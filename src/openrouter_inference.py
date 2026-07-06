#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib import error, request

from io_utils import read_jsonl, write_jsonl
from progress import progress, progress_context
from runtime_logging import configure_runtime_logging


configure_runtime_logging()

TRANSIENT_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _get(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _openrouter_cfg(row: Mapping[str, Any]) -> dict[str, Any]:
    inference = row.get("inference", {})
    if not isinstance(inference, Mapping):
        inference = {}
    cfg = inference.get("openrouter", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    return dict(cfg)


def _messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("prompt_messages", [])
    if isinstance(messages, list) and messages:
        out: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role and content:
                out.append({"role": role, "content": content})
        if out:
            return out
    return [{"role": "user", "content": str(row.get("prompt", ""))}]


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return ""


def _usage(payload: Mapping[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, Mapping):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            out[key] = int(value)
    return out


def _request_body(row: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = row.get("model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    decoding = row.get("decoding", {})
    if not isinstance(decoding, Mapping):
        decoding = {}
    cfg = _openrouter_cfg(row)

    model_name = str(model_cfg.get("name_or_path", "")).strip()
    if not model_name:
        raise RuntimeError("model.name_or_path is required for OpenRouter inference")
    body: dict[str, Any] = {
        "model": model_name,
        "messages": _messages(row),
        "temperature": float(decoding.get("temperature", 0.0) or 0.0),
        "top_p": float(decoding.get("top_p", 1.0) or 1.0),
        "max_tokens": int(decoding.get("max_new_tokens", 1024) or 1024),
    }
    provider = cfg.get("provider")
    if isinstance(provider, Mapping):
        body["provider"] = dict(provider)
    reasoning = cfg.get("reasoning")
    if isinstance(reasoning, Mapping):
        body["reasoning"] = dict(reasoning)
    reasoning_effort = str(cfg.get("reasoning_effort", "") or "").strip()
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    extra_body = cfg.get("extra_body", {})
    if isinstance(extra_body, Mapping):
        body = _deep_merge(body, extra_body)
    return body


def _headers(cfg: Mapping[str, Any], api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = str(cfg.get("site_url") or os.environ.get("OPENROUTER_SITE_URL", "")).strip()
    app_name = str(cfg.get("app_name") or os.environ.get("OPENROUTER_APP_NAME", "DQS Eval")).strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def _post_json(
    *,
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_s: float,
    max_retries: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        req = request.Request(url, data=payload, headers=dict(headers), method="POST")
        try:
            with request.urlopen(req, timeout=timeout_s) as response:
                parsed = json.loads(response.read().decode("utf-8"))
                if not isinstance(parsed, Mapping):
                    raise RuntimeError("OpenRouter response must be a JSON object")
                return dict(parsed)
        except error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {body_text[:1000]}")
            if exc.code not in TRANSIENT_HTTP_STATUS or attempt >= max_retries:
                raise last_error
            retry_after = exc.headers.get("retry-after")
            sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else retry_sleep_s
            time.sleep(sleep_s * (attempt + 1))
        except (error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
            time.sleep(retry_sleep_s * (attempt + 1))
    raise RuntimeError(f"OpenRouter request failed: {last_error}")


def _run_one(
    *,
    row: Mapping[str, Any],
    url: str,
    headers: Mapping[str, str],
    timeout_s: float,
    max_retries: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    payload = _post_json(
        url=url,
        headers=headers,
        body=_request_body(row),
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
    )
    choices = payload.get("choices", [])
    generation = choices[0] if isinstance(choices, list) and choices else {}
    message = generation.get("message", {}) if isinstance(generation, Mapping) else {}
    content = message.get("content") if isinstance(message, Mapping) else None
    text = _normalize_content(content)
    usage = _usage(payload)
    return {
        "id": str(row.get("id", "")),
        "row_id": str(row.get("row_id", "")),
        "order_idx": int(row.get("order_idx", 0)),
        "status": "ok" if text else "failed",
        "mt": text,
        "finish_reason": generation.get("finish_reason") if isinstance(generation, Mapping) else None,
        "generated_token_count": int(usage.get("completion_tokens", 0)),
        "usage": usage,
        "provider": _get(payload, "provider.name"),
        "model": payload.get("model"),
        "error": None if text else "OpenRouter generation returned empty translation",
    }


def run(input_path: Path, output_path: Path) -> None:
    rows = read_jsonl(input_path)
    if not rows:
        write_jsonl(output_path, [])
        return

    cfg = _openrouter_cfg(rows[0])
    api_key_env = str(cfg.get("api_key_env", "OPENROUTER_API_KEY") or "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing OpenRouter API key env var: {api_key_env}")

    url = str(cfg.get("base_url", "https://openrouter.ai/api/v1/chat/completions")).strip()
    timeout_s = float(cfg.get("timeout_s", 120) or 120)
    max_retries = int(cfg.get("max_retries", 5) or 5)
    retry_sleep_s = float(cfg.get("retry_sleep_s", 2.0) or 2.0)
    concurrency = max(1, int(cfg.get("concurrency", 4) or 4))
    headers = _headers(cfg, api_key)

    progress("openrouter dispatch", requests=len(rows), concurrency=concurrency)
    out_rows: list[dict[str, Any] | None] = [None] * len(rows)
    with progress_context("openrouter generate", requests=len(rows)):
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(
                    _run_one,
                    row=row,
                    url=url,
                    headers=headers,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    retry_sleep_s=retry_sleep_s,
                ): idx
                for idx, row in enumerate(rows)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                out_rows[idx] = future.result()

    write_jsonl(output_path, [row for row in out_rows if row is not None])
    progress("openrouter output written", output=output_path, rows=len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenRouter chat inference for DQS eval.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
