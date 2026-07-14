#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterable, Mapping
from urllib import error, request


ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = ROOT / "prepared" / "preference_candidates_strict_v2.jsonl"
DEFAULT_TOKENIZED = ROOT / "prepared" / "mpo_tokenized_pairs_strict_v2.jsonl"
DEFAULT_PROMPT = ROOT / "prompts" / "source_integrity_filter_v1.txt"
DEFAULT_RUN_DIR = ROOT / "source_quality" / "gpt54mini_source_integrity_v1"
DEFAULT_FILTERED_CANDIDATES = ROOT / "prepared" / "preference_candidates_strict_v3_gpt54mini.jsonl"
DEFAULT_FILTERED_TOKENIZED = ROOT / "prepared" / "mpo_tokenized_pairs_strict_v3_gpt54mini.jsonl"
DEFAULT_REJECTIONS = ROOT / "analysis" / "source_quality_gpt54mini_rejections.jsonl"
DEFAULT_SUMMARY = ROOT / "analysis" / "source_quality_gpt54mini_summary.json"
DEFAULT_CONTRACT = ROOT / "dataset_contract_strict_v3_gpt54mini.json"

SCHEMA_VERSION = "dqs.source_integrity_filter.v1"
MODEL = "openai/gpt-5.4-mini"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_PRICE_PER_MTOK_USD = 0.75
COMPLETION_PRICE_PER_MTOK_USD = 4.50
TRANSIENT_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}

KEEP_REASONS = {
    "intact_prose",
    "intact_fragment_or_heading",
    "intact_table_or_list",
    "minor_noise_readable",
}
REJECT_REASONS = {
    "lost_delimiters",
    "flattened_table",
    "ocr_or_encoding_corruption",
    "mixed_or_reordered_segments",
    "severe_truncation",
    "other_severe_corruption",
}
REVIEW_REASONS = {
    "borderline_delimiter_loss",
    "borderline_fragmentation",
    "uncertain_structure",
}
DECISION_REASONS = {
    "KEEP": KEEP_REASONS,
    "REJECT": REJECT_REASONS,
    "REVIEW": REVIEW_REASONS,
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "source_integrity_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["KEEP", "REJECT", "REVIEW"]},
                "reason_code": {
                    "type": "string",
                    "enum": sorted(KEEP_REASONS | REJECT_REASONS | REVIEW_REASONS),
                },
                "evidence": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["decision", "reason_code", "evidence", "explanation"],
            "additionalProperties": False,
        },
    },
}


class FilterError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FilterError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise FilterError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def _unique_by_pair_id(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise FilterError(f"{label}[{idx}] has no non-empty pair_id")
        if pair_id in by_id:
            raise FilterError(f"duplicate pair_id in {label}: {pair_id}")
        by_id[pair_id] = dict(row)
    return by_id


def _select_requests(
    candidate_rows: list[dict[str, Any]],
    tokenized_rows: list[dict[str, Any]],
    selected_pair_ids: list[str] | None,
) -> list[dict[str, Any]]:
    candidates = _unique_by_pair_id(candidate_rows, label="candidates")
    tokenized = _unique_by_pair_id(tokenized_rows, label="tokenized")
    tokenized_order = [str(row["pair_id"]) for row in tokenized_rows]
    missing_candidates = [pair_id for pair_id in tokenized_order if pair_id not in candidates]
    if missing_candidates:
        raise FilterError(
            f"tokenized pair_ids missing from candidates: count={len(missing_candidates)} "
            f"first={missing_candidates[:5]}"
        )

    if selected_pair_ids:
        if len(selected_pair_ids) != len(set(selected_pair_ids)):
            raise FilterError("--pair-id values must be unique")
        missing = [pair_id for pair_id in selected_pair_ids if pair_id not in tokenized]
        if missing:
            raise FilterError(f"requested pair_ids missing from final tokenized dataset: {missing}")
        order = selected_pair_ids
    else:
        order = tokenized_order

    requests: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for order_idx, pair_id in enumerate(order):
        candidate = candidates[pair_id]
        source = candidate.get("source")
        if not isinstance(source, str) or not source.strip():
            raise FilterError(f"candidate source is empty: {pair_id}")
        source_sha256 = _sha256_text(source)
        if not selected_pair_ids and source_sha256 in seen_sources:
            raise FilterError(f"duplicate source content in full dataset: {pair_id}")
        seen_sources.add(source_sha256)
        requests.append(
            {
                "schema_version": SCHEMA_VERSION,
                "order_idx": order_idx,
                "pair_id": pair_id,
                "source_sha256": source_sha256,
                "source": source,
            }
        )
    return requests


def _manifest_payload(
    *,
    candidates: Path,
    tokenized: Path,
    prompt_path: Path,
    prompt: str,
    requests: list[dict[str, Any]],
    model: str,
    base_url: str,
    selected_pair_ids: list[str] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "model": model,
        "base_url": base_url,
        "session_policy": "one_source_per_independent_stateless_request",
        "decision_policy": "only_KEEP_is_accepted",
        "candidate_path": str(candidates.resolve()),
        "candidate_sha256": _sha256_file(candidates),
        "tokenized_path": str(tokenized.resolve()),
        "tokenized_sha256": _sha256_file(tokenized),
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": _sha256_text(prompt),
        "request_count": len(requests),
        "selection": "explicit_pair_ids" if selected_pair_ids else "all_final_strict_v2_pair_ids",
        "selected_pair_ids": selected_pair_ids or [],
        "temperature": 0.0,
        "reasoning_effort": "none",
        "max_tokens": 220,
        "structured_output_requested": True,
        "provider_require_parameters": False,
        "response_schema": RESPONSE_SCHEMA,
    }


def prepare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    requests_path = run_dir / "requests.jsonl"
    if manifest_path.exists() or requests_path.exists():
        raise FilterError(
            f"run directory already contains a prepared session: {run_dir}; "
            "use a new --run-dir for a genuinely new session"
        )

    candidates = Path(args.candidates)
    tokenized = Path(args.tokenized)
    prompt_path = Path(args.prompt)
    for path in (candidates, tokenized, prompt_path):
        if not path.is_file():
            raise FilterError(f"required input not found: {path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise FilterError("source integrity prompt is empty")

    requests = _select_requests(
        _read_jsonl(candidates),
        _read_jsonl(tokenized),
        args.pair_id or None,
    )
    manifest = _manifest_payload(
        candidates=candidates,
        tokenized=tokenized,
        prompt_path=prompt_path,
        prompt=prompt,
        requests=requests,
        model=args.model,
        base_url=args.base_url,
        selected_pair_ids=args.pair_id or None,
    )
    _atomic_write_jsonl(requests_path, requests)
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "prepared",
                "run_dir": str(run_dir),
                "requests": len(requests),
                "model": args.model,
                "session_policy": manifest["session_policy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _load_and_validate_session(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    manifest_path = run_dir / "manifest.json"
    requests_path = run_dir / "requests.jsonl"
    if not manifest_path.is_file() or not requests_path.is_file():
        raise FilterError(f"session is not prepared: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise FilterError("session manifest schema mismatch")
    requests_rows = _read_jsonl(requests_path)
    _unique_by_pair_id(requests_rows, label="requests")
    if len(requests_rows) != manifest.get("request_count"):
        raise FilterError("request count does not match manifest")
    prompt_path = Path(str(manifest.get("prompt_path", "")))
    if not prompt_path.is_file():
        raise FilterError(f"prompt file referenced by manifest is missing: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if _sha256_text(prompt) != manifest.get("prompt_sha256"):
        raise FilterError("prompt changed after session preparation")
    candidates = Path(str(manifest.get("candidate_path", "")))
    tokenized = Path(str(manifest.get("tokenized_path", "")))
    if _sha256_file(candidates) != manifest.get("candidate_sha256"):
        raise FilterError("candidate input changed after session preparation")
    if _sha256_file(tokenized) != manifest.get("tokenized_sha256"):
        raise FilterError("tokenized input changed after session preparation")
    return manifest, requests_rows, prompt


def _validate_decision(payload: Mapping[str, Any], source: str) -> dict[str, str]:
    expected_keys = {"decision", "reason_code", "evidence", "explanation"}
    if set(payload) != expected_keys:
        raise FilterError(f"decision keys mismatch: expected={sorted(expected_keys)} actual={sorted(payload)}")
    decision = payload.get("decision")
    reason_code = payload.get("reason_code")
    evidence = payload.get("evidence")
    explanation = payload.get("explanation")
    if not all(isinstance(value, str) for value in (decision, reason_code, evidence, explanation)):
        raise FilterError("all decision values must be strings")
    assert isinstance(decision, str)
    assert isinstance(reason_code, str)
    assert isinstance(evidence, str)
    assert isinstance(explanation, str)
    if decision not in DECISION_REASONS:
        raise FilterError(f"unsupported decision: {decision}")
    if reason_code not in DECISION_REASONS[decision]:
        raise FilterError(f"reason_code {reason_code!r} is invalid for decision {decision}")
    if not explanation.strip() or len(explanation) > 320:
        raise FilterError("explanation must contain 1..320 characters")
    if decision == "KEEP":
        if evidence != "":
            raise FilterError("KEEP evidence must be the empty string")
    else:
        if not evidence or len(evidence) > 180:
            raise FilterError("REJECT/REVIEW evidence must contain 1..180 characters")
        if evidence not in source:
            raise FilterError("REJECT/REVIEW evidence is not an exact source substring")
    return {
        "decision": decision,
        "reason_code": reason_code,
        "evidence": evidence,
        "explanation": explanation.strip(),
    }


def _decision_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("decision", "reason_code", "evidence", "explanation")
    }


def _response_content(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise FilterError("OpenRouter response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise FilterError("OpenRouter choice has no message object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise FilterError("OpenRouter returned empty content")
    return content.strip(), choice


def _usage(payload: Mapping[str, Any]) -> dict[str, int]:
    raw = payload.get("usage")
    if not isinstance(raw, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key, 0)
        usage[key] = int(value) if isinstance(value, (int, float)) else 0
    return usage


def _request_body(*, model: str, prompt: str, pair_id: str, source: str) -> dict[str, Any]:
    user_payload = json.dumps({"source_id": pair_id, "source": source}, ensure_ascii=False)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.0,
        "max_tokens": 220,
        "include_reasoning": False,
        "reasoning": {"effort": "none", "exclude": True},
        "reasoning_effort": "none",
        "response_format": RESPONSE_SCHEMA,
    }


def _post_json(
    *,
    url: str,
    api_key: str,
    body: Mapping[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "DQS source integrity filter",
    }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=encoded, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        failure = FilterError(f"OpenRouter HTTP {exc.code}: {response_text[:800]}")
        setattr(failure, "http_status", exc.code)
        raise failure from exc
    except (error.URLError, TimeoutError) as exc:
        raise FilterError(f"OpenRouter request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FilterError(f"OpenRouter response was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FilterError("OpenRouter response must be a JSON object")
    return payload


def _classify_one(
    *,
    row: Mapping[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    prompt: str,
    timeout_s: float,
    max_retries: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    pair_id = str(row["pair_id"])
    source = str(row["source"])
    body = _request_body(model=model, prompt=prompt, pair_id=pair_id, source=source)
    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 2):
        started = time.perf_counter()
        try:
            response = _post_json(url=base_url, api_key=api_key, body=body, timeout_s=timeout_s)
            content, choice = _response_content(response)
            parsed = json.loads(content)
            if not isinstance(parsed, Mapping):
                raise FilterError("structured decision must be a JSON object")
            decision = _validate_decision(parsed, source)
            response_model = response.get("model")
            if not isinstance(response_model, str) or "gpt-5.4-mini" not in response_model:
                raise FilterError(f"unexpected response model identity: {response_model!r}")
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "order_idx": int(row["order_idx"]),
                "pair_id": pair_id,
                "source_sha256": str(row["source_sha256"]),
                **decision,
                "request_model": model,
                "response_model": response_model,
                "provider": response.get("provider"),
                "response_id": response.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "usage": _usage(response),
                "attempt_count": attempt,
                "latency_ms": elapsed_ms,
            }
        except (FilterError, json.JSONDecodeError) as exc:
            last_error = exc
            status = getattr(exc, "http_status", None)
            retryable = status is None or status in TRANSIENT_HTTP_STATUS
            if attempt > max_retries or not retryable:
                break
            time.sleep(retry_sleep_s * attempt)
    raise FilterError(f"classification failed for {pair_id} after retries: {last_error}")


def _load_journal(path: Path, requests_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    by_id = _unique_by_pair_id(rows, label="decision journal")
    for pair_id, row in by_id.items():
        request_row = requests_by_id.get(pair_id)
        if request_row is None:
            raise FilterError(f"decision journal contains an unknown pair_id: {pair_id}")
        if row.get("source_sha256") != request_row.get("source_sha256"):
            raise FilterError(f"decision journal source hash mismatch: {pair_id}")
        _validate_decision(_decision_fields(row), str(request_row["source"]))
    return by_id


def run_session(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest, requests_rows, prompt = _load_and_validate_session(run_dir)
    if args.model != manifest.get("model") or args.base_url != manifest.get("base_url"):
        raise FilterError("run model/base_url must exactly match the prepared manifest")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise FilterError(f"missing API key environment variable: {args.api_key_env}")

    requests_by_id = _unique_by_pair_id(requests_rows, label="requests")
    journal_path = run_dir / "decision_journal.jsonl"
    failure_path = run_dir / "failure_journal.jsonl"
    completed = _load_journal(journal_path, requests_by_id)
    pending = [row for row in requests_rows if row["pair_id"] not in completed]
    print(
        json.dumps(
            {
                "status": "dispatch",
                "total": len(requests_rows),
                "completed": len(completed),
                "pending": len(pending),
                "concurrency": args.concurrency,
                "model": args.model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not pending:
        print(json.dumps({"status": "complete", "rows": len(completed)}), flush=True)
        return

    lock = threading.Lock()
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    done_count = len(completed)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_to_row = {
            executor.submit(
                _classify_one,
                row=row,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                prompt=prompt,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                retry_sleep_s=args.retry_sleep_s,
            ): row
            for row in pending
        }
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            pair_id = str(row["pair_id"])
            try:
                decision = future.result()
            except BaseException as exc:
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": _utc_now(),
                    "pair_id": pair_id,
                    "source_sha256": row["source_sha256"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                _append_jsonl(failure_path, failure, lock)
            else:
                _append_jsonl(journal_path, decision, lock)
                done_count += 1
                if done_count % 25 == 0 or done_count == len(requests_rows):
                    elapsed = max(time.perf_counter() - started, 0.001)
                    session_done = done_count - len(completed)
                    rate = session_done / elapsed
                    remaining = len(requests_rows) - done_count
                    print(
                        json.dumps(
                            {
                                "status": "progress",
                                "done": done_count,
                                "total": len(requests_rows),
                                "failed_this_run": len(failures),
                                "rows_per_second": round(rate, 3),
                                "eta_seconds": round(remaining / rate, 1) if rate else None,
                            }
                        ),
                        flush=True,
                    )

    final = _load_journal(journal_path, requests_by_id)
    missing = sorted(set(requests_by_id) - set(final))
    if failures or missing:
        raise FilterError(
            f"session incomplete: completed={len(final)} total={len(requests_rows)} "
            f"failures_this_run={len(failures)} missing={len(missing)}; rerun the same command to resume"
        )
    print(json.dumps({"status": "complete", "rows": len(final)}), flush=True)


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def apply_decisions(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest, requests_rows, _ = _load_and_validate_session(run_dir)
    requests_by_id = _unique_by_pair_id(requests_rows, label="requests")
    if manifest.get("selection") != "all_final_strict_v2_pair_ids":
        raise FilterError("apply is forbidden for a calibration/partial session")

    journal_path = run_dir / "decision_journal.jsonl"
    decisions_by_id = _load_journal(journal_path, requests_by_id)
    missing = sorted(set(requests_by_id) - set(decisions_by_id))
    if missing:
        raise FilterError(f"cannot apply incomplete decisions: missing={len(missing)} first={missing[:5]}")
    if len(decisions_by_id) != len(requests_rows):
        raise FilterError("decision completeness mismatch")

    decisions = [decisions_by_id[str(row["pair_id"])] for row in requests_rows]
    canonical_decisions_path = run_dir / "decisions.jsonl"
    _atomic_write_jsonl(canonical_decisions_path, decisions)
    keep_ids = {row["pair_id"] for row in decisions if row["decision"] == "KEEP"}

    candidate_path = Path(str(manifest["candidate_path"]))
    tokenized_path = Path(str(manifest["tokenized_path"]))
    candidate_rows = _read_jsonl(candidate_path)
    tokenized_rows = _read_jsonl(tokenized_path)
    candidate_by_id = _unique_by_pair_id(candidate_rows, label="candidates")
    tokenized_by_id = _unique_by_pair_id(tokenized_rows, label="tokenized")
    request_ids = [str(row["pair_id"]) for row in requests_rows]
    if set(request_ids) != set(tokenized_by_id):
        raise FilterError("full-session request IDs do not exactly equal final tokenized IDs")

    filtered_candidates = [candidate_by_id[pair_id] for pair_id in request_ids if pair_id in keep_ids]
    filtered_tokenized = [tokenized_by_id[pair_id] for pair_id in request_ids if pair_id in keep_ids]
    rejected_rows: list[dict[str, Any]] = []
    for pair_id in request_ids:
        decision = decisions_by_id[pair_id]
        if decision["decision"] == "KEEP":
            continue
        candidate = candidate_by_id[pair_id]
        rejected_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "subset": candidate.get("subset"),
                "row_id": candidate.get("row_id"),
                "source": candidate.get("source"),
                "source_sha256": decision["source_sha256"],
                "decision": decision["decision"],
                "reason_code": decision["reason_code"],
                "evidence": decision["evidence"],
                "explanation": decision["explanation"],
                "response_id": decision.get("response_id"),
                "response_model": decision.get("response_model"),
            }
        )

    filtered_candidates_path = Path(args.filtered_candidates)
    filtered_tokenized_path = Path(args.filtered_tokenized)
    rejections_path = Path(args.rejections)
    summary_path = Path(args.summary)
    contract_path = Path(args.contract)
    _atomic_write_jsonl(filtered_candidates_path, filtered_candidates)
    _atomic_write_jsonl(filtered_tokenized_path, filtered_tokenized)
    _atomic_write_jsonl(rejections_path, rejected_rows)

    prompt_tokens = sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in decisions)
    completion_tokens = sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in decisions)
    total_tokens = sum(int(row.get("usage", {}).get("total_tokens", 0)) for row in decisions)
    estimated_cost = (
        prompt_tokens / 1_000_000 * PROMPT_PRICE_PER_MTOK_USD
        + completion_tokens / 1_000_000 * COMPLETION_PRICE_PER_MTOK_USD
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "model": manifest["model"],
        "session_policy": manifest["session_policy"],
        "acceptance_policy": manifest["decision_policy"],
        "input_rows": len(requests_rows),
        "output_rows": len(filtered_tokenized),
        "excluded_rows": len(rejected_rows),
        "decision_counts": _counter_dict(str(row["decision"]) for row in decisions),
        "reason_counts": _counter_dict(str(row["reason_code"]) for row in decisions),
        "excluded_by_subset": _counter_dict(str(row.get("subset")) for row in rejected_rows),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "estimated_openrouter_cost_usd": round(estimated_cost, 6),
        "prompt_price_per_mtok_usd": PROMPT_PRICE_PER_MTOK_USD,
        "completion_price_per_mtok_usd": COMPLETION_PRICE_PER_MTOK_USD,
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "decisions": str(canonical_decisions_path.resolve()),
            "filtered_candidates": str(filtered_candidates_path.resolve()),
            "filtered_tokenized": str(filtered_tokenized_path.resolve()),
            "rejections": str(rejections_path.resolve()),
        },
    }
    _atomic_write_json(summary_path, summary)

    contract = {
        "schema_version": "dqs.mpo_dataset_contract.strict_v3_gpt54mini.v1",
        "created_at": _utc_now(),
        "source_quality_model": manifest["model"],
        "source_quality_prompt_sha256": manifest["prompt_sha256"],
        "source_quality_session_policy": manifest["session_policy"],
        "source_quality_acceptance_policy": manifest["decision_policy"],
        "input_candidate_sha256": manifest["candidate_sha256"],
        "input_tokenized_sha256": manifest["tokenized_sha256"],
        "decision_count": len(decisions),
        "decision_journal_sha256": _sha256_file(journal_path),
        "canonical_decisions_sha256": _sha256_file(canonical_decisions_path),
        "output_rows": len(filtered_tokenized),
        "filtered_candidates_path": str(filtered_candidates_path.resolve()),
        "filtered_candidates_sha256": _sha256_file(filtered_candidates_path),
        "filtered_tokenized_path": str(filtered_tokenized_path.resolve()),
        "filtered_tokenized_sha256": _sha256_file(filtered_tokenized_path),
        "rejections_path": str(rejections_path.resolve()),
        "rejections_sha256": _sha256_file(rejections_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
        "hard_invariants": {
            "every_final_strict_v2_source_classified_exactly_once": True,
            "one_source_per_stateless_request": True,
            "only_keep_rows_retained": True,
            "reject_and_review_rows_retained": False,
            "model_fallback_allowed": False,
            "output_pair_ids_preserve_input_order": True,
            "candidate_and_tokenized_pair_ids_equal": True,
        },
    }
    _atomic_write_json(contract_path, contract)
    print(json.dumps({"status": "applied", **summary}, ensure_ascii=False), flush=True)


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify every final strict-v2 source with independent GPT-5.4-mini sessions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    _common_parser(prepare_parser)
    prepare_parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    prepare_parser.add_argument("--tokenized", default=str(DEFAULT_TOKENIZED))
    prepare_parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    prepare_parser.add_argument("--pair-id", action="append", default=[])
    prepare_parser.set_defaults(func=prepare)

    run_parser = subparsers.add_parser("run")
    _common_parser(run_parser)
    run_parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    run_parser.add_argument("--concurrency", type=int, default=4)
    run_parser.add_argument("--timeout-s", type=float, default=180.0)
    run_parser.add_argument("--max-retries", type=int, default=5)
    run_parser.add_argument("--retry-sleep-s", type=float, default=1.5)
    run_parser.set_defaults(func=run_session)

    apply_parser = subparsers.add_parser("apply")
    _common_parser(apply_parser)
    apply_parser.add_argument("--filtered-candidates", default=str(DEFAULT_FILTERED_CANDIDATES))
    apply_parser.add_argument("--filtered-tokenized", default=str(DEFAULT_FILTERED_TOKENIZED))
    apply_parser.add_argument("--rejections", default=str(DEFAULT_REJECTIONS))
    apply_parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    apply_parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    apply_parser.set_defaults(func=apply_decisions)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "concurrency", 1) < 1:
        parser.error("--concurrency must be >= 1")
    if getattr(args, "max_retries", 0) < 0:
        parser.error("--max-retries must be >= 0")
    try:
        args.func(args)
    except FilterError as exc:
        print(f"source-quality filter failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
