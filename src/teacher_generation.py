from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, ValidationError

from degeneration_filter import classify_student_output
from io_utils import write_jsonl
from progress import progress, progress_context


TeacherLabel = Literal["no_change", "minor", "major", "critical", "invalid"]
TEACHER_ACCEPTED_LABELS = ("no_change", "minor", "major", "critical")
TeacherErrorType = Literal[
    "mistranslation",
    "omission",
    "addition",
    "terminology",
    "acronym",
    "proper_noun",
    "number_unit",
    "grammar_fluency",
    "register",
    "structure",
    "punctuation",
    "parenthetical",
    "traceability",
    "other",
]


class TeacherErrorItem(BaseModel):
    error_span_target: str | None
    source_span: str | None
    error_type: TeacherErrorType
    correction: str | None
    reason_ko: str


class TeacherOutputItem(BaseModel):
    id: str
    label: TeacherLabel
    final_translation: str | None
    invalid_reason_ko: str | None
    errors: list[TeacherErrorItem] = Field(default_factory=list)


class TeacherBatchOutput(BaseModel):
    items: list[TeacherOutputItem]


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _save_all_step_artifacts(cfg: Mapping[str, Any]) -> bool:
    return bool(_get(cfg, "logging.save_all_step_artifacts", False))


def _enabled_providers(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    providers = _get(cfg, "teacher.providers", [])
    if not isinstance(providers, list):
        raise SystemExit("teacher.providers must be a list")
    enabled: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, Mapping):
            raise SystemExit(f"invalid teacher provider config: {provider!r}")
        weight = float(provider.get("weight", 0.0) or 0.0)
        if weight > 0:
            out = dict(provider)
            out["weight"] = weight
            enabled.append(out)
    if not enabled:
        raise SystemExit("teacher.providers has no enabled provider with weight > 0")
    return enabled


def _choose_provider(
    providers: list[dict[str, Any]],
    *,
    seed: int,
    batch_idx: int,
) -> dict[str, Any]:
    total = sum(float(provider["weight"]) for provider in providers)
    rng = random.Random(f"{seed}|teacher|{batch_idx}")
    pick = rng.random() * total
    cursor = 0.0
    for provider in providers:
        cursor += float(provider["weight"])
        if pick <= cursor:
            return provider
    return providers[-1]


def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + size] for idx in range(0, len(rows), size)]


def _candidate_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "source": str(row.get("source", "")),
        "draft": str(row.get("student_translation", "")),
    }


def _render_user_prompt(template: str, items: list[dict[str, Any]]) -> str:
    items_json = json.dumps(items, ensure_ascii=False, sort_keys=True, indent=2)
    item_blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        item_blocks.append(
            "\n".join(
                [
                    f"[ITEM {idx}]",
                    "[ID]",
                    str(item["id"]),
                    "",
                    "[SOURCE]",
                    str(item["source"]),
                    "",
                    "[DRAFT]",
                    str(item["draft"]),
                    f"[/ITEM {idx}]",
                ]
            )
        )
    return template.format(
        batch_size=len(items),
        items_json=items_json,
        items_text="\n\n".join(item_blocks),
    )


def _parse_teacher_json(text: str) -> TeacherBatchOutput:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    data = json.loads(cleaned)
    if hasattr(TeacherBatchOutput, "model_validate"):
        return TeacherBatchOutput.model_validate(data)
    return TeacherBatchOutput.parse_obj(data)


def _validate_batch_ids(parsed: TeacherBatchOutput, expected_ids: set[str]) -> None:
    actual_ids = [item.id for item in parsed.items]
    if len(actual_ids) != len(expected_ids):
        raise ValueError(f"teacher item count mismatch: expected={len(expected_ids)} actual={len(actual_ids)}")
    if set(actual_ids) != expected_ids:
        raise ValueError(f"teacher item ids mismatch: expected={sorted(expected_ids)} actual={sorted(actual_ids)}")


def _call_openai(
    *,
    provider: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required for OpenAI teacher provider") from exc
    api_key_env = str(provider.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env: {api_key_env}")
    client = OpenAI(api_key=api_key)
    started = time.perf_counter()
    response = client.responses.create(
        model=str(provider["model"]),
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for part in getattr(item, "content", None) or []:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        output_text = "".join(chunks)
    return {"text": output_text, "latency_ms": round(latency_ms, 3), "usage": {}}


def _call_anthropic(
    *,
    provider: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        raise RuntimeError("anthropic package is required for Anthropic teacher provider") from exc
    api_key_env = str(provider.get("api_key_env", "ANTHROPIC_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env: {api_key_env}")
    client = anthropic.Anthropic(api_key=api_key)
    started = time.perf_counter()
    response = client.messages.create(
        model=str(provider["model"]),
        max_tokens=max_output_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    chunks = [getattr(block, "text", "") or "" for block in getattr(response, "content", []) if getattr(block, "type", None) == "text"]
    return {"text": "".join(chunks), "latency_ms": round(latency_ms, 3), "usage": {}}


def _call_gemini(
    *,
    provider: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("requests package is required for Gemini teacher provider") from exc
    api_key_env = str(provider.get("api_key_env", "GEMINI_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env: {api_key_env}")
    model = str(provider["model"])
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    started = time.perf_counter()
    last_error = ""
    for attempt in range(5):
        response = requests.post(url, json=payload, timeout=180)
        if response.status_code < 300:
            data = response.json()
            chunks: list[str] = []
            for candidate in data.get("candidates", []) or []:
                for part in ((candidate.get("content") or {}).get("parts") or []):
                    if part.get("text"):
                        chunks.append(str(part["text"]))
            usage = data.get("usageMetadata") or {}
            return {
                "text": "".join(chunks),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "usage": usage,
            }
        last_error = response.text[:500]
        if response.status_code not in {429, 500, 502, 503, 504}:
            break
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Gemini API error {response.status_code}: {last_error}")


def _call_provider(
    *,
    provider: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    name = str(provider.get("name", "")).lower()
    if name == "openai":
        return _call_openai(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if name in {"anthropic", "claude"}:
        return _call_anthropic(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if name == "gemini":
        return _call_gemini(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    raise RuntimeError(f"unsupported teacher provider: {name}")


def _call_batch_with_retries(
    *,
    batch_idx: int,
    provider: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    expected_ids: set[str],
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    last_error = ""
    raw_text = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = _call_provider(
                provider=provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            raw_text = str(raw.get("text", "") or "")
            parsed = _parse_teacher_json(raw_text)
            _validate_batch_ids(parsed, expected_ids)
            return {
                "batch_idx": batch_idx,
                "status": "ok",
                "provider": provider.get("name"),
                "model": provider.get("model"),
                "attempt": attempt,
                "raw_text": raw_text,
                "parsed": parsed,
                "latency_ms": raw.get("latency_ms"),
                "usage": raw.get("usage", {}),
                "error": None,
            }
        except (ValidationError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 15))
    return {
        "batch_idx": batch_idx,
        "status": "failed",
        "provider": provider.get("name"),
        "model": provider.get("model"),
        "attempt": max_retries,
        "raw_text": raw_text,
        "parsed": None,
        "latency_ms": None,
        "usage": {},
        "error": last_error,
    }


def _raw_response_row(result: Mapping[str, Any], batch_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batch_idx": result["batch_idx"],
        "status": result["status"],
        "provider": result.get("provider"),
        "model": result.get("model"),
        "attempt": result.get("attempt"),
        "item_ids": [item["id"] for item in batch_items],
        "raw_text": result.get("raw_text", ""),
        "usage": result.get("usage", {}),
        "latency_ms": result.get("latency_ms"),
        "error": result.get("error"),
    }


def _teacher_rejection_counts(rejected: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    reason_counts = Counter(str(row.get("reject_reason", "unknown")) for row in rejected)
    flag_counts: Counter[str] = Counter()
    for row in rejected:
        flags = row.get("reject_flags", [])
        if isinstance(flags, list):
            flag_counts.update(str(flag) for flag in flags)
    return {
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "reject_flag_counts": dict(sorted(flag_counts.items())),
    }


def _teacher_label_summary(accepted: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    counts = {
        label: 0
        for label in TEACHER_ACCEPTED_LABELS
    }
    for row in accepted:
        label = str(row.get("teacher_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    total = len(accepted)
    ratios = {
        label: (count / total if total > 0 else 0.0)
        for label, count in counts.items()
    }
    return {
        "label_counts": dict(sorted(counts.items())),
        "label_ratios": dict(sorted(ratios.items())),
    }


def _run_teacher_batch_inputs(
    *,
    batch_inputs: list[tuple[int, dict[str, Any], str, str, list[dict[str, Any]], set[str]]],
    max_workers: int,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
) -> dict[int, dict[str, Any]]:
    if not batch_inputs:
        return {}
    results: dict[int, dict[str, Any]] = {}
    worker_count = min(max_workers, len(batch_inputs))
    progress("teacher api-window start", batches=len(batch_inputs), workers=worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _call_batch_with_retries,
                batch_idx=batch_idx,
                provider=provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                expected_ids=expected_ids,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                max_retries=max_retries,
            )
            for batch_idx, provider, system_prompt, user_prompt, _batch_rows, expected_ids in batch_inputs
        ]
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results[int(result["batch_idx"])] = result
            completed += 1
            progress(
                "teacher api-batch done",
                batch=result.get("batch_idx"),
                status=result.get("status"),
                completed=f"{completed}/{len(futures)}",
                provider=result.get("provider"),
                model=result.get("model"),
            )
    return results


def _record_teacher_batch_result(
    *,
    result: Mapping[str, Any],
    batch_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
) -> dict[str, TeacherOutputItem]:
    raw_rows.append(_raw_response_row(result, [_candidate_item(row) for row in batch_rows]))
    parsed_by_id: dict[str, TeacherOutputItem] = {}
    parsed = result.get("parsed")
    if isinstance(parsed, TeacherBatchOutput):
        for item in parsed.items:
            parsed_by_id[item.id] = item
            parsed_rows.append(
                {
                    "id": item.id,
                    "batch_idx": result["batch_idx"],
                    "provider": result.get("provider"),
                    "model": result.get("model"),
                    "label": item.label,
                    "final_translation": item.final_translation,
                    "invalid_reason_ko": item.invalid_reason_ko,
                    "errors": [error.model_dump() if hasattr(error, "model_dump") else error.dict() for error in item.errors],
                    "parse_status": "ok",
                    "parse_error": None,
                }
            )
    else:
        for row in batch_rows:
            parsed_rows.append(
                {
                    "id": row["id"],
                    "batch_idx": result["batch_idx"],
                    "provider": result.get("provider"),
                    "model": result.get("model"),
                    "label": None,
                    "final_translation": None,
                    "invalid_reason_ko": None,
                    "errors": [],
                    "parse_status": "failed",
                    "parse_error": result.get("error"),
                }
            )
    return parsed_by_id


def _write_teacher_artifacts(
    *,
    subset_dir: Path,
    request_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
) -> None:
    records: list[dict[str, Any]] = []
    for row in request_rows:
        records.append({"record_type": "teacher_request", **row})
    for row in raw_rows:
        records.append({"record_type": "teacher_raw_response", **row})
    for row in parsed_rows:
        records.append({"record_type": "teacher_parsed_item", **row})
    for row in rejected_rows:
        records.append({"record_type": "teacher_rejected_row", **row})
    write_jsonl(subset_dir / "teacher_artifacts.jsonl", records)


def _flush_teacher_outputs(
    *,
    cfg: Mapping[str, Any],
    subset_dir: Path,
    request_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
) -> None:
    _write_teacher_artifacts(
        subset_dir=subset_dir,
        request_rows=request_rows,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        rejected_rows=rejected_rows,
    )
    if _save_all_step_artifacts(cfg):
        write_jsonl(subset_dir / "teacher_requests.jsonl", request_rows)
        write_jsonl(subset_dir / "teacher_responses.raw.jsonl", raw_rows)
        write_jsonl(subset_dir / "teacher_parsed.jsonl", parsed_rows)
        write_jsonl(subset_dir / "teacher_rejected.jsonl", rejected_rows)
    write_jsonl(subset_dir / "golden_pairs.jsonl", accepted_rows)


def run_teacher_generation(
    *,
    cfg: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    subset_dir: Path,
) -> dict[str, Any]:
    teacher_cfg = _get(cfg, "teacher", {})
    if not isinstance(teacher_cfg, Mapping):
        raise SystemExit("teacher config must be a mapping")
    target = int(_get(cfg, "data.teacher_target_per_subset", 1000) or 1000)
    batch_size = max(1, int(teacher_cfg.get("batch_size", 4) or 4))
    max_workers = max(1, int(teacher_cfg.get("max_workers", 20) or 20))
    max_retries = max(1, int(teacher_cfg.get("max_retries_per_row", 3) or 3))
    max_output_tokens = max(512, int(teacher_cfg.get("max_output_tokens", 8192) or 8192))
    temperature = float(teacher_cfg.get("temperature", 0.0) or 0.0)
    refill_until_target = bool(teacher_cfg.get("refill_until_target", True))
    abort_on_all_failed_window = bool(teacher_cfg.get("abort_on_all_failed_window", True))

    system_prompt = _read_text(str(teacher_cfg.get("system_prompt_path", "prompts/teacher_system.txt")))
    user_template = _read_text(str(teacher_cfg.get("user_prompt_path", "prompts/teacher_user_batch.txt")))
    providers = _enabled_providers(cfg)
    seed = int(_get(cfg, "run.seed", 42) or 42)

    request_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    filter_cfg = _get(cfg, "data.degeneration_filter", {})
    if not isinstance(filter_cfg, Mapping):
        filter_cfg = {}
    teacher_filter_enabled = bool(filter_cfg.get("enabled", True)) and bool(filter_cfg.get("teacher_enabled", True))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    sorted_candidates = sorted(candidates, key=lambda row: int(row.get("selection_rank", 10**12)))
    batches = _chunks(sorted_candidates, batch_size)
    progress(
        "teacher start",
        candidates=len(candidates),
        batches=len(batches),
        target=target,
        batch_size=batch_size,
        max_workers=max_workers,
    )
    called_batches = 0
    requested_candidate_rows = 0
    batch_cursor = 0
    batch_window = max_workers if refill_until_target else max(len(batches), 1)

    while batch_cursor < len(batches):
        if refill_until_target and len(accepted) >= target:
            break
        window_batches = batches[batch_cursor : batch_cursor + batch_window]
        batch_inputs: list[tuple[int, dict[str, Any], str, str, list[dict[str, Any]], set[str]]] = []
        batch_rows_by_idx: dict[int, list[dict[str, Any]]] = {}
        for offset, batch_rows in enumerate(window_batches):
            batch_idx = batch_cursor + offset
            items = [_candidate_item(row) for row in batch_rows]
            provider = _choose_provider(providers, seed=seed, batch_idx=batch_idx)
            user_prompt = _render_user_prompt(user_template, items)
            expected_ids = {item["id"] for item in items}
            request_rows.append(
                {
                    "batch_idx": batch_idx,
                    "provider": provider.get("name"),
                    "model": provider.get("model"),
                    "item_ids": sorted(expected_ids),
                    "batch_size": len(items),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            batch_inputs.append((batch_idx, provider, system_prompt, user_prompt, batch_rows, expected_ids))
            batch_rows_by_idx[batch_idx] = batch_rows

        with progress_context(
            "teacher api-window",
            start_batch=batch_cursor,
            batches=len(batch_inputs),
            accepted=len(accepted),
        ):
            results = _run_teacher_batch_inputs(
                batch_inputs=batch_inputs,
                max_workers=max_workers,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                max_retries=max_retries,
            )
        called_batches += len(batch_inputs)
        requested_candidate_rows += sum(len(batch_rows_by_idx[batch_idx]) for batch_idx in results)

        for batch_idx in sorted(results):
            batch_rows = batch_rows_by_idx[batch_idx]
            parsed_by_id = _record_teacher_batch_result(
                result=results[batch_idx],
                batch_rows=batch_rows,
                raw_rows=raw_rows,
                parsed_rows=parsed_rows,
            )
            for candidate in batch_rows:
                if refill_until_target and len(accepted) >= target:
                    break
                item = parsed_by_id.get(str(candidate["id"]))
                if item is None:
                    rejected.append(_reject_row(candidate, reason="missing_or_unparsed_teacher_item", flags=[]))
                    continue
                if item.label == "invalid":
                    rejected.append(_reject_row(candidate, reason="teacher_invalid", flags=["teacher_invalid"], item=item))
                    continue
                translation = (item.final_translation or "").strip()
                if not translation:
                    rejected.append(_reject_row(candidate, reason="empty_teacher_translation", flags=["empty"], item=item))
                    continue
                if teacher_filter_enabled:
                    label, flags = classify_student_output(
                        source=str(candidate.get("source", "")),
                        mt=translation,
                        status="ok",
                        finish_reason=None,
                        config=filter_cfg,
                    )
                    if label != "clean":
                        rejected.append(_reject_row(candidate, reason=f"teacher_{label}", flags=flags, item=item))
                        continue
                accepted.append(_accept_row(candidate, item=item, rank=len(accepted) + 1))

        batch_cursor += len(window_batches)
        progress(
            "teacher window processed",
            requested=requested_candidate_rows,
            accepted=len(accepted),
            rejected=len(rejected),
            target=target,
        )
        if abort_on_all_failed_window and results and all(result.get("status") == "failed" for result in results.values()):
            sample_error = next(
                (
                    str(result.get("error", "")).strip()
                    for result in results.values()
                    if str(result.get("error", "")).strip()
                ),
                "unknown teacher provider error",
            )
            _flush_teacher_outputs(
                cfg=cfg,
                subset_dir=subset_dir,
                request_rows=request_rows,
                raw_rows=raw_rows,
                parsed_rows=parsed_rows,
                rejected_rows=rejected,
                accepted_rows=accepted,
            )
            raise RuntimeError(
                "teacher API window failed for every batch; "
                f"aborting before SFT. start_batch={batch_cursor} "
                f"batches={len(results)} sample_error={sample_error[:1000]}"
            )
        if not refill_until_target:
            break

    _flush_teacher_outputs(
        cfg=cfg,
        subset_dir=subset_dir,
        request_rows=request_rows,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        rejected_rows=rejected,
        accepted_rows=accepted,
    )
    shortfall = max(0, target - len(accepted))
    rejection_counts = _teacher_rejection_counts(rejected)
    label_summary = _teacher_label_summary(accepted)
    summary = {
        "teacher_candidate_rows": len(candidates),
        "teacher_candidate_batches": len(batches),
        "teacher_batches": called_batches,
        "teacher_requested_candidate_rows": requested_candidate_rows,
        "teacher_skipped_candidate_rows": max(0, len(candidates) - requested_candidate_rows),
        "teacher_accepted_rows": len(accepted),
        "teacher_rejected_rows": len(rejected),
        "teacher_label_counts": label_summary["label_counts"],
        "teacher_label_ratios": label_summary["label_ratios"],
        "teacher_reject_reason_counts": rejection_counts["reject_reason_counts"],
        "teacher_reject_flag_counts": rejection_counts["reject_flag_counts"],
        "teacher_target_rows": target,
        "teacher_shortfall_rows": shortfall,
        "teacher_exhausted_candidate_pool": bool(shortfall and requested_candidate_rows >= len(candidates)),
        "teacher_refill_until_target": refill_until_target,
        "teacher_degeneration_filter_enabled": teacher_filter_enabled,
    }
    (subset_dir / "teacher_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    progress(
        "teacher done",
        accepted=len(accepted),
        rejected=len(rejected),
        shortfall=shortfall,
        called_batches=called_batches,
    )
    return summary


def _item_errors(item: TeacherOutputItem | None) -> list[dict[str, Any]]:
    if item is None:
        return []
    return [error.model_dump() if hasattr(error, "model_dump") else error.dict() for error in item.errors]


def _reject_row(
    candidate: Mapping[str, Any],
    *,
    reason: str,
    flags: list[str],
    item: TeacherOutputItem | None = None,
) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "source": candidate.get("source", ""),
        "student_translation": candidate.get("student_translation", ""),
        "qe_score": candidate.get("qe_score"),
        "selection_rank": candidate.get("selection_rank"),
        "teacher_label": item.label if item is not None else None,
        "teacher_translation": item.final_translation if item is not None else None,
        "teacher_errors": _item_errors(item),
        "reject_reason": reason,
        "reject_flags": flags,
    }


def _accept_row(
    candidate: Mapping[str, Any],
    *,
    item: TeacherOutputItem,
    rank: int,
) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "source": candidate.get("source", ""),
        "target": item.final_translation,
        "student_translation": candidate.get("student_translation", ""),
        "teacher_label": item.label,
        "teacher_errors": _item_errors(item),
        "teacher_accept_rank": rank,
        "qe_score": candidate.get("qe_score"),
        "selection_rank": candidate.get("selection_rank"),
        "prompt": candidate.get("prompt"),
        "prompt_template_id": candidate.get("prompt_template_id"),
        "prompt_template_group": candidate.get("prompt_template_group"),
        "prompt_template_hash": candidate.get("prompt_template_hash"),
        "chat_template_applied": candidate.get("chat_template_applied"),
        "source_tokens": candidate.get("source_tokens"),
        "length_bucket_idx": candidate.get("length_bucket_idx"),
        "length_bucket": candidate.get("length_bucket"),
        "metadata": candidate.get("metadata", {}),
    }
