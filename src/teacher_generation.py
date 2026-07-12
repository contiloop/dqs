from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, ValidationError

from degeneration_filter import classify_student_output
from io_utils import write_jsonl
from progress import progress, progress_context
from text_tokenization import text_token_ids


TeacherLabel = Literal["no_change", "minor", "major", "critical", "invalid"]
TEACHER_ACCEPTED_LABELS = ("no_change", "minor", "major", "critical")
INVALID_REASON_KO = "독립적으로 번역하기 어려운 원문"
INVALID_DRAFT_FORMAT_REASON_KO = "번역문 외의 설명·선택지·분석이 포함된 초안"
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


def _normalize_teacher_output_item(item: TeacherOutputItem) -> TeacherOutputItem:
    if item.label == "invalid":
        item.final_translation = None
        if item.invalid_reason_ko not in {INVALID_REASON_KO, INVALID_DRAFT_FORMAT_REASON_KO}:
            item.invalid_reason_ko = INVALID_REASON_KO
        item.errors = []
    else:
        item.invalid_reason_ko = None
    return item


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _teacher_system_prompt(teacher_cfg: Mapping[str, Any]) -> str:
    system_prompt = _read_text(str(teacher_cfg.get("system_prompt_path", "prompts/teacher_system.txt")))
    marker = "{{DRAFT_FORMAT_POLICY}}"
    if marker not in system_prompt:
        return system_prompt
    policy_path = str(teacher_cfg.get("draft_format_policy_path", "")).strip()
    if not policy_path:
        raise SystemExit(
            "teacher system prompt contains {{DRAFT_FORMAT_POLICY}} but "
            "teacher.draft_format_policy_path is empty"
        )
    policy = _read_text(policy_path).strip()
    if not policy:
        raise SystemExit(f"teacher draft-format policy is empty: {policy_path}")
    return system_prompt.replace(marker, policy)


def _load_translation_tokenizer(cfg: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("missing transformers; run `make set` first") from exc

    model_cfg = _get(cfg, "model", {})
    if not isinstance(model_cfg, Mapping):
        raise SystemExit("model config must be a mapping")
    return AutoTokenizer.from_pretrained(
        str(model_cfg["name_or_path"]),
        revision=str(model_cfg.get("tokenizer_revision", "main")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )


def _translation_token_count(tokenizer: Any, text: str) -> int:
    return len(text_token_ids(tokenizer, text, add_special_tokens=False))


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


def _source_token_length(row: Mapping[str, Any]) -> int:
    raw = row.get("source_tokens")
    if isinstance(raw, bool):
        raw = None
    if isinstance(raw, int):
        return max(0, raw)
    if isinstance(raw, float):
        return max(0, int(raw))
    if isinstance(raw, str) and raw.strip():
        try:
            return max(0, int(float(raw.strip())))
        except ValueError:
            pass
    return max(1, len(str(row.get("source", "")).split()))


def _length_buckets(cfg: Mapping[str, Any]) -> list[tuple[int, int]]:
    raw = _get(cfg, "data.length_buckets", [])
    buckets: list[tuple[int, int]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if not isinstance(item, list) or len(item) != 2:
                raise SystemExit(f"data.length_buckets[{idx}] must be [min, max]")
            low = int(item[0])
            high = int(item[1])
            if low > high:
                raise SystemExit(f"data.length_buckets[{idx}] has min > max")
            buckets.append((low, high))
    if not buckets:
        buckets = [(1, 1280)]
    return buckets


def _bucket_index_for_candidate(row: Mapping[str, Any], buckets: list[tuple[int, int]]) -> int:
    length = _source_token_length(row)
    for idx, (low, high) in enumerate(buckets):
        if low <= length <= high:
            return idx
    if length < buckets[0][0]:
        return 0
    return len(buckets) - 1


def _bucket_weights(cfg: Mapping[str, Any], bucket_count: int) -> list[float] | None:
    raw = _get(cfg, "data.length_bucket_selection.weights")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != bucket_count:
        raise SystemExit(
            "data.length_bucket_selection.weights must be null or a list with "
            f"{bucket_count} values"
        )
    weights = [max(0.0, float(value)) for value in raw]
    if sum(weights) <= 0:
        raise SystemExit("data.length_bucket_selection.weights must contain a positive weight")
    return weights


def _quota_from_weights(total: int, weights: list[float]) -> list[int]:
    if total <= 0:
        return [0 for _ in weights]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise SystemExit("cannot allocate length bucket quotas because all bucket weights are zero")
    raw = [(total * weight / weight_sum) for weight in weights]
    quotas = [int(value) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(weights)),
        key=lambda idx: (-(raw[idx] - quotas[idx]), idx),
    )
    for idx in order[:remainder]:
        quotas[idx] += 1
    return quotas


def _teacher_bucket_plan(
    *,
    cfg: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    target: int,
) -> dict[str, Any]:
    selection_cfg = _get(cfg, "data.length_bucket_selection", {})
    if not isinstance(selection_cfg, Mapping) or not bool(selection_cfg.get("enabled", True)):
        return {
            "enabled": False,
            "buckets": [(0, 10**12)],
            "bucket_counts": [len(candidates)],
            "quotas": [target],
            "quota_strategy": "disabled",
            "explicit_weights": None,
            "positive_weight_bucket_indexes": {0},
            "allow_zero_weight_bucket_fill": True,
            "fill_remainder_from_global": True,
        }

    buckets = _length_buckets(cfg)
    bucket_counts = [0 for _ in buckets]
    for candidate in candidates:
        bucket_counts[_bucket_index_for_candidate(candidate, buckets)] += 1

    explicit_weights = _bucket_weights(cfg, len(buckets))
    quota_strategy = str(selection_cfg.get("quota_strategy", "proportional")).strip().lower()
    if explicit_weights is not None:
        weights = explicit_weights
    elif quota_strategy == "uniform":
        weights = [1.0 for _ in buckets]
    elif quota_strategy == "proportional":
        weights = [float(count) for count in bucket_counts]
    else:
        raise SystemExit("data.length_bucket_selection.quota_strategy must be uniform or proportional")

    return {
        "enabled": True,
        "buckets": buckets,
        "bucket_counts": bucket_counts,
        "quotas": _quota_from_weights(target, weights),
        "quota_strategy": quota_strategy,
        "explicit_weights": explicit_weights,
        "positive_weight_bucket_indexes": {
            idx for idx, weight in enumerate(weights)
            if weight > 0
        },
        "allow_zero_weight_bucket_fill": bool(selection_cfg.get("allow_zero_weight_bucket_fill", False)),
        "fill_remainder_from_global": bool(selection_cfg.get("fill_remainder_from_global", True)),
    }


def _bucket_label(buckets: list[tuple[int, int]], bucket_idx: int) -> str:
    low, high = buckets[bucket_idx]
    return f"{low}-{high}"


def _bucket_count_summary(
    *,
    buckets: list[tuple[int, int]],
    counts: list[int],
) -> dict[str, int]:
    return {
        _bucket_label(buckets, idx): int(count)
        for idx, count in enumerate(counts)
    }


def _teacher_bucket_queues(
    *,
    candidates: list[dict[str, Any]],
    buckets: list[tuple[int, int]],
) -> list[deque[dict[str, Any]]]:
    queues: list[deque[dict[str, Any]]] = [deque() for _ in buckets]
    sorted_candidates = sorted(candidates, key=lambda row: int(row.get("selection_rank", 10**12)))
    for candidate in sorted_candidates:
        bucket_idx = _bucket_index_for_candidate(candidate, buckets)
        queued = dict(candidate)
        queued["_teacher_bucket_idx"] = bucket_idx
        queued["_teacher_bucket"] = _bucket_label(buckets, bucket_idx)
        queues[bucket_idx].append(queued)
    return queues


def _candidate_qe_sort_score(row: Mapping[str, Any]) -> float:
    value = row.get("qe_score")
    if value is None:
        return 0.0
    return float(value)


def _choose_teacher_bucket(
    *,
    bucket_queues: list[deque[dict[str, Any]]],
    quotas: list[int],
    accepted_counts: list[int],
    scheduled_counts: list[int],
    fill_remainder_from_global: bool,
    explicit_weights: list[float] | None,
    positive_weight_bucket_indexes: set[int],
    allow_zero_weight_bucket_fill: bool,
) -> tuple[int, bool] | None:
    needed: list[int] = []
    for bucket_idx, queue in enumerate(bucket_queues):
        if not queue:
            continue
        if accepted_counts[bucket_idx] + scheduled_counts[bucket_idx] < quotas[bucket_idx]:
            needed.append(bucket_idx)
    if needed:
        return min(
            needed,
            key=lambda idx: (
                (accepted_counts[idx] + scheduled_counts[idx]) / max(quotas[idx], 1),
                idx,
            ),
        ), False

    if not fill_remainder_from_global:
        return None

    fallback: list[int] = []
    for bucket_idx, queue in enumerate(bucket_queues):
        if not queue:
            continue
        if explicit_weights is not None and not allow_zero_weight_bucket_fill:
            if bucket_idx not in positive_weight_bucket_indexes:
                continue
        fallback.append(bucket_idx)
    if not fallback:
        return None

    return min(
        fallback,
        key=lambda idx: (
            _candidate_qe_sort_score(bucket_queues[idx][0]),
            str(bucket_queues[idx][0].get("id", "")),
            idx,
        ),
    ), True


def _pop_teacher_batch_rows(
    *,
    bucket_queues: list[deque[dict[str, Any]]],
    quotas: list[int],
    accepted_counts: list[int],
    scheduled_counts: list[int],
    batch_size: int,
    fill_remainder_from_global: bool,
    explicit_weights: list[float] | None,
    positive_weight_bucket_indexes: set[int],
    allow_zero_weight_bucket_fill: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _ in range(batch_size):
        choice = _choose_teacher_bucket(
            bucket_queues=bucket_queues,
            quotas=quotas,
            accepted_counts=accepted_counts,
            scheduled_counts=scheduled_counts,
            fill_remainder_from_global=fill_remainder_from_global,
            explicit_weights=explicit_weights,
            positive_weight_bucket_indexes=positive_weight_bucket_indexes,
            allow_zero_weight_bucket_fill=allow_zero_weight_bucket_fill,
        )
        if choice is None:
            break
        bucket_idx, _ = choice
        candidate = bucket_queues[bucket_idx].popleft()
        scheduled_counts[bucket_idx] += 1
        rows.append(candidate)
    return rows


def _candidate_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "source": str(row.get("source", "")),
        "draft": str(row.get("student_translation", "")),
    }


def _validate_teacher_candidate_drafts(candidates: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    for row in candidates:
        # Only the legacy raw-random ablation may call the teacher without a
        # student draft. Normal low/high/random runs must pass SOURCE + DRAFT.
        raw_random = (
            row.get("selection_rule") == "random_raw_length_bucket_candidate_pool"
            or row.get("student_status") == "skipped_raw_random"
        )
        if not raw_random and not str(row.get("student_translation", "")).strip():
            missing.append(str(row.get("id", "")))
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            "teacher candidates are missing student_translation drafts; "
            f"first_missing_ids={preview} count={len(missing)}"
        )


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
    last_status: int | None = None
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, timeout=(30, 180))
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"Gemini API network error: {last_error}") from exc
        last_status = int(response.status_code)
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
    raise RuntimeError(f"Gemini API error {last_status}: {last_error}")


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
            item = _normalize_teacher_output_item(item)
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
    reject_over_max_output_tokens = bool(teacher_cfg.get("reject_over_max_output_tokens", True))
    target_max_output_tokens = max(
        1,
        int(teacher_cfg.get("target_max_output_tokens", _get(cfg, "data.max_output_tokens", 1500)) or 1500),
    )
    temperature = float(teacher_cfg.get("temperature", 0.0) or 0.0)
    refill_until_target = bool(teacher_cfg.get("refill_until_target", True))
    abort_on_all_failed_window = bool(teacher_cfg.get("abort_on_all_failed_window", True))

    system_prompt = _teacher_system_prompt(teacher_cfg)
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
    translation_tokenizer = _load_translation_tokenizer(cfg) if reject_over_max_output_tokens else None
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    _validate_teacher_candidate_drafts(candidates)
    bucket_plan = _teacher_bucket_plan(cfg=cfg, candidates=candidates, target=target)
    buckets = bucket_plan["buckets"]
    quotas = list(bucket_plan["quotas"])
    accepted_bucket_counts = [0 for _ in buckets]
    requested_bucket_counts = [0 for _ in buckets]
    bucket_queues = _teacher_bucket_queues(candidates=candidates, buckets=buckets)
    estimated_batches = (len(candidates) + batch_size - 1) // batch_size
    progress(
        "teacher start",
        candidates=len(candidates),
        batches=estimated_batches,
        target=target,
        batch_size=batch_size,
        max_workers=max_workers,
        bucket_strategy=bucket_plan["quota_strategy"],
        target_max_output_tokens=target_max_output_tokens if reject_over_max_output_tokens else "disabled",
    )
    called_batches = 0
    requested_candidate_rows = 0
    batch_idx_cursor = 0
    batch_window = max_workers if refill_until_target else max(estimated_batches, 1)

    while any(bucket_queues):
        if refill_until_target and len(accepted) >= target:
            break
        scheduled_bucket_counts = [0 for _ in buckets]
        batch_inputs: list[tuple[int, dict[str, Any], str, str, list[dict[str, Any]], set[str]]] = []
        batch_rows_by_idx: dict[int, list[dict[str, Any]]] = {}
        for _ in range(batch_window):
            batch_rows = _pop_teacher_batch_rows(
                bucket_queues=bucket_queues,
                quotas=quotas,
                accepted_counts=accepted_bucket_counts,
                scheduled_counts=scheduled_bucket_counts,
                batch_size=batch_size,
                fill_remainder_from_global=bool(bucket_plan["fill_remainder_from_global"]),
                explicit_weights=bucket_plan["explicit_weights"],
                positive_weight_bucket_indexes=bucket_plan["positive_weight_bucket_indexes"],
                allow_zero_weight_bucket_fill=bool(bucket_plan["allow_zero_weight_bucket_fill"]),
            )
            if not batch_rows:
                break
            batch_idx = batch_idx_cursor
            batch_idx_cursor += 1
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
            for candidate in batch_rows:
                bucket_idx = int(candidate.get("_teacher_bucket_idx", 0))
                requested_bucket_counts[bucket_idx] += 1
        if not batch_inputs:
            break

        with progress_context(
            "teacher api-window",
            start_batch=called_batches,
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
                bucket_idx = int(candidate.get("_teacher_bucket_idx", 0))
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
                translation_tokens = None
                if translation_tokenizer is not None:
                    translation_tokens = _translation_token_count(translation_tokenizer, translation)
                    if translation_tokens > target_max_output_tokens:
                        rejected.append(
                            _reject_row(
                                candidate,
                                reason="teacher_output_too_long",
                                flags=["teacher_output_too_long"],
                                item=item,
                                extra={
                                    "teacher_translation_tokens": translation_tokens,
                                    "teacher_max_output_tokens": target_max_output_tokens,
                                },
                            )
                        )
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
                accepted_bucket_counts[bucket_idx] += 1
                accepted.append(
                    _accept_row(
                        candidate,
                        item=item,
                        rank=len(accepted) + 1,
                        teacher_translation_tokens=translation_tokens,
                    )
                )

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
                f"aborting before SFT. start_batch={max(0, called_batches - len(results))} "
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
        "teacher_candidate_batches": estimated_batches,
        "teacher_batches": called_batches,
        "teacher_requested_candidate_rows": requested_candidate_rows,
        "teacher_skipped_candidate_rows": max(0, len(candidates) - requested_candidate_rows),
        "teacher_accepted_rows": len(accepted),
        "teacher_rejected_rows": len(rejected),
        "teacher_label_counts": label_summary["label_counts"],
        "teacher_label_ratios": label_summary["label_ratios"],
        "teacher_reject_reason_counts": rejection_counts["reject_reason_counts"],
        "teacher_reject_flag_counts": rejection_counts["reject_flag_counts"],
        "teacher_bucket_quota_strategy": bucket_plan["quota_strategy"],
        "teacher_bucket_candidate_counts": _bucket_count_summary(
            buckets=buckets,
            counts=list(bucket_plan["bucket_counts"]),
        ),
        "teacher_bucket_target_quotas": _bucket_count_summary(buckets=buckets, counts=quotas),
        "teacher_bucket_requested_counts": _bucket_count_summary(
            buckets=buckets,
            counts=requested_bucket_counts,
        ),
        "teacher_bucket_accepted_counts": _bucket_count_summary(
            buckets=buckets,
            counts=accepted_bucket_counts,
        ),
        "teacher_target_rows": target,
        "teacher_shortfall_rows": shortfall,
        "teacher_exhausted_candidate_pool": bool(shortfall and requested_candidate_rows >= len(candidates)),
        "teacher_refill_until_target": refill_until_target,
        "teacher_degeneration_filter_enabled": teacher_filter_enabled,
        "teacher_reject_over_max_output_tokens": reject_over_max_output_tokens,
        "teacher_target_max_output_tokens": target_max_output_tokens,
        "teacher_draft_format_policy_path": teacher_cfg.get("draft_format_policy_path"),
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
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
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
    if extra:
        row.update(dict(extra))
    return row


def _accept_row(
    candidate: Mapping[str, Any],
    *,
    item: TeacherOutputItem,
    rank: int,
    teacher_translation_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "source": candidate.get("source", ""),
        "target": item.final_translation,
        "student_translation": candidate.get("student_translation", ""),
        "teacher_label": item.label,
        "teacher_errors": _item_errors(item),
        "teacher_accept_rank": rank,
        "teacher_translation_tokens": teacher_translation_tokens,
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
