#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config, config_hash, save_effective_config
from degeneration_filter import classify_student_output
from io_utils import read_jsonl, write_jsonl
from progress import progress, progress_context
from prompting import load_student_templates, render_student_prompt
from qe_score import comet_scores
from runtime_logging import configure_runtime_logging
from sft_dataset import write_sft_dataset
from sft_train import run_sft_training
from teacher_generation import run_teacher_generation


configure_runtime_logging()

TRAIN_PHASES = (
    "input",
    "student-infer",
    "student-filter",
    "qe-select",
    "teacher",
    "sft-dataset",
    "sft",
)


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _save_all_step_artifacts(cfg: Mapping[str, Any]) -> bool:
    return bool(_get(cfg, "logging.save_all_step_artifacts", False))


def _remove_path_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _cleanup_compact_subset_artifacts(cfg: Mapping[str, Any], subset_dir: Path) -> None:
    if _save_all_step_artifacts(cfg):
        return
    rel_paths: list[str] = []
    if (subset_dir / "student_records.jsonl").exists():
        rel_paths.extend(
            [
                "input.jsonl",
                "student_translations.jsonl",
                "student_filtered.jsonl",
                "qe_scores.jsonl",
                "selected_for_teacher.jsonl",
                "filter_blocked_selection.jsonl",
                "runtime_io/infer-student.input.jsonl",
                "runtime_io/infer-student.output.jsonl",
                "runtime_io/qe-selection.input.jsonl",
                "runtime_io/qe-selection.output.jsonl",
            ]
        )
    if (subset_dir / "teacher_artifacts.jsonl").exists():
        rel_paths.extend(
            [
                "teacher_requests.jsonl",
                "teacher_responses.raw.jsonl",
                "teacher_parsed.jsonl",
                "teacher_rejected.jsonl",
            ]
        )
    for rel_path in rel_paths:
        _remove_path_if_exists(subset_dir / rel_path)


def _subset_root(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return Path(str(_get(cfg, "paths.subset_dir"))) / f"subset_{subset_idx:03d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_start_from_phase(start_from_phase: str | None) -> str | None:
    if start_from_phase is None:
        return None
    if start_from_phase not in TRAIN_PHASES:
        raise SystemExit(
            f"invalid start-from-phase={start_from_phase!r}; "
            f"valid phases: {', '.join(TRAIN_PHASES)}"
        )
    return start_from_phase


def _skipped_phases(start_from_phase: str | None) -> set[str]:
    start_from_phase = _validate_start_from_phase(start_from_phase)
    if start_from_phase is None:
        return set()
    return set(TRAIN_PHASES[:TRAIN_PHASES.index(start_from_phase)])


def _rewind_skipped_phases(skipped: set[str], phase: str) -> set[str]:
    if phase not in TRAIN_PHASES:
        return set(skipped)
    phase_idx = TRAIN_PHASES.index(phase)
    return {item for item in skipped if TRAIN_PHASES.index(item) < phase_idx}


def _require_jsonl(path: Path, artifact_name: str) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"cannot resume: {artifact_name} not found at {path}")
    return read_jsonl(path)


def _require_jsonl_file(path: Path, artifact_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"cannot resume: {artifact_name} not found at {path}")
    return read_jsonl(path)


def _require_json(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"cannot resume: {artifact_name} not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"cannot resume: {artifact_name} at {path} must be a JSON object")
    return data


def _phase_state_path(subset_dir: Path) -> Path:
    return subset_dir / "phase_state.json"


def _write_phase_state(
    *,
    cfg: Mapping[str, Any],
    subset_dir: Path,
    subset_idx: int,
    phase: str,
    status: str,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_id": _get(cfg, "run.id"),
        "subset_idx": subset_idx,
        "phase": phase,
        "status": status,
        "updated_at": _utc_now(),
    }
    if error:
        payload["error"] = error
    _phase_state_path(subset_dir).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_phase_state(subset_dir: Path) -> dict[str, Any] | None:
    path = _phase_state_path(subset_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _next_phase(phase: str) -> str | None:
    if phase not in TRAIN_PHASES:
        return None
    idx = TRAIN_PHASES.index(phase) + 1
    if idx >= len(TRAIN_PHASES):
        return None
    return TRAIN_PHASES[idx]


def _phase_to_resume_from_state(state: Mapping[str, Any]) -> str | None:
    phase = str(state.get("phase", ""))
    status = str(state.get("status", ""))
    if phase not in TRAIN_PHASES:
        return None
    if status in {"running", "failed"}:
        return phase
    if status == "phase_completed":
        return _next_phase(phase)
    return None


def _subset_idx_from_dir(path: Path) -> int | None:
    name = path.name
    if not name.startswith("subset_"):
        return None
    suffix = name.removeprefix("subset_")
    try:
        return int(suffix)
    except ValueError:
        return None


def _auto_resume_target(
    *,
    cfg: Mapping[str, Any],
    default_subset_idx: int,
    explicit_subset_idx: int | None,
) -> tuple[int, str | None]:
    subset_base = Path(str(_get(cfg, "paths.subset_dir")))
    subset_dirs: list[tuple[int, Path]] = []
    if explicit_subset_idx is not None:
        subset_dirs.append((explicit_subset_idx, _subset_root(cfg, explicit_subset_idx)))
    elif subset_base.exists():
        for path in subset_base.glob("subset_*"):
            if not path.is_dir():
                continue
            subset_idx = _subset_idx_from_dir(path)
            if subset_idx is not None:
                subset_dirs.append((subset_idx, path))
    for subset_idx, subset_dir in sorted(subset_dirs, key=lambda item: item[0], reverse=True):
        state = _read_phase_state(subset_dir)
        if not state:
            continue
        phase = _phase_to_resume_from_state(state)
        if phase is not None:
            print(
                f"[resume] auto: subset_{subset_idx:03d} from phase {phase} "
                f"(previous status={state.get('status')})",
                file=sys.stderr,
            )
            return subset_idx, phase
    return default_subset_idx, None


def _run_tracked_phase(
    *,
    cfg: Mapping[str, Any],
    subset_dir: Path,
    subset_idx: int,
    phase: str,
    func: Any,
) -> Any:
    event = f"subset_{subset_idx:03d}/{phase}"
    with progress_context(event, run=_get(cfg, "run.id")):
        _write_phase_state(
            cfg=cfg,
            subset_dir=subset_dir,
            subset_idx=subset_idx,
            phase=phase,
            status="running",
        )
        try:
            result = func()
        except BaseException as exc:
            _write_phase_state(
                cfg=cfg,
                subset_dir=subset_dir,
                subset_idx=subset_idx,
                phase=phase,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _write_phase_state(
            cfg=cfg,
            subset_dir=subset_dir,
            subset_idx=subset_idx,
            phase=phase,
            status="phase_completed",
        )
    return result


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("missing pyarrow; run `make set` first") from exc

    rows: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(str(path))
    for batch in parquet.iter_batches(batch_size=8192):
        rows.extend(dict(row) for row in pa.Table.from_batches([batch]).to_pylist())
    return rows


def _candidate_data_files(cfg: Mapping[str, Any], data_path: str | None) -> list[Path]:
    if data_path:
        path = Path(data_path)
        if path.is_dir():
            files = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.jsonl"))
        else:
            files = [path]
    else:
        local_dir = Path(str(_get(cfg, "data.prepared_download.local_dir")))
        files = sorted(local_dir.rglob("*.parquet")) + sorted(local_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit("no prepared data files found; run `make download-prepared-data` first")
    train_like = [p for p in files if "train" in p.name.lower() or "train" in str(p.parent).lower()]
    return train_like or files


def _load_pool_rows(cfg: Mapping[str, Any], data_path: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _candidate_data_files(cfg, data_path):
        if path.suffix.lower() == ".parquet":
            rows.extend(_read_parquet_rows(path))
        elif path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl(path))
    source_field = str(_get(cfg, "data.source_field", "source"))
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source = row.get(source_field)
        if not isinstance(source, str) or not source.strip():
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            row_id = f"row_{idx:012d}"
        metadata = row.get("metadata")
        normalized.append(
            {
                "id": row_id,
                "source": source,
                "target": row.get(str(_get(cfg, "data.target_field", "target"))),
                "source_tokens": row.get("source_tokens"),
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            }
        )
    if not normalized:
        raise SystemExit("prepared data did not contain usable source rows")
    return normalized


def _select_subset(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    subset_idx: int,
    subset_size: int,
) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    start = subset_idx * subset_size
    end = start + subset_size
    return shuffled[start:end]


def _materialize_input(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    subset_size: int,
    data_path: str | None,
    force: bool,
) -> list[dict[str, Any]]:
    subset_dir = _subset_root(cfg, subset_idx)
    input_path = subset_dir / "input.jsonl"
    if input_path.exists() and input_path.stat().st_size > 0 and not force:
        return read_jsonl(input_path)
    rows = _load_pool_rows(cfg, data_path)
    selected = _select_subset(
        rows,
        seed=int(_get(cfg, "run.seed", 42)),
        subset_idx=subset_idx,
        subset_size=subset_size,
    )
    if not selected:
        raise SystemExit(f"subset_{subset_idx:03d} is empty")
    write_jsonl(input_path, selected)
    return selected


def _checkpoint_dir(cfg: Mapping[str, Any]) -> Path:
    return Path(str(_get(cfg, "paths.checkpoint_dir")))


def _is_lora_adapter_path(path: Path) -> bool:
    return path.is_dir() and (
        (path / "adapter_config.json").exists()
        or (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


def _is_full_model_checkpoint_path(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _is_full_inference_model_path(path: Path) -> bool:
    if not _is_full_model_checkpoint_path(path):
        return False
    return any(
        (path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
        )
    )


def _previous_subset_stage_state(cfg: Mapping[str, Any], subset_idx: int) -> tuple[int, dict[str, Any]] | None:
    run_subset_start = int(_get(cfg, "run.subset_start", 0) or 0)
    if subset_idx <= run_subset_start:
        return None
    previous_subset_idx = subset_idx - 1
    state_path = _checkpoint_dir(cfg) / f"sft_stage_state_subset_{previous_subset_idx:03d}.json"
    state = _require_json(state_path, f"sft_stage_state_subset_{previous_subset_idx:03d}.json")
    if state.get("status") != "completed":
        raise SystemExit(
            f"cannot run subset_{subset_idx:03d} student inference: "
            f"previous subset_{previous_subset_idx:03d} SFT is not completed"
        )
    return previous_subset_idx, state


def _previous_subset_lora_adapter_path(cfg: Mapping[str, Any], subset_idx: int) -> Path | None:
    if str(_get(cfg, "training.tuning_mode", "")).strip().lower() != "lora":
        return None
    previous = _previous_subset_stage_state(cfg, subset_idx)
    if previous is None:
        return None
    previous_subset_idx, state = previous
    checkpoint_dir = Path(str(state.get("checkpoint_dir", "")))
    if not _is_lora_adapter_path(checkpoint_dir):
        raise SystemExit(
            f"cannot run subset_{subset_idx:03d} student inference with LoRA: "
            f"previous checkpoint is not a LoRA adapter directory: {checkpoint_dir}"
        )
    return checkpoint_dir


def _previous_subset_full_inference_model(
    cfg: Mapping[str, Any],
    subset_idx: int,
) -> tuple[Path, int, int] | None:
    if str(_get(cfg, "training.tuning_mode", "")).strip().lower() != "full":
        return None
    previous = _previous_subset_stage_state(cfg, subset_idx)
    if previous is None:
        return None
    previous_subset_idx, state = previous
    checkpoint_dir = Path(str(state.get("checkpoint_dir", "")))
    if not _is_full_model_checkpoint_path(checkpoint_dir):
        raise SystemExit(
            f"cannot run subset_{subset_idx:03d} student inference with full tuning: "
            f"previous subset_{previous_subset_idx:03d} optimizer checkpoint is not a full model: "
            f"{checkpoint_dir}"
        )
    final_dir = _checkpoint_dir(cfg) / "final"
    if not _is_full_inference_model_path(final_dir):
        raise SystemExit(
            f"cannot run subset_{subset_idx:03d} student inference with full tuning: "
            f"previous subset_{previous_subset_idx:03d} final model is not loadable: {final_dir}"
        )
    marker_path = final_dir / "dqs_stage_model.json"
    marker = _require_json(marker_path, "dqs_stage_model.json")
    marker_subset_idx = marker.get("subset_idx")
    marker_global_step = marker.get("global_step")
    state_global_step = state.get("actual_global_step")
    if marker_subset_idx != previous_subset_idx or marker_global_step != state_global_step:
        raise SystemExit(
            f"cannot run subset_{subset_idx:03d} student inference with full tuning: "
            f"final model marker does not match previous subset state; "
            f"expected_subset={previous_subset_idx} actual_subset={marker_subset_idx} "
            f"expected_step={state_global_step} actual_step={marker_global_step}"
        )
    return final_dir, previous_subset_idx, int(marker_global_step)


def _student_inference_model_cfg(cfg: Mapping[str, Any], subset_idx: int) -> dict[str, Any]:
    model_cfg = _get(cfg, "model", {})
    out = dict(model_cfg) if isinstance(model_cfg, Mapping) else {}
    full_inference_model = _previous_subset_full_inference_model(cfg, subset_idx)
    if full_inference_model is not None:
        full_model_path, checkpoint_subset_idx, checkpoint_global_step = full_inference_model
        out["name_or_path"] = str(full_model_path)
        out.pop("lora_adapter_path", None)
        out["dqs_checkpoint_subset_idx"] = checkpoint_subset_idx
        out["dqs_checkpoint_global_step"] = checkpoint_global_step
        return out
    adapter_path = _previous_subset_lora_adapter_path(cfg, subset_idx)
    if adapter_path is not None:
        out["lora_adapter_path"] = str(adapter_path)
    return out


def _student_request_model_error(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    requests: list[dict[str, Any]],
) -> str | None:
    if not requests:
        return "student inference request file is empty"
    expected = _student_inference_model_cfg(cfg, subset_idx)
    expected_model = str(expected.get("name_or_path", ""))
    expected_adapter = str(expected.get("lora_adapter_path", "") or "")
    expected_checkpoint_subset = expected.get("dqs_checkpoint_subset_idx")
    expected_checkpoint_step = expected.get("dqs_checkpoint_global_step")
    for idx, request in enumerate(requests):
        model = request.get("model", {})
        if not isinstance(model, Mapping):
            return f"student inference request {idx} has invalid model config"
        actual_model = str(model.get("name_or_path", ""))
        actual_adapter = str(model.get("lora_adapter_path", "") or "")
        if actual_model != expected_model:
            return (
                f"student inference request {idx} model mismatch: "
                f"expected={expected_model} actual={actual_model}"
            )
        if actual_adapter != expected_adapter:
            return (
                f"student inference request {idx} LoRA adapter mismatch: "
                f"expected={expected_adapter or '<none>'} actual={actual_adapter or '<none>'}"
            )
        if model.get("dqs_checkpoint_subset_idx") != expected_checkpoint_subset:
            return (
                f"student inference request {idx} checkpoint subset mismatch: "
                f"expected={expected_checkpoint_subset} actual={model.get('dqs_checkpoint_subset_idx')}"
            )
        if model.get("dqs_checkpoint_global_step") != expected_checkpoint_step:
            return (
                f"student inference request {idx} checkpoint step mismatch: "
                f"expected={expected_checkpoint_step} actual={model.get('dqs_checkpoint_global_step')}"
            )
    return None


def _remove_student_inference_dependents(subset_dir: Path) -> None:
    for rel_path in (
        "runtime_io/infer-student.output.jsonl",
        "student_translations.jsonl",
        "student_filtered.jsonl",
        "student_filter_summary.json",
        "student_records.jsonl",
        "runtime_io/qe-selection.input.jsonl",
        "runtime_io/qe-selection.output.jsonl",
        "qe_scores.jsonl",
        "selected_for_teacher.jsonl",
        "filter_blocked_selection.jsonl",
        "teacher_artifacts.jsonl",
        "teacher_requests.jsonl",
        "teacher_responses.raw.jsonl",
        "teacher_parsed.jsonl",
        "teacher_rejected.jsonl",
        "golden_pairs.jsonl",
        "sft_train.jsonl",
    ):
        _remove_path_if_exists(subset_dir / rel_path)


def _build_inference_requests(
    *,
    cfg: Mapping[str, Any],
    rows: list[dict[str, Any]],
    subset_idx: int,
    force: bool,
) -> list[dict[str, Any]]:
    subset_dir = _subset_root(cfg, subset_idx)
    request_path = subset_dir / "runtime_io" / "infer-student.input.jsonl"
    if request_path.exists() and request_path.stat().st_size > 0 and not force:
        existing_requests = read_jsonl(request_path)
        request_error = _student_request_model_error(
            cfg=cfg,
            subset_idx=subset_idx,
            requests=existing_requests,
        )
        if request_error is None:
            return existing_requests
        print(
            f"[resume] invalid existing student inference request; rebuilding: {request_error}",
            file=sys.stderr,
        )
        _remove_student_inference_dependents(subset_dir)

    prompt_cfg = _get(cfg, "prompts", {})
    if not isinstance(prompt_cfg, Mapping):
        raise SystemExit("prompts config must be a mapping")
    template_path = Path(str(prompt_cfg.get("student_templates_path", "prompts/student_templates.yaml")))
    template_cfg = load_student_templates(template_path)
    inference_cfg = _get(cfg, "inference", {})
    student_model_cfg = _student_inference_model_cfg(cfg, subset_idx)
    progress(
        "student inference model",
        subset=f"subset_{subset_idx:03d}",
        model=student_model_cfg.get("name_or_path"),
        lora_adapter=student_model_cfg.get("lora_adapter_path"),
    )

    requests: list[dict[str, Any]] = []
    for order_idx, row in enumerate(rows):
        row_id = str(row["id"])
        rendered = render_student_prompt(
            template_cfg=template_cfg,
            prompt_cfg=prompt_cfg,
            model_cfg=student_model_cfg,
            source=str(row["source"]),
            row_id=row_id,
            subset_idx=subset_idx,
        )
        requests.append(
            {
                "id": f"{_get(cfg, 'run.id')}/subsets/subset_{subset_idx:03d}/{row_id}/student",
                "run_id": str(_get(cfg, "run.id")),
                "subset_idx": subset_idx,
                "row_id": row_id,
                "order_idx": order_idx,
                "source": row["source"],
                "metadata": row.get("metadata", {}),
                "prompt": rendered.text,
                "prompt_template_id": rendered.template_id,
                "prompt_template_group": rendered.template_group,
                "prompt_template_hash": rendered.template_hash,
                "chat_template_applied": rendered.chat_template_applied,
                "model": dict(student_model_cfg),
                "inference": dict(inference_cfg) if isinstance(inference_cfg, Mapping) else {},
                "decoding": {
                    "temperature": _get(cfg, "inference.temperature", 0.0),
                    "top_p": _get(cfg, "inference.top_p", 1.0),
                    "max_new_tokens": _get(cfg, "inference.max_new_tokens", 1500),
                },
            }
        )
    write_jsonl(request_path, requests)
    return requests


def _validate_vllm_outputs(
    *,
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    allow_dry_run: bool,
) -> tuple[bool, str, list[dict[str, Any]]]:
    request_order_by_id: dict[str, int] = {}
    request_row_id_by_id: dict[str, str] = {}
    for idx, request in enumerate(requests):
        request_id = str(request.get("id", ""))
        if not request_id:
            return False, f"request at index {idx} is missing id", []
        if request_id in request_order_by_id:
            return False, f"duplicate request id: {request_id}", []
        request_order_by_id[request_id] = int(request.get("order_idx", idx))
        request_row_id_by_id[request_id] = str(request.get("row_id", ""))

    response_by_id: dict[str, dict[str, Any]] = {}
    duplicate_response_ids: set[str] = set()
    dry_run_ids: list[str] = []
    mismatched_row_ids: list[str] = []
    for idx, response in enumerate(responses):
        response_id = str(response.get("id", ""))
        if not response_id:
            return False, f"response at index {idx} is missing id", []
        if response_id in response_by_id:
            duplicate_response_ids.add(response_id)
        response_by_id[response_id] = dict(response)
        if not allow_dry_run and response.get("status") == "dry_run":
            dry_run_ids.append(response_id)
        expected_row_id = request_row_id_by_id.get(response_id)
        if expected_row_id is not None and str(response.get("row_id", "")) != expected_row_id:
            mismatched_row_ids.append(response_id)

    if duplicate_response_ids:
        sample = sorted(duplicate_response_ids)[:3]
        return False, f"duplicate response ids: {sample}", []
    request_ids = set(request_order_by_id)
    response_ids = set(response_by_id)
    missing_ids = sorted(request_ids - response_ids)
    extra_ids = sorted(response_ids - request_ids)
    if missing_ids:
        return False, f"missing response ids: {missing_ids[:3]} (count={len(missing_ids)})", []
    if extra_ids:
        return False, f"extra response ids: {extra_ids[:3]} (count={len(extra_ids)})", []
    if dry_run_ids:
        return False, f"dry-run responses cannot satisfy real vLLM output: {dry_run_ids[:3]}", []
    if mismatched_row_ids:
        return False, f"response row_id mismatch: {mismatched_row_ids[:3]}", []
    if len(responses) != len(requests):
        return False, f"row count mismatch: expected={len(requests)} actual={len(responses)}", []

    normalized: list[dict[str, Any]] = []
    for request_id, order_idx in sorted(request_order_by_id.items(), key=lambda item: item[1]):
        row = dict(response_by_id[request_id])
        row["order_idx"] = order_idx
        normalized.append(row)
    return True, "ok", normalized


def _validate_or_raise_vllm_outputs(
    *,
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    output_path: Path,
    allow_dry_run: bool = False,
) -> list[dict[str, Any]]:
    valid, reason, normalized = _validate_vllm_outputs(
        requests=requests,
        responses=responses,
        allow_dry_run=allow_dry_run,
    )
    if not valid:
        raise SystemExit(f"vLLM output validation failed: {output_path}: {reason}")
    return normalized


def _validated_existing_vllm_output_or_none(
    *,
    requests: list[dict[str, Any]],
    output_path: Path,
    allow_dry_run: bool,
) -> list[dict[str, Any]] | None:
    try:
        existing = read_jsonl(output_path)
    except (OSError, ValueError) as exc:
        print(f"[resume] invalid existing vLLM output; rerunning {output_path}: {exc}", file=sys.stderr)
        return None
    valid, reason, normalized = _validate_vllm_outputs(
        requests=requests,
        responses=existing,
        allow_dry_run=allow_dry_run,
    )
    if valid:
        return normalized
    print(f"[resume] invalid existing vLLM output; rerunning {output_path}: {reason}", file=sys.stderr)
    return None


def _run_vllm(request_path: Path, output_path: Path, *, force: bool) -> list[dict[str, Any]]:
    requests = read_jsonl(request_path)
    progress("vllm dispatch", requests=len(requests), input=request_path, output=output_path)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        normalized = _validated_existing_vllm_output_or_none(
            requests=requests,
            output_path=output_path,
            allow_dry_run=False,
        )
        if normalized is not None:
            return normalized
    shard_groups, shard_strategy = _resolve_vllm_shards(requests)
    if len(shard_groups) > 1 and len(requests) > 1:
        responses = _run_vllm_sharded(
            requests=requests,
            request_path=request_path,
            output_path=output_path,
            shard_groups=shard_groups,
            shard_strategy=shard_strategy,
        )
        return _validate_or_raise_vllm_outputs(
            requests=requests,
            responses=responses,
            output_path=output_path,
        )
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("vllm_inference.py")),
        "--input",
        str(request_path),
        "--output",
        str(output_path),
    ]
    with progress_context("vllm subprocess", requests=len(requests)):
        subprocess.run(cmd, check=True)
    responses = read_jsonl(output_path)
    return _validate_or_raise_vllm_outputs(
        requests=requests,
        responses=responses,
        output_path=output_path,
    )


def _sft_training_summary_path(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return Path(str(_get(cfg, "paths.checkpoint_dir"))) / f"sft_training_summary_subset_{subset_idx:03d}.json"


def _run_sft_training_phase(
    *,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    subset_idx: int,
    dataset_path: Path,
) -> dict[str, Any]:
    sft_rows = read_jsonl(dataset_path)
    if not sft_rows:
        raise SystemExit(f"SFT dataset is empty before training launch: {dataset_path}")

    nproc_per_node = max(1, int(args.sft_nproc_per_node or 1))
    if nproc_per_node == 1:
        return run_sft_training(
            cfg=cfg,
            subset_idx=subset_idx,
            dataset_path=dataset_path,
            stage_scheduler_total_steps=args.sft_scheduler_total_steps,
            force_save_checkpoint=args.sft_force_save_checkpoint,
        )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        str(Path(__file__).with_name("sft_train.py")),
        "--config",
        args.config,
        "--subset-idx",
        str(subset_idx),
        "--dataset-path",
        str(dataset_path),
    ]
    if args.sft_scheduler_total_steps is not None:
        cmd.extend(["--stage-scheduler-total-steps", str(args.sft_scheduler_total_steps)])
    if args.sft_force_save_checkpoint:
        cmd.append("--force-save-checkpoint")
    for override in args.override:
        cmd.extend(["--override", override])

    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    progress("sft ddp dispatch", subset=f"subset_{subset_idx:03d}", nproc_per_node=nproc_per_node)
    subprocess.run(cmd, check=True, env=env)

    summary_path = _sft_training_summary_path(cfg, subset_idx)
    if not summary_path.exists() or summary_path.stat().st_size <= 0:
        raise SystemExit(f"SFT DDP finished but summary was not written: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise SystemExit(f"SFT DDP summary must be a JSON object: {summary_path}")
    return summary


def _resolve_gpu_ids(inference_cfg: Mapping[str, Any]) -> list[int]:
    raw_gpu_ids = inference_cfg.get("gpu_ids")
    if raw_gpu_ids is None:
        num_gpus = int(inference_cfg.get("num_gpus", 1) or 1)
        if num_gpus < 1:
            raise SystemExit("inference.num_gpus must be >= 1")
        return list(range(num_gpus))
    if not isinstance(raw_gpu_ids, list):
        raise SystemExit("inference.gpu_ids must be null or a list of GPU ids")
    gpu_ids: list[int] = []
    for idx, gpu_id in enumerate(raw_gpu_ids):
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise SystemExit(f"inference.gpu_ids[{idx}] must be a non-negative integer")
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise SystemExit("inference.gpu_ids must not be empty when set")
    return gpu_ids


def _resolve_vllm_shards(requests: list[dict[str, Any]]) -> tuple[list[list[int]], str]:
    if not requests:
        return [[]], "order_split"
    inference_cfg = requests[0].get("inference", {})
    if not isinstance(inference_cfg, Mapping):
        inference_cfg = {}
    gpu_ids = _resolve_gpu_ids(inference_cfg)
    tensor_parallel_size = int(inference_cfg.get("tensor_parallel_size", 1) or 1)
    if tensor_parallel_size < 1:
        raise SystemExit("inference.tensor_parallel_size must be >= 1")
    if len(gpu_ids) % tensor_parallel_size != 0:
        raise SystemExit(
            "inference GPU count must be divisible by tensor_parallel_size: "
            f"gpu_ids={gpu_ids}, tensor_parallel_size={tensor_parallel_size}"
        )
    shard_strategy = str(inference_cfg.get("shard_strategy", "order_split")).strip() or "order_split"
    if shard_strategy not in {"order_split", "row_id_hash"}:
        raise SystemExit("inference.shard_strategy must be one of: order_split, row_id_hash")
    groups = [
        gpu_ids[idx : idx + tensor_parallel_size]
        for idx in range(0, len(gpu_ids), tensor_parallel_size)
    ]
    replicas = inference_cfg.get("data_parallel_replicas", "auto")
    if replicas != "auto":
        try:
            replica_count = int(replicas)
        except (TypeError, ValueError) as exc:
            raise SystemExit("inference.data_parallel_replicas must be 'auto' or an integer") from exc
        if replica_count != len(groups):
            raise SystemExit(
                "inference.data_parallel_replicas does not match num_gpus / tensor_parallel_size: "
                f"data_parallel_replicas={replica_count}, computed={len(groups)}"
            )
    return groups, shard_strategy


def _stable_shard_index(
    *,
    row: Mapping[str, Any],
    order_idx: int,
    shard_count: int,
    shard_strategy: str,
    total_rows: int,
) -> int:
    if shard_count <= 1:
        return 0
    if shard_strategy == "row_id_hash":
        row_id = str(row.get("row_id", row.get("id", order_idx)))
        digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % shard_count
    return min((order_idx * shard_count) // max(total_rows, 1), shard_count - 1)


def _request_for_shard(request: Mapping[str, Any], *, shard_gpu_count: int) -> dict[str, Any]:
    out = dict(request)
    inference_cfg = out.get("inference", {})
    if isinstance(inference_cfg, Mapping):
        local_inference = dict(inference_cfg)
        local_inference["num_gpus"] = shard_gpu_count
        out["inference"] = local_inference
    return out


def _run_vllm_part(input_path: Path, output_path: Path, gpu_ids: list[int]) -> list[dict[str, Any]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("vllm_inference.py")),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    with progress_context("vllm shard", gpus=",".join(str(gpu_id) for gpu_id in gpu_ids), input=input_path):
        subprocess.run(cmd, check=True, env=env)
    return read_jsonl(output_path)


def _run_vllm_sharded(
    *,
    requests: list[dict[str, Any]],
    request_path: Path,
    output_path: Path,
    shard_groups: list[list[int]],
    shard_strategy: str,
) -> list[dict[str, Any]]:
    runtime_dir = request_path.parent
    shard_rows: list[list[dict[str, Any]]] = [[] for _ in shard_groups]
    total_rows = len(requests)
    for idx, request in enumerate(requests):
        order_idx = int(request.get("order_idx", idx))
        shard_idx = _stable_shard_index(
            row=request,
            order_idx=order_idx,
            shard_count=len(shard_groups),
            shard_strategy=shard_strategy,
            total_rows=total_rows,
        )
        shard_rows[shard_idx].append(_request_for_shard(request, shard_gpu_count=len(shard_groups[shard_idx])))

    futures = []
    with ThreadPoolExecutor(max_workers=len(shard_groups)) as executor:
        for shard_idx, gpu_ids in enumerate(shard_groups):
            rows = shard_rows[shard_idx]
            if not rows:
                continue
            gpu_label = "_".join(str(gpu_id) for gpu_id in gpu_ids)
            input_part = runtime_dir / f"{request_path.stem}.part{shard_idx:03d}.gpu{gpu_label}.jsonl"
            output_part = runtime_dir / f"{output_path.stem}.part{shard_idx:03d}.gpu{gpu_label}.jsonl"
            write_jsonl(input_part, rows)
            futures.append(executor.submit(_run_vllm_part, input_part, output_part, gpu_ids))

    part_rows: list[dict[str, Any]] = []
    for future in futures:
        part_rows.extend(future.result())
    merged = _merge_vllm_shard_outputs(requests=requests, responses=part_rows)
    write_jsonl(output_path, merged)
    return merged


def _merge_vllm_shard_outputs(
    *,
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(responses) != len(requests):
        raise SystemExit(
            f"merged vLLM shard output row count mismatch: expected={len(requests)} actual={len(responses)}"
        )
    order_idx_by_id: dict[str, int] = {}
    for idx, request in enumerate(requests):
        request_id = str(request.get("id", ""))
        if not request_id:
            raise SystemExit("vLLM request row missing id")
        order_idx_by_id[request_id] = int(request.get("order_idx", idx))

    by_order_idx: dict[int, dict[str, Any]] = {}
    for response in responses:
        response_id = str(response.get("id", ""))
        if response_id not in order_idx_by_id:
            raise SystemExit(f"vLLM response id is not in request set: {response_id}")
        order_idx = int(response.get("order_idx", order_idx_by_id[response_id]))
        if order_idx in by_order_idx:
            raise SystemExit(f"duplicate vLLM response order_idx after shard merge: {order_idx}")
        merged = dict(response)
        merged["order_idx"] = order_idx
        by_order_idx[order_idx] = merged

    merged_rows: list[dict[str, Any]] = []
    for order_idx in range(len(requests)):
        row = by_order_idx.get(order_idx)
        if row is None:
            raise SystemExit(f"missing vLLM shard response for order_idx={order_idx}")
        merged_rows.append(row)
    return merged_rows


def _write_dry_run_outputs(requests: list[dict[str, Any]], output_path: Path, *, force: bool) -> list[dict[str, Any]]:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        normalized = _validated_existing_vllm_output_or_none(
            requests=requests,
            output_path=output_path,
            allow_dry_run=True,
        )
        if normalized is not None:
            return normalized
    rows = [
        {
            "id": request["id"],
            "row_id": request["row_id"],
            "order_idx": request["order_idx"],
            "status": "dry_run",
            "mt": "",
            "finish_reason": None,
            "generated_token_count": 0,
            "error": None,
        }
        for request in requests
    ]
    write_jsonl(output_path, rows)
    return rows


def _materialize_student_translations(
    *,
    cfg: Mapping[str, Any],
    input_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    subset_idx: int,
) -> list[dict[str, Any]]:
    by_id = {str(row.get("id")): row for row in responses}
    out_rows: list[dict[str, Any]] = []
    for source_row, request in zip(input_rows, requests):
        response = by_id.get(str(request["id"]))
        if response is None:
            raise SystemExit(f"missing vLLM response for request id={request['id']}")
        out_rows.append(
            {
                "id": source_row["id"],
                "source": source_row["source"],
                "target": source_row.get("target"),
                "metadata": source_row.get("metadata", {}),
                "source_tokens": source_row.get("source_tokens"),
                "student_translation": response.get("mt", ""),
                "student_status": response.get("status", "failed"),
                "student_error": response.get("error"),
                "finish_reason": response.get("finish_reason"),
                "generated_token_count": response.get("generated_token_count", 0),
                "request_id": request["id"],
                "run_id": request.get("run_id"),
                "subset_idx": request.get("subset_idx"),
                "order_idx": request.get("order_idx"),
                "prompt": request["prompt"],
                "prompt_template_id": request["prompt_template_id"],
                "prompt_template_group": request["prompt_template_group"],
                "prompt_template_hash": request["prompt_template_hash"],
                "chat_template_applied": request["chat_template_applied"],
                "model": request.get("model", {}),
                "inference": request.get("inference", {}),
                "decoding": request.get("decoding", {}),
            }
        )
    output_path = _subset_root(cfg, subset_idx) / "student_translations.jsonl"
    write_jsonl(output_path, out_rows)
    return out_rows


def _student_translation_artifact_error(
    *,
    input_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    student_rows: list[dict[str, Any]],
) -> str | None:
    if len(student_rows) != len(input_rows):
        return f"student_translations row count mismatch: expected={len(input_rows)} actual={len(student_rows)}"
    if len(requests) != len(input_rows):
        return f"infer-student input count mismatch: expected={len(input_rows)} actual={len(requests)}"

    if [str(row.get("id", "")) for row in student_rows] != [str(row.get("id", "")) for row in input_rows]:
        return "student_translations row ids do not match input row ids in order"

    request_by_row_id = {str(row.get("row_id", "")): row for row in requests}
    response_by_id = {str(row.get("id", "")): row for row in responses}
    for row in student_rows:
        row_id = str(row.get("id", ""))
        request = request_by_row_id.get(row_id)
        if request is None:
            return f"student row has no matching inference request: row_id={row_id}"
        request_id = str(request.get("id", ""))
        if str(row.get("request_id", "")) != request_id:
            return f"student row request_id mismatch: row_id={row_id}"
        if request_id not in response_by_id:
            return f"student row has no matching inference response: row_id={row_id}"
    return None


def _qe_response_artifact_error(
    *,
    qe_requests: list[dict[str, Any]],
    qe_responses: list[dict[str, Any]],
) -> str | None:
    request_by_id = {str(row.get("id", "")): row for row in qe_requests}
    if len(request_by_id) != len(qe_requests):
        return "qe-selection input has duplicate or missing request ids"

    response_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for response in qe_responses:
        response_id = str(response.get("id", ""))
        if not response_id:
            return "qe-selection output has a response without id"
        if response_id in response_by_id:
            duplicate_ids.add(response_id)
        response_by_id[response_id] = response
    if duplicate_ids:
        return f"qe-selection output has duplicate ids: {sorted(duplicate_ids)[:3]}"

    missing = sorted(set(request_by_id) - set(response_by_id))
    extra = sorted(set(response_by_id) - set(request_by_id))
    if missing:
        return f"qe-selection output is missing responses: {missing[:3]} count={len(missing)}"
    if extra:
        return f"qe-selection output has extra responses: {extra[:3]} count={len(extra)}"
    if len(qe_responses) != len(qe_requests):
        return f"qe-selection row count mismatch: expected={len(qe_requests)} actual={len(qe_responses)}"

    for request in qe_requests:
        response = response_by_id[str(request.get("id", ""))]
        if str(response.get("row_id", "")) != str(request.get("row_id", "")):
            return f"qe-selection row_id mismatch for request={request.get('id')}"
        if response.get("status") != "ok":
            return f"qe-selection response is not ok for request={request.get('id')}: {response.get('status')}"
        if not isinstance(response.get("score"), (int, float)):
            return f"qe-selection response has invalid score for request={request.get('id')}"
    return None


def _qe_selection_artifact_error(
    *,
    subset_dir: Path,
    student_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    student_filter_summary: Mapping[str, Any],
    selected_rows: list[dict[str, Any]],
) -> str | None:
    if len(filter_rows) != len(student_rows):
        return f"student_filtered row count mismatch: expected={len(student_rows)} actual={len(filter_rows)}"
    if [str(row.get("id", "")) for row in filter_rows] != [str(row.get("id", "")) for row in student_rows]:
        return "student_filtered row ids do not match student_translations row ids in order"
    summary_filter_rows = student_filter_summary.get("filter_input_rows")
    if isinstance(summary_filter_rows, int) and summary_filter_rows != len(filter_rows):
        return (
            "student_filter_summary filter_input_rows mismatch: "
            f"expected={len(filter_rows)} actual={summary_filter_rows}"
        )

    qe_requests = _require_jsonl_file(
        subset_dir / "runtime_io" / "qe-selection.input.jsonl",
        "qe-selection.input.jsonl",
    )
    qe_responses = _require_jsonl_file(
        subset_dir / "runtime_io" / "qe-selection.output.jsonl",
        "qe-selection.output.jsonl",
    )
    qe_scores = _require_jsonl_file(subset_dir / "qe_scores.jsonl", "qe_scores.jsonl")
    student_records = _require_jsonl_file(subset_dir / "student_records.jsonl", "student_records.jsonl")

    qe_eligible_ids = [
        str(row.get("id", ""))
        for row in filter_rows
        if row.get("student_status") == "ok" and str(row.get("student_translation", "")).strip()
    ]
    request_row_ids = [str(row.get("row_id", "")) for row in qe_requests]
    if request_row_ids != qe_eligible_ids:
        return "qe-selection input row_ids do not match QE-eligible student rows in order"

    response_error = _qe_response_artifact_error(qe_requests=qe_requests, qe_responses=qe_responses)
    if response_error is not None:
        return response_error

    if len(qe_scores) != len(qe_requests):
        return f"qe_scores row count mismatch: expected={len(qe_requests)} actual={len(qe_scores)}"
    if [str(row.get("request_id", "")) for row in qe_scores] != [str(row.get("id", "")) for row in qe_requests]:
        return "qe_scores request ids do not match qe-selection input ids in order"
    if len(student_records) != len(filter_rows):
        return f"student_records row count mismatch: expected={len(filter_rows)} actual={len(student_records)}"

    clean_ids = {str(row.get("id", "")) for row in filter_rows if row.get("degeneration_label") == "clean"}
    scored_ids = {str(row.get("row_id", "")) for row in qe_responses}
    selected_ids: set[str] = set()
    for row in selected_rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            return "selected_for_teacher contains a row without id"
        if row_id in selected_ids:
            return f"selected_for_teacher has duplicate id: {row_id}"
        selected_ids.add(row_id)
        if row_id not in clean_ids:
            return f"selected_for_teacher contains a non-clean row: {row_id}"
        if row_id not in scored_ids:
            return f"selected_for_teacher contains an unscored row: {row_id}"
        if not isinstance(row.get("qe_score"), (int, float)):
            return f"selected_for_teacher row has invalid qe_score: {row_id}"

    if qe_requests and not selected_rows:
        return "selected_for_teacher is empty despite non-empty qe-selection input"
    return None


def _run_qe_selection(
    *,
    cfg: Mapping[str, Any],
    student_rows: list[dict[str, Any]],
    subset_idx: int,
) -> list[dict[str, Any]]:
    subset_dir = _subset_root(cfg, subset_idx)
    filter_rows = _filter_student_rows(cfg=cfg, student_rows=student_rows)
    filtered_path = subset_dir / "student_filtered.jsonl"
    write_jsonl(filtered_path, filter_rows)
    filter_summary = _student_filter_summary(student_rows=student_rows, filter_rows=filter_rows)

    qe_eligible_rows = [
        row for row in filter_rows
        if row.get("student_status") == "ok" and str(row.get("student_translation", "")).strip()
    ]
    clean_rows = [row for row in qe_eligible_rows if row["degeneration_label"] == "clean"]
    clean_row_ids = {str(row["id"]) for row in clean_rows}
    qe_requests = _build_qe_requests(cfg=cfg, rows=qe_eligible_rows, subset_idx=subset_idx)
    qe_input_path = subset_dir / "runtime_io" / "qe-selection.input.jsonl"
    qe_output_path = subset_dir / "runtime_io" / "qe-selection.output.jsonl"
    write_jsonl(qe_input_path, qe_requests)
    if not qe_requests:
        filter_summary.update(_student_filter_blocked_selection_summary(blocked_rows=[]))
        (subset_dir / "student_filter_summary.json").write_text(
            json.dumps(filter_summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        write_jsonl(qe_output_path, [])
        write_jsonl(subset_dir / "qe_scores.jsonl", [])
        write_jsonl(subset_dir / "filter_blocked_selection.jsonl", [])
        write_jsonl(subset_dir / "selected_for_teacher.jsonl", [])
        write_jsonl(
            subset_dir / "student_records.jsonl",
            _student_records(
                filter_rows=filter_rows,
                qe_requests=[],
                qe_responses=[],
                selected_rows=[],
                blocked_selection=[],
            ),
        )
        return []

    qe_responses = _score_qe_requests(cfg=cfg, qe_requests=qe_requests)
    write_jsonl(qe_output_path, qe_responses)
    write_jsonl(
        subset_dir / "qe_scores.jsonl",
        _qe_score_records(qe_requests=qe_requests, qe_responses=qe_responses, filter_rows=filter_rows),
    )
    clean_qe_responses = [
        response for response in qe_responses
        if str(response.get("row_id", "")) in clean_row_ids
    ]
    qe_order_prefix = _qe_selection_rule_prefix(_qe_selection_order(cfg))
    selected = _select_for_teacher(
        cfg=cfg,
        subset_idx=subset_idx,
        filter_rows=filter_rows,
        qe_responses=clean_qe_responses,
        selection_rule=f"{qe_order_prefix}_clean_student_length_bucket_candidate_pool",
    )
    shadow_selected = _select_for_teacher(
        cfg=cfg,
        subset_idx=subset_idx,
        filter_rows=filter_rows,
        qe_responses=qe_responses,
        selection_rule=f"{qe_order_prefix}_qe_eligible_length_bucket_candidate_pool",
    )
    blocked_selection = [
        row for row in shadow_selected
        if row.get("degeneration_label") != "clean"
    ]
    write_jsonl(subset_dir / "filter_blocked_selection.jsonl", blocked_selection)
    filter_summary.update(_student_filter_blocked_selection_summary(blocked_rows=blocked_selection))
    (subset_dir / "student_filter_summary.json").write_text(
        json.dumps(filter_summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(subset_dir / "selected_for_teacher.jsonl", selected)
    write_jsonl(
        subset_dir / "student_records.jsonl",
        _student_records(
            filter_rows=filter_rows,
            qe_requests=qe_requests,
            qe_responses=qe_responses,
            selected_rows=selected,
            blocked_selection=blocked_selection,
        ),
    )
    if not selected:
        raise SystemExit(
            "teacher selection produced zero clean candidates after student filtering; "
            "not falling back to filtered or non-clean rows. Inspect "
            f"{subset_dir / 'student_filter_summary.json'} and "
            f"{subset_dir / 'filter_blocked_selection.jsonl'}."
        )
    return selected


def _run_raw_random_selection(
    *,
    cfg: Mapping[str, Any],
    input_rows: list[dict[str, Any]],
    subset_idx: int,
) -> list[dict[str, Any]]:
    subset_dir = _subset_root(cfg, subset_idx)
    runtime_dir = subset_dir / "runtime_io"
    # This path is intentionally a legacy ablation: it samples raw input rows
    # before student inference, student filtering, and QE. The normal
    # data.qe_selection_order=random path goes through _run_qe_selection().
    write_jsonl(runtime_dir / "infer-student.input.jsonl", [])
    write_jsonl(runtime_dir / "infer-student.output.jsonl", [])
    write_jsonl(subset_dir / "student_translations.jsonl", [])
    write_jsonl(subset_dir / "student_filtered.jsonl", [])
    write_jsonl(runtime_dir / "qe-selection.input.jsonl", [])
    write_jsonl(runtime_dir / "qe-selection.output.jsonl", [])
    write_jsonl(subset_dir / "qe_scores.jsonl", [])
    write_jsonl(subset_dir / "filter_blocked_selection.jsonl", [])
    write_jsonl(subset_dir / "student_records.jsonl", [])
    (subset_dir / "student_filter_summary.json").write_text(
        json.dumps(
            {
                "student_rows": 0,
                "filter_input_rows": 0,
                "filter_pass_rows": 0,
                "filter_fail_rows": 0,
                "filter_fail_ratio": 0.0,
                "degeneration_filter_enabled": False,
                "basic_validity_only": False,
                "filter_label_counts": {},
                "filter_blocked_selection_rows": 0,
                "filter_blocked_selection_label_counts": {},
                "raw_random_baseline": True,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    selected = _select_raw_random_for_teacher(
        cfg=cfg,
        subset_idx=subset_idx,
        input_rows=input_rows,
    )
    write_jsonl(subset_dir / "selected_for_teacher.jsonl", selected)
    if input_rows and not selected:
        raise SystemExit("raw random teacher selection produced zero candidates")
    return selected


def _student_filter_summary(
    *,
    student_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    label_counts = Counter(str(row.get("degeneration_label", "unknown")) for row in filter_rows)
    total = len(filter_rows)
    clean = int(label_counts.get("clean", 0))
    rejected = max(0, total - clean)
    filter_enabled = any(bool(row.get("degeneration_filter_enabled", False)) for row in filter_rows)
    basic_validity_only = any(bool(row.get("degeneration_basic_validity_only", False)) for row in filter_rows)
    return {
        "student_rows": len(student_rows),
        "filter_input_rows": total,
        "filter_pass_rows": clean,
        "filter_fail_rows": rejected,
        "filter_fail_ratio": rejected / max(total, 1),
        "degeneration_filter_enabled": filter_enabled,
        "basic_validity_only": basic_validity_only,
        "filter_label_counts": dict(sorted(label_counts.items())),
    }


def _student_filter_blocked_selection_summary(
    *,
    blocked_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    label_counts = Counter(str(row.get("degeneration_label", "unknown")) for row in blocked_rows)
    return {
        "filter_blocked_selection_rows": len(blocked_rows),
        "filter_blocked_selection_label_counts": dict(sorted(label_counts.items())),
    }


def _mean_qe_score(rows: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    for row in rows:
        value = row.get("qe_score")
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scores.append(float(value))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _mean_qe_score_from_file(path: Path) -> float | None:
    if not path.exists():
        return None
    return _mean_qe_score(read_jsonl(path))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _raw_random_baseline_enabled(cfg: Mapping[str, Any]) -> bool:
    enabled = bool(_get(cfg, "data.raw_random_baseline", False))
    if enabled and _qe_selection_order(cfg) != "random":
        raise SystemExit("data.raw_random_baseline=true requires data.qe_selection_order=random")
    return enabled


def _raw_random_selection_artifact_error(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    input_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> str | None:
    expected_rows = _select_raw_random_for_teacher(
        cfg=cfg,
        subset_idx=subset_idx,
        input_rows=input_rows,
    )
    expected_ids = [str(row.get("id", "")) for row in expected_rows]
    selected_ids = [str(row.get("id", "")) for row in selected_rows]
    if selected_ids != expected_ids:
        return "raw random selected_for_teacher ids do not match current seed/subset/config"
    for row in selected_rows:
        if row.get("selection_rule") != "random_raw_length_bucket_candidate_pool":
            return "raw random selected_for_teacher has an unexpected selection_rule"
        if row.get("student_status") != "skipped_raw_random":
            return "raw random selected_for_teacher has unexpected student_status"
    if input_rows and not selected_rows:
        return "raw random selected_for_teacher is empty despite non-empty input"
    return None


def _filter_student_rows(
    *,
    cfg: Mapping[str, Any],
    student_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filter_cfg = _get(cfg, "data.degeneration_filter", {})
    if not isinstance(filter_cfg, Mapping):
        filter_cfg = {}
    out_rows: list[dict[str, Any]] = []
    global_enabled = bool(filter_cfg.get("enabled", True))
    student_enabled = global_enabled and bool(filter_cfg.get("student_enabled", True))
    require_valid_output = bool(filter_cfg.get("student_require_valid_output", True))
    for row in student_rows:
        if student_enabled:
            label, flags = classify_student_output(
                source=str(row.get("source", "")),
                mt=str(row.get("student_translation", "")),
                status=str(row.get("student_status", "")),
                finish_reason=row.get("finish_reason"),
                config=filter_cfg,
            )
        elif require_valid_output and str(row.get("student_status", "")) != "ok":
            label, flags = ("invalid_status", ["invalid_status"])
        elif require_valid_output and not str(row.get("student_translation", "")).strip():
            label, flags = ("empty", ["empty"])
        else:
            label, flags = ("clean", [])
        out = dict(row)
        out["degeneration_label"] = label
        out["degeneration_flags"] = flags
        out["degeneration_filter_enabled"] = student_enabled
        out["degeneration_basic_validity_only"] = (not student_enabled and require_valid_output)
        out_rows.append(out)
    return out_rows


def _build_qe_requests(
    *,
    cfg: Mapping[str, Any],
    rows: list[dict[str, Any]],
    subset_idx: int,
) -> list[dict[str, Any]]:
    qe_cfg = _get(cfg, "qe.selection", {})
    if not isinstance(qe_cfg, Mapping):
        raise SystemExit("qe.selection config must be a mapping")
    backend = str(qe_cfg.get("backend", "comet")).strip()
    model = str(qe_cfg.get("model", "")).strip()
    requests: list[dict[str, Any]] = []
    for order_idx, row in enumerate(rows):
        request = {
            "id": f"{_get(cfg, 'run.id')}/subsets/subset_{subset_idx:03d}/{row['id']}/qe-selection",
            "row_id": row["id"],
            "order_idx": order_idx,
            "backend": backend,
            "model": model,
            "src": row["source"],
            "mt": row["student_translation"],
        }
        if bool(qe_cfg.get("requires_reference", False)):
            request["ref"] = row.get("target")
        requests.append(request)
    return requests


def _score_qe_requests(
    *,
    cfg: Mapping[str, Any],
    qe_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qe_cfg = _get(cfg, "qe.selection", {})
    if not isinstance(qe_cfg, Mapping):
        raise SystemExit("qe.selection config must be a mapping")
    backend = str(qe_cfg.get("backend", "comet")).strip().lower()
    model_name = str(qe_cfg.get("model", "")).strip()
    if backend != "comet":
        raise SystemExit(f"unsupported qe.selection.backend={backend!r}; currently supported: comet")
    if not model_name:
        raise SystemExit("qe.selection.model is required")

    raw_num_gpus = qe_cfg.get("num_gpus", 1)
    scores = comet_scores(
        qe_requests,
        model_name=model_name,
        batch_size=int(qe_cfg.get("batch_size", 512) or 512),
        python_env_var=str(qe_cfg.get("python_env_var", "COMET_PYTHON")),
        include_reference=bool(qe_cfg.get("requires_reference", False)),
        num_gpus=None if raw_num_gpus is None else int(raw_num_gpus),
        gpu_ids=qe_cfg.get("gpu_ids"),
        shard_strategy=str(qe_cfg.get("shard_strategy", "order_split")),
    )
    responses: list[dict[str, Any]] = []
    for request, score in zip(qe_requests, scores):
        responses.append(
            {
                "id": request["id"],
                "row_id": request["row_id"],
                "order_idx": request["order_idx"],
                "backend": backend,
                "model": model_name,
                "score": float(score),
                "status": "ok",
                "error": None,
            }
        )
    return responses


def _qe_score_records(
    *,
    qe_requests: list[dict[str, Any]],
    qe_responses: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    response_by_id = {str(row.get("id", "")): row for row in qe_responses}
    source_row_by_id = {str(row.get("id", "")): row for row in filter_rows}
    records: list[dict[str, Any]] = []
    for request in qe_requests:
        response = response_by_id.get(str(request.get("id", "")), {})
        row = source_row_by_id.get(str(request.get("row_id", "")), {})
        raw_score = response.get("score")
        records.append(
            {
                "id": request.get("row_id"),
                "request_id": request.get("id"),
                "order_idx": request.get("order_idx"),
                "source": request.get("src"),
                "target": request.get("ref", row.get("target")),
                "metadata": row.get("metadata", {}),
                "source_tokens": row.get("source_tokens"),
                "student_translation": request.get("mt"),
                "student_status": row.get("student_status"),
                "student_error": row.get("student_error"),
                "finish_reason": row.get("finish_reason"),
                "generated_token_count": row.get("generated_token_count"),
                "prompt_template_id": row.get("prompt_template_id"),
                "prompt_template_group": row.get("prompt_template_group"),
                "prompt_template_hash": row.get("prompt_template_hash"),
                "chat_template_applied": row.get("chat_template_applied"),
                "degeneration_label": row.get("degeneration_label"),
                "degeneration_flags": row.get("degeneration_flags", []),
                "degeneration_filter_enabled": row.get("degeneration_filter_enabled"),
                "qe_backend": request.get("backend"),
                "qe_model": request.get("model"),
                "qe_requires_reference": "ref" in request,
                "qe_status": response.get("status", "missing"),
                "qe_score": None if raw_score is None else float(raw_score),
                "qe_error": response.get("error"),
            }
        )
    return records


def _student_records(
    *,
    filter_rows: list[dict[str, Any]],
    qe_requests: list[dict[str, Any]],
    qe_responses: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    blocked_selection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qe_request_by_row_id = {str(row.get("row_id", "")): row for row in qe_requests}
    qe_response_by_row_id = {str(row.get("row_id", "")): row for row in qe_responses}
    selected_by_id = {str(row.get("id", "")): row for row in selected_rows}
    blocked_by_id = {str(row.get("id", "")): row for row in blocked_selection}

    records: list[dict[str, Any]] = []
    for row in filter_rows:
        row_id = str(row.get("id", ""))
        qe_request = qe_request_by_row_id.get(row_id)
        qe_response = qe_response_by_row_id.get(row_id)
        selected = selected_by_id.get(row_id)
        blocked = blocked_by_id.get(row_id)

        qe_payload = None
        if qe_request is not None or qe_response is not None:
            raw_score = None if qe_response is None else qe_response.get("score")
            qe_payload = {
                "backend": (qe_response or qe_request or {}).get("backend"),
                "model": (qe_response or qe_request or {}).get("model"),
                "requires_reference": bool(qe_request is not None and "ref" in qe_request),
                "status": "missing" if qe_response is None else qe_response.get("status", "missing"),
                "score": None if raw_score is None else float(raw_score),
                "error": None if qe_response is None else qe_response.get("error"),
                "request_id": None if qe_request is None else qe_request.get("id"),
                "order_idx": None if qe_request is None else qe_request.get("order_idx"),
            }

        selection_payload = {
            "selected_for_teacher": selected is not None,
            "selection_rank": None if selected is None else selected.get("selection_rank"),
            "selection_rule": None if selected is None else selected.get("selection_rule"),
            "length_bucket_idx": None if selected is None else selected.get("length_bucket_idx"),
            "length_bucket": None if selected is None else selected.get("length_bucket"),
            "blocked_by_filter": blocked is not None,
            "blocked_selection_rank": None if blocked is None else blocked.get("selection_rank"),
            "blocked_selection_rule": None if blocked is None else blocked.get("selection_rule"),
            "blocked_length_bucket_idx": None if blocked is None else blocked.get("length_bucket_idx"),
            "blocked_length_bucket": None if blocked is None else blocked.get("length_bucket"),
        }

        filter_label = str(row.get("degeneration_label", "unknown"))
        records.append(
            {
                "id": row.get("id"),
                "source": row.get("source", ""),
                "target": row.get("target"),
                "metadata": row.get("metadata", {}),
                "source_tokens": row.get("source_tokens"),
                "student": {
                    "translation": row.get("student_translation", ""),
                    "status": row.get("student_status", "failed"),
                    "error": row.get("student_error"),
                    "finish_reason": row.get("finish_reason"),
                    "generated_token_count": row.get("generated_token_count", 0),
                    "request_id": row.get("request_id"),
                    "order_idx": row.get("order_idx"),
                    "prompt_template_id": row.get("prompt_template_id"),
                    "prompt_template_group": row.get("prompt_template_group"),
                    "prompt_template_hash": row.get("prompt_template_hash"),
                    "chat_template_applied": row.get("chat_template_applied"),
                },
                "filter": {
                    "filtered": filter_label != "clean",
                    "label": filter_label,
                },
                "qe": qe_payload,
                "selection": selection_payload,
            }
        )
    return records


def _restore_compact_front_artifacts(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    subset_dir: Path,
) -> int:
    records = _require_jsonl(subset_dir / "student_records.jsonl", "student_records.jsonl")
    model_cfg = _student_inference_model_cfg(cfg, subset_idx)
    filter_summary = _read_json_if_exists(subset_dir / "student_filter_summary.json")
    filter_enabled = bool(filter_summary.get("degeneration_filter_enabled", True))

    input_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    student_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    qe_requests: list[dict[str, Any]] = []
    qe_responses: list[dict[str, Any]] = []
    qe_scores: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        student = record.get("student", {})
        filter_payload = record.get("filter", {})
        qe = record.get("qe")
        selection = record.get("selection", {})
        if not isinstance(student, Mapping) or not isinstance(filter_payload, Mapping) or not isinstance(selection, Mapping):
            raise SystemExit(f"invalid compact student record at index={idx}")

        row_id = str(record.get("id", ""))
        request_id = str(student.get("request_id", ""))
        if not row_id or not request_id:
            raise SystemExit(f"compact student record is missing id/request_id at index={idx}")
        order_idx = int(student.get("order_idx", idx) if student.get("order_idx") is not None else idx)
        source = str(record.get("source", ""))
        translation = str(student.get("translation", ""))
        status = str(student.get("status", "failed"))

        input_row = {
            "id": row_id,
            "source": source,
            "target": record.get("target"),
            "metadata": record.get("metadata", {}),
            "source_tokens": record.get("source_tokens"),
        }
        request = {
            "id": request_id,
            "run_id": str(_get(cfg, "run.id")),
            "subset_idx": subset_idx,
            "row_id": row_id,
            "order_idx": order_idx,
            "source": source,
            "metadata": record.get("metadata", {}),
            "prompt": "",
            "prompt_template_id": student.get("prompt_template_id"),
            "prompt_template_group": student.get("prompt_template_group"),
            "prompt_template_hash": student.get("prompt_template_hash"),
            "chat_template_applied": student.get("chat_template_applied"),
            "model": dict(model_cfg),
            "inference": {},
            "decoding": {},
        }
        response = {
            "id": request_id,
            "row_id": row_id,
            "order_idx": order_idx,
            "status": status,
            "mt": translation,
            "error": student.get("error"),
            "finish_reason": student.get("finish_reason"),
            "generated_token_count": student.get("generated_token_count", 0),
        }
        student_row = {
            **input_row,
            "student_translation": translation,
            "student_status": status,
            "student_error": student.get("error"),
            "finish_reason": student.get("finish_reason"),
            "generated_token_count": student.get("generated_token_count", 0),
            "request_id": request_id,
            "run_id": str(_get(cfg, "run.id")),
            "subset_idx": subset_idx,
            "order_idx": order_idx,
            "prompt_template_id": student.get("prompt_template_id"),
            "prompt_template_group": student.get("prompt_template_group"),
            "prompt_template_hash": student.get("prompt_template_hash"),
            "chat_template_applied": student.get("chat_template_applied"),
        }
        filter_row = {
            **student_row,
            "degeneration_label": str(filter_payload.get("label", "unknown")),
            "degeneration_flags": [],
            "degeneration_filter_enabled": filter_enabled,
            "degeneration_basic_validity_only": False,
        }

        input_rows.append(input_row)
        requests.append(request)
        responses.append(response)
        student_rows.append(student_row)
        filter_rows.append(filter_row)

        if isinstance(qe, Mapping):
            qe_request_id = str(qe.get("request_id", ""))
            if not qe_request_id:
                raise SystemExit(f"compact QE record is missing request_id for row_id={row_id}")
            qe_order_idx = int(qe.get("order_idx", len(qe_requests)) if qe.get("order_idx") is not None else len(qe_requests))
            qe_request = {
                "id": qe_request_id,
                "row_id": row_id,
                "order_idx": qe_order_idx,
                "backend": qe.get("backend"),
                "model": qe.get("model"),
                "src": source,
                "mt": translation,
            }
            if bool(qe.get("requires_reference", False)):
                qe_request["ref"] = record.get("target")
            score = qe.get("score")
            qe_response = {
                "id": qe_request_id,
                "row_id": row_id,
                "order_idx": qe_order_idx,
                "backend": qe.get("backend"),
                "model": qe.get("model"),
                "status": qe.get("status", "missing"),
                "score": score,
                "error": qe.get("error"),
            }
            qe_requests.append(qe_request)
            qe_responses.append(qe_response)
            qe_scores.append(
                {
                    "id": row_id,
                    "request_id": qe_request_id,
                    "qe_score": score,
                }
            )

        candidate = dict(filter_row)
        if isinstance(qe, Mapping):
            candidate.update(
                {
                    "qe_score": qe.get("score"),
                    "qe_backend": qe.get("backend"),
                    "qe_model": qe.get("model"),
                }
            )
        if bool(selection.get("selected_for_teacher", False)):
            candidate.update(
                {
                    "selection_rank": selection.get("selection_rank"),
                    "selection_rule": selection.get("selection_rule"),
                    "length_bucket_idx": selection.get("length_bucket_idx"),
                    "length_bucket": selection.get("length_bucket"),
                }
            )
            selected_rows.append(candidate)
        if bool(selection.get("blocked_by_filter", False)):
            blocked = dict(candidate)
            blocked.update(
                {
                    "selection_rank": selection.get("blocked_selection_rank"),
                    "selection_rule": selection.get("blocked_selection_rule"),
                    "length_bucket_idx": selection.get("blocked_length_bucket_idx"),
                    "length_bucket": selection.get("blocked_length_bucket"),
                }
            )
            blocked_rows.append(blocked)

    selected_rows.sort(key=lambda row: int(row.get("selection_rank") or 10**12))
    runtime_dir = subset_dir / "runtime_io"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset_dir / "input.jsonl", input_rows)
    write_jsonl(runtime_dir / "infer-student.input.jsonl", requests)
    write_jsonl(runtime_dir / "infer-student.output.jsonl", responses)
    write_jsonl(subset_dir / "student_translations.jsonl", student_rows)
    write_jsonl(subset_dir / "student_filtered.jsonl", filter_rows)
    write_jsonl(runtime_dir / "qe-selection.input.jsonl", qe_requests)
    write_jsonl(runtime_dir / "qe-selection.output.jsonl", qe_responses)
    write_jsonl(subset_dir / "qe_scores.jsonl", qe_scores)
    write_jsonl(subset_dir / "selected_for_teacher.jsonl", selected_rows)
    write_jsonl(subset_dir / "filter_blocked_selection.jsonl", blocked_rows)
    return len(selected_rows)


def _select_for_teacher(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    filter_rows: list[dict[str, Any]],
    qe_responses: list[dict[str, Any]],
    selection_rule: str = "bottom_qe_clean_student_length_bucket_candidate_pool",
) -> list[dict[str, Any]]:
    row_by_id = {str(row["id"]): row for row in filter_rows}
    scored_rows: list[dict[str, Any]] = []
    for response in qe_responses:
        if response.get("status") != "ok":
            continue
        row_id = str(response.get("row_id", ""))
        row = row_by_id.get(row_id)
        if row is None:
            raise SystemExit(f"QE response row_id not found in filtered rows: {row_id}")
        out = dict(row)
        out["qe_score"] = float(response["score"])
        out["qe_backend"] = response["backend"]
        out["qe_model"] = response["model"]
        scored_rows.append(out)

    if not scored_rows:
        return []

    candidate_count = _teacher_candidate_count(cfg, pool_size=len(filter_rows))
    qe_order = _qe_selection_order(cfg)
    scored_rows.sort(
        key=lambda row: _teacher_candidate_sort_key(
            row,
            qe_order,
            cfg=cfg,
            subset_idx=subset_idx,
        )
    )
    selected: list[dict[str, Any]] = []
    ranked_candidates = _rank_teacher_candidates_by_length_bucket(
        cfg=cfg,
        subset_idx=subset_idx,
        scored_rows=scored_rows,
        candidate_count=candidate_count,
    )
    for rank, row in enumerate(ranked_candidates, start=1):
        out = dict(row)
        out["selection_rank"] = rank
        out["selection_rule"] = selection_rule
        selected.append(out)
    return selected


def _select_raw_random_for_teacher(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    input_rows: list[dict[str, Any]],
    selection_rule: str = "random_raw_length_bucket_candidate_pool",
) -> list[dict[str, Any]]:
    if not input_rows:
        return []
    candidate_count = _teacher_candidate_count(cfg, pool_size=len(input_rows))
    candidate_rows = [dict(row) for row in input_rows]
    candidate_rows.sort(
        key=lambda row: _teacher_candidate_sort_key(
            row,
            "random",
            cfg=cfg,
            subset_idx=subset_idx,
        )
    )
    ranked_candidates = _rank_teacher_candidates_by_length_bucket(
        cfg=cfg,
        subset_idx=subset_idx,
        scored_rows=candidate_rows,
        candidate_count=candidate_count,
    )
    selected: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_candidates, start=1):
        out = dict(row)
        out.setdefault("student_translation", "")
        out.setdefault("student_status", "skipped_raw_random")
        out.setdefault("student_error", None)
        out["qe_score"] = None
        out["qe_backend"] = None
        out["qe_model"] = None
        out["selection_rank"] = rank
        out["selection_rule"] = selection_rule
        selected.append(out)
    return selected


def _teacher_candidate_count(cfg: Mapping[str, Any], *, pool_size: int) -> int:
    target = int(_get(cfg, "data.teacher_target_per_subset", 0) or 0)
    if target <= 0:
        ratio = float(_get(cfg, "data.selection_ratio", 0.01) or 0.01)
        target = max(1, int(pool_size * ratio + 0.999999))
    multiplier = float(_get(cfg, "teacher.candidate_multiplier", 1.0) or 1.0)
    return max(target, int(target * max(1.0, multiplier) + 0.999999))


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


def _bucket_index_for_row(row: Mapping[str, Any], buckets: list[tuple[int, int]]) -> int:
    length = _source_token_length(row)
    for idx, (low, high) in enumerate(buckets):
        if low <= length <= high:
            return idx
    if length < buckets[0][0]:
        return 0
    return len(buckets) - 1


def _qe_selection_order(cfg: Mapping[str, Any]) -> str:
    order = str(_get(cfg, "data.qe_selection_order", "low") or "low").strip().lower()
    if order in {"bottom", "lowest", "low_qe", "bottom_qe"}:
        return "low"
    if order in {"top", "highest", "high_qe", "top_qe"}:
        return "high"
    if order in {"rand", "random_qe"}:
        return "random"
    if order not in {"low", "high", "random"}:
        raise SystemExit("data.qe_selection_order must be low, high, or random")
    return order


def _qe_sort_key(
    row: Mapping[str, Any],
    order: str,
    *,
    cfg: Mapping[str, Any] | None = None,
    subset_idx: int | None = None,
) -> tuple[object, ...]:
    if order == "random":
        if cfg is None or subset_idx is None:
            raise SystemExit("random qe_selection_order requires run.seed and subset_idx")
        return _random_teacher_candidate_key(row, cfg=cfg, subset_idx=subset_idx)
    score = float(row["qe_score"])
    if order == "high":
        return (-score, str(row["id"]))
    return (score, str(row["id"]))


def _qe_selection_rule_prefix(order: str) -> str:
    if order == "high":
        return "top_qe"
    if order == "random":
        return "random"
    return "bottom_qe"


def _random_teacher_candidate_key(
    row: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
) -> tuple[str, str]:
    seed = str(_get(cfg, "run.seed", 42))
    row_id = str(row["id"])
    payload = (
        "dqs_teacher_selection_random_v1"
        f"|seed={seed}|subset_idx={subset_idx}|row_id={row_id}"
    )
    return (hashlib.sha256(payload.encode("utf-8")).hexdigest(), row_id)


def _teacher_candidate_sort_key(
    row: Mapping[str, Any],
    order: str,
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
) -> tuple[object, ...]:
    return _qe_sort_key(row, order, cfg=cfg, subset_idx=subset_idx)


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


def _rank_teacher_candidates_by_length_bucket(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    scored_rows: list[dict[str, Any]],
    candidate_count: int,
) -> list[dict[str, Any]]:
    if candidate_count <= 0 or not scored_rows:
        return []

    selection_cfg = _get(cfg, "data.length_bucket_selection", {})
    if not isinstance(selection_cfg, Mapping) or not bool(selection_cfg.get("enabled", True)):
        return scored_rows[:candidate_count]

    buckets = _length_buckets(cfg)
    bucket_rows: list[list[dict[str, Any]]] = [[] for _ in buckets]
    for row in scored_rows:
        bucket_rows[_bucket_index_for_row(row, buckets)].append(row)

    qe_order = _qe_selection_order(cfg)
    for rows in bucket_rows:
        rows.sort(
            key=lambda row: _teacher_candidate_sort_key(
                row,
                qe_order,
                cfg=cfg,
                subset_idx=subset_idx,
            )
        )

    explicit_weights = _bucket_weights(cfg, len(buckets))
    quota_strategy = str(selection_cfg.get("quota_strategy", "proportional")).strip().lower()
    if explicit_weights is not None:
        weights = explicit_weights
    elif quota_strategy == "uniform":
        weights = [1.0 for _ in buckets]
    elif quota_strategy == "proportional":
        weights = [float(len(rows)) for rows in bucket_rows]
    else:
        raise SystemExit("data.length_bucket_selection.quota_strategy must be uniform or proportional")

    quotas = _quota_from_weights(candidate_count, weights)
    positive_weight_bucket_indexes = {
        idx for idx, weight in enumerate(weights)
        if weight > 0
    }
    allow_zero_weight_bucket_fill = bool(selection_cfg.get("allow_zero_weight_bucket_fill", False))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket_idx, quota in enumerate(quotas):
        picked = 0
        for row in bucket_rows[bucket_idx]:
            if picked >= quota:
                break
            selected.append(_with_bucket_meta(row, buckets, bucket_idx))
            selected_ids.add(str(row["id"]))
            picked += 1

    if len(selected) < candidate_count and bool(selection_cfg.get("fill_remainder_from_global", True)):
        for row in scored_rows:
            if len(selected) >= candidate_count:
                break
            row_id = str(row["id"])
            if row_id in selected_ids:
                continue
            bucket_idx = _bucket_index_for_row(row, buckets)
            if explicit_weights is not None and not allow_zero_weight_bucket_fill:
                if bucket_idx not in positive_weight_bucket_indexes:
                    continue
            selected.append(_with_bucket_meta(row, buckets, bucket_idx))
            selected_ids.add(row_id)

    selected.sort(
        key=lambda row: (
            int(row["length_bucket_idx"]),
        ) + _teacher_candidate_sort_key(
            row,
            qe_order,
            cfg=cfg,
            subset_idx=subset_idx,
        )
    )
    return selected[:candidate_count]


def _with_bucket_meta(row: Mapping[str, Any], buckets: list[tuple[int, int]], bucket_idx: int) -> dict[str, Any]:
    out = dict(row)
    low, high = buckets[bucket_idx]
    out["source_tokens"] = _source_token_length(row)
    out["length_bucket_idx"] = bucket_idx
    out["length_bucket"] = f"{low}-{high}"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DQS subset front stage.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset-idx", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--start-from-phase", choices=list(TRAIN_PHASES), default=None)
    parser.add_argument("--resume", choices=["auto", "none"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sft-scheduler-total-steps", type=int, default=None)
    parser.add_argument("--sft-force-save-checkpoint", action="store_true")
    parser.add_argument("--sft-nproc-per-node", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = compose_config(args.config, overrides=args.override)
    default_subset_idx = int(args.subset_idx if args.subset_idx is not None else _get(cfg, "run.subset_start", 0))
    start_from_phase = args.start_from_phase
    if args.resume == "auto" and start_from_phase is None and not args.force:
        default_subset_idx, start_from_phase = _auto_resume_target(
            cfg=cfg,
            default_subset_idx=default_subset_idx,
            explicit_subset_idx=args.subset_idx,
        )
    skipped = _skipped_phases(start_from_phase)
    subset_idx = default_subset_idx
    subset_size = int(args.subset_size if args.subset_size is not None else _get(cfg, "data.subset_size", 100000))
    subset_dir = _subset_root(cfg, subset_idx)
    runtime_dir = subset_dir / "runtime_io"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    progress(
        "subset start",
        run=_get(cfg, "run.id"),
        subset=f"subset_{subset_idx:03d}",
        resume=args.resume,
        start_from=start_from_phase,
        dry_run=args.dry_run,
    )

    cfg_hash = config_hash(cfg)
    save_effective_config(_get(cfg, "paths.config_snapshot_path"), cfg)
    Path(str(_get(cfg, "paths.config_hash_path"))).write_text(f"{cfg_hash}\n", encoding="utf-8")

    request_path = runtime_dir / "infer-student.input.jsonl"
    response_path = runtime_dir / "infer-student.output.jsonl"
    compact_records_path = subset_dir / "student_records.jsonl"
    if (
        start_from_phase in {"teacher", "sft-dataset", "sft"}
        and not (subset_dir / "input.jsonl").exists()
        and compact_records_path.exists()
    ):
        restored_selected = _restore_compact_front_artifacts(
            cfg=cfg,
            subset_idx=subset_idx,
            subset_dir=subset_dir,
        )
        progress(
            "compact front artifacts restored",
            subset=f"subset_{subset_idx:03d}",
            selected_for_teacher=restored_selected,
        )
    if "input" in skipped:
        input_rows = _require_jsonl(subset_dir / "input.jsonl", "input.jsonl")
    else:
        input_rows = _run_tracked_phase(
            cfg=cfg,
            subset_dir=subset_dir,
            subset_idx=subset_idx,
            phase="input",
            func=lambda: _materialize_input(
                cfg=cfg,
                subset_idx=subset_idx,
                subset_size=subset_size,
                data_path=args.data_path,
                force=args.force,
            ),
        )

    raw_random_baseline = _raw_random_baseline_enabled(cfg)
    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    student_rows: list[dict[str, Any]] = []
    if raw_random_baseline:
        progress(
            "raw random baseline",
            subset=f"subset_{subset_idx:03d}",
            skip_student_infer=True,
            skip_qe_scoring=True,
        )
    else:
        if "student-infer" in skipped:
            valid = False
            reason = "unknown"
            normalized: list[dict[str, Any]] = []
            try:
                requests = _require_jsonl(request_path, "infer-student.input.jsonl")
                request_error = _student_request_model_error(
                    cfg=cfg,
                    subset_idx=subset_idx,
                    requests=requests,
                )
                if request_error is not None:
                    raise ValueError(request_error)
                responses = _require_jsonl(response_path, "infer-student.output.jsonl")
                valid, reason, normalized = _validate_vllm_outputs(
                    requests=requests,
                    responses=responses,
                    allow_dry_run=args.dry_run,
                )
            except (SystemExit, ValueError, json.JSONDecodeError) as exc:
                reason = str(exc)
            if valid:
                responses = normalized
            else:
                print(
                    f"[resume] invalid student-infer artifacts; rerunning student-infer: {reason}",
                    file=sys.stderr,
                )
                skipped = _rewind_skipped_phases(skipped, "student-infer")
        if "student-infer" not in skipped:
            def _student_infer_phase() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                built_requests = _build_inference_requests(
                    cfg=cfg,
                    rows=input_rows,
                    subset_idx=subset_idx,
                    force=args.force,
                )
                if args.dry_run:
                    built_responses = _write_dry_run_outputs(built_requests, response_path, force=args.force)
                else:
                    built_responses = _run_vllm(request_path, response_path, force=args.force)
                return built_requests, built_responses

            requests, responses = _run_tracked_phase(
                cfg=cfg,
                subset_dir=subset_dir,
                subset_idx=subset_idx,
                phase="student-infer",
                func=_student_infer_phase,
            )

        if "student-filter" in skipped:
            student_error = None
            try:
                student_rows = _require_jsonl(
                    subset_dir / "student_translations.jsonl",
                    "student_translations.jsonl",
                )
                student_error = _student_translation_artifact_error(
                    input_rows=input_rows,
                    requests=requests,
                    responses=responses,
                    student_rows=student_rows,
                )
            except (SystemExit, ValueError, json.JSONDecodeError) as exc:
                student_error = str(exc)
            if student_error is not None:
                print(
                    f"[resume] invalid student-filter artifacts; rerunning student-filter: {student_error}",
                    file=sys.stderr,
                )
                skipped = _rewind_skipped_phases(skipped, "student-filter")
        if "student-filter" not in skipped:
            student_rows = _run_tracked_phase(
                cfg=cfg,
                subset_dir=subset_dir,
                subset_idx=subset_idx,
                phase="student-filter",
                func=lambda: _materialize_student_translations(
                    cfg=cfg,
                    input_rows=input_rows,
                    requests=requests,
                    responses=responses,
                    subset_idx=subset_idx,
                ),
            )

    selected_rows: list[dict[str, Any]] = []
    teacher_summary = {}
    sft_summary = {}
    sft_training_summary = {}
    student_filter_summary = {}
    if not args.dry_run:
        if "qe-select" in skipped:
            qe_error = None
            try:
                selected_rows = _require_jsonl_file(
                    subset_dir / "selected_for_teacher.jsonl",
                    "selected_for_teacher.jsonl",
                )
                if raw_random_baseline:
                    student_filter_summary = _read_json_if_exists(subset_dir / "student_filter_summary.json")
                    qe_error = _raw_random_selection_artifact_error(
                        cfg=cfg,
                        subset_idx=subset_idx,
                        input_rows=input_rows,
                        selected_rows=selected_rows,
                    )
                else:
                    filter_rows = _require_jsonl_file(subset_dir / "student_filtered.jsonl", "student_filtered.jsonl")
                    student_filter_summary = _require_json(
                        subset_dir / "student_filter_summary.json",
                        "student_filter_summary.json",
                    )
                    qe_error = _qe_selection_artifact_error(
                        subset_dir=subset_dir,
                        student_rows=student_rows,
                        filter_rows=filter_rows,
                        student_filter_summary=student_filter_summary,
                        selected_rows=selected_rows,
                    )
            except (SystemExit, ValueError, json.JSONDecodeError) as exc:
                qe_error = str(exc)
            if qe_error is not None:
                print(
                    f"[resume] invalid qe-select artifacts; rerunning qe-select: {qe_error}",
                    file=sys.stderr,
                )
                skipped = _rewind_skipped_phases(skipped, "qe-select")
                student_filter_summary = {}
                selected_rows = []
        if "qe-select" not in skipped:
            if raw_random_baseline:
                qe_select_func = lambda: _run_raw_random_selection(
                    cfg=cfg,
                    input_rows=input_rows,
                    subset_idx=subset_idx,
                )
            else:
                qe_select_func = lambda: _run_qe_selection(
                    cfg=cfg,
                    student_rows=student_rows,
                    subset_idx=subset_idx,
                )
            selected_rows = _run_tracked_phase(
                cfg=cfg,
                subset_dir=subset_dir,
                subset_idx=subset_idx,
                phase="qe-select",
                func=qe_select_func,
            )
            student_filter_summary = _read_json_if_exists(subset_dir / "student_filter_summary.json")

        if "teacher" in skipped:
            teacher_summary = _require_json(subset_dir / "teacher_summary.json", "teacher_summary.json")
            _require_jsonl(subset_dir / "golden_pairs.jsonl", "golden_pairs.jsonl")
        else:
            teacher_summary = _run_tracked_phase(
                cfg=cfg,
                subset_dir=subset_dir,
                subset_idx=subset_idx,
                phase="teacher",
                func=lambda: run_teacher_generation(
                    cfg=cfg,
                    candidates=selected_rows,
                    subset_dir=subset_dir,
                ),
            )

        if "sft-dataset" in skipped:
            _require_jsonl(subset_dir / "golden_pairs.jsonl", "golden_pairs.jsonl")
            sft_rows = _require_jsonl(subset_dir / "sft_train.jsonl", "sft_train.jsonl")
            sft_summary = {
                "sft_rows": len(sft_rows),
                "sft_dataset_path": str(subset_dir / "sft_train.jsonl"),
                "golden_rows": len(read_jsonl(subset_dir / "golden_pairs.jsonl")),
            }
        else:
            sft_summary = _run_tracked_phase(
                cfg=cfg,
                subset_dir=subset_dir,
                subset_idx=subset_idx,
                phase="sft-dataset",
                func=lambda: write_sft_dataset(
                    cfg=cfg,
                    golden_path=subset_dir / "golden_pairs.jsonl",
                    output_path=subset_dir / "sft_train.jsonl",
                    subset_idx=subset_idx,
                ),
            )

        if "sft" in skipped:
            sft_training_summary = _read_json_if_exists(
                _sft_training_summary_path(cfg, subset_idx)
            )
        else:
            sft_training_summary = _run_tracked_phase(
                cfg=cfg,
                subset_dir=subset_dir,
                subset_idx=subset_idx,
                phase="sft",
                func=lambda: _run_sft_training_phase(
                    cfg=cfg,
                    args=args,
                    subset_idx=subset_idx,
                    dataset_path=subset_dir / "sft_train.jsonl",
                ),
            )
    summary = {
        "run_id": _get(cfg, "run.id"),
        "subset_idx": subset_idx,
        "subset_size": subset_size,
        "input_rows": len(input_rows),
        "inference_requests": len(requests),
        "student_rows": len(student_rows),
        "student_filter_input_rows": student_filter_summary.get("filter_input_rows", 0),
        "student_filter_pass_rows": student_filter_summary.get("filter_pass_rows", 0),
        "student_filter_fail_rows": student_filter_summary.get("filter_fail_rows", 0),
        "student_filter_fail_ratio": student_filter_summary.get("filter_fail_ratio", 0.0),
        "student_filter_label_counts": student_filter_summary.get("filter_label_counts", {}),
        "student_filter_blocked_selection_rows": student_filter_summary.get("filter_blocked_selection_rows", 0),
        "student_filter_blocked_selection_label_counts": student_filter_summary.get(
            "filter_blocked_selection_label_counts",
            {},
        ),
        "qe_selection_order": _qe_selection_order(cfg),
        "raw_random_baseline": raw_random_baseline,
        "selected_for_teacher_rows": len(selected_rows),
        "all_qe_score_mean": _mean_qe_score_from_file(subset_dir / "qe_scores.jsonl"),
        "selected_qe_score_mean": _mean_qe_score_from_file(subset_dir / "golden_pairs.jsonl"),
        "teacher_accepted_rows": teacher_summary.get("teacher_accepted_rows", 0),
        "teacher_rejected_rows": teacher_summary.get("teacher_rejected_rows", 0),
        "teacher_shortfall_rows": teacher_summary.get("teacher_shortfall_rows", 0),
        "teacher_label_counts": teacher_summary.get("teacher_label_counts", {}),
        "teacher_label_ratios": teacher_summary.get("teacher_label_ratios", {}),
        "sft_rows": sft_summary.get("sft_rows", 0),
        "sft_dataset_path": sft_summary.get("sft_dataset_path"),
        "sft_training_global_step": sft_training_summary.get("global_step"),
        "sft_training_output_dir": sft_training_summary.get("output_dir"),
        "sft_training_world_size": sft_training_summary.get("world_size"),
        "dry_run": bool(args.dry_run),
        "resume_mode": args.resume,
        "resumed_from": start_from_phase,
        "skipped_phases": sorted(skipped, key=TRAIN_PHASES.index),
        "subset_dir": str(subset_dir),
        "config_hash": cfg_hash,
    }
    if start_from_phase is None:
        summary.pop("resumed_from")
        summary.pop("skipped_phases")
    _write_phase_state(
        cfg=cfg,
        subset_dir=subset_dir,
        subset_idx=subset_idx,
        phase="sft" if not args.dry_run else "student-filter",
        status="completed",
    )
    (subset_dir / "front_stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _cleanup_compact_subset_artifacts(cfg, subset_dir)
    progress(
        "subset done",
        run=_get(cfg, "run.id"),
        subset=f"subset_{subset_idx:03d}",
        selected_for_teacher=len(selected_rows),
        sft_rows=summary.get("sft_rows"),
        global_step=summary.get("sft_training_global_step"),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
