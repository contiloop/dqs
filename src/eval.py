#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config, save_effective_config
from degeneration_filter import classify_student_output
from io_utils import read_jsonl, write_jsonl
from metricx_score import metricx_scores
from progress import progress, progress_context
from prompting import load_student_templates, render_student_prompt
from qe_score import comet_scores
from runtime_logging import configure_runtime_logging
from wandb_logging import eval_metric_payload, log_wandb_metrics


configure_runtime_logging()

def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


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


def _load_eval_rows(cfg: Mapping[str, Any], data_path: str | None, limit: int | None) -> list[dict[str, Any]]:
    path = Path(data_path or str(_get(cfg, "eval.dataset_path")))
    if not path.exists():
        raise SystemExit(f"eval dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        raw_rows = _read_parquet_rows(path)
    elif path.suffix.lower() == ".jsonl":
        raw_rows = read_jsonl(path)
    else:
        raise SystemExit(f"unsupported eval dataset extension: {path.suffix}")

    source_field = str(_get(cfg, "eval.source_field", _get(cfg, "data.source_field", "source")))
    target_field = str(_get(cfg, "eval.target_field", _get(cfg, "data.target_field", "target")))
    metadata_field = str(_get(cfg, "eval.metadata_field", _get(cfg, "data.metadata_field", "metadata")))
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        source = row.get(source_field)
        if not isinstance(source, str) or not source.strip():
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            row_id = f"eval_{idx:012d}"
        metadata = row.get(metadata_field)
        rows.append(
            {
                "id": row_id,
                "source": source,
                "target": row.get(target_field),
                "source_tokens": row.get("source_tokens"),
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise SystemExit("eval dataset did not contain usable source rows")
    return rows


def _auto_model_candidates(cfg: Mapping[str, Any]) -> list[Path]:
    checkpoint_dir = Path(str(_get(cfg, "paths.checkpoint_dir")))
    tuning_mode = str(_get(cfg, "training.tuning_mode", "")).strip().lower()
    if tuning_mode == "lora":
        names = ["merged_16bit", "adapter"]
    elif tuning_mode == "full":
        names = ["full_model", "final"]
    else:
        names = ["merged_16bit", "full_model", "final"]
    return [checkpoint_dir / name for name in names]


def _is_lora_adapter_path(path: Path) -> bool:
    return path.is_dir() and (
        (path / "adapter_config.json").exists()
        or (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


def _resolve_generation_model_spec(
    cfg: Mapping[str, Any],
    *,
    model_path: str | None,
    require_trained_artifact: bool,
) -> dict[str, str | None]:
    raw = str(model_path or _get(cfg, "eval.generation.model_path", "auto")).strip()
    if raw and raw != "auto":
        raw_path = Path(raw)
        if raw_path.exists() and _is_lora_adapter_path(raw_path):
            return {
                "model_path": str(_get(cfg, "model.name_or_path")),
                "lora_adapter_path": str(raw_path),
            }
        return {"model_path": raw, "lora_adapter_path": None}
    for candidate in _auto_model_candidates(cfg):
        if candidate.exists():
            if _is_lora_adapter_path(candidate):
                return {
                    "model_path": str(_get(cfg, "model.name_or_path")),
                    "lora_adapter_path": str(candidate),
                }
            return {"model_path": str(candidate), "lora_adapter_path": None}
    if not require_trained_artifact:
        return {"model_path": str(_get(cfg, "model.name_or_path")), "lora_adapter_path": None}
    candidates = ", ".join(str(path) for path in _auto_model_candidates(cfg))
    raise SystemExit(f"eval.generation.model_path=auto could not find a trained artifact; checked: {candidates}")


def _eval_inference_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    base = _get(cfg, "inference", {})
    inference = dict(base) if isinstance(base, Mapping) else {}
    generation = _get(cfg, "eval.generation", {})
    if not isinstance(generation, Mapping):
        generation = {}
    for key in ("num_gpus", "tensor_parallel_size", "gpu_memory_utilization"):
        if key in generation:
            inference[key] = generation[key]
    return inference


def _build_eval_requests(
    *,
    cfg: Mapping[str, Any],
    rows: list[dict[str, Any]],
    model_spec: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    prompt_cfg = _get(cfg, "prompts", {})
    if not isinstance(prompt_cfg, Mapping):
        raise SystemExit("prompts config must be a mapping")
    template_path = Path(str(prompt_cfg.get("student_templates_path", "prompts/student_templates.yaml")))
    template_cfg = load_student_templates(template_path)
    model_cfg = _get(cfg, "model", {})
    model_cfg = dict(model_cfg) if isinstance(model_cfg, Mapping) else {}
    model_cfg["name_or_path"] = str(model_spec["model_path"])
    if model_spec.get("lora_adapter_path"):
        model_cfg["lora_adapter_path"] = str(model_spec["lora_adapter_path"])
    inference_cfg = _eval_inference_cfg(cfg)
    profile = str(_get(cfg, "eval.profile", "train"))

    requests: list[dict[str, Any]] = []
    for order_idx, row in enumerate(rows):
        rendered = render_student_prompt(
            template_cfg=template_cfg,
            prompt_cfg=prompt_cfg,
            model_cfg=model_cfg,
            source=str(row["source"]),
            row_id=str(row["id"]),
            subset_idx=0,
        )
        requests.append(
            {
                "id": f"{_get(cfg, 'run.id')}/eval/{profile}/{row['id']}/student",
                "run_id": str(_get(cfg, "run.id")),
                "eval_profile": profile,
                "row_id": row["id"],
                "order_idx": order_idx,
                "source": row["source"],
                "metadata": row.get("metadata", {}),
                "prompt": rendered.text,
                "prompt_template_id": rendered.template_id,
                "prompt_template_group": rendered.template_group,
                "prompt_template_hash": rendered.template_hash,
                "chat_template_applied": rendered.chat_template_applied,
                "model": model_cfg,
                "inference": inference_cfg,
                "decoding": {
                    "temperature": _get(cfg, "eval.generation.temperature", 0.0),
                    "top_p": _get(cfg, "eval.generation.top_p", 1.0),
                    "max_new_tokens": _get(cfg, "eval.generation.max_new_tokens", 1500),
                },
            }
        )
    return requests


def _validate_generation_outputs(
    *,
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    allow_dry_run: bool,
) -> tuple[bool, str, list[dict[str, Any]]]:
    request_by_id = {str(row.get("id", "")): row for row in requests}
    if len(request_by_id) != len(requests):
        return False, "duplicate or missing request ids", []
    response_by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    dry_run_ids: list[str] = []
    for idx, response in enumerate(responses):
        response_id = str(response.get("id", ""))
        if not response_id:
            return False, f"response at index {idx} is missing id", []
        if response_id in response_by_id:
            duplicates.add(response_id)
        response_by_id[response_id] = dict(response)
        if not allow_dry_run and response.get("status") == "dry_run":
            dry_run_ids.append(response_id)
    if duplicates:
        return False, f"duplicate response ids: {sorted(duplicates)[:3]}", []
    request_ids = set(request_by_id)
    response_ids = set(response_by_id)
    missing = sorted(request_ids - response_ids)
    extra = sorted(response_ids - request_ids)
    if missing:
        return False, f"missing response ids: {missing[:3]} (count={len(missing)})", []
    if extra:
        return False, f"extra response ids: {extra[:3]} (count={len(extra)})", []
    if dry_run_ids:
        return False, f"dry-run responses cannot satisfy real eval output: {dry_run_ids[:3]}", []
    normalized: list[dict[str, Any]] = []
    for request in sorted(requests, key=lambda row: int(row.get("order_idx", 0))):
        response = dict(response_by_id[str(request["id"])])
        response["order_idx"] = int(request.get("order_idx", 0))
        if str(response.get("row_id", "")) != str(request.get("row_id", "")):
            return False, f"response row_id mismatch: {response.get('id')}", []
        normalized.append(response)
    return True, "ok", normalized


def _validated_existing_outputs_or_none(
    *,
    requests: list[dict[str, Any]],
    output_path: Path,
    allow_dry_run: bool,
) -> list[dict[str, Any]] | None:
    try:
        existing = read_jsonl(output_path)
    except (OSError, ValueError) as exc:
        print(f"[resume] invalid existing eval output; rerunning {output_path}: {exc}", file=sys.stderr)
        return None
    valid, reason, normalized = _validate_generation_outputs(
        requests=requests,
        responses=existing,
        allow_dry_run=allow_dry_run,
    )
    if valid:
        return normalized
    print(f"[resume] invalid existing eval output; rerunning {output_path}: {reason}", file=sys.stderr)
    return None


def _run_generation(
    *,
    requests: list[dict[str, Any]],
    request_path: Path,
    output_path: Path,
    force: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        normalized = _validated_existing_outputs_or_none(
            requests=requests,
            output_path=output_path,
            allow_dry_run=dry_run,
        )
        if normalized is not None:
            return normalized
    if dry_run:
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
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("vllm_inference.py")),
        "--input",
        str(request_path),
        "--output",
        str(output_path),
    ]
    with progress_context("eval generation subprocess", requests=len(requests)):
        subprocess.run(cmd, check=True)
    responses = read_jsonl(output_path)
    valid, reason, normalized = _validate_generation_outputs(
        requests=requests,
        responses=responses,
        allow_dry_run=False,
    )
    if not valid:
        raise SystemExit(f"eval output validation failed: {output_path}: {reason}")
    return normalized


def _materialize_eval_translations(
    *,
    rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row.get("id")): row for row in responses}
    out_rows: list[dict[str, Any]] = []
    for source_row, request in zip(rows, requests):
        response = by_id.get(str(request["id"]))
        if response is None:
            raise SystemExit(f"missing eval response for request id={request['id']}")
        out_rows.append(
            {
                "id": source_row["id"],
                "source": source_row["source"],
                "target": source_row.get("target"),
                "metadata": source_row.get("metadata", {}),
                "source_tokens": source_row.get("source_tokens"),
                "translation": response.get("mt", ""),
                "status": response.get("status", "failed"),
                "error": response.get("error"),
                "finish_reason": response.get("finish_reason"),
                "generated_token_count": response.get("generated_token_count", 0),
                "prompt": request["prompt"],
                "prompt_template_id": request["prompt_template_id"],
                "prompt_template_group": request["prompt_template_group"],
                "prompt_template_hash": request["prompt_template_hash"],
                "chat_template_applied": request["chat_template_applied"],
            }
        )
    return out_rows


def _filter_eval_rows(cfg: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filter_enabled = bool(_get(cfg, "eval.filter.enabled", True))
    filter_cfg = _get(cfg, "data.degeneration_filter", {})
    if not isinstance(filter_cfg, Mapping):
        filter_cfg = {}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if filter_enabled:
            label, flags = classify_student_output(
                source=str(row.get("source", "")),
                mt=str(row.get("translation", "")),
                status=str(row.get("status", "")),
                finish_reason=row.get("finish_reason"),
                config=filter_cfg,
            )
        elif str(row.get("status", "")) != "ok":
            label, flags = ("invalid_status", ["invalid_status"])
        elif not str(row.get("translation", "")).strip():
            label, flags = ("empty", ["empty"])
        else:
            label, flags = ("clean", [])
        out = dict(row)
        out["degeneration_label"] = label
        out["degeneration_flags"] = flags
        out["degeneration_filter_enabled"] = filter_enabled
        out_rows.append(out)
    return out_rows


def _metric_payload(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "src": str(row.get("source", "")),
            "mt": str(row.get("translation", "")),
            "ref": str(row.get("target", "")),
        }
        for row in rows
    ]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _score_eval_metrics(
    *,
    cfg: Mapping[str, Any],
    rows: list[dict[str, Any]],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    for row in rows:
        score_rows.append(
            {
                "id": row["id"],
                "source": row.get("source"),
                "reference": row.get("target"),
                "translation": row.get("translation"),
                "status": row.get("status"),
                "degeneration_label": row.get("degeneration_label"),
                "scores": {},
            }
        )
    if dry_run:
        return score_rows, {}

    metric_rows = _metric_payload(rows)
    metrics_cfg = _get(cfg, "eval.metrics", [])
    if not isinstance(metrics_cfg, list):
        raise SystemExit("eval.metrics must be a list")

    summary: dict[str, Any] = {}
    for metric in metrics_cfg:
        if not isinstance(metric, Mapping):
            raise SystemExit("each eval metric must be a mapping")
        metric_id = str(metric.get("id", metric.get("backend", "metric"))).strip()
        backend = str(metric.get("backend", "")).strip().lower()
        requires_reference = bool(metric.get("requires_reference", True))
        if requires_reference and any(not str(row.get("ref", "")).strip() for row in metric_rows):
            raise SystemExit(f"metric {metric_id} requires references, but some eval rows have empty target")

        with progress_context("eval metric", metric=metric_id, backend=backend, rows=len(metric_rows)):
            if backend == "sacrebleu":
                try:
                    import sacrebleu
                except ModuleNotFoundError as exc:
                    raise SystemExit("missing sacrebleu; run `make set` first") from exc
                metric_name = str(metric.get("metric", metric_id)).strip().lower()
                hypotheses = [row["mt"] for row in metric_rows]
                references = [row["ref"] for row in metric_rows]
                if metric_name == "bleu":
                    score = float(sacrebleu.corpus_bleu(hypotheses, [references]).score)
                elif metric_name == "chrf":
                    score = float(sacrebleu.corpus_chrf(hypotheses, [references]).score)
                else:
                    raise SystemExit(f"unsupported sacrebleu metric={metric_name!r}")
                summary[metric_id] = {
                    "backend": backend,
                    "metric": metric_name,
                    "score": score,
                    "higher_is_better": True,
                }
                continue

            if backend == "comet":
                scores = comet_scores(
                    metric_rows,
                    model_name=str(metric.get("model", "")).strip(),
                    batch_size=int(metric.get("batch_size", 512) or 512),
                    python_env_var=str(metric.get("python_env_var", "COMET_PYTHON")),
                    include_reference=requires_reference,
                )
                for row, score in zip(score_rows, scores):
                    row["scores"][metric_id] = float(score)
                summary[metric_id] = {
                    "backend": backend,
                    "model": str(metric.get("model", "")).strip(),
                    "mean": _mean(scores),
                    "higher_is_better": True,
                }
                continue

            if backend == "metricx":
                scores = metricx_scores(
                    metric_rows,
                    model_name=str(metric.get("model", "")).strip(),
                    tokenizer=str(metric.get("tokenizer", "google/mt5-xl")).strip(),
                    max_input_length=int(metric.get("max_input_length", 1536) or 1536),
                    batch_size=int(metric.get("batch_size", 1) or 1),
                    python_env_var=str(metric.get("python_env_var", "METRICX_PYTHON")),
                    module=str(metric.get("module", "metricx24.predict")),
                    include_reference=requires_reference,
                )
                for row, score in zip(score_rows, scores):
                    row["scores"][metric_id] = float(score)
                summary[metric_id] = {
                    "backend": backend,
                    "model": str(metric.get("model", "")).strip(),
                    "mean": _mean(scores),
                    "higher_is_better": False,
                }
                continue

        raise SystemExit(f"unsupported eval metric backend={backend!r}")
    return score_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQS evaluation.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-wandb-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = compose_config(args.config, overrides=args.override)
    output_dir = Path(args.output_dir or str(_get(cfg, "eval.output_dir")))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(output_dir / "effective_config.yaml", cfg)

    limit = args.limit if args.limit is not None else _get(cfg, "eval.limit")
    limit_int = None if limit is None else int(limit)
    progress("eval start", profile=_get(cfg, "eval.profile"), output_dir=output_dir, dry_run=args.dry_run)
    with progress_context("eval load-data", path=args.data_path or _get(cfg, "eval.dataset_path"), limit=limit_int):
        rows = _load_eval_rows(cfg, args.data_path, limit_int)
    model_spec = _resolve_generation_model_spec(
        cfg,
        model_path=args.model_path,
        require_trained_artifact=not args.dry_run,
    )
    progress(
        "eval model",
        model=model_spec["model_path"],
        lora_adapter=model_spec.get("lora_adapter_path"),
    )
    with progress_context("eval build-requests", rows=len(rows)):
        requests = _build_eval_requests(cfg=cfg, rows=rows, model_spec=model_spec)
    request_path = output_dir / "eval_requests.jsonl"
    output_path = output_dir / "eval_outputs.jsonl"
    write_jsonl(request_path, requests)
    with progress_context("eval generate", requests=len(requests)):
        responses = _run_generation(
            requests=requests,
            request_path=request_path,
            output_path=output_path,
            force=args.force,
            dry_run=args.dry_run,
        )
    with progress_context("eval materialize", responses=len(responses)):
        translations = _materialize_eval_translations(
            rows=rows,
            requests=requests,
            responses=responses,
        )
    write_jsonl(output_dir / "eval_translations.jsonl", translations)
    with progress_context("eval label-filter", rows=len(translations)):
        filtered = _filter_eval_rows(cfg, translations)
    write_jsonl(output_dir / "eval_filtered.jsonl", filtered)
    with progress_context("eval score", rows=len(filtered)):
        score_rows, metric_summary = _score_eval_metrics(
            cfg=cfg,
            rows=filtered,
            dry_run=args.dry_run,
        )
    write_jsonl(output_dir / "eval_scores.jsonl", score_rows)

    label_counts = Counter(str(row.get("degeneration_label", "unknown")) for row in filtered)
    ok_rows = sum(1 for row in filtered if row.get("status") == "ok")
    fail_rows = len(filtered) - int(label_counts.get("clean", 0))
    summary = {
        "run_id": _get(cfg, "run.id"),
        "eval_profile": _get(cfg, "eval.profile"),
        "dataset_path": str(args.data_path or _get(cfg, "eval.dataset_path")),
        "model_path": model_spec["model_path"],
        "lora_adapter_path": model_spec.get("lora_adapter_path"),
        "output_dir": str(output_dir),
        "rows": len(filtered),
        "generation_ok_rows": ok_rows,
        "filter_pass_rows": int(label_counts.get("clean", 0)),
        "filter_fail_rows": fail_rows,
        "filter_fail_ratio": fail_rows / max(len(filtered), 1),
        "filter_label_counts": dict(sorted(label_counts.items())),
        "metrics": metric_summary,
        "dry_run": bool(args.dry_run),
    }
    (output_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.skip_wandb_log:
        log_wandb_metrics(
            cfg,
            eval_metric_payload(summary, prefix=f"eval/{_get(cfg, 'eval.profile')}"),
            job_type="eval",
            finish=True,
        )
    progress(
        "eval done",
        profile=_get(cfg, "eval.profile"),
        rows=len(filtered),
        generation_ok=ok_rows,
        filter_fail=fail_rows,
        metrics=",".join(sorted(metric_summary)),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
