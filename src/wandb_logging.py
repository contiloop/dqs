from __future__ import annotations

import os
import re
import sys
from typing import Any, Mapping


_WANDB_IMPORT_WARNING_SHOWN = False


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def wandb_enabled(cfg: Mapping[str, Any]) -> bool:
    return bool(_get(cfg, "logging.wandb.enabled", False))


def _is_main_process() -> bool:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        raw = os.environ.get(key)
        if raw is not None and raw.strip():
            return raw.strip() == "0"
    return True


def _clean_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return (cleaned or "dqs")[:128]


def wandb_run_id(cfg: Mapping[str, Any]) -> str:
    configured = _get(cfg, "logging.wandb.run_id")
    raw = str(configured or _get(cfg, "run.id", "dqs"))
    return _clean_run_id(raw)


def configure_wandb_env(cfg: Mapping[str, Any]) -> None:
    if not wandb_enabled(cfg):
        return
    project = _get(cfg, "logging.wandb.project")
    entity = _get(cfg, "logging.wandb.entity")
    run_name = _get(cfg, "logging.wandb.run_name", _get(cfg, "run.id", "dqs"))
    tags = _get(cfg, "logging.wandb.tags", [])
    if project:
        os.environ["WANDB_PROJECT"] = str(project)
    if entity:
        os.environ["WANDB_ENTITY"] = str(entity)
    if run_name:
        os.environ["WANDB_NAME"] = str(run_name)
    os.environ["WANDB_RUN_ID"] = wandb_run_id(cfg)
    os.environ["WANDB_RESUME"] = str(_get(cfg, "logging.wandb.resume", "allow"))
    if isinstance(tags, list) and tags:
        os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)


def _wandb_init_kwargs(cfg: Mapping[str, Any], job_type: str | None) -> dict[str, Any]:
    tags = _get(cfg, "logging.wandb.tags", [])
    kwargs: dict[str, Any] = {
        "project": _get(cfg, "logging.wandb.project", "dqs"),
        "name": str(_get(cfg, "logging.wandb.run_name", _get(cfg, "run.id", "dqs"))),
        "id": wandb_run_id(cfg),
        "resume": str(_get(cfg, "logging.wandb.resume", "allow")),
    }
    entity = _get(cfg, "logging.wandb.entity")
    if entity:
        kwargs["entity"] = str(entity)
    if isinstance(tags, list):
        kwargs["tags"] = [str(tag) for tag in tags]
    if job_type:
        kwargs["job_type"] = job_type
    return kwargs


def init_wandb(cfg: Mapping[str, Any], *, job_type: str | None = None) -> Any | None:
    if not wandb_enabled(cfg) or not _is_main_process():
        return None
    configure_wandb_env(cfg)
    try:
        import wandb
    except ModuleNotFoundError:
        global _WANDB_IMPORT_WARNING_SHOWN
        if not _WANDB_IMPORT_WARNING_SHOWN:
            print("[wandb] package not installed; skipping wandb logging", file=sys.stderr)
            _WANDB_IMPORT_WARNING_SHOWN = True
        return None
    try:
        if wandb.run is None:
            run = wandb.init(**_wandb_init_kwargs(cfg, job_type))
        else:
            run = wandb.run
        _define_metrics_once(run)
        return run
    except Exception as exc:
        print(f"[wandb] init failed; skipping wandb logging: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _define_metrics_once(run: Any) -> None:
    if getattr(run, "_dqs_metrics_defined", False):
        return
    try:
        run.define_metric("subset/index")
        run.define_metric("train/global_step")
        run.define_metric("checkpoint/step")
        run.define_metric("subset/*", step_metric="subset/index")
        run.define_metric("stage/*", step_metric="subset/index")
        run.define_metric("eval/val/*", step_metric="train/global_step")
        run.define_metric("eval/final/*", step_metric="train/global_step")
        run.define_metric("eval_checkpoint/*", step_metric="checkpoint/step")
        setattr(run, "_dqs_metrics_defined", True)
    except Exception:
        return


def log_wandb_metrics(
    cfg: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    step: int | None = None,
    job_type: str | None = None,
    finish: bool = False,
) -> None:
    if not metrics:
        return
    run = init_wandb(cfg, job_type=job_type)
    if run is None:
        return
    payload = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }
    try:
        run.log(payload, step=step)
        if finish:
            run.finish()
    except Exception as exc:
        print(f"[wandb] log failed; continuing: {type(exc).__name__}: {exc}", file=sys.stderr)


def eval_metric_payload(summary: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        f"{prefix}/rows": summary.get("rows"),
        f"{prefix}/generation_ok_rows": summary.get("generation_ok_rows"),
        f"{prefix}/filter_pass_rows": summary.get("filter_pass_rows"),
        f"{prefix}/filter_fail_rows": summary.get("filter_fail_rows"),
        f"{prefix}/filter_fail_ratio": summary.get("filter_fail_ratio"),
    }
    metrics = summary.get("metrics", {})
    if isinstance(metrics, Mapping):
        for metric_id, metric in metrics.items():
            if not isinstance(metric, Mapping):
                continue
            value = metric.get("mean", metric.get("score"))
            if isinstance(value, (int, float)):
                payload[f"{prefix}/{metric_id}"] = float(value)
    label_counts = summary.get("filter_label_counts", {})
    if isinstance(label_counts, Mapping):
        for label, count in label_counts.items():
            if isinstance(count, (int, float)):
                payload[f"{prefix}/filter_label_counts/{label}"] = int(count)
    return payload


def subset_summary_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subset/index": summary.get("subset_idx"),
        "subset/input_rows": summary.get("input_rows"),
        "subset/student_rows": summary.get("student_rows"),
        "subset/student_filter_pass_rows": summary.get("student_filter_pass_rows"),
        "subset/student_filter_fail_rows": summary.get("student_filter_fail_rows"),
        "subset/student_filter_fail_ratio": summary.get("student_filter_fail_ratio"),
        "subset/filtered_rows": summary.get("student_filter_fail_rows"),
        "subset/filter_blocked_selection_rows": summary.get("student_filter_blocked_selection_rows"),
        "subset/selected_for_teacher_rows": summary.get("selected_for_teacher_rows"),
        "subset/selected_qe_score_mean": summary.get("selected_qe_score_mean"),
        "subset/teacher_accepted_rows": summary.get("teacher_accepted_rows"),
        "subset/teacher_rejected_rows": summary.get("teacher_rejected_rows"),
        "subset/teacher_shortfall_rows": summary.get("teacher_shortfall_rows"),
        "subset/sft_rows": summary.get("sft_rows"),
        "train/global_step": summary.get("sft_training_global_step"),
    }
