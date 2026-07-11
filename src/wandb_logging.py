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


def wandb_run_id(cfg: Mapping[str, Any]) -> str | None:
    configured = _get(cfg, "logging.wandb.run_id")
    if configured is None or not str(configured).strip():
        return None
    return _clean_run_id(str(configured))


def configure_wandb_env(cfg: Mapping[str, Any]) -> None:
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ.setdefault("WANDB_SILENT", "true")
    if not wandb_enabled(cfg):
        os.environ["WANDB_MODE"] = "disabled"
        return
    if not _is_main_process():
        os.environ["WANDB_MODE"] = "disabled"
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
    run_id = wandb_run_id(cfg)
    if run_id:
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = str(_get(cfg, "logging.wandb.resume", "allow"))
    else:
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
    if isinstance(tags, list) and tags:
        os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)


def _wandb_init_kwargs(cfg: Mapping[str, Any], job_type: str | None) -> dict[str, Any]:
    tags = _get(cfg, "logging.wandb.tags", [])
    kwargs: dict[str, Any] = {
        "project": _get(cfg, "logging.wandb.project", "dqs"),
        "name": str(_get(cfg, "logging.wandb.run_name", _get(cfg, "run.id", "dqs"))),
    }
    run_id = wandb_run_id(cfg)
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = str(_get(cfg, "logging.wandb.resume", "allow"))
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
        run.define_metric("train/*", step_metric="train/global_step")
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
        # Use explicit metric columns such as train/global_step, subset/index, and
        # checkpoint/step as chart axes. Passing wandb's internal step directly can
        # hide later subset/eval logs when separate resumed processes reuse a step.
        run.log(payload)
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
    payload: dict[str, Any] = {
        "subset/index": summary.get("subset_idx"),
        "subset/filtered_rows": summary.get("student_filter_fail_rows"),
        "subset/filter_blocked_selection_rows": summary.get("student_filter_blocked_selection_rows"),
        "subset/qe_selection_order": summary.get("qe_selection_order"),
        "subset/selected_for_teacher_rows": summary.get("selected_for_teacher_rows"),
        "subset/all_qe_score_mean": summary.get("all_qe_score_mean"),
        "subset/selected_qe_score_mean": summary.get("selected_qe_score_mean"),
        "subset/sft_rows": summary.get("sft_rows"),
        "train/global_step": summary.get("sft_training_global_step"),
    }
    teacher_label_ratios = summary.get("teacher_label_ratios", {})
    if isinstance(teacher_label_ratios, Mapping):
        for label, ratio in teacher_label_ratios.items():
            if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
                payload[f"subset/teacher_label_ratios/{label}"] = float(ratio)
    return payload
