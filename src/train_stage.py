#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config
from progress import progress, progress_context
from runtime_logging import configure_runtime_logging
from train import (
    TRAIN_PHASES,
    _get,
    _load_pool_rows,
    _read_phase_state,
    _subset_root,
)
from sft_train import estimate_update_steps_for_rows
from wandb_logging import eval_metric_payload, log_wandb_metrics, subset_summary_payload


configure_runtime_logging()

def _stage_summary_path(cfg: Mapping[str, Any]) -> Path:
    return Path(str(_get(cfg, "paths.artifact_root"))) / "train_stage_summary.json"


def _front_stage_summary_path(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return _subset_root(cfg, subset_idx) / "front_stage_summary.json"


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _is_completed_subset(cfg: Mapping[str, Any], subset_idx: int) -> bool:
    state = _read_phase_state(_subset_root(cfg, subset_idx))
    return bool(state and state.get("status") == "completed")


def _resolve_stage_end(
    *,
    cfg: Mapping[str, Any],
    data_path: str | None,
    subset_size: int,
    start_subset: int,
    end_subset: int | None,
    max_subsets: int | None,
) -> int:
    if subset_size <= 0:
        raise SystemExit("subset_size must be > 0")
    if end_subset is not None:
        resolved_end = end_subset
    else:
        configured_end = _get(cfg, "run.subset_end")
        if configured_end is not None:
            resolved_end = int(configured_end)
        else:
            rows = _load_pool_rows(cfg, data_path)
            resolved_end = int(math.ceil(len(rows) / subset_size))

    if max_subsets is not None:
        if max_subsets <= 0:
            raise SystemExit("stage_max_subsets must be > 0 when set")
        resolved_end = min(resolved_end, start_subset + max_subsets)

    if resolved_end <= start_subset:
        raise SystemExit(
            f"no subsets selected for train-stage: start={start_subset}, end={resolved_end}"
        )
    return resolved_end


def _train_subset_cmd(
    *,
    args: argparse.Namespace,
    subset_idx: int,
    start_from_phase: str | None,
    sft_scheduler_total_steps: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--config",
        args.config,
        "--subset-idx",
        str(subset_idx),
        "--resume",
        args.resume,
    ]
    if args.subset_size is not None:
        cmd.extend(["--subset-size", str(args.subset_size)])
    if args.data_path:
        cmd.extend(["--data-path", args.data_path])
    if start_from_phase:
        cmd.extend(["--start-from-phase", start_from_phase])
    if sft_scheduler_total_steps is not None and not args.dry_run:
        cmd.extend(["--sft-scheduler-total-steps", str(sft_scheduler_total_steps)])
        cmd.append("--sft-force-save-checkpoint")
    for override in args.override:
        cmd.extend(["--override", override])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")
    return cmd


def _planned_sft_rows_per_subset(cfg: Mapping[str, Any]) -> int:
    configured = _get(cfg, "training.stage_planned_rows_per_subset")
    if configured is not None:
        return max(1, int(configured))
    teacher_target = int(_get(cfg, "data.teacher_target_per_subset", 0) or 0)
    if teacher_target > 0:
        return teacher_target
    subset_size = int(_get(cfg, "data.subset_size", 100000) or 100000)
    selection_ratio = float(_get(cfg, "data.selection_ratio", 0.01) or 0.01)
    return max(1, math.ceil(subset_size * selection_ratio))


def _stage_scheduler_total_steps(
    *,
    cfg: Mapping[str, Any],
    stage_end: int,
) -> int:
    cycle_start = int(_get(cfg, "run.subset_start", 0) or 0)
    subset_count = max(1, stage_end - cycle_start)
    planned_rows = _planned_sft_rows_per_subset(cfg)
    planned_steps_per_subset = estimate_update_steps_for_rows(planned_rows, cfg)
    return subset_count * planned_steps_per_subset


def _eval_output_dir(
    *,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    subset_idx: int,
) -> Path:
    base = (
        Path(args.eval_output_dir)
        if args.eval_output_dir
        else Path(str(_get(cfg, "paths.artifact_root"))) / "eval" / args.eval_profile
    )
    return base / f"subset_{subset_idx:03d}"


def _eval_cmd(
    *,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    subset_idx: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("eval.py")),
        "--config",
        args.eval_config or args.config,
        "--override",
        f"eval={args.eval_profile}",
        "--output-dir",
        str(_eval_output_dir(cfg=cfg, args=args, subset_idx=subset_idx)),
        "--skip-wandb-log",
    ]
    for override in args.override:
        cmd.extend(["--override", override])
    for override in args.eval_override:
        cmd.extend(["--override", override])
    if args.eval_data_path:
        cmd.extend(["--data-path", args.eval_data_path])
    if args.eval_model_path:
        cmd.extend(["--model-path", args.eval_model_path])
    if args.eval_limit is not None:
        cmd.extend(["--limit", str(args.eval_limit)])
    if args.eval_metrics:
        cmd.extend(["--metrics", args.eval_metrics])
    if args.eval_dry_run:
        cmd.append("--dry-run")
    if args.eval_force:
        cmd.append("--force")
    return cmd


def _should_eval_after_subset(
    *,
    args: argparse.Namespace,
    stage_start: int,
    stage_end: int,
    subset_idx: int,
) -> bool:
    every_n = int(args.eval_every_n_subsets or 0)
    if every_n <= 0:
        return False
    offset = subset_idx - stage_start + 1
    if offset % every_n == 0:
        return True
    return bool(args.eval_on_final_subset and subset_idx == stage_end - 1)


def _write_stage_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _metric_step_from_subset_summary(summary: Mapping[str, Any]) -> int | None:
    value = summary.get("sft_training_global_step")
    return int(value) if isinstance(value, int) else None


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    cfg = compose_config(args.config, overrides=args.override)
    start_subset = int(args.subset_idx if args.subset_idx is not None else _get(cfg, "run.subset_start", 0))
    subset_size = int(args.subset_size if args.subset_size is not None else _get(cfg, "data.subset_size", 100000))
    stage_end = _resolve_stage_end(
        cfg=cfg,
        data_path=args.data_path,
        subset_size=subset_size,
        start_subset=start_subset,
        end_subset=args.stage_end_subset,
        max_subsets=args.stage_max_subsets,
    )
    sft_scheduler_total_steps = _stage_scheduler_total_steps(cfg=cfg, stage_end=stage_end)
    summary_path = _stage_summary_path(cfg)
    summary: dict[str, Any] = {
        "run_id": _get(cfg, "run.id"),
        "stage_start_subset": start_subset,
        "stage_end_subset": stage_end,
        "stage_end_subset_exclusive": True,
        "subset_size": subset_size,
        "resume_mode": args.resume,
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "eval_every_n_subsets": int(args.eval_every_n_subsets or 0),
        "eval_on_final_subset": bool(args.eval_on_final_subset),
        "eval_profile": args.eval_profile,
        "sft_scheduler_total_steps": sft_scheduler_total_steps,
        "sft_planned_rows_per_subset": _planned_sft_rows_per_subset(cfg),
        "subsets": [],
    }
    _write_stage_summary(summary_path, summary)
    progress(
        "stage start",
        run=_get(cfg, "run.id"),
        start_subset=start_subset,
        end_subset_exclusive=stage_end,
        subset_size=subset_size,
        eval_every_n_subsets=args.eval_every_n_subsets,
    )

    start_from_phase_used = False
    for subset_idx in range(start_subset, stage_end):
        subset_record: dict[str, Any] = {
            "subset_idx": subset_idx,
            "subset_name": f"subset_{subset_idx:03d}",
        }
        if _is_completed_subset(cfg, subset_idx) and not args.force:
            subset_record["status"] = "skipped_completed"
            subset_record["summary_path"] = str(_front_stage_summary_path(cfg, subset_idx))
            summary["subsets"].append(subset_record)
            _write_stage_summary(summary_path, summary)
            progress("stage subset skip", subset=f"subset_{subset_idx:03d}", reason="completed")
            continue

        start_from_phase = None
        if args.start_from_phase and not start_from_phase_used:
            start_from_phase = args.start_from_phase
            start_from_phase_used = True

        train_cmd = _train_subset_cmd(
            args=args,
            subset_idx=subset_idx,
            start_from_phase=start_from_phase,
            sft_scheduler_total_steps=sft_scheduler_total_steps,
        )
        subset_record["train_command"] = " ".join(train_cmd)
        try:
            with progress_context("stage subset", subset=f"subset_{subset_idx:03d}"):
                subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            subset_record["status"] = "failed"
            subset_record["error"] = f"train exited with code {exc.returncode}"
            subset_record["summary_path"] = str(_front_stage_summary_path(cfg, subset_idx))
            summary["subsets"].append(subset_record)
            summary["status"] = "failed"
            summary["failed_subset_idx"] = subset_idx
            _write_stage_summary(summary_path, summary)
            raise SystemExit(exc.returncode) from exc

        subset_record["status"] = "completed"
        subset_record["summary_path"] = str(_front_stage_summary_path(cfg, subset_idx))
        subset_record["summary"] = _read_json_if_exists(_front_stage_summary_path(cfg, subset_idx))
        metric_step = _metric_step_from_subset_summary(subset_record["summary"])
        subset_metrics = subset_summary_payload(subset_record["summary"])
        log_wandb_metrics(cfg, subset_metrics, step=metric_step, job_type="train-stage", finish=True)

        if _should_eval_after_subset(
            args=args,
            stage_start=start_subset,
            stage_end=stage_end,
            subset_idx=subset_idx,
        ):
            eval_cmd = _eval_cmd(cfg=cfg, args=args, subset_idx=subset_idx)
            subset_record["eval_command"] = " ".join(eval_cmd)
            try:
                with progress_context("stage eval", subset=f"subset_{subset_idx:03d}", profile=args.eval_profile):
                    subprocess.run(eval_cmd, check=True)
            except subprocess.CalledProcessError as exc:
                subset_record["eval_status"] = "failed"
                subset_record["eval_error"] = f"eval exited with code {exc.returncode}"
                summary["subsets"].append(subset_record)
                summary["status"] = "failed"
                summary["failed_subset_idx"] = subset_idx
                summary["failed_stage"] = "eval"
                _write_stage_summary(summary_path, summary)
                raise SystemExit(exc.returncode) from exc
            subset_record["eval_status"] = "completed"
            subset_record["eval_output_dir"] = str(
                _eval_output_dir(cfg=cfg, args=args, subset_idx=subset_idx)
            )
            eval_summary = _read_json_if_exists(
                _eval_output_dir(cfg=cfg, args=args, subset_idx=subset_idx) / "eval_summary.json"
            )
            eval_metrics = eval_metric_payload(eval_summary, prefix=f"eval/{args.eval_profile}")
            eval_metrics["subset/index"] = subset_idx
            if metric_step is not None:
                eval_metrics["train/global_step"] = metric_step
            log_wandb_metrics(cfg, eval_metrics, step=metric_step, job_type="eval", finish=True)

        summary["subsets"].append(subset_record)
        _write_stage_summary(summary_path, summary)

    summary["status"] = "completed"
    _write_stage_summary(summary_path, summary)
    progress("stage done", run=_get(cfg, "run.id"), subsets=len(summary["subsets"]))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQS training across multiple subsets.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset-idx", type=int, default=None, help="First subset index to run.")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--start-from-phase", choices=list(TRAIN_PHASES), default=None)
    parser.add_argument("--resume", choices=["auto", "none"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-end-subset", type=int, default=None)
    parser.add_argument("--stage-max-subsets", type=int, default=None)
    parser.add_argument("--eval-every-n-subsets", type=int, default=0)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--eval-profile", default="train")
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--eval-model-path", default=None)
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-metrics", default=None)
    parser.add_argument("--eval-dry-run", action="store_true")
    parser.add_argument("--eval-force", action="store_true")
    parser.add_argument("--eval-on-final-subset", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run_stage(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
