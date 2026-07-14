#!/usr/bin/env python3
"""Evaluate only the final mPO model into its isolated post-training run tree."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from .mpo_wandb import (
        MPOWandbConfig,
        initialize_wandb_run,
        require_wandb_runtime_version,
    )
except ImportError:  # Direct ``python post_training/eval_mpo.py`` execution.
    from mpo_wandb import (
        MPOWandbConfig,
        initialize_wandb_run,
        require_wandb_runtime_version,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
POST_TRAINING_ROOT = REPO_ROOT / "post_training"
DEFAULT_TRAIN_CONFIG = POST_TRAINING_ROOT / "configs" / "mpo_setting5.yaml"
_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class MPOEvalPaths:
    run_id: str
    final_model_dir: Path
    output_dir: Path
    base_eval_config: Path
    model_profile: str
    training_profile: str
    profile: str


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"post-training config section {key!r} must be a mapping")
    return value


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def load_post_training_config(path: str | Path) -> Mapping[str, Any]:
    config_path = _repo_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"post-training config must be a mapping: {config_path}")
    return payload


def resolve_eval_paths(
    payload: Mapping[str, Any],
    *,
    profile: str | None = None,
    require_final_artifact: bool = True,
) -> MPOEvalPaths:
    run_cfg = _mapping(payload, "run")
    eval_cfg = _mapping(payload, "evaluation")
    run_id = str(run_cfg.get("id", "")).strip()
    if not run_id:
        raise ValueError("run.id must be non-empty")

    run_output = _repo_path(str(run_cfg.get("output_dir", "")))
    if not _is_within(run_output, POST_TRAINING_ROOT):
        raise ValueError(f"post-training run output must stay under {POST_TRAINING_ROOT}: {run_output}")

    selected_profile = str(profile or eval_cfg.get("default_profile", "val")).strip()
    if not _PROFILE_RE.fullmatch(selected_profile):
        raise ValueError(f"invalid eval profile {selected_profile!r}")
    eval_group = REPO_ROOT / "configs" / "eval" / f"{selected_profile}.yaml"
    if not eval_group.is_file():
        raise ValueError(f"unknown DQS eval profile: {selected_profile}")

    output_subdir = str(eval_cfg.get("output_subdir", "eval")).strip()
    if output_subdir != "eval":
        raise ValueError("evaluation.output_subdir must be exactly 'eval'")
    eval_root = (run_output / output_subdir).resolve()
    output_dir = (eval_root / selected_profile).resolve()
    if not _is_within(output_dir, eval_root):
        raise ValueError(f"eval output escaped the post-training eval root: {output_dir}")

    final_model_dir = (run_output / "final").resolve()
    if require_final_artifact:
        marker_path = final_model_dir / "dqs_mpo_model.json"
        if not marker_path.is_file():
            raise ValueError(f"final mPO provenance marker is missing: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if str(marker.get("run_id", "")) != run_id or int(marker.get("paper_setting", -1)) != 5:
            raise ValueError(
                "final mPO provenance marker does not match this run: "
                f"expected run_id={run_id!r}, paper_setting=5"
            )
        weight_files = list(final_model_dir.glob("*.safetensors")) + list(
            final_model_dir.glob("pytorch_model*.bin")
        )
        if not (final_model_dir / "config.json").is_file() or not weight_files:
            raise ValueError(f"final mPO model artifact is incomplete: {final_model_dir}")

    base_eval_config = _repo_path(str(eval_cfg.get("base_config", "")))
    if not base_eval_config.is_file():
        raise ValueError(f"base eval config does not exist: {base_eval_config}")
    model_profile = str(eval_cfg.get("model_profile", "")).strip()
    training_profile = str(eval_cfg.get("training_profile", "")).strip()
    if not model_profile or not training_profile:
        raise ValueError("evaluation.model_profile and evaluation.training_profile must be non-empty")

    return MPOEvalPaths(
        run_id=run_id,
        final_model_dir=final_model_dir,
        output_dir=output_dir,
        base_eval_config=base_eval_config,
        model_profile=model_profile,
        training_profile=training_profile,
        profile=selected_profile,
    )


def build_eval_command(paths: MPOEvalPaths, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "eval.py"),
        "--config",
        str(paths.base_eval_config),
        "--override",
        f"eval={paths.profile}",
        "--override",
        f"run.id={paths.run_id}",
        "--override",
        f"model={paths.model_profile}",
        "--override",
        f"training={paths.training_profile}",
        "--model-path",
        str(paths.final_model_dir),
        "--output-dir",
        str(paths.output_dir),
        # The shared evaluator is best-effort. This wrapper owns the one
        # strict W&B append to the stable post-training run instead.
        "--skip-wandb-log",
    ]
    for override in args.override:
        command.extend(["--override", override])
    if args.data_path:
        command.extend(["--data-path", args.data_path])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.metrics:
        command.extend(["--metrics", args.metrics])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    return command


def log_eval_to_post_training_wandb(
    payload: Mapping[str, Any],
    paths: MPOEvalPaths,
) -> None:
    """Append one completed eval summary to the training run, or fail."""

    logging_cfg = _mapping(payload, "logging")
    wandb_config = MPOWandbConfig.from_logging_config(
        logging_cfg,
        post_training_run_id=paths.run_id,
    )
    require_wandb_runtime_version()

    marker_path = paths.final_model_dir / "dqs_mpo_model.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    global_step = int(marker.get("global_step", -1))
    if global_step < 0:
        raise ValueError(f"final mPO marker has invalid global_step: {marker_path}")

    summary_path = paths.output_dir / "eval_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"successful evaluator did not write its summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError(f"eval summary must be a mapping: {summary_path}")
    if str(summary.get("run_id", "")) != paths.run_id:
        raise ValueError("eval summary run_id does not match the post-training run")
    if str(summary.get("eval_profile", "")) != paths.profile:
        raise ValueError("eval summary profile does not match the requested post-training profile")
    observed_model = _repo_path(str(summary.get("model_path", "")))
    if observed_model != paths.final_model_dir:
        raise ValueError(
            "eval summary model is not the final post-training artifact: "
            f"observed={observed_model}, required={paths.final_model_dir}"
        )

    session = initialize_wandb_run(
        wandb_config,
        output_dir=paths.final_model_dir.parent,
        # Do not mutate the already established training config on resume.
        metadata=None,
        active_process=True,
    )
    try:
        session.log_eval_summary(
            profile=paths.profile,
            summary=summary,
            global_step=global_step,
        )
    except BaseException as eval_logging_error:
        try:
            session.finish(exit_code=1)
        except BaseException as finish_error:
            raise finish_error from eval_logging_error
        raise
    else:
        session.finish(exit_code=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--profile", default=None, help="DQS eval profile, usually val or final")
    parser.add_argument("--override", action="append", default=[], help="Additional DQS eval override")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wandb-log", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_post_training_config(args.config)
    paths = resolve_eval_paths(payload, profile=args.profile, require_final_artifact=True)
    command = build_eval_command(paths, args)
    if args.print_command:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not args.dry_run and not args.skip_wandb_log:
        log_eval_to_post_training_wandb(payload, paths)


if __name__ == "__main__":
    main()
