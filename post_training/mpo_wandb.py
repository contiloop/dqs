"""Strict W&B integration for the isolated DQS mPO post-training run.

This module deliberately does not import Transformers at module import time.
Unsloth must patch Transformers before the Trainer callback class is imported.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WANDB_REQUIRED_VERSION = "0.28.0"
_METRIC_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def require_wandb_runtime_version() -> None:
    try:
        observed = importlib.metadata.version("wandb")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"strict W&B logging requires wandb=={WANDB_REQUIRED_VERSION}; package is missing"
        ) from exc
    if observed != WANDB_REQUIRED_VERSION:
        raise RuntimeError(
            f"strict W&B logging requires wandb=={WANDB_REQUIRED_VERSION}, observed={observed}"
        )


@dataclass(frozen=True)
class MPOWandbConfig:
    """Validated W&B contract shared by training and post-training eval."""

    enabled: bool
    strict: bool
    mode: str
    project: str
    entity: str | None
    run_name: str
    run_id: str
    group: str
    job_type: str
    resume: str
    local_subdir: str
    log_smoke: bool
    log_model: bool
    watch: bool
    tags: tuple[str, ...]

    @classmethod
    def from_logging_config(
        cls,
        logging_cfg: Mapping[str, Any],
        *,
        post_training_run_id: str,
    ) -> "MPOWandbConfig":
        raw = logging_cfg.get("wandb")
        if not isinstance(raw, Mapping):
            raise ValueError("logging.wandb must be an explicit mapping")
        required = (
            "enabled",
            "strict",
            "mode",
            "project",
            "entity",
            "run_name",
            "run_id",
            "group",
            "job_type",
            "resume",
            "local_subdir",
            "log_smoke",
            "log_model",
            "watch",
            "tags",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"logging.wandb is missing explicit keys: {missing}")
        for key in ("enabled", "strict", "log_smoke", "log_model", "watch"):
            if type(raw[key]) is not bool:
                raise ValueError(f"logging.wandb.{key} must be an explicit boolean")
        for key in (
            "mode",
            "project",
            "run_name",
            "run_id",
            "group",
            "job_type",
            "resume",
            "local_subdir",
        ):
            if not isinstance(raw[key], str):
                raise ValueError(f"logging.wandb.{key} must be an explicit string")
        tags = raw["tags"]
        if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
            raise ValueError("logging.wandb.tags must be a sequence of non-empty strings")
        if any(not isinstance(tag, str) for tag in tags):
            raise ValueError("logging.wandb.tags must contain strings only")
        normalized_tags = tuple(tag.strip() for tag in tags)
        if not normalized_tags or any(not tag for tag in normalized_tags):
            raise ValueError("logging.wandb.tags must contain non-empty strings")
        entity_raw = raw["entity"]
        if entity_raw is not None and not isinstance(entity_raw, str):
            raise ValueError("logging.wandb.entity must be null or a string")
        if entity_raw is not None and not entity_raw.strip():
            raise ValueError("logging.wandb.entity must be null or a non-empty string")
        config = cls(
            enabled=bool(raw["enabled"]),
            strict=bool(raw["strict"]),
            mode=str(raw["mode"]).strip().lower(),
            project=str(raw["project"]).strip(),
            entity=None if entity_raw is None else entity_raw.strip(),
            run_name=str(raw["run_name"]).strip(),
            run_id=str(raw["run_id"]).strip(),
            group=str(raw["group"]).strip(),
            job_type=str(raw["job_type"]).strip(),
            resume=str(raw["resume"]).strip().lower(),
            local_subdir=str(raw["local_subdir"]).strip(),
            log_smoke=bool(raw["log_smoke"]),
            log_model=bool(raw["log_model"]),
            watch=bool(raw["watch"]),
            tags=normalized_tags,
        )
        config.validate(post_training_run_id=post_training_run_id)
        return config

    def validate(self, *, post_training_run_id: str) -> None:
        if not self.enabled:
            raise ValueError("logging.wandb.enabled must remain true for the full mPO run")
        if not self.strict:
            raise ValueError("logging.wandb.strict must remain true; silent logging fallback is forbidden")
        if self.mode != "online":
            raise ValueError("logging.wandb.mode must be exactly 'online'")
        for key, value in (
            ("project", self.project),
            ("run_name", self.run_name),
            ("run_id", self.run_id),
            ("group", self.group),
            ("job_type", self.job_type),
        ):
            if not value:
                raise ValueError(f"logging.wandb.{key} must be non-empty")
        if self.run_id != post_training_run_id:
            raise ValueError(
                "logging.wandb.run_id must exactly equal run.id so training, resume, and eval "
                "cannot split across W&B runs"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.run_id):
            raise ValueError(
                "logging.wandb.run_id must be 1-64 ASCII letters, digits, underscores, or hyphens; "
                "W&B must not normalize the configured stable ID"
            )
        if self.resume != "allow":
            raise ValueError("logging.wandb.resume must be exactly 'allow'")
        if self.local_subdir != "wandb":
            raise ValueError("logging.wandb.local_subdir must be exactly 'wandb'")
        if self.log_smoke:
            raise ValueError("logging.wandb.log_smoke must remain false; smoke must not pollute the full run")
        if self.log_model or self.watch:
            raise ValueError("W&B model upload and parameter watching must remain disabled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strict": self.strict,
            "mode": self.mode,
            "project": self.project,
            "entity": self.entity,
            "run_name": self.run_name,
            "run_id": self.run_id,
            "group": self.group,
            "job_type": self.job_type,
            "resume": self.resume,
            "local_subdir": self.local_subdir,
            "log_smoke": self.log_smoke,
            "log_model": self.log_model,
            "watch": self.watch,
            "tags": list(self.tags),
        }


def _sanitize_metric_segment(value: Any) -> str:
    segment = _METRIC_SEGMENT_RE.sub("_", str(value).strip()).strip("_")
    return segment or "unnamed"


def _flatten_numeric_metrics(
    payload: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, float | int]:
    """Flatten finite numeric leaves, rejecting ambiguous metric collisions."""

    flattened: dict[str, float | int] = {}

    def visit(value: Any, path: list[str]) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], [*path, _sanitize_metric_segment(key)])
            return
        if isinstance(value, bool):
            number: float | int = int(value)
        elif isinstance(value, int):
            number = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"non-finite W&B metric at {'/'.join(path)}: {value}")
            number = value
        else:
            return
        name = "/".join([prefix.rstrip("/"), *path])
        if name in flattened:
            raise ValueError(f"W&B metric name collision after sanitization: {name}")
        flattened[name] = number

    visit(payload, [])
    return flattened


def map_trainer_logs(logs: Mapping[str, Any]) -> dict[str, float | int]:
    """Map Trainer keys without Transformers' ``train/train`` rewrite."""

    exact = {
        "loss": "train/loss/hf_global",
        "grad_norm": "train/optimizer/grad_norm",
        "learning_rate": "train/optimizer/learning_rate",
        "epoch": "train/progress/epoch",
        "train_loss": "train/loss/hf_epoch",
        "train_runtime": "train/runtime/seconds",
        "train_samples_per_second": "train/runtime/samples_per_second",
        "train_steps_per_second": "train/runtime/steps_per_second",
        "total_flos": "train/runtime/total_flos",
        "eval_loss": "eval/loss/hf_global",
    }
    mapped: dict[str, float | int] = {}
    for raw_key, raw_value in logs.items():
        if isinstance(raw_value, bool):
            value: float | int = int(raw_value)
        elif isinstance(raw_value, (int, float)):
            value = raw_value
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite Trainer metric {raw_key}={value}")
        else:
            continue
        key = str(raw_key)
        if key.startswith(("train/", "eval/")):
            target = key
        elif key in exact:
            target = exact[key]
        elif key.startswith("eval_"):
            target = f"eval/hf/{_sanitize_metric_segment(key[5:])}"
        elif key.startswith("train_"):
            target = f"train/hf/{_sanitize_metric_segment(key[6:])}"
        else:
            target = f"train/hf/{_sanitize_metric_segment(key)}"
        if target in mapped:
            raise ValueError(f"duplicate mapped W&B metric: {target}")
        mapped[target] = value
    return mapped


class MPOWandbRun:
    """Thin strict wrapper; active method failures are intentionally fatal."""

    def __init__(self, run: Any | None = None) -> None:
        self._run = run
        self._finished = False

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, payload: Mapping[str, Any]) -> None:
        if not self.active:
            return
        if self._finished:
            raise RuntimeError("attempted to log to a finished W&B run")
        self._run.log(dict(payload), commit=True)

    def define_eval_metrics(self, profile: str) -> None:
        if self.active:
            safe_profile = _sanitize_metric_segment(profile)
            self._run.define_metric(
                f"eval/{safe_profile}/*",
                step_metric="train/global_step",
            )

    def log_eval_summary(
        self,
        *,
        profile: str,
        summary: Mapping[str, Any],
        global_step: int,
    ) -> None:
        if not self.active:
            return
        if global_step < 0:
            raise ValueError("eval W&B global_step must be non-negative")
        safe_profile = _sanitize_metric_segment(profile)
        self.define_eval_metrics(safe_profile)
        payload: dict[str, Any] = {
            "train/global_step": global_step,
            f"eval/{safe_profile}/completed": 1,
        }
        payload.update(_flatten_numeric_metrics(summary, prefix=f"eval/{safe_profile}"))
        self.log(payload)

    def log_final(
        self,
        *,
        global_step: int,
        final_saved: bool,
        train_metrics: Mapping[str, Any],
    ) -> None:
        if not self.active:
            return
        payload: dict[str, Any] = {
            "train/global_step": global_step,
            "train/status/completed": 1,
            "train/status/final_saved": int(final_saved),
        }
        payload.update(map_trainer_logs(train_metrics))
        self.log(payload)

    def log_failure(self, *, global_step: int) -> None:
        if self.active:
            self.log(
                {
                    "train/global_step": max(0, int(global_step)),
                    "train/status/failed": 1,
                }
            )

    def finish(self, *, exit_code: int) -> None:
        if not self.active or self._finished:
            return
        self._run.finish(exit_code=exit_code)
        self._finished = True


def _configure_environment(*, mode: str, output_dir: Path) -> None:
    os.environ["WANDB_MODE"] = mode
    # W&B creates its standard ``wandb/`` child under this base directory.
    os.environ["WANDB_DIR"] = str(output_dir)
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_DISABLE_CODE"] = "true"


def initialize_wandb_run(
    config: MPOWandbConfig,
    *,
    output_dir: Path,
    metadata: Mapping[str, Any] | None,
    active_process: bool,
    wandb_module: Any | None = None,
) -> MPOWandbRun:
    """Initialize only the designated rank; disabled ranks never import W&B."""

    resolved_output = output_dir.resolve()
    local_dir = (resolved_output / config.local_subdir).resolve()
    if resolved_output not in local_dir.parents:
        raise ValueError(f"W&B directory escaped post-training output: {local_dir}")
    mode = config.mode if active_process else "disabled"
    _configure_environment(mode=mode, output_dir=resolved_output)
    if not active_process:
        return MPOWandbRun()
    local_dir.mkdir(parents=True, exist_ok=True)
    if wandb_module is None:
        import wandb as wandb_module  # type: ignore[no-redef]
    init_kwargs: dict[str, Any] = {
        "project": config.project,
        "name": config.run_name,
        "id": config.run_id,
        "resume": config.resume,
        "group": config.group,
        "job_type": config.job_type,
        "tags": list(config.tags),
        "mode": config.mode,
        "dir": str(resolved_output),
        "config": dict(metadata or {}),
        # W&B otherwise permits an unauthenticated process to continue in
        # offline mode. The post-training contract forbids that fallback.
        "force": True,
        "save_code": False,
    }
    if config.entity is not None:
        init_kwargs["entity"] = config.entity
    run = wandb_module.init(**init_kwargs)
    if run is None:
        raise RuntimeError("wandb.init returned no run while strict online logging is enabled")
    if bool(getattr(run, "disabled", False)):
        raise RuntimeError("wandb.init returned a disabled run while strict online logging is enabled")
    session = MPOWandbRun(run)
    try:
        run.define_metric("train/global_step")
        run.define_metric("train/*", step_metric="train/global_step")
    except BaseException as define_error:
        try:
            session.finish(exit_code=1)
        except BaseException as finish_error:
            raise finish_error from define_error
        raise
    return session


def build_wandb_callback(session: MPOWandbRun) -> Any:
    """Create the callback only after Unsloth has imported Transformers."""

    from transformers import TrainerCallback

    class StrictMPOWandbCallback(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if session.active:
                session.log(
                    {
                        "train/global_step": int(state.global_step),
                        "train/status/started": 1,
                    }
                )

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            del args, control, kwargs
            if session.active and logs:
                payload: dict[str, Any] = {"train/global_step": int(state.global_step)}
                payload.update(map_trainer_logs(logs))
                session.log(payload)

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if session.active:
                session.log(
                    {
                        "train/global_step": int(state.global_step),
                        "train/checkpoint/saved": 1,
                    }
                )

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if session.active:
                session.log(
                    {
                        "train/global_step": int(state.global_step),
                        "train/status/trainer_completed": 1,
                    }
                )

    return StrictMPOWandbCallback()
