"""Shared strict runtime helpers for mPO, CPO, and DPO entry points.

This module deliberately contains only training-runtime concerns.  Dataset
synthesis and review code must not depend on it, and deployable trainers must
not import private helpers from another objective's entry point.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

try:
    from .mpo_wandb import WANDB_REQUIRED_VERSION
except ImportError:  # Direct execution from a flat source directory.
    from mpo_wandb import WANDB_REQUIRED_VERSION


SOURCE_ROOT = Path(__file__).resolve().parent
_RELEASE_MARKER = ".dqs-preference-release"
if SOURCE_ROOT.name == "src" and (SOURCE_ROOT.parent / _RELEASE_MARKER).is_file():
    # Generated deployment layout: <release>/src/*.py
    POST_TRAINING_ROOT = SOURCE_ROOT.parent
    REPO_ROOT = POST_TRAINING_ROOT
else:
    # Research layout: <dqs>/post_training/*.py
    POST_TRAINING_ROOT = SOURCE_ROOT
    REPO_ROOT = POST_TRAINING_ROOT.parent


REQUIRED_RUNTIME_VERSIONS = {
    "accelerate": "1.14.0",
    "datasets": "4.3.0",
    "huggingface-hub": "1.21.0",
    "pyyaml": "6.0.3",
    "tokenizers": "0.22.2",
    "trl": "0.24.0",
    "transformers": "5.5.3",
    "unsloth": "2026.7.2",
    "unsloth-zoo": "2026.7.2",
    "wandb": WANDB_REQUIRED_VERSION,
}


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("override key cannot be empty")
    current = config
    for part in parts[:-1]:
        nested = current.get(part)
        if nested is None:
            nested = {}
            current[part] = nested
        if not isinstance(nested, dict):
            raise ValueError(f"cannot descend through non-mapping override key {part!r}")
        current = nested
    current[parts[-1]] = value


def load_config(path: Path, overrides: Sequence[str]) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be KEY=VALUE: {override!r}")
        key, raw_value = override.split("=", 1)
        _set_dotted(payload, key.strip(), yaml.safe_load(raw_value))
    return payload


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _output_dir(run_cfg: Mapping[str, Any], *, smoke_step: bool) -> Path:
    raw = run_cfg.get("output_dir")
    if not raw:
        raise ValueError("run.output_dir must be configured explicitly")
    path = _resolve_repo_path(str(raw)).resolve()
    root = POST_TRAINING_ROOT.resolve()
    if path != root and root not in path.parents:
        raise ValueError(
            f"run.output_dir must stay under the post-training project root {root}: {path}"
        )
    if path == root:
        raise ValueError("run.output_dir cannot be the post-training project root itself")
    return path / "smoke_step" if smoke_step else path


def _world_size() -> int:
    value = int(os.environ.get("WORLD_SIZE", "1") or 1)
    if value <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {value}")
    return value


def _global_rank() -> int:
    rank = int(os.environ.get("RANK", "0") or 0)
    world_size = _world_size()
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK={rank} is outside WORLD_SIZE={world_size}")
    return rank


def _gradient_accumulation_steps(training_cfg: Mapping[str, Any]) -> int:
    configured = training_cfg.get("gradient_accumulation_steps", "auto")
    if str(configured).lower() != "auto":
        value = int(configured)
        if value <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        return value
    per_device = int(training_cfg.get("per_device_train_batch_size", 1))
    effective = int(training_cfg.get("effective_batch_size", 128))
    denominator = per_device * _world_size()
    if effective % denominator:
        raise ValueError(
            "effective_batch_size must be divisible by "
            "per_device_train_batch_size * WORLD_SIZE: "
            f"{effective} % {denominator} != 0"
        )
    return effective // denominator


def _training_argument_kwargs(
    *,
    run_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    output_dir: Path,
    has_eval: bool,
    smoke_step: bool,
) -> dict[str, Any]:
    report_to = training_cfg.get("report_to", [])
    if isinstance(report_to, str):
        report_to = [] if report_to.lower() in {"none", "[]", ""} else [report_to]
    eval_strategy = str(training_cfg.get("eval_strategy", "steps" if has_eval else "no"))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": _gradient_accumulation_steps(training_cfg),
        "learning_rate": float(training_cfg.get("learning_rate", 5e-6)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1)),
        "lr_scheduler_type": str(training_cfg.get("scheduler", "cosine")),
        "optim": str(training_cfg.get("optimizer", "adamw_torch")),
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 5.0)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 1.0)),
        "max_steps": 1 if smoke_step else int(training_cfg.get("max_steps", -1)),
        "save_strategy": "no" if smoke_step else str(training_cfg.get("save_strategy", "steps")),
        "save_steps": int(training_cfg.get("save_steps", 100)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "logging_steps": 1 if smoke_step else int(training_cfg.get("logging_steps", 1)),
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(training_cfg.get("dataloader_pin_memory", True)),
        "remove_unused_columns": False,
        "label_names": [],
        "prediction_loss_only": True,
        "seed": int(run_cfg.get("seed", 42)),
        "data_seed": int(run_cfg.get("seed", 42)),
        "bf16": True,
        "fp16": False,
        "report_to": list(report_to),
        "run_name": str(run_cfg.get("id", "dqs_preference_training")),
        "ddp_find_unused_parameters": (
            bool(training_cfg.get("ddp_find_unused_parameters", False))
            if _world_size() > 1
            else None
        ),
        "eval_steps": int(training_cfg.get("eval_steps", 250)),
    }
    kwargs["eval_strategy"] = eval_strategy
    if kwargs["ddp_find_unused_parameters"] is None:
        del kwargs["ddp_find_unused_parameters"]
    return kwargs


def _training_arguments(
    *,
    run_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    output_dir: Path,
    has_eval: bool,
    smoke_step: bool,
) -> Any:
    import torch
    from transformers import TrainingArguments

    dtype = str(training_cfg.get("dtype", "")).strip().lower()
    if dtype not in {"bf16", "bfloat16"}:
        raise ValueError(
            "training.dtype must be bfloat16; no automatic precision substitution is allowed"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the configured bfloat16 CUDA execution contract is unavailable")
    kwargs = _training_argument_kwargs(
        run_cfg=run_cfg,
        training_cfg=training_cfg,
        output_dir=output_dir,
        has_eval=has_eval,
        smoke_step=smoke_step,
    )
    # This is the exact Transformers 5.5.3 constructor surface. In v5,
    # overwrite_output_dir was removed and model saving is safetensors-only,
    # so save_safetensors was removed as well. Never signature-filter kwargs.
    return TrainingArguments(**kwargs)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def _resume_checkpoint(
    training_cfg: Mapping[str, Any], output_dir: Path, *, smoke_step: bool
) -> str | None:
    if smoke_step:
        return None
    raw = training_cfg.get("resume_from_checkpoint")
    if raw in {None, False, "", "none", "null", "false"}:
        if (output_dir / "final").exists() or (output_dir / "run_manifest.json").exists():
            raise ValueError(
                f"a completed post-training run already exists under {output_dir}; "
                "use a new run.output_dir"
            )
        if _latest_checkpoint(output_dir) is not None:
            raise ValueError(
                f"post-training checkpoints already exist under {output_dir}; "
                "set training.resume_from_checkpoint=auto or use a new run.output_dir"
            )
        return None
    if str(raw).lower() == "auto":
        latest = _latest_checkpoint(output_dir)
        if latest is None:
            raise ValueError(f"resume_from_checkpoint=auto but no checkpoint exists in {output_dir}")
        return str(latest)
    path = _resolve_repo_path(str(raw)).resolve()
    if output_dir.resolve() not in path.parents:
        raise ValueError(
            "Only checkpoints from this post-training output may be resumed; "
            "the prior SFT optimizer/scheduler must not be reused."
        )
    return str(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in (
        "torch",
        "torchvision",
        "triton",
        "bitsandbytes",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface-hub",
        "peft",
        "pyyaml",
        "tokenizers",
        "trl",
        "unsloth",
        "unsloth-zoo",
        "wandb",
        "xformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _require_runtime_versions(versions: Mapping[str, str]) -> None:
    mismatches = {
        package: {"observed": versions.get(package, "missing"), "required": required}
        for package, required in REQUIRED_RUNTIME_VERSIONS.items()
        if versions.get(package) != required
    }
    if mismatches:
        raise RuntimeError(
            "GPU runtime does not match the source-verified package pins: "
            + json.dumps(mismatches, sort_keys=True)
        )
    torch_version = str(versions.get("torch", "missing")).split("+", 1)[0]
    try:
        torch_parts = tuple(int(part) for part in torch_version.split(".")[:2])
    except ValueError as exc:
        raise RuntimeError(f"cannot parse torch version={torch_version!r}") from exc
    if not ((2, 4) <= torch_parts < (2, 11)):
        raise RuntimeError(
            f"torch={torch_version} is outside Unsloth 2026.7.2's supported range [2.4, 2.11)"
        )


def _runtime_hardware_contract() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to construct a smoke/full runtime contract")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(properties.total_memory),
            }
        )
    return {
        "torch_cuda_version": str(torch.version.cuda),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "world_size": _world_size(),
        "devices": devices,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_contract(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(model_cfg["name_or_path"])).expanduser().resolve()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name != ".DS_Store"
    )
    if not files:
        raise ValueError(f"SFT final model directory contains no files: {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        size = file_path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "path": str(path),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "directory_sha256": digest.hexdigest(),
    }


def _seed_python_and_torch(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def entrypoint_command(filename: str) -> str:
    """Return the correct source-relative command for research or release layout."""

    try:
        relative = (SOURCE_ROOT / filename).relative_to(REPO_ROOT)
    except ValueError:
        relative = SOURCE_ROOT / filename
    return f"python3 {relative.as_posix()}"
