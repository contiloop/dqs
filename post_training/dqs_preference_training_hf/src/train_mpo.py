#!/usr/bin/env python3
"""Standalone post-training entry point for DQS paper setting 5 (SFT + mPO)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

try:
    from .mpo_data import load_preference_datasets
    from .mpo_masking import MPOPreferenceCollator
    from .mpo_model import (
        configure_distributed_device,
        load_model_and_tokenizer,
        require_final_stage_model,
        text_tokenizer,
        validate_model_data_contract,
    )
    from .mpo_objective import Setting5LossConfig
    from .mpo_wandb import (
        WANDB_REQUIRED_VERSION,
        MPOWandbConfig,
        build_wandb_callback,
        initialize_wandb_run,
    )
    from .preference_runtime import (
        POST_TRAINING_ROOT,
        REPO_ROOT,
        SOURCE_ROOT,
        _training_arguments,
        entrypoint_command,
    )
except ImportError:  # Direct execution from a flat source directory.
    from mpo_data import load_preference_datasets
    from mpo_masking import MPOPreferenceCollator
    from mpo_model import (
        configure_distributed_device,
        load_model_and_tokenizer,
        require_final_stage_model,
        text_tokenizer,
        validate_model_data_contract,
    )
    from mpo_objective import Setting5LossConfig
    from mpo_wandb import (
        WANDB_REQUIRED_VERSION,
        MPOWandbConfig,
        build_wandb_callback,
        initialize_wandb_run,
    )
    from preference_runtime import (
        POST_TRAINING_ROOT,
        REPO_ROOT,
        SOURCE_ROOT,
        _training_arguments,
        entrypoint_command,
    )

REQUIRED_RUNTIME_VERSIONS = {
    "accelerate": "1.14.0",
    "datasets": "4.3.0",
    "huggingface-hub": "1.21.0",
    "pyyaml": "6.0.3",
    "tokenizers": "0.22.2",
    "trl": "0.24.0",
    "transformers": "5.5.0",
    "unsloth": "2026.7.2",
    "unsloth-zoo": "2026.7.2",
    "wandb": WANDB_REQUIRED_VERSION,
}

SMOKE_SOURCE_FILES = (
    "train_mpo.py",
    "preference_runtime.py",
    "mpo_model.py",
    "mpo_data.py",
    "mpo_masking.py",
    "mpo_objective.py",
    "mpo_trainer.py",
    "mpo_wandb.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=POST_TRAINING_ROOT / "configs" / "mpo_setting5.yaml",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="YAML-parsed dotted override; may be repeated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and every dataset row without loading a model",
    )
    parser.add_argument(
        "--smoke-step",
        action="store_true",
        help="Run exactly one optimizer step without saving a final model",
    )
    return parser.parse_args()


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


def _mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _output_dir(run_cfg: Mapping[str, Any], *, smoke_step: bool) -> Path:
    raw = run_cfg.get("output_dir")
    if not raw:
        raise ValueError("run.output_dir must be configured explicitly")
    path = _resolve_repo_path(str(raw)).resolve()
    if POST_TRAINING_ROOT.resolve() not in path.parents:
        raise ValueError(f"run.output_dir must stay under the post-training project root: {path}")
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


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def _resume_checkpoint(training_cfg: Mapping[str, Any], output_dir: Path, *, smoke_step: bool) -> str | None:
    if smoke_step:
        return None
    raw = training_cfg.get("resume_from_checkpoint")
    if raw in {None, False, "", "none", "null", "false"}:
        if (output_dir / "final").exists() or (output_dir / "run_manifest.json").exists():
            raise ValueError(
                f"a completed post-training run already exists under {output_dir}; use a new run.output_dir"
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


def _source_contract() -> dict[str, str]:
    return {
        filename: _file_sha256(SOURCE_ROOT / filename)
        for filename in SMOKE_SOURCE_FILES
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


def _smoke_contract(
    *,
    run_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    wandb_config: MPOWandbConfig,
    loss_cfg: Setting5LossConfig,
    dataset_contract: Mapping[str, Any],
    model_artifact: Mapping[str, Any] | None,
    hardware_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "model": {
            "name_or_path": str(model_cfg.get("name_or_path", "")),
            "revision": model_cfg.get("revision"),
            "subfolder": model_cfg.get("subfolder"),
            "unsloth_model_api": str(model_cfg.get("unsloth_model_api", "fast_model")),
            "expected_sft_run_id": str(model_cfg.get("expected_sft_run_id", "")),
            "expected_sft_subset_idx": int(model_cfg.get("expected_sft_subset_idx", -1)),
            "expected_sft_global_step": int(model_cfg.get("expected_sft_global_step", -1)),
            "artifact": dict(model_artifact) if model_artifact is not None else "not_checked_in_dry_run",
        },
        "dataset": dict(dataset_contract),
        "loss": loss_cfg.to_dict(),
        "run": {
            "id": str(run_cfg.get("id", "")),
            "seed": int(run_cfg.get("seed", 0)),
        },
        "training": dict(training_cfg),
        "logging": {"wandb": wandb_config.to_dict()},
        "source_sha256": _source_contract(),
        "runtime_versions": _runtime_versions(),
        "runtime_hardware": (
            dict(hardware_contract) if hardware_contract is not None else "not_checked_in_dry_run"
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def _require_smoke_receipt(
    *,
    run_cfg: Mapping[str, Any],
    base_output_dir: Path,
    expected_contract: Mapping[str, Any],
) -> None:
    if not bool(run_cfg.get("require_smoke_step_receipt", True)):
        raise ValueError(
            "run.require_smoke_step_receipt must remain true; full training has no bypass for the hard gate"
        )
    receipt_path = base_output_dir / "smoke_step" / "smoke_step_result.json"
    if not receipt_path.exists():
        raise ValueError(
            "Full training is gated on a successful one-step Unsloth run. Run first:\n"
            f"  {entrypoint_command('train_mpo.py')} --smoke-step\n"
            f"Expected receipt: {receipt_path}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    observed = receipt.get("smoke_contract", {})
    if observed.get("sha256") != expected_contract.get("sha256"):
        raise ValueError(
            "smoke-step receipt does not match the current model/data/loss/runtime contract; "
            "rerun --smoke-step in this exact environment"
        )
    if receipt.get("resolved_logits_projection") != "selected":
        raise ValueError("smoke-step receipt did not prove selected logits projection")


def _validate_hard_config(
    *,
    run_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    loss_values: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    logging_cfg: Mapping[str, Any],
) -> None:
    required = {
        "run": ("id", "seed", "output_dir", "require_smoke_step_receipt"),
        "model": (
            "name_or_path",
            "revision",
            "subfolder",
            "trust_remote_code",
            "require_final_stage_model",
            "unsloth_model_api",
            "expected_sft_run_id",
            "expected_sft_subset_idx",
            "expected_sft_global_step",
        ),
        "data": (
            "source",
            "path",
            "eval_path",
            "contract_path",
            "cache_dir",
            "hf_repo_id",
            "hf_revision",
            "hf_config_name",
            "hf_train_filename",
            "hf_eval_filename",
            "hf_train_split",
            "hf_eval_split",
            "hf_contract_filename",
        ),
        "loss": (
            "paper_setting",
            "lambda_sft",
            "lambda_mpo",
            "preference_beta",
            "smooth_l1_delta",
        ),
        "training": (
            "backend",
            "dtype",
            "load_in_4bit",
            "load_in_8bit",
            "freeze_embeddings",
            "gradient_checkpointing",
            "unsloth_fullgraph",
            "max_seq_length",
            "logits_projection",
            "token_logp_backend",
            "learning_rate",
            "warmup_ratio",
            "scheduler",
            "optimizer",
            "max_grad_norm",
            "weight_decay",
            "num_train_epochs",
            "max_steps",
            "effective_batch_size",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "logging_steps",
            "eval_strategy",
            "eval_steps",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "ddp_find_unused_parameters",
            "report_to",
            "resume_from_checkpoint",
        ),
    }
    sections = {
        "run": run_cfg,
        "model": model_cfg,
        "data": data_cfg,
        "loss": loss_values,
        "training": training_cfg,
    }
    for section_name, keys in required.items():
        missing = [key for key in keys if key not in sections[section_name]]
        if missing:
            raise ValueError(f"config section {section_name!r} is missing explicit keys: {missing}")
    if not str(run_cfg.get("id", "")).strip():
        raise ValueError("run.id must be configured explicitly")
    if "seed" not in run_cfg:
        raise ValueError("run.seed must be configured explicitly")
    if not bool(run_cfg.get("require_smoke_step_receipt", False)):
        raise ValueError("run.require_smoke_step_receipt must remain true; there is no bypass")
    if not bool(model_cfg.get("require_final_stage_model", False)):
        raise ValueError("model.require_final_stage_model must remain true")
    if str(model_cfg.get("unsloth_model_api", "")).strip().lower() != "fast_model":
        raise ValueError("model.unsloth_model_api must be fast_model")
    if str(data_cfg.get("source", "")).strip().lower() != "local":
        raise ValueError(
            "data.source must be local; run `make download-data` before training"
        )
    if str(training_cfg.get("backend", "")).strip().lower() != "unsloth":
        raise ValueError("This entry point requires training.backend=unsloth; no backend fallback is allowed")
    if str(training_cfg.get("dtype", "")).strip().lower() not in {"bf16", "bfloat16"}:
        raise ValueError("training.dtype must be bfloat16")
    if bool(training_cfg.get("load_in_4bit", False)) or bool(training_cfg.get("load_in_8bit", False)):
        raise ValueError("mPO full post-training forbids 4-bit and 8-bit model loading")
    if type(training_cfg.get("freeze_embeddings")) is not bool:
        raise ValueError("training.freeze_embeddings must be an explicit boolean")
    if str(training_cfg.get("gradient_checkpointing", "")).strip().lower() != "unsloth":
        raise ValueError("training.gradient_checkpointing must be unsloth")
    if bool(training_cfg.get("unsloth_fullgraph", True)):
        raise ValueError("training.unsloth_fullgraph must be false for variable selected-position tensors")
    if str(training_cfg.get("logits_projection", "")).strip().lower() != "selected":
        raise ValueError(
            "This entry point requires training.logits_projection=selected; "
            "no full-logits fallback is allowed"
        )
    if str(training_cfg.get("token_logp_backend", "")).strip().lower() != "unsloth_fused":
        raise ValueError(
            "This entry point requires training.token_logp_backend=unsloth_fused; "
            "no PyTorch CE fallback is allowed"
        )
    if training_cfg.get("report_to") != []:
        raise ValueError(
            "training.report_to must remain exactly []; the strict post-training W&B callback "
            "is the only permitted reporting path"
        )
    MPOWandbConfig.from_logging_config(
        logging_cfg,
        post_training_run_id=str(run_cfg.get("id", "")),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    run_cfg = _mapping(cfg, "run")
    model_cfg = _mapping(cfg, "model")
    data_cfg = _mapping(cfg, "data")
    loss_values = _mapping(cfg, "loss")
    training_cfg = _mapping(cfg, "training")
    logging_cfg = _mapping(cfg, "logging")
    _validate_hard_config(
        run_cfg=run_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        loss_values=loss_values,
        training_cfg=training_cfg,
        logging_cfg=logging_cfg,
    )
    wandb_config = MPOWandbConfig.from_logging_config(
        logging_cfg,
        post_training_run_id=str(run_cfg["id"]),
    )
    loss_cfg = Setting5LossConfig.from_mapping(loss_values)
    output_dir = _output_dir(run_cfg, smoke_step=args.smoke_step)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_preference_datasets(data_cfg, repo_root=REPO_ROOT)
    configured_max_length = int(training_cfg.get("max_seq_length", 0))
    contracted_max_length = int(bundle.contract["max_seq_length"])
    if configured_max_length != contracted_max_length:
        raise ValueError(
            "training.max_seq_length must exactly match the tokenized dataset contract: "
            f"configured={configured_max_length}, contract={contracted_max_length}"
        )

    model_artifact: Mapping[str, Any] | None = None
    hardware_contract: Mapping[str, Any] | None = None
    if not args.dry_run:
        versions = _runtime_versions()
        _require_runtime_versions(versions)
        configure_distributed_device()
        require_final_stage_model(model_cfg)
        model_artifact = _model_artifact_contract(model_cfg)
        hardware_contract = _runtime_hardware_contract()

    smoke_contract = _smoke_contract(
        run_cfg=run_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        wandb_config=wandb_config,
        loss_cfg=loss_cfg,
        dataset_contract=bundle.contract,
        model_artifact=model_artifact,
        hardware_contract=hardware_contract,
    )
    preflight = {
        "run_id": str(run_cfg["id"]),
        "mode": "dry_run" if args.dry_run else ("smoke_step" if args.smoke_step else "train"),
        "paper_setting": 5,
        "objective": loss_cfg.to_dict(),
        "dataset": bundle.summary,
        "fresh_post_training_optimizer": training_cfg.get("resume_from_checkpoint") in {
            None,
            False,
            "",
            "none",
            "null",
            "false",
        },
        "output_dir": str(output_dir),
        "smoke_contract": smoke_contract,
    }
    if _global_rank() == 0:
        _write_json(output_dir / "preflight.json", preflight)
    if args.dry_run:
        if _global_rank() == 0:
            print(json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2))
        return

    if not args.smoke_step:
        _require_smoke_receipt(
            run_cfg=run_cfg,
            base_output_dir=_output_dir(run_cfg, smoke_step=False),
            expected_contract=smoke_contract,
        )

    resume = _resume_checkpoint(training_cfg, output_dir, smoke_step=args.smoke_step)
    _seed_python_and_torch(int(run_cfg["seed"]))
    wandb_session = initialize_wandb_run(
        wandb_config,
        output_dir=output_dir,
        # On checkpoint resume, retain the config already stored in the W&B
        # run instead of trying to replace it with resume-only local state.
        metadata=None if resume is not None else {"post_training": preflight},
        # Smoke validates the exact W&B package and source hash above, but only
        # the full-run rank zero is allowed to open the stable online run.
        active_process=(not args.smoke_step and _global_rank() == 0),
    )
    trainer: Any | None = None
    try:
        model, tokenizer_or_processor, model_summary = load_model_and_tokenizer(
            model_cfg,
            training_cfg,
        )
        # Unsloth has now imported and patched Transformers. Importing Trainer
        # or TrainerCallback before load_model_and_tokenizer is a hard error.
        try:
            from .mpo_trainer import MPOTrainer
        except ImportError:
            from mpo_trainer import MPOTrainer
        validate_model_data_contract(
            tokenizer_or_processor,
            dataset_summary=bundle.summary,
            contract=bundle.contract,
        )
        tokenizer = text_tokenizer(tokenizer_or_processor)
        training_args = _training_arguments(
            run_cfg=run_cfg,
            training_cfg=training_cfg,
            output_dir=output_dir,
            has_eval=bundle.eval_dataset is not None,
            smoke_step=args.smoke_step,
        )
        trainer = MPOTrainer(
            model=model,
            args=training_args,
            train_dataset=bundle.train_dataset,
            eval_dataset=bundle.eval_dataset,
            data_collator=MPOPreferenceCollator(pad_token_id=int(tokenizer.pad_token_id)),
            loss_config=loss_cfg,
            logits_projection="selected",
            token_logp_backend="unsloth_fused",
            callbacks=[build_wandb_callback(wandb_session)],
        )
        train_result = trainer.train(resume_from_checkpoint=resume)
        manifest = {
            **preflight,
            "model": model_summary,
            "resolved_logits_projection": trainer._resolved_projection,
            "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
            "world_size": _world_size(),
            "train_metrics": dict(train_result.metrics),
            "resumed_post_training_checkpoint": resume,
            "global_step": int(trainer.state.global_step),
        }
        if args.smoke_step:
            if int(trainer.state.global_step) != 1:
                raise RuntimeError(
                    f"smoke run did not complete exactly one optimizer step: {trainer.state.global_step}"
                )
            train_loss = float(train_result.metrics.get("train_loss", float("nan")))
            if not math.isfinite(train_loss):
                raise RuntimeError(f"smoke run produced non-finite train_loss={train_loss}")
            if trainer.is_world_process_zero():
                _write_json(output_dir / "smoke_step_result.json", manifest)
                print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
            return

        trainer.save_state()
        final_dir = output_dir / "final"
        trainer.save_model(str(final_dir))
        if trainer.is_world_process_zero():
            tokenizer_or_processor.save_pretrained(final_dir)
            _write_json(
                final_dir / "dqs_mpo_model.json",
                {
                    "run_id": str(run_cfg["id"]),
                    "paper_setting": 5,
                    "global_step": int(trainer.state.global_step),
                    "objective": loss_cfg.to_dict(),
                    "source_model_artifact": dict(model_artifact or {}),
                    "dataset_contract": dict(bundle.contract),
                    "smoke_contract_sha256": str(smoke_contract["sha256"]),
                },
            )
            manifest["final_model_dir"] = str(final_dir)
            _write_json(output_dir / "run_manifest.json", manifest)
            wandb_session.log_final(
                global_step=int(trainer.state.global_step),
                final_saved=True,
                train_metrics=train_result.metrics,
            )
    except BaseException as training_error:
        if wandb_session.active:
            global_step = int(getattr(getattr(trainer, "state", None), "global_step", 0))
            try:
                try:
                    wandb_session.log_failure(global_step=global_step)
                finally:
                    wandb_session.finish(exit_code=1)
            except BaseException as wandb_error:
                raise wandb_error from training_error
        raise
    else:
        wandb_session.finish(exit_code=0)


if __name__ == "__main__":
    main()
