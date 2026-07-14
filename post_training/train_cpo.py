#!/usr/bin/env python3
"""Strict full-response CPO post-training on Teacher-vs-Student pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .cpo_objective import FullResponseCPOConfig
    from .mpo_data import load_preference_datasets
    from .mpo_masking import MPOPreferenceCollator
    from .mpo_model import (
        configure_distributed_device,
        load_model_and_tokenizer,
        require_final_stage_model,
        text_tokenizer,
        validate_model_data_contract,
    )
    from .mpo_wandb import (
        MPOWandbConfig,
        build_wandb_callback,
        initialize_wandb_run,
    )
    from .preference_runtime import (
        POST_TRAINING_ROOT,
        REPO_ROOT,
        SOURCE_ROOT,
        _file_sha256,
        _global_rank,
        _model_artifact_contract,
        _output_dir,
        _require_runtime_versions,
        _resume_checkpoint,
        _runtime_hardware_contract,
        _runtime_versions,
        _seed_python_and_torch,
        _training_arguments,
        _world_size,
        _write_json,
        entrypoint_command,
        load_config,
    )
except ImportError:
    from cpo_objective import FullResponseCPOConfig
    from mpo_data import load_preference_datasets
    from mpo_masking import MPOPreferenceCollator
    from mpo_model import (
        configure_distributed_device,
        load_model_and_tokenizer,
        require_final_stage_model,
        text_tokenizer,
        validate_model_data_contract,
    )
    from mpo_wandb import MPOWandbConfig, build_wandb_callback, initialize_wandb_run
    from preference_runtime import (
        POST_TRAINING_ROOT,
        REPO_ROOT,
        SOURCE_ROOT,
        _file_sha256,
        _global_rank,
        _model_artifact_contract,
        _output_dir,
        _require_runtime_versions,
        _resume_checkpoint,
        _runtime_hardware_contract,
        _runtime_versions,
        _seed_python_and_torch,
        _training_arguments,
        _world_size,
        _write_json,
        entrypoint_command,
        load_config,
    )


SOURCE_FILES = (
    "train_cpo.py",
    "preference_runtime.py",
    "cpo_objective.py",
    "cpo_trainer.py",
    "mpo_data.py",
    "mpo_masking.py",
    "mpo_model.py",
    "mpo_trainer.py",
    "mpo_wandb.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=POST_TRAINING_ROOT / "configs" / "cpo_full_response.yaml",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-step", action="store_true")
    return parser.parse_args()


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    run = _required_mapping(config, "run")
    model = _required_mapping(config, "model")
    data = _required_mapping(config, "data")
    loss = _required_mapping(config, "loss")
    training = _required_mapping(config, "training")
    logging = _required_mapping(config, "logging")
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
            "hf_train_filename",
            "hf_eval_filename",
            "hf_train_split",
            "hf_eval_split",
            "hf_contract_filename",
        ),
        "loss": ("objective", "beta", "cpo_alpha", "loss_type", "label_smoothing"),
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
            "effective_batch_size",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "report_to",
            "resume_from_checkpoint",
        ),
    }
    sections = {"run": run, "model": model, "data": data, "loss": loss, "training": training}
    for section, keys in required.items():
        missing = [key for key in keys if key not in sections[section]]
        if missing:
            raise ValueError(f"{section} is missing explicit keys: {missing}")
    if str(loss["objective"]).lower() != "cpo_full_response":
        raise ValueError("loss.objective must be cpo_full_response")
    FullResponseCPOConfig.from_mapping(loss)
    if str(training["backend"]).lower() != "unsloth":
        raise ValueError("CPO backend must be unsloth")
    if str(training["dtype"]).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("CPO dtype must be bfloat16")
    if bool(training["load_in_4bit"]) or bool(training["load_in_8bit"]):
        raise ValueError("CPO is full bfloat16 training, not quantized training")
    if type(training["freeze_embeddings"]) is not bool:
        raise ValueError("training.freeze_embeddings must be an explicit boolean")
    if training["logits_projection"] != "selected":
        raise ValueError("CPO selected-logit projection is mandatory")
    if training["token_logp_backend"] != "unsloth_fused":
        raise ValueError("CPO fused token log-prob backend is mandatory")
    if list(training["report_to"]) != []:
        raise ValueError("training.report_to must stay empty; strict W&B owns logging")
    if not bool(run["require_smoke_step_receipt"]):
        raise ValueError("the CPO full run cannot bypass its smoke-step receipt")
    MPOWandbConfig.from_logging_config(logging, post_training_run_id=str(run["id"]))
    if str(data["source"]).lower() != "local":
        raise ValueError("CPO data.source must be local; run `make download-data` first")
    if not bool(model["require_final_stage_model"]):
        raise ValueError("CPO must initialize from the verified SFT final model")


def source_contract() -> dict[str, str]:
    return {name: _file_sha256(SOURCE_ROOT / name) for name in SOURCE_FILES}


def smoke_contract(
    *,
    config: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    model_artifact: Mapping[str, Any] | None,
    hardware: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "run": dict(_required_mapping(config, "run")),
        "model": dict(_required_mapping(config, "model")),
        "data_contract": dict(dataset_contract),
        "loss": FullResponseCPOConfig.from_mapping(
            _required_mapping(config, "loss")
        ).to_dict(),
        "training": dict(_required_mapping(config, "training")),
        "logging": dict(_required_mapping(config, "logging")),
        "model_artifact": dict(model_artifact) if model_artifact else "not_checked",
        "hardware": dict(hardware) if hardware else "not_checked",
        "runtime_versions": _runtime_versions(),
        "source_sha256": source_contract(),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def require_smoke_receipt(output_dir: Path, expected: Mapping[str, Any]) -> None:
    path = output_dir / "smoke_step" / "smoke_step_result.json"
    if not path.is_file():
        raise ValueError(
            "full CPO is gated on a successful one-step run; execute "
            f"{entrypoint_command('train_cpo.py')} --smoke-step first"
        )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("smoke_contract", {}).get("sha256") != expected.get("sha256"):
        raise ValueError("CPO smoke receipt does not match the current exact contract")
    if receipt.get("resolved_logits_projection") != "selected":
        raise ValueError("CPO smoke receipt did not prove selected-logit execution")


def main() -> None:
    args = parse_args()
    if args.dry_run and args.smoke_step:
        raise ValueError("--dry-run and --smoke-step are mutually exclusive")
    config = load_config(args.config, args.overrides)
    validate_config(config)
    run_cfg = _required_mapping(config, "run")
    model_cfg = _required_mapping(config, "model")
    data_cfg = _required_mapping(config, "data")
    loss_cfg = FullResponseCPOConfig.from_mapping(_required_mapping(config, "loss"))
    training_cfg = _required_mapping(config, "training")
    logging_cfg = _required_mapping(config, "logging")
    wandb_cfg = MPOWandbConfig.from_logging_config(
        logging_cfg, post_training_run_id=str(run_cfg["id"])
    )
    bundle = load_preference_datasets(data_cfg, repo_root=REPO_ROOT)
    if bundle.contract.get("objective_family") != "CPO_full_response":
        raise ValueError("configured dataset contract is not the full-response CPO artifact")
    if bundle.contract.get("full_response_negative_policy") != (
        "original_full_student_response_no_synthetic_reversion"
    ):
        raise ValueError("CPO dataset rejected side is not the original Student response")

    base_output_dir = _output_dir(run_cfg, smoke_step=False)
    output_dir = _output_dir(run_cfg, smoke_step=args.smoke_step)

    model_artifact = None
    hardware = None
    if not args.dry_run:
        versions = _runtime_versions()
        _require_runtime_versions(versions)
        configure_distributed_device()
        require_final_stage_model(model_cfg)
        model_artifact = _model_artifact_contract(model_cfg)
        hardware = _runtime_hardware_contract()
    exact_contract = smoke_contract(
        config=config,
        dataset_contract=bundle.contract,
        model_artifact=model_artifact,
        hardware=hardware,
    )
    preflight = {
        "run_id": str(run_cfg["id"]),
        "mode": "dry_run" if args.dry_run else ("smoke_step" if args.smoke_step else "train"),
        "objective": loss_cfg.to_dict(),
        "dataset": bundle.summary,
        "output_dir": str(output_dir),
        "smoke_contract": exact_contract,
    }
    if _global_rank() == 0:
        _write_json(output_dir / "preflight.json", preflight)
    if args.dry_run:
        if _global_rank() == 0:
            print(json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2))
        return
    if not args.smoke_step:
        require_smoke_receipt(base_output_dir, exact_contract)

    resume = _resume_checkpoint(training_cfg, output_dir, smoke_step=args.smoke_step)
    _seed_python_and_torch(int(run_cfg["seed"]))
    wandb_session = initialize_wandb_run(
        wandb_cfg,
        output_dir=output_dir,
        metadata=None if resume is not None else {"post_training": preflight},
        active_process=(not args.smoke_step and _global_rank() == 0),
    )
    trainer: Any | None = None
    try:
        model, tokenizer_or_processor, model_summary = load_model_and_tokenizer(
            model_cfg, training_cfg
        )
        validate_model_data_contract(
            tokenizer_or_processor,
            dataset_summary=bundle.summary,
            contract=bundle.contract,
        )
        tokenizer = text_tokenizer(tokenizer_or_processor)
        # Import only after Unsloth has patched Transformers.
        try:
            from .cpo_trainer import FullResponseCPOTrainer
        except ImportError:
            from cpo_trainer import FullResponseCPOTrainer

        training_args = _training_arguments(
            run_cfg=run_cfg,
            training_cfg=training_cfg,
            output_dir=output_dir,
            has_eval=False,
            smoke_step=args.smoke_step,
        )
        trainer = FullResponseCPOTrainer(
            model=model,
            args=training_args,
            train_dataset=bundle.train_dataset,
            eval_dataset=None,
            data_collator=MPOPreferenceCollator(
                pad_token_id=int(tokenizer.pad_token_id)
            ),
            loss_config=loss_cfg,
            logits_projection="selected",
            token_logp_backend="unsloth_fused",
            callbacks=[build_wandb_callback(wandb_session)],
        )
        result = trainer.train(resume_from_checkpoint=resume)
        manifest = {
            **preflight,
            "model": model_summary,
            "resolved_logits_projection": "selected",
            "gradient_accumulation_steps": int(
                training_args.gradient_accumulation_steps
            ),
            "world_size": _world_size(),
            "train_metrics": dict(result.metrics),
            "resumed_post_training_checkpoint": resume,
            "global_step": int(trainer.state.global_step),
        }
        if args.smoke_step:
            if int(trainer.state.global_step) != 1:
                raise RuntimeError("CPO smoke did not complete exactly one optimizer step")
            train_loss = float(result.metrics.get("train_loss", float("nan")))
            if not math.isfinite(train_loss):
                raise RuntimeError(f"CPO smoke produced non-finite loss={train_loss}")
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
                final_dir / "dqs_cpo_model.json",
                {
                    "run_id": str(run_cfg["id"]),
                    "objective": loss_cfg.to_dict(),
                    "global_step": int(trainer.state.global_step),
                    "source_model_artifact": dict(model_artifact or {}),
                    "dataset_contract": dict(bundle.contract),
                    "smoke_contract_sha256": exact_contract["sha256"],
                },
            )
            manifest["final_model_dir"] = str(final_dir)
            _write_json(output_dir / "run_manifest.json", manifest)
            wandb_session.log_final(
                global_step=int(trainer.state.global_step),
                final_saved=True,
                train_metrics=result.metrics,
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
