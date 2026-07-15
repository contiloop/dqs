#!/usr/bin/env python3
"""Strict Unsloth DPO post-training on full Teacher-vs-Student responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .dpo_trainer import (
        DPOForwardContractMonitor,
        precompute_reference_logps_before_policy_restore,
        require_unsloth_dpo_patch,
        validate_dpo_trainer_contract,
        validate_prepared_dpo_dataset,
    )
    from .full_preference_data import (
        load_full_preference_dataset,
        validate_runtime_tokenization,
    )
    from .mpo_model import (
        configure_distributed_device,
        load_model_and_tokenizer,
        require_final_stage_model,
        text_tokenizer,
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
        _gradient_accumulation_steps,
        _model_artifact_contract,
        _output_dir,
        _require_runtime_versions,
        _resume_checkpoint,
        _runtime_hardware_contract,
        _runtime_versions,
        _seed_python_and_torch,
        _world_size,
        _write_json,
        entrypoint_command,
        load_config,
    )
except ImportError:
    from dpo_trainer import (
        DPOForwardContractMonitor,
        precompute_reference_logps_before_policy_restore,
        require_unsloth_dpo_patch,
        validate_dpo_trainer_contract,
        validate_prepared_dpo_dataset,
    )
    from full_preference_data import (
        load_full_preference_dataset,
        validate_runtime_tokenization,
    )
    from mpo_model import (
        configure_distributed_device,
        load_model_and_tokenizer,
        require_final_stage_model,
        text_tokenizer,
    )
    from mpo_wandb import MPOWandbConfig, build_wandb_callback, initialize_wandb_run
    from preference_runtime import (
        POST_TRAINING_ROOT,
        REPO_ROOT,
        SOURCE_ROOT,
        _file_sha256,
        _global_rank,
        _gradient_accumulation_steps,
        _model_artifact_contract,
        _output_dir,
        _require_runtime_versions,
        _resume_checkpoint,
        _runtime_hardware_contract,
        _runtime_versions,
        _seed_python_and_torch,
        _world_size,
        _write_json,
        entrypoint_command,
        load_config,
    )


SOURCE_FILES = (
    "train_dpo.py",
    "dpo_trainer.py",
    "full_preference_data.py",
    "mpo_model.py",
    "mpo_wandb.py",
    "preference_runtime.py",
)
EXPECTED_TRL_VERSION = "0.24.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=POST_TRAINING_ROOT / "configs" / "dpo_full_response.yaml",
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


def _loss_contract(
    loss: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "objective": "-logsigmoid(beta * ((logpi_chosen-logpi_rejected) - (logref_chosen-logref_rejected)))",
        "objective_family": str(loss["objective"]),
        "trainer": str(loss["trainer"]),
        "beta": float(loss["beta"]),
        "loss_type": str(loss["loss_type"]),
        "label_smoothing": float(loss["label_smoothing"]),
        "f_divergence_type": str(loss["f_divergence_type"]),
        "reference_free": bool(loss["reference_free"]),
        "rpo_alpha": loss["rpo_alpha"],
        "ld_alpha": loss["ld_alpha"],
        "loss_weights": loss["loss_weights"],
        "use_weighting": bool(loss["use_weighting"]),
        "use_liger_loss": bool(loss["use_liger_loss"]),
        "sync_ref_model": bool(loss["sync_ref_model"]),
        "logprob_aggregation": "sum_over_completion_including_eos",
        "reference": dict(reference),
    }


def _validate_length_contract(
    *,
    dataset_contract: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    tokenization = dataset_contract.get("tokenization_contract")
    if not isinstance(tokenization, Mapping):
        raise ValueError("DPO dataset contract has no tokenization_contract")
    if str(tokenization.get("trl_version")) != EXPECTED_TRL_VERSION:
        raise ValueError("DPO dataset contract is not pinned to TRL 0.24.0")
    if bool(tokenization.get("truncation_allowed", True)):
        raise ValueError("DPO dataset contract must forbid truncation")
    maxima_raw = tokenization.get("maxima")
    if not isinstance(maxima_raw, Mapping):
        raise ValueError("DPO tokenization contract has no maxima")
    maxima = {
        key: int(value["tokens"])
        for key, value in maxima_raw.items()
        if isinstance(value, Mapping) and "tokens" in value
    }
    required_maxima = {
        "prompt_tokens",
        "dpo_chosen_completion_tokens",
        "dpo_rejected_completion_tokens",
        "dpo_chosen_sequence_tokens",
        "dpo_rejected_sequence_tokens",
    }
    missing = sorted(required_maxima - set(maxima))
    if missing:
        raise ValueError(f"DPO tokenization contract is missing maxima: {missing}")
    max_length = int(training["max_length"])
    max_seq_length = int(training["max_seq_length"])
    max_prompt_length = int(training["max_prompt_length"])
    max_completion_length = int(training["max_completion_length"])
    if max_length != max_seq_length or max_length != int(tokenization["max_seq_length"]):
        raise ValueError(
            "training max_length/max_seq_length must equal the immutable dataset contract"
        )
    if max_prompt_length != maxima["prompt_tokens"]:
        raise ValueError("max_prompt_length must equal the observed no-truncation maximum")
    observed_completion_max = max(
        maxima["dpo_chosen_completion_tokens"],
        maxima["dpo_rejected_completion_tokens"],
    )
    if max_completion_length != observed_completion_max:
        raise ValueError("max_completion_length must equal the observed no-truncation maximum")
    observed_sequence_max = max(
        maxima["dpo_chosen_sequence_tokens"],
        maxima["dpo_rejected_sequence_tokens"],
    )
    if observed_sequence_max > max_length:
        raise ValueError("DPO data exceeds configured max_length")
    return {
        "trl_version": EXPECTED_TRL_VERSION,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "max_observed_sequence_tokens": observed_sequence_max,
        "truncation_allowed": False,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    run = _required_mapping(config, "run")
    logging = _required_mapping(config, "logging")
    model = _required_mapping(config, "model")
    data = _required_mapping(config, "data")
    loss = _required_mapping(config, "loss")
    reference = _required_mapping(config, "reference")
    training = _required_mapping(config, "training")
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
            "contract_path",
            "cache_dir",
            "hf_repo_id",
            "hf_revision",
            "hf_train_filename",
            "hf_train_split",
            "hf_contract_filename",
        ),
        "loss": (
            "objective",
            "trainer",
            "beta",
            "loss_type",
            "label_smoothing",
            "f_divergence_type",
            "reference_free",
            "rpo_alpha",
            "ld_alpha",
            "loss_weights",
            "use_weighting",
            "use_liger_loss",
            "sync_ref_model",
        ),
        "reference": (
            "mode",
            "precompute_ref_log_probs",
            "precompute_ref_batch_size",
            "require_ref_model_none",
            "precompute_before_resume_restore",
        ),
        "training": (
            "backend",
            "dtype",
            "load_in_4bit",
            "load_in_8bit",
            "freeze_embeddings",
            "gradient_checkpointing",
            "unsloth_compile",
            "unsloth_fullgraph",
            "max_seq_length",
            "max_length",
            "max_prompt_length",
            "max_completion_length",
            "truncation_mode",
            "padding_free",
            "use_logits_to_keep",
            "dataset_num_proc",
            "disable_dropout",
            "learning_rate",
            "warmup_steps",
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
            "smoke_sample_rows",
            "report_to",
            "resume_from_checkpoint",
        ),
    }
    sections = {
        "run": run,
        "model": model,
        "data": data,
        "loss": loss,
        "reference": reference,
        "training": training,
    }
    for section, keys in required.items():
        missing = [key for key in keys if key not in sections[section]]
        if missing:
            raise ValueError(f"{section} is missing explicit keys: {missing}")

    if not bool(run["require_smoke_step_receipt"]):
        raise ValueError("the DPO full run cannot bypass its smoke-step receipt")
    if not bool(model["require_final_stage_model"]):
        raise ValueError("DPO must initialize from the verified SFT final model")
    if str(model["unsloth_model_api"]).lower() != "fast_model":
        raise ValueError("DPO requires model.unsloth_model_api=fast_model")
    if str(data["source"]).lower() != "local":
        raise ValueError("DPO data.source must be local; run `make download-data` first")
    if str(loss["objective"]).lower() != "dpo_full_response":
        raise ValueError("loss.objective must be dpo_full_response")
    if str(loss["trainer"]) != "UnslothDPOTrainer":
        raise ValueError("loss.trainer must be UnslothDPOTrainer")
    if float(loss["beta"]) <= 0:
        raise ValueError("DPO beta must be positive")
    if str(loss["loss_type"]) != "sigmoid":
        raise ValueError("strict DPO implements sigmoid loss only")
    if float(loss["label_smoothing"]) != 0.0:
        raise ValueError("strict DPO requires label_smoothing=0")
    if str(loss["f_divergence_type"]) != "reverse_kl":
        raise ValueError("strict DPO requires reverse_kl")
    if bool(loss["reference_free"]):
        raise ValueError("reference-free DPO is forbidden")
    if loss["rpo_alpha"] is not None:
        raise ValueError("RPO/SFT mixing is forbidden in the pure DPO baseline")
    if loss["ld_alpha"] is not None:
        raise ValueError("LD-DPO is forbidden in the pure DPO baseline")
    if loss["loss_weights"] is not None:
        raise ValueError("multi-loss weighting is forbidden in the pure DPO baseline")
    for key in ("use_weighting", "use_liger_loss", "sync_ref_model"):
        if bool(loss[key]):
            raise ValueError(f"loss.{key} must remain false")

    exact_reference = {
        "mode": "initial_sft_policy_precomputed",
        "precompute_ref_log_probs": True,
        "precompute_ref_batch_size": 1,
        "require_ref_model_none": True,
        "precompute_before_resume_restore": True,
    }
    if dict(reference) != exact_reference:
        raise ValueError(
            "reference policy contract must remain exact: "
            f"observed={dict(reference)!r} expected={exact_reference!r}"
        )
    if str(training["backend"]).lower() != "unsloth":
        raise ValueError("DPO backend must be unsloth")
    if str(training["dtype"]).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("DPO dtype must be bfloat16")
    if bool(training["load_in_4bit"]) or bool(training["load_in_8bit"]):
        raise ValueError("DPO is full bfloat16 training, not quantized training")
    if type(training["freeze_embeddings"]) is not bool:
        raise ValueError("training.freeze_embeddings must be an explicit boolean")
    if str(training["gradient_checkpointing"]).lower() != "unsloth":
        raise ValueError("DPO gradient_checkpointing must be unsloth")
    if str(training["unsloth_compile"]).lower() != "disabled":
        raise ValueError("DPO unsloth_compile must be disabled")
    if bool(training["unsloth_fullgraph"]):
        raise ValueError("DPO unsloth_fullgraph must remain false")
    if str(training["truncation_mode"]) != "keep_end":
        raise ValueError("DPO truncation_mode must be keep_end")
    if bool(training["padding_free"]):
        raise ValueError("padding-free DPO is outside the pinned contract")
    if not bool(training["use_logits_to_keep"]):
        raise ValueError("DPO requires use_logits_to_keep=true")
    if not bool(training["disable_dropout"]):
        raise ValueError("DPO requires disable_dropout=true")
    if str(training["eval_strategy"]).lower() != "no":
        raise ValueError("DPO baseline has no eval split; training.eval_strategy must be no")
    if int(training["dataset_num_proc"]) <= 0:
        raise ValueError("training.dataset_num_proc must be positive")
    if int(training["smoke_sample_rows"]) != int(training["effective_batch_size"]):
        raise ValueError("smoke_sample_rows must equal one effective optimization batch")
    if list(training["report_to"]) != []:
        raise ValueError("training.report_to must stay empty; strict W&B owns logging")
    MPOWandbConfig.from_logging_config(logging, post_training_run_id=str(run["id"]))


def source_contract() -> dict[str, str]:
    return {name: _file_sha256(SOURCE_ROOT / name) for name in SOURCE_FILES}


def smoke_contract(
    *,
    config: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    model_artifact: Mapping[str, Any] | None,
    hardware: Mapping[str, Any] | None,
) -> dict[str, Any]:
    loss = _required_mapping(config, "loss")
    reference = _required_mapping(config, "reference")
    payload = {
        "run": dict(_required_mapping(config, "run")),
        "model": dict(_required_mapping(config, "model")),
        "dataset_contract": dict(dataset_contract),
        "loss": _loss_contract(loss, reference),
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
            "full DPO is gated on a successful one-step run; execute "
            f"{entrypoint_command('train_dpo.py')} --smoke-step first"
        )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("smoke_contract", {}).get("sha256") != expected.get("sha256"):
        raise ValueError("DPO smoke receipt does not match the current exact contract")
    if receipt.get("resolved_trainer_class") != "UnslothDPOTrainer":
        raise ValueError("DPO smoke receipt did not prove the patched Unsloth trainer")
    forward = receipt.get("forward_contract", {})
    if not forward.get("selected_suffix_logits_enforced") or forward.get(
        "full_logits_fallback", True
    ):
        raise ValueError("DPO smoke receipt did not prove the logits_to_keep contract")
    reference = receipt.get("reference_precompute", {})
    if not reference.get("precomputed_before_trainer_train"):
        raise ValueError("DPO smoke receipt did not prove fixed-reference precomputation")
    expected_rows = int(expected["payload"]["training"]["smoke_sample_rows"])
    if int(reference.get("rows", -1)) != expected_rows:
        raise ValueError("DPO smoke receipt has the wrong reference-precompute row count")
    if int(receipt.get("prepared_dataset", {}).get("rows", -1)) != expected_rows:
        raise ValueError("DPO smoke receipt has the wrong prepared-dataset row count")


def dpo_config_kwargs(
    *,
    run: Mapping[str, Any],
    loss: Mapping[str, Any],
    reference: Mapping[str, Any],
    training: Mapping[str, Any],
    output_dir: Path,
    pad_token: str,
    smoke_step: bool,
) -> dict[str, Any]:
    ddp_find_unused = (
        bool(training["ddp_find_unused_parameters"]) if _world_size() > 1 else None
    )
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": _gradient_accumulation_steps(training),
        "learning_rate": float(training["learning_rate"]),
        "warmup_steps": float(training["warmup_steps"]),
        "lr_scheduler_type": str(training["scheduler"]),
        "optim": str(training["optimizer"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "weight_decay": float(training["weight_decay"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "max_steps": 1 if smoke_step else int(training["max_steps"]),
        "save_strategy": "no" if smoke_step else str(training["save_strategy"]),
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "logging_steps": 1 if smoke_step else int(training["logging_steps"]),
        "eval_strategy": "no",
        "eval_steps": int(training["eval_steps"]),
        "dataloader_num_workers": int(training["dataloader_num_workers"]),
        "dataloader_pin_memory": bool(training["dataloader_pin_memory"]),
        "remove_unused_columns": True,
        "label_names": [],
        "prediction_loss_only": True,
        "seed": int(run["seed"]),
        "data_seed": int(run["seed"]),
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "report_to": [],
        "run_name": str(run["id"]),
        "max_length": int(training["max_length"]),
        "max_prompt_length": int(training["max_prompt_length"]),
        "max_completion_length": int(training["max_completion_length"]),
        "truncation_mode": str(training["truncation_mode"]),
        "padding_free": bool(training["padding_free"]),
        "pad_token": pad_token,
        "dataset_num_proc": int(training["dataset_num_proc"]),
        "precompute_ref_log_probs": bool(reference["precompute_ref_log_probs"]),
        "precompute_ref_batch_size": int(reference["precompute_ref_batch_size"]),
        "loss_type": [str(loss["loss_type"])],
        "beta": float(loss["beta"]),
        "f_divergence_type": str(loss["f_divergence_type"]),
        "reference_free": bool(loss["reference_free"]),
        "label_smoothing": float(loss["label_smoothing"]),
        "use_weighting": bool(loss["use_weighting"]),
        "rpo_alpha": loss["rpo_alpha"],
        "ld_alpha": loss["ld_alpha"],
        "loss_weights": loss["loss_weights"],
        "use_liger_loss": bool(loss["use_liger_loss"]),
        "sync_ref_model": bool(loss["sync_ref_model"]),
        "disable_dropout": bool(training["disable_dropout"]),
        "use_logits_to_keep": bool(training["use_logits_to_keep"]),
        "generate_during_eval": False,
    }
    if ddp_find_unused is not None:
        kwargs["ddp_find_unused_parameters"] = ddp_find_unused
    return kwargs


def _runtime_dataset(bundle: Any, *, smoke_step: bool, smoke_rows: int) -> Any:
    if not smoke_step:
        return bundle.train_dataset
    if smoke_rows <= 0 or smoke_rows > len(bundle.train_dataset):
        raise ValueError("invalid DPO smoke_sample_rows")
    return bundle.train_dataset.select(range(smoke_rows))


def main() -> None:
    args = parse_args()
    if args.dry_run and args.smoke_step:
        raise ValueError("--dry-run and --smoke-step are mutually exclusive")
    config = load_config(args.config, args.overrides)
    validate_config(config)
    run = _required_mapping(config, "run")
    logging = _required_mapping(config, "logging")
    model_cfg = _required_mapping(config, "model")
    data = _required_mapping(config, "data")
    loss = _required_mapping(config, "loss")
    reference = _required_mapping(config, "reference")
    training = _required_mapping(config, "training")
    wandb_cfg = MPOWandbConfig.from_logging_config(
        logging, post_training_run_id=str(run["id"])
    )
    bundle = load_full_preference_dataset(data, repo_root=REPO_ROOT)
    if "DPO" not in list(bundle.contract.get("compatible_objectives", [])):
        raise ValueError("configured preference dataset is not contracted for DPO")
    if bundle.contract.get("negative_policy") != (
        "original_full_student_response_no_synthetic_reversion"
    ):
        raise ValueError("DPO rejected side is not the original Student response")
    length_contract = _validate_length_contract(
        dataset_contract=bundle.contract, training=training
    )

    base_output_dir = _output_dir(run, smoke_step=False)
    output_dir = _output_dir(run, smoke_step=args.smoke_step)
    model_artifact = None
    hardware = None
    if not args.dry_run:
        versions = _runtime_versions()
        _require_runtime_versions(versions)
        if versions.get("trl") != EXPECTED_TRL_VERSION:
            raise RuntimeError(
                f"DPO requires trl=={EXPECTED_TRL_VERSION}, observed={versions.get('trl')}"
            )
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
        "run_id": str(run["id"]),
        "mode": "dry_run" if args.dry_run else ("smoke_step" if args.smoke_step else "train"),
        "objective": _loss_contract(loss, reference),
        "dataset": bundle.summary,
        "length_contract": length_contract,
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

    resume = _resume_checkpoint(training, output_dir, smoke_step=args.smoke_step)
    _seed_python_and_torch(int(run["seed"]))
    wandb_session = initialize_wandb_run(
        wandb_cfg,
        output_dir=output_dir,
        metadata=None if resume is not None else {"post_training": preflight},
        active_process=(not args.smoke_step and _global_rank() == 0),
    )
    trainer: Any | None = None
    forward_monitor: DPOForwardContractMonitor | None = None
    try:
        model, processor, model_summary = load_model_and_tokenizer(model_cfg, training)
        tokenizer = text_tokenizer(processor)
        if getattr(processor, "tokenizer", None) is None:
            raise RuntimeError(
                "Gemma4 DPO requires the full Processor; bare-tokenizer fallback is forbidden"
            )
        pad_token = getattr(tokenizer, "pad_token", None)
        if not isinstance(pad_token, str) or not pad_token:
            raise RuntimeError("Gemma4 DPO tokenizer must expose an explicit pad token string")
        runtime_tokenization = validate_runtime_tokenization(
            bundle,
            tokenizer,
            objective="dpo",
            max_length=int(training["max_length"]),
            max_prompt_length=int(training["max_prompt_length"]),
            max_completion_length=int(training["max_completion_length"]),
        )
        # Import only after FastModel has installed the exact Unsloth TRL patch.
        from trl import DPOConfig, DPOTrainer

        require_unsloth_dpo_patch(DPOTrainer, DPOConfig)
        dpo_args = DPOConfig(
            **dpo_config_kwargs(
                run=run,
                loss=loss,
                reference=reference,
                training=training,
                output_dir=output_dir,
                pad_token=pad_token,
                smoke_step=args.smoke_step,
            )
        )
        raw_runtime_dataset = _runtime_dataset(
            bundle,
            smoke_step=args.smoke_step,
            smoke_rows=int(training["smoke_sample_rows"]),
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=raw_runtime_dataset,
            eval_dataset=None,
            processing_class=processor,
            callbacks=[build_wandb_callback(wandb_session)],
        )
        validate_dpo_trainer_contract(trainer)
        prepared_summary = validate_prepared_dpo_dataset(
            raw_dataset=raw_runtime_dataset,
            prepared_dataset=trainer.train_dataset,
            tokenizer=tokenizer,
        )
        forward_monitor = DPOForwardContractMonitor()
        forward_monitor.install(trainer.model)
        # This explicit call is deliberately before trainer.train.  On resume,
        # Trainer has therefore not restored a changed policy checkpoint yet.
        reference_summary = precompute_reference_logps_before_policy_restore(trainer)
        # A resumed W&B run may already be at a later global step.  Do not emit
        # a new step-0 record; the recomputed reference remains in the manifest.
        if wandb_session.active and resume is None:
            wandb_session.log(
                {
                    "train/global_step": 0,
                    "train/reference/precompute_rows": reference_summary["rows"],
                    "train/reference/chosen_mean_logp": reference_summary["chosen_mean"],
                    "train/reference/rejected_mean_logp": reference_summary["rejected_mean"],
                    "train/reference/margin_mean": reference_summary["margin_mean"],
                    "train/reference/precomputed_before_policy_restore": 1,
                }
            )
        result = trainer.train(resume_from_checkpoint=resume)
        forward_summary = forward_monitor.summary()
        forward_monitor.remove()
        forward_monitor = None
        manifest = {
            **preflight,
            "model": model_summary,
            "resolved_trainer_class": trainer.__class__.__name__,
            "runtime_tokenization": runtime_tokenization,
            "prepared_dataset": prepared_summary,
            "reference_precompute": reference_summary,
            "forward_contract": forward_summary,
            "gradient_accumulation_steps": int(dpo_args.gradient_accumulation_steps),
            "world_size": _world_size(),
            "train_metrics": dict(result.metrics),
            "resumed_post_training_checkpoint": resume,
            "global_step": int(trainer.state.global_step),
        }
        if args.smoke_step:
            if int(trainer.state.global_step) != 1:
                raise RuntimeError("DPO smoke did not complete exactly one optimizer step")
            train_loss = float(result.metrics.get("train_loss", float("nan")))
            if not math.isfinite(train_loss):
                raise RuntimeError(f"DPO smoke produced non-finite loss={train_loss}")
            if trainer.is_world_process_zero():
                _write_json(output_dir / "smoke_step_result.json", manifest)
                print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
            return

        trainer.save_state()
        final_dir = output_dir / "final"
        trainer.save_model(str(final_dir))
        if trainer.is_world_process_zero():
            processor.save_pretrained(final_dir)
            _write_json(
                final_dir / "dqs_dpo_model.json",
                {
                    "run_id": str(run["id"]),
                    "objective": _loss_contract(loss, reference),
                    "global_step": int(trainer.state.global_step),
                    "source_model_artifact": dict(model_artifact or {}),
                    "dataset_contract": dict(bundle.contract),
                    "reference_precompute": reference_summary,
                    "forward_contract": forward_summary,
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
        if forward_monitor is not None:
            try:
                forward_monitor.remove()
            except BaseException as monitor_error:
                raise monitor_error from training_error
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
