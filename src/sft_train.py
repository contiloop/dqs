#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from config_loader import compose_config
from io_utils import read_jsonl
from wandb_logging import configure_wandb_env, log_wandb_metrics


IGNORE_INDEX = -100


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _subset_dir(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return Path(str(_get(cfg, "paths.subset_dir"))) / f"subset_{subset_idx:03d}"


def _checkpoint_dir(cfg: Mapping[str, Any]) -> Path:
    return Path(str(_get(cfg, "paths.checkpoint_dir")))


def _default_sft_dataset_path(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return _subset_dir(cfg, subset_idx) / "sft_train.jsonl"


def _default_max_seq_length(cfg: Mapping[str, Any]) -> int:
    configured = _get(cfg, "training.max_seq_length")
    if configured is not None:
        return int(configured)
    max_input = int(_get(cfg, "data.max_input_tokens", 1280) or 1280)
    max_output = int(_get(cfg, "data.max_output_tokens", 1500) or 1500)
    prompt_overhead = int(_get(cfg, "training.prompt_overhead_tokens", 128) or 128)
    return max_input + max_output + prompt_overhead


def _world_size() -> int:
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        raw = os.environ.get(key)
        if raw and raw.isdigit():
            return max(1, int(raw))
    return 1


def _gradient_accumulation_steps(cfg: Mapping[str, Any]) -> int:
    raw = _get(cfg, "training.gradient_accumulation_steps", "auto")
    if raw != "auto":
        return max(1, int(raw))
    effective = int(_get(cfg, "training.effective_batch_size", 128) or 128)
    per_device = int(_get(cfg, "training.per_device_train_batch_size", 1) or 1)
    denominator = max(1, per_device * _world_size())
    return max(1, math.ceil(effective / denominator))


def estimate_update_steps_for_rows(row_count: int, cfg: Mapping[str, Any]) -> int:
    per_device = int(_get(cfg, "training.per_device_train_batch_size", 1) or 1)
    micro_batch_rows = max(1, per_device * _world_size())
    micro_batches = max(1, math.ceil(max(1, row_count) / micro_batch_rows))
    return max(1, math.ceil(micro_batches / _gradient_accumulation_steps(cfg)))


def _torch_dtype(dtype_name: Any) -> Any:
    if dtype_name is None:
        return None
    text = str(dtype_name).strip().lower()
    if text in {"", "auto", "none", "null"}:
        return "auto"
    import torch

    if text in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "torch.float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise SystemExit(f"unsupported training.dtype={dtype_name!r}")


def _call_with_supported_kwargs(fn: Any, kwargs: Mapping[str, Any]) -> Any:
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**filtered)


def _resolve_model_api(cfg: Mapping[str, Any]) -> Any:
    model_cfg = _get(cfg, "model", {})
    use_vision_api = bool(isinstance(model_cfg, Mapping) and model_cfg.get("is_vision_model", False))
    try:
        if use_vision_api:
            from unsloth import FastVisionModel

            return FastVisionModel
        from unsloth import FastLanguageModel

        return FastLanguageModel
    except ModuleNotFoundError as exc:
        raise SystemExit("missing unsloth; run `make set` first") from exc


def _load_model_and_tokenizer(cfg: Mapping[str, Any], max_seq_length: int) -> tuple[Any, Any, Any]:
    model_cfg = _get(cfg, "model", {})
    training_cfg = _get(cfg, "training", {})
    if not isinstance(model_cfg, Mapping):
        raise SystemExit("model config must be a mapping")
    if not isinstance(training_cfg, Mapping):
        raise SystemExit("training config must be a mapping")
    if str(training_cfg.get("backend", "unsloth")).lower() != "unsloth":
        raise SystemExit("training.backend must be unsloth; fallback training backends are disabled")

    model_api = _resolve_model_api(cfg)
    dtype = _torch_dtype(training_cfg.get("dtype", "auto"))
    load_kwargs = {
        "model_name": str(model_cfg["name_or_path"]),
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "load_in_4bit": bool(training_cfg.get("load_in_4bit", False)),
        "load_in_8bit": bool(training_cfg.get("load_in_8bit", False)),
        "full_finetuning": str(training_cfg.get("tuning_mode", "")).lower() == "full",
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "revision": str(model_cfg.get("revision", model_cfg.get("tokenizer_revision", "main"))),
        "use_gradient_checkpointing": training_cfg.get("gradient_checkpointing", "unsloth"),
    }
    model, tokenizer = _call_with_supported_kwargs(model_api.from_pretrained, load_kwargs)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None and getattr(model, "config", None) is not None:
        model.config.pad_token_id = pad_token_id
    return model, tokenizer, model_api


def _linear_leaf_module_short_names(model: Any) -> list[str]:
    target_names: set[str] = set()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        children = list(module.children())
        if children:
            continue
        class_name = module.__class__.__name__.lower()
        if "linear" not in class_name:
            continue
        short_name = module_name.rsplit(".", 1)[-1]
        if short_name == "lm_head":
            continue
        target_names.add(short_name)
    if not target_names:
        raise SystemExit("could not discover LoRA target linear modules from the loaded model")
    return sorted(target_names)


def _write_lora_target_audit(path: Path, model: Any, target_modules: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = []
    for name, parameter in model.named_parameters():
        if getattr(parameter, "requires_grad", False):
            trainable.append(name)
    payload = {
        "target_modules": target_modules,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": len(trainable),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _apply_lora_if_needed(cfg: Mapping[str, Any], model: Any, model_api: Any, audit_dir: Path) -> Any:
    training_cfg = _get(cfg, "training", {})
    if not isinstance(training_cfg, Mapping):
        raise SystemExit("training config must be a mapping")
    if str(training_cfg.get("tuning_mode", "")).lower() != "lora":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model
    lora_cfg = training_cfg.get("lora")
    if not isinstance(lora_cfg, Mapping):
        raise SystemExit("training.lora must be configured for LoRA training")

    target_modules: Any = lora_cfg.get("target_modules", "auto_discover_text_layers")
    if target_modules == "auto_discover_text_layers":
        if bool(_get(cfg, "model.is_vision_model", False)):
            target_modules = "all-linear"
        else:
            target_modules = _linear_leaf_module_short_names(model)

    peft_kwargs = {
        "model": model,
        "r": int(lora_cfg.get("rank", 16) or 16),
        "target_modules": target_modules,
        "lora_alpha": int(lora_cfg.get("alpha", 16) or 16),
        "lora_dropout": float(lora_cfg.get("dropout", 0.0) or 0.0),
        "bias": str(lora_cfg.get("bias", "none")),
        "use_rslora": bool(lora_cfg.get("use_rslora", False)),
        "loftq_config": lora_cfg.get("loftq_config"),
        "random_state": int(_get(cfg, "run.seed", 42) or 42),
        "use_gradient_checkpointing": training_cfg.get("gradient_checkpointing", "unsloth"),
        "modules_to_save": lora_cfg.get("modules_to_save"),
        "finetune_vision_layers": bool(training_cfg.get("train_vision_layers", False)),
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
    }
    model = _call_with_supported_kwargs(model_api.get_peft_model, peft_kwargs)
    _write_lora_target_audit(audit_dir / "lora_target_modules.json", model, target_modules)
    return model


def _completion_text(row: Mapping[str, Any]) -> str:
    value = row.get("completion", row.get("response", row.get("target", "")))
    return str(value).strip()


def _tokenize_prompt_completion(
    *,
    tokenizer: Any,
    row: Mapping[str, Any],
    max_seq_length: int,
    append_eos_token: bool,
    prevent_truncation: bool,
    response_only_loss: bool,
) -> dict[str, Any]:
    prompt = str(row.get("prompt", "") or "")
    completion = _completion_text(row)
    if not prompt.strip() or not completion:
        raise ValueError(f"row {row.get('id')} must contain non-empty prompt and completion/response/target")

    eos_token = getattr(tokenizer, "eos_token", None)
    if append_eos_token and eos_token and not completion.endswith(str(eos_token)):
        completion_for_tokens = completion + str(eos_token)
    else:
        completion_for_tokens = completion

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_for_tokens, add_special_tokens=False)["input_ids"]
    input_ids = list(prompt_ids) + list(completion_ids)
    if response_only_loss:
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    else:
        labels = list(input_ids)
    if not completion_ids:
        raise ValueError(f"row {row.get('id')} produced no completion tokens")
    if len(input_ids) > max_seq_length:
        message = (
            f"row {row.get('id')} exceeds training.max_seq_length={max_seq_length}: "
            f"prompt_tokens={len(prompt_ids)} completion_tokens={len(completion_ids)} total={len(input_ids)}"
        )
        if prevent_truncation:
            raise SystemExit(message)
        keep_completion = max_seq_length - len(prompt_ids)
        if keep_completion <= 0:
            raise SystemExit(message)
        input_ids = list(prompt_ids) + list(completion_ids[:keep_completion])
        if response_only_loss:
            labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids[:keep_completion])
        else:
            labels = list(input_ids)
    return {
        "id": row.get("id"),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_token_count": len(prompt_ids),
        "completion_token_count": len(completion_ids),
        "supervised_token_count": sum(1 for value in labels if value != IGNORE_INDEX),
    }


def _prepare_tokenized_rows(
    *,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    max_seq_length: int,
) -> list[dict[str, Any]]:
    training_cfg = _get(cfg, "training", {})
    if not isinstance(training_cfg, Mapping):
        raise SystemExit("training config must be a mapping")
    tokenized = [
        _tokenize_prompt_completion(
            tokenizer=tokenizer,
            row=row,
            max_seq_length=max_seq_length,
            append_eos_token=bool(training_cfg.get("append_eos_token", True)),
            prevent_truncation=bool(training_cfg.get("prevent_template_truncation", True)),
            response_only_loss=bool(training_cfg.get("response_only_loss", True)),
        )
        for row in rows
    ]
    if not tokenized:
        raise SystemExit("SFT dataset has no usable rows")
    return tokenized


class PromptCompletionDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class CompletionOnlyCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * pad_len)
            attention_mask.append(list(feature["attention_mask"]) + [0] * pad_len)
            labels.append(list(feature["labels"]) + [IGNORE_INDEX] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _write_mask_audit(path: Path, tokenizer: Any, rows: Sequence[Mapping[str, Any]], sample_size: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in rows[:sample_size]:
        labels = list(row["labels"])
        first_supervised = next((idx for idx, value in enumerate(labels) if value != IGNORE_INDEX), None)
        supervised_ids = [value for value in labels if value != IGNORE_INDEX]
        decoded = tokenizer.decode(supervised_ids, skip_special_tokens=False)
        samples.append(
            {
                "id": row.get("id"),
                "prompt_token_count": row["prompt_token_count"],
                "completion_token_count": row["completion_token_count"],
                "supervised_token_count": row["supervised_token_count"],
                "first_supervised_token_index": first_supervised,
                "decoded_supervised_prefix": decoded[:500],
            }
        )
    payload = {
        "ignore_index": IGNORE_INDEX,
        "sample_count": len(samples),
        "samples": samples,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _latest_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if not checkpoints:
        return None
    return str(sorted(checkpoints, key=lambda item: item[0])[-1][1])


def _checkpoint_global_step(checkpoint: str | Path | bool | None) -> int:
    if checkpoint is None or isinstance(checkpoint, bool):
        return 0
    checkpoint_path = Path(checkpoint)
    state_path = checkpoint_path / "trainer_state.json"
    if state_path.exists():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            return int(data.get("global_step", 0) or 0)
    suffix = checkpoint_path.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else 0


def _sft_stage_state_path(output_dir: Path, subset_idx: int) -> Path:
    return output_dir / f"sft_stage_state_subset_{subset_idx:03d}.json"


def _read_sft_stage_state(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_sft_stage_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_stage_sft_plan(
    *,
    cfg: Mapping[str, Any],
    output_dir: Path,
    subset_idx: int,
    row_count: int,
    resume_from_checkpoint: str | bool | None,
    scheduler_total_steps: int | None,
) -> dict[str, Any] | None:
    if scheduler_total_steps is None:
        return None
    if scheduler_total_steps <= 0:
        raise SystemExit("stage scheduler total steps must be > 0")

    latest_global_step = _checkpoint_global_step(resume_from_checkpoint)
    run_subset_start = int(_get(cfg, "run.subset_start", 0) or 0)
    if subset_idx > run_subset_start and resume_from_checkpoint is None:
        raise SystemExit(
            "stage SFT requires a previous checkpoint before training a later subset; "
            f"no checkpoint found for subset_{subset_idx:03d}"
        )

    state_path = _sft_stage_state_path(output_dir, subset_idx)
    previous_state = _read_sft_stage_state(state_path)
    if previous_state and previous_state.get("status") != "completed":
        start_global_step = int(previous_state.get("stage_start_global_step", latest_global_step) or latest_global_step)
        subset_update_steps = int(
            previous_state.get("stage_subset_update_steps", estimate_update_steps_for_rows(row_count, cfg)) or 1
        )
        target_global_step = int(previous_state.get("stage_target_global_step", start_global_step + subset_update_steps))
    else:
        start_global_step = latest_global_step
        subset_update_steps = estimate_update_steps_for_rows(row_count, cfg)
        target_global_step = start_global_step + subset_update_steps

    if target_global_step > scheduler_total_steps:
        raise SystemExit(
            "stage SFT target global step exceeds planned scheduler total steps: "
            f"target={target_global_step}, scheduler_total={scheduler_total_steps}. "
            "Increase the stage scheduler plan or reduce the selected stage range."
        )

    return {
        "path": str(state_path),
        "status": "running",
        "subset_idx": subset_idx,
        "stage_scheduler_total_steps": scheduler_total_steps,
        "stage_start_global_step": start_global_step,
        "stage_resume_global_step": latest_global_step,
        "stage_subset_update_steps": subset_update_steps,
        "stage_target_global_step": target_global_step,
        "sft_rows": row_count,
    }


def _training_argument_kwargs(
    cfg: Mapping[str, Any],
    output_dir: Path,
    *,
    max_steps_override: int | None = None,
    ignore_data_skip_override: bool | None = None,
) -> dict[str, Any]:
    training_cfg = _get(cfg, "training", {})
    logging_cfg = _get(cfg, "logging", {})
    if not isinstance(training_cfg, Mapping):
        raise SystemExit("training config must be a mapping")
    if not isinstance(logging_cfg, Mapping):
        logging_cfg = {}
    import torch

    wandb_cfg = logging_cfg.get("wandb", {}) if isinstance(logging_cfg.get("wandb", {}), Mapping) else {}
    report_to: list[str] = ["wandb"] if bool(wandb_cfg.get("enabled", False)) else []
    configure_wandb_env(cfg)
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = bool(torch.cuda.is_available() and not bf16)
    return {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1) or 1),
        "gradient_accumulation_steps": _gradient_accumulation_steps(cfg),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-5) or 2e-5),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1) or 0.1),
        "lr_scheduler_type": str(training_cfg.get("scheduler", "cosine")),
        "optim": str(training_cfg.get("optimizer", "adamw_torch")),
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 1.0) or 1.0),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0) or 0.0),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 1) or 1),
        "max_steps": int(max_steps_override if max_steps_override is not None else (training_cfg.get("max_steps", -1) or -1)),
        "save_strategy": str(training_cfg.get("save_strategy", "steps")),
        "save_steps": int(training_cfg.get("save_steps", 100) or 100),
        "save_total_limit": (
            None
            if training_cfg.get("save_total_limit") is None
            else int(training_cfg.get("save_total_limit") or 0)
        ),
        "logging_steps": int(training_cfg.get("logging_steps", 10) or 10),
        "logging_dir": str(_get(cfg, "logging.local.root_dir", output_dir / "logs")),
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 0) or 0),
        "remove_unused_columns": False,
        "save_safetensors": True,
        "seed": int(_get(cfg, "run.seed", 42) or 42),
        "data_seed": int(_get(cfg, "run.seed", 42) or 42),
        "ignore_data_skip": bool(
            ignore_data_skip_override
            if ignore_data_skip_override is not None
            else training_cfg.get("ignore_data_skip", False)
        ),
        "bf16": bf16,
        "fp16": fp16,
        "report_to": report_to,
        "run_name": str(wandb_cfg.get("run_name", _get(cfg, "run.id", "dqs"))),
        "ddp_find_unused_parameters": False if _world_size() > 1 else None,
    }


def _make_training_arguments(
    cfg: Mapping[str, Any],
    output_dir: Path,
    *,
    max_steps_override: int | None = None,
    ignore_data_skip_override: bool | None = None,
) -> Any:
    from transformers import TrainingArguments

    kwargs = _training_argument_kwargs(
        cfg,
        output_dir,
        max_steps_override=max_steps_override,
        ignore_data_skip_override=ignore_data_skip_override,
    )
    signature = inspect.signature(TrainingArguments)
    kwargs = {key: value for key, value in kwargs.items() if value is not None and key in signature.parameters}
    return TrainingArguments(**kwargs)


def _resume_checkpoint(cfg: Mapping[str, Any], output_dir: Path) -> str | bool | None:
    raw = _get(cfg, "training.resume_from_checkpoint", "auto")
    if raw is None or raw is False or str(raw).lower() in {"false", "none", "null", "no"}:
        return None
    if str(raw).lower() == "auto":
        return _latest_checkpoint(output_dir)
    if raw is True or str(raw).lower() == "true":
        return True
    return str(raw)


def _save_checkpoint_at_current_step(trainer: Any) -> None:
    if hasattr(trainer, "_save_checkpoint"):
        _call_with_supported_kwargs(
            trainer._save_checkpoint,
            {
                "model": trainer.model,
                "trial": None,
            },
        )
        return
    raise SystemExit("Trainer does not expose checkpoint save API required for stage resume")


def _smoke_test_transformers_artifact(path: Path, *, mode: str, trust_remote_code: bool) -> dict[str, Any]:
    mode = mode.strip().lower()
    if mode in {"", "none", "skip", "false"}:
        return {"path": str(path), "mode": mode, "status": "skipped"}
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(str(path), trust_remote_code=trust_remote_code, local_files_only=True)
        AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust_remote_code, local_files_only=True)
        if mode == "load_model":
            from transformers import AutoModelForCausalLM

            AutoModelForCausalLM.from_pretrained(
                str(path),
                trust_remote_code=trust_remote_code,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        elif mode != "config_and_tokenizer":
            raise ValueError("mode must be config_and_tokenizer, load_model, or none")
    except Exception as exc:
        raise SystemExit(f"saved model smoke test failed for {path}: {exc}") from exc
    return {"path": str(path), "mode": mode, "status": "ok"}


def _save_model_artifacts(cfg: Mapping[str, Any], model: Any, tokenizer: Any, output_dir: Path) -> dict[str, Any]:
    training_cfg = _get(cfg, "training", {})
    tuning_mode = str(_get(cfg, "training.tuning_mode", "")).lower()
    artifacts: dict[str, Any] = {}
    smoke_tests: list[dict[str, Any]] = []
    trust_remote_code = bool(_get(cfg, "model.trust_remote_code", False))
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    artifacts["final_model_dir"] = str(final_dir)

    if tuning_mode == "lora":
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        artifacts["adapter_dir"] = str(adapter_dir)
        if bool(training_cfg.get("save_merged_model", False)):
            merged_dir = output_dir / "merged_16bit"
            if not hasattr(model, "save_pretrained_merged"):
                raise SystemExit("LoRA merged save requested, but model.save_pretrained_merged is unavailable")
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
            artifacts["merged_model_dir"] = str(merged_dir)
            if bool(training_cfg.get("merge_smoke_test_required", False)):
                smoke_tests.append(
                    _smoke_test_transformers_artifact(
                        merged_dir,
                        mode=str(training_cfg.get("merge_smoke_test_mode", "config_and_tokenizer")),
                        trust_remote_code=trust_remote_code,
                    )
                )
    elif bool(training_cfg.get("save_full_model", True)):
        full_dir = output_dir / "full_model"
        full_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(full_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(full_dir))
        artifacts["full_model_dir"] = str(full_dir)
    if smoke_tests:
        artifacts["smoke_tests"] = smoke_tests
    return artifacts


def run_sft_training(
    *,
    cfg: Mapping[str, Any],
    subset_idx: int,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    stage_scheduler_total_steps: int | None = None,
    force_save_checkpoint: bool = False,
) -> dict[str, Any]:
    dataset_path_obj = Path(dataset_path) if dataset_path else _default_sft_dataset_path(cfg, subset_idx)
    if not dataset_path_obj.exists():
        raise SystemExit(f"missing SFT dataset: {dataset_path_obj}")
    output_dir_obj = Path(output_dir) if output_dir else _checkpoint_dir(cfg)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(dataset_path_obj)
    if not rows:
        raise SystemExit(f"SFT dataset is empty: {dataset_path_obj}")

    max_seq_length = _default_max_seq_length(cfg)
    model, tokenizer, model_api = _load_model_and_tokenizer(cfg, max_seq_length=max_seq_length)
    tokenized_rows = _prepare_tokenized_rows(
        tokenizer=tokenizer,
        rows=rows,
        cfg=cfg,
        max_seq_length=max_seq_length,
    )
    audit_dir = output_dir_obj / "audit"
    _write_mask_audit(audit_dir / f"sft_mask_audit_subset_{subset_idx:03d}.json", tokenizer, tokenized_rows)
    max_observed_length = max(len(row["input_ids"]) for row in tokenized_rows)
    summary: dict[str, Any] = {
        "run_id": _get(cfg, "run.id"),
        "subset_idx": subset_idx,
        "sft_dataset_path": str(dataset_path_obj),
        "sft_rows": len(tokenized_rows),
        "max_seq_length": max_seq_length,
        "max_observed_length": max_observed_length,
        "response_only_loss": bool(_get(cfg, "training.response_only_loss", True)),
        "gradient_accumulation_steps": _gradient_accumulation_steps(cfg),
        "world_size": _world_size(),
        "output_dir": str(output_dir_obj),
        "dry_run": dry_run,
    }
    if dry_run:
        summary_path = output_dir_obj / f"sft_dry_run_summary_subset_{subset_idx:03d}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return summary

    model = _apply_lora_if_needed(cfg, model, model_api, audit_dir)
    from transformers import Trainer, set_seed

    set_seed(int(_get(cfg, "run.seed", 42) or 42))
    train_dataset = PromptCompletionDataset(tokenized_rows)
    resume_from_checkpoint = _resume_checkpoint(cfg, output_dir_obj)
    stage_plan = _resolve_stage_sft_plan(
        cfg=cfg,
        output_dir=output_dir_obj,
        subset_idx=subset_idx,
        row_count=len(tokenized_rows),
        resume_from_checkpoint=resume_from_checkpoint,
        scheduler_total_steps=stage_scheduler_total_steps,
    )

    class DQSStageSchedulerTrainer(Trainer):
        def __init__(self, *trainer_args: Any, dqs_scheduler_total_steps: int | None = None, **trainer_kwargs: Any) -> None:
            self._dqs_scheduler_total_steps = dqs_scheduler_total_steps
            super().__init__(*trainer_args, **trainer_kwargs)

        def create_scheduler(self, num_training_steps: int, optimizer: Any = None) -> Any:
            if self._dqs_scheduler_total_steps is not None:
                num_training_steps = self._dqs_scheduler_total_steps
            return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    training_args = _make_training_arguments(
        cfg,
        output_dir_obj,
        max_steps_override=stage_plan["stage_target_global_step"] if stage_plan else None,
        ignore_data_skip_override=True if stage_plan else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CompletionOnlyCollator(tokenizer),
    ) if stage_plan is None else DQSStageSchedulerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CompletionOnlyCollator(tokenizer),
        dqs_scheduler_total_steps=int(stage_plan["stage_scheduler_total_steps"]),
    )
    if stage_plan:
        _write_sft_stage_state(Path(stage_plan["path"]), stage_plan)
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_state()
        if force_save_checkpoint or stage_plan:
            _save_checkpoint_at_current_step(trainer)
    except BaseException as exc:
        if stage_plan:
            failed_plan = dict(stage_plan)
            failed_plan["status"] = "failed"
            failed_plan["error"] = f"{type(exc).__name__}: {exc}"
            failed_plan["actual_global_step"] = int(getattr(trainer.state, "global_step", 0))
            _write_sft_stage_state(Path(stage_plan["path"]), failed_plan)
        raise
    artifacts = _save_model_artifacts(cfg, model, tokenizer, output_dir_obj)
    summary.update(
        {
            "dry_run": False,
            "resume_from_checkpoint": resume_from_checkpoint,
            "global_step": int(getattr(trainer.state, "global_step", 0)),
            "train_loss": train_result.metrics.get("train_loss") if hasattr(train_result, "metrics") else None,
            "artifacts": artifacts,
        }
    )
    if stage_plan:
        completed_plan = dict(stage_plan)
        completed_plan["status"] = "completed"
        completed_plan["actual_global_step"] = summary["global_step"]
        completed_plan["checkpoint_dir"] = str(Path(output_dir_obj) / f"checkpoint-{summary['global_step']}")
        _write_sft_stage_state(Path(stage_plan["path"]), completed_plan)
        summary.update(
            {
                "stage_scheduler_total_steps": stage_plan["stage_scheduler_total_steps"],
                "stage_start_global_step": stage_plan["stage_start_global_step"],
                "stage_resume_global_step": stage_plan["stage_resume_global_step"],
                "stage_subset_update_steps": stage_plan["stage_subset_update_steps"],
                "stage_target_global_step": stage_plan["stage_target_global_step"],
                "stage_state_path": stage_plan["path"],
            }
        )
    log_wandb_metrics(
        cfg,
        {
            "train/global_step": summary.get("global_step"),
            "train/subset_idx": subset_idx,
            "train/subset_sft_rows": len(tokenized_rows),
            "train/subset_train_loss": summary.get("train_loss"),
            "stage/subset_update_steps": summary.get("stage_subset_update_steps"),
            "stage/scheduler_total_steps": summary.get("stage_scheduler_total_steps"),
        },
        step=summary.get("global_step") if isinstance(summary.get("global_step"), int) else None,
        job_type="sft",
    )
    summary_path = output_dir_obj / f"sft_training_summary_subset_{subset_idx:03d}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQS Unsloth SFT on an sft_train.jsonl file.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset-idx", type=int, default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = compose_config(args.config, overrides=args.override)
    subset_idx = int(args.subset_idx if args.subset_idx is not None else _get(cfg, "run.subset_start", 0))
    summary = run_sft_training(
        cfg=cfg,
        subset_idx=subset_idx,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
