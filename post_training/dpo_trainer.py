"""Strict guards around Unsloth's patched TRL DPOTrainer.

The production DPO loss stays inside ``UnslothDPOTrainer``.  This module only
enforces the DQS data, reference-policy, and selected-suffix-logit contracts;
it intentionally contains no alternate DPO implementation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


EXPECTED_DPO_TRAINER_CLASS = "UnslothDPOTrainer"
EXPECTED_DPO_CONFIG_CLASS = "UnslothDPOConfig"


def require_unsloth_dpo_patch(trainer_class: type[Any], config_class: type[Any]) -> None:
    """Reject pristine TRL classes or any compatibility fallback."""

    if trainer_class.__name__ != EXPECTED_DPO_TRAINER_CLASS:
        raise RuntimeError(
            "Unsloth did not patch TRL DPOTrainer: "
            f"observed={trainer_class.__module__}.{trainer_class.__name__}"
        )
    if config_class.__name__ != EXPECTED_DPO_CONFIG_CLASS:
        raise RuntimeError(
            "Unsloth did not patch TRL DPOConfig: "
            f"observed={config_class.__module__}.{config_class.__name__}"
        )


def _integer_ids(value: Any, *, field: str, row_index: int) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"prepared DPO row {row_index}: {field} is not a token sequence")
    try:
        result = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"prepared DPO row {row_index}: {field} contains non-integer values"
        ) from exc
    if not result:
        raise ValueError(f"prepared DPO row {row_index}: {field} is empty")
    return result


def validate_prepared_dpo_dataset(
    *,
    raw_dataset: Any,
    prepared_dataset: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Prove that Unsloth/TRL preserved the pre-serialized text exactly."""

    if len(raw_dataset) != len(prepared_dataset):
        raise ValueError(
            "Unsloth DPO preparation changed row count: "
            f"raw={len(raw_dataset)} prepared={len(prepared_dataset)}"
        )
    columns = set(getattr(prepared_dataset, "column_names", []))
    required = {
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
        "mm_token_type_ids",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"prepared DPO dataset is missing token columns: {missing}")
    forbidden_modal_columns = {
        "pixel_values",
        "pixel_attention_mask",
        "image_sizes",
        "pixel_position_ids",
        "image_position_ids",
    }
    present_modal = sorted(forbidden_modal_columns & columns)
    if present_modal:
        raise ValueError(
            "text-only DPO unexpectedly produced visual tensors: "
            f"{present_modal}"
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("DPO tokenizer has no eos_token_id")
    eos = int(eos_token_id)
    maxima = {"prompt_tokens": 0, "chosen_tokens": 0, "rejected_tokens": 0}
    for row_index in range(len(raw_dataset)):
        raw = raw_dataset[row_index]
        prepared = prepared_dataset[row_index]
        expected_prompt = [
            int(item)
            for item in tokenizer(
                str(raw["prompt"]), add_special_tokens=False
            )["input_ids"]
        ]
        expected_chosen = [
            int(item)
            for item in tokenizer(
                str(raw["chosen"]), add_special_tokens=False
            )["input_ids"]
        ] + [eos]
        expected_rejected = [
            int(item)
            for item in tokenizer(
                str(raw["rejected"]), add_special_tokens=False
            )["input_ids"]
        ] + [eos]
        observed = {
            "prompt": _integer_ids(
                prepared["prompt_input_ids"],
                field="prompt_input_ids",
                row_index=row_index,
            ),
            "chosen": _integer_ids(
                prepared["chosen_input_ids"],
                field="chosen_input_ids",
                row_index=row_index,
            ),
            "rejected": _integer_ids(
                prepared["rejected_input_ids"],
                field="rejected_input_ids",
                row_index=row_index,
            ),
        }
        expected = {
            "prompt": expected_prompt,
            "chosen": expected_chosen,
            "rejected": expected_rejected,
        }
        for side in ("prompt", "chosen", "rejected"):
            if observed[side] != expected[side]:
                raise ValueError(
                    f"prepared DPO row {row_index}: {side} tokenization drift"
                )
        for type_field in ("token_type_ids", "mm_token_type_ids"):
            if type_field not in columns:
                continue
            type_ids = _integer_ids(
                prepared[type_field], field=type_field, row_index=row_index
            )
            if len(type_ids) != len(expected_prompt) or any(type_ids):
                raise ValueError(
                    f"prepared DPO row {row_index}: {type_field} is not an all-zero prompt mask"
                )
        maxima["prompt_tokens"] = max(maxima["prompt_tokens"], len(expected_prompt))
        maxima["chosen_tokens"] = max(maxima["chosen_tokens"], len(expected_chosen))
        maxima["rejected_tokens"] = max(
            maxima["rejected_tokens"], len(expected_rejected)
        )
    return {
        "rows": len(raw_dataset),
        "exact_text_tokenization": True,
        "visual_tensor_columns": [],
        "maxima": maxima,
    }


def validate_dpo_trainer_contract(trainer: Any) -> None:
    """Validate the reference policy and pure-DPO runtime settings."""

    if trainer.__class__.__name__ != EXPECTED_DPO_TRAINER_CLASS:
        raise RuntimeError("production trainer is not UnslothDPOTrainer")
    if not bool(getattr(trainer, "precompute_ref_log_probs", False)):
        raise RuntimeError("DPO must precompute reference log-probabilities")
    if getattr(trainer, "ref_model", object()) is not None:
        raise RuntimeError(
            "DPO must use the initial policy for precomputation without a resident reference clone"
        )
    if bool(getattr(trainer, "reference_free", True)):
        raise RuntimeError("reference-free DPO is forbidden")
    if bool(getattr(trainer, "is_peft_model", True)):
        raise RuntimeError("DPO must full-train the verified SFT model, not a PEFT adapter")
    if bool(getattr(trainer, "is_encoder_decoder", True)):
        raise RuntimeError("DPO is pinned to Gemma4 causal language modeling")
    loss_types = getattr(trainer, "loss_type", None)
    if isinstance(loss_types, str):
        loss_types = [loss_types]
    if list(loss_types or []) != ["sigmoid"]:
        raise RuntimeError(f"strict DPO requires sigmoid loss only, got {loss_types!r}")
    if float(getattr(trainer, "label_smoothing", -1.0)) != 0.0:
        raise RuntimeError("strict DPO requires label_smoothing=0")
    f_divergence = getattr(trainer, "f_divergence_type", None)
    f_divergence_value = getattr(f_divergence, "value", f_divergence)
    if str(f_divergence_value) != "reverse_kl":
        raise RuntimeError("strict DPO requires reverse-KL regularization")
    if bool(getattr(trainer, "use_weighting", True)):
        raise RuntimeError("WPO weighting is forbidden in the pure DPO baseline")
    if bool(getattr(trainer, "aux_loss_enabled", True)):
        raise RuntimeError("MoE router auxiliary loss is forbidden in the pure DPO baseline")
    args = trainer.args
    processor = getattr(trainer, "processing_class", None)
    if processor is None or getattr(processor, "tokenizer", None) is None:
        raise RuntimeError("Gemma4 DPO requires the full Processor, not a bare tokenizer")
    if not bool(getattr(trainer, "is_vision_model", False)):
        raise RuntimeError("Gemma4 DPO must execute Unsloth's patched vision-model path")
    collator_class = getattr(getattr(trainer, "data_collator", None), "__class__", None)
    if collator_class is None or not bool(
        getattr(collator_class, "_unsloth_vision_keys_patch", False)
    ):
        raise RuntimeError("Unsloth did not patch the DPO collator for mm_token_type_ids")
    if not bool(getattr(args, "use_logits_to_keep", False)):
        raise RuntimeError("DPO requires use_logits_to_keep=True")
    if not bool(getattr(args, "gradient_checkpointing", False)):
        raise RuntimeError("DPO requires gradient checkpointing in TrainingArguments")
    if not bool(getattr(args, "disable_dropout", False)):
        raise RuntimeError("DPO requires dropout to be disabled for policy/reference parity")
    if bool(getattr(args, "padding_free", True)):
        raise RuntimeError("padding-free DPO is outside the pinned execution contract")
    if bool(getattr(args, "use_liger_loss", True)):
        raise RuntimeError("Liger DPO is outside the pinned execution contract")
    if getattr(args, "rpo_alpha", object()) is not None:
        raise RuntimeError("RPO/SFT mixing is forbidden in the pure DPO baseline")
    if getattr(args, "loss_weights", None) is not None:
        raise RuntimeError("multi-loss weighting is forbidden in the pure DPO baseline")
    if getattr(args, "ld_alpha", object()) is not None:
        raise RuntimeError("LD-DPO is forbidden in the pure DPO baseline")
    if bool(getattr(args, "sync_ref_model", True)):
        raise RuntimeError("TR-DPO reference synchronization is forbidden")


def precompute_reference_logps_before_policy_restore(trainer: Any) -> dict[str, Any]:
    """Precompute the fixed SFT reference before ``Trainer.train`` can resume.

    Calling this explicitly before ``trainer.train(resume_from_checkpoint=...)``
    guarantees that a resumed policy checkpoint can never become its own
    reference model.
    """

    validate_dpo_trainer_contract(trainer)
    if bool(getattr(trainer, "_precomputed_train_ref_log_probs", False)):
        raise RuntimeError("reference log-probabilities were already precomputed unexpectedly")
    if getattr(trainer, "optimizer", None) is not None:
        raise RuntimeError("reference log-probabilities must be computed before optimizer creation")
    trainer.get_train_dataloader()
    if not bool(getattr(trainer, "_precomputed_train_ref_log_probs", False)):
        raise RuntimeError("UnslothDPOTrainer did not mark reference precomputation complete")
    columns = set(getattr(trainer.train_dataset, "column_names", []))
    required = {"ref_chosen_logps", "ref_rejected_logps"}
    if not required.issubset(columns):
        raise RuntimeError("reference log-probability columns were not added to the dataset")

    import numpy as np

    chosen = np.asarray(trainer.train_dataset["ref_chosen_logps"], dtype=np.float32)
    rejected = np.asarray(trainer.train_dataset["ref_rejected_logps"], dtype=np.float32)
    if chosen.ndim != 1 or rejected.ndim != 1 or chosen.shape != rejected.shape:
        raise RuntimeError("reference log-probability arrays have invalid shapes")
    if chosen.size != len(trainer.train_dataset) or chosen.size == 0:
        raise RuntimeError("reference log-probability row count mismatch")
    if not np.isfinite(chosen).all() or not np.isfinite(rejected).all():
        raise RuntimeError("reference log-probabilities contain non-finite values")
    digest = hashlib.sha256()
    digest.update(chosen.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(rejected.astype("<f4", copy=False).tobytes(order="C"))
    return {
        "rows": int(chosen.size),
        "reference_source": "initial_sft_policy_before_any_dpo_update_or_resume_restore",
        "chosen_mean": float(chosen.mean()),
        "rejected_mean": float(rejected.mean()),
        "margin_mean": float((chosen - rejected).mean()),
        "float32_sha256": digest.hexdigest(),
        "precomputed_before_trainer_train": True,
    }


@dataclass
class DPOForwardContractMonitor:
    """Hard-fail if TRL omits or the model ignores ``logits_to_keep``."""

    calls: int = 0
    min_logits_to_keep: int | None = None
    max_logits_to_keep: int | None = None

    def __post_init__(self) -> None:
        self._pending: list[int] = []
        self._handles: list[Any] = []

    def _pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del module
        keep = kwargs.get("logits_to_keep")
        if type(keep) is not int or keep <= 0:
            raise RuntimeError(
                "Unsloth DPO forward omitted a positive integer logits_to_keep; no full-logits fallback is allowed"
            )
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or not hasattr(input_ids, "shape") or len(input_ids.shape) != 2:
            raise RuntimeError("DPO forward contract could not identify 2D input_ids")
        expected = min(int(keep), int(input_ids.shape[1]))
        self._pending.append(expected)
        self.min_logits_to_keep = (
            keep if self.min_logits_to_keep is None else min(self.min_logits_to_keep, keep)
        )
        self.max_logits_to_keep = (
            keep if self.max_logits_to_keep is None else max(self.max_logits_to_keep, keep)
        )

    def _post_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs
        if not self._pending:
            raise RuntimeError("DPO forward hook observed output without a matching input")
        expected = self._pending.pop()
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
        if logits is None or not hasattr(logits, "shape") or len(logits.shape) != 3:
            raise RuntimeError("DPO model did not return a real 3D logits tensor")
        if int(logits.shape[1]) != expected:
            raise RuntimeError(
                "DPO model ignored logits_to_keep: "
                f"returned={int(logits.shape[1])} expected={expected}"
            )
        self.calls += 1

    def install(self, model: Any) -> None:
        if self._handles:
            raise RuntimeError("DPO forward contract monitor is already installed")
        self._handles = [
            model.register_forward_pre_hook(self._pre_hook, with_kwargs=True),
            model.register_forward_hook(self._post_hook, with_kwargs=True),
        ]

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._pending:
            raise RuntimeError("DPO forward contract monitor ended with unfinished forwards")

    def summary(self) -> dict[str, Any]:
        if self.calls <= 0:
            raise RuntimeError("DPO forward contract was never exercised")
        if self._pending:
            raise RuntimeError("DPO forward contract has unfinished forwards")
        return {
            "calls": self.calls,
            "min_logits_to_keep": self.min_logits_to_keep,
            "max_logits_to_keep": self.max_logits_to_keep,
            "selected_suffix_logits_enforced": True,
            "full_logits_fallback": False,
        }
