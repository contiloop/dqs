"""Batch padding and causal-LM masked log-probability primitives for mPO.

The token masks in the prepared artifact are aligned to ``input_ids``.  A
causal LM predicts ``input_ids[:, 1:]`` from ``logits[:, :-1]``, so every loss
mask is shifted with ``[:, 1:]`` inside :func:`masked_causal_logp_mean`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MASK_FIELDS = ("completion_mask", "term_mask")


def _validate_binary_mask(values: Sequence[int], *, name: str, expected: int) -> list[int]:
    mask = [int(value) for value in values]
    if len(mask) != expected:
        raise ValueError(f"{name} length={len(mask)} does not match input length={expected}")
    if any(value not in (0, 1) for value in mask):
        raise ValueError(f"{name} must contain only 0/1 values")
    return mask


class MPOPreferenceCollator:
    """Right-pad chosen and rejected sequences while zero-padding every mask."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def _pad_side(self, features: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
        import torch

        ids_field = f"{side}_input_ids"
        sequences = [[int(value) for value in feature[ids_field]] for feature in features]
        if not sequences or any(not sequence for sequence in sequences):
            raise ValueError(f"{ids_field} must contain non-empty sequences")
        max_length = max(len(sequence) for sequence in sequences)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        padded_masks: dict[str, list[list[int]]] = {field: [] for field in MASK_FIELDS}
        for feature, sequence in zip(features, sequences, strict=True):
            length = len(sequence)
            pad_length = max_length - length
            input_ids.append(sequence + [self.pad_token_id] * pad_length)
            attention_mask.append([1] * length + [0] * pad_length)
            for field in MASK_FIELDS:
                source_name = f"{side}_{field}"
                mask = _validate_binary_mask(feature[source_name], name=source_name, expected=length)
                padded_masks[field].append(mask + [0] * pad_length)

        result = {
            f"{side}_input_ids": torch.tensor(input_ids, dtype=torch.long),
            f"{side}_attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }
        for field, values in padded_masks.items():
            result[f"{side}_{field}"] = torch.tensor(values, dtype=torch.bool)
        return result

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("MPOPreferenceCollator received an empty batch")
        batch: dict[str, Any] = {"pair_id": [str(feature["pair_id"]) for feature in features]}
        batch.update(self._pad_side(features, "chosen"))
        batch.update(self._pad_side(features, "rejected"))
        return batch


def _token_logps(logits: Any, labels: Any, *, backend: str) -> Any:
    """Return per-token log-probabilities with an explicit backend."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels shapes are incompatible")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    if backend == "torch":
        import torch.nn.functional as functional

        flat_losses = functional.cross_entropy(flat_logits, flat_labels, reduction="none")
    elif backend == "unsloth_fused":
        try:
            from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "required Unsloth fused per-token cross entropy is unavailable; no fallback is allowed"
            ) from exc
        # Gemma4's forward has already applied final-logit softcapping. This
        # kernel supplies stable FP32 per-token CE/logsumexp without a dense
        # log-softmax allocation.
        flat_losses = Fast_CrossEntropyLoss.apply(flat_logits, flat_labels, 0.0, 0.0)
    else:
        raise ValueError(f"unknown token log-probability backend={backend!r}")
    return -flat_losses.reshape_as(labels)


def shifted_token_logps(logits: Any, input_ids: Any, *, backend: str) -> Any:
    """Return log p(x_t | x_<t) aligned to ``input_ids[:, 1:]``."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, seq, vocab] and input_ids [batch, seq]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("logits/input_ids shapes are incompatible for causal shifting")
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    return _token_logps(shift_logits, shift_labels, backend=backend)


def causal_prediction_positions(*, token_mask: Any, attention_mask: Any) -> Any:
    """Return sequence positions whose logits are needed by any batch row.

    A target token at input position ``t`` is predicted by the logit at
    position ``t - 1``.  The returned indices therefore address the model's
    unshifted logits/hidden-state sequence, not ``input_ids[:, 1:]``.
    """

    import torch

    if token_mask.shape != attention_mask.shape or token_mask.ndim != 2:
        raise ValueError("token_mask and attention_mask must be matching 2D tensors")
    effective = token_mask[:, 1:].bool() & attention_mask[:, 1:].bool()
    per_row_counts = effective.sum(dim=-1)
    if torch.any(per_row_counts == 0):
        bad_rows = torch.nonzero(per_row_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"empty shifted loss mask for batch rows {bad_rows}")
    return torch.nonzero(effective.any(dim=0), as_tuple=False).flatten()


def selected_causal_token_logps(
    *,
    logits: Any,
    prediction_positions: Any,
    input_ids: Any,
    backend: str,
) -> tuple[Any, Any]:
    """Score selected causal targets exactly once with the requested backend.

    Unsloth's fused CE reuses its saved logits buffer during backward. Callers
    that need several masks over the same logits must therefore share this one
    token-logp tensor and aggregate it with :func:`selected_token_logp_mean`.
    """

    import torch

    if prediction_positions.ndim != 1 or prediction_positions.numel() == 0:
        raise ValueError("prediction_positions must be a non-empty 1D tensor")
    if torch.any(prediction_positions < 0) or torch.any(prediction_positions >= input_ids.shape[1] - 1):
        raise ValueError("prediction_positions are outside the causal prediction range")
    if logits.ndim != 3 or logits.shape[0] != input_ids.shape[0]:
        raise ValueError("selected logits must have shape [batch, selected_positions, vocab]")
    if logits.shape[1] != prediction_positions.numel():
        raise ValueError(
            "selected logits length does not match prediction_positions: "
            f"{logits.shape[1]} != {prediction_positions.numel()}"
        )

    target_positions = prediction_positions + 1
    target_ids = input_ids.index_select(1, target_positions)
    selected_logps = _token_logps(logits, target_ids, backend=backend)
    return selected_logps, target_positions


def selected_token_logp_mean(
    *,
    token_logps: Any,
    target_positions: Any,
    input_ids: Any,
    token_mask: Any,
    attention_mask: Any,
) -> tuple[Any, Any]:
    """Aggregate already-scored selected tokens with one independent mask."""

    import torch

    if token_mask.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("token_mask and attention_mask must match input_ids")
    if target_positions.ndim != 1 or target_positions.numel() == 0:
        raise ValueError("target_positions must be a non-empty 1D tensor")
    if torch.any(target_positions < 1) or torch.any(target_positions >= input_ids.shape[1]):
        raise ValueError("target_positions are outside the causal target range")
    if token_logps.ndim != 2 or token_logps.shape != (
        input_ids.shape[0],
        target_positions.numel(),
    ):
        raise ValueError("token_logps shape does not match batch and selected target positions")
    effective_mask = (
        token_mask.index_select(1, target_positions).bool()
        & attention_mask.index_select(1, target_positions).bool()
    )
    token_counts = effective_mask.sum(dim=-1)
    if torch.any(token_counts == 0):
        bad_rows = torch.nonzero(token_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"empty selected loss mask for batch rows {bad_rows}")
    totals = (token_logps * effective_mask.to(token_logps.dtype)).sum(dim=-1)
    return totals / token_counts.to(token_logps.dtype), token_counts


def selected_causal_logp_mean(
    *,
    logits: Any,
    prediction_positions: Any,
    input_ids: Any,
    token_mask: Any,
    attention_mask: Any,
    backend: str,
) -> tuple[Any, Any]:
    """Convenience wrapper for a single mask over selected causal logits."""

    token_logps, target_positions = selected_causal_token_logps(
        logits=logits,
        prediction_positions=prediction_positions,
        input_ids=input_ids,
        backend=backend,
    )
    return selected_token_logp_mean(
        token_logps=token_logps,
        target_positions=target_positions,
        input_ids=input_ids,
        token_mask=token_mask,
        attention_mask=attention_mask,
    )


def masked_causal_logp_mean(
    *,
    logits: Any,
    input_ids: Any,
    token_mask: Any,
    attention_mask: Any,
    backend: str,
) -> tuple[Any, Any]:
    """Compute a per-row masked mean after causal shift and padding removal.

    ``token_mask`` is input-aligned.  Prompt exclusion is expressed by zeros in
    that mask; padding exclusion is enforced independently with
    ``attention_mask``.  Normalization happens per row before a caller performs
    any batch reduction.
    """

    import torch

    if token_mask.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("token_mask and attention_mask must match input_ids")
    token_logps = shifted_token_logps(logits, input_ids, backend=backend)
    effective_mask = token_mask[:, 1:].bool() & attention_mask[:, 1:].bool()
    token_counts = effective_mask.sum(dim=-1)
    if torch.any(token_counts == 0):
        bad_rows = torch.nonzero(token_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"empty shifted loss mask for batch rows {bad_rows}")
    totals = (token_logps * effective_mask.to(token_logps.dtype)).sum(dim=-1)
    return totals / token_counts.to(token_logps.dtype), token_counts


def completion_sft_loss(
    *,
    logits: Any,
    input_ids: Any,
    completion_mask: Any,
    attention_mask: Any,
    backend: str,
) -> Any:
    """Full-completion SFT loss with per-row token normalization."""

    means, _ = masked_causal_logp_mean(
        logits=logits,
        input_ids=input_ids,
        token_mask=completion_mask,
        attention_mask=attention_mask,
        backend=backend,
    )
    return -means.mean()
