"""Hugging Face Trainer integration for DQS setting-5 post-training."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from transformers import Trainer

try:
    from .mpo_masking import (
        causal_prediction_positions,
        selected_causal_token_logps,
        selected_token_logp_mean,
    )
    from .mpo_objective import Setting5LossConfig, Setting5LossOutput, setting5_loss
except ImportError:  # Direct execution/tests with post_training on sys.path.
    from mpo_masking import (
        causal_prediction_positions,
        selected_causal_token_logps,
        selected_token_logp_mean,
    )
    from mpo_objective import Setting5LossConfig, Setting5LossOutput, setting5_loss


class SelectedLogitsUnsupported(RuntimeError):
    """Raised when a model ignores or cannot honor tensor logits_to_keep."""


@dataclass
class BatchLossResult:
    objective: Setting5LossOutput
    chosen_completion_token_counts: Any
    chosen_term_token_counts: Any
    rejected_term_token_counts: Any
    projection: str


def _accumulate_weighted_metrics(
    buffer: dict[str, list[float]],
    metrics: Mapping[str, float],
    *,
    weight: int,
) -> None:
    """Accumulate scalar batch means as row-weighted sums and counts."""

    if weight <= 0:
        raise ValueError(f"metric aggregation weight must be positive, got {weight}")
    for name, value in metrics.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"non-finite mPO metric {name}={numeric}")
        accumulator = buffer.setdefault(name, [0.0, 0.0])
        accumulator[0] += numeric * weight
        accumulator[1] += weight


def _distributed_weighted_metric_means(
    buffer: Mapping[str, list[float]],
    *,
    device: Any,
) -> dict[str, float]:
    """Reduce all custom metrics globally with one collective per log event."""

    import torch

    names = sorted(buffer)
    if not names:
        return {}
    packed = torch.tensor(
        [[float(buffer[name][0]), float(buffer[name][1])] for name in names],
        dtype=torch.float64,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    rows = packed.detach().cpu().tolist()
    means: dict[str, float] = {}
    for name, (weighted_sum, count) in zip(names, rows, strict=True):
        if not math.isfinite(weighted_sum) or not math.isfinite(count) or count <= 0:
            raise RuntimeError(
                f"invalid globally reduced mPO metric {name}: sum={weighted_sum}, count={count}"
            )
        means[name] = weighted_sum / count
    return means


def _extract_logits(outputs: Any) -> Any:
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, (tuple, list)) and outputs:
            logits = outputs[0]
    if logits is None or not hasattr(logits, "shape"):
        raise RuntimeError(
            "model forward did not return logits; for Unsloth load FastModel with return_logits=True"
        )
    if logits.ndim != 3:
        raise RuntimeError(f"expected [batch, sequence, vocab] logits, got shape={tuple(logits.shape)}")
    return logits


def _forward_logits(
    *,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    prediction_positions: Any | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }
    unwrapped = model
    while hasattr(unwrapped, "module") and getattr(unwrapped, "module") is not unwrapped:
        unwrapped = unwrapped.module
    model_config = getattr(unwrapped, "config", None)
    model_type = str(getattr(model_config, "model_type", "") or "").lower()
    if model_type.startswith("gemma4"):
        # Transformers 5.5 requires this during Gemma4 training. Zero denotes
        # ordinary causal text tokens; no vision/audio tokens enter this task.
        kwargs["mm_token_type_ids"] = input_ids.new_zeros(input_ids.shape)
    if prediction_positions is not None:
        kwargs["logits_to_keep"] = prediction_positions
    outputs = model(**kwargs)
    logits = _extract_logits(outputs)
    expected_length = input_ids.shape[1] if prediction_positions is None else prediction_positions.numel()
    if logits.shape[1] != expected_length:
        if prediction_positions is not None:
            raise SelectedLogitsUnsupported(
                "model did not honor tensor logits_to_keep: "
                f"returned sequence={logits.shape[1]}, expected selected={expected_length}"
            )
        raise RuntimeError(
            f"full-logits forward returned sequence={logits.shape[1]}, expected={expected_length}"
        )
    return logits


def _single_forward_preference_token_logps(
    *,
    model: Any,
    chosen_ids: Any,
    chosen_attention: Any,
    chosen_positions: Any,
    rejected_ids: Any,
    rejected_attention: Any,
    rejected_positions: Any,
    token_logp_backend: str,
) -> tuple[Any, Any, Any]:
    """Score both preference sides through one model forward and one CE node.

    Gemma4 E-series training uses activation checkpointing around compiled
    decoder layers.  Running chosen and rejected as separate forwards with
    different ``logits_to_keep`` lengths can warm a different compiled graph
    before the first graph is recomputed in backward.  Concatenating along the
    batch axis gives the model exactly one forward signature.  The union of
    selected sequence positions is only a projection optimization; each side's
    independent mask still determines which token log-probabilities contribute
    to its normalized loss.
    """

    import torch

    if chosen_ids.ndim != 2 or rejected_ids.ndim != 2:
        raise ValueError("chosen/rejected input tensors must be two-dimensional")
    if chosen_ids.shape != rejected_ids.shape:
        raise ValueError(
            "chosen/rejected input tensors must share one padded shape for a single "
            "concatenated preference forward"
        )
    if (
        chosen_attention.shape != chosen_ids.shape
        or rejected_attention.shape != rejected_ids.shape
    ):
        raise ValueError("chosen/rejected attention masks must match their input tensors")
    if chosen_ids.shape[0] <= 0:
        raise ValueError("preference forward requires at least one pair")
    for name, positions in (
        ("chosen_positions", chosen_positions),
        ("rejected_positions", rejected_positions),
    ):
        if positions.ndim != 1 or positions.numel() == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional tensor")

    prediction_positions = torch.unique(
        torch.cat((chosen_positions, rejected_positions), dim=0),
        sorted=True,
    )
    pair_batch_size = chosen_ids.shape[0]
    concatenated_ids = torch.cat((chosen_ids, rejected_ids), dim=0)
    concatenated_attention = torch.cat((chosen_attention, rejected_attention), dim=0)
    concatenated_logits = _forward_logits(
        model=model,
        input_ids=concatenated_ids,
        attention_mask=concatenated_attention,
        prediction_positions=prediction_positions,
    )
    concatenated_token_logps, target_positions = selected_causal_token_logps(
        logits=concatenated_logits,
        prediction_positions=prediction_positions,
        input_ids=concatenated_ids,
        backend=token_logp_backend,
    )
    expected_batch_size = pair_batch_size * 2
    if concatenated_token_logps.shape[0] != expected_batch_size:
        raise RuntimeError(
            "single preference forward returned an unexpected batch dimension: "
            f"{concatenated_token_logps.shape[0]} != {expected_batch_size}"
        )
    return (
        concatenated_token_logps[:pair_batch_size],
        concatenated_token_logps[pair_batch_size:],
        target_positions,
    )


def compute_setting5_batch_loss(
    *,
    model: Any,
    batch: Mapping[str, Any],
    config: Setting5LossConfig,
    projection: str,
    token_logp_backend: str,
) -> BatchLossResult:
    """Run one concatenated preference forward and compute setting-5 loss.

    ``projection='selected'`` asks the model to project logits only at the
    required causal positions: all chosen completion positions and only the
    rejected term positions. No full-logits execution path exists here.
    """

    if projection != "selected":
        raise ValueError("projection must be 'selected'; no full-logits path is implemented")

    chosen_ids = batch["chosen_input_ids"]
    chosen_attention = batch["chosen_attention_mask"]
    chosen_completion_mask = batch["chosen_completion_mask"]
    chosen_term_mask = batch["chosen_term_mask"]
    rejected_ids = batch["rejected_input_ids"]
    rejected_attention = batch["rejected_attention_mask"]
    rejected_term_mask = batch["rejected_term_mask"]

    chosen_positions = causal_prediction_positions(
        token_mask=chosen_completion_mask,
        attention_mask=chosen_attention,
    )
    rejected_positions = causal_prediction_positions(
        token_mask=rejected_term_mask,
        attention_mask=rejected_attention,
    )
    chosen_token_logps, rejected_token_logps, target_positions = (
        _single_forward_preference_token_logps(
            model=model,
            chosen_ids=chosen_ids,
            chosen_attention=chosen_attention,
            chosen_positions=chosen_positions,
            rejected_ids=rejected_ids,
            rejected_attention=rejected_attention,
            rejected_positions=rejected_positions,
            token_logp_backend=token_logp_backend,
        )
    )
    # One fused CE node owns the concatenated logits buffer. Chosen SFT, chosen
    # term mPO, and rejected term mPO remain separate masked reductions.
    chosen_completion_logps, chosen_completion_counts = selected_token_logp_mean(
        token_logps=chosen_token_logps,
        target_positions=target_positions,
        input_ids=chosen_ids,
        token_mask=chosen_completion_mask,
        attention_mask=chosen_attention,
    )
    chosen_term_logps, chosen_term_counts = selected_token_logp_mean(
        token_logps=chosen_token_logps,
        target_positions=target_positions,
        input_ids=chosen_ids,
        token_mask=chosen_term_mask,
        attention_mask=chosen_attention,
    )
    rejected_term_logps, rejected_term_counts = selected_token_logp_mean(
        token_logps=rejected_token_logps,
        target_positions=target_positions,
        input_ids=rejected_ids,
        token_mask=rejected_term_mask,
        attention_mask=rejected_attention,
    )

    objective = setting5_loss(
        chosen_completion_logps=chosen_completion_logps,
        chosen_term_logps=chosen_term_logps,
        rejected_term_logps=rejected_term_logps,
        config=config,
    )
    return BatchLossResult(
        objective=objective,
        chosen_completion_token_counts=chosen_completion_counts,
        chosen_term_token_counts=chosen_term_counts,
        rejected_term_token_counts=rejected_term_counts,
        projection=projection,
    )


class MPOTrainer(Trainer):
    """Trainer whose only objective is the isolated post-training setting 5."""

    def __init__(
        self,
        *args: Any,
        loss_config: Setting5LossConfig,
        logits_projection: str = "selected",
        token_logp_backend: str = "unsloth_fused",
        **kwargs: Any,
    ) -> None:
        if logits_projection != "selected":
            raise ValueError(
                "The production MPOTrainer requires logits_projection='selected'; "
                "there is no automatic or full-logits fallback."
            )
        self.loss_config = loss_config
        self.logits_projection = logits_projection
        if token_logp_backend != "unsloth_fused":
            raise ValueError(
                "The production MPOTrainer requires token_logp_backend='unsloth_fused'; "
                "there is no PyTorch CE fallback."
            )
        self.token_logp_backend = token_logp_backend
        self._resolved_projection = "selected"
        self._metric_buffer: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        super().__init__(*args, **kwargs)
        # Our loss is already normalized per row/token.  Trainer must not scale
        # it by num_items_in_batch as it may do for built-in token losses.
        self.model_accepts_loss_kwargs = False

    def _compute_with_projection(self, model: Any, inputs: Mapping[str, Any]) -> BatchLossResult:
        return compute_setting5_batch_loss(
            model=model,
            batch=inputs,
            config=self.loss_config,
            projection="selected",
            token_logp_backend=self.token_logp_backend,
        )

    def _store_metrics(self, result: BatchLossResult, *, train_mode: bool) -> None:
        import torch

        prefix = "train" if train_mode else "eval"
        output = result.objective
        target = output.margins.new_tensor(self.loss_config.target_margin)
        metrics = output.detached_metrics()
        metrics.update(
            {
                "loss/sft_weighted": output.sft_loss.detach() * self.loss_config.lambda_sft,
                "loss/mpo_weighted": output.mpo_loss.detach() * self.loss_config.lambda_mpo,
                "tokens/chosen_completion": result.chosen_completion_token_counts.detach().float().mean(),
                "tokens/chosen_term": result.chosen_term_token_counts.detach().float().mean(),
                "tokens/rejected_term": result.rejected_term_token_counts.detach().float().mean(),
                "margin/target": target,
                "margin/target_gap": output.margins.detach().mean() - target,
                "margin/mean_abs_error_to_target": (output.margins.detach() - target).abs().mean(),
                "margin/target_reached": (
                    output.margins.detach() >= self.loss_config.target_margin
                ).to(torch.float32).mean(),
                "mix/lambda_sft": output.loss.new_tensor(self.loss_config.lambda_sft),
                "mix/lambda_mpo": output.loss.new_tensor(self.loss_config.lambda_mpo),
                "projection/selected": output.loss.new_tensor(1.0 if result.projection == "selected" else 0.0),
            }
        )
        names = list(metrics)
        # One device synchronization per micro-batch instead of one per metric.
        values = torch.stack([metrics[name].detach().to(torch.float32).reshape(()) for name in names])
        numeric_metrics = {
            f"{prefix}/{name}": float(value)
            for name, value in zip(names, values.cpu().tolist(), strict=True)
        }
        _accumulate_weighted_metrics(
            self._metric_buffer,
            numeric_metrics,
            weight=int(output.margins.numel()),
        )

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        del num_items_in_batch
        # pair_id is audit metadata; it must never reach the model forward.
        inputs = {key: value for key, value in inputs.items() if key != "pair_id"}
        result = self._compute_with_projection(model, inputs)
        self._store_metrics(result, train_mode=bool(model.training))
        if return_outputs:
            # Do not return giant chosen/rejected logits to evaluation loops.
            return result.objective.loss, {"loss": result.objective.loss.detach()}
        return result.objective.loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        logs.update(
            _distributed_weighted_metric_means(
                self._metric_buffer,
                device=self.args.device,
            )
        )
        self._metric_buffer.clear()
        return super().log(logs, *args, **kwargs)
