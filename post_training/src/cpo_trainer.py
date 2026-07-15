"""Selected-logit full-response CPO Trainer for text-only Gemma4."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from transformers import Trainer

try:
    from .cpo_objective import (
        FullResponseCPOConfig,
        FullResponseCPOOutput,
        full_response_cpo_loss,
    )
    from .mpo_masking import (
        causal_prediction_positions,
        selected_token_logp_mean,
    )
    from .mpo_trainer import (
        _accumulate_weighted_metrics,
        _distributed_weighted_metric_means,
        _single_forward_preference_token_logps,
    )
except ImportError:
    from cpo_objective import (
        FullResponseCPOConfig,
        FullResponseCPOOutput,
        full_response_cpo_loss,
    )
    from mpo_masking import (
        causal_prediction_positions,
        selected_token_logp_mean,
    )
    from mpo_trainer import (
        _accumulate_weighted_metrics,
        _distributed_weighted_metric_means,
        _single_forward_preference_token_logps,
    )


@dataclass
class CPOBatchLossResult:
    objective: FullResponseCPOOutput
    chosen_token_counts: Any
    rejected_token_counts: Any


def compute_cpo_batch_loss(
    *,
    model: Any,
    batch: Mapping[str, Any],
    config: FullResponseCPOConfig,
    projection: str,
    token_logp_backend: str,
) -> CPOBatchLossResult:
    if projection != "selected":
        raise ValueError("full-response CPO has no full-logits execution path")
    chosen_ids = batch["chosen_input_ids"]
    chosen_attention = batch["chosen_attention_mask"]
    chosen_mask = batch["chosen_completion_mask"]
    rejected_ids = batch["rejected_input_ids"]
    rejected_attention = batch["rejected_attention_mask"]
    rejected_mask = batch["rejected_completion_mask"]

    chosen_positions = causal_prediction_positions(
        token_mask=chosen_mask, attention_mask=chosen_attention
    )
    rejected_positions = causal_prediction_positions(
        token_mask=rejected_mask, attention_mask=rejected_attention
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
    chosen_means, chosen_counts = selected_token_logp_mean(
        token_logps=chosen_token_logps,
        target_positions=target_positions,
        input_ids=chosen_ids,
        token_mask=chosen_mask,
        attention_mask=chosen_attention,
    )
    rejected_means, rejected_counts = selected_token_logp_mean(
        token_logps=rejected_token_logps,
        target_positions=target_positions,
        input_ids=rejected_ids,
        token_mask=rejected_mask,
        attention_mask=rejected_attention,
    )
    objective = full_response_cpo_loss(
        chosen_mean_logps=chosen_means,
        rejected_mean_logps=rejected_means,
        chosen_token_counts=chosen_counts,
        rejected_token_counts=rejected_counts,
        config=config,
    )
    return CPOBatchLossResult(
        objective=objective,
        chosen_token_counts=chosen_counts,
        rejected_token_counts=rejected_counts,
    )


class FullResponseCPOTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        loss_config: FullResponseCPOConfig,
        logits_projection: str = "selected",
        token_logp_backend: str = "unsloth_fused",
        **kwargs: Any,
    ) -> None:
        if logits_projection != "selected":
            raise ValueError("CPO requires selected logits; full-logits fallback is forbidden")
        if token_logp_backend != "unsloth_fused":
            raise ValueError("CPO requires Unsloth fused token log-probs; fallback is forbidden")
        self.loss_config = loss_config
        self.logits_projection = logits_projection
        self.token_logp_backend = token_logp_backend
        self._metric_buffer: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        del num_items_in_batch
        inputs = {key: value for key, value in inputs.items() if key != "pair_id"}
        result = compute_cpo_batch_loss(
            model=model,
            batch=inputs,
            config=self.loss_config,
            projection="selected",
            token_logp_backend="unsloth_fused",
        )
        output = result.objective
        metrics = output.detached_metrics()
        metrics.update(
            {
                "loss/chosen_nll_weighted": output.chosen_nll_loss.detach()
                * self.loss_config.cpo_alpha,
                "tokens/chosen_completion": result.chosen_token_counts.detach()
                .float()
                .mean(),
                "tokens/rejected_completion": result.rejected_token_counts.detach()
                .float()
                .mean(),
                "mix/beta": output.loss.new_tensor(self.loss_config.beta),
                "mix/cpo_alpha": output.loss.new_tensor(self.loss_config.cpo_alpha),
                "projection/selected": output.loss.new_tensor(1.0),
            }
        )
        names = list(metrics)
        import torch

        values = torch.stack(
            [metrics[name].detach().to(torch.float32).reshape(()) for name in names]
        )
        numeric = {
            f"{'train' if model.training else 'eval'}/{name}": float(value)
            for name, value in zip(names, values.cpu().tolist(), strict=True)
        }
        _accumulate_weighted_metrics(
            self._metric_buffer,
            numeric,
            weight=int(output.margins.numel()),
        )
        if return_outputs:
            return output.loss, {"loss": output.loss.detach()}
        return output.loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        logs.update(
            _distributed_weighted_metric_means(
                self._metric_buffer,
                device=self.args.device,
            )
        )
        self._metric_buffer.clear()
        return super().log(logs, *args, **kwargs)
