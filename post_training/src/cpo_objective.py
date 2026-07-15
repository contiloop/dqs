"""Reference-free full-response CPO objective for Teacher-vs-Student pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class FullResponseCPOConfig:
    beta: float = 0.1
    cpo_alpha: float = 1.0
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("CPO beta must be positive")
        if self.cpo_alpha <= 0:
            raise ValueError("CPO chosen-NLL coefficient must be positive")
        if self.loss_type != "sigmoid":
            raise ValueError("this strict full-response CPO trainer implements sigmoid loss only")
        if self.label_smoothing != 0.0:
            raise ValueError("label smoothing is disabled for the strict CPO baseline")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FullResponseCPOConfig":
        return cls(
            beta=float(values.get("beta", 0.1)),
            cpo_alpha=float(values.get("cpo_alpha", 1.0)),
            loss_type=str(values.get("loss_type", "sigmoid")),
            label_smoothing=float(values.get("label_smoothing", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "preference_logprob_aggregation": "sum_over_completion_including_eos",
            "chosen_nll_normalization": "per_row_completion_token_mean",
            "objective": "-logsigmoid(beta * (sum_logp_teacher - sum_logp_student)) + cpo_alpha * NLL_teacher",
        }


@dataclass
class FullResponseCPOOutput:
    loss: Any
    preference_loss: Any
    chosen_nll_loss: Any
    chosen_mean_logps: Any
    rejected_mean_logps: Any
    chosen_sum_logps: Any
    rejected_sum_logps: Any
    margins: Any

    def detached_metrics(self) -> dict[str, Any]:
        import torch

        return {
            "loss/total": self.loss.detach(),
            "loss/preference_unweighted": self.preference_loss.detach(),
            "loss/chosen_nll_unweighted": self.chosen_nll_loss.detach(),
            "logp/chosen_mean": self.chosen_mean_logps.detach().mean(),
            "logp/rejected_mean": self.rejected_mean_logps.detach().mean(),
            "logp/chosen_sum": self.chosen_sum_logps.detach().mean(),
            "logp/rejected_sum": self.rejected_sum_logps.detach().mean(),
            "margin/sum_logp_mean": self.margins.detach().mean(),
            "margin/preference_accuracy": (self.margins.detach() > 0)
            .to(torch.float32)
            .mean(),
        }


def full_response_cpo_loss(
    *,
    chosen_mean_logps: Any,
    rejected_mean_logps: Any,
    chosen_token_counts: Any,
    rejected_token_counts: Any,
    config: FullResponseCPOConfig,
) -> FullResponseCPOOutput:
    import torch
    import torch.nn.functional as functional

    tensors = (
        chosen_mean_logps,
        rejected_mean_logps,
        chosen_token_counts,
        rejected_token_counts,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("full-response CPO expects 1D per-row tensors")
    if not all(tensor.shape == chosen_mean_logps.shape for tensor in tensors):
        raise ValueError("full-response CPO tensor shapes do not match")
    if chosen_mean_logps.numel() == 0:
        raise ValueError("full-response CPO received an empty batch")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("non-finite value entered full-response CPO")
    if torch.any(chosen_token_counts <= 0) or torch.any(rejected_token_counts <= 0):
        raise ValueError("full-response CPO completion masks must be non-empty")

    chosen_sum = chosen_mean_logps * chosen_token_counts.to(chosen_mean_logps.dtype)
    rejected_sum = rejected_mean_logps * rejected_token_counts.to(rejected_mean_logps.dtype)
    margins = chosen_sum - rejected_sum
    preference = -functional.logsigmoid(config.beta * margins).mean()
    chosen_nll = -chosen_mean_logps.mean()
    total = preference + config.cpo_alpha * chosen_nll
    return FullResponseCPOOutput(
        loss=total,
        preference_loss=preference,
        chosen_nll_loss=chosen_nll,
        chosen_mean_logps=chosen_mean_logps,
        rejected_mean_logps=rejected_mean_logps,
        chosen_sum_logps=chosen_sum,
        rejected_sum_logps=rejected_sum,
        margins=margins,
    )
