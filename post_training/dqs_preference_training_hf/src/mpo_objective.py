"""Paper-setting-5 objective for DQS terminology post-training.

The atomic losses in this module are deliberately unweighted.  Mixing happens
once, at the final line:

    lambda_sft * L_SFT(y+) + lambda_mpo * L_mPO(y+, y-)

This avoids accidentally applying the paper's alpha twice (its equations (4),
(6), and (7) contain redundant-looking alpha notation).  The paper preset is
``lambda_sft=10`` and ``lambda_mpo=1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Setting5LossConfig:
    """Configuration for paper setting 5: full-completion SFT plus mPO."""

    paper_setting: int = 5
    lambda_sft: float = 10.0
    lambda_mpo: float = 1.0
    preference_beta: float = 0.25
    smooth_l1_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.paper_setting != 5:
            raise ValueError(
                "This trainer implements paper setting 5 (SFT + mPO). "
                "Paper setting 6 additionally requires mSFT and full-sequence PO."
            )
        if self.lambda_sft <= 0 or self.lambda_mpo <= 0:
            raise ValueError("paper setting 5 requires both loss mixing coefficients to be positive")
        if self.preference_beta <= 0:
            raise ValueError("preference_beta must be positive")
        if self.smooth_l1_delta <= 0:
            raise ValueError("smooth_l1_delta must be positive")

    @property
    def target_margin(self) -> float:
        """IPO-style target margin from the paper, 1 / (2 * beta)."""

        return 1.0 / (2.0 * self.preference_beta)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Setting5LossConfig":
        return cls(
            paper_setting=int(values.get("paper_setting", 5)),
            lambda_sft=float(values.get("lambda_sft", 10.0)),
            lambda_mpo=float(values.get("lambda_mpo", 1.0)),
            preference_beta=float(values.get("preference_beta", 0.25)),
            smooth_l1_delta=float(values.get("smooth_l1_delta", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_margin"] = self.target_margin
        payload["objective"] = "lambda_sft * sft(y+) + lambda_mpo * mpo(y+, y-)"
        return payload


@dataclass
class Setting5LossOutput:
    loss: Any
    sft_loss: Any
    mpo_loss: Any
    chosen_completion_logps: Any
    chosen_term_logps: Any
    rejected_term_logps: Any
    margins: Any

    def detached_metrics(self) -> dict[str, Any]:
        import torch

        return {
            "loss/total": self.loss.detach(),
            "loss/sft_unweighted": self.sft_loss.detach(),
            "loss/mpo_unweighted": self.mpo_loss.detach(),
            "logp/chosen_completion": self.chosen_completion_logps.detach().mean(),
            "logp/chosen_term": self.chosen_term_logps.detach().mean(),
            "logp/rejected_term": self.rejected_term_logps.detach().mean(),
            "margin/mean": self.margins.detach().mean(),
            "margin/preference_accuracy": (self.margins.detach() > 0).to(torch.float32).mean(),
        }


def setting5_loss(
    *,
    chosen_completion_logps: Any,
    chosen_term_logps: Any,
    rejected_term_logps: Any,
    config: Setting5LossConfig,
) -> Setting5LossOutput:
    """Combine per-row normalized log-probabilities into setting-5 loss."""

    import torch
    import torch.nn.functional as functional

    tensors = (chosen_completion_logps, chosen_term_logps, rejected_term_logps)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("setting5_loss expects three 1D per-row log-probability tensors")
    if not (
        chosen_completion_logps.shape
        == chosen_term_logps.shape
        == rejected_term_logps.shape
    ):
        raise ValueError("chosen/rejected log-probability tensors must have matching batch shapes")
    if chosen_completion_logps.numel() == 0:
        raise ValueError("setting5_loss received an empty batch")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("non-finite log-probability entered setting5_loss")

    # Atomic SFT is unweighted.  It is the mean of the already per-row,
    # completion-token-normalized chosen log-probabilities.
    sft = -chosen_completion_logps.mean()

    margins = chosen_term_logps - rejected_term_logps
    target = torch.full_like(margins, config.target_margin)
    mpo_per_row = functional.smooth_l1_loss(
        margins,
        target,
        reduction="none",
        beta=config.smooth_l1_delta,
    )
    mpo = mpo_per_row.mean()

    # Apply each mixing coefficient exactly once here.
    total = config.lambda_sft * sft + config.lambda_mpo * mpo
    return Setting5LossOutput(
        loss=total,
        sft_loss=sft,
        mpo_loss=mpo,
        chosen_completion_logps=chosen_completion_logps,
        chosen_term_logps=chosen_term_logps,
        rejected_term_logps=rejected_term_logps,
        margins=margins,
    )
