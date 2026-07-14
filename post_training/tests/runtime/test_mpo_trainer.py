from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpo_masking
from mpo_masking import MPOPreferenceCollator, masked_causal_logp_mean
from mpo_objective import Setting5LossConfig, setting5_loss
from mpo_trainer import (
    SelectedLogitsUnsupported,
    _accumulate_weighted_metrics,
    _distributed_weighted_metric_means,
    compute_setting5_batch_loss,
)


class TinyCausalLM:
    """Small differentiable model implementing tensor logits_to_keep."""

    def __init__(self, vocab_size: int = 13, hidden_size: int = 7) -> None:
        import torch

        self.module = torch.nn.Module()
        self.module.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.module.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache=False,
        return_dict=True,
        logits_to_keep=0,
    ):
        del attention_mask, use_cache, return_dict
        hidden = self.module.embedding(input_ids)
        if not isinstance(logits_to_keep, int):
            hidden = hidden.index_select(1, logits_to_keep)
        elif logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=self.module.projection(hidden))

    def parameters(self):
        return self.module.parameters()


class IgnoresSelectedPositions(TinyCausalLM):
    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache=False,
        return_dict=True,
        logits_to_keep=0,
    ):
        del attention_mask, use_cache, return_dict, logits_to_keep
        hidden = self.module.embedding(input_ids)
        return SimpleNamespace(logits=self.module.projection(hidden))


def batch():
    collator = MPOPreferenceCollator(pad_token_id=0)
    return collator(
        [
            {
                "pair_id": "a",
                "chosen_input_ids": [2, 3, 4, 5],
                "chosen_completion_mask": [0, 0, 1, 1],
                "chosen_term_mask": [0, 0, 1, 0],
                "rejected_input_ids": [2, 3, 6],
                "rejected_completion_mask": [0, 0, 1],
                "rejected_term_mask": [0, 0, 1],
            },
            {
                "pair_id": "b",
                "chosen_input_ids": [2, 7, 8],
                "chosen_completion_mask": [0, 1, 1],
                "chosen_term_mask": [0, 1, 1],
                "rejected_input_ids": [2, 9, 10, 11, 12],
                "rejected_completion_mask": [0, 1, 1, 1, 1],
                "rejected_term_mask": [0, 1, 1, 1, 0],
            },
        ]
    )


class TrainerLossTest(unittest.TestCase):
    def test_custom_metrics_are_row_weighted_across_micro_batches(self) -> None:
        buffer: dict[str, list[float]] = {}
        _accumulate_weighted_metrics(buffer, {"train/loss/total": 3.0}, weight=2)
        _accumulate_weighted_metrics(buffer, {"train/loss/total": 9.0}, weight=1)
        means = _distributed_weighted_metric_means(buffer, device="cpu")
        self.assertEqual(buffer["train/loss/total"], [15.0, 3.0])
        self.assertAlmostEqual(means["train/loss/total"], 5.0)

    def test_custom_metrics_use_one_global_sum_count_reduction(self) -> None:
        import torch

        buffer = {"train/margin/mean": [4.0, 2.0]}

        def add_other_rank(packed, *, op):
            self.assertIs(op, torch.distributed.ReduceOp.SUM)
            packed.add_(packed.new_tensor([[9.0, 3.0]]))

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce", side_effect=add_other_rank) as reduce,
        ):
            means = _distributed_weighted_metric_means(buffer, device="cpu")
        self.assertEqual(reduce.call_count, 1)
        self.assertAlmostEqual(means["train/margin/mean"], 13.0 / 5.0)

    def test_selected_projection_matches_full_logits(self) -> None:
        import torch

        torch.manual_seed(7)
        model = TinyCausalLM()
        config = Setting5LossConfig()
        selected = compute_setting5_batch_loss(
            model=model,
            batch=batch(),
            config=config,
            projection="selected",
            token_logp_backend="torch",
        )
        full_batch = batch()
        chosen_logits = model(
            input_ids=full_batch["chosen_input_ids"],
            attention_mask=full_batch["chosen_attention_mask"],
        ).logits
        rejected_logits = model(
            input_ids=full_batch["rejected_input_ids"],
            attention_mask=full_batch["rejected_attention_mask"],
        ).logits
        chosen_completion, _ = masked_causal_logp_mean(
            logits=chosen_logits,
            input_ids=full_batch["chosen_input_ids"],
            token_mask=full_batch["chosen_completion_mask"],
            attention_mask=full_batch["chosen_attention_mask"],
            backend="torch",
        )
        chosen_term, _ = masked_causal_logp_mean(
            logits=chosen_logits,
            input_ids=full_batch["chosen_input_ids"],
            token_mask=full_batch["chosen_term_mask"],
            attention_mask=full_batch["chosen_attention_mask"],
            backend="torch",
        )
        rejected_term, _ = masked_causal_logp_mean(
            logits=rejected_logits,
            input_ids=full_batch["rejected_input_ids"],
            token_mask=full_batch["rejected_term_mask"],
            attention_mask=full_batch["rejected_attention_mask"],
            backend="torch",
        )
        full = setting5_loss(
            chosen_completion_logps=chosen_completion,
            chosen_term_logps=chosen_term,
            rejected_term_logps=rejected_term,
            config=config,
        )
        self.assertTrue(torch.allclose(selected.objective.loss, full.loss, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                selected.objective.chosen_completion_logps,
                full.chosen_completion_logps,
                atol=1e-6,
            )
        )
        self.assertEqual(selected.chosen_term_token_counts.tolist(), [1, 2])
        self.assertEqual(selected.rejected_term_token_counts.tolist(), [1, 3])

    def test_selected_projection_backpropagates(self) -> None:
        import torch

        torch.manual_seed(11)
        model = TinyCausalLM()
        result = compute_setting5_batch_loss(
            model=model,
            batch=batch(),
            config=Setting5LossConfig(),
            projection="selected",
            token_logp_backend="torch",
        )
        result.objective.loss.backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_each_forward_uses_exactly_one_token_ce_node(self) -> None:
        model = TinyCausalLM()
        with patch.object(
            mpo_masking,
            "_token_logps",
            wraps=mpo_masking._token_logps,
        ) as token_scorer:
            result = compute_setting5_batch_loss(
                model=model,
                batch=batch(),
                config=Setting5LossConfig(),
                projection="selected",
                token_logp_backend="torch",
            )
            result.objective.loss.backward()

        # One node for chosen, shared by SFT and mPO; one for rejected mPO.
        self.assertEqual(token_scorer.call_count, 2)

    def test_ignored_selected_projection_hard_fails(self) -> None:
        with self.assertRaisesRegex(SelectedLogitsUnsupported, "did not honor"):
            compute_setting5_batch_loss(
                model=IgnoresSelectedPositions(),
                batch=batch(),
                config=Setting5LossConfig(),
                projection="selected",
                token_logp_backend="torch",
            )

    def test_full_logits_path_is_not_implemented(self) -> None:
        with self.assertRaisesRegex(ValueError, "no full-logits path"):
            compute_setting5_batch_loss(
                model=TinyCausalLM(),
                batch=batch(),
                config=Setting5LossConfig(),
                projection="full",
                token_logp_backend="torch",
            )


if __name__ == "__main__":
    unittest.main()
