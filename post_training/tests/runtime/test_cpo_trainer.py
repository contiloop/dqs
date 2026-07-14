from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from cpo_objective import FullResponseCPOConfig, full_response_cpo_loss
from cpo_trainer import compute_cpo_batch_loss
from mpo_masking import MPOPreferenceCollator, masked_causal_logp_mean


class TinyCausalLM:
    def __init__(self, vocab_size: int = 13, hidden_size: int = 7) -> None:
        self.module = torch.nn.Module()
        self.module.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.module.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.calls: list[tuple[int, int]] = []

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
        self.calls.append(tuple(input_ids.shape))
        hidden = self.module.embedding(input_ids)
        if not isinstance(logits_to_keep, int):
            hidden = hidden.index_select(1, logits_to_keep)
        elif logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=self.module.projection(hidden))

    def parameters(self):
        return self.module.parameters()


def batch():
    return MPOPreferenceCollator(pad_token_id=0)(
        [
            {
                "pair_id": "a",
                "chosen_input_ids": [2, 3, 4, 5],
                "chosen_completion_mask": [0, 0, 1, 1],
                "chosen_term_mask": [0, 0, 1, 1],
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
                "rejected_term_mask": [0, 1, 1, 1, 1],
            },
        ]
    )


class FullResponseCPOTrainerTest(unittest.TestCase):
    def test_pair_uses_one_concatenated_model_forward(self) -> None:
        model = TinyCausalLM()
        full_batch = batch()
        compute_cpo_batch_loss(
            model=model,
            batch=full_batch,
            config=FullResponseCPOConfig(),
            projection="selected",
            token_logp_backend="torch",
        )

        self.assertEqual(model.calls, [(4, full_batch["chosen_input_ids"].shape[1])])

    def test_selected_projection_matches_full_logits(self) -> None:
        torch.manual_seed(17)
        model = TinyCausalLM()
        config = FullResponseCPOConfig()
        selected = compute_cpo_batch_loss(
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
        chosen_mean, chosen_count = masked_causal_logp_mean(
            logits=chosen_logits,
            input_ids=full_batch["chosen_input_ids"],
            token_mask=full_batch["chosen_completion_mask"],
            attention_mask=full_batch["chosen_attention_mask"],
            backend="torch",
        )
        rejected_mean, rejected_count = masked_causal_logp_mean(
            logits=rejected_logits,
            input_ids=full_batch["rejected_input_ids"],
            token_mask=full_batch["rejected_completion_mask"],
            attention_mask=full_batch["rejected_attention_mask"],
            backend="torch",
        )
        full = full_response_cpo_loss(
            chosen_mean_logps=chosen_mean,
            rejected_mean_logps=rejected_mean,
            chosen_token_counts=chosen_count,
            rejected_token_counts=rejected_count,
            config=config,
        )
        self.assertTrue(torch.allclose(selected.objective.loss, full.loss, atol=1e-6))
        self.assertEqual(selected.chosen_token_counts.tolist(), [2, 2])
        self.assertEqual(selected.rejected_token_counts.tolist(), [1, 4])

    def test_selected_projection_backpropagates(self) -> None:
        model = TinyCausalLM()
        result = compute_cpo_batch_loss(
            model=model,
            batch=batch(),
            config=FullResponseCPOConfig(),
            projection="selected",
            token_logp_backend="torch",
        )
        result.objective.loss.backward()
        total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(total, 0.0)

    def test_full_logits_path_is_forbidden(self) -> None:
        with self.assertRaisesRegex(ValueError, "no full-logits"):
            compute_cpo_batch_loss(
                model=TinyCausalLM(),
                batch=batch(),
                config=FullResponseCPOConfig(),
                projection="full",
                token_logp_backend="torch",
            )


if __name__ == "__main__":
    unittest.main()
