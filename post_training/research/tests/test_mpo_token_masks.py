from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mpo_token_masks import TokenMaskRejected, build_tokenized_pair
from mpo_masking import (
    MPOPreferenceCollator,
    completion_sft_loss,
    masked_causal_logp_mean,
    shifted_token_logps,
)


class TableTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self, table: dict[str, tuple[list[int], list[tuple[int, int]]]], tokens: dict[int, str]) -> None:
        self.table = table
        self.tokens = tokens

    def __call__(self, text: str | None = None, **kwargs: Any) -> dict[str, Any]:
        assert text is not None
        ids, offsets = self.table[text]
        result: dict[str, Any] = {"input_ids": list(ids)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = list(offsets)
        return result

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.tokens[value] for value in ids]


def base_row() -> dict[str, Any]:
    return {
        "pair_id": "pair_1",
        "subset": "subset_000",
        "teacher_label": "minor",
        "term_annotation_count": 1,
        "replacement_span_count": 1,
        "has_quality_warnings": False,
        "quality_flags": [],
        "prompt": "P",
        "chosen": "국제 송금",
        "rejected": "transfer",
        "chosen_term_char_spans": [[0, 5]],
        "rejected_term_char_spans": [[0, 8]],
    }


class TokenMaskBuilderTest(unittest.TestCase):
    def test_chosen_and_rejected_masks_are_independent(self) -> None:
        tokenizer = TableTokenizer(
            {
                "P": ([10], [(0, 1)]),
                "국제 송금<eos>": (
                    [11, 12, 13, 14, 1],
                    [(0, 1), (1, 2), (2, 4), (4, 5), (5, 10)],
                ),
                "transfer<eos>": ([20, 1], [(0, 8), (8, 13)]),
            },
            {1: "<eos>", 10: "P", 11: "국", 12: "제", 13: "▁송", 14: "금", 20: "transfer"},
        )
        pair = build_tokenized_pair(
            tokenizer=tokenizer,
            row=base_row(),
            max_seq_length=32,
            append_eos=True,
        )
        self.assertEqual(pair["chosen_term_token_count"], 4)
        self.assertEqual(pair["rejected_term_token_count"], 1)
        self.assertTrue(pair["term_token_lengths_differ"])
        self.assertEqual(pair["chosen_completion_mask"], [0, 1, 1, 1, 1, 1])
        self.assertEqual(pair["chosen_term_mask"], [0, 1, 1, 1, 1, 0])
        self.assertEqual(pair["rejected_completion_mask"], [0, 1, 1])
        self.assertEqual(pair["rejected_term_mask"], [0, 1, 0])
        self.assertEqual(pair["chosen_term_prediction_indices"], [0, 1, 2, 3])
        self.assertEqual(pair["rejected_term_prediction_indices"], [0])

    def test_non_whitespace_boundary_crossing_is_rejected(self) -> None:
        row = base_row()
        row.update(
            {
                "chosen": "정답은",
                "rejected": "오답은",
                "chosen_term_char_spans": [[0, 2]],
                "rejected_term_char_spans": [[0, 2]],
            }
        )
        tokenizer = TableTokenizer(
            {
                "P": ([10], [(0, 1)]),
                "정답은<eos>": ([30, 1], [(0, 3), (3, 8)]),
                "오답은<eos>": ([31, 1], [(0, 3), (3, 8)]),
            },
            {1: "<eos>", 10: "P", 30: "정답은", 31: "오답은"},
        )
        with self.assertRaisesRegex(TokenMaskRejected, "chosen_token_crosses_term_boundary"):
            build_tokenized_pair(
                tokenizer=tokenizer,
                row=row,
                max_seq_length=32,
                append_eos=True,
            )

    def test_whitespace_only_boundary_overhang_is_allowed(self) -> None:
        row = base_row()
        row.update(
            {
                "chosen": "앞 정답",
                "rejected": "앞 오답",
                "chosen_term_char_spans": [[2, 4]],
                "rejected_term_char_spans": [[2, 4]],
            }
        )
        tokenizer = TableTokenizer(
            {
                "P": ([10], [(0, 1)]),
                "앞 정답<eos>": ([40, 41, 1], [(0, 1), (1, 4), (4, 9)]),
                "앞 오답<eos>": ([40, 42, 1], [(0, 1), (1, 4), (4, 9)]),
            },
            {1: "<eos>", 10: "P", 40: "앞", 41: "▁정답", 42: "▁오답"},
        )
        pair = build_tokenized_pair(
            tokenizer=tokenizer,
            row=row,
            max_seq_length=32,
            append_eos=True,
        )
        self.assertEqual(pair["chosen_term_token_count"], 1)
        self.assertTrue(pair["chosen_term_alignments"][0]["boundary_whitespace_only"])

    def test_sequence_is_rejected_instead_of_truncated(self) -> None:
        tokenizer = TableTokenizer(
            {
                "P": ([10], [(0, 1)]),
                "국제 송금<eos>": (
                    [11, 12, 13, 14, 1],
                    [(0, 1), (1, 2), (2, 4), (4, 5), (5, 10)],
                ),
                "transfer<eos>": ([20, 1], [(0, 8), (8, 13)]),
            },
            {1: "<eos>", 10: "P", 11: "국", 12: "제", 13: "▁송", 14: "금", 20: "transfer"},
        )
        with self.assertRaisesRegex(TokenMaskRejected, "chosen_sequence_too_long"):
            build_tokenized_pair(
                tokenizer=tokenizer,
                row=base_row(),
                max_seq_length=5,
                append_eos=True,
            )

    def test_completion_strip_matches_sft_and_shifts_char_spans(self) -> None:
        row = base_row()
        row.update(
            {
                "chosen": "  정답",
                "rejected": "  오답",
                "chosen_term_char_spans": [[2, 4]],
                "rejected_term_char_spans": [[2, 4]],
            }
        )
        tokenizer = TableTokenizer(
            {
                "P": ([10], [(0, 1)]),
                "정답<eos>": ([50, 1], [(0, 2), (2, 7)]),
                "오답<eos>": ([51, 1], [(0, 2), (2, 7)]),
            },
            {1: "<eos>", 10: "P", 50: "정답", 51: "오답"},
        )
        pair = build_tokenized_pair(
            tokenizer=tokenizer,
            row=row,
            max_seq_length=32,
            append_eos=True,
        )
        self.assertEqual(pair["chosen"], "정답")
        self.assertEqual(pair["rejected"], "오답")
        self.assertEqual(pair["chosen_term_char_spans"], [[0, 2]])
        self.assertEqual(pair["chosen_outer_whitespace_stripped_chars"], [2, 0])

    def test_tokenizer_equivalent_chosen_and_rejected_terms_are_rejected(self) -> None:
        row = base_row()
        row.update(
            {
                "chosen": "Ａ",
                "rejected": "A",
                "chosen_term_char_spans": [[0, 1]],
                "rejected_term_char_spans": [[0, 1]],
            }
        )
        tokenizer = TableTokenizer(
            {
                "P": ([10], [(0, 1)]),
                "Ａ<eos>": ([60, 1], [(0, 1), (1, 6)]),
                "A<eos>": ([60, 1], [(0, 1), (1, 6)]),
            },
            {1: "<eos>", 10: "P", 60: "A"},
        )
        with self.assertRaisesRegex(TokenMaskRejected, "identical_chosen_rejected_term_token_ids"):
            build_tokenized_pair(
                tokenizer=tokenizer,
                row=row,
                max_seq_length=32,
                append_eos=True,
            )


class MaskedLossTest(unittest.TestCase):
    def test_collator_zero_pads_attention_and_loss_masks(self) -> None:
        collator = MPOPreferenceCollator(pad_token_id=0)
        batch = collator(
            [
                {
                    "pair_id": "a",
                    "chosen_input_ids": [2, 3, 4],
                    "chosen_completion_mask": [0, 1, 1],
                    "chosen_term_mask": [0, 0, 1],
                    "rejected_input_ids": [2, 5],
                    "rejected_completion_mask": [0, 1],
                    "rejected_term_mask": [0, 1],
                },
                {
                    "pair_id": "b",
                    "chosen_input_ids": [2, 6],
                    "chosen_completion_mask": [0, 1],
                    "chosen_term_mask": [0, 1],
                    "rejected_input_ids": [2, 7, 8],
                    "rejected_completion_mask": [0, 1, 1],
                    "rejected_term_mask": [0, 1, 1],
                },
            ]
        )
        self.assertEqual(batch["chosen_attention_mask"].tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(batch["chosen_completion_mask"].tolist(), [[False, True, True], [False, True, False]])
        self.assertEqual(batch["chosen_term_mask"].tolist(), [[False, False, True], [False, True, False]])
        self.assertEqual(batch["rejected_attention_mask"].tolist(), [[True, True, False], [True, True, True]])
        self.assertEqual(batch["chosen_input_ids"].shape, batch["rejected_input_ids"].shape)

    def test_causal_shift_and_per_row_normalization(self) -> None:
        import torch
        import torch.nn.functional as functional

        input_ids = torch.tensor([[4, 1, 2, 0], [4, 3, 0, 0]])
        attention = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        completion = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
        term = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
        logits = torch.zeros((2, 4, 5), dtype=torch.float32)
        logits[0, 0, 1] = 2.0
        logits[0, 1, 2] = 4.0
        logits[0, 2, 2] = -9.0  # Must not score token 2; causal shift uses position 1.
        logits[1, 0, 3] = 3.0

        all_logps = shifted_token_logps(logits, input_ids, backend="torch")
        expected_term_0 = functional.log_softmax(logits[0, 1], dim=-1)[2]
        expected_term_1 = functional.log_softmax(logits[1, 0], dim=-1)[3]
        self.assertTrue(torch.allclose(all_logps[0, 1], expected_term_0))

        term_means, term_counts = masked_causal_logp_mean(
            logits=logits,
            input_ids=input_ids,
            token_mask=term,
            attention_mask=attention,
            backend="torch",
        )
        self.assertEqual(term_counts.tolist(), [1, 1])
        self.assertTrue(torch.allclose(term_means, torch.stack([expected_term_0, expected_term_1])))

        completion_means, completion_counts = masked_causal_logp_mean(
            logits=logits,
            input_ids=input_ids,
            token_mask=completion,
            attention_mask=attention,
            backend="torch",
        )
        self.assertEqual(completion_counts.tolist(), [2, 1])
        expected_loss = -completion_means.mean()
        actual_loss = completion_sft_loss(
            logits=logits,
            input_ids=input_ids,
            completion_mask=completion,
            attention_mask=attention,
            backend="torch",
        )
        self.assertTrue(torch.allclose(actual_loss, expected_loss))

    def test_empty_shifted_mask_is_rejected(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "empty shifted loss mask"):
            masked_causal_logp_mean(
                logits=torch.zeros((1, 3, 5)),
                input_ids=torch.tensor([[2, 3, 0]]),
                token_mask=torch.tensor([[0, 0, 0]], dtype=torch.bool),
                attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
                backend="torch",
            )


if __name__ == "__main__":
    unittest.main()
