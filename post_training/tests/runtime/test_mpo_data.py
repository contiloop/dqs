from __future__ import annotations

import copy
import unittest

from mpo_data import validate_dataset


class _Rows(list):
    @property
    def column_names(self):
        return list(self[0]) if self else []


class DatasetContractTest(unittest.TestCase):
    def row(self):
        return {
            "pair_id": "pair-1",
            "schema_version": "dqs_mpo_token_masks_v1",
            "prompt_token_count": 2,
            "chosen_input_ids": [2, 10, 11, 1],
            "chosen_completion_mask": [0, 0, 1, 1],
            "chosen_term_mask": [0, 0, 1, 0],
            "rejected_input_ids": [2, 10, 12, 1],
            "rejected_completion_mask": [0, 0, 1, 1],
            "rejected_term_mask": [0, 0, 1, 0],
        }

    def test_semantic_checksum_detects_training_tensor_change(self) -> None:
        base_contract = {
            "schema_version": "dqs_mpo_token_masks_v1",
            "max_seq_length": 8,
            "row_count": 1,
        }
        row = self.row()
        initial = validate_dataset(_Rows([row]), split_name="train", contract=base_contract)
        pinned_contract = {
            **base_contract,
            "training_semantic_sha256": initial["training_semantic_sha256"],
        }
        validate_dataset(_Rows([row]), split_name="train", contract=pinned_contract)

        changed = copy.deepcopy(row)
        changed["rejected_input_ids"][2] = 13
        with self.assertRaisesRegex(ValueError, "semantic checksum mismatch"):
            validate_dataset(_Rows([changed]), split_name="train", contract=pinned_contract)


if __name__ == "__main__":
    unittest.main()
