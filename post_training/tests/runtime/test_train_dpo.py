from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from train_dpo import (
    _validate_length_contract,
    dpo_config_kwargs,
    require_smoke_receipt,
    validate_config,
)
from train_mpo import _output_dir


POST_TRAINING_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = POST_TRAINING_ROOT / "dqs_preference_training_hf"
CONFIG_ROOT = PACKAGE_ROOT / "configs"
CONTRACT_ROOT = PACKAGE_ROOT / "data" / "contracts"


class TrainDPOConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(
            (CONFIG_ROOT / "dpo.yaml").read_text(encoding="utf-8")
        )

    def test_default_config_is_strict_and_distinct(self) -> None:
        validate_config(self.config)
        dpo_output = _output_dir(self.config["run"], smoke_step=False)
        self.assertTrue(str(dpo_output).startswith(str(PACKAGE_ROOT.resolve())))
        cpo = yaml.safe_load(
            (CONFIG_ROOT / "cpo.yaml").read_text(encoding="utf-8")
        )
        mpo = yaml.safe_load(
            (CONFIG_ROOT / "mpo.yaml").read_text(encoding="utf-8")
        )
        self.assertNotEqual(self.config["run"]["id"], cpo["run"]["id"])
        self.assertNotEqual(self.config["run"]["id"], mpo["run"]["id"])
        self.assertNotEqual(self.config["run"]["output_dir"], cpo["run"]["output_dir"])
        self.assertNotEqual(self.config["run"]["output_dir"], mpo["run"]["output_dir"])

    def test_semantic_fallbacks_are_rejected(self) -> None:
        mutations = (
            ("run", "require_smoke_step_receipt", False),
            ("model", "require_final_stage_model", False),
            ("data", "source", "hf"),
            ("loss", "trainer", "DPOTrainer"),
            ("loss", "reference_free", True),
            ("loss", "rpo_alpha", 1.0),
            ("loss", "ld_alpha", 0.5),
            ("loss", "loss_weights", [1.0]),
            ("loss", "use_liger_loss", True),
            ("reference", "precompute_ref_log_probs", False),
            ("reference", "precompute_before_resume_restore", False),
            ("training", "load_in_4bit", True),
            ("training", "gradient_checkpointing", False),
            ("training", "unsloth_compile", "enabled"),
            ("training", "use_logits_to_keep", False),
            ("training", "disable_dropout", False),
            ("training", "padding_free", True),
            ("training", "eval_strategy", "steps"),
            ("training", "report_to", ["wandb"]),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                changed = deepcopy(self.config)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_no_truncation_contract_matches_data(self) -> None:
        contract = json.loads(
            (CONTRACT_ROOT / "dpo.json").read_text(
                encoding="utf-8"
            )
        )
        resolved = _validate_length_contract(
            dataset_contract=contract, training=self.config["training"]
        )
        self.assertEqual(resolved["max_observed_sequence_tokens"], 2825)
        self.assertEqual(resolved["max_prompt_length"], 1398)
        self.assertEqual(resolved["max_completion_length"], 1501)
        self.assertFalse(resolved["truncation_allowed"])
        self.assertEqual(math.ceil(contract["row_count"] / 128), 41)

        changed = deepcopy(self.config["training"])
        changed["max_completion_length"] = 1500
        with self.assertRaisesRegex(ValueError, "no-truncation"):
            _validate_length_contract(dataset_contract=contract, training=changed)

    def test_dpo_config_kwargs_keep_reference_and_memory_contract(self) -> None:
        with patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0"}, clear=False):
            kwargs = dpo_config_kwargs(
                run=self.config["run"],
                loss=self.config["loss"],
                reference=self.config["reference"],
                training=self.config["training"],
                output_dir=PACKAGE_ROOT / "outputs" / "test",
                pad_token="<pad>",
                smoke_step=False,
            )
        self.assertTrue(kwargs["precompute_ref_log_probs"])
        self.assertTrue(kwargs["use_logits_to_keep"])
        self.assertTrue(kwargs["gradient_checkpointing"])
        self.assertFalse(kwargs["reference_free"])
        self.assertIsNone(kwargs["rpo_alpha"])
        self.assertEqual(kwargs["loss_type"], ["sigmoid"])
        self.assertEqual(kwargs["learning_rate"], 5e-6)
        self.assertEqual(kwargs["warmup_steps"], 0.1)
        self.assertNotIn("warmup_ratio", kwargs)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 128)

    def test_smoke_receipt_must_cover_the_whole_smoke_batch(self) -> None:
        expected = {
            "sha256": "contract",
            "payload": {"training": {"smoke_sample_rows": 128}},
        }
        receipt = {
            "smoke_contract": {"sha256": "contract"},
            "resolved_trainer_class": "UnslothDPOTrainer",
            "forward_contract": {
                "selected_suffix_logits_enforced": True,
                "full_logits_fallback": False,
            },
            "reference_precompute": {
                "precomputed_before_trainer_train": True,
                "rows": 128,
            },
            "prepared_dataset": {"rows": 128},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            smoke_dir = Path(temp_dir) / "smoke_step"
            smoke_dir.mkdir()
            path = smoke_dir / "smoke_step_result.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            require_smoke_receipt(Path(temp_dir), expected)
            receipt["reference_precompute"]["rows"] = 127
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row count"):
                require_smoke_receipt(Path(temp_dir), expected)


if __name__ == "__main__":
    unittest.main()
