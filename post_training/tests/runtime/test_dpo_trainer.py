from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from dpo_trainer import (
    DPOForwardContractMonitor,
    precompute_reference_logps_before_policy_restore,
    require_unsloth_dpo_patch,
    validate_dpo_trainer_contract,
    validate_prepared_dpo_dataset,
)


class _Tokenizer:
    eos_token_id = 99

    _ids = {"P": [10, 11], "Teacher": [20, 21], "Student": [30]}

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("DPO validation must not add tokenizer special tokens")
        return {"input_ids": list(self._ids[text])}


class _Rows(list):
    @property
    def column_names(self):
        return list(self[0]) if self else []


class _ReferenceRows(_Rows):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)

    def add_column(self, *, name, column):
        return _ReferenceRows(
            [{**row, name: float(value)} for row, value in zip(self, column, strict=True)]
        )


class _PatchedCollator:
    _unsloth_vision_keys_patch = True


class UnslothDPOTrainer:
    def __init__(self):
        self.precompute_ref_log_probs = True
        self.ref_model = None
        self.reference_free = False
        self.is_peft_model = False
        self.is_encoder_decoder = False
        self.loss_type = ["sigmoid"]
        self.label_smoothing = 0.0
        self.f_divergence_type = "reverse_kl"
        self.use_weighting = False
        self.aux_loss_enabled = False
        self.processing_class = SimpleNamespace(tokenizer=object())
        self.is_vision_model = True
        self.data_collator = _PatchedCollator()
        self.args = SimpleNamespace(
            use_logits_to_keep=True,
            gradient_checkpointing=True,
            disable_dropout=True,
            padding_free=False,
            use_liger_loss=False,
            rpo_alpha=None,
            loss_weights=None,
            ld_alpha=None,
            sync_ref_model=False,
        )
        self.optimizer = None
        self._precomputed_train_ref_log_probs = False
        self.train_dataset = _ReferenceRows([{"row": 0}, {"row": 1}])

    def get_train_dataloader(self):
        self.train_dataset = self.train_dataset.add_column(
            name="ref_chosen_logps", column=[-1.0, -2.0]
        )
        self.train_dataset = self.train_dataset.add_column(
            name="ref_rejected_logps", column=[-3.0, -4.0]
        )
        self._precomputed_train_ref_log_probs = True
        return object()


class _GoodModel(torch.nn.Module):
    def forward(self, *, input_ids, logits_to_keep):
        keep = min(int(logits_to_keep), int(input_ids.shape[1]))
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], keep, 7))


class _BadModel(torch.nn.Module):
    def forward(self, *, input_ids, logits_to_keep):
        del logits_to_keep
        return SimpleNamespace(
            logits=torch.zeros(input_ids.shape[0], input_ids.shape[1], 7)
        )


class DPOTrainerContractTest(unittest.TestCase):
    def test_only_unsloth_patched_classes_are_accepted(self) -> None:
        UnslothDPOConfig = type("UnslothDPOConfig", (), {})
        require_unsloth_dpo_patch(UnslothDPOTrainer, UnslothDPOConfig)
        with self.assertRaisesRegex(RuntimeError, "did not patch"):
            require_unsloth_dpo_patch(type("DPOTrainer", (), {}), UnslothDPOConfig)

    def test_prepared_dataset_is_exact_and_text_only(self) -> None:
        raw = _Rows([{"prompt": "P", "chosen": "Teacher", "rejected": "Student"}])
        prepared = _Rows(
            [
                {
                    "prompt_input_ids": [10, 11],
                    "chosen_input_ids": [20, 21, 99],
                    "rejected_input_ids": [30, 99],
                    "mm_token_type_ids": [0, 0],
                }
            ]
        )
        summary = validate_prepared_dpo_dataset(
            raw_dataset=raw, prepared_dataset=prepared, tokenizer=_Tokenizer()
        )
        self.assertTrue(summary["exact_text_tokenization"])
        self.assertEqual(summary["rows"], 1)

        prepared[0]["chosen_input_ids"][0] = 404
        with self.assertRaisesRegex(ValueError, "tokenization drift"):
            validate_prepared_dpo_dataset(
                raw_dataset=raw, prepared_dataset=prepared, tokenizer=_Tokenizer()
            )

    def test_visual_tensors_and_missing_mm_mask_are_rejected(self) -> None:
        raw = _Rows([{"prompt": "P", "chosen": "Teacher", "rejected": "Student"}])
        base = {
            "prompt_input_ids": [10, 11],
            "chosen_input_ids": [20, 21, 99],
            "rejected_input_ids": [30, 99],
        }
        with self.assertRaisesRegex(ValueError, "missing token columns"):
            validate_prepared_dpo_dataset(
                raw_dataset=raw,
                prepared_dataset=_Rows([dict(base)]),
                tokenizer=_Tokenizer(),
            )
        with self.assertRaisesRegex(ValueError, "visual tensors"):
            validate_prepared_dpo_dataset(
                raw_dataset=raw,
                prepared_dataset=_Rows(
                    [{**base, "mm_token_type_ids": [0, 0], "pixel_values": [[1.0]]}]
                ),
                tokenizer=_Tokenizer(),
            )

    def test_runtime_contract_and_reference_precompute(self) -> None:
        trainer = UnslothDPOTrainer()
        validate_dpo_trainer_contract(trainer)
        summary = precompute_reference_logps_before_policy_restore(trainer)
        self.assertEqual(summary["rows"], 2)
        self.assertTrue(summary["precomputed_before_trainer_train"])
        self.assertEqual(len(summary["float32_sha256"]), 64)
        with self.assertRaisesRegex(RuntimeError, "already precomputed"):
            precompute_reference_logps_before_policy_restore(trainer)

    def test_forward_monitor_proves_selected_suffix(self) -> None:
        model = _GoodModel()
        monitor = DPOForwardContractMonitor()
        monitor.install(model)
        model(input_ids=torch.ones(2, 5, dtype=torch.long), logits_to_keep=3)
        summary = monitor.summary()
        monitor.remove()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["max_logits_to_keep"], 3)
        self.assertFalse(summary["full_logits_fallback"])

    def test_forward_monitor_rejects_ignored_projection(self) -> None:
        model = _BadModel()
        monitor = DPOForwardContractMonitor()
        monitor.install(model)
        with self.assertRaisesRegex(RuntimeError, "ignored logits_to_keep"):
            model(input_ids=torch.ones(1, 5, dtype=torch.long), logits_to_keep=2)
        monitor.remove()


if __name__ == "__main__":
    unittest.main()
