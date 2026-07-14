from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpo_model import (
    _configure_unsloth_compile_contract,
    prepare_full_finetuning_parameters,
)


def _tiny_gemma4(*, tied: bool) -> Any:
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="gemma4",
                num_hidden_layers=0,
                num_kv_shared_layers=0,
            )
            self.embed_tokens = nn.Embedding(7, 4)
            self.body = nn.Linear(4, 4)
            self.visual = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 7, bias=False)
            if tied:
                self.lm_head.weight = self.embed_tokens.weight

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    return Model()


class FullFinetuningParameterTest(unittest.TestCase):
    def test_compile_disabled_contract_is_applied_before_unsloth_import(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _configure_unsloth_compile_contract({"unsloth_compile": "disabled"})
            self.assertEqual(os.environ["UNSLOTH_COMPILE_DISABLE"], "1")

    def test_compile_contract_has_no_enabled_or_late_import_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be disabled"):
            _configure_unsloth_compile_contract({"unsloth_compile": "enabled"})
        with patch.dict(sys.modules, {"unsloth": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "after Unsloth import"):
                _configure_unsloth_compile_contract({"unsloth_compile": "disabled"})

    def test_default_full_text_training_keeps_embeddings_trainable(self) -> None:
        model = _tiny_gemma4(tied=True)
        summary = prepare_full_finetuning_parameters(model, freeze_embeddings=False)

        self.assertTrue(model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.body.weight.requires_grad)
        self.assertFalse(model.visual.weight.requires_grad)
        self.assertTrue(summary["full_finetuning"])
        self.assertFalse(summary["freeze_embeddings"])
        self.assertEqual(summary["frozen_embedding_parameter_count"], 0)
        self.assertTrue(summary["input_embedding_output_weight_tied"])

    def test_freeze_embeddings_also_freezes_a_tied_lm_head(self) -> None:
        model = _tiny_gemma4(tied=True)
        summary = prepare_full_finetuning_parameters(model, freeze_embeddings=True)

        self.assertFalse(model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertTrue(model.body.weight.requires_grad)
        self.assertTrue(summary["freeze_embeddings"])
        self.assertEqual(summary["frozen_embedding_parameter_count"], 1)
        self.assertTrue(summary["input_embedding_output_weight_tied"])

    def test_untied_output_head_remains_trainable(self) -> None:
        model = _tiny_gemma4(tied=False)
        summary = prepare_full_finetuning_parameters(model, freeze_embeddings=True)

        self.assertFalse(model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)
        self.assertFalse(summary["input_embedding_output_weight_tied"])

    def test_freeze_embeddings_requires_a_real_boolean(self) -> None:
        model = _tiny_gemma4(tied=True)
        with self.assertRaisesRegex(TypeError, "explicit boolean"):
            prepare_full_finetuning_parameters(model, freeze_embeddings="false")


if __name__ == "__main__":
    unittest.main()
