from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mpo_model import _dtype, require_final_stage_model
from mpo_masking import (
    completion_sft_loss,
    masked_causal_logp_mean,
    selected_causal_token_logps,
    selected_causal_logp_mean,
    shifted_token_logps,
)
from train_mpo import (
    REQUIRED_RUNTIME_VERSIONS,
    _require_runtime_versions,
    _validate_hard_config,
    _world_size,
)


class NoFallbackContractTest(unittest.TestCase):
    def setUp(self) -> None:
        post_training_root = Path(__file__).resolve().parents[2]
        config_path = post_training_root / "dqs_preference_training_hf" / "configs" / "mpo.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.run = payload["run"]
        self.model = payload["model"]
        self.data = payload["data"]
        self.loss = payload["loss"]
        self.training = payload["training"]
        self.logging = payload["logging"]

    def validate(self) -> None:
        _validate_hard_config(
            run_cfg=self.run,
            model_cfg=self.model,
            data_cfg=self.data,
            loss_values=self.loss,
            training_cfg=self.training,
            logging_cfg=self.logging,
        )

    def test_exact_contract_passes(self) -> None:
        self.validate()

    def test_every_semantic_substitution_is_rejected(self) -> None:
        mutations = (
            (self.run, "require_smoke_step_receipt", False),
            (self.model, "require_final_stage_model", False),
            (self.model, "unsloth_model_api", "fast_language_model"),
            (self.data, "source", "hf"),
            (self.training, "backend", "transformers"),
            (self.training, "dtype", "float16"),
            (self.training, "load_in_4bit", True),
            (self.training, "load_in_8bit", True),
            (self.training, "gradient_checkpointing", "true"),
            (self.training, "unsloth_fullgraph", True),
            (self.training, "logits_projection", "full"),
            (self.training, "token_logp_backend", "torch"),
            (self.training, "report_to", ["wandb"]),
            (self.logging["wandb"], "enabled", False),
            (self.logging["wandb"], "strict", False),
            (self.logging["wandb"], "mode", "offline"),
            (self.logging["wandb"], "run_id", "another-run"),
            (self.logging["wandb"], "log_smoke", True),
            (self.logging["wandb"], "log_model", True),
            (self.logging["wandb"], "watch", True),
        )
        for mapping, key, bad_value in mutations:
            with self.subTest(key=key, bad_value=bad_value):
                original = mapping[key]
                mapping[key] = bad_value
                try:
                    with self.assertRaises(ValueError):
                        self.validate()
                finally:
                    mapping[key] = original

    def test_auto_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "automatic dtype"):
            _dtype("auto")

    def test_embedding_freeze_is_an_explicit_boolean_option(self) -> None:
        self.training["freeze_embeddings"] = True
        self.validate()
        self.training["freeze_embeddings"] = "true"
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            self.validate()

    def test_low_level_loss_backend_must_be_explicit(self) -> None:
        for function in (
            shifted_token_logps,
            selected_causal_token_logps,
            selected_causal_logp_mean,
            masked_causal_logp_mean,
            completion_sft_loss,
        ):
            with self.subTest(function=function.__name__):
                backend = inspect.signature(function).parameters["backend"]
                self.assertIs(backend.default, inspect.Parameter.empty)

    def test_runtime_versions_are_exact(self) -> None:
        exact = {**REQUIRED_RUNTIME_VERSIONS, "torch": "2.10.0"}
        _require_runtime_versions(exact)
        wrong = dict(exact)
        wrong["transformers"] = "5.5.1"
        with self.assertRaises(RuntimeError):
            _require_runtime_versions(wrong)
        wrong_torch = dict(exact)
        wrong_torch["torch"] = "2.11.0"
        with self.assertRaises(RuntimeError):
            _require_runtime_versions(wrong_torch)

    def test_invalid_world_size_is_not_coerced(self) -> None:
        with patch.dict(os.environ, {"WORLD_SIZE": "0"}):
            with self.assertRaises(ValueError):
                _world_size()

    def test_final_sft_marker_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = Path(temp_dir) / "sft_final"
            final_dir.mkdir()
            config = {
                "name_or_path": str(final_dir),
                "require_final_stage_model": True,
                "expected_sft_run_id": "source-run",
                "expected_sft_subset_idx": 22,
                "expected_sft_global_step": 184,
            }
            with self.assertRaisesRegex(ValueError, "provenance marker"):
                require_final_stage_model(config)
            (final_dir / "dqs_stage_model.json").write_text(
                json.dumps(
                    {
                        "tuning_mode": "full",
                        "run_id": "source-run",
                        "subset_idx": 22,
                        "global_step": 184,
                    }
                ),
                encoding="utf-8",
            )
            (final_dir / "config.json").write_text("{}", encoding="utf-8")
            (final_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (final_dir / "model.safetensors").write_bytes(b"weights")
            require_final_stage_model(config)


if __name__ == "__main__":
    unittest.main()
