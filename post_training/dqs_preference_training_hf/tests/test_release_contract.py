from __future__ import annotations

from copy import deepcopy
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preference_runtime import (  # noqa: E402
    POST_TRAINING_ROOT,
    REQUIRED_RUNTIME_VERSIONS,
    REPO_ROOT,
    _output_dir,
    _training_argument_kwargs,
)
from train_cpo import validate_config as validate_cpo_config  # noqa: E402
from train_dpo import dpo_config_kwargs, validate_config as validate_dpo_config  # noqa: E402
from train_mpo import _validate_hard_config  # noqa: E402
from mpo_masking import MPOPreferenceCollator  # noqa: E402
from mpo_objective import Setting5LossConfig  # noqa: E402
from mpo_trainer import compute_setting5_batch_loss  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from validate_bundle import validate_model_eos_profile  # noqa: E402


class ReleaseContractTest(unittest.TestCase):
    def load(self, name: str) -> dict:
        return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))

    def test_release_layout_is_detected(self) -> None:
        self.assertEqual(REPO_ROOT.resolve(), ROOT.resolve())
        self.assertEqual(POST_TRAINING_ROOT.resolve(), ROOT.resolve())

    def test_all_objective_configs_are_strict(self) -> None:
        mpo = self.load("mpo.yaml")
        _validate_hard_config(
            run_cfg=mpo["run"],
            model_cfg=mpo["model"],
            data_cfg=mpo["data"],
            loss_values=mpo["loss"],
            training_cfg=mpo["training"],
            logging_cfg=mpo["logging"],
        )
        validate_cpo_config(self.load("cpo.yaml"))
        validate_dpo_config(self.load("dpo.yaml"))

    def test_all_mpo_variants_satisfy_the_hard_config_contract(self) -> None:
        for name in ("mpo.yaml", "mpo_constant.yaml", "mpo_constant_lambda5.yaml"):
            with self.subTest(config=name):
                mpo = self.load(name)
                _validate_hard_config(
                    run_cfg=mpo["run"],
                    model_cfg=mpo["model"],
                    data_cfg=mpo["data"],
                    loss_values=mpo["loss"],
                    training_cfg=mpo["training"],
                    logging_cfg=mpo["logging"],
                )

    def test_argument_kwargs_match_the_pinned_v5_constructor_surfaces(self) -> None:
        from transformers import TrainingArguments

        removed_v5_fields = {"overwrite_output_dir", "save_safetensors"}
        training_parameters = set(inspect.signature(TrainingArguments).parameters)
        dpo_024_parameters_used = {
            "beta",
            "dataset_num_proc",
            "disable_dropout",
            "f_divergence_type",
            "generate_during_eval",
            "label_smoothing",
            "ld_alpha",
            "loss_type",
            "loss_weights",
            "max_completion_length",
            "max_length",
            "max_prompt_length",
            "pad_token",
            "padding_free",
            "precompute_ref_batch_size",
            "precompute_ref_log_probs",
            "reference_free",
            "rpo_alpha",
            "sync_ref_model",
            "truncation_mode",
            "use_liger_loss",
            "use_logits_to_keep",
            "use_weighting",
        }
        with patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0"}, clear=False):
            for objective in ("mpo", "cpo"):
                config = self.load(f"{objective}.yaml")
                kwargs = _training_argument_kwargs(
                    run_cfg=config["run"],
                    training_cfg=config["training"],
                    output_dir=ROOT / "outputs" / "constructor-test" / objective,
                    has_eval=False,
                    smoke_step=True,
                )
                self.assertFalse(removed_v5_fields & set(kwargs))
                self.assertEqual(set(kwargs) - training_parameters, set())
                self.assertEqual(kwargs["warmup_steps"], 0.1)
                self.assertNotIn("warmup_ratio", kwargs)

            config = self.load("dpo.yaml")
            kwargs = dpo_config_kwargs(
                run=config["run"],
                loss=config["loss"],
                reference=config["reference"],
                training=config["training"],
                output_dir=ROOT / "outputs" / "constructor-test" / "dpo",
                pad_token="<pad>",
                smoke_step=True,
            )
        self.assertFalse(removed_v5_fields & set(kwargs))
        self.assertEqual(set(kwargs) - training_parameters, dpo_024_parameters_used)
        self.assertEqual(kwargs["warmup_steps"], 0.1)
        self.assertNotIn("warmup_ratio", kwargs)

    def test_all_objectives_require_compile_disabled_unsloth(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("UNSLOTH_RUNTIME_ENV := UNSLOTH_COMPILE_DISABLE=1", makefile)
        for objective in ("mpo", "cpo", "dpo"):
            training = self.load(f"{objective}.yaml")["training"]
            self.assertEqual(training["backend"], "unsloth")
            self.assertEqual(training["gradient_checkpointing"], "unsloth")
            self.assertEqual(training["unsloth_compile"], "disabled")
            self.assertEqual(training["warmup_steps"], 0.1)
            self.assertNotIn("warmup_ratio", training)

    def test_outputs_are_isolated_and_distinct(self) -> None:
        outputs = []
        for name in ("mpo.yaml", "cpo.yaml", "dpo.yaml"):
            config = self.load(name)
            output = _output_dir(config["run"], smoke_step=False)
            self.assertIn(ROOT.resolve(), output.resolve().parents)
            outputs.append(output)
        self.assertEqual(len(set(outputs)), 3)

    def test_manifest_declares_exact_objectives(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["objectives"]), {"mpo", "cpo", "dpo"})
        self.assertEqual(manifest["schema_version"], "dqs_preference_release.v1")
        self.assertEqual(manifest["data_access"], "explicit_download_then_local")

    def test_gemma4_transformers_override_is_explicit_and_exact(self) -> None:
        bootstrap = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
        override = (ROOT / "requirements-transformers-gemma4.txt").read_text(
            encoding="utf-8"
        )
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertIn("transformers==5.5.0", bootstrap)
        self.assertIn("transformers==5.5.3", override)
        self.assertEqual(REQUIRED_RUNTIME_VERSIONS["transformers"], "5.5.3")
        self.assertIn(
            "pip install --no-deps --force-reinstall -r "
            "$(TRANSFORMERS_OVERRIDE_REQUIREMENTS)",
            makefile,
        )
        self.assertEqual(
            manifest["runtime_override_requirements"],
            "requirements-transformers-gemma4.txt",
        )

    def test_mpo_collator_uses_one_shared_forward_length(self) -> None:
        batch = MPOPreferenceCollator(pad_token_id=0)(
            [
                {
                    "pair_id": "different-term-token-lengths",
                    "chosen_input_ids": [2, 3, 4, 5],
                    "chosen_completion_mask": [0, 1, 1, 1],
                    "chosen_term_mask": [0, 0, 1, 1],
                    "rejected_input_ids": [2, 6],
                    "rejected_completion_mask": [0, 1],
                    "rejected_term_mask": [0, 1],
                }
            ]
        )
        self.assertEqual(batch["chosen_input_ids"].shape, batch["rejected_input_ids"].shape)
        self.assertEqual(batch["rejected_attention_mask"].tolist(), [[True, True, False, False]])
        self.assertEqual(batch["rejected_term_mask"].tolist(), [[False, True, False, False]])

    def test_mpo_release_uses_one_concatenated_preference_forward(self) -> None:
        class RecordingModel:
            def __init__(self) -> None:
                self.embedding = torch.nn.Embedding(8, 5)
                self.projection = torch.nn.Linear(5, 8, bias=False)
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
                hidden = self.embedding(input_ids)
                if not isinstance(logits_to_keep, int):
                    hidden = hidden.index_select(1, logits_to_keep)
                return SimpleNamespace(logits=self.projection(hidden))

        preference_batch = MPOPreferenceCollator(pad_token_id=0)(
            [
                {
                    "pair_id": "single-forward-contract",
                    "chosen_input_ids": [2, 3, 4, 5],
                    "chosen_completion_mask": [0, 1, 1, 1],
                    "chosen_term_mask": [0, 0, 1, 1],
                    "rejected_input_ids": [2, 6],
                    "rejected_completion_mask": [0, 1],
                    "rejected_term_mask": [0, 1],
                }
            ]
        )
        model = RecordingModel()
        compute_setting5_batch_loss(
            model=model,
            batch=preference_batch,
            config=Setting5LossConfig(),
            projection="selected",
            token_logp_backend="torch",
        )

        self.assertEqual(model.calls, [(2, 4)])

    def test_manifest_pins_the_exact_full_sft_model(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        model = manifest["sft_model"]
        self.assertEqual(model["repo_id"], "alwaysgood/dqs-runs")
        self.assertEqual(model["repo_type"], "dataset")
        self.assertEqual(
            model["revision"], "a58b1878988efcecc9a2644f8324bd00131864b5"
        )
        self.assertEqual(
            model["remote_dir"],
            "gemma4_e2b_it_full_iter_lowqe_sf_on_seed42/checkpoints/final",
        )
        self.assertEqual(model["local_dir"], "models/sft_final")
        self.assertEqual(model["file_count"], 8)
        self.assertEqual(model["total_size_bytes"], 10_279_726_920)
        self.assertEqual(
            model["files"]["model.safetensors"]["sha256"],
            "304387c31d762065420035d711ebed0eb6e296d0ee28c8918645ed3943fdaf4e",
        )

    def test_all_objectives_pin_the_final_sft_eos_profile(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        model = manifest["sft_model"]
        expected_source = {
            "repo_id": model["repo_id"],
            "repo_type": model["repo_type"],
            "revision": model["revision"],
            "subfolder": model["remote_dir"],
            "tokenizer_config_sha256": model["files"]["tokenizer_config.json"][
                "sha256"
            ],
        }
        for objective in ("mpo", "cpo", "dpo"):
            contract = json.loads(
                (ROOT / f"data/contracts/{objective}.json").read_text(encoding="utf-8")
            )
            tokenization = (
                contract["tokenization_contract"] if objective == "dpo" else contract
            )
            self.assertEqual(tokenization["eos_token"], "<turn|>")
            self.assertEqual(tokenization["eos_token_id"], 106)
            alignment = tokenization["post_sft_tokenizer_alignment"]
            self.assertEqual(alignment["source_eos_token_id"], 1)
            self.assertEqual(alignment["target_eos_token_id"], 106)
            self.assertEqual(alignment["repair_or_fallback"], "none")
            source = alignment["final_sft_tokenizer"]
            self.assertEqual(
                {key: source[key] for key in expected_source}, expected_source
            )

    def test_downloaded_model_eos_profile_is_fail_closed(self) -> None:
        tokenizer = {
            "eos_token": "<turn|>",
            "added_tokens_decoder": {
                "1": {"content": "<eos>", "special": True},
                "106": {"content": "<turn|>", "special": True},
            },
        }
        model = {"eos_token_id": 106, "text_config": {"eos_token_id": 1}}
        generation = {"eos_token_id": [1, 106, 50]}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for filename, value in (
                ("tokenizer_config.json", tokenizer),
                ("config.json", model),
                ("generation_config.json", generation),
            ):
                (root / filename).write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                validate_model_eos_profile(root)["tokenizer_eos_token_id"], 106
            )
            tokenizer["eos_token"] = "<eos>"
            (root / "tokenizer_config.json").write_text(
                json.dumps(tokenizer), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "EOS profile"):
                validate_model_eos_profile(root)

    def test_training_configs_are_local_only_after_explicit_download(self) -> None:
        for objective in ("mpo", "cpo", "dpo"):
            config = self.load(f"{objective}.yaml")
            data = config["data"]
            self.assertEqual(data["source"], "local")
            self.assertEqual(data["path"], f"data/train/{objective}.jsonl")
            self.assertEqual(data["cache_dir"], f".cache/datasets/{objective}")
            self.assertIsNone(data["hf_repo_id"])
            self.assertIsNone(data["hf_revision"])

    def test_all_trainers_reject_lazy_hf_data_loading(self) -> None:
        mpo = self.load("mpo.yaml")
        changed_mpo = deepcopy(mpo)
        changed_mpo["data"]["source"] = "hf"
        with self.assertRaisesRegex(ValueError, "must be local"):
            _validate_hard_config(
                run_cfg=changed_mpo["run"],
                model_cfg=changed_mpo["model"],
                data_cfg=changed_mpo["data"],
                loss_values=changed_mpo["loss"],
                training_cfg=changed_mpo["training"],
                logging_cfg=changed_mpo["logging"],
            )
        for filename, validator in (
            ("cpo.yaml", validate_cpo_config),
            ("dpo.yaml", validate_dpo_config),
        ):
            changed = deepcopy(self.load(filename))
            changed["data"]["source"] = "hf"
            with self.assertRaisesRegex(ValueError, "must be local"):
                validator(changed)


if __name__ == "__main__":
    unittest.main()
