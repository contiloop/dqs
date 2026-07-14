from __future__ import annotations

from copy import deepcopy
import json
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preference_runtime import POST_TRAINING_ROOT, REPO_ROOT, _output_dir  # noqa: E402
from train_cpo import validate_config as validate_cpo_config  # noqa: E402
from train_dpo import validate_config as validate_dpo_config  # noqa: E402
from train_mpo import _validate_hard_config  # noqa: E402


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
