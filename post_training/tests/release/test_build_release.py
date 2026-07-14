from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build_release import build  # noqa: E402


class BuildReleaseTest(unittest.TestCase):
    def args(self, output: Path, **overrides: object) -> argparse.Namespace:
        values = {
            "output": output,
            "data_mode": "hf",
            "hf_repo_id": "alwaysgood/dqs-post-training",
            "hf_revision": "a" * 40,
            "replace": False,
            "archive": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_hf_release_stages_download_before_local_only_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "release"
            result = build(self.args(output))
            self.assertEqual(result["data_mode"], "hf")
            self.assertFalse(any((output / "data" / "train").iterdir()))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["hf_dataset"]["revision"], "a" * 40)
            self.assertEqual(manifest["data_access"], "explicit_download_then_local")
            self.assertEqual(set(manifest["objectives"]), {"mpo", "cpo", "dpo"})
            for objective in ("mpo", "cpo", "dpo"):
                config = yaml.safe_load(
                    (output / "configs" / f"{objective}.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(config["data"]["source"], "local")
                self.assertEqual(config["data"]["path"], f"data/train/{objective}.jsonl")
                self.assertEqual(
                    config["data"]["cache_dir"], f".cache/datasets/{objective}"
                )
                self.assertIsNone(config["data"]["hf_repo_id"])
                self.assertIsNone(config["data"]["hf_revision"])
                self.assertFalse(manifest["objectives"][objective]["data_bundled"])
                self.assertEqual(
                    manifest["objectives"][objective]["data"],
                    f"data/train/{objective}.jsonl",
                )
            self.assertTrue((output / "scripts" / "download_data.py").is_file())
            self.assertTrue((output / "scripts" / "download_model.py").is_file())
            self.assertEqual(manifest["sft_model"]["repo_id"], "alwaysgood/dqs-runs")
            self.assertEqual(
                manifest["sft_model"]["revision"],
                "a58b1878988efcecc9a2644f8324bd00131864b5",
            )
            self.assertEqual(manifest["sft_model"]["file_count"], 8)
            completed = subprocess.run(
                [sys.executable, "scripts/validate_bundle.py"],
                cwd=output,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            missing_data = subprocess.run(
                [sys.executable, "scripts/download_data.py", "--check"],
                cwd=output,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(missing_data.returncode, 0)
            self.assertIn("make download-data", missing_data.stderr)

    def test_replace_refuses_an_unmarked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "not-a-release"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "unmarked"):
                build(self.args(output, replace=True))

    def test_hf_revision_must_be_exact_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "release"
            with self.assertRaisesRegex(ValueError, "40-hex"):
                build(self.args(output, hf_revision="main"))


if __name__ == "__main__":
    unittest.main()
