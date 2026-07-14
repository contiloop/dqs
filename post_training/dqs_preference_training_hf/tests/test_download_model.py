from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_model import download_model, load_model_spec, validate_model_dir  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class DownloadModelContractTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        expected_sft = {
            "run_id": "source-run",
            "subset_idx": 22,
            "global_step": 184,
            "tuning_mode": "full",
        }
        payloads = {
            "config.json": b"{}\n",
            "dqs_stage_model.json": (
                json.dumps(expected_sft, sort_keys=True) + "\n"
            ).encode(),
            "model.safetensors": b"weights",
            "tokenizer_config.json": b"{}\n",
        }
        model_dir = root / "models" / "sft_final"
        model_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            (model_dir / name).write_bytes(payload)
        files = {
            name: {"sha256": sha256_bytes(payload), "size": len(payload)}
            for name, payload in payloads.items()
        }
        objective_expected = {key: expected_sft[key] for key in ("run_id", "subset_idx", "global_step")}
        manifest = {
            "schema_version": "dqs_preference_release.v1",
            "objectives": {
                name: {"expected_sft": objective_expected}
                for name in ("mpo", "cpo", "dpo")
            },
            "sft_model": {
                "repo_id": "org/dataset",
                "repo_type": "dataset",
                "revision": "a" * 40,
                "remote_dir": "run/checkpoints/final",
                "local_dir": "models/sft_final",
                "file_count": len(files),
                "total_size_bytes": sum(value["size"] for value in files.values()),
                "files": files,
                "expected_sft": expected_sft,
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return model_dir

    def test_offline_validation_accepts_exact_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = self.fixture(root)
            _, spec = load_model_spec(root)
            result = validate_model_dir(model_dir, spec)
            self.assertEqual(result["file_count"], 4)
            self.assertEqual(result["expected_sft"]["global_step"], 184)

    def test_offline_validation_rejects_corrupt_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = self.fixture(root)
            _, spec = load_model_spec(root)
            (model_dir / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "size mismatch|SHA256 mismatch"):
                validate_model_dir(model_dir, spec)

    def test_offline_validation_rejects_unexpected_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = self.fixture(root)
            _, spec = load_model_spec(root)
            (model_dir / "optimizer.pt").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                validate_model_dir(model_dir, spec)

    def test_model_and_objective_provenance_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.fixture(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["objectives"]["dpo"]["expected_sft"]["global_step"] = 183
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance requirements disagree"):
                load_model_spec(root)

    def test_download_uses_only_the_pinned_snapshot_then_installs_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = self.fixture(root)
            payloads = {
                path.name: path.read_bytes()
                for path in model_dir.iterdir()
                if path.is_file()
            }
            for path in model_dir.iterdir():
                path.unlink()
            model_dir.rmdir()

            calls: list[dict[str, object]] = []

            def fake_snapshot_download(**kwargs: object) -> str:
                calls.append(kwargs)
                snapshot = Path(str(kwargs["local_dir"]))
                remote = snapshot / "run" / "checkpoints" / "final"
                remote.mkdir(parents=True)
                for name, payload in payloads.items():
                    (remote / name).write_bytes(payload)
                return str(snapshot)

            with patch("huggingface_hub.snapshot_download", fake_snapshot_download):
                result = download_model(root=root, output=model_dir, workers=3)

            self.assertEqual(result["mode"], "explicit-download-then-local")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["repo_id"], "org/dataset")
            self.assertEqual(calls[0]["repo_type"], "dataset")
            self.assertEqual(calls[0]["revision"], "a" * 40)
            self.assertEqual(calls[0]["max_workers"], 3)
            _, spec = load_model_spec(root)
            validate_model_dir(model_dir, spec)


if __name__ == "__main__":
    unittest.main()
