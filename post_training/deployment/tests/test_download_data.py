from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_data import validate_installed_data  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class DownloadDataContractTest(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        objectives: dict[str, dict[str, object]] = {}
        for name in ("mpo", "cpo", "dpo"):
            artifact = (json.dumps({"pair_id": name}) + "\n").encode()
            artifact_sha = sha256_bytes(artifact)
            contract_payload = {"artifact_sha256": artifact_sha, "row_count": 1}
            contract = (
                json.dumps(contract_payload, sort_keys=True, indent=2) + "\n"
            ).encode()
            contract_path = root / "data" / "contracts" / f"{name}.json"
            data_path = root / "data" / "train" / f"{name}.jsonl"
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            contract_path.write_bytes(contract)
            data_path.write_bytes(artifact)
            objectives[name] = {
                "data": f"data/train/{name}.jsonl",
                "data_bundled": False,
                "contract": f"data/contracts/{name}.json",
                "artifact_sha256": artifact_sha,
                "release_contract_sha256": sha256_bytes(contract),
                "row_count": 1,
                "hf_train_filename": f"{name}/train.jsonl",
                "hf_contract_filename": f"{name}/dataset_contract.json",
            }
        manifest = {
            "schema_version": "dqs_preference_release.v1",
            "data_mode": "hf",
            "data_access": "explicit_download_then_local",
            "hf_dataset": {"repo_id": "org/repo", "revision": "a" * 40},
            "objectives": objectives,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def test_offline_validation_accepts_exact_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.fixture(root)
            result = validate_installed_data(root)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(set(result["objectives"]), {"mpo", "cpo", "dpo"})

    def test_offline_validation_rejects_corrupt_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.fixture(root)
            (root / "data" / "train" / "mpo.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mpo train data hash mismatch"):
                validate_installed_data(root)

    def test_offline_validation_requires_downloaded_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.fixture(root)
            (root / "data" / "train" / "dpo.jsonl").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "make download-data"):
                validate_installed_data(root)


if __name__ == "__main__":
    unittest.main()
