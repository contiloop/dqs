from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval import _save_eval_config_provenance


def _load_yaml(path: Path) -> dict[str, object]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


class EvalConfigProvenanceTest(TestCase):
    def test_separates_eval_defaults_from_training_selection_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            output_dir = root / "eval" / "final"
            model_path = root / "checkpoints" / "final"
            model_path.mkdir(parents=True)
            training_config_path = root / "effective_config.yaml"
            training_config_path.write_text(
                "data:\n  selection_ratio: 0.02\n  qe_selection_order: random\n",
                encoding="utf-8",
            )
            cfg = {
                "paths": {"config_snapshot_path": str(training_config_path)},
                "data": {"selection_ratio": 0.01, "qe_selection_order": "low"},
            }

            manifest = _save_eval_config_provenance(
                cfg=cfg,
                output_dir=output_dir,
                model_path=model_path,
            )

            eval_cfg = _load_yaml(output_dir / "eval_effective_config.yaml")
            training_cfg = _load_yaml(output_dir / "training_effective_config.yaml")
            legacy_text = (output_dir / "effective_config.yaml").read_text(encoding="utf-8")
            saved_manifest = json.loads((output_dir / "config_provenance.json").read_text(encoding="utf-8"))

            self.assertEqual(eval_cfg["data"]["selection_ratio"], 0.01)  # type: ignore[index]
            self.assertEqual(training_cfg["data"]["selection_ratio"], 0.02)  # type: ignore[index]
            self.assertTrue(legacy_text.startswith("# WARNING: evaluation-scope config only"))
            self.assertEqual(saved_manifest, manifest)
            self.assertTrue(manifest["training_config"]["available"])
            self.assertFalse(manifest["eval_config"]["authoritative_for_training"])

    def test_finds_training_config_from_local_model_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            output_dir = Path(tmp) / "standalone-eval"
            model_path = root / "checkpoints" / "final"
            model_path.mkdir(parents=True)
            (root / "effective_config.yaml").write_text(
                "data:\n  selection_ratio: 0.02\n",
                encoding="utf-8",
            )
            cfg = {
                "paths": {"config_snapshot_path": str(Path(tmp) / "missing.yaml")},
                "data": {"selection_ratio": 0.01},
            }

            manifest = _save_eval_config_provenance(
                cfg=cfg,
                output_dir=output_dir,
                model_path=model_path,
            )

            self.assertTrue(manifest["training_config"]["available"])
            self.assertEqual(
                _load_yaml(output_dir / "training_effective_config.yaml")["data"]["selection_ratio"],  # type: ignore[index]
                0.02,
            )

    def test_records_missing_training_config_without_stale_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"
            output_dir.mkdir(parents=True)
            (output_dir / "training_effective_config.yaml").write_text("stale: true\n", encoding="utf-8")

            manifest = _save_eval_config_provenance(
                cfg={"data": {"selection_ratio": 0.01}},
                output_dir=output_dir,
                model_path="remote/model-id",
            )

            self.assertFalse(manifest["training_config"]["available"])
            self.assertFalse((output_dir / "training_effective_config.yaml").exists())
