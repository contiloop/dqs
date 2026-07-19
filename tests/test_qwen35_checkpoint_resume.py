from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen35_checkpoint_keys import (
    assert_qwen35_checkpoint_compatible,
    normalize_qwen35_checkpoint_keys,
    qwen35_checkpoint_keys,
)
from sft_train import (
    _assert_qwen35_resume_checkpoint,
    _save_repaired_checkpoint_at_current_step,
)


BAD_PREFIX = "model.language_model.language_model.language_model."
GOOD_PREFIX = "model.language_model."


def _cfg() -> dict[str, object]:
    return {
        "model": {"family": "qwen3.5"},
        "training": {
            "tuning_mode": "full",
            "normalize_full_weight_checkpoint_keys": True,
            "assert_full_weight_checkpoint_keys": True,
        },
    }


def _bad_tensors(count: int = 100) -> dict[str, torch.Tensor]:
    return {
        f"{BAD_PREFIX}layers.{idx}.weight": torch.tensor([float(idx)])
        for idx in range(count)
    }


def _runtime_state(count: int = 100) -> dict[str, torch.Tensor]:
    state = {
        f"{GOOD_PREFIX}layers.{idx}.weight": torch.tensor([float(idx)])
        for idx in range(count)
    }
    # Qwen3.5 can omit this tied/deduplicated key from a safe checkpoint.
    state["lm_head.weight"] = torch.tensor([0.0])
    return state


class _FakeModel:
    def __init__(self) -> None:
        self._state = _runtime_state()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state


class _FakeTrainer:
    def __init__(self, output_dir: Path, model: _FakeModel) -> None:
        self.args = SimpleNamespace(output_dir=str(output_dir))
        self.state = SimpleNamespace(global_step=8)
        self.model = model

    def _save_checkpoint(self, model: object, trial: object | None = None) -> None:
        checkpoint_dir = Path(self.args.output_dir) / "checkpoint-8"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_file(_bad_tensors(), str(checkpoint_dir / "model.safetensors"))


class Qwen35CheckpointResumeTest(TestCase):
    def test_rejects_zero_overlap_legacy_checkpoint_before_resume(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoint_dir = output_dir / "checkpoint-8"
            checkpoint_dir.mkdir()
            save_file(_bad_tensors(), str(checkpoint_dir / "model.safetensors"))

            with self.assertRaisesRegex(SystemExit, "bad Qwen3.5 checkpoint keys remain"):
                _assert_qwen35_resume_checkpoint(
                    _cfg(),
                    _FakeModel(),
                    str(checkpoint_dir),
                    output_dir,
                )

    def test_manual_repair_makes_known_good_checkpoint_resume_compatible(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoint_dir = output_dir / "checkpoint-8"
            checkpoint_dir.mkdir()
            save_file(_bad_tensors(), str(checkpoint_dir / "model.safetensors"))

            normalized = normalize_qwen35_checkpoint_keys(checkpoint_dir)
            compatibility = _assert_qwen35_resume_checkpoint(
                _cfg(),
                _FakeModel(),
                str(checkpoint_dir),
                output_dir,
            )

            self.assertEqual(normalized["renamed_key_count"], 100)
            self.assertIsNotNone(compatibility)
            self.assertEqual(compatibility["matched_key_count"], 100)
            self.assertEqual(compatibility["checkpoint_only_count"], 0)
            self.assertGreaterEqual(compatibility["runtime_coverage"], 0.99)

    def test_stage_checkpoint_is_repaired_and_checked_immediately_after_save(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            model = _FakeModel()
            trainer = _FakeTrainer(output_dir, model)

            checkpoint_dir, normalized, compatibility = _save_repaired_checkpoint_at_current_step(
                _cfg(),
                trainer,
                model,
            )

            self.assertEqual(checkpoint_dir, output_dir / "checkpoint-8")
            self.assertIsNotNone(normalized)
            self.assertEqual(normalized["renamed_key_count"], 100)
            self.assertIsNotNone(compatibility)
            self.assertEqual(compatibility["matched_key_count"], 100)
            self.assertNotIn(
                f"{BAD_PREFIX}layers.0.weight",
                qwen35_checkpoint_keys(checkpoint_dir),
            )
            self.assertIn(
                f"{GOOD_PREFIX}layers.0.weight",
                qwen35_checkpoint_keys(checkpoint_dir),
            )

    def test_compatibility_guard_rejects_partial_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            save_file(
                {f"{GOOD_PREFIX}layers.0.weight": torch.tensor([0.0])},
                str(checkpoint_dir / "model.safetensors"),
            )

            with self.assertRaisesRegex(SystemExit, "refusing partial/zero-weight load"):
                assert_qwen35_checkpoint_compatible(
                    _runtime_state().keys(),
                    checkpoint_dir,
                )
