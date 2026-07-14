from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from eval_mpo import (
    POST_TRAINING_ROOT,
    build_eval_command,
    log_eval_to_post_training_wandb,
    resolve_eval_paths,
)


class MPOEvalIsolationTest(unittest.TestCase):
    def base_payload(self, output_dir: Path) -> dict:
        return {
            "run": {"id": "mpo-test-run", "output_dir": str(output_dir)},
            "evaluation": {
                "base_config": "configs/config.yaml",
                "model_profile": "gemma4_e2b_it",
                "training_profile": "full",
                "default_profile": "val",
                "output_subdir": "eval",
            },
        }

    def test_eval_model_and_output_are_isolated_under_post_training(self) -> None:
        payload = yaml.safe_load(
            (POST_TRAINING_ROOT / "configs" / "mpo_setting5.yaml").read_text(encoding="utf-8")
        )
        paths = resolve_eval_paths(payload, require_final_artifact=False)
        self.assertEqual(paths.final_model_dir, Path(payload["run"]["output_dir"]).resolve() / "final")
        self.assertEqual(paths.output_dir, Path(payload["run"]["output_dir"]).resolve() / "eval" / "val")
        self.assertIn(POST_TRAINING_ROOT.resolve(), paths.output_dir.parents)

        args = argparse.Namespace(
            override=[], data_path=None, limit=None, metrics=None, force=False,
            dry_run=True, skip_wandb_log=False,
        )
        command = build_eval_command(paths, args)
        self.assertEqual(command[command.index("--model-path") + 1], str(paths.final_model_dir))
        self.assertEqual(command[command.index("--output-dir") + 1], str(paths.output_dir))
        self.assertEqual(command.count("--skip-wandb-log"), 1)

    def test_output_outside_post_training_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = self.base_payload(Path(temp_dir) / "run")
            with self.assertRaisesRegex(ValueError, "must stay under"):
                resolve_eval_paths(payload, require_final_artifact=False)

    def test_final_marker_and_weights_are_required(self) -> None:
        outputs_root = POST_TRAINING_ROOT / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=outputs_root) as temp_dir:
            run_output = Path(temp_dir)
            payload = self.base_payload(run_output)
            with self.assertRaisesRegex(ValueError, "provenance marker"):
                resolve_eval_paths(payload)

            final_dir = run_output / "final"
            final_dir.mkdir()
            (final_dir / "dqs_mpo_model.json").write_text(
                json.dumps({"run_id": "mpo-test-run", "paper_setting": 5}), encoding="utf-8"
            )
            (final_dir / "config.json").write_text("{}", encoding="utf-8")
            (final_dir / "model.safetensors").write_bytes(b"weights")
            paths = resolve_eval_paths(payload)
            self.assertEqual(paths.final_model_dir, final_dir.resolve())

    def test_completed_eval_appends_to_same_wandb_run_at_final_step(self) -> None:
        outputs_root = POST_TRAINING_ROOT / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=outputs_root) as temp_dir:
            run_output = Path(temp_dir)
            payload = yaml.safe_load(
                (POST_TRAINING_ROOT / "configs" / "mpo_setting5.yaml").read_text(
                    encoding="utf-8"
                )
            )
            payload["run"]["id"] = "mpo-test-run"
            payload["run"]["output_dir"] = str(run_output)
            payload["logging"]["wandb"]["run_id"] = "mpo-test-run"
            payload["logging"]["wandb"]["run_name"] = "mpo-test-run"

            final_dir = run_output / "final"
            final_dir.mkdir()
            (final_dir / "dqs_mpo_model.json").write_text(
                json.dumps(
                    {"run_id": "mpo-test-run", "paper_setting": 5, "global_step": 53}
                ),
                encoding="utf-8",
            )
            (final_dir / "config.json").write_text("{}", encoding="utf-8")
            (final_dir / "model.safetensors").write_bytes(b"weights")
            paths = resolve_eval_paths(payload)
            paths.output_dir.mkdir(parents=True)
            (paths.output_dir / "eval_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": paths.run_id,
                        "eval_profile": paths.profile,
                        "model_path": str(paths.final_model_dir),
                        "rows": 100,
                        "metrics": {"comet": {"mean": 0.8}},
                    }
                ),
                encoding="utf-8",
            )

            session = MagicMock()
            with (
                patch("eval_mpo.require_wandb_runtime_version") as require_version,
                patch("eval_mpo.initialize_wandb_run", return_value=session) as initialize,
            ):
                log_eval_to_post_training_wandb(payload, paths)

            require_version.assert_called_once_with()
            initialize.assert_called_once()
            self.assertEqual(initialize.call_args.kwargs["output_dir"], run_output)
            session.log_eval_summary.assert_called_once()
            self.assertEqual(session.log_eval_summary.call_args.kwargs["global_step"], 53)
            session.finish.assert_called_once_with(exit_code=0)


if __name__ == "__main__":
    unittest.main()
