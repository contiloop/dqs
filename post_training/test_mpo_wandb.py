from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mpo_wandb import (
    MPOWandbConfig,
    initialize_wandb_run,
    map_trainer_logs,
    require_wandb_runtime_version,
)


class FakeRun:
    def __init__(self, *, fail_log: bool = False) -> None:
        self.fail_log = fail_log
        self.defined: list[tuple[tuple, dict]] = []
        self.logged: list[dict] = []
        self.finished: list[int] = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, payload, *, commit):
        if self.fail_log:
            raise RuntimeError("synthetic W&B log failure")
        self.logged.append({"payload": dict(payload), "commit": commit})

    def finish(self, *, exit_code):
        self.finished.append(exit_code)


class FakeWandb:
    def __init__(self, run) -> None:
        self.run = run
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


class MPOWandbTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).parent / "configs" / "mpo_setting5.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.spec = MPOWandbConfig.from_logging_config(
            payload["logging"],
            post_training_run_id=payload["run"]["id"],
        )

    def test_trainer_mapping_preserves_custom_namespace(self) -> None:
        mapped = map_trainer_logs(
            {
                "loss": 3.0,
                "grad_norm": 1.5,
                "learning_rate": 5e-6,
                "train/loss/sft_weighted": 2.0,
                "train/margin/mean": 0.25,
            }
        )
        self.assertEqual(mapped["train/loss/hf_global"], 3.0)
        self.assertEqual(mapped["train/loss/sft_weighted"], 2.0)
        self.assertFalse(any(key.startswith("train/train/") for key in mapped))

    def test_rank_zero_initializes_stable_run_and_logs_eval_at_train_step(self) -> None:
        run = FakeRun()
        wandb = FakeWandb(run)
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=False):
            session = initialize_wandb_run(
                self.spec,
                output_dir=Path(temp_dir),
                metadata={"post_training": {"paper_setting": 5}},
                active_process=True,
                wandb_module=wandb,
            )
            session.log_eval_summary(
                profile="val",
                summary={
                    "rows": 10,
                    "filter_fail_ratio": 0.1,
                    "metrics": {"comet": {"mean": 0.75}},
                    "model_path": "/not/a/metric",
                },
                global_step=53,
            )
            session.finish(exit_code=0)

            self.assertEqual(os.environ["WANDB_MODE"], "online")
            self.assertEqual(Path(os.environ["WANDB_DIR"]), Path(temp_dir).resolve())
        self.assertEqual(len(wandb.init_calls), 1)
        self.assertEqual(wandb.init_calls[0]["id"], self.spec.run_id)
        self.assertEqual(wandb.init_calls[0]["resume"], "allow")
        self.assertEqual(wandb.init_calls[0]["mode"], "online")
        self.assertIs(wandb.init_calls[0]["force"], True)
        self.assertIs(wandb.init_calls[0]["save_code"], False)
        self.assertEqual(run.finished, [0])
        payload = run.logged[-1]["payload"]
        self.assertEqual(payload["train/global_step"], 53)
        self.assertEqual(payload["eval/val/metrics/comet/mean"], 0.75)
        self.assertNotIn("eval/val/model_path", payload)

    def test_smoke_or_nonzero_rank_does_not_import_or_initialize_wandb(self) -> None:
        wandb = FakeWandb(FakeRun())
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=False):
            session = initialize_wandb_run(
                self.spec,
                output_dir=Path(temp_dir),
                metadata=None,
                active_process=False,
                wandb_module=wandb,
            )
            self.assertFalse(session.active)
            self.assertEqual(os.environ["WANDB_MODE"], "disabled")
        self.assertEqual(wandb.init_calls, [])

    def test_init_and_log_failures_are_not_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "returned no run"):
                initialize_wandb_run(
                    self.spec,
                    output_dir=Path(temp_dir),
                    metadata=None,
                    active_process=True,
                    wandb_module=FakeWandb(None),
                )
            session = initialize_wandb_run(
                self.spec,
                output_dir=Path(temp_dir),
                metadata=None,
                active_process=True,
                wandb_module=FakeWandb(FakeRun(fail_log=True)),
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic W&B log failure"):
                session.log({"train/global_step": 1})

    def test_eval_requires_exact_wandb_runtime(self) -> None:
        with patch("mpo_wandb.importlib.metadata.version", return_value="0.28.1"):
            with self.assertRaisesRegex(RuntimeError, "requires wandb==0.28.0"):
                require_wandb_runtime_version()


if __name__ == "__main__":
    unittest.main()
