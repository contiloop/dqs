from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_stage import _should_eval_after_subset


class TrainStageEvalCadenceTest(TestCase):
    def test_three_subset_cadence_stays_anchored_to_run_start_after_resume(self) -> None:
        args = SimpleNamespace(
            eval_every_n_subsets=3,
            eval_on_final_subset=True,
        )

        evaluated = {
            subset_idx
            for subset_idx in range(2, 23)
            if _should_eval_after_subset(
                args=args,
                run_start=0,
                stage_end=23,
                subset_idx=subset_idx,
            )
        }

        self.assertEqual(evaluated, {3, 6, 9, 12, 15, 18, 21, 22})

    def test_subset_zero_is_an_eval_point(self) -> None:
        args = SimpleNamespace(
            eval_every_n_subsets=3,
            eval_on_final_subset=False,
        )

        self.assertTrue(
            _should_eval_after_subset(
                args=args,
                run_start=0,
                stage_end=23,
                subset_idx=0,
            )
        )
        self.assertFalse(
            _should_eval_after_subset(
                args=args,
                run_start=0,
                stage_end=23,
                subset_idx=2,
            )
        )
