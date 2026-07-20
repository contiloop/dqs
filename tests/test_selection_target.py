from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from selection_target import selection_target_per_subset
from train import _teacher_candidate_count
from train_stage import _planned_sft_rows_per_subset


class SelectionTargetTest(TestCase):
    def test_target_is_derived_from_subset_size_and_ratio(self) -> None:
        cfg = {
            "data": {"subset_size": 100_000, "selection_ratio": 0.02},
            "teacher": {"candidate_multiplier": 4},
            "training": {"stage_planned_rows_per_subset": None},
        }

        self.assertEqual(selection_target_per_subset(cfg), 2000)
        self.assertEqual(_teacher_candidate_count(cfg), 8000)
        self.assertEqual(_planned_sft_rows_per_subset(cfg), 2000)

    def test_stage_plan_can_still_be_overridden_explicitly(self) -> None:
        cfg = {
            "data": {"subset_size": 100_000, "selection_ratio": 0.02},
            "training": {"stage_planned_rows_per_subset": 123},
        }

        self.assertEqual(_planned_sft_rows_per_subset(cfg), 123)
