from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from io_utils import read_jsonl, write_jsonl
from config_loader import compose_config
from teacher_generation import TeacherBatchOutput, TeacherOutputItem, run_teacher_generation


def _candidate(item_id: str, rank: int) -> dict[str, object]:
    return {
        "id": item_id,
        "source": f"source {item_id}",
        "student_translation": f"draft {item_id}",
        "selection_rank": rank,
        "qe_score": float(rank),
    }


class TeacherPartialResumeTest(TestCase):
    def test_default_config_enables_resume_and_uses_fifty_workers(self) -> None:
        cfg = compose_config(REPO_ROOT / "configs/config.yaml")

        self.assertTrue(cfg["teacher"]["resume_partial"])
        self.assertEqual(cfg["teacher"]["max_workers"], 50)

    def test_preserves_completed_rows_and_retries_latest_failed_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            subset_dir = Path(tmp)
            system_prompt_path = subset_dir / "system.txt"
            user_prompt_path = subset_dir / "user.txt"
            system_prompt_path.write_text("system", encoding="utf-8")
            user_prompt_path.write_text("{items_json}", encoding="utf-8")

            candidates = [
                _candidate("a", 1),
                _candidate("b", 2),
                _candidate("c", 3),
                _candidate("d", 4),
                _candidate("e", 5),
            ]
            write_jsonl(
                subset_dir / "teacher_artifacts.jsonl",
                [
                    {
                        "record_type": "teacher_request",
                        "batch_idx": 0,
                        "item_ids": ["a"],
                    },
                    {
                        "record_type": "teacher_request",
                        "batch_idx": 1,
                        "item_ids": ["b"],
                    },
                    {
                        "record_type": "teacher_request",
                        "batch_idx": 2,
                        "item_ids": ["c"],
                    },
                    {
                        "record_type": "teacher_raw_response",
                        "batch_idx": 0,
                        "status": "ok",
                        "item_ids": ["a"],
                    },
                    {
                        "record_type": "teacher_raw_response",
                        "batch_idx": 1,
                        "status": "failed",
                        "item_ids": ["b"],
                    },
                    {
                        "record_type": "teacher_raw_response",
                        "batch_idx": 2,
                        "status": "ok",
                        "item_ids": ["c"],
                    },
                    {
                        "record_type": "teacher_parsed_item",
                        "batch_idx": 0,
                        "id": "a",
                        "parse_status": "ok",
                    },
                    {
                        "record_type": "teacher_parsed_item",
                        "batch_idx": 1,
                        "id": "b",
                        "parse_status": "failed",
                    },
                    {
                        "record_type": "teacher_parsed_item",
                        "batch_idx": 2,
                        "id": "c",
                        "parse_status": "ok",
                    },
                    {
                        "record_type": "teacher_rejected_row",
                        "id": "b",
                        "reject_reason": "missing_or_unparsed_teacher_item",
                        "reject_flags": [],
                    },
                    {
                        "record_type": "teacher_rejected_row",
                        "id": "c",
                        "reject_reason": "teacher_invalid",
                        "reject_flags": ["teacher_invalid"],
                    },
                ],
            )
            write_jsonl(
                subset_dir / "golden_pairs.jsonl",
                [
                    {
                        "id": "a",
                        "source": "source a",
                        "target": "target a",
                        "teacher_label": "minor",
                        "teacher_accept_rank": 1,
                    }
                ],
            )

            cfg = {
                "run": {"seed": 42},
                "logging": {"save_all_step_artifacts": False},
                "data": {
                    "teacher_target_per_subset": 3,
                    "max_output_tokens": 100,
                    "length_bucket_selection": {"enabled": False},
                    "degeneration_filter": {"enabled": False},
                },
                "teacher": {
                    "batch_size": 1,
                    "max_workers": 2,
                    "max_retries_per_row": 1,
                    "max_output_tokens": 512,
                    "reject_over_max_output_tokens": False,
                    "refill_until_target": True,
                    "abort_on_all_failed_window": True,
                    "resume_partial": True,
                    "system_prompt_path": str(system_prompt_path),
                    "user_prompt_path": str(user_prompt_path),
                    "providers": [
                        {
                            "name": "gemini",
                            "model": "test-model",
                            "weight": 1.0,
                            "api_key_env": "IGNORED_IN_TEST",
                        }
                    ],
                },
            }
            called_ids: list[str] = []

            def fake_run_teacher_batch_inputs(**kwargs: object) -> dict[int, dict[str, object]]:
                results: dict[int, dict[str, object]] = {}
                for batch_input in kwargs["batch_inputs"]:  # type: ignore[index]
                    batch_idx, provider, _system, _user, batch_rows, _expected = batch_input
                    item_id = str(batch_rows[0]["id"])
                    called_ids.append(item_id)
                    results[int(batch_idx)] = {
                        "batch_idx": batch_idx,
                        "status": "ok",
                        "provider": provider["name"],
                        "model": provider["model"],
                        "attempt": 1,
                        "raw_text": "{}",
                        "parsed": TeacherBatchOutput(
                            items=[
                                TeacherOutputItem(
                                    id=item_id,
                                    label="minor",
                                    final_translation=f"target {item_id}",
                                    invalid_reason_ko=None,
                                    errors=[],
                                )
                            ]
                        ),
                        "latency_ms": 1.0,
                        "usage": {},
                        "error": None,
                    }
                return results

            with patch(
                "teacher_generation._run_teacher_batch_inputs",
                side_effect=fake_run_teacher_batch_inputs,
            ):
                summary = run_teacher_generation(
                    cfg=cfg,
                    candidates=candidates,
                    subset_dir=subset_dir,
                )

            self.assertEqual(called_ids, ["b", "d"])
            self.assertEqual(
                [row["id"] for row in read_jsonl(subset_dir / "golden_pairs.jsonl")],
                ["a", "b", "d"],
            )
            rejected = [
                row
                for row in read_jsonl(subset_dir / "teacher_artifacts.jsonl")
                if row["record_type"] == "teacher_rejected_row"
            ]
            self.assertEqual([row["id"] for row in rejected], ["c"])
            self.assertTrue(summary["teacher_resumed_partial"])
            self.assertEqual(summary["teacher_resumed_accepted_rows"], 1)
            self.assertEqual(summary["teacher_resumed_rejected_rows"], 1)
            self.assertEqual(summary["teacher_resumed_retry_rows"], 1)
            self.assertEqual(summary["teacher_requested_candidate_rows"], 4)
            self.assertEqual(summary["teacher_skipped_candidate_rows"], 1)
