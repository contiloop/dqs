from __future__ import annotations

import unittest

import finalize_preference_datasets as module


class FinalizePreferenceDatasetsTest(unittest.TestCase):
    def test_promoted_review_is_kept(self) -> None:
        pair_id = "subset_002:row_000001518947"
        decision, reason, explanation = module.resolve_review(
            pair_id,
            {"reason_code": "borderline_delimiter_loss"},
        )
        self.assertEqual(decision, "KEEP")
        self.assertEqual(reason, "intact_fragment_or_heading")
        self.assertIn("GOODWILL", explanation)

    def test_unpromoted_review_fails_closed(self) -> None:
        decision, reason, _ = module.resolve_review(
            "subset_000:row_000001490192",
            {"reason_code": "borderline_fragmentation"},
        )
        self.assertEqual(decision, "REJECT")
        self.assertEqual(reason, "severe_truncation")

    def test_full_pair_uses_original_student_not_synthetic_negative(self) -> None:
        row = module.build_full_pair(
            {
                "pair_id": "subset_000:row_x",
                "subset": "subset_000",
                "row_id": "row_x",
                "prompt": "PROMPT",
                "chosen": "TEACHER",
                "rejected": "SYNTHETIC MINIMAL NEGATIVE",
                "student_translation": "ORIGINAL STUDENT",
                "source": "SOURCE",
                "teacher_label": "major_error",
                "qe_score": 0.1,
                "term_annotation_count": 1,
            }
        )
        self.assertEqual(row["chosen"], "TEACHER")
        self.assertEqual(row["rejected"], "ORIGINAL STUDENT")
        self.assertNotEqual(row["rejected"], "SYNTHETIC MINIMAL NEGATIVE")

    def test_known_identical_response_is_excluded_from_all_preferences(self) -> None:
        pair_id = "subset_012:row_000001531000"
        reason_code, explanation = module.PREFERENCE_EXCLUSIONS[pair_id]
        self.assertEqual(
            reason_code, "teacher_student_identical_annotation_misalignment"
        )
        self.assertIn("NFC-identical", explanation)


if __name__ == "__main__":
    unittest.main()
