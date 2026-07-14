from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_preference_pairs import RowRejected, build_pair_content


def term_error(source: str, bad: str, good: str) -> dict[str, str]:
    return {
        "error_type": "terminology",
        "source_span": source,
        "error_span_target": bad,
        "correction": good,
        "reason_ko": "test",
    }


def build(row: dict[str, object]) -> dict[str, object]:
    pair = build_pair_content(
        row=row,
        subset="subset_000",
        subset_idx=0,
        max_term_chars=64,
        max_term_whitespace_tokens=8,
    )
    assert pair is not None
    return pair


class PreferencePairBuilderTest(unittest.TestCase):
    def test_multiple_annotations_are_jointly_reverted(self) -> None:
        pair = build(
            {
                "id": "row_multi",
                "source": "alpha and beta",
                "student_translation": "오류A 및 오류B",
                "target": "정답A 및 정답B",
                "teacher_errors": [
                    term_error("alpha", "오류A", "정답A"),
                    term_error("beta", "오류B", "정답B"),
                ],
            }
        )
        self.assertEqual(pair["rejected"], "오류A 및 오류B")
        self.assertEqual(pair["term_annotation_count"], 2)
        self.assertEqual(pair["replacement_span_count"], 2)
        self.assertTrue(pair["roundtrip_strict"])

    def test_balanced_repeated_mapping_reverts_every_occurrence(self) -> None:
        pair = build(
            {
                "id": "row_repeat",
                "source": "alpha, alpha",
                "student_translation": "오류, 오류",
                "target": "정답, 정답",
                "teacher_errors": [term_error("alpha", "오류", "정답")],
            }
        )
        self.assertEqual(pair["rejected"], "오류, 오류")
        self.assertEqual(pair["replacement_span_count"], 2)
        self.assertTrue(pair["has_repeated_mapping"])

    def test_ambiguous_repeated_mapping_rejects_whole_row(self) -> None:
        with self.assertRaisesRegex(RowRejected, "ambiguous_repeated_teacher_term"):
            build(
                {
                    "id": "row_ambiguous",
                    "source": "alpha, alpha",
                    "student_translation": "오류",
                    "target": "정답, 정답",
                    "teacher_errors": [term_error("alpha", "오류", "정답")],
                }
            )

    def test_josa_mismatch_is_warning_not_rejection(self) -> None:
        pair = build(
            {
                "id": "row_josa",
                "source": "calf scours",
                "student_translation": "우유소(calf scours)는",
                "target": "송아지 설사병(calf scours)은",
                "teacher_errors": [
                    term_error(
                        "calf scours",
                        "우유소(calf scours)",
                        "송아지 설사병(calf scours)",
                    )
                ],
            }
        )
        self.assertEqual(pair["rejected"], "우유소(calf scours)은")
        self.assertIn("josa_incompatible_after_reversion", pair["quality_flags"])

    def test_duplicate_following_gloss_is_warning_not_rejection(self) -> None:
        pair = build(
            {
                "id": "row_gloss",
                "source": "vicarious liability",
                "student_translation": "Vicarious liability",
                "target": "사용자 책임(vicarious liability)",
                "teacher_errors": [
                    term_error("vicarious liability", "Vicarious liability", "사용자 책임")
                ],
            }
        )
        self.assertEqual(pair["rejected"], "Vicarious liability(vicarious liability)")
        self.assertIn("duplicate_parenthetical_after_reversion", pair["quality_flags"])

    def test_terminal_punctuation_mismatch_is_warning_not_rejection(self) -> None:
        pair = build(
            {
                "id": "row_punctuation",
                "source": "Patents",
                "student_translation": "Patents는",
                "target": "특허:",
                "teacher_errors": [term_error("Patents", "Patents는", "특허:")],
            }
        )
        self.assertEqual(pair["rejected"], "Patents는")
        self.assertIn("terminal_punctuation_mismatch", pair["quality_flags"])

    def test_unique_teacher_term_allows_repeated_student_term(self) -> None:
        pair = build(
            {
                "id": "row_repeated_student",
                "source": "alpha beta",
                "student_translation": "오류, 오류",
                "target": "정답, 다른 표현",
                "teacher_errors": [term_error("alpha", "오류", "정답")],
            }
        )
        self.assertEqual(pair["rejected"], "오류, 다른 표현")
        self.assertEqual(pair["term_mappings"][0]["student_occurrence_count"], 2)

    def test_source_mismatch_and_long_span_are_warnings(self) -> None:
        long_source = " ".join(f"source{i}" for i in range(12))
        pair = build(
            {
                "id": "row_warning",
                "source": "actual source",
                "student_translation": "짧은 오류",
                "target": "짧은 정답",
                "teacher_errors": [term_error(long_source, "짧은 오류", "짧은 정답")],
            }
        )
        self.assertIn("source_span_not_found", pair["quality_flags"])
        self.assertIn("long_source_span_whitespace_tokens", pair["quality_flags"])


if __name__ == "__main__":
    unittest.main()
