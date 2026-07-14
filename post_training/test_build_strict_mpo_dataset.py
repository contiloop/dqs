from __future__ import annotations

import unittest

from build_strict_mpo_dataset import classify_quality


def _row(
    *,
    source: str = "The complaint was consolidated under In re Example Litigation.",
    chosen: str = "해당 소송은 In re Example Litigation이라는 사건명으로 통합되었습니다.",
    rejected: str = "해당 소송은 In re Example 소송이라는 사건명으로 통합되었습니다.",
    source_term: str = "In re Example Litigation",
    teacher_term: str = "In re Example Litigation이라는 사건명으로",
    student_term: str = "In re Example 소송이라는 사건명으로",
    quality_flags: list[str] | None = None,
) -> dict:
    chosen_start = chosen.index(teacher_term)
    rejected_start = rejected.index(student_term)
    flags = list(quality_flags or [])
    return {
        "pair_id": "pair-1",
        "source": source,
        "chosen": chosen,
        "rejected": rejected,
        "quality_flags": flags,
        "has_quality_warnings": bool(flags),
        "chosen_term_char_spans": [[chosen_start, chosen_start + len(teacher_term)]],
        "rejected_term_char_spans": [[rejected_start, rejected_start + len(student_term)]],
        "term_mappings": [
            {
                "source_terms": [source_term],
                "source_occurrence_count": source.count(source_term),
            }
        ],
        "term_replacements": [
            {
                "teacher_term": teacher_term,
                "student_term": student_term,
                "chosen_char_span": [chosen_start, chosen_start + len(teacher_term)],
                "rejected_char_span": [rejected_start, rejected_start + len(student_term)],
            }
        ],
    }


class StrictQualityFilterTest(unittest.TestCase):
    def test_hard_quality_flag_rejects_without_repair(self) -> None:
        decision = classify_quality(
            _row(quality_flags=["josa_incompatible_after_reversion"])
        )
        self.assertFalse(decision.accepted)
        self.assertIn(
            "quality_flag:josa_incompatible_after_reversion",
            decision.reasons,
        )

    def test_embedded_long_legal_name_is_kept(self) -> None:
        decision = classify_quality(
            _row(quality_flags=["long_teacher_term_whitespace_tokens"])
        )
        self.assertTrue(decision.accepted)

    def test_response_wide_sentence_is_rejected(self) -> None:
        chosen = "대손상각은 대손충당금에서 차감됩니다."
        rejected = "대손상각은 대손충당금에 대항하여 처리됩니다."
        decision = classify_quality(
            _row(
                source="Loan write-downs are charged against the allowance.",
                chosen=chosen,
                rejected=rejected,
                source_term="Loan write-downs are charged against the allowance.",
                teacher_term=chosen,
                student_term=rejected,
                quality_flags=["long_source_span_whitespace_tokens"],
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("response_wide_sentence_replacement", decision.reasons)

    def test_response_wide_nominal_title_is_kept(self) -> None:
        chosen = "전자급 황산 합작투자."
        rejected = "전자 등급 황산 연합 사업."
        decision = classify_quality(
            _row(
                source="Electronic Grade Sulfuric Acid Joint Venture.",
                chosen=chosen,
                rejected=rejected,
                source_term="Electronic Grade Sulfuric Acid Joint Venture.",
                teacher_term="전자급 황산 합작투자",
                student_term="전자 등급 황산 연합 사업",
            )
        )
        self.assertTrue(decision.accepted)

    def test_missing_exact_source_term_is_rejected(self) -> None:
        decision = classify_quality(_row(source_term="not present"))
        self.assertFalse(decision.accepted)
        self.assertTrue(
            any(reason.startswith("exact_source_term_not_found:") for reason in decision.reasons)
        )

    def test_unknown_warning_flag_is_rejected(self) -> None:
        decision = classify_quality(_row(quality_flags=["new_unreviewed_warning"]))
        self.assertFalse(decision.accepted)
        self.assertIn("unknown_quality_flag:new_unreviewed_warning", decision.reasons)

    def test_duplicated_suffix_at_reversion_boundary_is_rejected(self) -> None:
        chosen = "그것은 도로 절개지입니다."
        rejected = "그것은 도로입니다입니다."
        decision = classify_quality(
            _row(
                source="That is a road cut.",
                chosen=chosen,
                rejected=rejected,
                source_term="road cut",
                teacher_term="도로 절개지",
                student_term="도로입니다",
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn(
            "duplicated_korean_suffix_after_reversion:입니다",
            decision.reasons,
        )


if __name__ == "__main__":
    unittest.main()
