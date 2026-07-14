from __future__ import annotations

import unittest
from typing import Any

try:
    from .build_preference_pairs_strict_v2 import strict_reasons
except ImportError:
    from build_preference_pairs_strict_v2 import strict_reasons


def one_term_pair(
    *,
    source: str,
    chosen: str,
    rejected: str,
    student: str,
    teacher_term: str,
    student_term: str,
    source_term: str | None = None,
    quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    chosen_start = chosen.index(teacher_term)
    rejected_start = rejected.index(student_term)
    resolved_source_term = source if source_term is None else source_term
    return {
        "pair_id": "test:row",
        "source": source,
        "chosen": chosen,
        "rejected": rejected,
        "student_translation": student,
        "quality_flags": list(quality_flags or []),
        "term_mappings": [
            {
                "source_terms": [resolved_source_term],
                "student_term": student_term,
                "teacher_term": teacher_term,
            }
        ],
        "term_replacements": [
            {
                "mapping_index": 0,
                "teacher_term": teacher_term,
                "student_term": student_term,
                "chosen_char_span": [chosen_start, chosen_start + len(teacher_term)],
                "rejected_char_span": [
                    rejected_start,
                    rejected_start + len(student_term),
                ],
            }
        ],
        "chosen_term_char_spans": [
            [chosen_start, chosen_start + len(teacher_term)]
        ],
        "rejected_term_char_spans": [
            [rejected_start, rejected_start + len(student_term)]
        ],
    }


def codes(pair: dict[str, Any]) -> set[str]:
    return {reason.code for reason in strict_reasons(pair, {})}


class StrictV2SynthesisTests(unittest.TestCase):
    def test_clean_local_replacement_passes(self) -> None:
        pair = one_term_pair(
            source="The transfer was processed.",
            source_term="transfer",
            chosen="국제 송금을 처리했다.",
            rejected="transfer를 처리했다.",
            student="transfer를 처리했다.",
            teacher_term="국제 송금",
            student_term="transfer",
        )
        self.assertEqual(codes(pair), set())

    def test_any_base_quality_warning_is_hard_rejected(self) -> None:
        pair = one_term_pair(
            source="term",
            chosen="정답",
            rejected="오답",
            student="오답",
            teacher_term="정답",
            student_term="오답",
            quality_flags=["long_teacher_term_whitespace_tokens"],
        )
        self.assertIn("quality_flag", codes(pair))

    def test_new_duplicate_against_teacher_context_is_rejected(self) -> None:
        pair = one_term_pair(
            source="a huge old master",
            source_term="huge old master",
            chosen="거대하고 음탕한 노장",
            rejected="거대하고 거대하고 더러운 노장",
            student="거대하고 더러운 노장",
            teacher_term="음탕한 노장",
            student_term="거대하고 더러운 노장",
        )
        self.assertIn("introduced_left_boundary_overlap", codes(pair))

    def test_duplicate_already_attested_in_student_is_allowed(self) -> None:
        pair = one_term_pair(
            source="cosmetology, aesthetics",
            source_term="aesthetics",
            chosen="미용, 에스테틱",
            rejected="미용, 미용",
            student="미용, 미용",
            teacher_term="에스테틱",
            student_term="미용",
        )
        self.assertFalse(any("duplicate" in code for code in codes(pair)))

    def test_adjacent_annotated_replacements_are_not_splice_duplicates(self) -> None:
        chosen = "미식축구, 미식축구에서"
        rejected = "축구, 축구에서"
        pair: dict[str, Any] = {
            "pair_id": "test:adjacent",
            "source": "football, football",
            "chosen": chosen,
            "rejected": rejected,
            "student_translation": rejected,
            "quality_flags": [],
            "term_mappings": [
                {
                    "source_terms": ["football"],
                    "student_term": "축구",
                    "teacher_term": "미식축구",
                }
            ],
            "term_replacements": [
                {
                    "mapping_index": 0,
                    "teacher_term": "미식축구",
                    "student_term": "축구",
                    "chosen_char_span": [0, 4],
                    "rejected_char_span": [0, 2],
                },
                {
                    "mapping_index": 0,
                    "teacher_term": "미식축구",
                    "student_term": "축구",
                    "chosen_char_span": [6, 10],
                    "rejected_char_span": [4, 6],
                },
            ],
            "chosen_term_char_spans": [[0, 4], [6, 10]],
            "rejected_term_char_spans": [[0, 2], [4, 6]],
        }
        self.assertFalse(any("duplicate" in code for code in codes(pair)))

    def test_particle_after_delimiter_collision_is_rejected(self) -> None:
        pair = one_term_pair(
            source="UTB",
            chosen="UTB의 정의",
            rejected="UTB)의의 정의",
            student="UTB)의 정의",
            teacher_term="UTB",
            student_term="UTB)의",
        )
        self.assertIn("introduced_particle_or_ending_collision", codes(pair))

    def test_adnominal_ending_plus_particle_collision_is_rejected(self) -> None:
        pair = one_term_pair(
            source="skip",
            chosen="생략을 선택한다",
            rejected="건너뛰는을 선택한다",
            student="건너뛰는 선택",
            teacher_term="생략",
            student_term="건너뛰는",
        )
        self.assertIn("introduced_particle_or_ending_collision", codes(pair))

    def test_stackable_postposition_is_allowed(self) -> None:
        pair = one_term_pair(
            source="from the previous currency",
            chosen="이전 실적 발표에서도 확인된다",
            rejected="이전 통화에서도 확인된다",
            student="이전 통화에서 확인된다",
            teacher_term="이전 실적 발표에서",
            student_term="이전 통화에서",
        )
        self.assertNotIn("introduced_particle_or_ending_collision", codes(pair))

    def test_lexical_syllable_plus_particle_is_not_blanket_rejected(self) -> None:
        pair = one_term_pair(
            source="reverse acquisition",
            chosen="역합병의 완료",
            rejected="역수역합의의 완료",
            student="역수역합의 완료",
            teacher_term="역합병",
            student_term="역수역합의",
        )
        self.assertNotIn("introduced_particle_or_ending_collision", codes(pair))

    def test_delimiter_damage_is_rejected(self) -> None:
        pair = one_term_pair(
            source="the term",
            source_term="term",
            chosen="그는 “용어”를 사용했다.",
            rejected="그는 ““이상한”를 사용했다.",
            student="“이상한 표현",
            teacher_term="용어",
            student_term="“이상한",
        )
        self.assertIn("delimiter_structure_changed", codes(pair))

    def test_teacher_output_delimiter_defect_is_rejected_when_source_balanced(self) -> None:
        pair = one_term_pair(
            source="term",
            chosen="정답)",
            rejected="오답)",
            student="오답",
            teacher_term="정답",
            student_term="오답",
        )
        self.assertIn("chosen_delimiter_unbalanced", codes(pair))

    def test_response_wide_sentence_annotation_is_rejected(self) -> None:
        pair = one_term_pair(
            source="Source sentence.",
            chosen="정답입니다.",
            rejected="오답입니다.",
            student="오답입니다.",
            teacher_term="정답입니다.",
            student_term="오답입니다.",
        )
        self.assertIn("response_wide_sentence_replacement", codes(pair))

    def test_whole_completion_nominal_annotation_is_rejected(self) -> None:
        pair = one_term_pair(
            source="SEGMENTS AND CONCENTRATIONS OF RISK",
            chosen="부문 및 위험 집중도",
            rejected="위험의 부문 및 집중도",
            student="위험의 부문 및 집중도",
            teacher_term="부문 및 위험 집중도",
            student_term="위험의 부문 및 집중도",
        )
        self.assertIn("whole_completion_term_replacement", codes(pair))


if __name__ == "__main__":
    unittest.main()
