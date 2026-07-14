from __future__ import annotations

import unittest

import full_preference_data as module


class _Dataset(list):
    @property
    def column_names(self):
        return list(self[0])


class FullPreferenceDataTest(unittest.TestCase):
    def test_validate_rows_accepts_teacher_student_pair(self) -> None:
        rows = _Dataset(
            [
                {
                    "schema_version": "schema",
                    "pair_id": "p1",
                    "prompt": "P",
                    "chosen": "Teacher",
                    "rejected": "Student",
                    "chosen_source": "teacher_post_edit_raw_target",
                    "rejected_source": "student_translation_raw_output",
                }
            ]
        )
        contract = {
            "artifact_schema_version": "schema",
            "row_count": 1,
            "ordered_pair_ids_sha256": module.sha256_lines(["p1"]),
        }
        summary = module.validate_rows(rows, contract)
        self.assertEqual(summary["rows"], 1)

    def test_validate_rows_rejects_identical_pair(self) -> None:
        rows = _Dataset(
            [
                {
                    "schema_version": "schema",
                    "pair_id": "p1",
                    "prompt": "P",
                    "chosen": "same",
                    "rejected": "same",
                    "chosen_source": "teacher_post_edit_raw_target",
                    "rejected_source": "student_translation_raw_output",
                }
            ]
        )
        contract = {
            "artifact_schema_version": "schema",
            "row_count": 1,
            "ordered_pair_ids_sha256": module.sha256_lines(["p1"]),
        }
        with self.assertRaises(ValueError):
            module.validate_rows(rows, contract)


if __name__ == "__main__":
    unittest.main()
