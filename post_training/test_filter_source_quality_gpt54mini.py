from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import filter_source_quality_gpt54mini as module


class SourceQualityFilterTest(unittest.TestCase):
    def test_select_requests_follows_tokenized_order(self) -> None:
        candidates = [
            {"pair_id": "b", "source": "Second source."},
            {"pair_id": "a", "source": "First source."},
            {"pair_id": "unused", "source": "Not tokenized."},
        ]
        tokenized = [{"pair_id": "a"}, {"pair_id": "b"}]
        rows = module._select_requests(candidates, tokenized, None)
        self.assertEqual([row["pair_id"] for row in rows], ["a", "b"])
        self.assertEqual([row["order_idx"] for row in rows], [0, 1])

    def test_validate_keep_requires_empty_evidence(self) -> None:
        valid = {
            "decision": "KEEP",
            "reason_code": "intact_prose",
            "evidence": "",
            "explanation": "The prose is readable and structurally intact.",
        }
        self.assertEqual(module._validate_decision(valid, "Readable source."), valid)
        invalid = dict(valid, evidence="Readable")
        with self.assertRaises(module.FilterError):
            module._validate_decision(invalid, "Readable source.")

    def test_validate_reject_requires_exact_evidence(self) -> None:
        valid = {
            "decision": "REJECT",
            "reason_code": "lost_delimiters",
            "evidence": "202320222021",
            "explanation": "Year columns are fused without delimiters.",
        }
        source = "31,202320222021PRETAX"
        self.assertEqual(module._validate_decision(valid, source), valid)
        invalid = dict(valid, evidence="2023 2022 2021")
        with self.assertRaises(module.FilterError):
            module._validate_decision(invalid, source)

    def test_journal_rejects_source_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            row = {
                "pair_id": "x",
                "source_sha256": "wrong",
                "decision": "KEEP",
                "reason_code": "intact_prose",
                "evidence": "",
                "explanation": "Readable prose.",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            requests = {
                "x": {
                    "pair_id": "x",
                    "source": "Readable prose.",
                    "source_sha256": "expected",
                }
            }
            with self.assertRaises(module.FilterError):
                module._load_journal(path, requests)

    def test_journal_accepts_valid_metadata_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            source = "Readable prose."
            source_hash = module._sha256_text(source)
            row = {
                "pair_id": "x",
                "source_sha256": source_hash,
                "decision": "KEEP",
                "reason_code": "intact_prose",
                "evidence": "",
                "explanation": "Readable prose.",
                "response_id": "response-1",
                "usage": {"prompt_tokens": 1},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            requests = {
                "x": {
                    "pair_id": "x",
                    "source": source,
                    "source_sha256": source_hash,
                }
            }
            loaded = module._load_journal(path, requests)
            self.assertEqual(loaded["x"]["response_id"], "response-1")


if __name__ == "__main__":
    unittest.main()
