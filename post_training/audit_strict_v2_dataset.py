#!/usr/bin/env python3
"""Run final semantic-quality audit checks on the strict-v2 train artifact."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_preference_pairs import file_sha256
    from .build_preference_pairs_strict_v2 import strict_reasons
    from .validate_preference_pairs_strict_v2 import KNOWN_BAD_PAIR_IDS
except ImportError:
    from build_preference_pairs import file_sha256
    from build_preference_pairs_strict_v2 import strict_reasons
    from validate_preference_pairs_strict_v2 import KNOWN_BAD_PAIR_IDS


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates_strict_v2.jsonl",
    )
    parser.add_argument(
        "--tokenized",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs_strict_v2.jsonl",
    )
    parser.add_argument(
        "--token-rejections",
        type=Path,
        default=root / "analysis" / "strict_v2_mpo_token_mask_rejections.jsonl",
    )
    parser.add_argument(
        "--synthesis-rejections",
        type=Path,
        default=root / "analysis" / "strict_v2_synthesis_rejections.jsonl",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=root / "dataset_contract_strict_v2.json",
    )
    parser.add_argument(
        "--old-tokenized",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs_strict.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "analysis" / "strict_v2_final_audit_report.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def normalized_term(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text).casefold()
        if character.isalnum()
    )


def main() -> None:
    args = parse_args()
    candidate_rows = read_jsonl(args.candidates)
    tokenized_rows = read_jsonl(args.tokenized)
    token_rejections = read_jsonl(args.token_rejections)
    synthesis_rejections = read_jsonl(args.synthesis_rejections)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))

    candidates = {str(row["pair_id"]): row for row in candidate_rows}
    final_ids = [str(row["pair_id"]) for row in tokenized_rows]
    token_rejection_ids = [str(row["pair_id"]) for row in token_rejections]
    if len(candidates) != len(candidate_rows):
        raise AssertionError("Duplicate candidate pair_id")
    if len(final_ids) != len(set(final_ids)):
        raise AssertionError("Duplicate final pair_id")
    if len(token_rejection_ids) != len(set(token_rejection_ids)):
        raise AssertionError("Duplicate token rejection pair_id")
    final_id_set = set(final_ids)
    token_rejection_set = set(token_rejection_ids)
    if final_id_set & token_rejection_set:
        raise AssertionError("Final and token rejection sets overlap")
    if final_id_set | token_rejection_set != set(candidates):
        raise AssertionError("Token stage does not partition strict char candidates")
    if len(tokenized_rows) != int(contract["row_count"]):
        raise AssertionError("Contract row count mismatch")
    if file_sha256(args.tokenized) != contract["artifact_sha256"]:
        raise AssertionError("Contract artifact hash mismatch")

    labels: Counter[str] = Counter()
    annotation_counts: Counter[int] = Counter()
    subsets: Counter[str] = Counter()
    orthography_only: list[dict[str, Any]] = []
    replacement_spans = 0
    roundtrip_rows = 0
    multiple_annotation_rows = 0
    repeated_mapping_rows = 0
    for tokenized in tokenized_rows:
        pair_id = str(tokenized["pair_id"])
        candidate = candidates[pair_id]
        remaining = strict_reasons(candidate, {})
        if remaining:
            raise AssertionError(f"{pair_id}: strict synthesis reason remains: {remaining}")
        if candidate.get("quality_flags") or candidate.get("has_quality_warnings"):
            raise AssertionError(f"{pair_id}: quality warning remains")
        for mapping in candidate["term_mappings"]:
            student_term = str(mapping["student_term"])
            teacher_term = str(mapping["teacher_term"])
            if (
                len(student_term) > 64
                or len(teacher_term) > 64
                or len(student_term.split()) > 8
                or len(teacher_term.split()) > 8
            ):
                raise AssertionError(f"{pair_id}: long target term remains")
            if normalized_term(student_term) == normalized_term(teacher_term):
                orthography_only.append(
                    {
                        "pair_id": pair_id,
                        "student_term": student_term,
                        "teacher_term": teacher_term,
                        "reason_ko": mapping.get("reason_ko", []),
                    }
                )
        for side in ("chosen", "rejected"):
            completion_tokens = sum(int(value) for value in tokenized[f"{side}_completion_mask"])
            term_tokens = sum(int(value) for value in tokenized[f"{side}_term_mask"])
            if term_tokens <= 0:
                raise AssertionError(f"{pair_id}: empty {side} term mask")
            # EOS is the one supervised completion token that is deliberately
            # excluded from the term mask. At least one other completion token
            # must remain outside the term mask.
            if term_tokens >= completion_tokens - 1:
                raise AssertionError(f"{pair_id}: {side} term mask covers the whole completion")

        labels[str(candidate["teacher_label"])] += 1
        annotation_counts[int(candidate["term_annotation_count"])] += 1
        subsets[str(candidate["subset"])] += 1
        replacement_spans += int(candidate["replacement_span_count"])
        roundtrip_rows += int(bool(candidate["roundtrip_strict"]))
        multiple_annotation_rows += int(bool(candidate["has_multiple_term_annotations"]))
        repeated_mapping_rows += int(bool(candidate["has_repeated_mapping"]))

    known_bad_present = sorted(final_id_set & KNOWN_BAD_PAIR_IDS)
    if known_bad_present:
        raise AssertionError(f"Known-bad pair IDs remain: {known_bad_present}")

    comparison: dict[str, Any] | None = None
    if args.old_tokenized.exists():
        old_ids = {str(row["pair_id"]) for row in read_jsonl(args.old_tokenized)}
        removed_ids = old_ids - final_id_set
        synthesis_by_id = {
            str(row["pair_id"]): row for row in synthesis_rejections
        }
        comparison = {
            "old_rows": len(old_ids),
            "new_rows": len(final_id_set),
            "new_only_rows": len(final_id_set - old_ids),
            "removed_from_old_rows": len(removed_ids),
            "removed_primary_reasons": dict(
                sorted(
                    Counter(
                        str(synthesis_by_id[pair_id]["primary_reason"])
                        for pair_id in removed_ids
                    ).items()
                )
            ),
        }

    report = {
        "status": "passed",
        "final_rows": len(tokenized_rows),
        "strict_char_candidates": len(candidate_rows),
        "token_boundary_rejections": len(token_rejections),
        "replacement_spans": replacement_spans,
        "roundtrip_rows": roundtrip_rows,
        "multiple_annotation_rows": multiple_annotation_rows,
        "repeated_mapping_rows": repeated_mapping_rows,
        "teacher_labels": dict(sorted(labels.items())),
        "term_annotation_count_distribution": {
            str(key): value for key, value in sorted(annotation_counts.items())
        },
        "subset_rows": dict(sorted(subsets.items())),
        "known_bad_rows_present": known_bad_present,
        "quality_warning_rows": 0,
        "long_target_term_rows": 0,
        "whole_completion_term_mask_rows": 0,
        "orthography_only_mapping_count": len(orthography_only),
        "orthography_only_mappings": orthography_only,
        "artifact_sha256": file_sha256(args.tokenized),
        "training_semantic_sha256": contract["training_semantic_sha256"],
        "comparison_to_v1": comparison,
        "invariants": [
            "strict char candidates partition exactly into final or token-boundary rejection",
            "no synthesis rejection reason remains in final rows",
            "no quality warning or long target term remains",
            "term masks are non-empty and do not cover whole completions",
            "all 34 previously identified malformed pairs are absent",
            "artifact row count and SHA256 match the immutable contract",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
