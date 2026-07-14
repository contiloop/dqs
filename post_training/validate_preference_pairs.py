#!/usr/bin/env python3
"""Independently validate the generated character-span preference artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates.jsonl",
    )
    parser.add_argument(
        "--roundtrip",
        type=Path,
        default=root / "prepared" / "roundtrip_strict.jsonl",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=root / "analysis" / "multi_term_rejections.jsonl",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=root / "analysis" / "sample_10.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "analysis" / "multi_term_synthesis_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "analysis" / "validation_report.json",
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
                raise AssertionError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def span(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise AssertionError(f"Invalid span: {value!r}")
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise AssertionError(f"Invalid span bounds: {value!r}")
    return start, end


def validate_pair(pair: dict[str, Any]) -> None:
    pair_id = pair.get("pair_id")
    for field in ("pair_id", "source", "prompt", "chosen", "rejected", "student_translation"):
        if not nonempty(pair.get(field)):
            raise AssertionError(f"{pair_id}: empty {field}")
    chosen = str(pair["chosen"])
    rejected = str(pair["rejected"])
    if chosen == rejected:
        raise AssertionError(f"{pair_id}: chosen equals rejected")

    mappings = pair.get("term_mappings")
    replacements = pair.get("term_replacements")
    if not isinstance(mappings, list) or not isinstance(replacements, list):
        raise AssertionError(f"{pair_id}: missing mapping/replacement lists")
    if len(replacements) != pair.get("replacement_span_count"):
        raise AssertionError(f"{pair_id}: replacement count mismatch")
    if len(mappings) != pair.get("term_mapping_count"):
        raise AssertionError(f"{pair_id}: mapping count mismatch")

    error_indices = {
        error_index
        for mapping in mappings
        for error_index in mapping.get("error_indices", [])
    }
    if len(error_indices) != pair.get("term_annotation_count"):
        raise AssertionError(f"{pair_id}: terminology annotation coverage mismatch")

    ordered = sorted(replacements, key=lambda item: span(item["chosen_char_span"]))
    if ordered != replacements:
        raise AssertionError(f"{pair_id}: replacements are not in chosen order")
    chosen_cursor = 0
    rejected_cursor = 0
    recovered_parts: list[str] = []
    previous_chosen_end = 0
    previous_rejected_end = 0
    for replacement in replacements:
        mapping_index = replacement.get("mapping_index")
        if not isinstance(mapping_index, int) or not 0 <= mapping_index < len(mappings):
            raise AssertionError(f"{pair_id}: invalid mapping index")
        mapping = mappings[mapping_index]
        teacher_term = str(mapping["teacher_term"])
        student_term = str(mapping["student_term"])
        chosen_start, chosen_end = span(replacement["chosen_char_span"])
        rejected_start, rejected_end = span(replacement["rejected_char_span"])
        if chosen_start < previous_chosen_end or rejected_start < previous_rejected_end:
            raise AssertionError(f"{pair_id}: overlapping spans")
        if chosen[chosen_start:chosen_end] != teacher_term:
            raise AssertionError(f"{pair_id}: chosen span does not match teacher term")
        if rejected[rejected_start:rejected_end] != student_term:
            raise AssertionError(f"{pair_id}: rejected span does not match student term")
        chosen_gap = chosen[chosen_cursor:chosen_start]
        rejected_gap = rejected[rejected_cursor:rejected_start]
        if chosen_gap != rejected_gap:
            raise AssertionError(f"{pair_id}: non-term text differs before a replacement")
        recovered_parts.extend((rejected_gap, teacher_term))
        chosen_cursor = chosen_end
        rejected_cursor = rejected_end
        previous_chosen_end = chosen_end
        previous_rejected_end = rejected_end
    if chosen[chosen_cursor:] != rejected[rejected_cursor:]:
        raise AssertionError(f"{pair_id}: non-term suffix differs")
    recovered_parts.append(rejected[rejected_cursor:])
    if "".join(recovered_parts) != chosen:
        raise AssertionError(f"{pair_id}: rejected spans do not reconstruct chosen")

    expected_roundtrip = rejected == str(pair["student_translation"])
    if bool(pair.get("roundtrip_strict")) != expected_roundtrip:
        raise AssertionError(f"{pair_id}: roundtrip flag mismatch")
    if pair.get("all_terminology_errors_reverted") is not True:
        raise AssertionError(f"{pair_id}: atomic terminology flag is not true")


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    roundtrip = read_jsonl(args.roundtrip)
    rejections = read_jsonl(args.rejections)
    samples = read_jsonl(args.samples)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    for pair in candidates:
        validate_pair(pair)

    candidate_ids = [str(pair["pair_id"]) for pair in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AssertionError("Duplicate candidate pair_id")
    candidate_id_set = set(candidate_ids)
    roundtrip_ids = {str(pair["pair_id"]) for pair in roundtrip}
    expected_roundtrip_ids = {
        str(pair["pair_id"]) for pair in candidates if pair["roundtrip_strict"]
    }
    if roundtrip_ids != expected_roundtrip_ids:
        raise AssertionError("roundtrip_strict.jsonl does not match candidate flags")
    sample_ids = [str(pair["pair_id"]) for pair in samples]
    if len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= candidate_id_set:
        raise AssertionError("sample_10.jsonl is duplicated or not a candidate subset")

    counts = summary["counts"]
    if len(candidates) != counts["rows_accepted"]:
        raise AssertionError("summary accepted count mismatch")
    if len(rejections) != counts["rows_rejected"]:
        raise AssertionError("summary rejected count mismatch")
    if len(roundtrip) != counts["roundtrip_strict_rows"]:
        raise AssertionError("summary roundtrip count mismatch")
    if len(candidates) + len(rejections) != counts["rows_with_terminology"]:
        raise AssertionError("terminology row accounting mismatch")
    candidate_digest = sha256(args.candidates)
    if candidate_digest != summary["outputs"]["candidate_sha256"]:
        raise AssertionError("candidate SHA-256 mismatch")

    report = {
        "status": "passed",
        "validated_candidates": len(candidates),
        "validated_rejections": len(rejections),
        "validated_roundtrip_rows": len(roundtrip),
        "validated_samples": len(samples),
        "candidate_sha256": candidate_digest,
        "invariants": [
            "unique pair_id",
            "all term annotations covered",
            "chosen/rejected spans non-overlapping and exact",
            "all non-term text identical",
            "rejected term spans reconstruct chosen byte-for-byte",
            "roundtrip flags and subset exact",
            "summary counts and candidate checksum exact",
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
