#!/usr/bin/env python3
"""Validate strict-v2 synthesis against the raw golden-pair source of truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

DATA_PREP_DIR = Path(__file__).resolve().parents[1] / "data_prep"
if str(DATA_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PREP_DIR))

try:
    from .build_preference_pairs import TERM_ERROR_TYPE, file_sha256, read_jsonl
    from .build_preference_pairs_strict_v2 import (
        STRICT_SCHEMA_VERSION,
        SYNTHESIS_VERSION,
        strict_reasons,
    )
    from .validate_preference_pairs import validate_pair
except ImportError:
    from build_preference_pairs import TERM_ERROR_TYPE, file_sha256, read_jsonl
    from build_preference_pairs_strict_v2 import (
        STRICT_SCHEMA_VERSION,
        SYNTHESIS_VERSION,
        strict_reasons,
    )
    from validate_preference_pairs import validate_pair


KNOWN_BAD_PAIR_IDS = frozenset(
    {
        "subset_000:row_000001022425",
        "subset_001:row_000001475319",
        "subset_001:row_000001568899",
        "subset_002:row_000000021673",
        "subset_002:row_000000541750",
        "subset_003:row_000002257035",
        "subset_004:row_000000478480",
        "subset_004:row_000000633139",
        "subset_005:row_000000517989",
        "subset_006:row_000000131566",
        "subset_006:row_000001288489",
        "subset_007:row_000000348974",
        "subset_007:row_000001007046",
        "subset_007:row_000002036440",
        "subset_008:row_000001593725",
        "subset_008:row_000001657797",
        "subset_009:row_000000177886",
        "subset_009:row_000000382620",
        "subset_010:row_000000448170",
        "subset_010:row_000001008664",
        "subset_011:row_000000560786",
        "subset_011:row_000001484613",
        "subset_012:row_000001976345",
        "subset_016:row_000000152184",
        "subset_016:row_000000564892",
        "subset_017:row_000001195636",
        "subset_017:row_000002115601",
        "subset_019:row_000000316555",
        "subset_019:row_000000496771",
        "subset_019:row_000001706899",
        "subset_020:row_000001684566",
        "subset_020:row_000002121122",
        "subset_021:row_000000552252",
        "subset_022:row_000000722138",
    }
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "raw" / "golden_pairs")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates_strict_v2.jsonl",
    )
    parser.add_argument(
        "--roundtrip",
        type=Path,
        default=root / "prepared" / "roundtrip_strict_v2.jsonl",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=root / "analysis" / "strict_v2_synthesis_rejections.jsonl",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=root / "analysis" / "strict_v2_sample_10.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "analysis" / "strict_v2_synthesis_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "analysis" / "strict_v2_synthesis_validation_report.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AssertionError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def terminology_count(row: Mapping[str, Any]) -> int:
    errors = row.get("teacher_errors")
    if not isinstance(errors, list):
        return 0
    return sum(
        1
        for error in errors
        if isinstance(error, Mapping) and error.get("error_type") == TERM_ERROR_TYPE
    )


def input_manifest_digest(manifest: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(manifest.items()):
        digest.update(f"{payload['sha256']}  {name}\n".encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    candidates = load_jsonl(args.candidates)
    roundtrip = load_jsonl(args.roundtrip)
    rejections = load_jsonl(args.rejections)
    samples = load_jsonl(args.samples)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    raw_by_id: dict[str, dict[str, Any]] = {}
    terminology_ids: set[str] = set()
    manifest: dict[str, dict[str, Any]] = {}
    raw_rows = 0
    for path in sorted(args.input_dir.glob("subset_*.jsonl")):
        subset = path.stem
        file_rows = 0
        for row in read_jsonl(path):
            file_rows += 1
            raw_rows += 1
            pair_id = f"{subset}:{row.get('id')}"
            if pair_id in raw_by_id:
                raise AssertionError(f"Duplicate raw pair_id: {pair_id}")
            raw_by_id[pair_id] = row
            if terminology_count(row):
                terminology_ids.add(pair_id)
        manifest[path.name] = {"rows": file_rows, "sha256": file_sha256(path)}

    candidate_ids = [str(pair.get("pair_id", "")) for pair in candidates]
    rejection_ids = [str(row.get("pair_id", "")) for row in rejections]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AssertionError("Duplicate candidate pair_id")
    if len(rejection_ids) != len(set(rejection_ids)):
        raise AssertionError("Duplicate rejection pair_id")
    accepted = set(candidate_ids)
    rejected = set(rejection_ids)
    if accepted & rejected:
        raise AssertionError("Accepted and rejected pair_id sets overlap")
    if accepted | rejected != terminology_ids:
        missing = sorted(terminology_ids - accepted - rejected)[:10]
        extra = sorted((accepted | rejected) - terminology_ids)[:10]
        raise AssertionError(f"Terminology partition mismatch: missing={missing}, extra={extra}")

    replacement_spans = 0
    roundtrip_expected: set[str] = set()
    for pair in candidates:
        pair_id = str(pair["pair_id"])
        raw = raw_by_id[pair_id]
        validate_pair(pair)
        if pair.get("schema_version") != STRICT_SCHEMA_VERSION:
            raise AssertionError(f"{pair_id}: schema version mismatch")
        if pair.get("synthesis_version") != SYNTHESIS_VERSION:
            raise AssertionError(f"{pair_id}: synthesis version mismatch")
        if pair.get("quality_flags") != [] or pair.get("has_quality_warnings") is not False:
            raise AssertionError(f"{pair_id}: accepted row retains quality warnings")
        if pair.get("strict_synthesis_checks", {}).get("repair_or_fallback") != "none":
            raise AssertionError(f"{pair_id}: repair/fallback contract mismatch")
        remaining = strict_reasons(pair, raw)
        if remaining:
            raise AssertionError(f"{pair_id}: strict reason remains: {remaining}")
        expected_values = {
            "source": raw.get("source"),
            "student_translation": raw.get("student_translation"),
            "chosen": raw.get("target"),
        }
        for field, expected in expected_values.items():
            if pair.get(field) != expected:
                raise AssertionError(f"{pair_id}: raw provenance mismatch for {field}")
        if int(pair.get("term_annotation_count", -1)) != terminology_count(raw):
            raise AssertionError(f"{pair_id}: raw terminology count mismatch")
        replacement_spans += int(pair["replacement_span_count"])
        if pair["roundtrip_strict"]:
            roundtrip_expected.add(pair_id)

    observed_roundtrip = {str(row["pair_id"]) for row in roundtrip}
    if observed_roundtrip != roundtrip_expected:
        raise AssertionError("Roundtrip artifact does not equal the candidate subset")
    sample_ids = [str(row.get("pair_id", "")) for row in samples]
    if len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= accepted:
        raise AssertionError("Samples are duplicated or outside the accepted candidate set")
    if accepted & KNOWN_BAD_PAIR_IDS:
        raise AssertionError(f"Known-bad rows accepted: {sorted(accepted & KNOWN_BAD_PAIR_IDS)}")

    for row in rejections:
        pair_id = str(row.get("pair_id", ""))
        if pair_id not in raw_by_id or not row.get("reasons") or not row.get("primary_reason"):
            raise AssertionError(f"{pair_id}: incomplete rejection ledger row")
        if row.get("decision") != "reject" or row.get("schema_version") != STRICT_SCHEMA_VERSION:
            raise AssertionError(f"{pair_id}: invalid rejection metadata")

    counts = summary["counts"]
    assertions = {
        "raw_rows": raw_rows,
        "terminology_rows": len(terminology_ids),
        "rows_accepted": len(candidates),
        "roundtrip_strict_rows": len(roundtrip),
        "accepted_replacement_spans": replacement_spans,
    }
    for field, observed in assertions.items():
        if int(counts.get(field, -1)) != observed:
            raise AssertionError(
                f"Summary count mismatch for {field}: {counts.get(field)} != {observed}"
            )
    if int(counts["rows_rejected_base_mapping"]) + int(
        counts["rows_rejected_strict_checks"]
    ) != len(rejections):
        raise AssertionError("Summary rejection count mismatch")
    if summary["input"]["files"] != manifest:
        raise AssertionError("Raw input manifest mismatch")
    if summary["input"]["manifest_sha256"] != input_manifest_digest(manifest):
        raise AssertionError("Raw input manifest digest mismatch")
    output_hashes = {
        "candidate_sha256": args.candidates,
        "roundtrip_sha256": args.roundtrip,
        "rejections_sha256": args.rejections,
    }
    for field, path in output_hashes.items():
        if summary["outputs"][field] != file_sha256(path):
            raise AssertionError(f"Output checksum mismatch for {field}")

    report = {
        "status": "passed",
        "schema_version": STRICT_SCHEMA_VERSION,
        "synthesis_version": SYNTHESIS_VERSION,
        "validated_raw_rows": raw_rows,
        "validated_terminology_rows": len(terminology_ids),
        "validated_candidates": len(candidates),
        "validated_rejections": len(rejections),
        "validated_roundtrip_rows": len(roundtrip),
        "validated_replacement_spans": replacement_spans,
        "validated_samples": len(samples),
        "candidate_sha256": file_sha256(args.candidates),
        "raw_manifest_sha256": summary["input"]["manifest_sha256"],
        "known_bad_rows_absent": len(KNOWN_BAD_PAIR_IDS),
        "invariants": [
            "raw terminology rows partition exactly into accepted or rejected",
            "source/student/teacher fields equal the raw golden row byte-for-byte",
            "every terminology annotation is represented atomically",
            "only recorded term spans differ between chosen and rejected",
            "rejected term spans reconstruct chosen byte-for-byte",
            "accepted rows have no warnings, repair, or fallback",
            "roundtrip/sample subsets and all output checksums are exact",
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
