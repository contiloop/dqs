#!/usr/bin/env python3
"""Validate the strict quality partition, lineage, hashes, and contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DATA_PREP_DIR = RESEARCH_ROOT / "data_prep"
RUNTIME_SRC = (
    RESEARCH_ROOT.parent / "dqs_preference_training_hf" / "src"
)
for import_path in (DATA_PREP_DIR, RUNTIME_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from .build_strict_mpo_dataset import (
        AUDIT_ONLY_LONG_FLAGS,
        FILTER_SCHEMA_VERSION,
        FILTER_VERSION,
        HARD_REJECT_FLAGS,
        classify_quality,
        sha256,
    )
    from .mpo_data import TRAIN_COLUMNS
except ImportError:  # Direct script execution from the research layout.
    from build_strict_mpo_dataset import (
        AUDIT_ONLY_LONG_FLAGS,
        FILTER_SCHEMA_VERSION,
        FILTER_VERSION,
        HARD_REJECT_FLAGS,
        classify_quality,
        sha256,
    )
    from mpo_data import TRAIN_COLUMNS


def parse_args() -> argparse.Namespace:
    root = RESEARCH_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter-source",
        type=Path,
        default=root / "data_prep" / "build_strict_mpo_dataset.py",
    )
    parser.add_argument(
        "--parent-candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates.jsonl",
    )
    parser.add_argument(
        "--parent-tokenized",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs.jsonl",
    )
    parser.add_argument(
        "--strict-candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates_strict.jsonl",
    )
    parser.add_argument(
        "--strict-tokenized",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs_strict.jsonl",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=root / "analysis" / "strict_quality_rejections.jsonl",
    )
    parser.add_argument(
        "--quality-summary",
        type=Path,
        default=root / "analysis" / "strict_quality_filter_summary.json",
    )
    parser.add_argument(
        "--token-summary",
        type=Path,
        default=root / "analysis" / "strict_mpo_token_mask_summary.json",
    )
    parser.add_argument(
        "--token-validation-report",
        type=Path,
        default=root / "analysis" / "strict_mpo_token_mask_validation_report.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=root / "dataset_contract_strict.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "analysis" / "strict_dataset_validation_report.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return payload


def _load_parent_candidates(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id", ""))
            if not pair_id or pair_id in rows:
                raise AssertionError(f"empty/duplicate parent candidate at line {line_number}")
            rows[pair_id] = row
    return rows


def _load_strict_candidates(
    path: Path,
    *,
    parent: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    pair_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id", ""))
            if pair_id in pair_ids:
                raise AssertionError(f"duplicate strict candidate at line {line_number}: {pair_id}")
            if parent.get(pair_id) != row:
                raise AssertionError(f"strict candidate is not an exact parent row: {pair_id}")
            decision = classify_quality(row)
            if not decision.accepted:
                raise AssertionError(f"rejected row entered strict candidates: {pair_id}")
            flags = {str(flag) for flag in row.get("quality_flags", [])}
            if flags & set(HARD_REJECT_FLAGS):
                raise AssertionError(f"hard warning entered strict candidates: {pair_id}")
            if flags - set(AUDIT_ONLY_LONG_FLAGS):
                raise AssertionError(f"unreviewed warning entered strict candidates: {pair_id}")
            pair_ids.append(pair_id)
    return pair_ids


def _load_rejections(
    path: Path,
    *,
    parent: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], Counter[str]]:
    pair_ids: list[str] = []
    primary_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            ledger = json.loads(line)
            pair_id = str(ledger.get("pair_id", ""))
            if pair_id in pair_ids:
                raise AssertionError(f"duplicate strict rejection at line {line_number}: {pair_id}")
            candidate = parent.get(pair_id)
            if candidate is None:
                raise AssertionError(f"unknown strict rejection pair_id: {pair_id}")
            decision = classify_quality(candidate)
            if decision.accepted:
                raise AssertionError(f"accepted row entered rejection ledger: {pair_id}")
            if list(decision.reasons) != ledger.get("reasons"):
                raise AssertionError(f"rejection reasons do not reproduce: {pair_id}")
            if dict(decision.metrics) != ledger.get("metrics"):
                raise AssertionError(f"rejection metrics do not reproduce: {pair_id}")
            if ledger.get("decision") != "reject":
                raise AssertionError(f"invalid ledger decision: {pair_id}")
            if ledger.get("schema_version") != FILTER_SCHEMA_VERSION:
                raise AssertionError(f"invalid ledger schema: {pair_id}")
            if ledger.get("filter_version") != FILTER_VERSION:
                raise AssertionError(f"invalid ledger filter version: {pair_id}")
            primary_counts[str(ledger["primary_reason"])] += 1
            pair_ids.append(pair_id)
    return pair_ids, primary_counts


def _semantic_update(digest: Any, row: Mapping[str, Any]) -> None:
    semantic_row = {"pair_id": str(row["pair_id"])}
    for field in TRAIN_COLUMNS[1:]:
        semantic_row[field] = [int(value) for value in row[field]]
    digest.update(
        json.dumps(
            semantic_row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _validate_tokenized_partition(
    *,
    parent_path: Path,
    strict_path: Path,
    candidates: Mapping[str, Mapping[str, Any]],
    strict_candidate_ids: list[str],
    rejection_ids: set[str],
) -> tuple[int, str]:
    accepted_index = 0
    parent_ids: set[str] = set()
    semantic_digest = hashlib.sha256()
    with (
        parent_path.open("r", encoding="utf-8") as parent_handle,
        strict_path.open("r", encoding="utf-8") as strict_handle,
    ):
        strict_iterator = iter(strict_handle)
        for parent_line_number, line in enumerate(parent_handle, start=1):
            if not line.strip():
                continue
            parent_row = json.loads(line)
            pair_id = str(parent_row.get("pair_id", ""))
            if not pair_id or pair_id in parent_ids:
                raise AssertionError(
                    f"empty/duplicate parent tokenized pair at line {parent_line_number}"
                )
            parent_ids.add(pair_id)
            candidate = candidates.get(pair_id)
            if candidate is None:
                raise AssertionError(f"parent tokenized row lacks candidate: {pair_id}")
            decision = classify_quality(candidate)
            if decision.accepted:
                while True:
                    try:
                        strict_line = next(strict_iterator)
                    except StopIteration as exc:
                        raise AssertionError("strict tokenized artifact ended early") from exc
                    if strict_line.strip():
                        break
                strict_row = json.loads(strict_line)
                if strict_row != parent_row:
                    raise AssertionError(f"strict tokenized row differs from parent: {pair_id}")
                if accepted_index >= len(strict_candidate_ids):
                    raise AssertionError("strict candidate artifact ended before tokenized artifact")
                if strict_candidate_ids[accepted_index] != pair_id:
                    raise AssertionError(f"strict candidate/tokenized order mismatch: {pair_id}")
                _semantic_update(semantic_digest, strict_row)
                accepted_index += 1
            elif pair_id not in rejection_ids:
                raise AssertionError(f"classified rejection missing from ledger: {pair_id}")

        if any(line.strip() for line in strict_iterator):
            raise AssertionError("strict tokenized artifact has extra rows")
    if accepted_index != len(strict_candidate_ids):
        raise AssertionError("strict candidate artifact has extra rows")
    if parent_ids != set(strict_candidate_ids) | rejection_ids:
        raise AssertionError("strict accept/reject IDs do not partition parent tokenized IDs")
    if set(strict_candidate_ids) & rejection_ids:
        raise AssertionError("strict accept/reject partitions overlap")
    return len(parent_ids), semantic_digest.hexdigest()


def main() -> None:
    args = parse_args()
    quality_summary = _load_json(args.quality_summary)
    token_summary = _load_json(args.token_summary)
    token_validation = _load_json(args.token_validation_report)
    contract = _load_json(args.contract)
    parent_candidates = _load_parent_candidates(args.parent_candidates)
    strict_candidate_ids = _load_strict_candidates(
        args.strict_candidates,
        parent=parent_candidates,
    )
    rejection_id_list, primary_counts = _load_rejections(
        args.rejections,
        parent=parent_candidates,
    )
    parent_count, semantic_sha = _validate_tokenized_partition(
        parent_path=args.parent_tokenized,
        strict_path=args.strict_tokenized,
        candidates=parent_candidates,
        strict_candidate_ids=strict_candidate_ids,
        rejection_ids=set(rejection_id_list),
    )

    strict_count = len(strict_candidate_ids)
    rejection_count = len(rejection_id_list)
    if parent_count != strict_count + rejection_count:
        raise AssertionError("strict row accounting mismatch")
    expected_counts = quality_summary["counts"]
    if int(expected_counts["input_rows"]) != parent_count:
        raise AssertionError("quality summary input count mismatch")
    if int(expected_counts["accepted_rows"]) != strict_count:
        raise AssertionError("quality summary accepted count mismatch")
    if int(expected_counts["rejected_rows"]) != rejection_count:
        raise AssertionError("quality summary rejected count mismatch")
    if dict(sorted(primary_counts.items())) != quality_summary["primary_rejection_reasons"]:
        raise AssertionError("quality summary primary rejection counts mismatch")

    strict_candidate_sha = sha256(args.strict_candidates)
    strict_tokenized_sha = sha256(args.strict_tokenized)
    rejection_sha = sha256(args.rejections)
    outputs = quality_summary["outputs"]
    if outputs["strict_candidates_sha256"] != strict_candidate_sha:
        raise AssertionError("quality summary strict candidate hash mismatch")
    if outputs["strict_tokenized_sha256"] != strict_tokenized_sha:
        raise AssertionError("quality summary strict tokenized hash mismatch")
    if outputs["quality_rejections_sha256"] != rejection_sha:
        raise AssertionError("quality summary rejection hash mismatch")

    if int(contract["row_count"]) != strict_count:
        raise AssertionError("strict contract row count mismatch")
    if contract["artifact_sha256"] != strict_tokenized_sha:
        raise AssertionError("strict contract artifact hash mismatch")
    if contract["training_semantic_sha256"] != semantic_sha:
        raise AssertionError("strict contract semantic hash mismatch")
    if contract["strict_candidate_sha256"] != strict_candidate_sha:
        raise AssertionError("strict contract candidate hash mismatch")
    if contract["strict_quality_rejection_ledger_sha256"] != rejection_sha:
        raise AssertionError("strict contract rejection hash mismatch")
    if contract["strict_quality_filter_version"] != FILTER_VERSION:
        raise AssertionError("strict contract filter version mismatch")
    if contract["strict_quality_filter_source_sha256"] != sha256(args.filter_source):
        raise AssertionError("strict contract filter source hash mismatch")

    if token_summary["input"]["candidate_sha256"] != strict_candidate_sha:
        raise AssertionError("strict token summary candidate hash mismatch")
    if token_summary["outputs"]["tokenized_sha256"] != strict_tokenized_sha:
        raise AssertionError("strict token summary tokenized hash mismatch")
    if int(token_summary["counts"]["accepted_rows"]) != strict_count:
        raise AssertionError("strict token summary row count mismatch")
    if token_validation.get("status") != "passed":
        raise AssertionError("strict independent token validation has not passed")
    if int(token_validation["validated_tokenized_pairs"]) != strict_count:
        raise AssertionError("strict token validation row count mismatch")
    if token_validation["tokenized_sha256"] != strict_tokenized_sha:
        raise AssertionError("strict token validation artifact hash mismatch")

    report = {
        "status": "passed",
        "filter_version": FILTER_VERSION,
        "validated_parent_tokenized_rows": parent_count,
        "validated_strict_rows": strict_count,
        "validated_quality_rejections": rejection_count,
        "accepted_and_rejected_partition_parent_exactly": True,
        "strict_rows_are_exact_parent_rows": True,
        "accepted_warning_policy": list(AUDIT_ONLY_LONG_FLAGS),
        "strict_tokenized_sha256": strict_tokenized_sha,
        "strict_training_semantic_sha256": semantic_sha,
        "invariants": [
            "no term text is repaired or rewritten by the strict stage",
            "strict candidate and tokenized rows equal their parent JSON values",
            "every parent tokenized pair is accepted or appears once in the rejection ledger",
            "hard, unknown, source-alignment, and response-wide sentence failures are rejected",
            "accepted warnings are limited to audited long-span flags",
            "quality summaries, hashes, row counts, and rejection reasons reproduce",
            "dataset contract artifact and training-semantic hashes reproduce",
            "fresh independent Gemma token-mask validation passed for every strict row",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
