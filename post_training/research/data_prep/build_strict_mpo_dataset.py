#!/usr/bin/env python3
"""Build the strict, training-ready subset from the validated mPO artifact.

This stage never repairs a rejected term.  It either preserves the original
candidate/tokenized row byte-for-byte at the JSON-value level or records a
row-level rejection with auditable reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RUNTIME_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

try:
    from .build_mpo_token_masks import percentile, sample_markdown
    from .mpo_data import TRAIN_COLUMNS
except ImportError:  # Direct script execution from the research layout.
    from build_mpo_token_masks import percentile, sample_markdown
    from mpo_data import TRAIN_COLUMNS


FILTER_SCHEMA_VERSION = "dqs_mpo_strict_quality_filter_v1"
FILTER_VERSION = "dqs_mpo_strict_quality_v1"
RESPONSE_WIDE_COVERAGE_THRESHOLD = 0.80

HARD_REJECT_FLAGS = (
    "source_span_not_found",
    "josa_incompatible_after_reversion",
    "terminal_punctuation_mismatch",
    "duplicate_parenthetical_after_reversion",
)

# These warnings describe size, not corruption.  They remain admissible unless
# the response-wide sentence rule below also fires.
AUDIT_ONLY_LONG_FLAGS = (
    "long_source_span_chars",
    "long_source_span_whitespace_tokens",
    "long_student_term_chars",
    "long_student_term_whitespace_tokens",
    "long_teacher_term_chars",
    "long_teacher_term_whitespace_tokens",
)

KNOWN_FLAGS = frozenset(HARD_REJECT_FLAGS) | frozenset(AUDIT_ONLY_LONG_FLAGS)

# A long title/name is allowed.  A replacement ending in a finite Korean
# predicate is sentence-like; combined with >=80% source/chosen/rejected
# coverage, it is a whole-clause annotation rather than a localized term.
KOREAN_FINITE_ENDING = re.compile(
    r"(?:니다|십시오|세요|했다|한다|된다|이다|있다|없다|않다|되었다|하였다|였다)"
    r"(?:[.!?…]*[\"'”’）)\]]*)?$"
)

BOUNDARY_DUPLICATE_SUFFIXES = (
    "있습니다",
    "없습니다",
    "했습니다",
    "였습니다",
    "입니다",
    "합니다",
    "됩니다",
    "으로",
    "에서",
    "에게",
    "까지",
    "부터",
    "보다",
    "처럼",
    "만큼",
)


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates.jsonl",
    )
    parser.add_argument(
        "--tokenized",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs.jsonl",
    )
    parser.add_argument(
        "--parent-contract",
        type=Path,
        default=root / "dataset_contract.json",
    )
    parser.add_argument(
        "--parent-token-summary",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_summary.json",
    )
    parser.add_argument(
        "--output-candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates_strict.jsonl",
    )
    parser.add_argument(
        "--output-tokenized",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs_strict.jsonl",
    )
    parser.add_argument(
        "--quality-rejections",
        type=Path,
        default=root / "analysis" / "strict_quality_rejections.jsonl",
    )
    parser.add_argument(
        "--token-rejections",
        type=Path,
        default=root / "analysis" / "strict_mpo_token_mask_rejections.jsonl",
    )
    parser.add_argument(
        "--token-summary",
        type=Path,
        default=root / "analysis" / "strict_mpo_token_mask_summary.json",
    )
    parser.add_argument(
        "--sample-jsonl",
        type=Path,
        default=root / "analysis" / "strict_mpo_token_mask_sample_10.jsonl",
    )
    parser.add_argument(
        "--sample-markdown",
        type=Path,
        default=root / "analysis" / "STRICT_MPO_TOKEN_MASK_SAMPLE_10.md",
    )
    parser.add_argument(
        "--quality-summary",
        type=Path,
        default=root / "analysis" / "strict_quality_filter_summary.json",
    )
    parser.add_argument(
        "--output-contract",
        type=Path,
        default=root / "dataset_contract_strict.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dump(row) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_candidates(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows: dict[str, dict[str, Any]] = {}
    line_numbers: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            pair_id = str(row.get("pair_id", "")).strip()
            if not pair_id:
                raise ValueError(f"empty pair_id at {path}:{line_number}")
            if pair_id in rows:
                raise ValueError(f"duplicate pair_id={pair_id} at {path}:{line_number}")
            rows[pair_id] = row
            line_numbers[pair_id] = line_number
    return rows, line_numbers


def _all_exact_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        index = text.find(needle, cursor)
        if index < 0:
            return spans
        spans.append((index, index + len(needle)))
        cursor = index + len(needle)


def _union_length(spans: Sequence[tuple[int, int]]) -> int:
    total = 0
    previous_end = -1
    for start, end in sorted(set(spans)):
        if start >= previous_end:
            total += end - start
        elif end > previous_end:
            total += end - previous_end
        previous_end = max(previous_end, end)
    return total


def _visible_coverage(text: str, spans: Sequence[tuple[int, int]]) -> float:
    visible_start = len(text) - len(text.lstrip())
    visible_end = len(text.rstrip())
    denominator = visible_end - visible_start
    if denominator <= 0:
        return 0.0
    clipped = [
        (max(start, visible_start), min(end, visible_end))
        for start, end in spans
        if max(start, visible_start) < min(end, visible_end)
    ]
    return _union_length(clipped) / denominator


def _target_spans(row: Mapping[str, Any], side: str) -> tuple[list[tuple[int, int]], bool]:
    text = str(row.get(side, ""))
    parsed: list[tuple[int, int]] = []
    raw_spans = row.get(f"{side}_term_char_spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        return [], False
    for raw in raw_spans:
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or not all(isinstance(value, int) for value in raw)
        ):
            return [], False
        start, end = int(raw[0]), int(raw[1])
        if not 0 <= start < end <= len(text):
            return [], False
        parsed.append((start, end))
    ordered = sorted(parsed)
    if any(start < previous_end for (_, previous_end), (start, _) in zip(ordered, ordered[1:])):
        return [], False
    return parsed, True


def _source_alignment(row: Mapping[str, Any]) -> tuple[list[tuple[int, int]], list[str]]:
    source = str(row.get("source", ""))
    mappings = row.get("term_mappings")
    if not isinstance(mappings, list) or not mappings:
        return [], ["missing_term_mappings"]
    all_spans: list[tuple[int, int]] = []
    reasons: list[str] = []
    for mapping_index, mapping in enumerate(mappings):
        if not isinstance(mapping, Mapping):
            reasons.append(f"invalid_term_mapping:{mapping_index}")
            continue
        source_terms = mapping.get("source_terms")
        if not isinstance(source_terms, list) or not source_terms:
            reasons.append(f"missing_source_terms:{mapping_index}")
            continue
        mapping_spans: list[tuple[int, int]] = []
        for source_term in source_terms:
            term = str(source_term)
            occurrences = _all_exact_occurrences(source, term)
            if not occurrences:
                reasons.append(f"exact_source_term_not_found:{mapping_index}")
            mapping_spans.extend(occurrences)
        unique_mapping_spans = sorted(set(mapping_spans))
        expected_occurrences = mapping.get("source_occurrence_count")
        if expected_occurrences is not None and int(expected_occurrences) != len(unique_mapping_spans):
            reasons.append(f"source_occurrence_count_mismatch:{mapping_index}")
        all_spans.extend(unique_mapping_spans)
    return sorted(set(all_spans)), reasons


def _sentence_like_term(row: Mapping[str, Any]) -> bool:
    replacements = row.get("term_replacements")
    if not isinstance(replacements, list) or not replacements:
        return False
    for replacement in replacements:
        if not isinstance(replacement, Mapping):
            continue
        for field in ("teacher_term", "student_term"):
            if KOREAN_FINITE_ENDING.search(str(replacement.get(field, "")).strip()):
                return True
    return False


def _duplicated_boundary_suffixes(row: Mapping[str, Any]) -> list[str]:
    rejected = str(row.get("rejected", ""))
    replacements = row.get("term_replacements")
    if not isinstance(replacements, list):
        return []
    duplicates: list[str] = []
    for replacement in replacements:
        if not isinstance(replacement, Mapping):
            continue
        span = replacement.get("rejected_char_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) for value in span)
        ):
            continue
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= len(rejected):
            continue
        term = rejected[start:end]
        suffix = rejected[end:]
        for candidate in BOUNDARY_DUPLICATE_SUFFIXES:
            if term.endswith(candidate) and suffix.startswith(candidate):
                duplicates.append(candidate)
    return list(dict.fromkeys(duplicates))


def classify_quality(row: Mapping[str, Any]) -> QualityDecision:
    reasons: list[str] = []
    flags_value = row.get("quality_flags", [])
    flags = {str(flag) for flag in flags_value} if isinstance(flags_value, list) else set()
    if not isinstance(flags_value, list):
        reasons.append("invalid_quality_flags")
    if bool(row.get("has_quality_warnings")) != bool(flags):
        reasons.append("quality_warning_metadata_mismatch")
    for flag in HARD_REJECT_FLAGS:
        if flag in flags:
            reasons.append(f"quality_flag:{flag}")
    for flag in sorted(flags - KNOWN_FLAGS):
        reasons.append(f"unknown_quality_flag:{flag}")

    source_spans, source_reasons = _source_alignment(row)
    reasons.extend(source_reasons)
    chosen_spans, chosen_valid = _target_spans(row, "chosen")
    rejected_spans, rejected_valid = _target_spans(row, "rejected")
    if not chosen_valid:
        reasons.append("invalid_target_term_spans:chosen")
    if not rejected_valid:
        reasons.append("invalid_target_term_spans:rejected")

    source_coverage = _visible_coverage(str(row.get("source", "")), source_spans)
    chosen_coverage = _visible_coverage(str(row.get("chosen", "")), chosen_spans)
    rejected_coverage = _visible_coverage(str(row.get("rejected", "")), rejected_spans)
    sentence_like = _sentence_like_term(row)
    duplicated_suffixes = _duplicated_boundary_suffixes(row)
    reasons.extend(
        f"duplicated_korean_suffix_after_reversion:{suffix}"
        for suffix in duplicated_suffixes
    )
    exact_source_alignment = bool(source_spans) and not source_reasons
    response_wide = (
        exact_source_alignment
        and chosen_valid
        and rejected_valid
        and source_coverage >= RESPONSE_WIDE_COVERAGE_THRESHOLD
        and chosen_coverage >= RESPONSE_WIDE_COVERAGE_THRESHOLD
        and rejected_coverage >= RESPONSE_WIDE_COVERAGE_THRESHOLD
        and sentence_like
    )
    if response_wide:
        reasons.append("response_wide_sentence_replacement")

    deduplicated_reasons = tuple(dict.fromkeys(reasons))
    return QualityDecision(
        accepted=not deduplicated_reasons,
        reasons=deduplicated_reasons,
        metrics={
            "source_term_char_coverage": round(source_coverage, 6),
            "chosen_term_char_coverage": round(chosen_coverage, 6),
            "rejected_term_char_coverage": round(rejected_coverage, 6),
            "sentence_like_term": sentence_like,
            "duplicated_korean_boundary_suffixes": duplicated_suffixes,
            "response_wide_threshold": RESPONSE_WIDE_COVERAGE_THRESHOLD,
            "response_wide_sentence_replacement": response_wide,
        },
    )


def _primary_reason(reasons: Sequence[str]) -> str:
    prefixes = (
        "quality_flag:source_span_not_found",
        "quality_flag:josa_incompatible_after_reversion",
        "quality_flag:terminal_punctuation_mismatch",
        "quality_flag:duplicate_parenthetical_after_reversion",
        "unknown_quality_flag:",
        "exact_source_term_not_found:",
        "source_occurrence_count_mismatch:",
        "duplicated_korean_suffix_after_reversion:",
        "response_wide_sentence_replacement",
    )
    for prefix in prefixes:
        for reason in reasons:
            if reason.startswith(prefix):
                return reason
    return str(reasons[0])


def _semantic_update(digest: Any, row: Mapping[str, Any]) -> None:
    semantic_row: dict[str, Any] = {"pair_id": str(row["pair_id"])}
    for field in TRAIN_COLUMNS[1:]:
        value = row.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{row['pair_id']}: missing semantic tensor field {field}")
        semantic_row[field] = [int(item) for item in value]
    digest.update(
        json.dumps(
            semantic_row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _assert_parent_contract(
    *,
    candidates_path: Path,
    tokenized_path: Path,
    contract: Mapping[str, Any],
    token_summary: Mapping[str, Any],
) -> None:
    observed_tokenized_sha = sha256(tokenized_path)
    if observed_tokenized_sha != str(contract.get("artifact_sha256")):
        raise ValueError("parent tokenized artifact does not match dataset_contract.json")
    if observed_tokenized_sha != str(token_summary["outputs"]["tokenized_sha256"]):
        raise ValueError("parent tokenized artifact does not match token-mask summary")
    observed_candidate_sha = sha256(candidates_path)
    if observed_candidate_sha != str(token_summary["input"]["candidate_sha256"]):
        raise ValueError("parent candidate artifact does not match token-mask summary")


def main() -> None:
    args = parse_args()
    parent_contract = _load_json(args.parent_contract)
    parent_token_summary = _load_json(args.parent_token_summary)
    _assert_parent_contract(
        candidates_path=args.candidates,
        tokenized_path=args.tokenized,
        contract=parent_contract,
        token_summary=parent_token_summary,
    )
    candidates, candidate_line_numbers = _load_candidates(args.candidates)

    output_paths = (
        args.output_candidates,
        args.output_tokenized,
        args.quality_rejections,
        args.token_rejections,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = {path: path.with_suffix(path.suffix + ".tmp") for path in output_paths}

    counts: Counter[str] = Counter()
    input_flags: Counter[str] = Counter()
    accepted_flags: Counter[str] = Counter()
    reason_occurrences: Counter[str] = Counter()
    primary_reasons: Counter[str] = Counter()
    subset_input: Counter[str] = Counter()
    subset_accepted: Counter[str] = Counter()
    subset_rejected: Counter[str] = Counter()
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    chosen_term_counts: list[int] = []
    rejected_term_counts: list[int] = []
    samples: list[dict[str, Any]] = []
    semantic_digest = hashlib.sha256()
    seen_ids: set[str] = set()

    try:
        with (
            args.tokenized.open("r", encoding="utf-8") as source,
            temporary_paths[args.output_candidates].open("w", encoding="utf-8") as candidate_out,
            temporary_paths[args.output_tokenized].open("w", encoding="utf-8") as tokenized_out,
            temporary_paths[args.quality_rejections].open("w", encoding="utf-8") as rejection_out,
            temporary_paths[args.token_rejections].open("w", encoding="utf-8"),
        ):
            for token_line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                tokenized = json.loads(line)
                if not isinstance(tokenized, dict):
                    raise ValueError(f"expected JSON object at {args.tokenized}:{token_line_number}")
                pair_id = str(tokenized.get("pair_id", "")).strip()
                if not pair_id or pair_id in seen_ids:
                    raise ValueError(f"empty or duplicate tokenized pair_id={pair_id!r}")
                seen_ids.add(pair_id)
                candidate = candidates.get(pair_id)
                if candidate is None:
                    raise ValueError(f"tokenized pair has no candidate: {pair_id}")
                if str(candidate.get("chosen", "")).strip() != str(tokenized.get("chosen", "")):
                    raise ValueError(f"parent candidate/tokenized chosen mismatch: {pair_id}")
                if str(candidate.get("rejected", "")).strip() != str(tokenized.get("rejected", "")):
                    raise ValueError(f"parent candidate/tokenized rejected mismatch: {pair_id}")
                if list(candidate.get("quality_flags", [])) != list(tokenized.get("quality_flags", [])):
                    raise ValueError(f"parent candidate/tokenized quality flag mismatch: {pair_id}")

                counts["input_rows"] += 1
                subset = str(candidate.get("subset", ""))
                subset_input[subset] += 1
                flags = [str(flag) for flag in candidate.get("quality_flags", [])]
                input_flags.update(flags)
                decision = classify_quality(candidate)
                if not decision.accepted:
                    counts["rejected_rows"] += 1
                    subset_rejected[subset] += 1
                    reason_occurrences.update(decision.reasons)
                    primary = _primary_reason(decision.reasons)
                    primary_reasons[primary] += 1
                    ledger_row = {
                        "schema_version": FILTER_SCHEMA_VERSION,
                        "filter_version": FILTER_VERSION,
                        "decision": "reject",
                        "pair_id": pair_id,
                        "candidate_line_number": candidate_line_numbers[pair_id],
                        "tokenized_line_number": token_line_number,
                        "primary_reason": primary,
                        "reasons": list(decision.reasons),
                        "metrics": dict(decision.metrics),
                        "quality_flags": flags,
                        "subset": subset,
                        "source": candidate.get("source"),
                        "chosen": candidate.get("chosen"),
                        "rejected": candidate.get("rejected"),
                        "term_replacements": candidate.get("term_replacements"),
                    }
                    rejection_out.write(_json_dump(ledger_row) + "\n")
                    continue

                counts["accepted_rows"] += 1
                subset_accepted[subset] += 1
                if flags:
                    counts["accepted_rows_with_audit_only_long_warnings"] += 1
                    accepted_flags.update(flags)
                else:
                    counts["accepted_rows_without_quality_warnings"] += 1
                chosen_term_count = int(tokenized["chosen_term_token_count"])
                rejected_term_count = int(tokenized["rejected_term_token_count"])
                counts["chosen_term_tokens"] += chosen_term_count
                counts["rejected_term_tokens"] += rejected_term_count
                counts["rows_with_different_term_token_counts"] += int(
                    bool(tokenized["term_token_lengths_differ"])
                )
                chosen_lengths.append(int(tokenized["chosen_sequence_token_count"]))
                rejected_lengths.append(int(tokenized["rejected_sequence_token_count"]))
                chosen_term_counts.append(chosen_term_count)
                rejected_term_counts.append(rejected_term_count)
                candidate_out.write(_json_dump(candidate) + "\n")
                tokenized_out.write(_json_dump(tokenized) + "\n")
                _semantic_update(semantic_digest, tokenized)
                if (
                    len(samples) < 10
                    and not flags
                    and len(str(tokenized.get("chosen", ""))) <= 500
                    and len(str(tokenized.get("rejected", ""))) <= 500
                ):
                    samples.append(tokenized)

        if counts["input_rows"] != int(parent_contract["row_count"]):
            raise ValueError(
                f"parent row count mismatch: observed={counts['input_rows']}, "
                f"contract={parent_contract['row_count']}"
            )
        if counts["accepted_rows"] + counts["rejected_rows"] != counts["input_rows"]:
            raise ValueError("strict quality partition does not cover every tokenized parent row")
        if len(samples) != 10:
            raise ValueError(f"could not select ten strict samples; found {len(samples)}")
        for path, temporary in temporary_paths.items():
            temporary.replace(path)
    except Exception:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise

    _write_jsonl_atomic(args.sample_jsonl, samples)
    args.sample_markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown_tmp = args.sample_markdown.with_suffix(args.sample_markdown.suffix + ".tmp")
    markdown_tmp.write_text(sample_markdown(samples), encoding="utf-8")
    markdown_tmp.replace(args.sample_markdown)

    strict_candidate_sha = sha256(args.output_candidates)
    strict_tokenized_sha = sha256(args.output_tokenized)
    quality_rejection_sha = sha256(args.quality_rejections)
    token_rejection_sha = sha256(args.token_rejections)
    semantic_sha = semantic_digest.hexdigest()
    token_counts = {
        "input_rows": counts["accepted_rows"],
        "accepted_rows": counts["accepted_rows"],
        "rejected_rows": 0,
        "chosen_term_tokens": counts["chosen_term_tokens"],
        "rejected_term_tokens": counts["rejected_term_tokens"],
        "rows_with_different_term_token_counts": counts[
            "rows_with_different_term_token_counts"
        ],
        "accepted_rows_with_char_quality_warnings": counts[
            "accepted_rows_with_audit_only_long_warnings"
        ],
    }
    token_summary = {
        "schema_version": parent_token_summary["schema_version"],
        "input": {
            "candidate_path": str(args.output_candidates.resolve()),
            "candidate_sha256": strict_candidate_sha,
        },
        "tokenizer": parent_token_summary["tokenizer"],
        "training_contract": parent_token_summary["training_contract"],
        "counts": token_counts,
        "reject_reasons": {},
        "samples": {
            "requested_ids": [str(row["pair_id"]) for row in samples],
            "requested_ids_rejected": [],
            "written_ids": [str(row["pair_id"]) for row in samples],
        },
        "lengths": {
            "chosen_sequence": {
                "min": min(chosen_lengths),
                "p50": percentile(chosen_lengths, 0.50),
                "p95": percentile(chosen_lengths, 0.95),
                "max": max(chosen_lengths),
            },
            "rejected_sequence": {
                "min": min(rejected_lengths),
                "p50": percentile(rejected_lengths, 0.50),
                "p95": percentile(rejected_lengths, 0.95),
                "max": max(rejected_lengths),
            },
            "chosen_term_tokens": {
                "min": min(chosen_term_counts),
                "p50": percentile(chosen_term_counts, 0.50),
                "p95": percentile(chosen_term_counts, 0.95),
                "max": max(chosen_term_counts),
            },
            "rejected_term_tokens": {
                "min": min(rejected_term_counts),
                "p50": percentile(rejected_term_counts, 0.50),
                "p95": percentile(rejected_term_counts, 0.95),
                "max": max(rejected_term_counts),
            },
        },
        "outputs": {
            "tokenized_path": str(args.output_tokenized.resolve()),
            "tokenized_sha256": strict_tokenized_sha,
            "rejections_path": str(args.token_rejections.resolve()),
            "rejections_sha256": token_rejection_sha,
            "sample_jsonl_path": str(args.sample_jsonl.resolve()),
            "sample_markdown_path": str(args.sample_markdown.resolve()),
        },
    }
    _write_json_atomic(args.token_summary, token_summary)

    strict_contract = dict(parent_contract)
    strict_contract.update(
        {
            "artifact_sha256": strict_tokenized_sha,
            "training_semantic_sha256": semantic_sha,
            "row_count": counts["accepted_rows"],
            "strict_quality_filter_version": FILTER_VERSION,
            "strict_quality_filter_schema_version": FILTER_SCHEMA_VERSION,
            "strict_quality_filter_source_sha256": sha256(Path(__file__)),
            "strict_quality_response_wide_coverage_threshold": (
                RESPONSE_WIDE_COVERAGE_THRESHOLD
            ),
            "strict_candidate_sha256": strict_candidate_sha,
            "strict_quality_rejection_ledger_sha256": quality_rejection_sha,
            "parent_artifact_sha256": parent_contract["artifact_sha256"],
            "parent_row_count": parent_contract["row_count"],
            "accepted_audit_only_long_warning_rows": counts[
                "accepted_rows_with_audit_only_long_warnings"
            ],
        }
    )
    _write_json_atomic(args.output_contract, strict_contract)

    quality_summary = {
        "schema_version": FILTER_SCHEMA_VERSION,
        "filter_version": FILTER_VERSION,
        "policy": {
            "repair_or_fallback": "none; row-level accept/reject only",
            "hard_reject_quality_flags": list(HARD_REJECT_FLAGS),
            "audit_only_long_flags": list(AUDIT_ONLY_LONG_FLAGS),
            "unknown_quality_flags": "reject",
            "source_alignment": "case-sensitive exact substring only",
            "duplicated_reversion_boundary_suffixes": list(
                BOUNDARY_DUPLICATE_SUFFIXES
            ),
            "response_wide_sentence_rule": {
                "source_term_char_coverage_gte": RESPONSE_WIDE_COVERAGE_THRESHOLD,
                "chosen_term_char_coverage_gte": RESPONSE_WIDE_COVERAGE_THRESHOLD,
                "rejected_term_char_coverage_gte": RESPONSE_WIDE_COVERAGE_THRESHOLD,
                "requires_korean_finite_ending": True,
            },
        },
        "inputs": {
            "filter_source_path": str(Path(__file__).resolve()),
            "filter_source_sha256": sha256(Path(__file__)),
            "candidates_path": str(args.candidates.resolve()),
            "candidates_sha256": sha256(args.candidates),
            "tokenized_path": str(args.tokenized.resolve()),
            "tokenized_sha256": sha256(args.tokenized),
            "parent_contract_path": str(args.parent_contract.resolve()),
            "parent_contract_sha256": sha256(args.parent_contract),
        },
        "counts": dict(sorted(counts.items())),
        "input_quality_flag_occurrences": dict(sorted(input_flags.items())),
        "accepted_quality_flag_occurrences": dict(sorted(accepted_flags.items())),
        "rejection_reason_occurrences": dict(sorted(reason_occurrences.items())),
        "primary_rejection_reasons": dict(sorted(primary_reasons.items())),
        "subsets": {
            "input": dict(sorted(subset_input.items())),
            "accepted": dict(sorted(subset_accepted.items())),
            "rejected": dict(sorted(subset_rejected.items())),
        },
        "outputs": {
            "strict_candidates_path": str(args.output_candidates.resolve()),
            "strict_candidates_sha256": strict_candidate_sha,
            "strict_tokenized_path": str(args.output_tokenized.resolve()),
            "strict_tokenized_sha256": strict_tokenized_sha,
            "quality_rejections_path": str(args.quality_rejections.resolve()),
            "quality_rejections_sha256": quality_rejection_sha,
            "dataset_contract_path": str(args.output_contract.resolve()),
            "dataset_contract_sha256": sha256(args.output_contract),
            "token_summary_path": str(args.token_summary.resolve()),
            "token_summary_sha256": sha256(args.token_summary),
        },
    }
    _write_json_atomic(args.quality_summary, quality_summary)
    print(json.dumps(quality_summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
