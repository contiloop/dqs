#!/usr/bin/env python3
"""Profile whether DQS golden pairs can yield term-local preference pairs.

This script is deliberately read-only with respect to the downloaded golden-pair
files.  It writes aggregate diagnostics and small audit samples; it does not
materialize a training set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TERM_ERROR_TYPE = "terminology"
AUDIT_SAMPLE_SIZE = 25
LOCALITY_PRESETS = {
    "tokens_le_4_chars_le_32": (4, 32),
    "tokens_le_6_chars_le_48": (6, 48),
    "tokens_le_8_chars_le_64": (8, 64),
    "tokens_le_12_chars_le_96": (12, 96),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "raw" / "golden_pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "analysis",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalized_term_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def local_context_matches(
    chosen: str,
    chosen_span: tuple[int, int],
    student: str,
    student_span: tuple[int, int],
    width: int,
) -> tuple[bool, bool]:
    chosen_start, chosen_end = chosen_span
    student_start, student_end = student_span
    chosen_left = chosen[max(0, chosen_start - width) : chosen_start]
    student_left = student[max(0, student_start - width) : student_start]
    chosen_right = chosen[chosen_end : chosen_end + width]
    student_right = student[student_end : student_end + width]
    return chosen_left == student_left, chosen_right == student_right


def hangul_jongseong_index(character: str) -> int | None:
    if len(character) != 1:
        return None
    codepoint = ord(character)
    if not 0xAC00 <= codepoint <= 0xD7A3:
        return None
    return (codepoint - 0xAC00) % 28


def josa_compatibility(rejected: str, rejected_span: tuple[int, int]) -> bool | None:
    """Check common Korean postpositions immediately following the inserted term.

    None means that the case is not safely checkable (no known postposition or
    the inserted term does not end in a Hangul syllable).
    """

    _, end = rejected_span
    suffix = rejected[end:]
    matched = next(
        (
            particle
            for particle in ("으로", "은", "는", "이", "가", "을", "를", "과", "와", "로")
            if suffix.startswith(particle)
        ),
        None,
    )
    if matched is None or end == 0:
        return None
    jongseong = hangul_jongseong_index(rejected[end - 1])
    if jongseong is None:
        return None
    has_batchim = jongseong != 0
    if matched in {"은", "는"}:
        return matched == ("은" if has_batchim else "는")
    if matched in {"이", "가"}:
        return matched == ("이" if has_batchim else "가")
    if matched in {"을", "를"}:
        return matched == ("을" if has_batchim else "를")
    if matched in {"과", "와"}:
        return matched == ("과" if has_batchim else "와")
    if matched in {"으로", "로"}:
        # A final rieul (jongseong index 8) takes 로, like a vowel-final word.
        return matched == ("으로" if has_batchim and jongseong != 8 else "로")
    return None


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def stable_score(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatchResult:
    fields_present: bool
    distinct_terms: bool
    raw_good_in_chosen: int
    raw_bad_in_student: int
    raw_source_in_source: int
    normalized_good_in_chosen: int
    normalized_bad_in_student: int
    normalized_source_in_source: int

    @property
    def raw_exact_unique_without_source(self) -> bool:
        return (
            self.fields_present
            and self.distinct_terms
            and self.raw_good_in_chosen == 1
            and self.raw_bad_in_student == 1
        )

    @property
    def raw_exact_unique(self) -> bool:
        return self.raw_exact_unique_without_source and self.raw_source_in_source == 1

    @property
    def normalized_exact_unique(self) -> bool:
        return (
            self.fields_present
            and self.distinct_terms
            and self.normalized_good_in_chosen == 1
            and self.normalized_bad_in_student == 1
            and self.normalized_source_in_source == 1
        )


def match_result(row: dict[str, Any], error: dict[str, Any]) -> MatchResult:
    source = row.get("source")
    student = row.get("student_translation")
    chosen = row.get("target")
    source_span = error.get("source_span")
    bad = error.get("error_span_target")
    good = error.get("correction")
    fields = (source, student, chosen, source_span, bad, good)
    fields_present = all(nonempty_string(value) for value in fields)
    if not fields_present:
        return MatchResult(False, False, 0, 0, 0, 0, 0, 0)

    assert isinstance(source, str)
    assert isinstance(student, str)
    assert isinstance(chosen, str)
    assert isinstance(source_span, str)
    assert isinstance(bad, str)
    assert isinstance(good, str)
    return MatchResult(
        fields_present=True,
        distinct_terms=nfc(good) != nfc(bad),
        raw_good_in_chosen=chosen.count(good),
        raw_bad_in_student=student.count(bad),
        raw_source_in_source=source.count(source_span),
        normalized_good_in_chosen=nfc(chosen).count(nfc(good)),
        normalized_bad_in_student=nfc(student).count(nfc(bad)),
        normalized_source_in_source=nfc(source).count(nfc(source_span)),
    )


def build_negative(row: dict[str, Any], error: dict[str, Any]) -> tuple[str, tuple[int, int], tuple[int, int]]:
    chosen = row["target"]
    good = error["correction"]
    bad = error["error_span_target"]
    start = chosen.index(good)
    end = start + len(good)
    rejected = chosen[:start] + bad + chosen[end:]
    return rejected, (start, end), (start, start + len(bad))


def replacement_payload(
    row: dict[str, Any],
    error: dict[str, Any],
    subset: str,
    error_index: int,
) -> dict[str, Any]:
    rejected, chosen_span, rejected_span = build_negative(row, error)
    student = row["student_translation"]
    return {
        "subset": subset,
        "id": row.get("id"),
        "teacher_label": row.get("teacher_label"),
        "source": row.get("source"),
        "student_translation": student,
        "chosen": row.get("target"),
        "rejected_synthetic": rejected,
        "reconstructs_student_raw": rejected == student,
        "reconstructs_student_nfc": nfc(rejected) == nfc(student),
        "error_index": error_index,
        "total_teacher_errors": len(row.get("teacher_errors") or []),
        "source_term": error.get("source_span"),
        "student_term": error.get("error_span_target"),
        "teacher_term": error.get("correction"),
        "chosen_term_char_span": list(chosen_span),
        "rejected_term_char_span": list(rejected_span),
        "reason_ko": error.get("reason_ko"),
    }


def add_audit_sample(
    samples: dict[str, list[tuple[str, dict[str, Any]]]],
    category: str,
    payload: dict[str, Any],
) -> None:
    bucket = samples[category]
    bucket.append((stable_score(payload), payload))
    bucket.sort(key=lambda item: item[0])
    del bucket[AUDIT_SAMPLE_SIZE:]


def record_term_usage(
    usage_by_tier: dict[str, dict[str, dict[str, Any]]],
    tier: str,
    payload: dict[str, Any],
) -> None:
    source_term = payload["source_term"]
    student_term = payload["student_term"]
    teacher_term = payload["teacher_term"]
    assert isinstance(source_term, str)
    assert isinstance(student_term, str)
    assert isinstance(teacher_term, str)
    key = normalized_term_key(source_term)
    entry = usage_by_tier[tier].setdefault(
        key,
        {"count": 0, "source_forms": set(), "student_terms": set(), "teacher_terms": set()},
    )
    entry["count"] += 1
    entry["source_forms"].add(source_term)
    entry["student_terms"].add(nfc(student_term))
    entry["teacher_terms"].add(nfc(teacher_term))


def summarize_term_usage(usage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    frequencies = [int(entry["count"]) for entry in usage.values()]
    repeated = {key: entry for key, entry in usage.items() if int(entry["count"]) >= 2}
    ambiguous = {
        key: entry for key, entry in usage.items() if len(entry["teacher_terms"]) >= 2
    }
    return {
        "rows": sum(frequencies),
        "unique_normalized_source_terms": len(usage),
        "singleton_source_terms": sum(1 for value in frequencies if value == 1),
        "source_terms_frequency_ge_2": sum(1 for value in frequencies if value >= 2),
        "source_terms_frequency_ge_3": sum(1 for value in frequencies if value >= 3),
        "source_terms_frequency_ge_5": sum(1 for value in frequencies if value >= 5),
        "rows_covered_by_frequency_ge_2": sum(int(entry["count"]) for entry in repeated.values()),
        "ambiguous_source_terms_with_multiple_teacher_terms": len(ambiguous),
        "rows_covered_by_ambiguous_source_terms": sum(
            int(entry["count"]) for entry in ambiguous.values()
        ),
        "max_source_term_frequency": max(frequencies) if frequencies else 0,
        "top_source_terms": [
            {
                "normalized_source_term": key,
                "count": int(entry["count"]),
                "source_forms": sorted(entry["source_forms"]),
                "student_term_count": len(entry["student_terms"]),
                "teacher_term_count": len(entry["teacher_terms"]),
                "teacher_terms": sorted(entry["teacher_terms"])[:20],
            }
            for key, entry in sorted(
                usage.items(), key=lambda item: (-int(item[1]["count"]), item[0])
            )[:30]
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("subset_*.jsonl"))
    if not input_paths:
        raise SystemExit(f"No subset_*.jsonl files found in {args.input_dir}")

    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    term_errors_per_row: Counter[int] = Counter()
    total_errors_per_term_row: Counter[int] = Counter()
    failure_reasons: Counter[str] = Counter()
    per_subset: dict[str, Counter[str]] = defaultdict(Counter)
    tier_labels: dict[str, Counter[str]] = defaultdict(Counter)
    locality_yields: dict[str, Counter[str]] = defaultdict(Counter)
    surface_diagnostics: Counter[str] = Counter()
    proposed_tiers: Counter[str] = Counter()
    proposed_tier_labels: dict[str, Counter[str]] = defaultdict(Counter)
    term_usage_by_tier: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    span_lengths: dict[str, list[int]] = defaultdict(list)
    span_token_lengths: dict[str, list[int]] = defaultdict(list)
    samples: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    ids: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    candidate_keys: Counter[str] = Counter()

    for path in input_paths:
        subset = path.stem
        for row in read_jsonl(path):
            counts["rows"] += 1
            per_subset[subset]["rows"] += 1
            label = str(row.get("teacher_label") or "<null>")
            labels[label] += 1
            per_subset[subset][f"label_{label}"] += 1
            if nonempty_string(row.get("id")):
                ids[row["id"]] += 1
            if nonempty_string(row.get("source")):
                sources[nfc(row["source"])] += 1

            raw_errors = row.get("teacher_errors")
            if not isinstance(raw_errors, list):
                counts["rows_invalid_teacher_errors"] += 1
                continue
            errors = [error for error in raw_errors if isinstance(error, dict)]
            counts["teacher_errors"] += len(errors)
            counts["non_object_teacher_errors"] += len(raw_errors) - len(errors)
            for error in errors:
                error_types[str(error.get("error_type") or "<null>")] += 1

            term_entries = [
                (index, error)
                for index, error in enumerate(errors)
                if error.get("error_type") == TERM_ERROR_TYPE
            ]
            term_errors_per_row[len(term_entries)] += 1
            if not term_entries:
                continue

            counts["rows_with_terminology"] += 1
            per_subset[subset]["rows_with_terminology"] += 1
            counts["terminology_errors"] += len(term_entries)
            per_subset[subset]["terminology_errors"] += len(term_entries)
            total_errors_per_term_row[len(errors)] += 1
            if len(term_entries) == 1:
                counts["rows_with_exactly_one_terminology"] += 1
                per_subset[subset]["rows_with_exactly_one_terminology"] += 1
            if len(term_entries) == 1 and len(errors) == 1:
                counts["rows_with_only_one_error_and_it_is_terminology"] += 1
                per_subset[subset]["rows_only_one_terminology_error"] += 1

            for error_index, error in term_entries:
                match = match_result(row, error)
                for key, field in (
                    ("source", "source_span"),
                    ("bad", "error_span_target"),
                    ("good", "correction"),
                ):
                    value = error.get(field)
                    if nonempty_string(value):
                        span_lengths[key].append(len(value))
                        span_token_lengths[key].append(whitespace_tokens(value))

                if not match.fields_present:
                    failure_reasons["missing_or_empty_required_field"] += 1
                    add_audit_sample(
                        samples,
                        "missing_field",
                        {"subset": subset, "id": row.get("id"), "error": error},
                    )
                    continue
                counts["term_errors_all_required_fields"] += 1
                if not match.distinct_terms:
                    failure_reasons["same_good_and_bad_after_nfc"] += 1
                    continue
                counts["term_errors_distinct_good_bad"] += 1

                if match.raw_good_in_chosen != 1:
                    failure_reasons[f"good_in_chosen_count_{match.raw_good_in_chosen}"] += 1
                if match.raw_bad_in_student != 1:
                    failure_reasons[f"bad_in_student_count_{match.raw_bad_in_student}"] += 1
                if match.raw_source_in_source != 1:
                    failure_reasons[f"source_span_in_source_count_{match.raw_source_in_source}"] += 1

                if match.normalized_exact_unique and not match.raw_exact_unique:
                    counts["term_errors_recoverable_only_after_nfc"] += 1

                if not match.raw_exact_unique_without_source:
                    add_audit_sample(
                        samples,
                        "not_exact_unique_target_spans",
                        {
                            "subset": subset,
                            "id": row.get("id"),
                            "source": row.get("source"),
                            "student_translation": row.get("student_translation"),
                            "target": row.get("target"),
                            "error": error,
                            "match": match.__dict__,
                        },
                    )
                    continue

                counts["replaceable_term_errors_without_source_anchor"] += 1
                if not match.raw_exact_unique:
                    add_audit_sample(
                        samples,
                        "replaceable_but_source_not_unique",
                        {
                            "subset": subset,
                            "id": row.get("id"),
                            "source": row.get("source"),
                            "error": error,
                            "match": match.__dict__,
                        },
                    )
                    continue

                counts["replaceable_term_errors"] += 1
                per_subset[subset]["replaceable_term_errors"] += 1
                payload = replacement_payload(row, error, subset, error_index)
                record_term_usage(term_usage_by_tier, "all_replaceable", payload)
                if len(term_entries) == 1:
                    record_term_usage(
                        term_usage_by_tier, "one_terminology_error_in_row", payload
                    )
                chosen = payload["chosen"]
                student = payload["student_translation"]
                bad = payload["student_term"]
                assert isinstance(chosen, str)
                assert isinstance(student, str)
                assert isinstance(bad, str)
                chosen_span = tuple(payload["chosen_term_char_span"])
                rejected_span = tuple(payload["rejected_term_char_span"])
                student_start = student.index(bad)
                student_span = (student_start, student_start + len(bad))
                for width in (1, 2, 4, 8):
                    left_equal, right_equal = local_context_matches(
                        chosen,
                        chosen_span,
                        student,
                        student_span,
                        width,
                    )
                    if left_equal:
                        surface_diagnostics[f"local_left_context_{width}_equal"] += 1
                    if right_equal:
                        surface_diagnostics[f"local_right_context_{width}_equal"] += 1
                    if left_equal and right_equal:
                        surface_diagnostics[f"local_both_context_{width}_equal"] += 1

                left_equal_2, right_equal_2 = local_context_matches(
                    chosen,
                    chosen_span,
                    student,
                    student_span,
                    2,
                )

                josa = josa_compatibility(payload["rejected_synthetic"], rejected_span)
                if josa is True:
                    surface_diagnostics["josa_checked_compatible"] += 1
                elif josa is False:
                    surface_diagnostics["josa_checked_incompatible"] += 1
                    add_audit_sample(samples, "josa_incompatible", payload)
                else:
                    surface_diagnostics["josa_not_applicable_or_unknown"] += 1

                term_values = [
                    payload["source_term"],
                    payload["student_term"],
                    payload["teacher_term"],
                ]
                assert all(isinstance(value, str) for value in term_values)
                tiers = ["all_replaceable"]
                tier_labels["all_replaceable"][label] += 1
                if len(term_entries) == 1:
                    tiers.append("one_terminology_error_in_row")
                    tier_labels["one_terminology_error_in_row"][label] += 1
                if len(term_entries) == 1 and len(errors) == 1:
                    tiers.append("only_one_teacher_error_terminology")
                    tier_labels["only_one_teacher_error_terminology"][label] += 1
                if payload["reconstructs_student_raw"]:
                    tiers.append("reconstructs_student_raw")
                    tier_labels["reconstructs_student_raw"][label] += 1
                if len(term_entries) == 1 and len(errors) == 1 and payload["reconstructs_student_raw"]:
                    tiers.append("paper_like_strict")
                    tier_labels["paper_like_strict"][label] += 1

                for preset_name, (max_tokens, max_chars) in LOCALITY_PRESETS.items():
                    passes = all(
                        whitespace_tokens(value) <= max_tokens and len(value) <= max_chars
                        for value in term_values
                    )
                    if passes:
                        for tier in tiers:
                            locality_yields[preset_name][tier] += 1

                max_term_tokens = max(whitespace_tokens(value) for value in term_values)
                max_term_chars = max(len(value) for value in term_values)
                surface_safe = max_term_tokens <= 8 and max_term_chars <= 64 and josa is not False
                if surface_safe:
                    surface_diagnostics["surface_safe_tokens8_chars64_josa"] += 1
                    for width in (1, 2, 4):
                        left_equal, right_equal = local_context_matches(
                            chosen,
                            chosen_span,
                            student,
                            student_span,
                            width,
                        )
                        if left_equal and right_equal:
                            surface_diagnostics[
                                f"surface_safe_and_local_both_context_{width}"
                            ] += 1

                allowed_label = label in {"minor", "major"}
                tier_a = (
                    allowed_label
                    and len(term_entries) == 1
                    and len(errors) == 1
                    and payload["reconstructs_student_raw"]
                    and max_term_tokens <= 8
                    and max_term_chars <= 64
                )
                tier_b = (
                    allowed_label
                    and len(term_entries) == 1
                    and surface_safe
                    and left_equal_2
                    and right_equal_2
                    and not tier_a
                )
                tier_c = (
                    allowed_label
                    and len(term_entries) == 1
                    and surface_safe
                    and not tier_a
                    and not tier_b
                )
                if tier_a:
                    proposed_tiers["tier_a_paper_like_strict"] += 1
                    proposed_tier_labels["tier_a_paper_like_strict"][label] += 1
                    per_subset[subset]["tier_a_paper_like_strict"] += 1
                    record_term_usage(term_usage_by_tier, "tier_a_paper_like_strict", payload)
                    record_term_usage(term_usage_by_tier, "tier_a_plus_b", payload)
                    add_audit_sample(samples, "proposed_tier_a", payload)
                elif tier_b:
                    proposed_tiers["tier_b_controlled_synthetic"] += 1
                    proposed_tier_labels["tier_b_controlled_synthetic"][label] += 1
                    per_subset[subset]["tier_b_controlled_synthetic"] += 1
                    record_term_usage(term_usage_by_tier, "tier_b_controlled_synthetic", payload)
                    record_term_usage(term_usage_by_tier, "tier_a_plus_b", payload)
                    add_audit_sample(samples, "proposed_tier_b", payload)
                elif tier_c:
                    proposed_tiers["tier_c_manual_review_queue"] += 1
                    proposed_tier_labels["tier_c_manual_review_queue"][label] += 1
                    per_subset[subset]["tier_c_manual_review_queue"] += 1
                    record_term_usage(term_usage_by_tier, "tier_c_manual_review_queue", payload)
                    add_audit_sample(samples, "proposed_tier_c", payload)
                key_payload = {
                    "source": payload["source"],
                    "chosen": payload["chosen"],
                    "rejected": payload["rejected_synthetic"],
                    "source_term": payload["source_term"],
                }
                candidate_keys[stable_score(key_payload)] += 1

                if payload["reconstructs_student_raw"]:
                    counts["replaceable_reconstructs_student_raw"] += 1
                    per_subset[subset]["reconstructs_student_raw"] += 1
                    add_audit_sample(samples, "reconstructs_student", payload)
                elif payload["reconstructs_student_nfc"]:
                    counts["replaceable_reconstructs_student_nfc_only"] += 1
                else:
                    add_audit_sample(samples, "synthetic_counterfactual", payload)

                if len(term_entries) == 1:
                    counts["rows_exactly_one_term_and_replaceable"] += 1
                    per_subset[subset]["rows_exactly_one_term_and_replaceable"] += 1
                if len(term_entries) == 1 and len(errors) == 1:
                    counts["rows_only_one_term_error_and_replaceable"] += 1
                    per_subset[subset]["rows_only_one_term_error_and_replaceable"] += 1
                    if payload["reconstructs_student_raw"]:
                        counts["strict_rows_reconstruct_student_raw"] += 1
                        per_subset[subset]["strict_rows_reconstruct_student_raw"] += 1
                    else:
                        add_audit_sample(samples, "strict_annotation_but_not_reconstruction", payload)

    counts["unique_ids"] = len(ids)
    counts["duplicate_id_values"] = sum(1 for count in ids.values() if count > 1)
    counts["duplicate_id_rows_beyond_first"] = sum(count - 1 for count in ids.values() if count > 1)
    counts["unique_normalized_sources"] = len(sources)
    counts["duplicate_source_values"] = sum(1 for count in sources.values() if count > 1)
    counts["duplicate_source_rows_beyond_first"] = sum(
        count - 1 for count in sources.values() if count > 1
    )
    counts["unique_replaceable_candidate_keys"] = len(candidate_keys)
    counts["duplicate_candidate_keys"] = sum(
        1 for count in candidate_keys.values() if count > 1
    )
    counts["duplicate_candidate_rows_beyond_first"] = sum(
        count - 1 for count in candidate_keys.values() if count > 1
    )

    summary = {
        "input": {
            "directory": str(args.input_dir.resolve()),
            "files": len(input_paths),
            "file_names": [path.name for path in input_paths],
        },
        "counts": dict(counts),
        "teacher_labels": dict(labels),
        "error_types": dict(error_types),
        "term_errors_per_row": {str(key): value for key, value in sorted(term_errors_per_row.items())},
        "total_errors_per_terminology_row": {
            str(key): value for key, value in sorted(total_errors_per_term_row.items())
        },
        "match_failure_reasons": dict(failure_reasons.most_common()),
        "replaceable_tier_teacher_labels": {
            tier: dict(counter) for tier, counter in sorted(tier_labels.items())
        },
        "locality_yields": {
            preset: dict(counter) for preset, counter in sorted(locality_yields.items())
        },
        "surface_diagnostics": dict(surface_diagnostics),
        "proposed_noncritical_tiers": dict(proposed_tiers),
        "proposed_noncritical_tier_teacher_labels": {
            tier: dict(counter) for tier, counter in sorted(proposed_tier_labels.items())
        },
        "source_term_repetition": {
            tier: summarize_term_usage(usage)
            for tier, usage in sorted(term_usage_by_tier.items())
        },
        "span_character_lengths": {
            key: distribution(values) for key, values in sorted(span_lengths.items())
        },
        "span_whitespace_token_lengths": {
            key: distribution(values) for key, values in sorted(span_token_lengths.items())
        },
    }
    write_json(args.output_dir / "profile.json", summary)

    subset_columns = [
        "subset",
        "rows",
        "label_no_change",
        "label_minor",
        "label_major",
        "label_critical",
        "rows_with_terminology",
        "terminology_errors",
        "rows_with_exactly_one_terminology",
        "rows_only_one_terminology_error",
        "replaceable_term_errors",
        "rows_exactly_one_term_and_replaceable",
        "rows_only_one_term_error_and_replaceable",
        "strict_rows_reconstruct_student_raw",
        "tier_a_paper_like_strict",
        "tier_b_controlled_synthetic",
        "tier_c_manual_review_queue",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "profile_by_subset.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=subset_columns)
        writer.writeheader()
        for subset in sorted(per_subset):
            row = {column: per_subset[subset].get(column, 0) for column in subset_columns}
            row["subset"] = subset
            writer.writerow(row)

    audit_rows: list[dict[str, Any]] = []
    for category in sorted(samples):
        for _, payload in samples[category]:
            audit_rows.append({"audit_category": category, **payload})
    write_jsonl(args.output_dir / "audit_samples.jsonl", audit_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
