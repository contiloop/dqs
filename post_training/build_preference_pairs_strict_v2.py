#!/usr/bin/env python3
"""Build strict mPO preference pairs directly from raw DQS golden pairs.

This is a single-pass, strict-by-construction builder.  A raw terminology row
is either emitted once as a complete preference pair or written once to the
rejection ledger.  It never reads, repairs, or filters a previously synthesized
candidate artifact.

The base character mapping is shared with ``build_preference_pairs.py`` so the
prompt and annotation contracts remain identical.  Before a pair is emitted,
this builder additionally proves that transplanting the student's annotated
term into the teacher post-edit does not introduce a new boundary duplicate,
Korean particle/ending collision, delimiter defect, or response-wide sentence
replacement.  Long annotation spans are rejected rather than retained as
warnings.  There is no trimming, fallback, or partial-row recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .build_preference_pairs import (
        HF_REPO,
        HF_REPO_REVISION,
        HF_RUN,
        TERM_ERROR_TYPE,
        PromptResolver,
        RowRejected,
        build_pair_content,
        candidate_sort_key,
        file_sha256,
        read_jsonl,
        samples_markdown,
        select_review_samples,
    )
except ImportError:  # Direct script execution.
    from build_preference_pairs import (
        HF_REPO,
        HF_REPO_REVISION,
        HF_RUN,
        TERM_ERROR_TYPE,
        PromptResolver,
        RowRejected,
        build_pair_content,
        candidate_sort_key,
        file_sha256,
        read_jsonl,
        samples_markdown,
        select_review_samples,
    )


STRICT_SCHEMA_VERSION = "dqs_mpo_strict_synthesis_v2"
SYNTHESIS_VERSION = "dqs_teacher_postedit_term_reversion_v4_strict_by_construction"
MAX_TERM_CHARS = 64
MAX_TERM_WHITESPACE_TOKENS = 8
RESPONSE_WIDE_COVERAGE_THRESHOLD = 0.75

KOREAN_FINITE_ENDING = re.compile(
    r"(?:니다|십시오|세요|했다|한다|된다|이다|있다|없다|않다|되었다|하였다|였다)"
    r"(?:[.!?…]*[\"'”’）)\]]*)?$"
)
LEXICAL_TOKEN = re.compile(r"[A-Za-z0-9]+|[가-힣]+")
LEADING_PARTICLES = tuple(
    sorted(
        (
            "으로써",
            "으로서",
            "에게서",
            "한테서",
            "으로",
            "에서",
            "에게",
            "한테",
            "까지",
            "부터",
            "처럼",
            "보다",
            "마저",
            "조차",
            "밖에",
            "이랑",
            "이라도",
            "이라고",
            "이라는",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "과",
            "와",
            "의",
            "에",
            "도",
            "만",
            "로",
            "께",
        ),
        key=len,
        reverse=True,
    )
)
MULTI_CHAR_SYNTACTIC_ENDINGS = (
    "으로써",
    "으로서",
    "에게서",
    "한테서",
    "으로",
    "에서",
    "에게",
    "한테",
    "까지",
    "부터",
)
STACKABLE_AFTER_SYNTACTIC_ENDING = (
    "은",
    "는",
    "도",
    "만",
    "조차",
    "마저",
    "의",
    "부터",
    "까지",
)
ADNOMINAL_ENDINGS = ("는", "던")
NOMINAL_PARTICLES = ("을", "를", "이", "가", "은", "는")
EXPLICIT_TRAILING_PARTICLE = re.compile(r"[)\]}>”’'\"][은는이가을를의]$")
PAIRED_DELIMITERS = (
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("“", "”"),
    ("「", "」"),
    ("『", "』"),
)


@dataclass(frozen=True)
class StrictReason:
    code: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "raw" / "golden_pairs")
    parser.add_argument("--repo-root", type=Path, default=root.parent)
    parser.add_argument(
        "--effective-config",
        type=Path,
        default=root / "raw" / "effective_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "prepared" / "preference_candidates_strict_v2.jsonl",
    )
    parser.add_argument(
        "--roundtrip-output",
        type=Path,
        default=root / "prepared" / "roundtrip_strict_v2.jsonl",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=root / "analysis" / "strict_v2_synthesis_rejections.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "analysis" / "strict_v2_synthesis_summary.json",
    )
    parser.add_argument(
        "--sample-jsonl",
        type=Path,
        default=root / "analysis" / "strict_v2_sample_10.jsonl",
    )
    parser.add_argument(
        "--sample-markdown",
        type=Path,
        default=root / "analysis" / "STRICT_V2_SAMPLE_10.md",
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        default=root / "analysis" / "strict_v2_sample_ids.txt",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    return parser.parse_args()


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


def _delimiter_state(text: str, opening: str, closing: str) -> tuple[int, int]:
    depth = 0
    minimum = 0
    for character in text:
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            minimum = min(minimum, depth)
    return depth, minimum


def delimiter_profile(text: str) -> dict[str, tuple[int, int] | int]:
    profile: dict[str, tuple[int, int] | int] = {
        opening + closing: _delimiter_state(text, opening, closing)
        for opening, closing in PAIRED_DELIMITERS
    }
    # Straight double quotes are symmetric.  Only parity is a structural
    # invariant; changing two straight quotes to a balanced Korean quote pair
    # is allowed.
    profile['"'] = text.count('"') % 2
    return profile


def delimiter_profile_is_balanced(profile: Mapping[str, tuple[int, int] | int]) -> bool:
    for value in profile.values():
        if isinstance(value, tuple):
            if value != (0, 0):
                return False
        elif value != 0:
            return False
    return True


def _maximum_overlap(left: str, right: str, *, limit: int = 64) -> str:
    for length in range(min(limit, len(left), len(right)), 1, -1):
        if left[-length:] == right[:length]:
            overlap = left[-length:]
            if overlap.strip():
                return overlap
    return ""


def _lexical_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in LEXICAL_TOKEN.finditer(text)]


def _contains_lexical_text(text: str) -> bool:
    return LEXICAL_TOKEN.search(text) is not None


def _left_boundary_fragment(prefix: str, term: str) -> str:
    prefix_matches = list(LEXICAL_TOKEN.finditer(prefix))
    term_match = LEXICAL_TOKEN.search(term)
    if not prefix_matches or term_match is None:
        return ""
    return prefix[prefix_matches[-1].start() :] + term[: term_match.end()]


def _right_boundary_fragment(term: str, suffix: str) -> str:
    term_matches = list(LEXICAL_TOKEN.finditer(term))
    suffix_match = LEXICAL_TOKEN.search(suffix)
    if not term_matches or suffix_match is None:
        return ""
    return term[term_matches[-1].start() :] + suffix[: suffix_match.end()]


def _leading_particle(text: str) -> str | None:
    return next((particle for particle in LEADING_PARTICLES if text.startswith(particle)), None)


def _original_term_is_detached(student: str, term: str) -> bool:
    cursor = 0
    suffixes: list[str] = []
    while True:
        start = student.find(term, cursor)
        if start < 0:
            break
        end = start + len(term)
        suffixes.append(student[end : end + 1])
        cursor = end
    if not suffixes:
        return False
    return all(
        not suffix or suffix.isspace() or suffix in ".,;:!?)]}>”’"
        for suffix in suffixes
    )


def _syntactic_term_ending(term: str, following_particle: str) -> str | None:
    stripped = term.rstrip()
    if EXPLICIT_TRAILING_PARTICLE.search(stripped):
        return "explicit_particle_after_delimiter"
    multi = next(
        (ending for ending in MULTI_CHAR_SYNTACTIC_ENDINGS if stripped.endswith(ending)),
        None,
    )
    if multi is not None and following_particle not in STACKABLE_AFTER_SYNTACTIC_ENDING:
        return multi
    if following_particle in NOMINAL_PARTICLES and stripped.endswith(ADNOMINAL_ENDINGS):
        return stripped[-1]
    return None


def _visible_coverage(text: str, spans: Sequence[tuple[int, int]]) -> float:
    visible_indices = {index for index, character in enumerate(text) if not character.isspace()}
    if not visible_indices:
        return 0.0
    covered = {
        index
        for start, end in spans
        for index in range(max(0, start), min(len(text), end))
        if not text[index].isspace()
    }
    return len(covered) / len(visible_indices)


def _source_spans(pair: Mapping[str, Any]) -> list[tuple[int, int]]:
    source = str(pair["source"])
    spans: list[tuple[int, int]] = []
    for mapping in pair["term_mappings"]:
        for raw_term in mapping["source_terms"]:
            term = str(raw_term)
            cursor = 0
            while True:
                start = source.find(term, cursor)
                if start < 0:
                    break
                spans.append((start, start + len(term)))
                cursor = start + len(term)
    return sorted(set(spans))


def _target_spans(pair: Mapping[str, Any], side: str) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in pair[f"{side}_term_char_spans"]]


def strict_reasons(pair: Mapping[str, Any], raw_row: Mapping[str, Any]) -> list[StrictReason]:
    """Return every strict rejection reason for one synthesized pair."""

    reasons: list[StrictReason] = []
    quality_flags = [str(flag) for flag in pair.get("quality_flags", [])]
    reasons.extend(
        StrictReason("quality_flag", flag)
        for flag in quality_flags
    )

    source = str(pair["source"])
    chosen = str(pair["chosen"])
    rejected = str(pair["rejected"])
    student = str(pair["student_translation"])

    source_profile = delimiter_profile(source)
    chosen_profile = delimiter_profile(chosen)
    rejected_profile = delimiter_profile(rejected)
    if delimiter_profile_is_balanced(source_profile) and not delimiter_profile_is_balanced(
        chosen_profile
    ):
        reasons.append(
            StrictReason(
                "chosen_delimiter_unbalanced",
                f"source={source_profile}, chosen={chosen_profile}",
            )
        )
    if chosen_profile != rejected_profile:
        reasons.append(
            StrictReason(
                "delimiter_structure_changed",
                f"chosen={chosen_profile}, rejected={rejected_profile}",
            )
        )

    mappings = pair["term_mappings"]
    replacements = pair["term_replacements"]
    replacement_spans = [
        tuple(int(value) for value in replacement["rejected_char_span"])
        for replacement in replacements
    ]
    for replacement_index, replacement in enumerate(replacements):
        mapping_index = int(replacement["mapping_index"])
        mapping = mappings[mapping_index]
        student_term = str(mapping["student_term"])
        start, end = (int(value) for value in replacement["rejected_char_span"])
        prefix = rejected[:start]
        suffix = rejected[end:]

        # A lexical boundary supplied by another replacement is not a splice
        # defect: both sides are explicitly annotated error terms and both are
        # masked.  Only a collision with unchanged teacher context is tested.
        left_is_replacement_boundary = False
        if replacement_index > 0:
            previous_end = replacement_spans[replacement_index - 1][1]
            left_is_replacement_boundary = not _contains_lexical_text(
                rejected[previous_end:start]
            )
        right_is_replacement_boundary = False
        if replacement_index + 1 < len(replacement_spans):
            next_start = replacement_spans[replacement_index + 1][0]
            right_is_replacement_boundary = not _contains_lexical_text(
                rejected[end:next_start]
            )

        left_overlap = _maximum_overlap(prefix, student_term)
        left_overlap_fragment = (
            prefix[-len(left_overlap) :] + student_term[: len(left_overlap)]
            if left_overlap
            else ""
        )
        if (
            left_overlap
            and not left_is_replacement_boundary
            and left_overlap_fragment not in student
        ):
            reasons.append(
                StrictReason(
                    "introduced_left_boundary_overlap",
                    f"replacement={replacement_index}, overlap={left_overlap!r}",
                )
            )
        right_overlap = _maximum_overlap(student_term, suffix)
        right_overlap_fragment = (
            student_term[-len(right_overlap) :] + suffix[: len(right_overlap)]
            if right_overlap
            else ""
        )
        if (
            right_overlap
            and not right_is_replacement_boundary
            and right_overlap_fragment not in student
        ):
            reasons.append(
                StrictReason(
                    "introduced_right_boundary_overlap",
                    f"replacement={replacement_index}, overlap={right_overlap!r}",
                )
            )

        prefix_tokens = _lexical_tokens(prefix)
        term_tokens = _lexical_tokens(student_term)
        suffix_tokens = _lexical_tokens(suffix)
        if prefix_tokens and term_tokens and prefix_tokens[-1] == term_tokens[0]:
            token = term_tokens[0]
            boundary_fragment = _left_boundary_fragment(prefix, student_term)
            if (
                len(token) >= 2
                and not left_is_replacement_boundary
                and boundary_fragment not in student
            ):
                reasons.append(
                    StrictReason(
                        "introduced_left_boundary_token_duplicate",
                        f"replacement={replacement_index}, token={token!r}",
                    )
                )
        if term_tokens and suffix_tokens and term_tokens[-1] == suffix_tokens[0]:
            token = term_tokens[-1]
            boundary_fragment = _right_boundary_fragment(student_term, suffix)
            if (
                len(token) >= 2
                and not right_is_replacement_boundary
                and boundary_fragment not in student
            ):
                reasons.append(
                    StrictReason(
                        "introduced_right_boundary_token_duplicate",
                        f"replacement={replacement_index}, token={token!r}",
                    )
                )

        following_particle = _leading_particle(suffix)
        if following_particle is not None and _original_term_is_detached(student, student_term):
            ending = _syntactic_term_ending(student_term, following_particle)
            if ending is not None:
                reasons.append(
                    StrictReason(
                        "introduced_particle_or_ending_collision",
                        (
                            f"replacement={replacement_index}, ending={ending!r}, "
                            f"following={following_particle!r}"
                        ),
                    )
                )

    source_spans = _source_spans(pair)
    chosen_spans = _target_spans(pair, "chosen")
    rejected_spans = _target_spans(pair, "rejected")
    sentence_like = any(
        KOREAN_FINITE_ENDING.search(str(replacement[field]).strip())
        for replacement in pair["term_replacements"]
        for field in ("teacher_term", "student_term")
    )
    coverage = {
        "source": _visible_coverage(source, source_spans),
        "chosen": _visible_coverage(chosen, chosen_spans),
        "rejected": _visible_coverage(rejected, rejected_spans),
    }
    if coverage["chosen"] == 1.0 and coverage["rejected"] == 1.0:
        reasons.append(
            StrictReason(
                "whole_completion_term_replacement",
                f"coverage={coverage}",
            )
        )
    if sentence_like and all(
        value >= RESPONSE_WIDE_COVERAGE_THRESHOLD for value in coverage.values()
    ):
        reasons.append(
            StrictReason(
                "response_wide_sentence_replacement",
                f"coverage={coverage}, threshold={RESPONSE_WIDE_COVERAGE_THRESHOLD}",
            )
        )

    # Deduplicate deterministic reasons while retaining discovery order.
    deduplicated: list[StrictReason] = []
    seen: set[tuple[str, str]] = set()
    for reason in reasons:
        key = (reason.code, reason.detail)
        if key not in seen:
            seen.add(key)
            deduplicated.append(reason)
    return deduplicated


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("subset_*.jsonl"))
    if not input_paths:
        raise SystemExit(f"No subset_*.jsonl files found in {args.input_dir}")

    resolver = PromptResolver(
        repo_root=args.repo_root,
        effective_config_path=args.effective_config,
    )
    counts: Counter[str] = Counter()
    rejection_codes: Counter[str] = Counter()
    accepted_labels: Counter[str] = Counter()
    accepted_annotations: Counter[int] = Counter()
    per_subset: dict[str, Counter[str]] = defaultdict(Counter)
    candidates: list[dict[str, Any]] = []
    rejection_ledger: list[dict[str, Any]] = []

    for path in input_paths:
        subset = path.stem
        try:
            subset_idx = int(subset.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Cannot parse subset index from {path.name}") from exc
        for row in read_jsonl(path):
            counts["raw_rows"] += 1
            per_subset[subset]["raw_rows"] += 1
            raw_errors = row.get("teacher_errors")
            term_count = (
                sum(
                    1
                    for error in raw_errors
                    if isinstance(error, Mapping) and error.get("error_type") == TERM_ERROR_TYPE
                )
                if isinstance(raw_errors, list)
                else 0
            )
            if term_count == 0:
                counts["rows_without_terminology"] += 1
                continue
            counts["terminology_rows"] += 1
            counts["terminology_annotations"] += term_count
            per_subset[subset]["terminology_rows"] += 1

            pair_id = f"{subset}:{row.get('id')}"
            try:
                pair = build_pair_content(
                    row=row,
                    subset=subset,
                    subset_idx=subset_idx,
                    max_term_chars=MAX_TERM_CHARS,
                    max_term_whitespace_tokens=MAX_TERM_WHITESPACE_TOKENS,
                )
                assert pair is not None
                prompt, prompt_status = resolver.resolve(row=row, subset_idx=subset_idx)
                pair["prompt"] = prompt
                pair["prompt_status"] = prompt_status
            except RowRejected as exc:
                counts["rows_rejected_base_mapping"] += 1
                rejection_codes[f"base_mapping:{exc.code}"] += 1
                per_subset[subset]["rows_rejected"] += 1
                rejection_ledger.append(
                    {
                        "schema_version": STRICT_SCHEMA_VERSION,
                        "synthesis_version": SYNTHESIS_VERSION,
                        "decision": "reject",
                        "stage": "base_mapping",
                        "pair_id": pair_id,
                        "subset": subset,
                        "subset_idx": subset_idx,
                        "row_id": row.get("id"),
                        "teacher_label": row.get("teacher_label"),
                        "term_annotation_count": term_count,
                        "primary_reason": f"base_mapping:{exc.code}",
                        "reasons": [
                            {"code": f"base_mapping:{exc.code}", "detail": exc.detail}
                        ],
                    }
                )
                continue

            reasons = strict_reasons(pair, row)
            if reasons:
                counts["rows_rejected_strict_checks"] += 1
                per_subset[subset]["rows_rejected"] += 1
                for reason in reasons:
                    rejection_codes[reason.code] += 1
                rejection_ledger.append(
                    {
                        "schema_version": STRICT_SCHEMA_VERSION,
                        "synthesis_version": SYNTHESIS_VERSION,
                        "decision": "reject",
                        "stage": "strict_checks",
                        "pair_id": pair_id,
                        "subset": subset,
                        "subset_idx": subset_idx,
                        "row_id": row.get("id"),
                        "teacher_label": row.get("teacher_label"),
                        "term_annotation_count": term_count,
                        "primary_reason": reasons[0].code,
                        "reasons": [reason.as_dict() for reason in reasons],
                        "quality_flags": pair.get("quality_flags", []),
                        "term_mappings": pair.get("term_mappings", []),
                    }
                )
                continue

            pair["schema_version"] = STRICT_SCHEMA_VERSION
            pair["synthesis_version"] = SYNTHESIS_VERSION
            pair["synthesis_policy"] = (
                "strict_by_construction_all_terminology_annotations_or_reject_row"
            )
            pair["has_quality_warnings"] = False
            pair["quality_flags"] = []
            pair["strict_synthesis_checks"] = {
                "repair_or_fallback": "none",
                "long_span_policy": "reject",
                "boundary_overlap": "reject_if_introduced_by_transplant",
                "particle_or_ending_collision": "reject",
                "delimiter_structure_change": "reject",
                "whole_completion_term_replacement": "reject",
                "response_wide_sentence_coverage_threshold": (
                    RESPONSE_WIDE_COVERAGE_THRESHOLD
                ),
            }
            counts["rows_accepted"] += 1
            per_subset[subset]["rows_accepted"] += 1
            accepted_labels[str(pair.get("teacher_label") or "<null>")] += 1
            accepted_annotations[int(pair["term_annotation_count"])] += 1
            counts["accepted_replacement_spans"] += int(pair["replacement_span_count"])
            counts["roundtrip_strict_rows"] += int(bool(pair["roundtrip_strict"]))
            candidates.append(pair)

    candidates.sort(key=candidate_sort_key)
    ids = [str(pair["pair_id"]) for pair in candidates]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate pair_id values in strict-v2 candidates")
    rejected_rows = (
        counts["rows_rejected_base_mapping"] + counts["rows_rejected_strict_checks"]
    )
    if counts["terminology_rows"] != counts["rows_accepted"] + rejected_rows:
        raise AssertionError("Strict-v2 terminology row accounting mismatch")

    _write_jsonl_atomic(args.output, candidates)
    _write_jsonl_atomic(
        args.roundtrip_output,
        (pair for pair in candidates if pair["roundtrip_strict"]),
    )
    _write_jsonl_atomic(args.rejections, rejection_ledger)
    sample_ids_path = args.sample_ids if args.sample_ids.exists() else None
    samples = select_review_samples(candidates, args.sample_size, sample_ids_path)
    _write_jsonl_atomic(args.sample_jsonl, samples)
    args.sample_markdown.parent.mkdir(parents=True, exist_ok=True)
    sample_tmp = args.sample_markdown.with_suffix(args.sample_markdown.suffix + ".tmp")
    sample_tmp.write_text(samples_markdown(samples), encoding="utf-8")
    sample_tmp.replace(args.sample_markdown)

    input_manifest = {
        path.name: {"rows": sum(1 for _ in read_jsonl(path)), "sha256": file_sha256(path)}
        for path in input_paths
    }
    manifest_digest = hashlib.sha256()
    for name, payload in sorted(input_manifest.items()):
        manifest_digest.update(
            f"{payload['sha256']}  {name}\n".encode("utf-8")
        )
    summary = {
        "schema_version": STRICT_SCHEMA_VERSION,
        "synthesis_version": SYNTHESIS_VERSION,
        "implementation": {
            "strict_builder_path": str(Path(__file__).resolve()),
            "strict_builder_sha256": file_sha256(Path(__file__).resolve()),
            "base_mapping_builder_path": str(
                Path(__file__).with_name("build_preference_pairs.py").resolve()
            ),
            "base_mapping_builder_sha256": file_sha256(
                Path(__file__).with_name("build_preference_pairs.py").resolve()
            ),
        },
        "policy": {
            "input": "raw golden_pairs only; no parent candidate artifact",
            "positive": "teacher target unchanged",
            "negative": "teacher target with every accepted terminology mapping reverted",
            "row_atomicity": "all terminology annotations pass or the entire row is rejected",
            "repair_or_fallback": "none",
            "max_term_chars": MAX_TERM_CHARS,
            "max_term_whitespace_tokens": MAX_TERM_WHITESPACE_TOKENS,
            "long_spans": "reject",
            "whole_completion_term_replacement": "reject",
            "response_wide_sentence_coverage_threshold": (
                RESPONSE_WIDE_COVERAGE_THRESHOLD
            ),
            "tokenization": "not run in this stage",
        },
        "input": {
            "directory": str(args.input_dir.resolve()),
            "file_count": len(input_paths),
            "manifest_sha256": manifest_digest.hexdigest(),
            "files": input_manifest,
            "hf_repo": HF_REPO,
            "hf_run": HF_RUN,
            "hf_repo_revision": HF_REPO_REVISION,
        },
        "counts": dict(sorted(counts.items())),
        "accepted_teacher_labels": _counter_dict(accepted_labels),
        "accepted_term_annotation_count_distribution": _counter_dict(
            accepted_annotations
        ),
        "rejection_reason_occurrences": dict(sorted(rejection_codes.items())),
        "subsets": {
            subset: dict(sorted(counter.items()))
            for subset, counter in sorted(per_subset.items())
        },
        "outputs": {
            "candidate_path": str(args.output.resolve()),
            "candidate_sha256": file_sha256(args.output),
            "roundtrip_path": str(args.roundtrip_output.resolve()),
            "roundtrip_sha256": file_sha256(args.roundtrip_output),
            "rejections_path": str(args.rejections.resolve()),
            "rejections_sha256": file_sha256(args.rejections),
            "sample_jsonl_path": str(args.sample_jsonl.resolve()),
            "sample_markdown_path": str(args.sample_markdown.resolve()),
        },
    }
    _write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
