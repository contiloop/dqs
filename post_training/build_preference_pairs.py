#!/usr/bin/env python3
"""Build controlled multi-term preference candidates from DQS golden pairs.

The preferred completion is the teacher post-edit.  The rejected completion is
constructed from that same post-edit by reverting every safely alignable
terminology correction in the row to the student's annotated term.  Rows are
handled all-or-nothing: if any terminology annotation cannot be mapped safely,
the row is written to the rejection ledger instead of being partially used.

This is a character-span artifact.  Model/tokenizer-specific term masks and
sequence-length checks intentionally happen in a later preparation stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


TERM_ERROR_TYPE = "terminology"
SYNTHESIS_VERSION = "dqs_teacher_postedit_term_reversion_v3_mechanical_exact"
HF_REPO = "alwaysgood/dqs-runs"
HF_RUN = "gemma4_e2b_it_full_iter_lowqe_sf_on_seed42"
HF_REPO_REVISION = "a58b1878988efcecc9a2644f8324bd00131864b5"


class RowRejected(ValueError):
    """A terminology row that does not meet the exact synthesis contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def as_list(self) -> list[int]:
        return [self.start, self.end]


@dataclass
class MappingGroup:
    student_term: str
    teacher_term: str
    source_terms: list[str]
    error_indices: list[int]
    reasons_ko: list[str]
    source_occurrence_count: int
    student_occurrence_count: int
    student_occurrences: list[Span]
    teacher_occurrences: list[Span]
    occurrence_mode: str
    quality_flags: list[str]


@dataclass
class Replacement:
    mapping_index: int
    occurrence_index: int
    occurrence_count: int
    chosen_span: Span
    rejected_span: Span | None = None
    quality_flags: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "raw" / "golden_pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "prepared",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=root / "analysis",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=root.parent,
    )
    parser.add_argument(
        "--effective-config",
        type=Path,
        default=root / "raw" / "effective_config.yaml",
    )
    parser.add_argument(
        "--max-term-chars",
        type=int,
        default=64,
        help="Warning threshold only; long spans are not rejected.",
    )
    parser.add_argument(
        "--max-term-whitespace-tokens",
        type=int,
        default=8,
        help="Warning threshold only; long spans are not rejected.",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument(
        "--sample-ids",
        type=Path,
        default=root / "analysis" / "sample_ids.txt",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield item


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def terminal_control_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if stripped and stripped[-1] in ":;.!?。；：！？":
        return stripped[-1]
    return ""


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_occurrences(text: str, needle: str) -> list[Span]:
    """Return non-overlapping exact occurrences, left to right."""

    if not needle:
        return []
    spans: list[Span] = []
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return spans
        end = start + len(needle)
        spans.append(Span(start, end))
        cursor = end


def maximum_nonoverlapping_span_count(spans: Iterable[Span]) -> int:
    """Maximum number of non-overlapping intervals (earliest-finish greedy)."""

    count = 0
    cursor = -1
    for span in sorted(set(spans), key=lambda item: (item.end, item.start)):
        if span.start >= cursor:
            count += 1
            cursor = span.end
    return count


def hangul_jongseong_index(character: str) -> int | None:
    if len(character) != 1:
        return None
    codepoint = ord(character)
    if not 0xAC00 <= codepoint <= 0xD7A3:
        return None
    return (codepoint - 0xAC00) % 28


def rightmost_hangul_jongseong_index(text: str) -> int | None:
    """Find the Korean head ending even when a Latin gloss trails in parentheses."""

    for character in reversed(text):
        jongseong = hangul_jongseong_index(character)
        if jongseong is not None:
            return jongseong
    return None


def leading_parenthetical(text: str) -> tuple[str, str] | None:
    """Split one balanced leading parenthetical from text, if present."""

    if not text.startswith("("):
        return None
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[1:index], text[index + 1 :]
    return None


def normalized_gloss(text: str) -> str:
    return "".join(character for character in nfc(text).casefold() if character.isalnum())


def duplicates_following_parenthetical(chosen: str, chosen_span: Span, inserted_term: str) -> bool:
    """Detect outputs such as `Vicarious liability(vicarious liability)`."""

    split = leading_parenthetical(chosen[chosen_span.end :])
    if split is None:
        return False
    gloss, _ = split
    gloss_key = normalized_gloss(gloss)
    inserted_key = normalized_gloss(inserted_term)
    if bool(gloss_key) and gloss_key == inserted_key:
        return True
    inserted = inserted_term.rstrip()
    opening = inserted.rfind("(")
    if opening >= 0 and inserted.endswith(")"):
        inserted_gloss_key = normalized_gloss(inserted[opening + 1 : -1])
        return bool(gloss_key) and inserted_gloss_key == gloss_key
    return False


def josa_compatibility(
    chosen: str,
    chosen_span: Span,
    inserted_term: str,
) -> bool | None:
    """Check common Korean particles immediately following an insertion.

    True/False means that a known particle was checkable; None means the check
    was not applicable.  Closing punctuation at the end of the inserted term is
    deliberately not guessed through.
    """

    suffix = chosen[chosen_span.end :]
    # If the teacher correction excludes an immediately following source gloss,
    # the Korean particle can occur after `(English gloss)` rather than directly
    # after the replaced span.
    parenthetical = leading_parenthetical(suffix)
    if parenthetical is not None:
        _, after_parenthetical = parenthetical
        if after_parenthetical and not after_parenthetical[0].isspace():
            suffix = after_parenthetical

    regular_pairs = (
        ("이라든지", "라든지"),
        ("이라는", "라는"),
        ("이라고", "라고"),
        ("이라도", "라도"),
        ("이든지", "든지"),
        ("이랑", "랑"),
        ("이나", "나"),
        ("이란", "란"),
        ("이든", "든"),
        ("이야", "야"),
        ("이여", "여"),
        ("은", "는"),
        ("이", "가"),
        ("을", "를"),
        ("과", "와"),
    )
    matched_pair = next(
        (pair for pair in regular_pairs if suffix.startswith(pair[0]) or suffix.startswith(pair[1])),
        None,
    )
    matched_ro = next(
        (particle for particle in ("으로", "로") if suffix.startswith(particle)),
        None,
    )
    if matched_pair is None and matched_ro is None:
        return None
    jongseong = rightmost_hangul_jongseong_index(inserted_term)
    if jongseong is None:
        return None
    has_batchim = jongseong != 0
    if matched_pair is not None:
        expected = matched_pair[0] if has_batchim else matched_pair[1]
        return suffix.startswith(expected)
    assert matched_ro is not None
    expected_ro = "으로" if has_batchim and jongseong != 8 else "로"
    return suffix.startswith(expected_ro)


def _term_fields(
    error: Mapping[str, Any],
    error_index: int,
) -> tuple[str | None, str, str]:
    fields = {
        "error_span_target": error.get("error_span_target"),
        "correction": error.get("correction"),
    }
    missing = [key for key, value in fields.items() if not nonempty_text(value)]
    if missing:
        raise RowRejected(
            "missing_term_field",
            f"teacher_errors[{error_index}] missing or empty: {', '.join(missing)}",
        )
    source_term = error.get("source_span")
    return (
        str(source_term) if nonempty_text(source_term) else None,
        str(fields["error_span_target"]),
        str(fields["correction"]),
    )


def _term_quality_flags(
    *,
    source: str,
    source_term: str | None,
    student_term: str,
    teacher_term: str,
    max_term_chars: int,
    max_term_whitespace_tokens: int,
) -> list[str]:
    flags: set[str] = set()
    values = {
        "student_term": student_term,
        "teacher_term": teacher_term,
    }
    if source_term is None:
        flags.add("missing_source_span")
    else:
        values["source_span"] = source_term
        if not exact_occurrences(source, source_term):
            flags.add("source_span_not_found")
    for field_name, value in values.items():
        if len(value) > max_term_chars:
            flags.add(f"long_{field_name}_chars")
        if whitespace_token_count(value) > max_term_whitespace_tokens:
            flags.add(f"long_{field_name}_whitespace_tokens")
    if terminal_control_punctuation(student_term) != terminal_control_punctuation(teacher_term):
        flags.add("terminal_punctuation_mismatch")
    return sorted(flags)


def build_mapping_groups(
    *,
    source: str,
    student: str,
    chosen: str,
    term_entries: Sequence[tuple[int, Mapping[str, Any]]],
    max_term_chars: int,
    max_term_whitespace_tokens: int,
) -> list[MappingGroup]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for error_index, error in term_entries:
        source_term, student_term, teacher_term = _term_fields(error, error_index)
        quality_flags = _term_quality_flags(
            source=source,
            source_term=source_term,
            student_term=student_term,
            teacher_term=teacher_term,
            max_term_chars=max_term_chars,
            max_term_whitespace_tokens=max_term_whitespace_tokens,
        )
        if nfc(student_term) == nfc(teacher_term):
            raise RowRejected(
                "same_student_teacher_term_after_nfc",
                f"teacher_errors[{error_index}] has no term contrast",
            )
        if not exact_occurrences(student, student_term):
            raise RowRejected(
                "student_term_not_found",
                f"teacher_errors[{error_index}] error_span_target={student_term!r}",
            )
        if not exact_occurrences(chosen, teacher_term):
            raise RowRejected(
                "teacher_term_not_found",
                f"teacher_errors[{error_index}] correction={teacher_term!r}",
            )

        key = (student_term, teacher_term)
        item = grouped.setdefault(
            key,
            {
                "source_terms": [],
                "error_indices": [],
                "reasons_ko": [],
                "quality_flags": set(),
            },
        )
        if source_term is not None and source_term not in item["source_terms"]:
            item["source_terms"].append(source_term)
        item["error_indices"].append(error_index)
        item["quality_flags"].update(quality_flags)
        reason = error.get("reason_ko")
        if nonempty_text(reason) and str(reason) not in item["reasons_ko"]:
            item["reasons_ko"].append(str(reason))

    groups: list[MappingGroup] = []
    for (student_term, teacher_term), item in grouped.items():
        teacher_occurrences = exact_occurrences(chosen, teacher_term)
        student_occurrences = exact_occurrences(student, student_term)
        student_occurrence_count = len(student_occurrences)
        source_spans = [
            span
            for source_term in item["source_terms"]
            for span in exact_occurrences(source, source_term)
        ]
        source_occurrence_count = maximum_nonoverlapping_span_count(source_spans)

        if len(teacher_occurrences) == 1:
            occurrence_mode = "unique_teacher_occurrence"
        elif len(teacher_occurrences) == student_occurrence_count:
            # A repeated terminology annotation often describes a term type,
            # not each surface occurrence separately. Expand every teacher
            # occurrence when the original student contains the same number.
            occurrence_mode = "balanced_repeated_occurrences"
        else:
            raise RowRejected(
                "ambiguous_repeated_teacher_term",
                (
                    f"student_term={student_term!r}, teacher_term={teacher_term!r}, "
                    f"source_occurrences={source_occurrence_count}, "
                    f"student_occurrences={student_occurrence_count}, "
                    f"teacher_occurrences={len(teacher_occurrences)}"
                ),
            )

        groups.append(
            MappingGroup(
                student_term=student_term,
                teacher_term=teacher_term,
                source_terms=list(item["source_terms"]),
                error_indices=list(item["error_indices"]),
                reasons_ko=list(item["reasons_ko"]),
                source_occurrence_count=source_occurrence_count,
                student_occurrence_count=student_occurrence_count,
                student_occurrences=student_occurrences,
                teacher_occurrences=teacher_occurrences,
                occurrence_mode=occurrence_mode,
                quality_flags=sorted(item["quality_flags"]),
            )
        )
    return groups


def build_negative(
    chosen: str,
    groups: Sequence[MappingGroup],
) -> tuple[str, list[Replacement]]:
    replacements: list[Replacement] = []
    for mapping_index, group in enumerate(groups):
        for occurrence_index, chosen_span in enumerate(group.teacher_occurrences):
            quality_flags: list[str] = []
            if duplicates_following_parenthetical(chosen, chosen_span, group.student_term):
                quality_flags.append("duplicate_parenthetical_after_reversion")
            compatibility = josa_compatibility(chosen, chosen_span, group.student_term)
            if compatibility is False:
                quality_flags.append("josa_incompatible_after_reversion")
            replacements.append(
                Replacement(
                    mapping_index=mapping_index,
                    occurrence_index=occurrence_index,
                    occurrence_count=len(group.teacher_occurrences),
                    chosen_span=chosen_span,
                    quality_flags=quality_flags,
                )
            )

    replacements.sort(key=lambda item: (item.chosen_span.start, item.chosen_span.end))
    for previous, current in zip(replacements, replacements[1:]):
        if current.chosen_span.start < previous.chosen_span.end:
            raise RowRejected(
                "overlapping_teacher_term_spans",
                f"spans={previous.chosen_span.as_list()} and {current.chosen_span.as_list()}",
            )

    parts: list[str] = []
    chosen_cursor = 0
    rejected_length = 0
    for replacement in replacements:
        group = groups[replacement.mapping_index]
        unchanged = chosen[chosen_cursor : replacement.chosen_span.start]
        parts.append(unchanged)
        rejected_length += len(unchanged)
        rejected_span = Span(rejected_length, rejected_length + len(group.student_term))
        parts.append(group.student_term)
        rejected_length = rejected_span.end
        replacement.rejected_span = rejected_span
        chosen_cursor = replacement.chosen_span.end
    parts.append(chosen[chosen_cursor:])
    rejected = "".join(parts)

    if rejected == chosen:
        raise RowRejected("negative_equals_positive", "term replacements produced no contrast")

    # Strong locality invariant: replacing exactly the recorded rejected spans
    # with their teacher terms must recover y+ byte-for-byte.
    recovered_parts: list[str] = []
    rejected_cursor = 0
    for replacement in replacements:
        assert replacement.rejected_span is not None
        group = groups[replacement.mapping_index]
        recovered_parts.append(rejected[rejected_cursor : replacement.rejected_span.start])
        recovered_parts.append(group.teacher_term)
        rejected_cursor = replacement.rejected_span.end
    recovered_parts.append(rejected[rejected_cursor:])
    recovered = "".join(recovered_parts)
    if recovered != chosen:
        raise AssertionError("locality invariant failed: y- could not reconstruct y+")
    return rejected, replacements


class PromptResolver:
    def __init__(self, *, repo_root: Path, effective_config_path: Path) -> None:
        with effective_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, Mapping):
            raise ValueError(f"Expected mapping in {effective_config_path}")
        self.repo_root = repo_root.resolve()
        self.prompt_cfg = config.get("prompts", {})
        self.model_cfg = config.get("model", {})
        if not isinstance(self.prompt_cfg, Mapping) or not isinstance(self.model_cfg, Mapping):
            raise ValueError("effective config requires prompts and model mappings")

        template_path = Path(
            str(self.prompt_cfg.get("student_templates_path", "prompts/student_templates.yaml"))
        )
        if not template_path.is_absolute():
            template_path = self.repo_root / template_path
        with template_path.open("r", encoding="utf-8") as handle:
            self.template_cfg = yaml.safe_load(handle) or {}
        group_name = (
            "instruct_templates"
            if bool(self.model_cfg.get("use_hf_chat_template", False))
            or bool(self.model_cfg.get("use_chat_messages", False))
            else "base_templates"
        )
        templates = self.template_cfg.get(group_name, [])
        self.group_name = group_name
        self.expected_hashes: dict[str, str] = {}
        for index, template in enumerate(templates):
            if not isinstance(template, Mapping):
                continue
            template_id = str(template.get("id", f"{group_name}_{index:03d}"))
            system_text = str(template.get("system", ""))
            user_text = str(template.get("user", template.get("text", "")))
            hash_source = f"system:{system_text}\nuser:{user_text}"
            self.expected_hashes[template_id] = stable_hash(hash_source)

    def resolve(
        self,
        *,
        row: Mapping[str, Any],
        subset_idx: int,
    ) -> tuple[str, str]:
        template_id = str(row.get("prompt_template_id") or "")
        stored_hash = str(row.get("prompt_template_hash") or "")
        expected_hash = self.expected_hashes.get(template_id)
        if expected_hash is None:
            raise RowRejected("unknown_prompt_template", f"template_id={template_id!r}")
        if stored_hash != expected_hash:
            raise RowRejected(
                "prompt_template_hash_mismatch",
                f"template_id={template_id!r}, stored={stored_hash}, expected={expected_hash}",
            )

        prompt = row.get("prompt")
        if nonempty_text(prompt):
            return str(prompt), "stored"

        source = str(row.get("source") or "")
        row_id = str(row.get("id") or "")
        src_path = self.repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from prompting import render_student_prompt  # Imported lazily for tokenizer startup.

        rendered = render_student_prompt(
            template_cfg=self.template_cfg,
            prompt_cfg=self.prompt_cfg,
            model_cfg=self.model_cfg,
            source=source,
            row_id=row_id,
            subset_idx=subset_idx,
            fixed_template_id=template_id,
        )
        if rendered.template_hash != stored_hash:
            raise RowRejected(
                "regenerated_prompt_hash_mismatch",
                f"template_id={template_id!r}",
            )
        if bool(row.get("chat_template_applied")) != rendered.chat_template_applied:
            raise RowRejected(
                "regenerated_chat_template_mode_mismatch",
                f"template_id={template_id!r}",
            )
        return rendered.text, "regenerated"


def build_pair_content(
    *,
    row: Mapping[str, Any],
    subset: str,
    subset_idx: int,
    max_term_chars: int,
    max_term_whitespace_tokens: int,
) -> dict[str, Any] | None:
    source = row.get("source")
    student = row.get("student_translation")
    chosen = row.get("target")
    if not nonempty_text(source) or not nonempty_text(student) or not nonempty_text(chosen):
        raw_errors = row.get("teacher_errors")
        has_term = isinstance(raw_errors, list) and any(
            isinstance(error, Mapping) and error.get("error_type") == TERM_ERROR_TYPE
            for error in raw_errors
        )
        if has_term:
            raise RowRejected("missing_row_text", "source, student_translation, or target is empty")
        return None
    assert isinstance(source, str)
    assert isinstance(student, str)
    assert isinstance(chosen, str)

    raw_errors = row.get("teacher_errors")
    if not isinstance(raw_errors, list):
        return None
    errors = [error for error in raw_errors if isinstance(error, Mapping)]
    term_entries = [
        (index, error)
        for index, error in enumerate(errors)
        if error.get("error_type") == TERM_ERROR_TYPE
    ]
    if not term_entries:
        return None

    groups = build_mapping_groups(
        source=source,
        student=student,
        chosen=chosen,
        term_entries=term_entries,
        max_term_chars=max_term_chars,
        max_term_whitespace_tokens=max_term_whitespace_tokens,
    )
    rejected, replacements = build_negative(chosen, groups)

    replacement_rows: list[dict[str, Any]] = []
    for replacement in replacements:
        assert replacement.rejected_span is not None
        group = groups[replacement.mapping_index]
        replacement_rows.append(
            {
                "mapping_index": replacement.mapping_index,
                "occurrence_index": replacement.occurrence_index,
                "occurrence_count": replacement.occurrence_count,
                "error_indices": group.error_indices,
                "source_terms": group.source_terms,
                "student_term": group.student_term,
                "teacher_term": group.teacher_term,
                "chosen_char_span": replacement.chosen_span.as_list(),
                "rejected_char_span": replacement.rejected_span.as_list(),
                "reason_ko": group.reasons_ko,
                "quality_flags": replacement.quality_flags,
            }
        )

    mapping_rows = [
        {
            "mapping_index": index,
            "error_indices": group.error_indices,
            "source_terms": group.source_terms,
            "student_term": group.student_term,
            "teacher_term": group.teacher_term,
            "source_occurrence_count": group.source_occurrence_count,
            "student_occurrence_count": group.student_occurrence_count,
            "teacher_occurrence_count": len(group.teacher_occurrences),
            "occurrence_mode": group.occurrence_mode,
            "reason_ko": group.reasons_ko,
            "quality_flags": group.quality_flags,
        }
        for index, group in enumerate(groups)
    ]
    quality_flags = sorted(
        {
            flag
            for group in groups
            for flag in group.quality_flags
        }
        | {
            flag
            for replacement in replacements
            for flag in replacement.quality_flags
        }
    )
    has_repeated_mapping = any(
        group.occurrence_mode == "balanced_repeated_occurrences" for group in groups
    )
    row_id = str(row.get("id") or "")
    return {
        "pair_id": f"{subset}:{row_id}",
        "synthesis_version": SYNTHESIS_VERSION,
        "synthesis_policy": "mechanical_exact_all_terminology_annotations_or_reject_row",
        "source": source,
        "prompt": None,
        "chosen": chosen,
        "rejected": rejected,
        "student_translation": student,
        "term_annotation_count": len(term_entries),
        "term_mapping_count": len(groups),
        "replacement_span_count": len(replacements),
        "has_multiple_term_annotations": len(term_entries) > 1,
        "has_repeated_mapping": has_repeated_mapping,
        "roundtrip_strict": rejected == student,
        "all_terminology_errors_reverted": True,
        "has_quality_warnings": bool(quality_flags),
        "quality_flags": quality_flags,
        "term_mappings": mapping_rows,
        "term_replacements": replacement_rows,
        "chosen_term_char_spans": [
            replacement["chosen_char_span"] for replacement in replacement_rows
        ],
        "rejected_term_char_spans": [
            replacement["rejected_char_span"] for replacement in replacement_rows
        ],
        "subset": subset,
        "subset_idx": subset_idx,
        "row_id": row_id,
        "teacher_label": row.get("teacher_label"),
        "total_teacher_error_count": len(errors),
        "non_term_teacher_error_count": len(errors) - len(term_entries),
        "qe_score": row.get("qe_score"),
        "selection_rank": row.get("selection_rank"),
        "teacher_accept_rank": row.get("teacher_accept_rank"),
        "prompt_template_id": row.get("prompt_template_id"),
        "prompt_template_group": row.get("prompt_template_group"),
        "prompt_template_hash": row.get("prompt_template_hash"),
        "chat_template_applied": row.get("chat_template_applied"),
        "source_tokens": row.get("source_tokens"),
        "length_bucket_idx": row.get("length_bucket_idx"),
        "length_bucket": row.get("length_bucket"),
        "hf_repo": HF_REPO,
        "hf_run": HF_RUN,
        "hf_repo_revision": HF_REPO_REVISION,
    }


def candidate_sort_key(pair: Mapping[str, Any]) -> tuple[int, str]:
    return int(pair["subset_idx"]), str(pair["row_id"])


def _sample_hash(pair: Mapping[str, Any]) -> str:
    return stable_hash(str(pair["pair_id"]))


def select_review_samples(
    candidates: Sequence[dict[str, Any]],
    size: int,
    sample_ids_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Choose a compact, deterministic mix of single/multi/repeated examples."""

    if sample_ids_path is not None and sample_ids_path.exists():
        requested_ids = [
            line.strip()
            for line in sample_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        by_id = {str(pair["pair_id"]): pair for pair in candidates}
        missing = [pair_id for pair_id in requested_ids if pair_id not in by_id]
        if missing:
            raise ValueError(f"Requested sample pair_ids are not accepted: {missing}")
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError(f"Duplicate pair_ids in {sample_ids_path}")
        if len(requested_ids) != size:
            raise ValueError(
                f"Expected {size} pair_ids in {sample_ids_path}, found {len(requested_ids)}"
            )
        return [by_id[pair_id] for pair_id in requested_ids]

    readable = [
        pair
        for pair in candidates
        if len(str(pair["source"])) <= 360
        and len(str(pair["chosen"])) <= 360
        and len(str(pair["rejected"])) <= 360
    ]
    ordered = sorted(readable, key=_sample_hash)
    predicates = [
        lambda p: int(p["term_annotation_count"]) >= 3,
        lambda p: bool(p["has_repeated_mapping"]),
        lambda p: int(p["term_annotation_count"]) == 2 and bool(p["roundtrip_strict"]),
        lambda p: int(p["term_annotation_count"]) == 2 and not bool(p["roundtrip_strict"]),
        lambda p: int(p["term_annotation_count"]) >= 2 and p["teacher_label"] == "critical",
        lambda p: int(p["term_annotation_count"]) == 2,
        lambda p: int(p["term_annotation_count"]) == 1 and bool(p["roundtrip_strict"]),
        lambda p: int(p["term_annotation_count"]) == 1 and not bool(p["roundtrip_strict"]),
        lambda p: p["teacher_label"] == "major",
        lambda p: True,
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for predicate in predicates:
        match = next(
            (
                pair
                for pair in ordered
                if str(pair["pair_id"]) not in seen and predicate(pair)
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            seen.add(str(match["pair_id"]))
        if len(selected) >= size:
            return selected
    for pair in ordered:
        if str(pair["pair_id"]) in seen:
            continue
        selected.append(pair)
        seen.add(str(pair["pair_id"]))
        if len(selected) >= size:
            break
    return selected


def markdown_fence(text: str) -> str:
    fence = "```"
    if fence in text:
        fence = "````"
    return f"{fence}\n{text}\n{fence}"


def samples_markdown(samples: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# 다중 용어 역치환 샘플 10개",
        "",
        "각 `y-`는 `y+`에서 아래에 표시된 용어 span만 student 표현으로 되돌린 결과다. "
        "`roundtrip_strict=false`는 원래 student의 다른 오류를 복원하지 않고 teacher 수정 상태로 유지했다는 뜻이다. "
        "아래 10개는 모두 자동 품질 경고가 없는 예시다.",
        "",
    ]
    for index, pair in enumerate(samples, start=1):
        lines.extend(
            [
                f"## {index}. {pair['pair_id']}",
                "",
                (
                    f"- label: `{pair['teacher_label']}`; terminology annotations: "
                    f"`{pair['term_annotation_count']}`; replacement spans: "
                    f"`{pair['replacement_span_count']}`; roundtrip_strict: "
                    f"`{str(bool(pair['roundtrip_strict'])).lower()}`"
                ),
                "",
                "- 역치환:",
                "",
            ]
        )
        for mapping in pair["term_mappings"]:
            source_terms = ", ".join(repr(term) for term in mapping["source_terms"])
            lines.append(
                f"  - source {source_terms}: `{mapping['teacher_term']}` → "
                f"`{mapping['student_term']}` ({mapping['teacher_occurrence_count']}회)"
            )
        lines.extend(
            [
                "",
                "Source",
                "",
                markdown_fence(str(pair["source"])),
                "",
                "Student 원출력",
                "",
                markdown_fence(str(pair["student_translation"])),
                "",
                "y+ (teacher post-edit)",
                "",
                markdown_fence(str(pair["chosen"])),
                "",
                "y- (term-reverted synthetic negative)",
                "",
                markdown_fence(str(pair["rejected"])),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("subset_*.jsonl"))
    if not input_paths:
        raise SystemExit(f"No subset_*.jsonl files found in {args.input_dir}")

    os.environ.setdefault(
        "HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface")
    )
    resolver = PromptResolver(
        repo_root=args.repo_root,
        effective_config_path=args.effective_config,
    )

    counts: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    accepted_labels: Counter[str] = Counter()
    accepted_annotations: Counter[int] = Counter()
    accepted_mappings: Counter[int] = Counter()
    accepted_replacements: Counter[int] = Counter()
    quality_warning_rows: Counter[str] = Counter()
    rejected_annotations: Counter[int] = Counter()
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
            counts["rows_total"] += 1
            per_subset[subset]["rows_total"] += 1
            raw_errors = row.get("teacher_errors")
            term_annotation_count = (
                sum(
                    1
                    for error in raw_errors
                    if isinstance(error, Mapping) and error.get("error_type") == TERM_ERROR_TYPE
                )
                if isinstance(raw_errors, list)
                else 0
            )
            if term_annotation_count == 0:
                counts["rows_without_terminology"] += 1
                continue
            counts["rows_with_terminology"] += 1
            counts["terminology_annotations"] += term_annotation_count
            if term_annotation_count > 1:
                counts["rows_with_multiple_terminology_annotations"] += 1

            try:
                pair = build_pair_content(
                    row=row,
                    subset=subset,
                    subset_idx=subset_idx,
                    max_term_chars=args.max_term_chars,
                    max_term_whitespace_tokens=args.max_term_whitespace_tokens,
                )
                assert pair is not None
                prompt, prompt_status = resolver.resolve(row=row, subset_idx=subset_idx)
                pair["prompt"] = prompt
                pair["prompt_status"] = prompt_status
            except RowRejected as exc:
                counts["rows_rejected"] += 1
                reject_reasons[exc.code] += 1
                rejected_annotations[term_annotation_count] += 1
                per_subset[subset]["rows_rejected"] += 1
                rejection_ledger.append(
                    {
                        "subset": subset,
                        "subset_idx": subset_idx,
                        "row_id": row.get("id"),
                        "teacher_label": row.get("teacher_label"),
                        "term_annotation_count": term_annotation_count,
                        "reject_reason": exc.code,
                        "reject_detail": exc.detail,
                    }
                )
                continue

            counts["rows_accepted"] += 1
            per_subset[subset]["rows_accepted"] += 1
            accepted_labels[str(pair.get("teacher_label") or "<null>")] += 1
            accepted_annotations[int(pair["term_annotation_count"])] += 1
            accepted_mappings[int(pair["term_mapping_count"])] += 1
            accepted_replacements[int(pair["replacement_span_count"])] += 1
            counts["accepted_term_mapping_groups"] += int(pair["term_mapping_count"])
            counts["accepted_replacement_spans"] += int(pair["replacement_span_count"])
            if pair["has_multiple_term_annotations"]:
                counts["accepted_rows_with_multiple_terminology_annotations"] += 1
            if pair["has_repeated_mapping"]:
                counts["accepted_rows_with_balanced_repeated_occurrences"] += 1
            if pair["roundtrip_strict"]:
                counts["roundtrip_strict_rows"] += 1
            if pair["has_quality_warnings"]:
                counts["accepted_rows_with_quality_warnings"] += 1
                for quality_flag in pair["quality_flags"]:
                    quality_warning_rows[str(quality_flag)] += 1
            else:
                counts["accepted_rows_without_quality_warnings"] += 1
            if pair["prompt_status"] == "regenerated":
                counts["prompts_regenerated"] += 1
            candidates.append(pair)

    candidates.sort(key=candidate_sort_key)
    ids = [str(pair["pair_id"]) for pair in candidates]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate pair_id values in accepted candidates")
    if counts["rows_with_terminology"] != counts["rows_accepted"] + counts["rows_rejected"]:
        raise AssertionError("Terminology row accounting mismatch")

    candidate_path = args.output_dir / "preference_candidates.jsonl"
    roundtrip_path = args.output_dir / "roundtrip_strict.jsonl"
    reject_path = args.analysis_dir / "multi_term_rejections.jsonl"
    sample_jsonl_path = args.analysis_dir / "sample_10.jsonl"
    sample_markdown_path = args.analysis_dir / "SAMPLE_10.md"
    summary_path = args.analysis_dir / "multi_term_synthesis_summary.json"

    write_jsonl(candidate_path, candidates)
    write_jsonl(roundtrip_path, (pair for pair in candidates if pair["roundtrip_strict"]))
    write_jsonl(reject_path, rejection_ledger)
    samples = select_review_samples(candidates, args.sample_size, args.sample_ids)
    write_jsonl(sample_jsonl_path, samples)
    sample_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    sample_markdown_path.write_text(samples_markdown(samples), encoding="utf-8")

    summary = {
        "synthesis_version": SYNTHESIS_VERSION,
        "policy": {
            "positive": "teacher target unchanged",
            "negative": "teacher target with every mechanically aligned terminology correction reverted",
            "row_atomicity": "all terminology annotations in a row must pass; no partial rows",
            "multiple_terminology_errors": "allowed and jointly reverted in one pair",
            "repeated_surface_occurrences": (
                "replace all teacher occurrences when teacher and student surface counts match"
            ),
            "student_surface_occurrences": (
                "student term must occur at least once; uniqueness is not required"
            ),
            "source_span": "retained as metadata; missing or non-exact source spans are warnings",
            "right_boundary": "not checked and not required",
            "non_terminology_teacher_edits": "preserved identically in chosen and rejected",
            "teacher_labels": "minor, major, and critical are eligible; label is not a filter",
            "long_span_warning_chars": args.max_term_chars,
            "long_span_warning_whitespace_tokens": args.max_term_whitespace_tokens,
            "josa_incompatibility": "warning only",
            "terminal_punctuation_mismatch": "warning only",
            "duplicate_parenthetical": "warning only",
            "tokenizer_stage": "not run; this is a character-span candidate artifact",
        },
        "input": {
            "directory": str(args.input_dir.resolve()),
            "file_count": len(input_paths),
            "hf_repo": HF_REPO,
            "hf_run": HF_RUN,
            "hf_repo_revision": HF_REPO_REVISION,
        },
        "counts": dict(counts),
        "accepted_teacher_labels": _counter_dict(accepted_labels),
        "accepted_term_annotation_count_distribution": _counter_dict(accepted_annotations),
        "accepted_term_mapping_count_distribution": _counter_dict(accepted_mappings),
        "accepted_replacement_span_count_distribution": _counter_dict(accepted_replacements),
        "quality_warning_rows": _counter_dict(quality_warning_rows),
        "rejected_term_annotation_count_distribution": _counter_dict(rejected_annotations),
        "reject_reasons": dict(reject_reasons.most_common()),
        "per_subset": {
            subset: dict(counter) for subset, counter in sorted(per_subset.items())
        },
        "outputs": {
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": file_sha256(candidate_path),
            "roundtrip_strict_path": str(roundtrip_path.resolve()),
            "rejection_ledger_path": str(reject_path.resolve()),
            "sample_jsonl_path": str(sample_jsonl_path.resolve()),
            "sample_markdown_path": str(sample_markdown_path.resolve()),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
