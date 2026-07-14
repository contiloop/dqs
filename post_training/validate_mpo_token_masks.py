#!/usr/bin/env python3
"""Independently re-tokenize and validate the prepared mPO mask artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
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
        "--rejections",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_rejections.jsonl",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_sample_10.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_validation_report.json",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def encode(tokenizer: Any, text: str, *, offsets: bool) -> tuple[list[int], list[tuple[int, int]]]:
    kwargs: dict[str, Any] = {"add_special_tokens": False}
    if offsets:
        kwargs["return_offsets_mapping"] = True
    encoded = tokenizer(text, **kwargs)
    ids = [int(value) for value in encoded["input_ids"]]
    parsed_offsets = (
        [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        if offsets
        else []
    )
    return ids, parsed_offsets


def adjusted_spans(candidate: Mapping[str, Any], side: str) -> tuple[str, list[list[int]], list[int]]:
    raw = str(candidate[side])
    stripped = raw.strip()
    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw) - len(raw.rstrip())
    spans = []
    for start, end in candidate[f"{side}_term_char_spans"]:
        spans.append([int(start) - left_trim, int(end) - left_trim])
    return stripped, spans, [left_trim, right_trim]


def expected_term_indices(
    *,
    pair_id: str,
    side: str,
    text: str,
    spans: list[list[int]],
    offsets: list[tuple[int, int]],
    prompt_count: int,
) -> tuple[list[int], list[list[int]]]:
    union: set[int] = set()
    per_span: list[list[int]] = []
    for span_index, (span_start, span_end) in enumerate(spans):
        if not 0 <= span_start < span_end <= len(text):
            raise AssertionError(f"{pair_id}: invalid adjusted {side} span {span_index}")
        selected = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and max(token_start, span_start) < min(token_end, span_end)
        ]
        if not selected:
            raise AssertionError(f"{pair_id}: empty {side} span mask {span_index}")
        for index in selected:
            token_start, token_end = offsets[index]
            visible_end = min(token_end, len(text))
            left = text[token_start:min(visible_end, span_start)] if token_start < span_start else ""
            right_start = max(token_start, span_end)
            right = text[right_start:visible_end] if visible_end > span_end else ""
            if (left + right).strip():
                raise AssertionError(f"{pair_id}: {side} token crosses non-whitespace boundary")
            if index in union:
                raise AssertionError(f"{pair_id}: {side} term spans share token {index}")
            union.add(index)
        per_span.append([prompt_count + index for index in selected])
    return sorted(prompt_count + index for index in union), per_span


def validate_binary_mask(pair_id: str, name: str, value: Any, expected_length: int) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise AssertionError(f"{pair_id}: {name} length mismatch")
    mask = [int(item) for item in value]
    if any(item not in (0, 1) for item in mask):
        raise AssertionError(f"{pair_id}: {name} is not binary")
    return mask


def validate_row(
    *,
    tokenizer: Any,
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
    eos_token: str,
    append_eos: bool,
    max_seq_length: int,
) -> None:
    pair_id = str(row["pair_id"])
    prompt = str(candidate["prompt"])
    prompt_ids, _ = encode(tokenizer, prompt, offsets=False)
    if int(row["prompt_token_count"]) != len(prompt_ids):
        raise AssertionError(f"{pair_id}: prompt token count mismatch")
    if row["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise AssertionError(f"{pair_id}: prompt checksum mismatch")

    side_term_counts: dict[str, int] = {}
    for side in ("chosen", "rejected"):
        text, spans, trim_counts = adjusted_spans(candidate, side)
        if row[side] != text or row[f"{side}_term_char_spans"] != spans:
            raise AssertionError(f"{pair_id}: {side} stripped text/span mismatch")
        if row[f"{side}_outer_whitespace_stripped_chars"] != trim_counts:
            raise AssertionError(f"{pair_id}: {side} strip-count mismatch")
        completion_for_tokens = text
        if append_eos and not completion_for_tokens.endswith(eos_token):
            completion_for_tokens += eos_token
        completion_ids, offsets = encode(tokenizer, completion_for_tokens, offsets=True)
        expected_ids = prompt_ids + completion_ids
        if row[f"{side}_input_ids"] != expected_ids:
            raise AssertionError(f"{pair_id}: {side} input_ids differ from fresh tokenization")
        if len(expected_ids) > max_seq_length:
            raise AssertionError(f"{pair_id}: accepted {side} exceeds max_seq_length")

        completion_mask = validate_binary_mask(
            pair_id,
            f"{side}_completion_mask",
            row[f"{side}_completion_mask"],
            len(expected_ids),
        )
        expected_completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        if completion_mask != expected_completion_mask:
            raise AssertionError(f"{pair_id}: {side} completion mask includes prompt or excludes completion")

        expected_indices, per_span_indices = expected_term_indices(
            pair_id=pair_id,
            side=side,
            text=text,
            spans=spans,
            offsets=offsets,
            prompt_count=len(prompt_ids),
        )
        term_mask = validate_binary_mask(
            pair_id,
            f"{side}_term_mask",
            row[f"{side}_term_mask"],
            len(expected_ids),
        )
        actual_indices = [index for index, value in enumerate(term_mask) if value]
        if actual_indices != expected_indices or row[f"{side}_term_token_indices"] != expected_indices:
            raise AssertionError(f"{pair_id}: {side} term mask/index mismatch")
        if any(term_mask[: len(prompt_ids)]):
            raise AssertionError(f"{pair_id}: {side} term mask includes prompt")
        if not actual_indices:
            raise AssertionError(f"{pair_id}: empty {side} term mask")
        expected_prediction_indices = [index - 1 for index in expected_indices]
        if row[f"{side}_term_prediction_indices"] != expected_prediction_indices:
            raise AssertionError(f"{pair_id}: {side} causal shift index mismatch")
        if int(row[f"{side}_term_token_count"]) != len(expected_indices):
            raise AssertionError(f"{pair_id}: {side} term token count mismatch")
        if int(row[f"{side}_completion_token_count"]) != len(completion_ids):
            raise AssertionError(f"{pair_id}: {side} completion token count mismatch")
        if int(row[f"{side}_sequence_token_count"]) != len(expected_ids):
            raise AssertionError(f"{pair_id}: {side} sequence token count mismatch")
        alignments = row[f"{side}_term_alignments"]
        if len(alignments) != len(per_span_indices):
            raise AssertionError(f"{pair_id}: {side} alignment count mismatch")
        for alignment, expected_span_indices in zip(alignments, per_span_indices, strict=True):
            if alignment["input_token_indices"] != expected_span_indices:
                raise AssertionError(f"{pair_id}: {side} per-span token alignment mismatch")
            if alignment["prediction_logit_indices"] != [index - 1 for index in expected_span_indices]:
                raise AssertionError(f"{pair_id}: {side} per-span shift mismatch")
        side_term_counts[side] = len(expected_indices)

    expected_different = side_term_counts["chosen"] != side_term_counts["rejected"]
    if bool(row["term_token_lengths_differ"]) != expected_different:
        raise AssertionError(f"{pair_id}: independent term-length flag mismatch")
    chosen_term_ids = [row["chosen_input_ids"][index] for index in row["chosen_term_token_indices"]]
    rejected_term_ids = [row["rejected_input_ids"][index] for index in row["rejected_term_token_indices"]]
    if chosen_term_ids == rejected_term_ids:
        raise AssertionError(f"{pair_id}: chosen/rejected term token IDs are identical")


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    tokenizer_cfg = summary["tokenizer"]
    contract = summary["training_contract"]
    os.environ.setdefault(
        "HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface")
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["name"],
        revision=tokenizer_cfg["requested_revision"],
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if not tokenizer.is_fast:
        raise AssertionError("validator requires a fast tokenizer")

    candidates = load_jsonl(args.candidates)
    candidate_by_id = {str(row["pair_id"]): row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise AssertionError("duplicate candidate pair_id")
    rejections = load_jsonl(args.rejections)
    rejected_ids = [str(row["pair_id"]) for row in rejections]
    if len(rejected_ids) != len(set(rejected_ids)):
        raise AssertionError("duplicate token-mask rejection pair_id")
    samples = load_jsonl(args.samples)
    sample_by_id = {str(row["pair_id"]): row for row in samples}
    if len(sample_by_id) != len(samples):
        raise AssertionError("duplicate token-mask sample pair_id")

    accepted_ids: list[str] = []
    seen_samples: set[str] = set()
    chosen_term_tokens = 0
    rejected_term_tokens = 0
    different_term_counts = 0
    with args.tokenized.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id", ""))
            candidate = candidate_by_id.get(pair_id)
            if candidate is None:
                raise AssertionError(f"unknown tokenized pair_id at line {line_number}: {pair_id}")
            validate_row(
                tokenizer=tokenizer,
                candidate=candidate,
                row=row,
                eos_token=str(tokenizer.eos_token),
                append_eos=bool(contract["append_eos"]),
                max_seq_length=int(contract["max_seq_length"]),
            )
            accepted_ids.append(pair_id)
            chosen_term_tokens += int(row["chosen_term_token_count"])
            rejected_term_tokens += int(row["rejected_term_token_count"])
            different_term_counts += int(bool(row["term_token_lengths_differ"]))
            if pair_id in sample_by_id:
                if row != sample_by_id[pair_id]:
                    raise AssertionError(f"sample row differs from tokenized artifact: {pair_id}")
                seen_samples.add(pair_id)

    if len(accepted_ids) != len(set(accepted_ids)):
        raise AssertionError("duplicate accepted tokenized pair_id")
    accepted_set = set(accepted_ids)
    rejected_set = set(rejected_ids)
    if accepted_set & rejected_set:
        raise AssertionError("accepted/rejected token-mask partitions overlap")
    if accepted_set | rejected_set != set(candidate_by_id):
        raise AssertionError("accepted/rejected token-mask partitions do not cover all candidates")
    if seen_samples != set(sample_by_id):
        raise AssertionError("sample artifact is not an exact accepted subset")

    counts = summary["counts"]
    expected_counts = {
        "input_rows": len(candidates),
        "accepted_rows": len(accepted_ids),
        "rejected_rows": len(rejections),
        "chosen_term_tokens": chosen_term_tokens,
        "rejected_term_tokens": rejected_term_tokens,
        "rows_with_different_term_token_counts": different_term_counts,
    }
    for key, expected in expected_counts.items():
        if int(counts[key]) != expected:
            raise AssertionError(f"summary count mismatch for {key}")
    if sha256(args.candidates) != summary["input"]["candidate_sha256"]:
        raise AssertionError("candidate checksum mismatch")
    if sha256(args.tokenized) != summary["outputs"]["tokenized_sha256"]:
        raise AssertionError("tokenized checksum mismatch")
    if sha256(args.rejections) != summary["outputs"]["rejections_sha256"]:
        raise AssertionError("rejection checksum mismatch")

    report = {
        "status": "passed",
        "validated_candidates": len(candidates),
        "validated_tokenized_pairs": len(accepted_ids),
        "validated_rejections": len(rejections),
        "validated_samples": len(samples),
        "rows_with_independent_term_token_counts": different_term_counts,
        "tokenizer": tokenizer_cfg,
        "tokenized_sha256": summary["outputs"]["tokenized_sha256"],
        "invariants": [
            "fresh prompt and completion tokenization exactly reproduces input_ids",
            "prompt tokens are zero in completion and term masks",
            "EOS is supervised by completion SFT but excluded from term masks",
            "chosen and rejected term masks are independently offset-aligned",
            "all term subtokens are selected and masks are non-empty",
            "chosen and rejected masked token-ID sequences are distinct",
            "non-whitespace token boundary crossings are rejected",
            "prediction indices equal input token indices minus one",
            "no accepted row is truncated or exceeds max_seq_length",
            "accepted and rejected rows exactly partition the char-span candidates",
            "summary counts and checksums are exact",
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
