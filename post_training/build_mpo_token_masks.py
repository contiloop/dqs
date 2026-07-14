#!/usr/bin/env python3
"""Convert char-span preference candidates into Gemma-aligned mPO masks.

This mirrors the repository's SFT serialization contract:

1. tokenize the stored prompt with ``add_special_tokens=False``;
2. strip the completion, append the configured EOS string, and tokenize the
   completion separately with ``add_special_tokens=False``;
3. concatenate prompt and completion token IDs;
4. keep input-aligned completion and term masks.  The causal one-token shift is
   applied later by ``mpo_masking.masked_causal_logp_mean``.

Chosen and rejected completions are encoded independently.  A row is rejected
instead of truncated, and a term token that crosses into non-whitespace text
outside the annotated char span is rejected because it cannot express a clean
term-only objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dqs_mpo_token_masks_v1"


class TokenMaskRejected(ValueError):
    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates.jsonl",
    )
    parser.add_argument(
        "--effective-config",
        type=Path,
        default=root / "raw" / "effective_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs.jsonl",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_rejections.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_summary.json",
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        default=root / "analysis" / "sample_ids.txt",
    )
    parser.add_argument(
        "--sample-jsonl",
        type=Path,
        default=root / "analysis" / "mpo_token_mask_sample_10.jsonl",
    )
    parser.add_argument(
        "--sample-markdown",
        type=Path,
        default=root / "analysis" / "MPO_TOKEN_MASK_SAMPLE_10.md",
    )
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_contract(path: Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, Mapping):
        raise SystemExit(f"Invalid config mapping: {path}")
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    training = cfg.get("training", {})
    if not all(isinstance(value, Mapping) for value in (model, data, training)):
        raise SystemExit("model/data/training config sections must be mappings")
    configured_max = training.get("max_seq_length")
    if configured_max is None:
        configured_max = (
            int(data.get("max_input_tokens", 1280) or 1280)
            + int(data.get("max_output_tokens", 1500) or 1500)
            + int(training.get("prompt_overhead_tokens", 128) or 128)
        )
    return {
        "tokenizer_name": str(model.get("name_or_path", "")).strip(),
        "tokenizer_revision": str(model.get("tokenizer_revision", "main")),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
        "max_seq_length": int(configured_max),
        "append_eos": bool(training.get("append_eos_token", True)),
        "response_only_loss": bool(training.get("response_only_loss", True)),
        "prevent_truncation": bool(training.get("prevent_template_truncation", True)),
    }


def _flatten(values: Any, *, field: str) -> list[Any]:
    if values is None:
        raise TypeError(f"tokenizer output is missing {field}")
    result = list(values)
    if result and isinstance(result[0], (list, tuple)) and field != "offset_mapping":
        if len(result) != 1:
            raise TypeError(f"batched tokenizer output is not supported for {field}")
        result = list(result[0])
    if field == "offset_mapping" and result and isinstance(result[0], list):
        first = result[0]
        if first and isinstance(first[0], (list, tuple)):
            if len(result) != 1:
                raise TypeError("batched offset_mapping is not supported")
            result = list(first)
    return result


def encode(
    tokenizer: Any,
    text: str,
    *,
    offsets: bool,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    kwargs = {"add_special_tokens": False}
    if offsets:
        kwargs["return_offsets_mapping"] = True
    try:
        encoded = tokenizer(text=text, **kwargs)
    except TypeError:
        encoded = tokenizer(text, **kwargs)
    if isinstance(encoded, Mapping):
        raw_ids = encoded.get("input_ids")
        raw_offsets = encoded.get("offset_mapping")
    else:
        raw_ids = getattr(encoded, "input_ids", None)
        raw_offsets = getattr(encoded, "offset_mapping", None)
    ids = [int(value) for value in _flatten(raw_ids, field="input_ids")]
    if not offsets:
        return ids, None
    parsed_offsets = [
        (int(value[0]), int(value[1]))
        for value in _flatten(raw_offsets, field="offset_mapping")
    ]
    if len(parsed_offsets) != len(ids):
        raise TypeError("offset_mapping length does not match input_ids")
    return ids, parsed_offsets


def parse_spans(value: Any, *, pair_id: str, side: str, text_length: int) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise TokenMaskRejected(f"{side}_missing_term_char_spans", pair_id=pair_id)
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for span_index, raw in enumerate(value):
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or not isinstance(raw[0], int)
            or not isinstance(raw[1], int)
        ):
            raise TokenMaskRejected(
                f"{side}_invalid_term_char_span",
                pair_id=pair_id,
                span_index=span_index,
                span=raw,
            )
        start, end = int(raw[0]), int(raw[1])
        if not 0 <= start < end <= text_length or start < previous_end:
            raise TokenMaskRejected(
                f"{side}_invalid_term_char_span",
                pair_id=pair_id,
                span_index=span_index,
                span=[start, end],
                text_length=text_length,
            )
        spans.append((start, end))
        previous_end = end
    return spans


def _non_whitespace_gap(text: str, start: int, end: int) -> str:
    if end <= start:
        return ""
    value = text[start:end]
    return value if value.strip() else ""


def align_spans(
    *,
    tokenizer: Any,
    pair_id: str,
    side: str,
    text: str,
    spans: Sequence[tuple[int, int]],
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    prompt_token_count: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    local_mask = [0] * len(token_ids)
    used_by_span: dict[int, int] = {}
    token_strings = [str(value) for value in tokenizer.convert_ids_to_tokens(list(token_ids))]
    alignments: list[dict[str, Any]] = []

    for span_index, (span_start, span_end) in enumerate(spans):
        selected = [
            token_index
            for token_index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and max(token_start, span_start) < min(token_end, span_end)
        ]
        if not selected:
            raise TokenMaskRejected(
                f"{side}_empty_term_mask",
                pair_id=pair_id,
                span_index=span_index,
                char_span=[span_start, span_end],
                term_text=text[span_start:span_end],
            )

        clipped_intervals: list[tuple[int, int]] = []
        boundary_whitespace = ""
        for token_index in selected:
            token_start, token_end = offsets[token_index]
            visible_end = min(token_end, len(text))
            left_outside = text[token_start:min(visible_end, span_start)] if token_start < span_start else ""
            right_start = max(token_start, span_end)
            right_outside = text[right_start:visible_end] if visible_end > span_end else ""
            outside = left_outside + right_outside
            if outside and outside.strip():
                raise TokenMaskRejected(
                    f"{side}_token_crosses_term_boundary",
                    pair_id=pair_id,
                    span_index=span_index,
                    char_span=[span_start, span_end],
                    term_text=text[span_start:span_end],
                    local_token_index=token_index,
                    token_id=int(token_ids[token_index]),
                    token=token_strings[token_index],
                    token_offset=[token_start, token_end],
                    token_surface=text[token_start:visible_end],
                    outside_text=outside,
                )
            boundary_whitespace += outside
            clipped_intervals.append((max(token_start, span_start), min(token_end, span_end)))
            if token_index in used_by_span:
                raise TokenMaskRejected(
                    f"{side}_term_spans_share_token",
                    pair_id=pair_id,
                    first_span_index=used_by_span[token_index],
                    second_span_index=span_index,
                    local_token_index=token_index,
                )
            used_by_span[token_index] = span_index

        cursor = span_start
        for covered_start, covered_end in sorted(clipped_intervals):
            if covered_start > cursor:
                gap = _non_whitespace_gap(text, cursor, covered_start)
                if gap:
                    raise TokenMaskRejected(
                        f"{side}_term_chars_not_token_covered",
                        pair_id=pair_id,
                        span_index=span_index,
                        char_span=[span_start, span_end],
                        uncovered_text=gap,
                    )
            cursor = max(cursor, covered_end)
        if cursor < span_end:
            gap = _non_whitespace_gap(text, cursor, span_end)
            if gap:
                raise TokenMaskRejected(
                    f"{side}_term_chars_not_token_covered",
                    pair_id=pair_id,
                    span_index=span_index,
                    char_span=[span_start, span_end],
                    uncovered_text=gap,
                )

        global_indices = [prompt_token_count + index for index in selected]
        if any(index <= 0 for index in global_indices):
            raise TokenMaskRejected(
                f"{side}_term_token_has_no_causal_predictor",
                pair_id=pair_id,
                span_index=span_index,
            )
        for token_index in selected:
            local_mask[token_index] = 1
        alignments.append(
            {
                "span_index": span_index,
                "char_span": [span_start, span_end],
                "term_text": text[span_start:span_end],
                "local_token_indices": selected,
                "input_token_indices": global_indices,
                "prediction_logit_indices": [index - 1 for index in global_indices],
                "token_ids": [int(token_ids[index]) for index in selected],
                "tokens": [token_strings[index] for index in selected],
                "token_offsets": [[int(offsets[index][0]), int(offsets[index][1])] for index in selected],
                "boundary_whitespace_only": bool(boundary_whitespace),
            }
        )

    if sum(local_mask) == 0:
        raise TokenMaskRejected(f"{side}_empty_term_mask", pair_id=pair_id)
    return local_mask, alignments


def build_side(
    *,
    tokenizer: Any,
    pair_id: str,
    side: str,
    prompt_ids: Sequence[int],
    text: str,
    raw_spans: Any,
    append_eos: bool,
) -> dict[str, Any]:
    if not text.strip():
        raise TokenMaskRejected(f"{side}_empty_completion_text", pair_id=pair_id)
    raw_spans_parsed = parse_spans(
        raw_spans,
        pair_id=pair_id,
        side=side,
        text_length=len(text),
    )
    completion_text = text.strip()
    left_trim = len(text) - len(text.lstrip())
    right_trim = len(text) - len(text.rstrip())
    spans: list[tuple[int, int]] = []
    for span_index, (start, end) in enumerate(raw_spans_parsed):
        adjusted = (start - left_trim, end - left_trim)
        if not 0 <= adjusted[0] < adjusted[1] <= len(completion_text):
            raise TokenMaskRejected(
                f"{side}_term_span_removed_by_completion_strip",
                pair_id=pair_id,
                span_index=span_index,
                raw_char_span=[start, end],
                left_trim=left_trim,
                right_trim=right_trim,
            )
        spans.append(adjusted)
    eos_token = getattr(tokenizer, "eos_token", None)
    if append_eos and not eos_token:
        raise TokenMaskRejected(f"{side}_tokenizer_has_no_eos", pair_id=pair_id)
    eos_appended = bool(append_eos and eos_token and not completion_text.endswith(str(eos_token)))
    completion_for_tokens = completion_text + str(eos_token) if eos_appended else completion_text
    completion_ids, offsets = encode(tokenizer, completion_for_tokens, offsets=True)
    assert offsets is not None
    if not completion_ids:
        raise TokenMaskRejected(f"{side}_empty_completion_tokens", pair_id=pair_id)
    local_term_mask, alignments = align_spans(
        tokenizer=tokenizer,
        pair_id=pair_id,
        side=side,
        text=completion_text,
        spans=spans,
        token_ids=completion_ids,
        offsets=offsets,
        prompt_token_count=len(prompt_ids),
    )
    input_ids = list(prompt_ids) + completion_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    term_mask = [0] * len(prompt_ids) + local_term_mask
    term_indices = [index for index, value in enumerate(term_mask) if value]
    return {
        "completion_text": completion_text,
        "term_char_spans": [[start, end] for start, end in spans],
        "outer_whitespace_stripped_chars": [left_trim, right_trim],
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "term_mask": term_mask,
        "sequence_token_count": len(input_ids),
        "completion_token_count": len(completion_ids),
        "term_token_count": len(term_indices),
        "term_token_indices": term_indices,
        "term_prediction_indices": [index - 1 for index in term_indices],
        "term_alignments": alignments,
        "eos_appended": eos_appended,
    }


def build_tokenized_pair(
    *,
    tokenizer: Any,
    row: Mapping[str, Any],
    max_seq_length: int,
    append_eos: bool,
) -> dict[str, Any]:
    pair_id = str(row.get("pair_id", ""))
    prompt = str(row.get("prompt", "") or "")
    if not pair_id or not prompt.strip():
        raise TokenMaskRejected("missing_pair_id_or_prompt", pair_id=pair_id)
    prompt_ids, _ = encode(tokenizer, prompt, offsets=False)
    if not prompt_ids:
        raise TokenMaskRejected("empty_prompt_tokens", pair_id=pair_id)

    chosen = build_side(
        tokenizer=tokenizer,
        pair_id=pair_id,
        side="chosen",
        prompt_ids=prompt_ids,
        text=str(row.get("chosen", "")),
        raw_spans=row.get("chosen_term_char_spans"),
        append_eos=append_eos,
    )
    rejected = build_side(
        tokenizer=tokenizer,
        pair_id=pair_id,
        side="rejected",
        prompt_ids=prompt_ids,
        text=str(row.get("rejected", "")),
        raw_spans=row.get("rejected_term_char_spans"),
        append_eos=append_eos,
    )
    for side, payload in (("chosen", chosen), ("rejected", rejected)):
        if payload["sequence_token_count"] > max_seq_length:
            raise TokenMaskRejected(
                f"{side}_sequence_too_long",
                pair_id=pair_id,
                sequence_token_count=payload["sequence_token_count"],
                max_seq_length=max_seq_length,
            )
    chosen_term_ids = [chosen["input_ids"][index] for index in chosen["term_token_indices"]]
    rejected_term_ids = [rejected["input_ids"][index] for index in rejected["term_token_indices"]]
    if chosen_term_ids == rejected_term_ids:
        raise TokenMaskRejected(
            "identical_chosen_rejected_term_token_ids",
            pair_id=pair_id,
            term_token_ids=chosen_term_ids,
        )

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "subset": row.get("subset"),
        "teacher_label": row.get("teacher_label"),
        "term_annotation_count": row.get("term_annotation_count"),
        "replacement_span_count": row.get("replacement_span_count"),
        "has_quality_warnings": bool(row.get("has_quality_warnings")),
        "quality_flags": list(row.get("quality_flags", [])),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_count": len(prompt_ids),
        "chosen": chosen["completion_text"],
        "rejected": rejected["completion_text"],
        "chosen_term_char_spans": chosen["term_char_spans"],
        "rejected_term_char_spans": rejected["term_char_spans"],
        "term_token_lengths_differ": chosen["term_token_count"] != rejected["term_token_count"],
        "mask_alignment": "input_aligned_shift_with_mask[:,1:]_against_logits[:,:-1]",
    }
    for side, payload in (("chosen", chosen), ("rejected", rejected)):
        for key, value in payload.items():
            if key == "completion_text":
                continue
            output[f"{side}_{key}"] = value
    return output


def percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return int(ordered[index])


def read_sample_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def cached_resolved_revision(tokenizer_name: str, requested_revision: str) -> str:
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        hub_root = Path(hub_cache)
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub_root = hf_home / "hub"
    ref_path = (
        hub_root
        / f"models--{tokenizer_name.replace('/', '--')}"
        / "refs"
        / requested_revision
    )
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    return ""


def sample_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Gemma mPO token-mask 샘플 10개",
        "",
        "각 mask는 최종 `input_ids`에 정렬되어 있다. `prediction_logit_indices`는 causal shift 후 "
        "실제로 해당 token을 예측하는 `logits` 위치다. Prompt와 EOS는 term mask에서 0이다.",
        "",
    ]
    for sample_index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {sample_index}. {row['pair_id']}",
                "",
                (
                    f"- prompt tokens: `{row['prompt_token_count']}`; chosen sequence: "
                    f"`{row['chosen_sequence_token_count']}`; rejected sequence: "
                    f"`{row['rejected_sequence_token_count']}`"
                ),
                (
                    f"- M+ tokens: `{row['chosen_term_token_count']}`; M- tokens: "
                    f"`{row['rejected_term_token_count']}`; lengths differ: "
                    f"`{str(bool(row['term_token_lengths_differ'])).lower()}`"
                ),
                "",
                "### M+ (chosen)",
                "",
            ]
        )
        for alignment in row["chosen_term_alignments"]:
            token_view = ", ".join(
                f"{token!r}@input[{input_index}]/logits[{logit_index}]"
                for token, input_index, logit_index in zip(
                    alignment["tokens"],
                    alignment["input_token_indices"],
                    alignment["prediction_logit_indices"],
                    strict=True,
                )
            )
            lines.append(f"- `{alignment['term_text']}` → {token_view}")
        lines.extend(["", "### M- (rejected)", ""])
        for alignment in row["rejected_term_alignments"]:
            token_view = ", ".join(
                f"{token!r}@input[{input_index}]/logits[{logit_index}]"
                for token, input_index, logit_index in zip(
                    alignment["tokens"],
                    alignment["input_token_indices"],
                    alignment["prediction_logit_indices"],
                    strict=True,
                )
            )
            lines.append(f"- `{alignment['term_text']}` → {token_view}")
        lines.extend(
            [
                "",
                "y+",
                "",
                "```text",
                str(row["chosen"]),
                "```",
                "",
                "y-",
                "",
                "```text",
                str(row["rejected"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    contract = load_contract(args.effective_config)
    tokenizer_name = args.tokenizer_name or contract["tokenizer_name"]
    tokenizer_revision = args.tokenizer_revision or contract["tokenizer_revision"]
    max_seq_length = args.max_seq_length or contract["max_seq_length"]
    append_eos = contract["append_eos"] if args.append_eos is None else bool(args.append_eos)
    if not tokenizer_name:
        raise SystemExit("tokenizer name is empty")
    if not contract["response_only_loss"]:
        raise SystemExit("This artifact requires training.response_only_loss=true")
    if not contract["prevent_truncation"]:
        raise SystemExit("This artifact requires training.prevent_template_truncation=true")

    os.environ.setdefault(
        "HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface")
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=contract["trust_remote_code"],
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise SystemExit("A fast tokenizer with offset_mapping support is required")

    for path in (args.output, args.rejections, args.summary, args.sample_jsonl, args.sample_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)

    requested_sample_ids = read_sample_ids(args.sample_ids)
    requested_set = set(requested_sample_ids)
    samples: dict[str, dict[str, Any]] = {}
    supplemental_samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    chosen_term_counts: list[int] = []
    rejected_term_counts: list[int] = []

    with (
        args.candidates.open("r", encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as accepted_handle,
        args.rejections.open("w", encoding="utf-8") as rejected_handle,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise SystemExit(f"Expected JSON object at {args.candidates}:{line_number}")
            counts["input_rows"] += 1
            try:
                tokenized = build_tokenized_pair(
                    tokenizer=tokenizer,
                    row=row,
                    max_seq_length=max_seq_length,
                    append_eos=append_eos,
                )
            except TokenMaskRejected as exc:
                counts["rejected_rows"] += 1
                reject_reasons[exc.reason] += 1
                write_jsonl_row(
                    rejected_handle,
                    {
                        "pair_id": row.get("pair_id"),
                        "reason": exc.reason,
                        "details": exc.details,
                        "chosen_term_char_spans": row.get("chosen_term_char_spans"),
                        "rejected_term_char_spans": row.get("rejected_term_char_spans"),
                    },
                )
                continue

            counts["accepted_rows"] += 1
            counts["chosen_term_tokens"] += int(tokenized["chosen_term_token_count"])
            counts["rejected_term_tokens"] += int(tokenized["rejected_term_token_count"])
            if tokenized["term_token_lengths_differ"]:
                counts["rows_with_different_term_token_counts"] += 1
            if tokenized["has_quality_warnings"]:
                counts["accepted_rows_with_char_quality_warnings"] += 1
            chosen_lengths.append(int(tokenized["chosen_sequence_token_count"]))
            rejected_lengths.append(int(tokenized["rejected_sequence_token_count"]))
            chosen_term_counts.append(int(tokenized["chosen_term_token_count"]))
            rejected_term_counts.append(int(tokenized["rejected_term_token_count"]))
            write_jsonl_row(accepted_handle, tokenized)
            pair_id = str(tokenized["pair_id"])
            if pair_id in requested_set:
                samples[pair_id] = tokenized
            if (
                len(supplemental_samples) < max(30, len(requested_sample_ids) * 3)
                and not tokenized["has_quality_warnings"]
                and len(str(tokenized["chosen"])) <= 500
                and len(str(tokenized["rejected"])) <= 500
            ):
                supplemental_samples.append(tokenized)

    missing_samples = [pair_id for pair_id in requested_sample_ids if pair_id not in samples]
    ordered_samples = [samples[pair_id] for pair_id in requested_sample_ids if pair_id in samples]
    sample_target = max(10, len(requested_sample_ids))
    selected_ids = {str(row["pair_id"]) for row in ordered_samples}
    for row in supplemental_samples:
        if len(ordered_samples) >= sample_target:
            break
        if str(row["pair_id"]) in selected_ids:
            continue
        ordered_samples.append(row)
        selected_ids.add(str(row["pair_id"]))
    if len(ordered_samples) < sample_target:
        raise SystemExit(
            f"Could not assemble {sample_target} accepted token-mask samples; found {len(ordered_samples)}"
        )
    with args.sample_jsonl.open("w", encoding="utf-8") as handle:
        for row in ordered_samples:
            write_jsonl_row(handle, row)
    args.sample_markdown.write_text(sample_markdown(ordered_samples), encoding="utf-8")

    resolved_revision = str(getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or "")
    if not resolved_revision:
        resolved_revision = cached_resolved_revision(tokenizer_name, tokenizer_revision)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "candidate_path": str(args.candidates.resolve()),
            "candidate_sha256": sha256(args.candidates),
        },
        "tokenizer": {
            "name": tokenizer_name,
            "requested_revision": tokenizer_revision,
            "resolved_revision": resolved_revision,
            "class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
        },
        "training_contract": {
            "prompt_and_completion_tokenized_separately": True,
            "add_special_tokens": False,
            "append_eos": append_eos,
            "response_only_sft": True,
            "max_seq_length": max_seq_length,
            "truncation": "reject row",
            "padding": "dynamic right padding; attention/completion/term masks pad with zero",
            "causal_shift": "logits[:, :-1] predicts input_ids[:, 1:]; use mask[:, 1:]",
            "chosen_rejected_masks": "independent offset alignment",
            "normalization": "per row and per mask token count before batch reduction",
            "boundary_policy": "allow whitespace-only token overhang; reject non-whitespace overhang",
        },
        "counts": dict(sorted(counts.items())),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "samples": {
            "requested_ids": requested_sample_ids,
            "requested_ids_rejected": missing_samples,
            "written_ids": [str(row["pair_id"]) for row in ordered_samples],
        },
        "lengths": {
            "chosen_sequence": {
                "min": min(chosen_lengths, default=0),
                "p50": percentile(chosen_lengths, 0.50),
                "p95": percentile(chosen_lengths, 0.95),
                "max": max(chosen_lengths, default=0),
            },
            "rejected_sequence": {
                "min": min(rejected_lengths, default=0),
                "p50": percentile(rejected_lengths, 0.50),
                "p95": percentile(rejected_lengths, 0.95),
                "max": max(rejected_lengths, default=0),
            },
            "chosen_term_tokens": {
                "min": min(chosen_term_counts, default=0),
                "p50": percentile(chosen_term_counts, 0.50),
                "p95": percentile(chosen_term_counts, 0.95),
                "max": max(chosen_term_counts, default=0),
            },
            "rejected_term_tokens": {
                "min": min(rejected_term_counts, default=0),
                "p50": percentile(rejected_term_counts, 0.50),
                "p95": percentile(rejected_term_counts, 0.95),
                "max": max(rejected_term_counts, default=0),
            },
        },
        "outputs": {
            "tokenized_path": str(args.output.resolve()),
            "tokenized_sha256": sha256(args.output),
            "rejections_path": str(args.rejections.resolve()),
            "rejections_sha256": sha256(args.rejections),
            "sample_jsonl_path": str(args.sample_jsonl.resolve()),
            "sample_markdown_path": str(args.sample_markdown.resolve()),
        },
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
