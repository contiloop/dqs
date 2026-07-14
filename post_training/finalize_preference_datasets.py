#!/usr/bin/env python3
"""Finalize source-quality judgments and build mPO plus DPO/CPO datasets.

This is intentionally a fail-closed finalization step.  It combines the two
already-saved source judgment journals, resolves the 93 REVIEW rows with the
explicit adjudication policy below, and never repairs or substitutes a row.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "source_quality" / "gpt54mini_source_integrity_v1"

DEFAULT_REQUESTS = RUN_DIR / "requests.jsonl"
DEFAULT_MODEL_DECISIONS = RUN_DIR / "decision_journal.jsonl"
DEFAULT_CODEX_DECISIONS = RUN_DIR / "codex_missing_decisions.jsonl"
DEFAULT_CANDIDATES = ROOT / "prepared" / "preference_candidates_strict_v2.jsonl"
DEFAULT_TOKENIZED = ROOT / "prepared" / "mpo_tokenized_pairs_strict_v2.jsonl"
DEFAULT_PARENT_CONTRACT = ROOT / "dataset_contract_strict_v2.json"
DEFAULT_RAW_DIR = ROOT / "raw" / "golden_pairs"

DEFAULT_FINAL_DECISIONS = RUN_DIR / "final_decisions.jsonl"
DEFAULT_HUMAN_LEDGER = ROOT / "analysis" / "source_quality_human_adjudications.jsonl"
DEFAULT_FINAL_CANDIDATES = (
    ROOT / "prepared" / "preference_candidates_final_source_filtered.jsonl"
)
DEFAULT_FINAL_TOKENIZED = (
    ROOT / "prepared" / "mpo_tokenized_pairs_final_source_filtered.jsonl"
)
DEFAULT_REJECTIONS = ROOT / "analysis" / "source_quality_final_rejections.jsonl"
DEFAULT_PREFERENCE_REJECTIONS = (
    ROOT / "analysis" / "preference_quality_rejections.jsonl"
)
DEFAULT_SUMMARY = ROOT / "analysis" / "source_quality_final_summary.json"
DEFAULT_MPO_CONTRACT = ROOT / "dataset_contract_final_source_filtered.json"
DEFAULT_FULL_PAIRS = ROOT / "prepared" / "full_response_preference_pairs_final.jsonl"
DEFAULT_FULL_PAIR_REJECTIONS = (
    ROOT / "analysis" / "full_response_preference_rejections.jsonl"
)
DEFAULT_FULL_PAIR_CONTRACT = ROOT / "dataset_contract_full_response_preference.json"
DEFAULT_FULL_PAIR_SUMMARY = ROOT / "analysis" / "full_response_preference_summary.json"
DEFAULT_CPO_TOKENIZED = (
    ROOT / "prepared" / "cpo_tokenized_full_response_pairs_final.jsonl"
)
DEFAULT_CPO_CONTRACT = ROOT / "dataset_contract_cpo_full_response.json"

SOURCE_DECISION_SCHEMA = "dqs.source_integrity_final.v1"
HUMAN_ADJUDICATION_SCHEMA = "dqs.source_integrity_human_adjudication.v1"
PREFERENCE_REJECTION_SCHEMA = "dqs.preference_quality_rejection.v1"
FULL_PAIR_SCHEMA = "dqs.full_response_preference.v1"
CPO_TOKEN_SCHEMA = "dqs_full_response_cpo_token_masks_v1"
FINALIZATION_POLICY = "direct_review_93_fail_closed_v1"

EXPECTED_REQUEST_COUNT = 5_505
EXPECTED_INPUT_DECISIONS = {"KEEP": 5_196, "REJECT": 216, "REVIEW": 93}
EXPECTED_FINAL_DECISIONS = {"KEEP": 5_201, "REJECT": 304}
EXPECTED_PREFERENCE_REJECTIONS = 1
EXPECTED_MPO_ROWS = 5_200
EXPECTED_FULL_PAIR_ROWS = 5_200

TRAIN_COLUMNS = (
    "pair_id",
    "chosen_input_ids",
    "chosen_completion_mask",
    "chosen_term_mask",
    "rejected_input_ids",
    "rejected_completion_mask",
    "rejected_term_mask",
)


# Only these five REVIEW rows have a fully recoverable heading/document
# boundary.  Every other REVIEW row was directly inspected and is excluded.
PROMOTED_REVIEW_RATIONALES: dict[str, tuple[str, str]] = {
    "subset_002:row_000001518947": (
        "intact_fragment_or_heading",
        "GOODWILL is an unmistakable section heading fused to an otherwise complete sentence; the source meaning is unambiguous.",
    ),
    "subset_005:row_000001483865": (
        "intact_fragment_or_heading",
        "EXIT ACTIVITIES is an unmistakable section heading fused to an otherwise complete sentence; the source meaning is unambiguous.",
    ),
    "subset_009:row_000002190520": (
        "intact_fragment_or_heading",
        "Income Taxes (continued) is an unmistakable continued-section heading; the following clause remains unambiguous.",
    ),
    "subset_011:row_000000151470": (
        "minor_noise_readable",
        "Holocaust Survivor is a recognizable subheading inside a complete and coherent review; no semantic content is lost.",
    ),
    "subset_014:row_000000011691": (
        "minor_noise_readable",
        "The rating-to-title transition is a recoverable boundary between two complete review blocks; both blocks remain coherent.",
    ),
}

REVIEW_REJECTION_POLICY: dict[str, tuple[str, str]] = {
    "borderline_delimiter_loss": (
        "lost_delimiters",
        "Direct review found that missing delimiters or layout make a boundary or field association unsafe.",
    ),
    "borderline_fragmentation": (
        "severe_truncation",
        "Direct review found material truncation, fragmentation, or an incomplete semantic unit.",
    ),
    "uncertain_structure": (
        "mixed_or_reordered_segments",
        "Direct review could not reconstruct the source structure unambiguously.",
    ),
}


# The source itself is usable, but the stored Teacher and Student responses are
# identical while a stale terminology error still points to text at another
# semantic position.  Reverting that annotation would fabricate an mPO
# preference that the post-edit does not demonstrate, so this row is excluded
# after source review and before every preference objective.
PREFERENCE_EXCLUSIONS: dict[str, tuple[str, str]] = {
    "subset_012:row_000001531000": (
        "teacher_student_identical_annotation_misalignment",
        "Raw Teacher target and Student translation are NFC-identical; the stored terminology annotation is positionally inconsistent and cannot support a faithful synthetic mPO negative.",
    ),
}


class FinalizationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FinalizationError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise FinalizationError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def unique_by_id(
    rows: Iterable[Mapping[str, Any]], *, key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        row_id = row.get(key)
        if not isinstance(row_id, str) or not row_id:
            raise FinalizationError(f"{label}[{index}] has no non-empty {key}")
        if row_id in result:
            raise FinalizationError(f"duplicate {key} in {label}: {row_id}")
        result[row_id] = dict(row)
    return result


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def semantic_tokenized_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        semantic: dict[str, Any] = {"pair_id": pair_id}
        for field in TRAIN_COLUMNS[1:]:
            values = row.get(field)
            if not isinstance(values, list):
                raise FinalizationError(f"{pair_id}: missing tensor field {field}")
            semantic[field] = [int(value) for value in values]
        digest.update(
            json.dumps(
                semantic,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_raw_golden(raw_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    files = sorted(raw_dir.glob("subset_*.jsonl"))
    if len(files) != 23:
        raise FinalizationError(f"expected 23 raw subset files, found {len(files)}")
    for path in files:
        for raw in read_jsonl(path):
            row_id = raw.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise FinalizationError(f"raw row without id: {path}")
            pair_id = f"{path.stem}:{row_id}"
            if pair_id in rows:
                raise FinalizationError(f"duplicate raw pair_id: {pair_id}")
            rows[pair_id] = raw
    return rows


def resolve_review(
    pair_id: str, original: Mapping[str, Any]
) -> tuple[str, str, str]:
    if pair_id in PROMOTED_REVIEW_RATIONALES:
        reason_code, explanation = PROMOTED_REVIEW_RATIONALES[pair_id]
        return "KEEP", reason_code, explanation
    original_reason = str(original.get("reason_code", ""))
    if original_reason not in REVIEW_REJECTION_POLICY:
        raise FinalizationError(
            f"{pair_id}: unsupported REVIEW reason_code={original_reason!r}"
        )
    reason_code, explanation = REVIEW_REJECTION_POLICY[original_reason]
    return "REJECT", reason_code, explanation


def finalize_decisions(
    requests: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    codex_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(requests) != EXPECTED_REQUEST_COUNT:
        raise FinalizationError(
            f"request count drift: {len(requests)} != {EXPECTED_REQUEST_COUNT}"
        )
    request_by_id = unique_by_id(requests, key="pair_id", label="requests")
    decisions: dict[str, dict[str, Any]] = {}
    for label, rows in (("model decisions", model_rows), ("Codex decisions", codex_rows)):
        for pair_id, row in unique_by_id(rows, key="pair_id", label=label).items():
            if pair_id in decisions:
                raise FinalizationError(f"decision journals overlap at {pair_id}")
            decisions[pair_id] = row
    if set(decisions) != set(request_by_id):
        missing = sorted(set(request_by_id) - set(decisions))
        extra = sorted(set(decisions) - set(request_by_id))
        raise FinalizationError(
            f"decision coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    observed = Counter(str(row.get("decision")) for row in decisions.values())
    if dict(observed) != EXPECTED_INPUT_DECISIONS:
        raise FinalizationError(
            f"input decision drift: {dict(observed)} != {EXPECTED_INPUT_DECISIONS}"
        )

    review_ids = {
        pair_id for pair_id, row in decisions.items() if row.get("decision") == "REVIEW"
    }
    missing_promotions = sorted(set(PROMOTED_REVIEW_RATIONALES) - review_ids)
    if missing_promotions:
        raise FinalizationError(f"promoted IDs are not REVIEW rows: {missing_promotions}")

    final_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    for request in requests:
        pair_id = str(request["pair_id"])
        original = decisions[pair_id]
        if original.get("source_sha256") != request.get("source_sha256"):
            raise FinalizationError(f"source hash drift for {pair_id}")
        decision = str(original["decision"])
        reason_code = str(original["reason_code"])
        evidence = str(original.get("evidence", ""))
        explanation = str(original.get("explanation", ""))
        decision_source = "saved_judgment"
        if decision == "REVIEW":
            decision, reason_code, explanation = resolve_review(pair_id, original)
            evidence = "" if decision == "KEEP" else evidence
            decision_source = "direct_review_adjudication"
            human_rows.append(
                {
                    "schema_version": HUMAN_ADJUDICATION_SCHEMA,
                    "policy_version": FINALIZATION_POLICY,
                    "order_idx": int(request["order_idx"]),
                    "pair_id": pair_id,
                    "source": request["source"],
                    "source_sha256": request["source_sha256"],
                    "original_decision": original["decision"],
                    "original_reason_code": original["reason_code"],
                    "original_evidence": original.get("evidence", ""),
                    "original_explanation": original.get("explanation", ""),
                    "final_decision": decision,
                    "final_reason_code": reason_code,
                    "final_explanation": explanation,
                    "reviewer": "root_codex_direct_review",
                }
            )

        final_rows.append(
            {
                "schema_version": SOURCE_DECISION_SCHEMA,
                "policy_version": FINALIZATION_POLICY,
                "order_idx": int(request["order_idx"]),
                "pair_id": pair_id,
                "source_sha256": request["source_sha256"],
                "decision": decision,
                "reason_code": reason_code,
                "evidence": evidence,
                "explanation": explanation,
                "decision_source": decision_source,
                "original_decision": original["decision"],
                "original_reason_code": original["reason_code"],
                "judge_backend": original.get("judge_backend", original.get("provider")),
                "judge_model": original.get(
                    "judge_model", original.get("response_model", original.get("request_model"))
                ),
            }
        )

    final_counts = Counter(row["decision"] for row in final_rows)
    if dict(final_counts) != EXPECTED_FINAL_DECISIONS:
        raise FinalizationError(
            f"final decision drift: {dict(final_counts)} != {EXPECTED_FINAL_DECISIONS}"
        )
    if len(human_rows) != EXPECTED_INPUT_DECISIONS["REVIEW"]:
        raise FinalizationError("human adjudication ledger does not contain all REVIEW rows")
    return final_rows, human_rows


def validate_candidate_against_raw(
    candidate: Mapping[str, Any], raw: Mapping[str, Any]
) -> None:
    pair_id = str(candidate["pair_id"])
    chosen = candidate.get("chosen")
    student = candidate.get("student_translation")
    if chosen != raw.get("target"):
        raise FinalizationError(f"{pair_id}: chosen is not byte-identical to raw target")
    if student != raw.get("student_translation"):
        raise FinalizationError(
            f"{pair_id}: student_translation is not byte-identical to raw student output"
        )
    for field, value in (("prompt", candidate.get("prompt")), ("chosen", chosen), ("student", student)):
        if not isinstance(value, str) or not value:
            raise FinalizationError(f"{pair_id}: empty {field}")


def responses_are_distinct(candidate: Mapping[str, Any]) -> bool:
    return unicodedata.normalize("NFC", str(candidate["chosen"])) != unicodedata.normalize(
        "NFC", str(candidate["student_translation"])
    )


def build_full_pair(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": FULL_PAIR_SCHEMA,
        "pair_id": candidate["pair_id"],
        "subset": candidate["subset"],
        "row_id": candidate["row_id"],
        "prompt": candidate["prompt"],
        "chosen": candidate["chosen"],
        "rejected": candidate["student_translation"],
        "chosen_source": "teacher_post_edit_raw_target",
        "rejected_source": "student_translation_raw_output",
        "source": candidate["source"],
        "teacher_label": candidate.get("teacher_label"),
        "qe_score": candidate.get("qe_score"),
        "term_annotation_count": candidate.get("term_annotation_count"),
    }


def full_pair_tokenization_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the exact TRL 0.24 DPO/CPO tokenization paths."""

    os.environ.setdefault("HF_HOME", str((ROOT / ".cache" / "huggingface").resolve()))
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise FinalizationError(
            "transformers is required to pin full-response token lengths"
        ) from exc

    tokenizer_name = str(parent_contract["tokenizer_name"])
    tokenizer_revision = str(parent_contract["tokenizer_resolved_revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        local_files_only=True,
    )
    if tokenizer.pad_token_id != int(parent_contract["pad_token_id"]):
        raise FinalizationError("full-pair tokenizer pad_token_id drift")
    if tokenizer.eos_token_id != int(parent_contract["eos_token_id"]):
        raise FinalizationError("full-pair tokenizer eos_token_id drift")

    maxima: dict[str, tuple[int, str]] = {
        "prompt_tokens": (0, ""),
        "dpo_chosen_completion_tokens": (0, ""),
        "dpo_rejected_completion_tokens": (0, ""),
        "dpo_chosen_sequence_tokens": (0, ""),
        "dpo_rejected_sequence_tokens": (0, ""),
        "cpo_chosen_sequence_tokens": (0, ""),
        "cpo_rejected_sequence_tokens": (0, ""),
    }

    def record(name: str, value: int, pair_id: str) -> None:
        if value > maxima[name][0]:
            maxima[name] = (value, pair_id)

    eos = int(tokenizer.eos_token_id)
    bos = tokenizer.bos_token_id
    for row in rows:
        pair_id = str(row["pair_id"])
        prompt = str(row["prompt"])
        chosen = str(row["chosen"])
        rejected = str(row["rejected"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"] + [eos]
        rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"] + [eos]
        record("prompt_tokens", len(prompt_ids), pair_id)
        record("dpo_chosen_completion_tokens", len(chosen_ids), pair_id)
        record("dpo_rejected_completion_tokens", len(rejected_ids), pair_id)
        record("dpo_chosen_sequence_tokens", len(prompt_ids) + len(chosen_ids), pair_id)
        record("dpo_rejected_sequence_tokens", len(prompt_ids) + len(rejected_ids), pair_id)

        # CPOTrainer 0.24 tokenizes prompt+answer jointly, preserves an existing
        # BOS, and appends EOS only when absent.
        for side, answer in (("chosen", chosen), ("rejected", rejected)):
            ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
            if bos is not None and (not ids or ids[0] != int(bos)):
                ids = [int(bos), *ids]
            if not ids or ids[-1] != eos:
                ids = [*ids, eos]
            record(f"cpo_{side}_sequence_tokens", len(ids), pair_id)

    max_sequence = max(
        maxima[name][0]
        for name in (
            "dpo_chosen_sequence_tokens",
            "dpo_rejected_sequence_tokens",
            "cpo_chosen_sequence_tokens",
            "cpo_rejected_sequence_tokens",
        )
    )
    max_allowed = int(parent_contract["max_seq_length"])
    if max_sequence > max_allowed:
        raise FinalizationError(
            f"full-response sequence exceeds model contract: {max_sequence} > {max_allowed}"
        )
    return {
        "trl_version": "0.24.0",
        "tokenizer_name": tokenizer_name,
        "tokenizer_resolved_revision": tokenizer_revision,
        "tokenizer_vocab_sha256": parent_contract["tokenizer_vocab_sha256"],
        "tokenizer_backend_core_sha256": parent_contract[
            "tokenizer_backend_core_sha256"
        ],
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": eos,
        "max_seq_length": max_allowed,
        "max_observed_sequence_tokens": max_sequence,
        "maxima": {
            name: {"tokens": value, "pair_id": pair_id}
            for name, (value, pair_id) in sorted(maxima.items())
        },
        "truncation_allowed": False,
    }


def build_cpo_tokenized_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Tokenize full Teacher/Student completions without truncation or repair."""

    os.environ.setdefault("HF_HOME", str((ROOT / ".cache" / "huggingface").resolve()))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(parent_contract["tokenizer_name"]),
        revision=str(parent_contract["tokenizer_resolved_revision"]),
        local_files_only=True,
    )
    eos = int(tokenizer.eos_token_id)
    max_length = int(parent_contract["max_seq_length"])
    output: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        prompt = str(row["prompt"])
        prompt_ids = [
            int(value)
            for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ]
        sides: dict[str, dict[str, Any]] = {}
        for side in ("chosen", "rejected"):
            completion = str(row[side])
            completion_ids = [
                int(value)
                for value in tokenizer(completion, add_special_tokens=False)[
                    "input_ids"
                ]
            ]
            joint_ids = [
                int(value)
                for value in tokenizer(
                    prompt + completion, add_special_tokens=False
                )["input_ids"]
            ]
            if joint_ids != prompt_ids + completion_ids:
                raise FinalizationError(
                    f"{pair_id}: prompt/{side} tokenizer boundary is not additive"
                )
            if not completion_ids or completion_ids[-1] != eos:
                completion_ids.append(eos)
            input_ids = prompt_ids + completion_ids
            if len(input_ids) > max_length:
                raise FinalizationError(
                    f"{pair_id}: {side} sequence exceeds max length without truncation"
                )
            completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            token_indices = list(range(len(prompt_ids), len(input_ids)))
            if not token_indices or token_indices[0] <= 0:
                raise FinalizationError(f"{pair_id}: {side} has no causal completion mask")
            sides[side] = {
                f"{side}_input_ids": input_ids,
                f"{side}_completion_mask": completion_mask,
                # The shared mPO collator/loader calls this a term mask.  For
                # full-response CPO it deliberately equals the full completion.
                f"{side}_term_mask": completion_mask,
                f"{side}_completion_token_count": len(completion_ids),
                f"{side}_term_token_count": len(completion_ids),
                f"{side}_term_token_indices": token_indices,
                f"{side}_term_prediction_indices": [index - 1 for index in token_indices],
                f"{side}_sequence_token_count": len(input_ids),
            }
        output.append(
            {
                "schema_version": CPO_TOKEN_SCHEMA,
                "pair_id": pair_id,
                "subset": row["subset"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_token_count": len(prompt_ids),
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                **sides["chosen"],
                **sides["rejected"],
                "term_token_lengths_differ": (
                    sides["chosen"]["chosen_term_token_count"]
                    != sides["rejected"]["rejected_term_token_count"]
                ),
                "mask_semantics": "full_completion_including_eos",
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--model-decisions", type=Path, default=DEFAULT_MODEL_DECISIONS)
    parser.add_argument("--codex-decisions", type=Path, default=DEFAULT_CODEX_DECISIONS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--tokenized", type=Path, default=DEFAULT_TOKENIZED)
    parser.add_argument("--parent-contract", type=Path, default=DEFAULT_PARENT_CONTRACT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--final-decisions", type=Path, default=DEFAULT_FINAL_DECISIONS)
    parser.add_argument("--human-ledger", type=Path, default=DEFAULT_HUMAN_LEDGER)
    parser.add_argument("--final-candidates", type=Path, default=DEFAULT_FINAL_CANDIDATES)
    parser.add_argument("--final-tokenized", type=Path, default=DEFAULT_FINAL_TOKENIZED)
    parser.add_argument("--rejections", type=Path, default=DEFAULT_REJECTIONS)
    parser.add_argument(
        "--preference-rejections",
        type=Path,
        default=DEFAULT_PREFERENCE_REJECTIONS,
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--mpo-contract", type=Path, default=DEFAULT_MPO_CONTRACT)
    parser.add_argument("--full-pairs", type=Path, default=DEFAULT_FULL_PAIRS)
    parser.add_argument(
        "--full-pair-rejections", type=Path, default=DEFAULT_FULL_PAIR_REJECTIONS
    )
    parser.add_argument("--full-pair-contract", type=Path, default=DEFAULT_FULL_PAIR_CONTRACT)
    parser.add_argument("--full-pair-summary", type=Path, default=DEFAULT_FULL_PAIR_SUMMARY)
    parser.add_argument("--cpo-tokenized", type=Path, default=DEFAULT_CPO_TOKENIZED)
    parser.add_argument("--cpo-contract", type=Path, default=DEFAULT_CPO_CONTRACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_files = (
        args.requests,
        args.model_decisions,
        args.codex_decisions,
        args.candidates,
        args.tokenized,
        args.parent_contract,
    )
    for path in required_files:
        if not path.is_file():
            raise FinalizationError(f"missing required input: {path}")

    requests = read_jsonl(args.requests)
    final_decisions, human_ledger = finalize_decisions(
        requests,
        read_jsonl(args.model_decisions),
        read_jsonl(args.codex_decisions),
    )
    candidates = unique_by_id(read_jsonl(args.candidates), key="pair_id", label="candidates")
    tokenized = unique_by_id(read_jsonl(args.tokenized), key="pair_id", label="tokenized")
    raw_golden = load_raw_golden(args.raw_dir)
    request_ids = [str(row["pair_id"]) for row in requests]
    if set(request_ids) != set(tokenized):
        raise FinalizationError("request IDs do not exactly equal strict-v2 tokenized IDs")
    missing_candidates = sorted(set(request_ids) - set(candidates))
    if missing_candidates:
        raise FinalizationError(f"candidates missing request IDs: {missing_candidates[:5]}")

    decision_by_id = unique_by_id(final_decisions, key="pair_id", label="final decisions")
    source_accepted_ids = [
        pair_id for pair_id in request_ids if decision_by_id[pair_id]["decision"] == "KEEP"
    ]
    rejected_ids = [
        pair_id for pair_id in request_ids if decision_by_id[pair_id]["decision"] == "REJECT"
    ]
    if len(source_accepted_ids) != EXPECTED_FINAL_DECISIONS["KEEP"]:
        raise FinalizationError("accepted row count drift")
    unknown_preference_exclusions = sorted(
        set(PREFERENCE_EXCLUSIONS) - set(source_accepted_ids)
    )
    if unknown_preference_exclusions:
        raise FinalizationError(
            "preference exclusions are not source-quality KEEP rows: "
            f"{unknown_preference_exclusions}"
        )

    final_candidates: list[dict[str, Any]] = []
    final_tokenized: list[dict[str, Any]] = []
    preference_rejections: list[dict[str, Any]] = []
    full_pairs: list[dict[str, Any]] = []
    full_pair_rejections: list[dict[str, Any]] = []
    for pair_id in source_accepted_ids:
        candidate = candidates[pair_id]
        raw = raw_golden.get(pair_id)
        if raw is None:
            raise FinalizationError(f"missing raw golden row: {pair_id}")
        validate_candidate_against_raw(candidate, raw)
        if pair_id in PREFERENCE_EXCLUSIONS:
            if responses_are_distinct(candidate):
                raise FinalizationError(
                    f"{pair_id}: pinned identical-response exclusion no longer matches raw data"
                )
            reason_code, explanation = PREFERENCE_EXCLUSIONS[pair_id]
            preference_rejections.append(
                {
                    "schema_version": PREFERENCE_REJECTION_SCHEMA,
                    "pair_id": pair_id,
                    "subset": candidate["subset"],
                    "row_id": candidate["row_id"],
                    "reason_code": reason_code,
                    "explanation": explanation,
                    "source_quality_decision": "KEEP",
                    "chosen": candidate["chosen"],
                    "synthetic_rejected": candidate["rejected"],
                    "student_translation": candidate["student_translation"],
                    "term_mappings": candidate.get("term_mappings", []),
                }
            )
            full_pair_rejections.append(
                {
                    "schema_version": FULL_PAIR_SCHEMA,
                    "pair_id": pair_id,
                    "subset": candidate["subset"],
                    "row_id": candidate["row_id"],
                    "reason_code": "teacher_student_identical",
                    "explanation": (
                        "The raw Teacher target and raw Student translation are NFC-identical, "
                        "so this row has no full-response preference signal."
                    ),
                    "chosen": candidate["chosen"],
                    "rejected": candidate["student_translation"],
                }
            )
            continue
        final_candidates.append(candidate)
        final_tokenized.append(tokenized[pair_id])
        if not responses_are_distinct(candidate):
            raise FinalizationError(
                f"{pair_id}: unpinned identical Teacher/Student preference row"
            )
        full_pairs.append(build_full_pair(candidate))

    if (
        len(final_tokenized) != EXPECTED_MPO_ROWS
        or len(preference_rejections) != EXPECTED_PREFERENCE_REJECTIONS
    ):
        raise FinalizationError(
            "mPO preference-quality count drift: "
            f"accepted={len(final_tokenized)} rejected={len(preference_rejections)}"
        )
    if (
        len(full_pairs) != EXPECTED_FULL_PAIR_ROWS
        or len(full_pair_rejections) != EXPECTED_PREFERENCE_REJECTIONS
    ):
        raise FinalizationError(
            "full-response pair count drift: "
            f"accepted={len(full_pairs)} identical={len(full_pair_rejections)}"
        )

    rejection_rows: list[dict[str, Any]] = []
    for pair_id in rejected_ids:
        candidate = candidates[pair_id]
        decision = decision_by_id[pair_id]
        rejection_rows.append(
            {
                "schema_version": SOURCE_DECISION_SCHEMA,
                "pair_id": pair_id,
                "subset": candidate.get("subset"),
                "row_id": candidate.get("row_id"),
                "source": candidate.get("source"),
                "source_sha256": decision["source_sha256"],
                "decision": decision["decision"],
                "reason_code": decision["reason_code"],
                "evidence": decision["evidence"],
                "explanation": decision["explanation"],
                "decision_source": decision["decision_source"],
                "original_decision": decision["original_decision"],
            }
        )

    atomic_write_jsonl(args.final_decisions, final_decisions)
    atomic_write_jsonl(args.human_ledger, human_ledger)
    atomic_write_jsonl(args.final_candidates, final_candidates)
    atomic_write_jsonl(args.final_tokenized, final_tokenized)
    atomic_write_jsonl(args.rejections, rejection_rows)
    atomic_write_jsonl(args.preference_rejections, preference_rejections)
    atomic_write_jsonl(args.full_pairs, full_pairs)
    atomic_write_jsonl(args.full_pair_rejections, full_pair_rejections)

    output_pair_ids = [str(row["pair_id"]) for row in final_candidates]
    output_ids_sha = sha256_lines(output_pair_ids)
    parent_contract = read_json(args.parent_contract)
    mpo_contract = dict(parent_contract)
    mpo_contract.update(
        {
            "artifact_sha256": sha256_file(args.final_tokenized),
            "training_semantic_sha256": semantic_tokenized_sha(final_tokenized),
            "row_count": len(final_tokenized),
            "parent_contract_path": str(args.parent_contract.resolve()),
            "parent_contract_sha256": sha256_file(args.parent_contract),
            "parent_artifact_sha256": parent_contract["artifact_sha256"],
            "finalization_policy": FINALIZATION_POLICY,
            "finalizer_source_sha256": sha256_file(Path(__file__)),
            "source_quality_input_rows": len(requests),
            "source_quality_decision_counts": EXPECTED_FINAL_DECISIONS,
            "source_quality_final_decisions_sha256": sha256_file(args.final_decisions),
            "source_quality_human_adjudications_sha256": sha256_file(args.human_ledger),
            "source_quality_rejections_sha256": sha256_file(args.rejections),
            "source_quality_keep_rows": len(source_accepted_ids),
            "preference_quality_exclusion_count": len(preference_rejections),
            "preference_quality_rejections_path": str(
                args.preference_rejections.resolve()
            ),
            "preference_quality_rejections_sha256": sha256_file(
                args.preference_rejections
            ),
            "ordered_pair_ids_sha256": output_ids_sha,
            "filtered_candidates_sha256": sha256_file(args.final_candidates),
            "hard_source_filter_invariants": {
                "all_5505_sources_have_one_final_decision": True,
                "no_review_decisions_remain": True,
                "only_final_keep_rows_retained": True,
                "known_preference_inconsistencies_excluded": True,
                "training_rows_equal_source_keep_minus_preference_exclusions": True,
                "repair_or_fallback": "none",
                "candidate_and_tokenized_pair_ids_equal": True,
                "input_order_preserved": True,
            },
        }
    )
    atomic_write_json(args.mpo_contract, mpo_contract)

    cpo_tokenized = build_cpo_tokenized_rows(
        full_pairs,
        parent_contract=parent_contract,
    )
    atomic_write_jsonl(args.cpo_tokenized, cpo_tokenized)
    cpo_contract = dict(mpo_contract)
    cpo_contract.update(
        {
            "artifact_sha256": sha256_file(args.cpo_tokenized),
            "training_semantic_sha256": semantic_tokenized_sha(cpo_tokenized),
            "row_count": len(cpo_tokenized),
            "schema_version": CPO_TOKEN_SCHEMA,
            "objective_family": "CPO_full_response",
            "full_response_pair_artifact_sha256": sha256_file(args.full_pairs),
            "full_response_pair_contract_path": str(
                args.full_pair_contract.resolve()
            ),
            "full_response_negative_policy": (
                "original_full_student_response_no_synthetic_reversion"
            ),
            "mask_semantics": {
                "chosen_completion_mask": "all Teacher completion tokens including EOS",
                "rejected_completion_mask": "all Student completion tokens including EOS",
                "chosen_term_mask": "alias of chosen completion mask for shared loader",
                "rejected_term_mask": "alias of rejected completion mask for shared loader",
            },
            "ordered_pair_ids_sha256": sha256_lines(
                [str(row["pair_id"]) for row in cpo_tokenized]
            ),
            "repair_or_fallback": "none",
        }
    )
    atomic_write_json(args.cpo_contract, cpo_contract)

    summary = {
        "schema_version": SOURCE_DECISION_SCHEMA,
        "finalization_policy": FINALIZATION_POLICY,
        "input_rows": len(requests),
        "input_decision_counts": EXPECTED_INPUT_DECISIONS,
        "human_review_rows": len(human_ledger),
        "human_review_promoted_to_keep": len(PROMOTED_REVIEW_RATIONALES),
        "human_review_rejected": len(human_ledger) - len(PROMOTED_REVIEW_RATIONALES),
        "final_decision_counts": EXPECTED_FINAL_DECISIONS,
        "source_quality_keep_rows": len(source_accepted_ids),
        "preference_quality_excluded_rows": len(preference_rejections),
        "output_rows": len(final_tokenized),
        "ordered_pair_ids_sha256": output_ids_sha,
        "outputs": {
            "final_decisions": str(args.final_decisions.resolve()),
            "human_adjudications": str(args.human_ledger.resolve()),
            "mPO_candidates": str(args.final_candidates.resolve()),
            "mPO_tokenized": str(args.final_tokenized.resolve()),
            "full_response_dpo_cpo_pairs": str(args.full_pairs.resolve()),
            "full_response_dpo_cpo_rejections": str(args.full_pair_rejections.resolve()),
            "full_response_cpo_tokenized": str(args.cpo_tokenized.resolve()),
            "source_rejections": str(args.rejections.resolve()),
            "preference_rejections": str(args.preference_rejections.resolve()),
        },
    }
    atomic_write_json(args.summary, summary)

    full_pair_ids = [str(row["pair_id"]) for row in full_pairs]
    full_pair_tokenization = full_pair_tokenization_contract(
        full_pairs,
        parent_contract=parent_contract,
    )
    full_pair_contract = {
        "schema_version": "dqs.full_response_preference_contract.v1",
        "artifact_schema_version": FULL_PAIR_SCHEMA,
        "artifact_path": str(args.full_pairs.resolve()),
        "artifact_sha256": sha256_file(args.full_pairs),
        "row_count": len(full_pairs),
        "ordered_pair_ids_sha256": sha256_lines(full_pair_ids),
        "source_quality_contract_path": str(args.mpo_contract.resolve()),
        "source_quality_contract_sha256": sha256_file(args.mpo_contract),
        "parent_candidate_sha256": sha256_file(args.candidates),
        "raw_golden_pair_provenance": {
            "chosen": "target",
            "rejected": "student_translation",
            "byte_identity_checked_for_every_row": True,
        },
        "prompt_contract": {
            "format": "already_serialized_chat_template_ending_at_model_prefix",
            "apply_chat_template_again": False,
            "chosen_and_rejected_are_completion_only": True,
        },
        "tokenization_contract": full_pair_tokenization,
        "cpo_tokenized_artifact_path": str(args.cpo_tokenized.resolve()),
        "cpo_tokenized_artifact_sha256": sha256_file(args.cpo_tokenized),
        "cpo_tokenized_contract_path": str(args.cpo_contract.resolve()),
        "cpo_tokenized_contract_sha256": sha256_file(args.cpo_contract),
        "compatible_objectives": ["DPO", "CPO"],
        "negative_policy": "original_full_student_response_no_synthetic_reversion",
        "excluded_identical_response_rows": len(full_pair_rejections),
        "rejection_ledger_path": str(args.full_pair_rejections.resolve()),
        "rejection_ledger_sha256": sha256_file(args.full_pair_rejections),
        "repair_or_fallback": "none",
        "hard_invariants": {
            "pair_ids_are_final_mpo_keep_subset": True,
            "pair_ids_equal_final_mpo_rows": True,
            "only_nfc_identical_teacher_student_rows_are_excluded": True,
            "teacher_and_student_nonempty": True,
            "teacher_and_student_nfc_distinct": True,
            "teacher_equals_raw_target_byte_for_byte": True,
            "student_equals_raw_student_translation_byte_for_byte": True,
            "input_order_preserved": True,
        },
    }
    atomic_write_json(args.full_pair_contract, full_pair_contract)
    full_pair_summary = {
        "schema_version": FULL_PAIR_SCHEMA,
        "rows": len(full_pairs),
        "excluded_identical_response_rows": len(full_pair_rejections),
        "chosen_source": "raw golden_pairs target (Teacher post-edit)",
        "rejected_source": "raw golden_pairs student_translation (Student full output)",
        "prompt_format": "pre-serialized; do not apply a second chat template",
        "max_observed_sequence_tokens": full_pair_tokenization[
            "max_observed_sequence_tokens"
        ],
        "artifact_sha256": sha256_file(args.full_pairs),
        "contract_sha256": sha256_file(args.full_pair_contract),
    }
    atomic_write_json(args.full_pair_summary, full_pair_summary)

    print(
        json.dumps(
            {
                "status": "finalized",
                "source_decisions": EXPECTED_FINAL_DECISIONS,
                "source_quality_keep_rows": len(source_accepted_ids),
                "preference_quality_rows_excluded": len(preference_rejections),
                "human_reviews": len(human_ledger),
                "human_review_promotions": len(PROMOTED_REVIEW_RATIONALES),
                "mpo_rows": len(final_tokenized),
                "dpo_cpo_rows": len(full_pairs),
                "cpo_tokenized_rows": len(cpo_tokenized),
                "dpo_cpo_identical_rows_excluded": len(full_pair_rejections),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
