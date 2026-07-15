#!/usr/bin/env python3
"""Build a pinned training contract for a validated tokenized mPO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

RUNTIME_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

try:
    from .build_preference_pairs import file_sha256
    from .mpo_data import TRAIN_COLUMNS
except ImportError:
    from build_preference_pairs import file_sha256
    from mpo_data import TRAIN_COLUMNS


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenized",
        type=Path,
        default=root / "prepared" / "mpo_tokenized_pairs_strict_v2.jsonl",
    )
    parser.add_argument(
        "--token-summary",
        type=Path,
        default=root / "analysis" / "strict_v2_mpo_token_mask_summary.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "prepared" / "preference_candidates_strict_v2.jsonl",
    )
    parser.add_argument(
        "--synthesis-summary",
        type=Path,
        default=root / "analysis" / "strict_v2_synthesis_summary.json",
    )
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=root / "dataset_contract.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset_contract_strict_v2.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def semantic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = str(row.get("pair_id", ""))
    if not pair_id:
        raise ValueError("Empty pair_id in tokenized artifact")
    result: dict[str, Any] = {"pair_id": pair_id}
    for field in TRAIN_COLUMNS[1:]:
        value = row.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{pair_id}: missing training tensor field {field}")
        result[field] = [int(item) for item in value]
    return result


def scan_tokenized(path: Path, expected_schema: str) -> tuple[int, str]:
    count = 0
    ids: set[str] = set()
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            if str(row.get("schema_version")) != expected_schema:
                raise ValueError(f"Schema mismatch at {path}:{line_number}")
            tensors = semantic_row(row)
            pair_id = str(tensors["pair_id"])
            if pair_id in ids:
                raise ValueError(f"Duplicate pair_id in tokenized artifact: {pair_id}")
            ids.add(pair_id)
            digest.update(
                json.dumps(
                    tensors,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def main() -> None:
    args = parse_args()
    token_summary = load_json(args.token_summary)
    synthesis_summary = load_json(args.synthesis_summary)
    base_contract = load_json(args.base_contract)

    tokenized_sha = file_sha256(args.tokenized)
    candidate_sha = file_sha256(args.candidates)
    if tokenized_sha != token_summary["outputs"]["tokenized_sha256"]:
        raise ValueError("Tokenized artifact does not match token-mask summary")
    if candidate_sha != token_summary["input"]["candidate_sha256"]:
        raise ValueError("Candidate artifact does not match token-mask summary")
    if candidate_sha != synthesis_summary["outputs"]["candidate_sha256"]:
        raise ValueError("Candidate artifact does not match synthesis summary")

    tokenizer = token_summary["tokenizer"]
    tokenizer_checks = {
        "tokenizer_name": "name",
        "tokenizer_requested_revision": "requested_revision",
        "tokenizer_resolved_revision": "resolved_revision",
        "pad_token_id": "pad_token_id",
        "eos_token_id": "eos_token_id",
    }
    for contract_field, summary_field in tokenizer_checks.items():
        if base_contract.get(contract_field) != tokenizer.get(summary_field):
            raise ValueError(
                f"Tokenizer mismatch for {contract_field}: "
                f"{base_contract.get(contract_field)!r} != {tokenizer.get(summary_field)!r}"
            )

    schema_version = str(token_summary["schema_version"])
    row_count, semantic_sha = scan_tokenized(args.tokenized, schema_version)
    if row_count != int(token_summary["counts"]["accepted_rows"]):
        raise ValueError("Tokenized row count does not match token-mask summary")
    if int(token_summary["counts"]["input_rows"]) != int(
        synthesis_summary["counts"]["rows_accepted"]
    ):
        raise ValueError("Synthesis and tokenization input counts do not match")

    contract = {
        "artifact_sha256": tokenized_sha,
        "training_semantic_sha256": semantic_sha,
        "row_count": row_count,
        "schema_version": schema_version,
        "max_seq_length": int(token_summary["training_contract"]["max_seq_length"]),
        "pad_token_id": int(tokenizer["pad_token_id"]),
        "eos_token_id": int(tokenizer["eos_token_id"]),
        "tokenizer_name": str(tokenizer["name"]),
        "tokenizer_requested_revision": str(tokenizer["requested_revision"]),
        "tokenizer_resolved_revision": str(tokenizer["resolved_revision"]),
        "tokenizer_vocab_sha256": str(base_contract["tokenizer_vocab_sha256"]),
        "tokenizer_backend_core_sha256": str(
            base_contract["tokenizer_backend_core_sha256"]
        ),
        "source_hf_repo": str(synthesis_summary["input"]["hf_repo"]),
        "source_hf_revision": str(synthesis_summary["input"]["hf_repo_revision"]),
        "source_run": str(synthesis_summary["input"]["hf_run"]),
        "source_subset_count": int(synthesis_summary["input"]["file_count"]),
        "raw_input_manifest_sha256": str(
            synthesis_summary["input"]["manifest_sha256"]
        ),
        "synthesis_schema_version": str(synthesis_summary["schema_version"]),
        "synthesis_version": str(synthesis_summary["synthesis_version"]),
        "synthesis_builder_source_sha256": str(
            synthesis_summary["implementation"]["strict_builder_sha256"]
        ),
        "base_mapping_builder_source_sha256": str(
            synthesis_summary["implementation"]["base_mapping_builder_sha256"]
        ),
        "synthesis_candidate_sha256": candidate_sha,
        "synthesis_rejection_ledger_sha256": str(
            synthesis_summary["outputs"]["rejections_sha256"]
        ),
        "synthesis_summary_sha256": file_sha256(args.synthesis_summary),
        "token_mask_rejection_ledger_sha256": str(
            token_summary["outputs"]["rejections_sha256"]
        ),
        "token_mask_summary_sha256": file_sha256(args.token_summary),
        "token_mask_builder_source_sha256": file_sha256(
            Path(__file__).with_name("build_mpo_token_masks.py")
        ),
        "base_tokenizer_contract_sha256": file_sha256(args.base_contract),
        "strict_row_atomicity": "all_terminology_annotations_or_reject_row",
        "repair_or_fallback": "none",
        "chosen_rejected_masks": "independent_offset_alignment",
        "causal_shift": "mask[:, 1:] aligns with logits[:, :-1]",
        "loss_normalization": "each row and each mask normalized by its own token count",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
