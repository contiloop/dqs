"""Load and validate explicitly staged local tokenized mPO pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRAIN_COLUMNS = (
    "pair_id",
    "chosen_input_ids",
    "chosen_completion_mask",
    "chosen_term_mask",
    "rejected_input_ids",
    "rejected_completion_mask",
    "rejected_term_mask",
)

REQUIRED_CONTRACT_FIELDS = (
    "artifact_sha256",
    "training_semantic_sha256",
    "row_count",
    "schema_version",
    "max_seq_length",
    "pad_token_id",
    "eos_token_id",
    "tokenizer_name",
    "tokenizer_resolved_revision",
    "tokenizer_vocab_sha256",
    "tokenizer_backend_core_sha256",
)


@dataclass
class PreferenceDatasetBundle:
    train_dataset: Any
    eval_dataset: Any | None
    summary: dict[str, Any]
    contract: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(raw: str | Path, *, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_contract(data_cfg: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    source = str(data_cfg.get("source", "")).strip().lower()
    if source != "local":
        raise ValueError("data.source must be local; run `make download-data` first")
    raw = data_cfg.get("contract_path")
    if not raw:
        raise ValueError("data.contract_path is required for local data")
    path = _resolve_path(str(raw), repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(f"missing local dataset contract: {path}")
    contract = _load_json(path)

    missing = [field for field in REQUIRED_CONTRACT_FIELDS if not contract.get(field) and contract.get(field) != 0]
    if missing:
        raise ValueError(f"dataset contract is missing required fields: {missing}")
    return contract


def _load_split(
    data_cfg: Mapping[str, Any],
    *,
    split: str,
    repo_root: Path,
    cache_dir: Path,
) -> Any:
    from datasets import load_dataset

    source = str(data_cfg.get("source", "")).strip().lower()
    if source != "local":
        raise ValueError("data.source must be local; run `make download-data` first")
    key = "path" if split == "train" else "eval_path"
    raw_path = data_cfg.get(key)
    if not raw_path:
        return None
    path = _resolve_path(str(raw_path), repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {split} dataset: {path}; run `make download-data` first"
        )
    return load_dataset(
        "json",
        data_files={split: str(path)},
        split=split,
        cache_dir=str(cache_dir),
    )


def _as_int_list(values: Any, *, field: str, pair_id: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{pair_id}: {field} must be a sequence")
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{pair_id}: {field} contains non-integer values") from exc


def _validate_side(
    row: Mapping[str, Any],
    *,
    side: str,
    pair_id: str,
    max_seq_length: int,
) -> tuple[int, int, int, int]:
    input_ids = _as_int_list(row.get(f"{side}_input_ids"), field=f"{side}_input_ids", pair_id=pair_id)
    completion = _as_int_list(
        row.get(f"{side}_completion_mask"),
        field=f"{side}_completion_mask",
        pair_id=pair_id,
    )
    term = _as_int_list(row.get(f"{side}_term_mask"), field=f"{side}_term_mask", pair_id=pair_id)
    length = len(input_ids)
    if length < 2 or length > max_seq_length:
        raise ValueError(f"{pair_id}: {side} sequence length {length} outside [2, {max_seq_length}]")
    if len(completion) != length or len(term) != length:
        raise ValueError(f"{pair_id}: {side} masks do not match input length")
    if any(value not in (0, 1) for value in completion + term):
        raise ValueError(f"{pair_id}: {side} masks must be binary")
    if any(term_value and not completion_value for term_value, completion_value in zip(term, completion)):
        raise ValueError(f"{pair_id}: {side} term mask is not a subset of completion mask")
    if sum(completion[1:]) == 0 or sum(term[1:]) == 0:
        raise ValueError(f"{pair_id}: {side} has an empty mask after causal shift")

    prompt_tokens = int(row.get("prompt_token_count", 0) or 0)
    if prompt_tokens < 1 or prompt_tokens >= length:
        raise ValueError(f"{pair_id}: invalid prompt_token_count={prompt_tokens}")
    if any(completion[:prompt_tokens]) or any(term[:prompt_tokens]):
        raise ValueError(f"{pair_id}: {side} prompt tokens enter a loss mask")
    expected_completion = [0] * prompt_tokens + [1] * (length - prompt_tokens)
    if completion != expected_completion:
        raise ValueError(f"{pair_id}: {side} completion mask is not prompt-zero/completion-one")

    expected_prediction_indices = [index - 1 for index, enabled in enumerate(term) if enabled]
    stored_prediction_indices = row.get(f"{side}_term_prediction_indices")
    if stored_prediction_indices is not None and [int(value) for value in stored_prediction_indices] != expected_prediction_indices:
        raise ValueError(f"{pair_id}: {side} stored term prediction indices are inconsistent")
    return length, sum(completion), sum(term), max(input_ids)


def validate_dataset(
    dataset: Any,
    *,
    split_name: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if dataset is None:
        return {"split": split_name, "rows": 0}
    columns = set(getattr(dataset, "column_names", []))
    required = set(TRAIN_COLUMNS) | {"schema_version", "prompt_token_count"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{split_name}: dataset is missing required columns: {missing}")

    expected_schema = str(contract.get("schema_version", "dqs_mpo_token_masks_v1"))
    max_seq_length = int(contract.get("max_seq_length", 2908))
    pair_ids: set[str] = set()
    max_token_id = -1
    chosen_max = 0
    rejected_max = 0
    chosen_term_tokens = 0
    rejected_term_tokens = 0
    different_term_lengths = 0
    semantic_digest = hashlib.sha256()
    for row_index, row in enumerate(dataset):
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError(f"{split_name}[{row_index}]: empty pair_id")
        if pair_id in pair_ids:
            raise ValueError(f"{split_name}: duplicate pair_id={pair_id}")
        pair_ids.add(pair_id)
        if str(row.get("schema_version")) != expected_schema:
            raise ValueError(
                f"{pair_id}: schema={row.get('schema_version')!r}, expected={expected_schema!r}"
            )
        chosen = _validate_side(
            row,
            side="chosen",
            pair_id=pair_id,
            max_seq_length=max_seq_length,
        )
        rejected = _validate_side(
            row,
            side="rejected",
            pair_id=pair_id,
            max_seq_length=max_seq_length,
        )
        chosen_max = max(chosen_max, chosen[0])
        rejected_max = max(rejected_max, rejected[0])
        chosen_term_tokens += chosen[2]
        rejected_term_tokens += rejected[2]
        max_token_id = max(max_token_id, chosen[3], rejected[3])
        different_term_lengths += int(chosen[2] != rejected[2])
        semantic_row: dict[str, Any] = {"pair_id": pair_id}
        for field in TRAIN_COLUMNS[1:]:
            semantic_row[field] = _as_int_list(row.get(field), field=field, pair_id=pair_id)
        semantic_digest.update(
            json.dumps(
                semantic_row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        semantic_digest.update(b"\n")

    expected_rows = contract.get("row_count") if split_name == "train" else None
    if expected_rows is not None and len(dataset) != int(expected_rows):
        raise ValueError(f"{split_name}: row count={len(dataset)}, expected={expected_rows}")
    observed_semantic_sha = semantic_digest.hexdigest()
    expected_semantic_sha = contract.get("training_semantic_sha256") if split_name == "train" else None
    if expected_semantic_sha and observed_semantic_sha != str(expected_semantic_sha):
        raise ValueError(
            f"{split_name}: semantic checksum mismatch: "
            f"observed={observed_semantic_sha}, expected={expected_semantic_sha}"
        )
    return {
        "split": split_name,
        "rows": len(dataset),
        "schema_version": expected_schema,
        "max_seq_length_contract": max_seq_length,
        "chosen_max_sequence_length": chosen_max,
        "rejected_max_sequence_length": rejected_max,
        "chosen_term_tokens": chosen_term_tokens,
        "rejected_term_tokens": rejected_term_tokens,
        "rows_with_different_term_token_counts": different_term_lengths,
        "max_token_id": max_token_id,
        "training_semantic_sha256": observed_semantic_sha,
    }


def load_preference_datasets(
    data_cfg: Mapping[str, Any],
    *,
    repo_root: Path,
) -> PreferenceDatasetBundle:
    cache_raw = data_cfg.get("cache_dir")
    if not cache_raw:
        raise ValueError("data.cache_dir must be configured explicitly")
    cache_dir = _resolve_path(str(cache_raw), repo_root=repo_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract = _load_contract(data_cfg, repo_root=repo_root)
    train_dataset = _load_split(
        data_cfg,
        split="train",
        repo_root=repo_root,
        cache_dir=cache_dir,
    )
    if train_dataset is None:
        raise ValueError("training dataset is not configured")
    eval_dataset = _load_split(
        data_cfg,
        split="eval",
        repo_root=repo_root,
        cache_dir=cache_dir,
    )

    source = str(data_cfg.get("source", "")).strip().lower()
    artifact_path = _resolve_path(str(data_cfg["path"]), repo_root=repo_root)
    observed = sha256(artifact_path)
    if observed != str(contract["artifact_sha256"]):
        raise ValueError(
            "tokenized artifact checksum mismatch: "
            f"observed={observed}, expected={contract['artifact_sha256']}"
        )

    train_summary = validate_dataset(train_dataset, split_name="train", contract=contract)
    eval_summary = (
        validate_dataset(eval_dataset, split_name="eval", contract=contract)
        if eval_dataset is not None
        else None
    )
    # Drop audit-heavy strings and span structures before Trainer row access.
    train_dataset = train_dataset.select_columns(list(TRAIN_COLUMNS))
    if eval_dataset is not None:
        eval_dataset = eval_dataset.select_columns(list(TRAIN_COLUMNS))
    return PreferenceDatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        summary={
            "source": source,
            "train": train_summary,
            "eval": eval_summary,
            "cache_dir": str(cache_dir),
        },
        contract=contract,
    )
