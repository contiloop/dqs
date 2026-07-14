"""Fail-closed loader for full-response Teacher-vs-Student preference pairs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_CONTRACT_FIELDS = (
    "artifact_sha256",
    "artifact_schema_version",
    "row_count",
    "ordered_pair_ids_sha256",
    "negative_policy",
    "tokenization_contract",
)


@dataclass
class FullPreferenceBundle:
    train_dataset: Any
    summary: dict[str, Any]
    contract: dict[str, Any]


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


def _resolve_path(raw: str | Path, *, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_inputs(
    data_cfg: Mapping[str, Any], *, repo_root: Path
) -> tuple[Path, dict[str, Any], Path]:
    cache_raw = data_cfg.get("cache_dir")
    if not cache_raw:
        raise ValueError("data.cache_dir must be configured explicitly")
    cache_dir = _resolve_path(str(cache_raw), repo_root=repo_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = str(data_cfg.get("source", "")).strip().lower()
    if source != "local":
        raise ValueError("data.source must be local; run `make download-data` first")
    path_raw = data_cfg.get("path")
    contract_raw = data_cfg.get("contract_path")
    if not path_raw or not contract_raw:
        raise ValueError("local data requires data.path and data.contract_path")
    artifact_path = _resolve_path(str(path_raw), repo_root=repo_root)
    contract_path = _resolve_path(str(contract_raw), repo_root=repo_root)
    if not artifact_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError(
            f"missing preference artifact or contract: {artifact_path}, {contract_path}"
        )
    contract = _load_json(contract_path)
    missing = [
        key
        for key in REQUIRED_CONTRACT_FIELDS
        if key not in contract or contract[key] in (None, "")
    ]
    if missing:
        raise ValueError(f"full preference contract missing fields: {missing}")
    return artifact_path, contract, cache_dir


def validate_rows(dataset: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    required_columns = {
        "schema_version",
        "pair_id",
        "prompt",
        "chosen",
        "rejected",
        "chosen_source",
        "rejected_source",
    }
    columns = set(getattr(dataset, "column_names", []))
    missing = sorted(required_columns - columns)
    if missing:
        raise ValueError(f"full preference dataset missing columns: {missing}")
    expected_schema = str(contract["artifact_schema_version"])
    pair_ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(dataset):
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id in seen:
            raise ValueError(f"invalid or duplicate pair_id at row {index}: {pair_id!r}")
        seen.add(pair_id)
        pair_ids.append(pair_id)
        if str(row.get("schema_version")) != expected_schema:
            raise ValueError(f"{pair_id}: schema mismatch")
        for field in ("prompt", "chosen", "rejected"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{pair_id}: {field} must be a non-empty string")
        if unicodedata.normalize("NFC", row["chosen"]) == unicodedata.normalize(
            "NFC", row["rejected"]
        ):
            raise ValueError(f"{pair_id}: chosen and rejected are NFC-identical")
        if row.get("chosen_source") != "teacher_post_edit_raw_target":
            raise ValueError(f"{pair_id}: chosen provenance mismatch")
        if row.get("rejected_source") != "student_translation_raw_output":
            raise ValueError(f"{pair_id}: rejected provenance mismatch")
    if len(pair_ids) != int(contract["row_count"]):
        raise ValueError(
            f"full preference row count={len(pair_ids)}, expected={contract['row_count']}"
        )
    observed_ids_sha = sha256_lines(pair_ids)
    if observed_ids_sha != str(contract["ordered_pair_ids_sha256"]):
        raise ValueError("full preference ordered pair ID checksum mismatch")
    return {
        "rows": len(pair_ids),
        "schema_version": expected_schema,
        "ordered_pair_ids_sha256": observed_ids_sha,
    }


def load_full_preference_dataset(
    data_cfg: Mapping[str, Any], *, repo_root: Path
) -> FullPreferenceBundle:
    artifact_path, contract, cache_dir = _load_inputs(data_cfg, repo_root=repo_root)
    observed_sha = sha256_file(artifact_path)
    if observed_sha != str(contract["artifact_sha256"]):
        raise ValueError(
            "full preference artifact checksum mismatch: "
            f"observed={observed_sha}, expected={contract['artifact_sha256']}"
        )
    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files={"train": str(artifact_path)},
        split="train",
        cache_dir=str(cache_dir),
    )
    summary = validate_rows(dataset, contract)
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
    return FullPreferenceBundle(
        train_dataset=dataset,
        summary={
            **summary,
            "source": str(data_cfg.get("source", "")).strip().lower(),
            "artifact_path": str(artifact_path),
            "artifact_sha256": observed_sha,
            "cache_dir": str(cache_dir),
        },
        contract=contract,
    )


def _runtime_tokenizer_sha(tokenizer: Any) -> tuple[str, str]:
    vocab_payload = json.dumps(
        tokenizer.get_vocab(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("full preference training requires a fast tokenizer")
    backend_payload = json.loads(backend.to_str())
    backend_payload.pop("padding", None)
    backend_payload.pop("truncation", None)
    backend_canonical = json.dumps(
        backend_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(vocab_payload).hexdigest(), hashlib.sha256(
        backend_canonical
    ).hexdigest()


def measure_tokenization(dataset: Any, tokenizer: Any) -> dict[str, Any]:
    maxima = {
        "prompt_tokens": 0,
        "dpo_chosen_completion_tokens": 0,
        "dpo_rejected_completion_tokens": 0,
        "dpo_chosen_sequence_tokens": 0,
        "dpo_rejected_sequence_tokens": 0,
        "cpo_chosen_sequence_tokens": 0,
        "cpo_rejected_sequence_tokens": 0,
    }
    eos = int(tokenizer.eos_token_id)
    bos = tokenizer.bos_token_id
    for row in dataset:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"] + [eos]
        rejected_ids = tokenizer(row["rejected"], add_special_tokens=False)["input_ids"] + [eos]
        maxima["prompt_tokens"] = max(maxima["prompt_tokens"], len(prompt_ids))
        maxima["dpo_chosen_completion_tokens"] = max(
            maxima["dpo_chosen_completion_tokens"], len(chosen_ids)
        )
        maxima["dpo_rejected_completion_tokens"] = max(
            maxima["dpo_rejected_completion_tokens"], len(rejected_ids)
        )
        maxima["dpo_chosen_sequence_tokens"] = max(
            maxima["dpo_chosen_sequence_tokens"], len(prompt_ids) + len(chosen_ids)
        )
        maxima["dpo_rejected_sequence_tokens"] = max(
            maxima["dpo_rejected_sequence_tokens"], len(prompt_ids) + len(rejected_ids)
        )
        for side in ("chosen", "rejected"):
            ids = tokenizer(
                row["prompt"] + row[side], add_special_tokens=False
            )["input_ids"]
            if bos is not None and (not ids or ids[0] != int(bos)):
                ids = [int(bos), *ids]
            if not ids or ids[-1] != eos:
                ids = [*ids, eos]
            key = f"cpo_{side}_sequence_tokens"
            maxima[key] = max(maxima[key], len(ids))
    return maxima


def validate_runtime_tokenization(
    bundle: FullPreferenceBundle,
    tokenizer: Any,
    *,
    objective: str,
    max_length: int,
    max_prompt_length: int,
    max_completion_length: int | None,
) -> dict[str, Any]:
    expected = bundle.contract["tokenization_contract"]
    if int(tokenizer.pad_token_id) != int(expected["pad_token_id"]):
        raise ValueError("runtime tokenizer pad_token_id mismatch")
    if int(tokenizer.eos_token_id) != int(expected["eos_token_id"]):
        raise ValueError("runtime tokenizer eos_token_id mismatch")
    vocab_sha, backend_sha = _runtime_tokenizer_sha(tokenizer)
    if vocab_sha != str(expected["tokenizer_vocab_sha256"]):
        raise ValueError("runtime tokenizer vocabulary checksum mismatch")
    if backend_sha != str(expected["tokenizer_backend_core_sha256"]):
        raise ValueError("runtime tokenizer backend checksum mismatch")
    observed = measure_tokenization(bundle.train_dataset, tokenizer)
    expected_maxima = {
        key: int(value["tokens"]) for key, value in expected["maxima"].items()
    }
    if observed != expected_maxima:
        raise ValueError(
            f"runtime full preference tokenization drift: {observed} != {expected_maxima}"
        )
    if max_length != int(expected["max_seq_length"]):
        raise ValueError("training max_length must equal the pinned model sequence contract")
    if observed["prompt_tokens"] > max_prompt_length:
        raise ValueError("configured max_prompt_length would truncate a prompt")
    if objective == "dpo":
        if max_completion_length is None:
            raise ValueError("DPO requires explicit max_completion_length")
        if max(
            observed["dpo_chosen_completion_tokens"],
            observed["dpo_rejected_completion_tokens"],
        ) > max_completion_length:
            raise ValueError("configured max_completion_length would truncate a response")
        max_observed = max(
            observed["dpo_chosen_sequence_tokens"],
            observed["dpo_rejected_sequence_tokens"],
        )
    elif objective == "cpo":
        max_observed = max(
            observed["cpo_chosen_sequence_tokens"],
            observed["cpo_rejected_sequence_tokens"],
        )
    else:
        raise ValueError(f"unsupported full preference objective: {objective}")
    if max_observed > max_length:
        raise ValueError("configured max_length would truncate a full response pair")
    return {"objective": objective, "maxima": observed, "max_observed": max_observed}
