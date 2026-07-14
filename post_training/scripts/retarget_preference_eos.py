#!/usr/bin/env python3
"""Retarget the released preference data to the exact final-SFT EOS token.

This is a fail-closed, one-way migration of the original EOS=1 release.  It
changes only the appended completion terminator in the pre-tokenized mPO/CPO
artifacts.  DPO text is copied byte-for-byte and only its runtime tokenization
contract changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SOURCE_EOS_ID = 1
SOURCE_EOS_TOKEN = "<eos>"
TARGET_EOS_ID = 106
TARGET_EOS_TOKEN = "<turn|>"
PAD_TOKEN_ID = 0
EXPECTED_ROWS = 5_200
EXPECTED_VOCAB_SHA256 = "d78b3ddf966363e3cd1f7242f4ac5590a3f1bac57c7be03b199dbccc85426e42"
EXPECTED_BACKEND_SHA256 = "1e31c483598073ccd8ff2bc1beaf77cb83fb923968487f05a1877a9f6441c627"
FINAL_TOKENIZER_CONFIG_SHA256 = (
    "de3cba60561eb2ee6362e34274bb7196b87d8febf01d54a2356a1b1f5a14284b"
)
FINAL_SFT_SOURCE = {
    "repo_id": "alwaysgood/dqs-runs",
    "repo_type": "dataset",
    "revision": "a58b1878988efcecc9a2644f8324bd00131864b5",
    "subfolder": (
        "gemma4_e2b_it_full_iter_lowqe_sf_on_seed42/checkpoints/final"
    ),
    "run_id": "gemma4_e2b_it_full_iter_lowqe_sf_on_seed42",
    "subset_idx": 22,
    "global_step": 184,
}
EXPECTED_INPUT_ARTIFACTS = {
    "mpo": "a85d052985e87316b3086acc1903ac0f74958a935094146a59a1e047ef8bb287",
    "cpo": "54c96f01ac416c90d24bb2ff757c0e8db5cde7161220a639051c984acda30d2f",
    "dpo": "4ff1fe26d35518b4c76ddc50f34ce48def8df73b0f9aec3f61ab97aba00e6187",
}
EXPECTED_INPUT_CONTRACTS = {
    "mpo": "184178b8b8f04cb6a580e1899aacc16cf4cc65d5d4c3f9e3c4489d6ea1dc8a46",
    "cpo": "6b8e0f2f029d0096bf5c58ef5ff28a24008cb333f88a0d2490add945483eb01e",
    "dpo": "e3e2e608724a29a64213f951de5ca1d1e5e777f2eb3c66be6605b005420aeb7b",
}
SEMANTIC_FIELDS = (
    "chosen_input_ids",
    "chosen_completion_mask",
    "chosen_term_mask",
    "rejected_input_ids",
    "rejected_completion_mask",
    "rejected_term_mask",
)


class RetargetError(ValueError):
    """Raised when any migration precondition is not exact."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-tokenizer-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetargetError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _tokenizer_hashes(tokenizer: Any) -> tuple[str, str]:
    vocab = json.dumps(
        tokenizer.get_vocab(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise RetargetError("final SFT tokenizer must be a fast tokenizer")
    backend_payload = json.loads(backend.to_str())
    backend_payload.pop("padding", None)
    backend_payload.pop("truncation", None)
    backend_core = json.dumps(
        backend_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(vocab).hexdigest(), hashlib.sha256(backend_core).hexdigest()


def validate_final_tokenizer(path: Path) -> dict[str, Any]:
    config_path = path / "tokenizer_config.json"
    if not config_path.is_file() or not (path / "tokenizer.json").is_file():
        raise RetargetError(f"incomplete final SFT tokenizer directory: {path}")
    config_sha = sha256_file(config_path)
    if config_sha != FINAL_TOKENIZER_CONFIG_SHA256:
        raise RetargetError(
            f"final tokenizer_config SHA256={config_sha}, "
            f"expected={FINAL_TOKENIZER_CONFIG_SHA256}"
        )
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RetargetError("transformers is required for tokenizer validation") from exc
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    observed = {
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "eos_token": str(tokenizer.eos_token),
        "source_token": str(tokenizer.convert_ids_to_tokens(SOURCE_EOS_ID)),
        "target_token": str(tokenizer.convert_ids_to_tokens(TARGET_EOS_ID)),
        "tokenizer_length": len(tokenizer),
    }
    expected = {
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": TARGET_EOS_ID,
        "eos_token": TARGET_EOS_TOKEN,
        "source_token": SOURCE_EOS_TOKEN,
        "target_token": TARGET_EOS_TOKEN,
        "tokenizer_length": 262_144,
    }
    if observed != expected:
        raise RetargetError(f"final SFT tokenizer profile mismatch: {observed} != {expected}")
    if TARGET_EOS_ID not in set(tokenizer.all_special_ids):
        raise RetargetError("target EOS id is not registered as a special token")
    vocab_sha, backend_sha = _tokenizer_hashes(tokenizer)
    if vocab_sha != EXPECTED_VOCAB_SHA256 or backend_sha != EXPECTED_BACKEND_SHA256:
        raise RetargetError(
            "final SFT tokenizer vocabulary/backend differs from the data tokenizer: "
            f"vocab={vocab_sha}, backend={backend_sha}"
        )
    return {
        **FINAL_SFT_SOURCE,
        "tokenizer_config_sha256": config_sha,
        "vocab_sha256": vocab_sha,
        "backend_core_sha256": backend_sha,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token": TARGET_EOS_TOKEN,
        "eos_token_id": TARGET_EOS_ID,
    }


def _require_int_list(row: Mapping[str, Any], field: str, pair_id: str) -> list[int]:
    value = row.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RetargetError(f"{pair_id}: {field} must be a sequence")
    if any(type(item) is not int for item in value):
        raise RetargetError(f"{pair_id}: {field} must contain exact integers")
    return list(value)


def _semantic_update(digest: Any, row: Mapping[str, Any]) -> None:
    value: dict[str, Any] = {"pair_id": str(row["pair_id"])}
    for field in SEMANTIC_FIELDS:
        value[field] = _require_int_list(row, field, str(row["pair_id"]))
    digest.update(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _retarget_side(row: dict[str, Any], *, objective: str, side: str, pair_id: str) -> None:
    ids_field = f"{side}_input_ids"
    completion_field = f"{side}_completion_mask"
    term_field = f"{side}_term_mask"
    ids = _require_int_list(row, ids_field, pair_id)
    completion = _require_int_list(row, completion_field, pair_id)
    term = _require_int_list(row, term_field, pair_id)
    if len(ids) < 2 or len(completion) != len(ids) or len(term) != len(ids):
        raise RetargetError(f"{pair_id}: {side} sequence/mask lengths are inconsistent")
    if ids[-1] != SOURCE_EOS_ID or ids.count(SOURCE_EOS_ID) != 1:
        raise RetargetError(
            f"{pair_id}: {side} must contain EOS={SOURCE_EOS_ID} exactly once at the end"
        )
    if completion[-1] != 1:
        raise RetargetError(f"{pair_id}: {side} appended EOS is outside completion mask")
    expected_term_eos = 0 if objective == "mpo" else 1
    if term[-1] != expected_term_eos:
        raise RetargetError(
            f"{pair_id}: {side} EOS term-mask value={term[-1]}, expected={expected_term_eos}"
        )
    if int(row.get(f"{side}_sequence_token_count", -1)) != len(ids):
        raise RetargetError(f"{pair_id}: {side} sequence-token count is inconsistent")
    if int(row.get(f"{side}_completion_token_count", -1)) != sum(completion):
        raise RetargetError(f"{pair_id}: {side} completion-token count is inconsistent")
    if int(row.get(f"{side}_term_token_count", -1)) != sum(term):
        raise RetargetError(f"{pair_id}: {side} term-token count is inconsistent")
    if objective == "mpo" and row.get(f"{side}_eos_appended") is not True:
        raise RetargetError(f"{pair_id}: {side} EOS was not explicitly appended")
    ids[-1] = TARGET_EOS_ID
    row[ids_field] = ids


def retarget_artifact(source: Path, destination: Path, *, objective: str) -> dict[str, Any]:
    if objective not in {"mpo", "cpo"}:
        raise RetargetError(f"unsupported tokenized objective: {objective}")
    expected_schema = (
        "dqs_mpo_token_masks_v1"
        if objective == "mpo"
        else "dqs_full_response_cpo_token_masks_v1"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    semantic = hashlib.sha256()
    seen: set[str] = set()
    rows = 0
    with source.open("r", encoding="utf-8") as input_handle, destination.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                raise RetargetError(f"blank line at {source}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RetargetError(f"non-object row at {source}:{line_number}")
            pair_id = str(row.get("pair_id", "")).strip()
            if not pair_id or pair_id in seen:
                raise RetargetError(f"invalid/duplicate pair_id at line {line_number}: {pair_id!r}")
            seen.add(pair_id)
            if row.get("schema_version") != expected_schema:
                raise RetargetError(f"{pair_id}: unexpected schema_version")
            for side in ("chosen", "rejected"):
                _retarget_side(row, objective=objective, side=side, pair_id=pair_id)
            _semantic_update(semantic, row)
            output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    if rows != EXPECTED_ROWS:
        raise RetargetError(f"{objective}: rows={rows}, expected={EXPECTED_ROWS}")
    return {
        "rows": rows,
        "artifact_sha256": sha256_file(destination),
        "training_semantic_sha256": semantic.hexdigest(),
        "changed_token_positions": rows * 2,
    }


def _alignment_metadata(
    *, script_sha256: str, tokenizer_source: Mapping[str, Any], artifact_scope: str
) -> dict[str, Any]:
    return {
        "mode": "deterministic_appended_completion_eos_retarget_v1",
        "source_eos_token": SOURCE_EOS_TOKEN,
        "source_eos_token_id": SOURCE_EOS_ID,
        "target_eos_token": TARGET_EOS_TOKEN,
        "target_eos_token_id": TARGET_EOS_ID,
        "artifact_change_scope": artifact_scope,
        "sequence_lengths_unchanged": True,
        "masks_unchanged": True,
        "non_terminator_token_ids_unchanged": True,
        "repair_or_fallback": "none",
        "script_sha256": script_sha256,
        "final_sft_tokenizer": dict(tokenizer_source),
    }


def _validate_input_release(input_dir: Path) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for objective in ("mpo", "cpo", "dpo"):
        artifact = input_dir / objective / "train.jsonl"
        contract_path = input_dir / objective / "dataset_contract.json"
        observed_artifact = sha256_file(artifact)
        observed_contract = sha256_file(contract_path)
        if observed_artifact != EXPECTED_INPUT_ARTIFACTS[objective]:
            raise RetargetError(
                f"{objective} source artifact SHA256={observed_artifact}, "
                f"expected={EXPECTED_INPUT_ARTIFACTS[objective]}"
            )
        if observed_contract != EXPECTED_INPUT_CONTRACTS[objective]:
            raise RetargetError(
                f"{objective} source contract SHA256={observed_contract}, "
                f"expected={EXPECTED_INPUT_CONTRACTS[objective]}"
            )
        contract = _load_json(contract_path)
        if str(contract.get("artifact_sha256")) != observed_artifact:
            raise RetargetError(f"{objective} source contract/artifact hash disagreement")
        contracts[objective] = contract
    return contracts


def _rewrite_readme(
    source: Path, destination: Path, artifact_hashes: Mapping[str, str]
) -> None:
    text = source.read_text(encoding="utf-8")
    for objective, old_hash in EXPECTED_INPUT_ARTIFACTS.items():
        new_hash = artifact_hashes[objective]
        count = text.count(old_hash)
        if count != 1:
            raise RetargetError(
                f"README expected one {objective} artifact hash, found {count}"
            )
        text = text.replace(old_hash, new_hash)
    anchor = "- Sequence truncation is forbidden by the contracts.\n"
    addition = (
        "- Completion EOS is `<turn|>` (token id 106), matching the exact final SFT "
        "tokenizer; the earlier base-tokenizer EOS id 1 is forbidden.\n"
    )
    if text.count(anchor) != 1:
        raise RetargetError("README strict-invariant anchor drift")
    text = text.replace(anchor, anchor + addition)
    provenance = (
        "- Final SFT tokenizer: `alwaysgood/dqs-runs@"
        "a58b1878988efcecc9a2644f8324bd00131864b5`, "
        "`gemma4_e2b_it_full_iter_lowqe_sf_on_seed42/checkpoints/final`\n"
    )
    license_anchor = "\nNo license is asserted by this dataset card."
    if text.count(license_anchor) != 1:
        raise RetargetError("README provenance anchor drift")
    text = text.replace(license_anchor, "\n" + provenance + license_anchor)
    destination.write_text(text, encoding="utf-8")


def migrate(input_dir: Path, output_dir: Path, final_tokenizer_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    contracts = _validate_input_release(input_dir)
    tokenizer_source = validate_final_tokenizer(final_tokenizer_dir)
    script_sha = sha256_file(Path(__file__).resolve())
    output_dir.mkdir(parents=True)

    artifact_results = {
        objective: retarget_artifact(
            input_dir / objective / "train.jsonl",
            output_dir / objective / "train.jsonl",
            objective=objective,
        )
        for objective in ("mpo", "cpo")
    }
    (output_dir / "dpo").mkdir(parents=True)
    shutil.copy2(input_dir / "dpo" / "train.jsonl", output_dir / "dpo" / "train.jsonl")
    dpo_sha = sha256_file(output_dir / "dpo" / "train.jsonl")
    if dpo_sha != EXPECTED_INPUT_ARTIFACTS["dpo"]:
        raise RetargetError("DPO text artifact changed during byte-for-byte copy")
    artifact_results["dpo"] = {
        "rows": EXPECTED_ROWS,
        "artifact_sha256": dpo_sha,
        "changed_token_positions": 0,
    }

    mpo = dict(contracts["mpo"])
    mpo["artifact_sha256"] = artifact_results["mpo"]["artifact_sha256"]
    mpo["training_semantic_sha256"] = artifact_results["mpo"][
        "training_semantic_sha256"
    ]
    mpo["eos_token_id"] = TARGET_EOS_ID
    mpo["eos_token"] = TARGET_EOS_TOKEN
    mpo["post_sft_tokenizer_alignment"] = _alignment_metadata(
        script_sha256=script_sha,
        tokenizer_source=tokenizer_source,
        artifact_scope="final appended completion token on chosen and rejected sides",
    )
    mpo_contract_path = output_dir / "mpo" / "dataset_contract.json"
    _write_json(mpo_contract_path, mpo)
    mpo_contract_sha = sha256_file(mpo_contract_path)

    cpo = dict(contracts["cpo"])
    cpo["artifact_sha256"] = artifact_results["cpo"]["artifact_sha256"]
    cpo["training_semantic_sha256"] = artifact_results["cpo"][
        "training_semantic_sha256"
    ]
    cpo["eos_token_id"] = TARGET_EOS_ID
    cpo["eos_token"] = TARGET_EOS_TOKEN
    cpo["post_sft_tokenizer_alignment"] = _alignment_metadata(
        script_sha256=script_sha,
        tokenizer_source=tokenizer_source,
        artifact_scope="final appended completion token on chosen and rejected sides",
    )
    cpo_contract_path = output_dir / "cpo" / "dataset_contract.json"
    _write_json(cpo_contract_path, cpo)
    cpo_contract_sha = sha256_file(cpo_contract_path)

    dpo = dict(contracts["dpo"])
    tokenization = dict(dpo["tokenization_contract"])
    tokenization["eos_token_id"] = TARGET_EOS_ID
    tokenization["eos_token"] = TARGET_EOS_TOKEN
    tokenization["post_sft_tokenizer_alignment"] = _alignment_metadata(
        script_sha256=script_sha,
        tokenizer_source=tokenizer_source,
        artifact_scope="runtime tokenization contract only; raw prompt/completion text unchanged",
    )
    dpo["tokenization_contract"] = tokenization
    dpo["source_quality_contract_sha256"] = mpo_contract_sha
    dpo["cpo_tokenized_artifact_sha256"] = artifact_results["cpo"][
        "artifact_sha256"
    ]
    dpo["cpo_tokenized_contract_sha256"] = cpo_contract_sha
    dpo_contract_path = output_dir / "dpo" / "dataset_contract.json"
    _write_json(dpo_contract_path, dpo)
    dpo_contract_sha = sha256_file(dpo_contract_path)

    contract_hashes = {
        "mpo": mpo_contract_sha,
        "cpo": cpo_contract_sha,
        "dpo": dpo_contract_sha,
    }
    artifact_hashes = {
        objective: str(result["artifact_sha256"])
        for objective, result in artifact_results.items()
    }
    source_manifest = _load_json(input_dir / "manifest.json")
    manifest = dict(source_manifest)
    manifest["tokenizer"] = {
        **dict(source_manifest["tokenizer"]),
        "eos_token": TARGET_EOS_TOKEN,
        "eos_token_id": TARGET_EOS_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "special_tokens_source": tokenizer_source,
    }
    manifest["post_sft_tokenizer_alignment"] = _alignment_metadata(
        script_sha256=script_sha,
        tokenizer_source=tokenizer_source,
        artifact_scope="mPO/CPO appended EOS; DPO runtime tokenization contract",
    )
    objectives = {key: dict(value) for key, value in source_manifest["objectives"].items()}
    for objective in ("mpo", "cpo", "dpo"):
        objectives[objective]["artifact_sha256"] = artifact_hashes[objective]
        objectives[objective]["contract_sha256"] = contract_hashes[objective]
    manifest["objectives"] = objectives
    _write_json(output_dir / "manifest.json", manifest)
    _rewrite_readme(input_dir / "README.md", output_dir / "README.md", artifact_hashes)

    return {
        "status": "ok",
        "source_eos_token_id": SOURCE_EOS_ID,
        "target_eos_token_id": TARGET_EOS_ID,
        "rows": EXPECTED_ROWS,
        "artifacts": artifact_results,
        "contracts": contract_hashes,
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "script_sha256": script_sha,
    }


def main() -> None:
    args = parse_args()
    result = migrate(
        args.input_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.final_tokenizer_dir.expanduser().resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
