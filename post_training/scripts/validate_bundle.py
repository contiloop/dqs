#!/usr/bin/env python3
"""Fail-closed validation for a generated DQS preference-training bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

import yaml

from download_model import load_model_spec, validate_model_dir


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OBJECTIVES = {"mpo", "cpo", "dpo"}
EXPECTED_OUTPUT_DIRS = {
    "mpo": "outputs/mpo",
    "cpo": "outputs/cpo",
    "dpo": "outputs/dpo",
}
EXPECTED_RUNTIME = {
    "accelerate": "1.14.0",
    "datasets": "4.3.0",
    "huggingface-hub": "1.21.0",
    "pyyaml": "6.0.3",
    "tokenizers": "0.22.2",
    "trl": "0.24.0",
    "transformers": "5.5.3",
    "unsloth": "2026.7.2",
    "unsloth-zoo": "2026.7.2",
    "wandb": "0.28.0",
}
EXPECTED_EOS_TOKEN = "<turn|>"
EXPECTED_EOS_TOKEN_ID = 106
BASE_EOS_TOKEN = "<eos>"
BASE_EOS_TOKEN_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also require the source-verified package versions and supported torch range",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="also verify the full-SFT final model provenance marker",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def resolve_bundle_path(raw: str, *, field: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a bundle-relative path: {raw!r}")
    resolved = (ROOT / path).resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ValueError(f"{field} escapes the bundle: {raw!r}")
    return resolved


def validate_manifest() -> dict[str, Any]:
    marker = ROOT / ".dqs-preference-release"
    if not marker.is_file():
        raise FileNotFoundError(f"release marker is missing: {marker}")
    manifest = load_json(ROOT / "manifest.json")
    if manifest.get("schema_version") != "dqs_preference_release.v1":
        raise ValueError("unsupported release manifest schema")
    objectives = manifest.get("objectives")
    if not isinstance(objectives, dict) or set(objectives) != EXPECTED_OBJECTIVES:
        raise ValueError("manifest must contain exactly mpo/cpo/dpo")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest has no immutable file inventory")
    observed: dict[str, str] = {}
    for relative, expected in files.items():
        path = resolve_bundle_path(str(relative), field="manifest.files")
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"bundle file hash mismatch for {relative}: observed={digest} expected={expected}"
            )
        observed[str(relative)] = digest
    aggregate = hashlib.sha256()
    for relative, digest in sorted(observed.items()):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if aggregate.hexdigest() != manifest.get("bundle_content_sha256"):
        raise ValueError("bundle aggregate SHA256 does not match manifest")
    model_manifest, _ = load_model_spec(ROOT)
    if model_manifest != manifest:
        raise ValueError("downloadable SFT model contract uses a different manifest")
    return manifest


def _line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank JSONL line is forbidden: {path}:{count + 1}")
            count += 1
    return count


def validate_objective(
    name: str,
    objective: dict[str, Any],
    *,
    data_mode: str,
    hf_dataset: Any,
    sft_model: Any,
) -> dict[str, Any]:
    config_path = resolve_bundle_path(str(objective["config"]), field=f"{name}.config")
    contract_path = resolve_bundle_path(str(objective["contract"]), field=f"{name}.contract")
    entrypoint = resolve_bundle_path(str(objective["entrypoint"]), field=f"{name}.entrypoint")
    if not entrypoint.is_file():
        raise FileNotFoundError(f"{name} entrypoint is missing: {entrypoint}")
    config = load_yaml(config_path)
    contract = load_json(contract_path)
    data = config.get("data")
    run = config.get("run")
    model = config.get("model")
    if not all(isinstance(value, dict) for value in (data, run, model)):
        raise ValueError(f"{name} config is missing run/model/data mappings")
    if run.get("output_dir") != EXPECTED_OUTPUT_DIRS[name]:
        raise ValueError(
            f"{name} output must be {EXPECTED_OUTPUT_DIRS[name]!r}, "
            f"got {run.get('output_dir')!r}"
        )
    if model.get("name_or_path") != "models/sft_final":
        raise ValueError(f"{name} default model path must be models/sft_final")
    if data.get("cache_dir") != f".cache/datasets/{name}":
        raise ValueError(f"{name} datasets cache must stay under .cache/datasets/{name}")
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True)
    if "post_training/" in serialized or "CHANGE_ME" in serialized:
        raise ValueError(f"{name} config contains a research-layout or placeholder path")
    contract_serialized = json.dumps(contract, ensure_ascii=False, sort_keys=True)
    if re.search(r'"/(?:Users|home|root|workspace)/', contract_serialized):
        raise ValueError(f"{name} contract contains a workstation-specific absolute path")
    if sha256_file(contract_path) != objective.get("release_contract_sha256"):
        raise ValueError(f"{name} release contract hash differs from manifest")
    source_contract_sha = str(objective.get("source_contract_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", source_contract_sha) is None:
        raise ValueError(f"{name} source contract SHA256 is missing")
    if int(contract.get("row_count", -1)) != int(objective["row_count"]):
        raise ValueError(f"{name} contract row count differs from manifest")
    if contract.get("artifact_sha256") != objective.get("artifact_sha256"):
        raise ValueError(f"{name} contract artifact hash differs from manifest")

    tokenization = contract.get("tokenization_contract") if name == "dpo" else contract
    if not isinstance(tokenization, dict):
        raise ValueError(f"{name} tokenizer contract is missing")
    if int(tokenization.get("eos_token_id", -1)) != EXPECTED_EOS_TOKEN_ID:
        raise ValueError(
            f"{name} EOS id must match the final SFT tokenizer: {EXPECTED_EOS_TOKEN_ID}"
        )
    if tokenization.get("eos_token") != EXPECTED_EOS_TOKEN:
        raise ValueError(f"{name} EOS token must be {EXPECTED_EOS_TOKEN!r}")
    alignment = tokenization.get("post_sft_tokenizer_alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"{name} has no final-SFT tokenizer alignment receipt")
    expected_alignment = {
        "source_eos_token": BASE_EOS_TOKEN,
        "source_eos_token_id": BASE_EOS_TOKEN_ID,
        "target_eos_token": EXPECTED_EOS_TOKEN,
        "target_eos_token_id": EXPECTED_EOS_TOKEN_ID,
        "repair_or_fallback": "none",
    }
    observed_alignment = {key: alignment.get(key) for key in expected_alignment}
    if observed_alignment != expected_alignment:
        raise ValueError(
            f"{name} final-SFT tokenizer alignment mismatch: "
            f"{observed_alignment} != {expected_alignment}"
        )
    tokenizer_source = alignment.get("final_sft_tokenizer")
    if not isinstance(tokenizer_source, dict) or not isinstance(sft_model, dict):
        raise ValueError(f"{name} final-SFT tokenizer provenance is missing")
    source_binding = {
        "repo_id": tokenizer_source.get("repo_id"),
        "repo_type": tokenizer_source.get("repo_type"),
        "revision": tokenizer_source.get("revision"),
        "subfolder": tokenizer_source.get("subfolder"),
        "tokenizer_config_sha256": tokenizer_source.get("tokenizer_config_sha256"),
    }
    expected_binding = {
        "repo_id": sft_model.get("repo_id"),
        "repo_type": sft_model.get("repo_type"),
        "revision": sft_model.get("revision"),
        "subfolder": sft_model.get("remote_dir"),
        "tokenizer_config_sha256": (
            sft_model.get("files", {}).get("tokenizer_config.json", {}).get("sha256")
        ),
    }
    if source_binding != expected_binding:
        raise ValueError(
            f"{name} tokenizer source is not the downloadable final SFT model: "
            f"{source_binding} != {expected_binding}"
        )

    if data_mode == "local":
        if (
            data.get("source") != "local"
            or not objective.get("data")
            or not bool(objective.get("data_bundled"))
        ):
            raise ValueError(f"{name} is not configured for bundled local data")
        data_path = resolve_bundle_path(str(objective["data"]), field=f"{name}.data")
        configured_path = resolve_bundle_path(str(data.get("path")), field=f"{name}.data.path")
        if data_path != configured_path:
            raise ValueError(f"{name} config and manifest point to different data files")
        digest = sha256_file(data_path)
        if digest != objective["artifact_sha256"]:
            raise ValueError(f"{name} local data hash differs from immutable contract")
        rows = _line_count(data_path)
        if rows != int(objective["row_count"]):
            raise ValueError(f"{name} local JSONL rows={rows}, expected={objective['row_count']}")
    elif data_mode == "hf":
        if data.get("source") != "local" or bool(objective.get("data_bundled")):
            raise ValueError(f"{name} HF-backed release must train from explicitly staged local data")
        configured_path = resolve_bundle_path(str(data.get("path")), field=f"{name}.data.path")
        target_path = resolve_bundle_path(str(objective.get("data")), field=f"{name}.data")
        if configured_path != target_path:
            raise ValueError(f"{name} config and download target point to different files")
        for field in (
            "hf_repo_id",
            "hf_revision",
            "hf_config_name",
            "hf_train_filename",
            "hf_eval_filename",
            "hf_train_split",
            "hf_eval_split",
            "hf_contract_filename",
        ):
            if data.get(field) is not None:
                raise ValueError(f"{name} trainer config must not carry remote field data.{field}")
        if not isinstance(hf_dataset, dict):
            raise ValueError("manifest.hf_dataset is missing")
        revision = str(hf_dataset.get("revision", ""))
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError("HF dataset revision is not an exact 40-hex commit")
        for field in ("hf_train_filename", "hf_contract_filename"):
            if not str(objective.get(field, "")).strip():
                raise ValueError(f"{name} manifest is missing {field}")
    else:
        raise ValueError(f"unsupported data_mode={data_mode!r}")
    return {
        "rows": int(objective["row_count"]),
        "artifact_sha256": str(objective["artifact_sha256"]),
        "config": objective["config"],
        "eos_token_id": int(tokenization["eos_token_id"]),
    }


def validate_runtime() -> dict[str, str]:
    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for package, required in EXPECTED_RUNTIME.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        observed[package] = version
        if version != required:
            mismatches[package] = {"observed": version, "required": required}
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "missing"
    observed["torch"] = torch_version
    normalized = torch_version.split("+", 1)[0]
    try:
        parts = tuple(int(part) for part in normalized.split(".")[:2])
    except ValueError as exc:
        raise RuntimeError(f"cannot parse torch version={torch_version!r}") from exc
    if not ((2, 4) <= parts < (2, 11)):
        mismatches["torch"] = {"observed": torch_version, "required": ">=2.4,<2.11"}
    if mismatches:
        raise RuntimeError("runtime version mismatch: " + json.dumps(mismatches, sort_keys=True))
    return observed


def validate_model_eos_profile(model_dir: Path) -> dict[str, Any]:
    tokenizer_config = load_json(model_dir / "tokenizer_config.json")
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise ValueError("final SFT tokenizer has no added-token decoder")
    target = decoder.get(str(EXPECTED_EOS_TOKEN_ID))
    source = decoder.get(str(BASE_EOS_TOKEN_ID))
    if (
        tokenizer_config.get("eos_token") != EXPECTED_EOS_TOKEN
        or not isinstance(target, dict)
        or target.get("content") != EXPECTED_EOS_TOKEN
        or target.get("special") is not True
        or not isinstance(source, dict)
        or source.get("content") != BASE_EOS_TOKEN
    ):
        raise ValueError("downloaded final SFT tokenizer EOS profile is not exact")
    model_config = load_json(model_dir / "config.json")
    generation_config = load_json(model_dir / "generation_config.json")
    text_config = model_config.get("text_config")
    generation_eos = generation_config.get("eos_token_id")
    if (
        int(model_config.get("eos_token_id", -1)) != EXPECTED_EOS_TOKEN_ID
        or not isinstance(text_config, dict)
        or int(text_config.get("eos_token_id", -1)) != BASE_EOS_TOKEN_ID
        or not isinstance(generation_eos, list)
        or BASE_EOS_TOKEN_ID not in generation_eos
        or EXPECTED_EOS_TOKEN_ID not in generation_eos
    ):
        raise ValueError("downloaded final SFT model/generation EOS profile is not exact")
    return {
        "tokenizer_eos_token": EXPECTED_EOS_TOKEN,
        "tokenizer_eos_token_id": EXPECTED_EOS_TOKEN_ID,
        "generation_eos_token_ids": generation_eos,
    }


def validate_model(model_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    model_manifest, spec = load_model_spec(ROOT)
    if model_manifest != manifest:
        raise ValueError("supplied manifest does not match the model download contract")
    summary = validate_model_dir(model_dir, spec)
    return {**summary, **validate_model_eos_profile(model_dir)}


def main() -> None:
    args = parse_args()
    manifest = validate_manifest()
    data_mode = str(manifest.get("data_mode", ""))
    data_access = str(manifest.get("data_access", ""))
    expected_access = "bundled_local" if data_mode == "local" else "explicit_download_then_local"
    if data_access != expected_access:
        raise ValueError(
            f"manifest data_access={data_access!r}, expected={expected_access!r}"
        )
    objectives = {
        name: validate_objective(
            name,
            dict(value),
            data_mode=data_mode,
            hf_dataset=manifest.get("hf_dataset"),
            sft_model=manifest.get("sft_model"),
        )
        for name, value in manifest["objectives"].items()
    }
    result: dict[str, Any] = {
        "status": "ok",
        "bundle_content_sha256": manifest["bundle_content_sha256"],
        "data_mode": data_mode,
        "data_access": data_access,
        "objectives": objectives,
    }
    if args.runtime:
        result["runtime"] = validate_runtime()
    if args.model_dir is not None:
        result["model"] = validate_model(args.model_dir, manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
