#!/usr/bin/env python3
"""Build a non-destructive vLLM compatibility view of a trained Gemma-4 model.

Transformers omits the structurally unused K/V projection and K-norm weights
from Gemma-4 shared-KV layers when a fine-tuned model is saved.  vLLM 0.19.1
still instantiates those parameters and requires them during strict checkpoint
loading, even though its shared-KV forward path does not consume them.

This script hard-links the trained checkpoint shards into a new directory and
adds only the missing shared-KV weights from the frozen source SFT checkpoint.
The trained model directory is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SHARED_WEIGHT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<part>k_norm|k_proj|v_proj)\.weight$"
)
_REQUIRED_SHARED_PART = "k_norm"
_COMPAT_MARKER = "dqs_vllm_compat.json"


@dataclass(frozen=True)
class WeightSet:
    directory: Path
    weight_map: dict[str, str]
    files: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-model", type=Path, required=True)
    parser.add_argument("--source-sft-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace only a previous output carrying dqs_vllm_compat.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be a mapping: {path}")
    return payload


def _text_config(model_dir: Path) -> Mapping[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    config = _load_json(config_path)
    nested = config.get("text_config", config)
    if not isinstance(nested, Mapping):
        raise ValueError(f"text_config must be a mapping: {config_path}")
    model_type = str(nested.get("model_type", "")).lower()
    if not model_type.startswith("gemma4"):
        raise ValueError(f"expected Gemma-4 text config, observed model_type={model_type!r}")
    return nested


def _architecture_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model_type",
        "num_hidden_layers",
        "num_kv_shared_layers",
        "hidden_size",
        "head_dim",
        "global_head_dim",
        "num_key_value_heads",
        "num_global_key_value_heads",
        "attention_k_eq_v",
        "layer_types",
    )
    return {key: config.get(key) for key in keys}


def _validate_architecture(
    trained_config: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> tuple[int, int]:
    trained_contract = _architecture_contract(trained_config)
    source_contract = _architecture_contract(source_config)
    if trained_contract != source_contract:
        differing = sorted(
            key
            for key in trained_contract
            if trained_contract[key] != source_contract[key]
        )
        raise ValueError(
            "trained and source SFT Gemma-4 architectures differ: "
            + ", ".join(differing)
        )
    num_layers = int(trained_config.get("num_hidden_layers", 0) or 0)
    shared_layers = int(trained_config.get("num_kv_shared_layers", 0) or 0)
    if num_layers <= 0 or shared_layers <= 0 or shared_layers >= num_layers:
        raise ValueError(
            "Gemma-4 shared-KV contract is invalid: "
            f"num_hidden_layers={num_layers}, num_kv_shared_layers={shared_layers}"
        )
    return num_layers, shared_layers


def _safetensor_keys(path: Path) -> list[str]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise RuntimeError("safetensors is required to prepare the vLLM checkpoint") from exc
    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def _weight_set(model_dir: Path) -> WeightSet:
    model_dir = model_dir.expanduser().resolve()
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, Mapping) or not raw_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        weight_map = {str(key): str(value) for key, value in raw_map.items()}
        files = tuple(sorted(set(weight_map.values())))
    else:
        candidates = tuple(
            sorted(
                path.name
                for path in model_dir.glob("*.safetensors")
                if path.is_file()
            )
        )
        if not candidates:
            raise FileNotFoundError(f"no safetensors checkpoint found: {model_dir}")
        weight_map: dict[str, str] = {}
        for filename in candidates:
            for key in _safetensor_keys(model_dir / filename):
                if key in weight_map:
                    raise ValueError(f"duplicate tensor key {key!r} in {model_dir}")
                weight_map[key] = filename
        files = candidates
    for filename in files:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"indexed weight shard is missing: {path}")
    return WeightSet(directory=model_dir, weight_map=weight_map, files=files)


def _shared_key(key: str, *, first_shared_layer: int) -> tuple[int, str] | None:
    match = _SHARED_WEIGHT_RE.search(key)
    if match is None:
        return None
    layer = int(match.group("layer"))
    if layer < first_shared_layer:
        return None
    return layer, match.group("part")


def _missing_shared_weights(
    trained: WeightSet,
    source: WeightSet,
    *,
    num_layers: int,
    shared_layers: int,
) -> list[str]:
    first_shared = num_layers - shared_layers
    trained_keys = set(trained.weight_map)
    selected = sorted(
        key
        for key in source.weight_map
        if key not in trained_keys
        and _shared_key(key, first_shared_layer=first_shared) is not None
    )
    required_layers = {
        layer
        for key in [*trained.weight_map, *selected]
        if (parsed := _shared_key(key, first_shared_layer=first_shared)) is not None
        for layer, part in [parsed]
        if part == _REQUIRED_SHARED_PART
    }
    expected_layers = set(range(first_shared, num_layers))
    if required_layers != expected_layers:
        missing = sorted(expected_layers - required_layers)
        raise ValueError(
            "source SFT checkpoint cannot restore every missing shared-KV k_norm; "
            f"missing_layers={missing}"
        )
    return selected


def _load_selected_tensors(weights: WeightSet, keys: list[str]) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise RuntimeError("safetensors is required to prepare the vLLM checkpoint") from exc
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weights.weight_map[key], []).append(key)
    tensors: dict[str, Any] = {}
    for filename, shard_keys in sorted(by_file.items()):
        with safe_open(weights.directory / filename, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
        return "copy"
    return "hardlink"


def _prepare_output(output: Path, *, replace: bool) -> None:
    if not output.exists():
        return
    marker = output / _COMPAT_MARKER
    if not replace:
        raise FileExistsError(f"output already exists; pass --replace: {output}")
    if not marker.is_file():
        raise ValueError(f"refusing to replace an unmarked directory: {output}")
    shutil.rmtree(output)


def prepare_vllm_checkpoint(
    *,
    trained_model: Path,
    source_sft_model: Path,
    output: Path,
    replace: bool = False,
) -> dict[str, Any]:
    trained_model = trained_model.expanduser().resolve()
    source_sft_model = source_sft_model.expanduser().resolve()
    output = output.expanduser().resolve()
    if output in {trained_model, source_sft_model}:
        raise ValueError("compatibility output must differ from both input model directories")

    trained_config = _text_config(trained_model)
    source_config = _text_config(source_sft_model)
    num_layers, shared_layers = _validate_architecture(trained_config, source_config)
    trained_weights = _weight_set(trained_model)
    source_weights = _weight_set(source_sft_model)
    supplemental_keys = _missing_shared_weights(
        trained_weights,
        source_weights,
        num_layers=num_layers,
        shared_layers=shared_layers,
    )
    supplemental_tensors = _load_selected_tensors(source_weights, supplemental_keys)

    _prepare_output(output, replace=replace)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        weight_sources = set(trained_weights.files)
        for path in sorted(trained_model.iterdir()):
            if not path.is_file() or path.name in weight_sources:
                continue
            if path.name == "model.safetensors.index.json":
                continue
            _link_or_copy(path, staging / path.name)

        shard_count = len(trained_weights.files) + 1
        renamed: dict[str, str] = {}
        link_modes: set[str] = set()
        for index, filename in enumerate(trained_weights.files, start=1):
            target_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            renamed[filename] = target_name
            link_modes.add(
                _link_or_copy(
                    trained_weights.directory / filename,
                    staging / target_name,
                )
            )

        supplement_name = f"model-{shard_count:05d}-of-{shard_count:05d}.safetensors"
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("safetensors is required to prepare the vLLM checkpoint") from exc
        save_file(supplemental_tensors, staging / supplement_name)

        weight_map = {
            key: renamed[filename]
            for key, filename in trained_weights.weight_map.items()
        }
        weight_map.update({key: supplement_name for key in supplemental_keys})
        total_size = sum(
            (staging / filename).stat().st_size
            for filename in sorted(set(weight_map.values()))
        )
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        marker = {
            "schema_version": "dqs.gemma4_vllm_compat.v1",
            "trained_model": str(trained_model),
            "source_sft_model": str(source_sft_model),
            "num_hidden_layers": num_layers,
            "num_kv_shared_layers": shared_layers,
            "first_shared_layer": num_layers - shared_layers,
            "supplemental_keys": supplemental_keys,
            "supplemental_key_count": len(supplemental_keys),
            "supplement_sha256": _sha256(staging / supplement_name),
            "base_shard_materialization": sorted(link_modes),
            "trained_model_modified": False,
        }
        (staging / _COMPAT_MARKER).write_text(
            json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**marker, "output": str(output)}


def main() -> None:
    args = parse_args()
    result = prepare_vllm_checkpoint(
        trained_model=args.trained_model,
        source_sft_model=args.source_sft_model,
        output=args.output,
        replace=bool(args.replace),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
