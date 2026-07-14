"""Model loading and trainable-parameter safeguards for mPO post-training."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _dtype(value: Any) -> Any:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError("training.dtype must be set explicitly")
    if text in {"auto", "none", "null"}:
        raise ValueError("automatic dtype selection is forbidden for mPO post-training")
    if text not in {"bf16", "bfloat16"}:
        raise ValueError("mPO post-training requires bfloat16 exactly")
    import torch

    return torch.bfloat16


def configure_distributed_device() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("mPO post-training requires CUDA; CPU/MPS execution is not supported")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "mPO post-training is pinned to bfloat16, but this CUDA device does not support BF16"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} is outside available CUDA devices={device_count}")
    torch.cuda.set_device(local_rank)


def text_tokenizer(tokenizer_or_processor: Any) -> Any:
    current = tokenizer_or_processor
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        nested = getattr(current, "tokenizer", None)
        if nested is None or nested is current:
            break
        current = nested
    return current


def _contains_non_text_module(name: str) -> bool:
    return any(part == "visual" or "vision" in part or "audio" in part for part in name.split("."))


def _gemma4_unused_shared_kv_names(model: Any) -> list[str]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    model_type = str(getattr(text_config, "model_type", "") or "").lower()
    if not model_type.startswith("gemma4"):
        return []
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    num_kv_shared_layers = int(getattr(text_config, "num_kv_shared_layers", 0) or 0)
    if num_hidden_layers <= 0 or num_kv_shared_layers <= 0:
        return []
    first_shared_layer = num_hidden_layers - num_kv_shared_layers
    unused_parts = {"k_norm", "k_proj", "v_proj"}
    names: list[str] = []
    for name, _ in model.named_parameters():
        if _contains_non_text_module(name):
            continue
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part != "layers" or index + 3 >= len(parts):
                continue
            layer_text = parts[index + 1]
            if not layer_text.isdigit() or parts[index + 2] != "self_attn":
                continue
            if int(layer_text) >= first_shared_layer and parts[index + 3] in unused_parts:
                names.append(name)
            break
    return sorted(names)


def _embedding_parameter_contract(model: Any) -> tuple[list[Any], list[str], bool]:
    getter = getattr(model, "get_input_embeddings", None)
    if not callable(getter):
        raise RuntimeError("Gemma4 model does not expose get_input_embeddings()")
    input_embeddings = getter()
    if input_embeddings is None:
        raise RuntimeError("Gemma4 model returned no input embedding module")
    by_id = {id(parameter): parameter for parameter in input_embeddings.parameters()}
    if not by_id:
        raise RuntimeError("Gemma4 input embedding module has no parameters")

    output_getter = getattr(model, "get_output_embeddings", None)
    output_embeddings = output_getter() if callable(output_getter) else None
    output_ids = (
        {id(parameter) for parameter in output_embeddings.parameters()}
        if output_embeddings is not None
        else set()
    )
    input_ids = set(by_id)
    names = sorted(
        name for name, parameter in model.named_parameters() if id(parameter) in input_ids
    )
    if not names:
        raise RuntimeError("could not map Gemma4 input embedding parameters to model names")
    return list(by_id.values()), names, bool(input_ids & output_ids)


def prepare_full_finetuning_parameters(
    model: Any,
    *,
    freeze_embeddings: bool,
) -> dict[str, Any]:
    """Full-tune text parameters with explicit optional token-embedding freeze."""

    if type(freeze_embeddings) is not bool:
        raise TypeError("freeze_embeddings must be an explicit boolean")
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    model_type = str(getattr(text_config, "model_type", "") or "").lower()
    if not model_type.startswith("gemma4"):
        raise RuntimeError(f"mPO post-training requires a Gemma4 model, got model_type={model_type!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    embedding_parameters, embedding_names, embedding_output_tied = _embedding_parameter_contract(model)
    frozen_non_text: list[str] = []
    for name, parameter in model.named_parameters():
        if _contains_non_text_module(name):
            parameter.requires_grad_(False)
            frozen_non_text.append(name)
    frozen_structural = _gemma4_unused_shared_kv_names(model)
    frozen_structural_set = set(frozen_structural)
    for name, parameter in model.named_parameters():
        if name in frozen_structural_set:
            parameter.requires_grad_(False)
    if freeze_embeddings:
        for parameter in embedding_parameters:
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in embedding_parameters):
            raise RuntimeError("failed to freeze every input embedding parameter")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError("model has no trainable text parameters")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "full_finetuning": True,
        "freeze_embeddings": bool(freeze_embeddings),
        "input_embedding_parameter_count": len(embedding_parameters),
        "input_embedding_parameter_names": embedding_names,
        "input_embedding_numel": sum(parameter.numel() for parameter in embedding_parameters),
        "input_embedding_output_weight_tied": embedding_output_tied,
        "frozen_embedding_parameter_count": len(embedding_parameters) if freeze_embeddings else 0,
        "frozen_embedding_parameter_names": embedding_names if freeze_embeddings else [],
        "frozen_non_text_parameter_count": len(frozen_non_text),
        "frozen_structurally_unused_parameter_count": len(frozen_structural),
        "frozen_structurally_unused_parameter_names": frozen_structural,
    }


def require_final_stage_model(model_cfg: Mapping[str, Any]) -> None:
    if not bool(model_cfg.get("require_final_stage_model", True)):
        raise ValueError(
            "model.require_final_stage_model must remain true; base-model initialization is forbidden"
        )
    raw = str(model_cfg.get("name_or_path", "")).strip()
    if not raw:
        raise ValueError("model.name_or_path must point to the local SFT checkpoints/final directory")
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            "model.name_or_path must be a downloaded local SFT final directory so its provenance "
            f"and weights can be verified: {path}"
        )
    if not path.is_dir():
        raise ValueError(f"model.name_or_path is not a directory: {path}")
    if path.name != "final":
        raise ValueError(
            "post-training must initialize from the SFT final model directory, "
            f"not an optimizer checkpoint: {path}"
        )
    marker = path / "dqs_stage_model.json"
    if not marker.exists():
        raise ValueError(f"missing final-stage provenance marker: {marker}")
    if not (path / "config.json").is_file():
        raise ValueError(f"missing model config in final-stage directory: {path / 'config.json'}")
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError(
            f"missing tokenizer config in final-stage directory: {path / 'tokenizer_config.json'}"
        )
    weight_files = [
        candidate
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for candidate in path.glob(pattern)
        if candidate.is_file()
    ]
    if not weight_files:
        raise ValueError(f"missing model weight files in final-stage directory: {path}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if str(payload.get("tuning_mode", "")).lower() != "full":
        raise ValueError(f"final-stage marker is not a full-tuning model: {marker}")
    expected_marker = {
        "run_id": str(model_cfg.get("expected_sft_run_id", "")),
        "subset_idx": int(model_cfg.get("expected_sft_subset_idx", -1)),
        "global_step": int(model_cfg.get("expected_sft_global_step", -1)),
    }
    observed_marker = {
        "run_id": str(payload.get("run_id", "")),
        "subset_idx": int(payload.get("subset_idx", -1)),
        "global_step": int(payload.get("global_step", -1)),
    }
    if observed_marker != expected_marker:
        raise ValueError(
            "SFT final marker does not match the configured source checkpoint: "
            f"observed={observed_marker}, expected={expected_marker}"
        )


def _load_unsloth(model_cfg: Mapping[str, Any], training_cfg: Mapping[str, Any]) -> tuple[Any, Any, str]:
    # Set this before importing Unsloth. Its compiler otherwise may replace
    # logits with EMPTY_LOGITS even when the custom trainer needs raw logits.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    if "transformers" in sys.modules and "unsloth" not in sys.modules:
        raise RuntimeError(
            "transformers was imported before unsloth; restart the process and use train_mpo.py so patches are deterministic"
        )
    try:
        from unsloth import FastModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unsloth is required. Install the pinned GPU environment; no alternate backend is allowed."
        ) from exc

    api_name = str(model_cfg["unsloth_model_api"]).strip().lower()
    if api_name != "fast_model":
        raise ValueError("Gemma4 mPO requires model.unsloth_model_api=fast_model exactly")
    kwargs: dict[str, Any] = {
        "model_name": str(model_cfg["name_or_path"]),
        "max_seq_length": int(training_cfg["max_seq_length"]),
        "dtype": _dtype(training_cfg["dtype"]),
        "load_in_4bit": bool(training_cfg["load_in_4bit"]),
        "load_in_8bit": bool(training_cfg["load_in_8bit"]),
        "load_in_16bit": False,
        "full_finetuning": True,
        "trust_remote_code": bool(model_cfg["trust_remote_code"]),
        "use_gradient_checkpointing": training_cfg["gradient_checkpointing"],
        # Unsloth can replace normal logits with an empty sentinel when its
        # fused built-in loss is expected.  mPO owns the token log-probs.
        "return_logits": True,
        "fullgraph": bool(training_cfg["unsloth_fullgraph"]),
        "fast_inference": False,
        "text_only": False,
    }
    if model_cfg.get("revision"):
        kwargs["revision"] = str(model_cfg["revision"])
    if model_cfg.get("subfolder"):
        kwargs["subfolder"] = str(model_cfg["subfolder"])
    # Do not filter unsupported arguments.  In particular, silently dropping
    # full_finetuning or return_logits would change the objective.  A runtime
    # that cannot accept this exact call must fail here.
    model, tokenizer = FastModel.from_pretrained(**kwargs)
    return model, tokenizer, FastModel.__name__


def load_model_and_tokenizer(
    model_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    require_final_stage_model(model_cfg)
    backend = str(training_cfg.get("backend", "")).strip().lower()
    if backend != "unsloth":
        raise ValueError("mPO post-training requires backend=unsloth; no backend fallback is implemented")
    model, tokenizer, api_name = _load_unsloth(model_cfg, training_cfg)

    tokenizer_backend = text_tokenizer(tokenizer)
    if getattr(tokenizer_backend, "pad_token_id", None) is None:
        raise RuntimeError("tokenizer has no pad_token_id; automatic EOS-as-padding is forbidden")
    if getattr(tokenizer_backend, "eos_token_id", None) is None:
        raise RuntimeError("tokenizer has no eos_token_id")
    if getattr(model, "config", None) is None:
        raise RuntimeError("loaded model has no config")
    model.config.pad_token_id = int(tokenizer_backend.pad_token_id)
    model.config.use_cache = False
    parameter_summary = prepare_full_finetuning_parameters(
        model,
        freeze_embeddings=bool(training_cfg["freeze_embeddings"]),
    )
    return model, tokenizer, {
        "backend": backend,
        "model_api": api_name,
        "pad_token_id": int(tokenizer_backend.pad_token_id),
        "eos_token_id": int(tokenizer_backend.eos_token_id),
        "tokenizer_length": len(tokenizer_backend),
        **parameter_summary,
    }


def validate_model_data_contract(
    tokenizer_or_processor: Any,
    *,
    dataset_summary: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    tokenizer = text_tokenizer(tokenizer_or_processor)
    expected_pad = contract.get("pad_token_id")
    expected_eos = contract.get("eos_token_id")
    if expected_pad is None or expected_eos is None:
        raise ValueError("dataset contract must pin pad_token_id and eos_token_id")
    if int(tokenizer.pad_token_id) != int(expected_pad):
        raise ValueError(f"tokenizer pad_token_id={tokenizer.pad_token_id}, expected={expected_pad}")
    if int(tokenizer.eos_token_id) != int(expected_eos):
        raise ValueError(f"tokenizer eos_token_id={tokenizer.eos_token_id}, expected={expected_eos}")
    expected_vocab_sha = str(contract.get("tokenizer_vocab_sha256", ""))
    expected_backend_sha = str(contract.get("tokenizer_backend_core_sha256", ""))
    if not expected_vocab_sha or not expected_backend_sha:
        raise ValueError("dataset contract must pin tokenizer vocab and backend SHA256 values")
    vocab_payload = json.dumps(
        tokenizer.get_vocab(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_vocab_sha = hashlib.sha256(vocab_payload).hexdigest()
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is None:
        raise ValueError("the runtime tokenizer is not a fast tokenizer with a backend contract")
    backend_payload = json.loads(backend_tokenizer.to_str())
    # Padding/truncation are runtime batching settings and may be installed by
    # Unsloth after load.  The immutable normalization, pre-tokenization,
    # model, decoder, post-processor, and added-token pipeline stays pinned.
    backend_payload.pop("padding", None)
    backend_payload.pop("truncation", None)
    backend_canonical = json.dumps(
        backend_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_backend_sha = hashlib.sha256(backend_canonical).hexdigest()
    if observed_vocab_sha != expected_vocab_sha:
        raise ValueError(
            f"tokenizer vocabulary mismatch: observed={observed_vocab_sha}, expected={expected_vocab_sha}"
        )
    if observed_backend_sha != expected_backend_sha:
        raise ValueError(
            "tokenizer implementation mismatch: "
            f"observed={observed_backend_sha}, expected={expected_backend_sha}"
        )
    max_token_id = int(dataset_summary["train"]["max_token_id"])
    if max_token_id >= len(tokenizer):
        raise ValueError(
            f"dataset token id {max_token_id} is outside model tokenizer length {len(tokenizer)}"
        )
