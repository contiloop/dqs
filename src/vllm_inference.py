#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Mapping

from io_utils import read_jsonl, write_jsonl
from runtime_logging import configure_runtime_logging, quiet_enabled


configure_runtime_logging()

def _patch_layer_type_validation_compat() -> None:
    try:
        from transformers import configuration_utils as cfg_utils  # type: ignore
    except Exception:
        return
    fn = getattr(cfg_utils, "layer_type_validation", None)
    if fn is None or getattr(fn, "_dqs_compat", False):
        return
    try:
        if len(inspect.signature(fn).parameters) != 1:
            return
    except Exception:
        return

    def _compat(layer_types: Any, num_hidden_layers: Any = None) -> Any:
        return fn(layer_types)

    setattr(_compat, "_dqs_compat", True)
    cfg_utils.layer_type_validation = _compat


def _sampling_params(row: Mapping[str, Any]) -> Any:
    from vllm import SamplingParams

    decoding = row.get("decoding", {})
    if not isinstance(decoding, Mapping):
        decoding = {}
    max_tokens = int(decoding.get("max_new_tokens", 1500) or 1500)
    temperature = float(decoding.get("temperature", 0.0) or 0.0)
    top_p = float(decoding.get("top_p", 1.0) or 1.0)
    if temperature == 0.0:
        return SamplingParams(max_tokens=max_tokens, temperature=0.0)
    return SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)


def _engine_kwargs(first: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = first.get("model", {})
    inference_cfg = first.get("inference", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    if not isinstance(inference_cfg, Mapping):
        inference_cfg = {}

    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "dtype": str(model_cfg.get("dtype", "auto")),
        "language_model_only": True,
    }
    max_model_len = int(model_cfg.get("max_seq_length") or 8192)
    kwargs["max_model_len"] = max_model_len
    gpu_memory_utilization = inference_cfg.get("gpu_memory_utilization")
    if isinstance(gpu_memory_utilization, (int, float)):
        kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
    tensor_parallel_size = int(inference_cfg.get("tensor_parallel_size", 1) or 1)
    if tensor_parallel_size > 1:
        kwargs["tensor_parallel_size"] = tensor_parallel_size
    if model_cfg.get("lora_adapter_path"):
        kwargs["enable_lora"] = True
        kwargs["max_loras"] = 1
    return kwargs


def _lora_request(first: Mapping[str, Any]) -> Any | None:
    model_cfg = first.get("model", {})
    if not isinstance(model_cfg, Mapping):
        return None
    adapter_path = str(model_cfg.get("lora_adapter_path", "") or "").strip()
    if not adapter_path:
        return None
    try:
        from vllm.lora.request import LoRARequest
    except ModuleNotFoundError as exc:
        raise SystemExit("vLLM LoRA evaluation requires vllm.lora.request.LoRARequest") from exc
    return LoRARequest("dqs_checkpoint_adapter", 1, adapter_path)


def run(input_path: Path, output_path: Path) -> None:
    requests = read_jsonl(input_path)
    if not requests:
        write_jsonl(output_path, [])
        return
    _patch_layer_type_validation_compat()
    try:
        from vllm import LLM
    except ModuleNotFoundError as exc:
        raise SystemExit("missing vllm; run `make set` first") from exc

    first = requests[0]
    model_cfg = first.get("model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    model_name = str(model_cfg.get("name_or_path", "")).strip()
    if not model_name:
        raise SystemExit("model.name_or_path is required")

    kwargs = _engine_kwargs(first)
    if not quiet_enabled():
        print(f"[vllm] loading model={model_name} kwargs={kwargs}", file=sys.stderr)
    llm = LLM(model=model_name, **kwargs)
    try:
        lora_request = _lora_request(first)
        prompts = [str(row.get("prompt", "")) for row in requests]
        params = [_sampling_params(row) for row in requests]
        first_param = params[0]
        uniform = all(
            getattr(param, "max_tokens", None) == getattr(first_param, "max_tokens", None)
            and getattr(param, "temperature", None) == getattr(first_param, "temperature", None)
            and getattr(param, "top_p", None) == getattr(first_param, "top_p", None)
            for param in params[1:]
        )
        generate_kwargs: dict[str, Any] = {"sampling_params": first_param if uniform else params}
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request
        outputs = llm.generate(prompts, **generate_kwargs)
        rows: list[dict[str, Any]] = []
        for idx, output in enumerate(outputs):
            request = requests[idx]
            generation = output.outputs[0] if output.outputs else None
            text = generation.text.strip() if generation is not None else ""
            rows.append(
                {
                    "id": str(request.get("id", "")),
                    "row_id": str(request.get("row_id", "")),
                    "order_idx": int(request.get("order_idx", idx)),
                    "status": "ok" if text else "failed",
                    "mt": text,
                    "finish_reason": getattr(generation, "finish_reason", None) if generation else None,
                    "generated_token_count": len(getattr(generation, "token_ids", []) or []) if generation else 0,
                    "error": None if text else "vllm generation returned empty translation",
                }
            )
        write_jsonl(output_path, rows)
    finally:
        try:
            del llm
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM inference for DQS student translations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
