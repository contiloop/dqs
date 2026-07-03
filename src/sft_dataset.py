from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from io_utils import read_jsonl, write_jsonl
from prompting import load_student_templates, render_student_prompt


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _join_prompt_response(prompt: str, response: str) -> tuple[str, int]:
    prompt = prompt.rstrip("\0")
    response = response.strip()
    if not prompt:
        return response, 0
    if prompt.endswith((" ", "\n", "\t")):
        return f"{prompt}{response}", len(prompt)
    return f"{prompt}\n{response}", len(prompt) + 1


def _fallback_prompt(
    *,
    cfg: Mapping[str, Any],
    source: str,
    row_id: str,
    subset_idx: int,
) -> dict[str, Any]:
    prompt_cfg = _get(cfg, "prompts", {})
    if not isinstance(prompt_cfg, Mapping):
        raise SystemExit("prompts config must be a mapping")
    model_cfg = _get(cfg, "model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    template_path = Path(str(prompt_cfg.get("student_templates_path", "prompts/student_templates.yaml")))
    template_cfg = load_student_templates(template_path)
    rendered = render_student_prompt(
        template_cfg=template_cfg,
        prompt_cfg=prompt_cfg,
        model_cfg=model_cfg,
        source=source,
        row_id=row_id,
        subset_idx=subset_idx,
    )
    return {
        "prompt": rendered.text,
        "prompt_template_id": rendered.template_id,
        "prompt_template_group": rendered.template_group,
        "prompt_template_hash": rendered.template_hash,
        "chat_template_applied": rendered.chat_template_applied,
    }


def build_sft_rows(
    *,
    cfg: Mapping[str, Any],
    golden_rows: list[dict[str, Any]],
    subset_idx: int,
) -> list[dict[str, Any]]:
    model_cfg = _get(cfg, "model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    out_rows: list[dict[str, Any]] = []
    for order_idx, row in enumerate(golden_rows):
        row_id = str(row.get("id", f"sft_{order_idx:012d}"))
        source = str(row.get("source", ""))
        target = str(row.get("target", "")).strip()
        if not source.strip() or not target:
            continue

        prompt = str(row.get("prompt", "") or "")
        prompt_meta = {
            "prompt_template_id": row.get("prompt_template_id"),
            "prompt_template_group": row.get("prompt_template_group"),
            "prompt_template_hash": row.get("prompt_template_hash"),
            "chat_template_applied": row.get("chat_template_applied"),
        }
        if not prompt:
            fallback = _fallback_prompt(cfg=cfg, source=source, row_id=row_id, subset_idx=subset_idx)
            prompt = str(fallback["prompt"])
            prompt_meta.update(
                {
                    "prompt_template_id": fallback["prompt_template_id"],
                    "prompt_template_group": fallback["prompt_template_group"],
                    "prompt_template_hash": fallback["prompt_template_hash"],
                    "chat_template_applied": fallback["chat_template_applied"],
                }
            )

        text, response_char_start = _join_prompt_response(prompt, target)
        out_rows.append(
            {
                "id": row_id,
                "order_idx": order_idx,
                "source": source,
                "target": target,
                "prompt": prompt,
                "response": target,
                "text": text,
                "prompt_char_len": len(prompt),
                "response_char_start": response_char_start,
                "sft_format": "prompt_completion_text",
                "model_name_or_path": model_cfg.get("name_or_path"),
                "model_variant": model_cfg.get("variant"),
                "use_hf_chat_template": bool(model_cfg.get("use_hf_chat_template", False)),
                "prompt_template_id": prompt_meta["prompt_template_id"],
                "prompt_template_group": prompt_meta["prompt_template_group"],
                "prompt_template_hash": prompt_meta["prompt_template_hash"],
                "chat_template_applied": bool(prompt_meta["chat_template_applied"]),
                "teacher_label": row.get("teacher_label"),
                "teacher_errors": row.get("teacher_errors", []),
                "qe_score": row.get("qe_score"),
                "selection_rank": row.get("selection_rank"),
                "teacher_accept_rank": row.get("teacher_accept_rank"),
                "source_tokens": row.get("source_tokens"),
                "length_bucket_idx": row.get("length_bucket_idx"),
                "length_bucket": row.get("length_bucket"),
                "metadata": row.get("metadata", {}),
            }
        )
    return out_rows


def write_sft_dataset(
    *,
    cfg: Mapping[str, Any],
    golden_path: str | Path,
    output_path: str | Path,
    subset_idx: int,
) -> dict[str, Any]:
    golden_rows = read_jsonl(golden_path)
    sft_rows = build_sft_rows(cfg=cfg, golden_rows=golden_rows, subset_idx=subset_idx)
    write_jsonl(output_path, sft_rows)
    return {
        "sft_rows": len(sft_rows),
        "sft_dataset_path": str(output_path),
        "golden_rows": len(golden_rows),
    }
