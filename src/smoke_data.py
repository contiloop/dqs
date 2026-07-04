#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config, save_effective_config
from io_utils import write_jsonl
from prompting import load_student_templates, render_student_prompt
from sft_train import _default_max_seq_length, _tokenize_prompt_completion
from text_tokenization import text_decode, text_token_ids


SOURCE_SENTENCES = [
    "Revenue increased as enterprise customers renewed multi-year contracts and expanded usage across regulated financial workflows.",
    "Operating expenses remained disciplined while the company continued to invest in compliance, data infrastructure, and customer support.",
    "Management expects demand to remain resilient, although foreign exchange volatility and interest-rate uncertainty may affect reported results.",
    "The board authorized additional capital expenditures for risk controls, cybersecurity systems, and automation of internal reporting processes.",
    "Cash flow from operations improved because collections accelerated and working capital requirements declined during the quarter.",
    "The company noted that backlog includes cancellable commitments and should not be interpreted as a guaranteed revenue forecast.",
    "Gross margin benefited from a favorable product mix, partially offset by higher hosting costs and professional service delivery expenses.",
    "These statements involve risks and uncertainties that could cause actual results to differ materially from current expectations.",
]

TARGET_SENTENCES = [
    "기업 고객이 다년 계약을 갱신하고 규제 산업의 금융 업무 전반에서 사용량을 확대하면서 매출이 증가했다.",
    "회사는 컴플라이언스, 데이터 인프라, 고객 지원에 계속 투자하는 가운데 영업비용을 절제된 수준으로 관리했다.",
    "경영진은 수요가 견조하게 유지될 것으로 예상하지만 환율 변동성과 금리 불확실성이 보고 실적에 영향을 줄 수 있다고 밝혔다.",
    "이사회는 리스크 관리, 사이버보안 시스템, 내부 보고 자동화를 위한 추가 자본 지출을 승인했다.",
    "매출채권 회수가 빨라지고 운전자본 부담이 낮아지면서 영업활동 현금흐름이 개선됐다.",
    "회사는 수주잔고에 취소 가능한 약정이 포함되어 있으며 이를 확정 매출 전망으로 해석해서는 안 된다고 설명했다.",
    "매출총이익률은 제품 구성 개선의 수혜를 받았으나 호스팅 비용과 전문 서비스 제공 비용 증가로 일부 상쇄됐다.",
    "이러한 진술은 실제 결과가 현재 예상과 크게 달라질 수 있는 위험과 불확실성을 포함한다.",
]

VAL_ROWS = [
    {
        "source": "Net sales increased 8.5% due to higher demand in North America.",
        "target": "북미 지역 수요 증가로 순매출은 8.5% 증가했다.",
    },
    {
        "source": "The company expects capital expenditures to remain elevated through fiscal 2026.",
        "target": "회사는 2026 회계연도까지 자본 지출이 높은 수준을 유지할 것으로 예상한다.",
    },
    {
        "source": "Foreign currency movements reduced operating income by approximately $12 million.",
        "target": "환율 변동은 영업이익을 약 1,200만 달러 감소시켰다.",
    },
    {
        "source": "Deferred revenue primarily reflects advance billings for subscription contracts.",
        "target": "이연수익은 주로 구독 계약에 대한 선청구 금액을 반영한다.",
    },
]


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _load_tokenizer(cfg: Mapping[str, Any], *, local_files_only: bool = False) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("missing transformers; run `make set` first") from exc

    model_cfg = _get(cfg, "model", {})
    if not isinstance(model_cfg, Mapping):
        raise SystemExit("model config must be a mapping")
    return AutoTokenizer.from_pretrained(
        str(model_cfg["name_or_path"]),
        revision=str(model_cfg.get("tokenizer_revision", "main")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=local_files_only,
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return text_token_ids(tokenizer, text, add_special_tokens=False)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(_token_ids(tokenizer, text))


def _fit_text_to_tokens(
    tokenizer: Any,
    sentences: list[str],
    *,
    target_tokens: int,
    prefix: str = "",
) -> str:
    if target_tokens <= 0:
        raise SystemExit("target token count must be > 0")
    text = prefix.strip()
    cursor = 0
    while _token_count(tokenizer, text) < target_tokens:
        sentence = sentences[cursor % len(sentences)]
        text = f"{text} {sentence}".strip()
        cursor += 1

    for _ in range(12):
        ids = _token_ids(tokenizer, text)
        if len(ids) == target_tokens:
            return text.strip()
        if len(ids) > target_tokens:
            text = text_decode(tokenizer, ids[:target_tokens], skip_special_tokens=True).strip()
        else:
            sentence = sentences[cursor % len(sentences)]
            text = f"{text} {sentence}".strip()
            cursor += 1
    ids = _token_ids(tokenizer, text)
    if len(ids) > target_tokens:
        return text_decode(tokenizer, ids[:target_tokens], skip_special_tokens=True).strip()
    return text.strip()


def _template_group(cfg: Mapping[str, Any]) -> str:
    return "instruct_templates" if bool(_get(cfg, "model.use_hf_chat_template", False)) else "base_templates"


def _render_with_template(
    *,
    cfg: Mapping[str, Any],
    template_cfg: Mapping[str, Any],
    template: Mapping[str, Any],
    source: str,
    row_id: str,
) -> Any:
    group = _template_group(cfg)
    scoped_template_cfg = dict(template_cfg)
    scoped_template_cfg[group] = [dict(template)]
    return render_student_prompt(
        template_cfg=scoped_template_cfg,
        prompt_cfg=_get(cfg, "prompts", {}),
        model_cfg=_get(cfg, "model", {}),
        source=source,
        row_id=row_id,
        subset_idx=0,
    )


def _rank_templates_by_prompt_tokens(
    *,
    cfg: Mapping[str, Any],
    template_cfg: Mapping[str, Any],
    tokenizer: Any,
    source: str,
) -> list[tuple[int, Mapping[str, Any]]]:
    group = _template_group(cfg)
    templates = template_cfg.get(group, [])
    if not isinstance(templates, list) or not templates:
        raise SystemExit(f"missing smoke templates for {group}")
    ranked: list[tuple[int, Mapping[str, Any]]] = []
    for index, template in enumerate(templates):
        if not isinstance(template, Mapping):
            continue
        rendered = _render_with_template(
            cfg=cfg,
            template_cfg=template_cfg,
            template=template,
            source=source,
            row_id=f"smoke_template_rank_{index:03d}",
        )
        ranked.append((_token_count(tokenizer, rendered.text), template))
    return sorted(ranked, key=lambda item: (item[0], str(item[1].get("id", ""))), reverse=True)


def _build_max_context_sft_rows(
    *,
    cfg: Mapping[str, Any],
    template_cfg: Mapping[str, Any],
    tokenizer: Any,
    row_count: int,
    source_tokens: int,
    target_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _fit_text_to_tokens(
        tokenizer,
        SOURCE_SENTENCES,
        target_tokens=source_tokens,
        prefix="Smoke max context source.",
    )
    target = _fit_text_to_tokens(
        tokenizer,
        TARGET_SENTENCES,
        target_tokens=target_tokens,
        prefix="스모크 최대 컨텍스트 번역.",
    )
    ranked_templates = _rank_templates_by_prompt_tokens(
        cfg=cfg,
        template_cfg=template_cfg,
        tokenizer=tokenizer,
        source=source,
    )
    max_seq_length = _default_max_seq_length(cfg)
    training_cfg = _get(cfg, "training", {})
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}

    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for idx in range(row_count):
        _prompt_tokens, template = ranked_templates[idx % len(ranked_templates)]
        row_id = f"smoke_max_context_{idx:03d}"
        rendered = _render_with_template(
            cfg=cfg,
            template_cfg=template_cfg,
            template=template,
            source=source,
            row_id=row_id,
        )
        row = {
            "id": row_id,
            "order_idx": idx,
            "source": source,
            "target": target,
            "response": target,
            "prompt": rendered.text,
            "prompt_template_id": rendered.template_id,
            "prompt_template_group": rendered.template_group,
            "prompt_template_hash": rendered.template_hash,
            "chat_template_applied": rendered.chat_template_applied,
            "source_tokens": _token_count(tokenizer, source),
            "target_text_tokens": _token_count(tokenizer, target),
            "metadata": {"smoke_kind": "max_context_sft"},
        }
        tokenized = _tokenize_prompt_completion(
            tokenizer=tokenizer,
            row=row,
            max_seq_length=max_seq_length,
            append_eos_token=bool(training_cfg.get("append_eos_token", True)),
            prevent_truncation=True,
            response_only_loss=bool(training_cfg.get("response_only_loss", True)),
        )
        rows.append(row)
        stats.append(
            {
                "id": row_id,
                "template_id": rendered.template_id,
                "chat_template_applied": rendered.chat_template_applied,
                "source_tokens": row["source_tokens"],
                "target_text_tokens": row["target_text_tokens"],
                "prompt_token_count": tokenized["prompt_token_count"],
                "completion_token_count": tokenized["completion_token_count"],
                "supervised_token_count": tokenized["supervised_token_count"],
                "total_token_count": len(tokenized["input_ids"]),
                "max_seq_length": max_seq_length,
                "remaining_tokens": max_seq_length - len(tokenized["input_ids"]),
            }
        )
    return rows, stats


def _build_cycle_rows(tokenizer: Any, *, row_count: int, source_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for idx in range(row_count):
        if idx % 4 == 0:
            source = _fit_text_to_tokens(
                tokenizer,
                SOURCE_SENTENCES,
                target_tokens=source_tokens,
                prefix=f"Smoke cycle long source {idx}.",
            )
        else:
            source = SOURCE_SENTENCES[idx % len(SOURCE_SENTENCES)]
        target = TARGET_SENTENCES[idx % len(TARGET_SENTENCES)]
        source_count = _token_count(tokenizer, source)
        row = {
            "id": f"smoke_cycle_{idx:03d}",
            "source": source,
            "target": target,
            "source_tokens": source_count,
            "metadata": {
                "smoke_kind": "cycle",
                "smoke_row_index": idx,
                "long_context": bool(idx % 4 == 0),
            },
        }
        rows.append(row)
        stats.append(
            {
                "id": row["id"],
                "source_tokens": source_count,
                "target_tokens": _token_count(tokenizer, target),
                "long_context": row["metadata"]["long_context"],
            }
        )
    return rows, stats


def _build_val_rows(tokenizer: Any, *, row_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for idx in range(row_count):
        base = VAL_ROWS[idx % len(VAL_ROWS)]
        row = {
            "id": f"smoke_val_{idx:03d}",
            "source": base["source"],
            "target": base["target"],
            "source_tokens": _token_count(tokenizer, base["source"]),
            "metadata": {"smoke_kind": "val", "smoke_row_index": idx},
        }
        rows.append(row)
        stats.append(
            {
                "id": row["id"],
                "source_tokens": row["source_tokens"],
                "target_tokens": _token_count(tokenizer, row["target"]),
            }
        )
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DQS smoke-run datasets.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", default="data/smoke")
    parser.add_argument("--max-context-rows", type=int, default=4)
    parser.add_argument("--cycle-subset-size", type=int, default=4)
    parser.add_argument("--cycle-subsets", type=int, default=2)
    parser.add_argument("--val-rows", type=int, default=4)
    parser.add_argument("--source-tokens", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = compose_config(args.config, overrides=args.override)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(output_dir / "effective_config.yaml", cfg)

    tokenizer = _load_tokenizer(cfg, local_files_only=args.local_files_only)
    template_cfg = load_student_templates(str(_get(cfg, "prompts.student_templates_path")))
    source_tokens = int(args.source_tokens or _get(cfg, "data.max_input_tokens", 1280))
    target_tokens = int(args.target_tokens or _get(cfg, "data.max_output_tokens", 1500))
    cycle_rows_count = max(1, int(args.cycle_subset_size) * max(1, int(args.cycle_subsets)))

    max_rows, max_stats = _build_max_context_sft_rows(
        cfg=cfg,
        template_cfg=template_cfg,
        tokenizer=tokenizer,
        row_count=max(1, int(args.max_context_rows)),
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )
    cycle_rows, cycle_stats = _build_cycle_rows(
        tokenizer,
        row_count=cycle_rows_count,
        source_tokens=source_tokens,
    )
    val_rows, val_stats = _build_val_rows(tokenizer, row_count=max(1, int(args.val_rows)))

    max_path = output_dir / "max_context_sft.jsonl"
    cycle_path = output_dir / "cycle.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(max_path, max_rows)
    write_jsonl(cycle_path, cycle_rows)
    write_jsonl(val_path, val_rows)

    summary = {
        "model": _get(cfg, "model.name_or_path"),
        "model_variant": _get(cfg, "model.variant"),
        "template_group": _template_group(cfg),
        "max_seq_length": _default_max_seq_length(cfg),
        "max_input_tokens": _get(cfg, "data.max_input_tokens"),
        "max_output_tokens": _get(cfg, "data.max_output_tokens"),
        "prompt_overhead_tokens": _get(cfg, "training.prompt_overhead_tokens"),
        "files": {
            "max_context_sft": str(max_path),
            "cycle": str(cycle_path),
            "val": str(val_path),
            "effective_config": str(output_dir / "effective_config.yaml"),
        },
        "max_context_sft": max_stats,
        "cycle": cycle_stats,
        "val": val_stats,
    }
    stats_path = output_dir / "smoke_stats.json"
    stats_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
