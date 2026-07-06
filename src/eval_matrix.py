#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

from io_utils import write_jsonl
from progress import progress


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"matrix config must be a YAML mapping: {path}")
    return payload


def _as_model_list(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise SystemExit("matrix config `models` must be a list")
    out: list[dict[str, Any]] = []
    for idx, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise SystemExit(f"models[{idx}] must be a mapping")
        slug = str(model.get("slug", "")).strip()
        model_id = str(model.get("model", "")).strip()
        if not slug or not model_id:
            raise SystemExit(f"models[{idx}] requires slug and model")
        out.append(dict(model))
    return out


def _selected_models(models: list[dict[str, Any]], selected: str | None) -> list[dict[str, Any]]:
    if selected:
        wanted = [item.strip() for item in selected.split(",") if item.strip()]
        by_slug = {str(model["slug"]): model for model in models}
        missing = sorted(set(wanted) - set(by_slug))
        if missing:
            raise SystemExit(f"unknown model slug(s): {', '.join(missing)}")
        return [by_slug[slug] for slug in wanted]
    return [model for model in models if bool(model.get("enabled", True))]


def _override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    return f"{key}={value}"


def _metric_value(summary: Mapping[str, Any], metric_id: str) -> Any:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_id)
    if not isinstance(metric, Mapping):
        return None
    if "score" in metric:
        return metric.get("score")
    return metric.get("mean")


def _cost_usd(summary: Mapping[str, Any], model: Mapping[str, Any]) -> float | None:
    usage = summary.get("usage", {})
    if not isinstance(usage, Mapping):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, (int, float)) or not isinstance(completion_tokens, (int, float)):
        return None
    prompt_price = model.get("prompt_price_per_mtok_usd")
    completion_price = model.get("completion_price_per_mtok_usd")
    if not isinstance(prompt_price, (int, float)) or not isinstance(completion_price, (int, float)):
        return None
    return (float(prompt_tokens) / 1_000_000 * float(prompt_price)) + (
        float(completion_tokens) / 1_000_000 * float(completion_price)
    )


def _summary_row(
    *,
    model: Mapping[str, Any],
    summary: Mapping[str, Any],
    output_dir: Path,
    metric_ids: list[str],
) -> dict[str, Any]:
    usage = summary.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    row: dict[str, Any] = {
        "slug": model.get("slug"),
        "model": model.get("model"),
        "provider": model.get("provider", "openrouter"),
        "output_dir": str(output_dir),
        "rows": summary.get("rows"),
        "generation_ok_rows": summary.get("generation_ok_rows"),
        "filter_pass_rows": summary.get("filter_pass_rows"),
        "filter_fail_rows": summary.get("filter_fail_rows"),
        "filter_fail_ratio": summary.get("filter_fail_ratio"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": _cost_usd(summary, model),
        "dry_run": summary.get("dry_run"),
    }
    for metric_id in metric_ids:
        row[metric_id] = _metric_value(summary, metric_id)
    return row


def _write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "summary.jsonl", rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _eval_cmd(
    *,
    args: argparse.Namespace,
    suite: Mapping[str, Any],
    openrouter: Mapping[str, Any],
    model: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    eval_config = str(suite.get("eval_config", args.eval_config))
    profile = str(args.profile or suite.get("eval_profile", "final"))
    metrics = str(args.metrics or suite.get("metrics", "bleu,chrf"))
    max_new_tokens = int(args.max_new_tokens or suite.get("max_new_tokens", 1024) or 1024)
    temperature = float(suite.get("temperature", 0.0) or 0.0)
    top_p = float(suite.get("top_p", 1.0) or 1.0)
    slug = str(model["slug"])
    model_id = str(model["model"])

    overrides: list[str] = [
        _override("eval", profile),
        _override("run.id", f"{suite.get('name', 'openrouter_api')}_{slug}"),
        _override("model.name_or_path", model_id),
        _override("model.family", model.get("family", "api")),
        _override("model.size", model.get("size", "unknown")),
        _override("model.variant", model.get("variant", "api")),
        _override("model.use_chat_messages", True),
        _override("model.use_hf_chat_template", False),
        _override("model.trust_remote_code", False),
        _override("model.enable_thinking", False),
        _override("model.require_thinking_control", False),
        _override("inference.backend", "openrouter"),
        _override("inference.openrouter.api_key_env", openrouter.get("api_key_env", "OPENROUTER_API_KEY")),
        _override(
            "inference.openrouter.base_url",
            openrouter.get("base_url", "https://openrouter.ai/api/v1/chat/completions"),
        ),
        _override("inference.openrouter.concurrency", model.get("concurrency", openrouter.get("concurrency", 4))),
        _override("inference.openrouter.timeout_s", openrouter.get("timeout_s", 120)),
        _override("inference.openrouter.max_retries", openrouter.get("max_retries", 5)),
        _override("inference.openrouter.retry_sleep_s", openrouter.get("retry_sleep_s", 2.0)),
        _override("eval.generation.max_new_tokens", model.get("max_new_tokens", max_new_tokens)),
        _override("eval.generation.temperature", model.get("temperature", temperature)),
        _override("eval.generation.top_p", model.get("top_p", top_p)),
        _override("logging.wandb.enabled", False),
    ]
    reasoning_effort = model.get("reasoning_effort", suite.get("reasoning_effort"))
    if reasoning_effort:
        overrides.append(_override("inference.openrouter.reasoning_effort", reasoning_effort))
    app_name = openrouter.get("app_name")
    if app_name:
        overrides.append(_override("inference.openrouter.app_name", app_name))
    site_url = openrouter.get("site_url")
    if site_url:
        overrides.append(_override("inference.openrouter.site_url", site_url))

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("eval.py")),
        "--config",
        eval_config,
        "--model-path",
        model_id,
        "--output-dir",
        str(output_dir),
        "--metrics",
        metrics,
        "--skip-wandb-log",
    ]
    for override in overrides:
        cmd.extend(["--override", override])
    if args.data_path:
        cmd.extend(["--data-path", args.data_path])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")
    return cmd


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.matrix_config)
    payload = _load_yaml(config_path)
    suite = payload.get("suite", {})
    if not isinstance(suite, Mapping):
        raise SystemExit("matrix config `suite` must be a mapping")
    openrouter = payload.get("openrouter", {})
    if not isinstance(openrouter, Mapping):
        raise SystemExit("matrix config `openrouter` must be a mapping")
    models = _selected_models(_as_model_list(payload), args.models)
    output_root = Path(args.output_dir or str(suite.get("output_dir", "outputs/results/openrouter_api")))
    profile = str(args.profile or suite.get("eval_profile", "final"))
    metric_ids = [item.strip() for item in str(args.metrics or suite.get("metrics", "bleu,chrf")).split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    for model in models:
        slug = str(model["slug"])
        output_dir = output_root / slug / profile
        progress("eval matrix model start", slug=slug, model=model["model"], output_dir=output_dir)
        cmd = _eval_cmd(
            args=args,
            suite=suite,
            openrouter=openrouter,
            model=model,
            output_dir=output_dir,
        )
        subprocess.run(cmd, check=True)
        summary_path = output_dir / "eval_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(_summary_row(model=model, summary=summary, output_dir=output_dir, metric_ids=metric_ids))
        _write_summary(output_root, rows)
        progress("eval matrix model done", slug=slug, output_dir=output_dir)

    _write_summary(output_root, rows)
    print(json.dumps({"output_dir": str(output_root), "models": len(rows)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQS eval across an API model matrix.")
    parser.add_argument("--matrix-config", default="configs/eval_matrix/openrouter.yaml")
    parser.add_argument("--eval-config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model slugs to run.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", default=None, help="Comma-separated metric ids.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
