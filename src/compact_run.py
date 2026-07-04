#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config


SUBSET_COMPACT_RULES = (
    (
        "student_records.jsonl",
        (
            "input.jsonl",
            "student_translations.jsonl",
            "student_filtered.jsonl",
            "qe_scores.jsonl",
            "selected_for_teacher.jsonl",
            "filter_blocked_selection.jsonl",
            "runtime_io/infer-student.input.jsonl",
            "runtime_io/infer-student.output.jsonl",
            "runtime_io/qe-selection.input.jsonl",
            "runtime_io/qe-selection.output.jsonl",
        ),
    ),
    (
        "qe_scores.jsonl",
        (
            "runtime_io/qe-selection.input.jsonl",
            "runtime_io/qe-selection.output.jsonl",
        ),
    ),
    (
        "teacher_artifacts.jsonl",
        (
            "teacher_requests.jsonl",
            "teacher_responses.raw.jsonl",
            "teacher_parsed.jsonl",
            "teacher_rejected.jsonl",
        ),
    ),
)

EVAL_COMPACT_RULES = (
    (
        "eval_records.jsonl",
        (
            "eval_requests.jsonl",
            "eval_translations.jsonl",
            "eval_filtered.jsonl",
            "eval_scores.jsonl",
        ),
    ),
)


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _remove_path(path: Path, *, dry_run: bool) -> int:
    if not path.exists():
        return 0
    if dry_run:
        return path.stat().st_size if path.is_file() else 0
    size = path.stat().st_size if path.is_file() else 0
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return size


def _compact_rooted_rules(root: Path, rules: tuple[tuple[str, tuple[str, ...]], ...], *, dry_run: bool) -> tuple[int, int]:
    removed_paths = 0
    removed_bytes = 0
    for sentinel, rel_paths in rules:
        if not (root / sentinel).exists():
            continue
        for rel_path in rel_paths:
            path = root / rel_path
            if not path.exists():
                continue
            removed_paths += 1
            removed_bytes += _remove_path(path, dry_run=dry_run)
    return removed_paths, removed_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove nonessential DQS run debug artifacts.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.override)
    if args.run_id:
        overrides.append(f"run.id={args.run_id}")
    cfg = compose_config(args.config, overrides=overrides)
    run_dir = Path(args.run_dir or str(_get(cfg, "paths.artifact_root"))).expanduser()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")

    removed_paths = 0
    total_bytes = 0
    for subset_dir in sorted(run_dir.glob("subsets/subset_*")):
        if not subset_dir.is_dir():
            continue
        count, size = _compact_rooted_rules(subset_dir, SUBSET_COMPACT_RULES, dry_run=args.dry_run)
        removed_paths += count
        total_bytes += size

    eval_dirs = sorted({path.parent for path in run_dir.glob("eval/**/eval_summary.json")})
    for eval_dir in eval_dirs:
        count, size = _compact_rooted_rules(eval_dir, EVAL_COMPACT_RULES, dry_run=args.dry_run)
        removed_paths += count
        total_bytes += size

    summary = {
        "run_dir": str(run_dir),
        "removed_paths": removed_paths,
        "removed_bytes": total_bytes,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
