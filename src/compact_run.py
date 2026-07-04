#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config


COMPACT_PATTERNS = (
    "subsets/subset_*/runtime_io/qe-selection.input.jsonl",
    "subsets/subset_*/runtime_io/qe-selection.output.jsonl",
    "subsets/subset_*/filter_blocked_selection.jsonl",
    "subsets/subset_*/teacher_requests.jsonl",
    "subsets/subset_*/teacher_responses.raw.jsonl",
    "subsets/subset_*/teacher_parsed.jsonl",
    "subsets/subset_*/teacher_rejected.jsonl",
    "eval/*/eval_requests.jsonl",
    "eval/*/eval_translations.jsonl",
    "eval/*/eval_filtered.jsonl",
    "eval/*/*/eval_requests.jsonl",
    "eval/*/*/eval_translations.jsonl",
    "eval/*/*/eval_filtered.jsonl",
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

    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in COMPACT_PATTERNS:
        for path in run_dir.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)

    total_bytes = 0
    for path in sorted(paths):
        total_bytes += _remove_path(path, dry_run=args.dry_run)

    summary = {
        "run_dir": str(run_dir),
        "removed_paths": len(paths),
        "removed_bytes": total_bytes,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
