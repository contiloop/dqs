#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping

from config_loader import compose_config


DEFAULT_IGNORE_PATTERNS = (
    ".DS_Store",
    "__pycache__/*",
    "*/__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".ipynb_checkpoints/*",
    "*/.ipynb_checkpoints/*",
)


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    current = float(value)
    for unit in units:
        if current < 1024 or unit == units[-1]:
            return f"{current:.1f} {unit}" if unit != "B" else f"{value} B"
        current /= 1024
    return f"{value} B"


def _matches_any(path: Path, patterns: Iterable[str]) -> bool:
    rel = path.as_posix()
    name = path.name
    return any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _iter_upload_files(run_dir: Path, ignore_patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)
        if _matches_any(rel, ignore_patterns):
            continue
        files.append(path)
    return sorted(files)


def _resolve_path_in_repo(path_in_repo: str | None, run_id: str) -> str | None:
    raw = (path_in_repo or run_id).strip()
    if raw in {"", "."}:
        return None
    return raw.strip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a complete DQS run folder to a Hugging Face dataset repo.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    parser.add_argument("--repo", required=True, help="Hugging Face dataset repo id, e.g. username/dqs-runs")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--path-in-repo")
    parser.add_argument("--commit-message")
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--create-repo", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--delete-existing-path", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.override)
    if args.run_id:
        overrides.append(f"run.id={args.run_id}")
    cfg = compose_config(args.config, overrides=overrides)

    run_dir = Path(args.run_dir or str(_get(cfg, "paths.artifact_root"))).expanduser()
    run_id = str(_get(cfg, "run.id"))
    if args.run_dir and not args.run_id:
        run_id = run_dir.name
    path_in_repo = _resolve_path_in_repo(args.path_in_repo, run_id)
    ignore_patterns = [*DEFAULT_IGNORE_PATTERNS, *args.ignore_pattern]

    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")

    files = _iter_upload_files(run_dir, ignore_patterns)
    total_bytes = sum(path.stat().st_size for path in files)
    if not files:
        raise SystemExit(f"run directory has no uploadable files: {run_dir}")

    destination = f"{args.repo}/{path_in_repo}" if path_in_repo else args.repo
    print(f"upload-run: run_id={run_id}")
    print(f"upload-run: source={run_dir}")
    print(f"upload-run: destination=dataset:{destination}")
    print(f"upload-run: files={len(files)} bytes={total_bytes} ({_format_bytes(total_bytes)})")
    if args.delete_existing_path:
        if path_in_repo is None:
            raise SystemExit("delete-existing-path requires a non-root path_in_repo/run_id")
        print(f"upload-run: delete_existing_path={path_in_repo}/**")

    if args.dry_run:
        for path in files[:20]:
            print(f"  {path.relative_to(run_dir)}")
        if len(files) > 20:
            print(f"  ... {len(files) - 20} more files")
        print("upload-run: dry run complete")
        return

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit("missing huggingface_hub; run `make set` first") from exc

    api = HfApi()
    if args.create_repo:
        api.create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    commit_info = api.upload_folder(
        folder_path=run_dir,
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message or f"Upload DQS run {run_id}",
        ignore_patterns=ignore_patterns,
        delete_patterns=f"{path_in_repo}/**" if args.delete_existing_path and path_in_repo else None,
    )
    commit_url = getattr(commit_info, "commit_url", None)
    if commit_url:
        print(f"upload-run: commit={commit_url}")
    else:
        print("upload-run: upload complete")


if __name__ == "__main__":
    main()
