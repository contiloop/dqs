#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"invalid yaml object: {path}")
    return data


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download prepared DQS parquet data from Hugging Face.")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--repo", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--local-dir", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    cfg = load_yaml(Path(args.config))["data"]
    download_cfg = cfg.get("prepared_download", {})

    repo = args.repo or cfg["prepared_dataset_repo"]
    revision = args.revision or download_cfg.get("revision", "main")
    local_dir = Path(args.local_dir or download_cfg["local_dir"])
    workers = int(args.workers or download_cfg.get("workers", 16))
    allow_patterns = download_cfg.get("allow_patterns", ["*.parquet", "README.md"])

    print(f"repo={repo}")
    print(f"revision={revision}")
    print(f"local_dir={local_dir}")
    print(f"workers={workers}")
    print(f"allow_patterns={allow_patterns}")

    if args.dry_run:
        return

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit("missing huggingface_hub; run `make set` first") from exc

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=workers,
    )

    parquet_files = sorted(Path(path).rglob("*.parquet"))
    total_bytes = sum(p.stat().st_size for p in parquet_files)
    print(f"downloaded_dir={path}")
    print(f"parquet_files={len(parquet_files)}")
    print(f"parquet_bytes={total_bytes}")


if __name__ == "__main__":
    main()
