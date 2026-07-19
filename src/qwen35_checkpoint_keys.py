#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


QWEN35_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("model.language_model.language_model.language_model.", "model.language_model."),
    ("model.language_model.visual.", "model.visual."),
)


def normalize_qwen35_key(key: str) -> str:
    for bad_prefix, good_prefix in QWEN35_KEY_REPLACEMENTS:
        if key.startswith(bad_prefix):
            return good_prefix + key[len(bad_prefix):]
    return key


def is_bad_qwen35_key(key: str) -> bool:
    return normalize_qwen35_key(key) != key


def _model_safetensor_files(model_dir: Path) -> list[Path]:
    files = []
    for path in sorted(model_dir.glob("*.safetensors")):
        if path.name == "adapter_model.safetensors":
            continue
        files.append(path)
    return files


def qwen35_checkpoint_keys(model_dir: str | Path) -> set[str]:
    """Return all full-model safetensor keys stored in ``model_dir``."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    files = _model_safetensor_files(model_dir)
    if not files:
        raise SystemExit(f"no model safetensors files found in {model_dir}")

    keys: set[str] = set()
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in keys:
                    raise SystemExit(f"duplicate tensor key across checkpoint shards in {model_dir}: {key}")
                keys.add(key)
    return keys


def qwen35_checkpoint_compatibility(
    runtime_keys: Iterable[str],
    model_dir: str | Path,
) -> dict[str, Any]:
    """Compare a loaded Qwen3.5 model namespace with an on-disk checkpoint."""
    runtime = {str(key) for key in runtime_keys}
    checkpoint = qwen35_checkpoint_keys(model_dir)
    matched = runtime & checkpoint
    runtime_only = runtime - checkpoint
    checkpoint_only = checkpoint - runtime
    runtime_coverage = len(matched) / len(runtime) if runtime else 0.0
    checkpoint_coverage = len(matched) / len(checkpoint) if checkpoint else 0.0
    return {
        "model_dir": str(model_dir),
        "runtime_key_count": len(runtime),
        "checkpoint_key_count": len(checkpoint),
        "matched_key_count": len(matched),
        "runtime_only_count": len(runtime_only),
        "checkpoint_only_count": len(checkpoint_only),
        "runtime_coverage": runtime_coverage,
        "checkpoint_coverage": checkpoint_coverage,
        "runtime_only_examples": sorted(runtime_only)[:20],
        "checkpoint_only_examples": sorted(checkpoint_only)[:20],
    }


def assert_qwen35_checkpoint_compatible(
    runtime_keys: Iterable[str],
    model_dir: str | Path,
    *,
    min_runtime_coverage: float = 0.99,
) -> dict[str, Any]:
    """Fail before resume when a Qwen3.5 checkpoint cannot load its full weights."""
    summary = qwen35_checkpoint_compatibility(runtime_keys, model_dir)
    if (
        summary["checkpoint_only_count"] > 0
        or summary["runtime_coverage"] < min_runtime_coverage
    ):
        raise SystemExit(
            "incompatible Qwen3.5 resume checkpoint; refusing partial/zero-weight load: "
            f"checkpoint={model_dir} "
            f"matched={summary['matched_key_count']} "
            f"runtime={summary['runtime_key_count']} "
            f"checkpoint_keys={summary['checkpoint_key_count']} "
            f"runtime_coverage={summary['runtime_coverage']:.6f} "
            f"checkpoint_only={summary['checkpoint_only_count']} "
            f"checkpoint_only_examples={summary['checkpoint_only_examples']}"
        )
    return summary


def _replace_file(path: Path, tmp_path: Path, *, backup: bool) -> str | None:
    backup_path = path.with_suffix(path.suffix + ".badkeys.bak")
    if backup:
        if backup_path.exists():
            raise SystemExit(f"backup already exists: {backup_path}")
        os.replace(path, backup_path)
        os.replace(tmp_path, path)
        return str(backup_path)
    path.unlink()
    os.replace(tmp_path, path)
    return None


def _load_key_summary(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    bad_keys = [key for key in keys if is_bad_qwen35_key(key)]
    return {
        "path": str(path),
        "key_count": len(keys),
        "bad_key_count": len(bad_keys),
        "bad_key_examples": bad_keys[:20],
    }


def _normalize_safetensors_file(path: Path, *, backup: bool, dry_run: bool) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    before = _load_key_summary(path)
    if before["bad_key_count"] <= 0:
        return {
            **before,
            "changed": False,
            "renamed_key_count": 0,
            "backup_path": None,
        }
    if dry_run:
        return {
            **before,
            "changed": False,
            "would_change": True,
            "renamed_key_count": before["bad_key_count"],
            "backup_path": None,
        }

    tensors = {}
    metadata = None
    renamed = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            new_key = normalize_qwen35_key(key)
            if new_key != key:
                renamed += 1
            if new_key in tensors:
                raise SystemExit(f"normalizing {path} would collide on tensor key: {new_key}")
            tensors[new_key] = handle.get_tensor(key)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    save_file(tensors, str(tmp_path), metadata=metadata)
    backup_path = _replace_file(path, tmp_path, backup=backup)
    after = _load_key_summary(path)
    return {
        **after,
        "changed": True,
        "renamed_key_count": renamed,
        "backup_path": backup_path,
    }


def _normalize_index_file(model_dir: Path, *, backup: bool, dry_run: bool) -> dict[str, Any] | None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        return {
            "path": str(index_path),
            "changed": False,
            "renamed_key_count": 0,
        }

    new_weight_map = {}
    renamed = 0
    for key, value in weight_map.items():
        new_key = normalize_qwen35_key(str(key))
        if new_key != key:
            renamed += 1
        if new_key in new_weight_map:
            raise SystemExit(f"normalizing {index_path} would collide on tensor key: {new_key}")
        new_weight_map[new_key] = value

    if renamed <= 0:
        return {
            "path": str(index_path),
            "changed": False,
            "renamed_key_count": 0,
        }
    if dry_run:
        return {
            "path": str(index_path),
            "changed": False,
            "would_change": True,
            "renamed_key_count": renamed,
        }

    data["weight_map"] = new_weight_map
    backup_path = index_path.with_suffix(index_path.suffix + ".badkeys.bak")
    if backup:
        if backup_path.exists():
            raise SystemExit(f"backup already exists: {backup_path}")
        os.replace(index_path, backup_path)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, index_path)
    return {
        "path": str(index_path),
        "changed": True,
        "renamed_key_count": renamed,
        "backup_path": str(backup_path) if backup else None,
    }


def normalize_qwen35_checkpoint_keys(
    model_dir: str | Path,
    *,
    backup: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise SystemExit(f"missing model dir: {model_dir}")

    files = _model_safetensor_files(model_dir)
    if not files:
        return {
            "model_dir": str(model_dir),
            "status": "skipped",
            "reason": "no model safetensors files found",
            "files": [],
            "bad_key_count_after": 0,
        }

    file_summaries = [
        _normalize_safetensors_file(path, backup=backup, dry_run=dry_run)
        for path in files
    ]
    index_summary = _normalize_index_file(model_dir, backup=backup, dry_run=dry_run)
    bad_after = sum(int(item.get("bad_key_count", 0) or 0) for item in file_summaries)
    renamed = sum(int(item.get("renamed_key_count", 0) or 0) for item in file_summaries)
    return {
        "model_dir": str(model_dir),
        "status": "ok" if bad_after == 0 or dry_run else "bad_keys_remaining",
        "dry_run": dry_run,
        "files": file_summaries,
        "index": index_summary,
        "renamed_key_count": renamed,
        "bad_key_count_after": bad_after,
    }


def assert_no_bad_qwen35_checkpoint_keys(model_dir: str | Path) -> None:
    summary = normalize_qwen35_checkpoint_keys(model_dir, dry_run=True)
    bad = sum(int(item.get("bad_key_count", 0) or 0) for item in summary.get("files", []))
    if bad:
        examples = []
        for item in summary.get("files", []):
            examples.extend(item.get("bad_key_examples", []))
        preview = "\n".join(str(key) for key in examples[:20])
        raise SystemExit(f"bad Qwen3.5 checkpoint keys remain in {model_dir}: count={bad}\n{preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair Qwen3.5 checkpoint keys affected by transformers save mapping bugs.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = normalize_qwen35_checkpoint_keys(
        args.model_dir,
        backup=bool(args.backup),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    if not args.dry_run and int(summary.get("bad_key_count_after", 0) or 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
