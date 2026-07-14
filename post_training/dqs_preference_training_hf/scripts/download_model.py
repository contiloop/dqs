#!/usr/bin/env python3
"""Explicitly download and verify the exact full-SFT model before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"[0-9a-fA-F]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ModelFileSpec:
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    repo_type: str
    revision: str
    remote_dir: str
    local_dir: Path
    files: tuple[ModelFileSpec, ...]
    expected_sft: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="perform an offline validation of an already downloaded model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="local model directory; defaults to manifest.sft_model.local_dir",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an incomplete or corrupt known-file-only model directory",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_relative(raw: str, *, field: str) -> str:
    path = Path(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {raw!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"{field} must not be empty")
    return normalized


def _bundle_path(root: Path, raw: str, *, field: str) -> Path:
    relative = _safe_relative(raw, field=field)
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"{field} escapes the bundle: {raw!r}")
    return resolved


def load_model_spec(root: Path = ROOT) -> tuple[dict[str, Any], ModelSpec]:
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema_version") != "dqs_preference_release.v1":
        raise ValueError("unsupported release manifest schema")
    raw = manifest.get("sft_model")
    if not isinstance(raw, dict):
        raise ValueError("manifest.sft_model is missing")

    repo_id = str(raw.get("repo_id", "")).strip()
    if not repo_id or repo_id.count("/") != 1:
        raise ValueError("manifest.sft_model.repo_id is invalid")
    repo_type = str(raw.get("repo_type", ""))
    if repo_type != "dataset":
        raise ValueError("manifest.sft_model.repo_type must be dataset")
    revision = str(raw.get("revision", ""))
    if HEX40.fullmatch(revision) is None:
        raise ValueError("manifest SFT model revision must be an exact 40-hex commit")
    remote_dir = _safe_relative(str(raw.get("remote_dir", "")), field="sft_model.remote_dir")
    local_dir = _bundle_path(root, str(raw.get("local_dir", "")), field="sft_model.local_dir")

    raw_files = raw.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("manifest.sft_model.files is missing")
    files: list[ModelFileSpec] = []
    for relative, metadata in sorted(raw_files.items()):
        safe_relative = _safe_relative(str(relative), field="sft_model.files")
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid model metadata for {safe_relative}")
        digest = str(metadata.get("sha256", ""))
        size = int(metadata.get("size", -1))
        if HEX64.fullmatch(digest) is None or size <= 0:
            raise ValueError(f"invalid immutable model metadata for {safe_relative}")
        files.append(ModelFileSpec(safe_relative, digest, size))

    total_size = sum(item.size for item in files)
    if int(raw.get("total_size_bytes", -1)) != total_size:
        raise ValueError("manifest SFT model total_size_bytes is inconsistent")
    if int(raw.get("file_count", -1)) != len(files):
        raise ValueError("manifest SFT model file_count is inconsistent")

    expected_sft = raw.get("expected_sft")
    if not isinstance(expected_sft, dict):
        raise ValueError("manifest.sft_model.expected_sft is missing")
    expected = {
        "run_id": str(expected_sft.get("run_id", "")),
        "subset_idx": int(expected_sft.get("subset_idx", -1)),
        "global_step": int(expected_sft.get("global_step", -1)),
        "tuning_mode": str(expected_sft.get("tuning_mode", "")),
    }
    if not expected["run_id"] or expected["subset_idx"] < 0 or expected["global_step"] < 0:
        raise ValueError("manifest SFT provenance values are invalid")
    if expected["tuning_mode"] != "full":
        raise ValueError("manifest SFT model must require full tuning")

    objectives = manifest.get("objectives")
    if not isinstance(objectives, dict) or not objectives:
        raise ValueError("manifest objectives are missing")
    objective_expected = {
        (
            str(value.get("expected_sft", {}).get("run_id", "")),
            int(value.get("expected_sft", {}).get("subset_idx", -1)),
            int(value.get("expected_sft", {}).get("global_step", -1)),
        )
        for value in objectives.values()
        if isinstance(value, dict)
    }
    model_expected = (expected["run_id"], expected["subset_idx"], expected["global_step"])
    if objective_expected != {model_expected}:
        raise ValueError("SFT model source and objective provenance requirements disagree")

    return manifest, ModelSpec(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        remote_dir=remote_dir,
        local_dir=local_dir,
        files=tuple(files),
        expected_sft=expected,
    )


def _inventory(path: Path) -> set[str]:
    inventory: set[str] = set()
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"symlinks are forbidden in the SFT model directory: {candidate}")
        if candidate.is_file():
            inventory.add(candidate.relative_to(path).as_posix())
        elif not candidate.is_dir():
            raise ValueError(f"unsupported filesystem entry in SFT model: {candidate}")
    return inventory


def validate_model_dir(path: Path, spec: ModelSpec) -> dict[str, Any]:
    model_dir = Path(os.path.abspath(path.expanduser()))
    if model_dir.is_symlink():
        raise ValueError(f"SFT model directory must not be a symlink: {model_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"SFT final model is missing: {model_dir}; run `make download-model` first"
        )
    expected_files = {item.relative_path for item in spec.files}
    observed_files = _inventory(model_dir)
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        unexpected = sorted(observed_files - expected_files)
        raise ValueError(
            f"SFT final model inventory mismatch: missing={missing}, unexpected={unexpected}"
        )

    verified: dict[str, dict[str, Any]] = {}
    for item in spec.files:
        file_path = model_dir / item.relative_path
        observed_size = file_path.stat().st_size
        if observed_size != item.size:
            raise ValueError(
                f"SFT model size mismatch for {item.relative_path}: "
                f"observed={observed_size} expected={item.size}"
            )
        observed_sha = sha256_file(file_path)
        if observed_sha != item.sha256:
            raise ValueError(
                f"SFT model SHA256 mismatch for {item.relative_path}: "
                f"observed={observed_sha} expected={item.sha256}"
            )
        verified[item.relative_path] = {"size": observed_size, "sha256": observed_sha}

    marker = load_json(model_dir / "dqs_stage_model.json")
    observed_sft = {
        "run_id": str(marker.get("run_id", "")),
        "subset_idx": int(marker.get("subset_idx", -1)),
        "global_step": int(marker.get("global_step", -1)),
        "tuning_mode": str(marker.get("tuning_mode", "")),
    }
    if observed_sft != spec.expected_sft:
        raise ValueError(
            f"SFT provenance mismatch: observed={observed_sft!r} "
            f"expected={spec.expected_sft!r}"
        )
    return {
        "path": str(model_dir),
        "file_count": len(verified),
        "total_size_bytes": sum(value["size"] for value in verified.values()),
        "expected_sft": observed_sft,
    }


def _assert_replaceable(path: Path, spec: ModelSpec) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"refusing to replace a non-directory or symlink: {path}")
    expected = {item.relative_path for item in spec.files}
    observed = _inventory(path)
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(
            "refusing to delete an existing directory with unknown files; "
            f"move it aside manually first: unexpected={unexpected}"
        )


def _downloaded_file(snapshot: Path, spec: ModelSpec, relative: str) -> Path:
    path = (snapshot / spec.remote_dir / relative).resolve()
    if snapshot.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"downloaded SFT model file is missing: {relative}")
    return path


def download_model(
    *,
    root: Path = ROOT,
    output: Path | None = None,
    replace: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("--workers must be positive")
    _, spec = load_model_spec(root)
    target = Path(os.path.abspath((output or spec.local_dir).expanduser()))
    if target.is_symlink():
        raise ValueError(f"SFT model output must not be a symlink: {target}")
    if target.exists():
        try:
            result = validate_model_dir(target, spec)
            return {
                "status": "ok",
                "mode": "existing-local-model",
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "model": result,
            }
        except (OSError, ValueError):
            if not replace:
                raise ValueError(
                    "existing SFT model is invalid; rerun with "
                    "`make download-model MODEL_REPLACE=1` to replace known files"
                )
            _assert_replaceable(target, spec)

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("missing huggingface_hub; run `make set` first") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    allow_patterns = [f"{spec.remote_dir}/{item.relative_path}" for item in spec.files]
    with tempfile.TemporaryDirectory(prefix=".download-model-", dir=target.parent) as temp_dir:
        staging_root = Path(temp_dir)
        snapshot = Path(
            snapshot_download(
                repo_id=spec.repo_id,
                repo_type=spec.repo_type,
                revision=spec.revision,
                local_dir=str(staging_root / "snapshot"),
                allow_patterns=allow_patterns,
                max_workers=workers,
            )
        )
        install = staging_root / "install"
        install.mkdir()
        for item in spec.files:
            source = _downloaded_file(snapshot, spec, item.relative_path)
            destination = install / item.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        staged_result = validate_model_dir(install, spec)

        backup = staging_root / "previous"
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(install, target)
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    result = validate_model_dir(target, spec)
    if result != {**staged_result, "path": str(target)}:
        raise RuntimeError("installed SFT model differs from the verified staging directory")
    return {
        "status": "ok",
        "mode": "explicit-download-then-local",
        "repo_id": spec.repo_id,
        "repo_type": spec.repo_type,
        "revision": spec.revision,
        "remote_dir": spec.remote_dir,
        "model": result,
    }


def main() -> None:
    args = parse_args()
    _, spec = load_model_spec()
    output = Path(os.path.abspath((args.output or spec.local_dir).expanduser()))
    result = (
        {
            "status": "ok",
            "mode": "offline-local-validation",
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "model": validate_model_dir(output, spec),
        }
        if args.check
        else download_model(output=output, replace=args.replace, workers=args.workers)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
