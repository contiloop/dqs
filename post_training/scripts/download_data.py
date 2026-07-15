#!/usr/bin/env python3
"""Explicitly download and verify all preference JSONL files before training."""

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
EXPECTED_OBJECTIVES = ("mpo", "cpo", "dpo")
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    data_path: Path
    contract_path: Path
    remote_data_path: str
    remote_contract_path: str
    artifact_sha256: str
    contract_sha256: str
    row_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="perform an offline validation of already downloaded files",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing corrupt/mismatched train file after verification",
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


def _bundle_path(root: Path, raw: str, *, field: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a bundle-relative path: {raw!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{field} escapes the bundle: {raw!r}")
    return resolved


def _remote_path(raw: str, *, field: str) -> str:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{field} must be a safe repository-relative path: {raw!r}")
    return path.as_posix()


def load_release(root: Path = ROOT) -> tuple[dict[str, Any], list[ObjectiveSpec]]:
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema_version") != "dqs_preference_release.v1":
        raise ValueError("unsupported release manifest schema")
    if manifest.get("data_mode") != "hf":
        raise ValueError("download-data requires an HF-backed release")
    if manifest.get("data_access") != "explicit_download_then_local":
        raise ValueError("release does not require explicit local data staging")
    hf_dataset = manifest.get("hf_dataset")
    if not isinstance(hf_dataset, dict) or not str(hf_dataset.get("repo_id", "")).strip():
        raise ValueError("manifest.hf_dataset.repo_id is missing")
    if HEX40.fullmatch(str(hf_dataset.get("revision", ""))) is None:
        raise ValueError("manifest HF revision must be an exact 40-hex commit")
    objectives = manifest.get("objectives")
    if not isinstance(objectives, dict) or set(objectives) != set(EXPECTED_OBJECTIVES):
        raise ValueError("manifest must contain exactly mpo/cpo/dpo")

    specs: list[ObjectiveSpec] = []
    for name in EXPECTED_OBJECTIVES:
        value = objectives[name]
        if not isinstance(value, dict):
            raise ValueError(f"manifest objective {name!r} is not an object")
        artifact_sha = str(value.get("artifact_sha256", ""))
        contract_sha = str(value.get("release_contract_sha256", ""))
        if HEX64.fullmatch(artifact_sha) is None or HEX64.fullmatch(contract_sha) is None:
            raise ValueError(f"{name} immutable SHA256 metadata is missing")
        if bool(value.get("data_bundled")):
            raise ValueError(f"{name} unexpectedly declares bundled train data")
        specs.append(
            ObjectiveSpec(
                name=name,
                data_path=_bundle_path(root, str(value.get("data", "")), field=f"{name}.data"),
                contract_path=_bundle_path(
                    root, str(value.get("contract", "")), field=f"{name}.contract"
                ),
                remote_data_path=_remote_path(
                    str(value.get("hf_train_filename", "")),
                    field=f"{name}.hf_train_filename",
                ),
                remote_contract_path=_remote_path(
                    str(value.get("hf_contract_filename", "")),
                    field=f"{name}.hf_contract_filename",
                ),
                artifact_sha256=artifact_sha,
                contract_sha256=contract_sha,
                row_count=int(value.get("row_count", -1)),
            )
        )
    return manifest, specs


def _validate_contract(spec: ObjectiveSpec) -> None:
    if not spec.contract_path.is_file():
        raise FileNotFoundError(f"{spec.name} bundled contract is missing: {spec.contract_path}")
    observed = sha256_file(spec.contract_path)
    if observed != spec.contract_sha256:
        raise ValueError(
            f"{spec.name} bundled contract hash mismatch: "
            f"observed={observed} expected={spec.contract_sha256}"
        )
    contract = load_json(spec.contract_path)
    if str(contract.get("artifact_sha256", "")) != spec.artifact_sha256:
        raise ValueError(f"{spec.name} contract and release disagree on artifact SHA256")
    if int(contract.get("row_count", -1)) != spec.row_count or spec.row_count <= 0:
        raise ValueError(f"{spec.name} contract and release disagree on row count")


def _validate_jsonl(path: Path, *, expected_rows: int) -> int:
    rows = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line is forbidden: {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"JSONL rows={rows}, expected={expected_rows}: {path}")
    return rows


def _validate_artifact(spec: ObjectiveSpec, path: Path | None = None) -> dict[str, Any]:
    artifact = path or spec.data_path
    if not artifact.is_file():
        raise FileNotFoundError(
            f"{spec.name} train data is missing: {artifact}; run `make download-data` first"
        )
    observed = sha256_file(artifact)
    if observed != spec.artifact_sha256:
        raise ValueError(
            f"{spec.name} train data hash mismatch: "
            f"observed={observed} expected={spec.artifact_sha256}"
        )
    rows = _validate_jsonl(artifact, expected_rows=spec.row_count)
    return {
        "path": str(artifact),
        "rows": rows,
        "artifact_sha256": observed,
    }


def validate_installed_data(
    root: Path = ROOT, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    loaded_manifest, specs = load_release(root)
    if manifest is not None and manifest != loaded_manifest:
        raise ValueError("supplied manifest does not match the release manifest on disk")
    results: dict[str, Any] = {}
    for spec in specs:
        _validate_contract(spec)
        results[spec.name] = _validate_artifact(spec)
    return {
        "status": "ok",
        "mode": "offline-local-validation",
        "objectives": results,
    }


def _downloaded_path(download_root: Path, relative: str, *, field: str) -> Path:
    path = _bundle_path(download_root, relative, field=field)
    if not path.is_file():
        raise FileNotFoundError(f"downloaded repository file is missing: {relative}")
    return path


def download_all(*, root: Path = ROOT, replace: bool = False, workers: int = 8) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("--workers must be positive")
    manifest, specs = load_release(root)
    pending: list[ObjectiveSpec] = []
    results: dict[str, Any] = {}
    for spec in specs:
        _validate_contract(spec)
        if not spec.data_path.exists():
            pending.append(spec)
            continue
        try:
            results[spec.name] = {"status": "existing", **_validate_artifact(spec)}
        except (OSError, ValueError):
            if not replace:
                raise ValueError(
                    f"{spec.name} existing train file is invalid; rerun with "
                    "`make download-data DOWNLOAD_REPLACE=1` to replace it"
                )
            pending.append(spec)

    if pending:
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:
            raise RuntimeError("missing huggingface_hub; run `make set` first") from exc

        data_root = root / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        allow_patterns = sorted(
            {
                path
                for spec in pending
                for path in (spec.remote_data_path, spec.remote_contract_path)
            }
        )
        hf_dataset = manifest["hf_dataset"]
        with tempfile.TemporaryDirectory(prefix=".download-data-", dir=data_root) as temp_dir:
            staging_root = Path(temp_dir)
            downloaded_root = Path(
                snapshot_download(
                    repo_id=str(hf_dataset["repo_id"]),
                    repo_type="dataset",
                    revision=str(hf_dataset["revision"]),
                    local_dir=str(staging_root / "snapshot"),
                    allow_patterns=allow_patterns,
                    max_workers=workers,
                )
            )
            install_root = staging_root / "install"
            install_root.mkdir()
            verified: dict[str, Path] = {}
            for spec in pending:
                remote_contract = _downloaded_path(
                    downloaded_root,
                    spec.remote_contract_path,
                    field=f"{spec.name}.remote_contract",
                )
                observed_contract = sha256_file(remote_contract)
                if observed_contract != spec.contract_sha256:
                    raise ValueError(
                        f"{spec.name} remote contract hash mismatch: "
                        f"observed={observed_contract} expected={spec.contract_sha256}"
                    )
                remote_data = _downloaded_path(
                    downloaded_root,
                    spec.remote_data_path,
                    field=f"{spec.name}.remote_data",
                )
                staged = install_root / f"{spec.name}.jsonl"
                shutil.copyfile(remote_data, staged)
                _validate_artifact(spec, staged)
                verified[spec.name] = staged

            # Install only after every pending objective has passed every check.
            for spec in pending:
                spec.data_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(verified[spec.name], spec.data_path)
                results[spec.name] = {"status": "downloaded", **_validate_artifact(spec)}

    final = validate_installed_data(root, manifest)
    return {
        "status": "ok",
        "mode": "explicit-download-then-local",
        "repo_id": manifest["hf_dataset"]["repo_id"],
        "revision": manifest["hf_dataset"]["revision"],
        "objectives": {
            name: {**final["objectives"][name], "status": results[name]["status"]}
            for name in EXPECTED_OBJECTIVES
        },
    }


def main() -> None:
    args = parse_args()
    result = (
        validate_installed_data()
        if args.check
        else download_all(replace=args.replace, workers=args.workers)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
