#!/usr/bin/env python3
"""Build a self-contained, strict preference-training deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


POST_TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = POST_TRAINING_ROOT.parent
RESEARCH_ROOT = POST_TRAINING_ROOT / "research"
PACKAGE_ROOT = POST_TRAINING_ROOT / "dqs_preference_training_hf"
DEPLOYMENT_ASSETS = PACKAGE_ROOT
RUNTIME_SOURCE_ROOT = PACKAGE_ROOT / "src"
DEFAULT_OUTPUT = POST_TRAINING_ROOT / "dist" / "dqs_preference_training_hf"
RELEASE_MARKER = ".dqs-preference-release"

RUNTIME_SOURCE_FILES = (
    "preference_runtime.py",
    "train_mpo.py",
    "train_cpo.py",
    "train_dpo.py",
    "mpo_data.py",
    "mpo_masking.py",
    "mpo_model.py",
    "mpo_objective.py",
    "mpo_trainer.py",
    "mpo_wandb.py",
    "cpo_objective.py",
    "cpo_trainer.py",
    "dpo_trainer.py",
    "full_preference_data.py",
)

OBJECTIVES: dict[str, dict[str, str]] = {
    "mpo": {
        "config_source": "mpo_setting5.yaml",
        "config_release": "mpo.yaml",
        "entrypoint": "train_mpo.py",
        "data_source": "prepared/mpo_tokenized_pairs_final_source_filtered.jsonl",
        "data_release": "mpo.jsonl",
        "contract_source": "dataset_contract_final_source_filtered.json",
        "hf_train_filename": "mpo/train.jsonl",
        "hf_contract_filename": "mpo/dataset_contract.json",
    },
    "cpo": {
        "config_source": "cpo_full_response.yaml",
        "config_release": "cpo.yaml",
        "entrypoint": "train_cpo.py",
        "data_source": "prepared/cpo_tokenized_full_response_pairs_final.jsonl",
        "data_release": "cpo.jsonl",
        "contract_source": "dataset_contract_cpo_full_response.json",
        "hf_train_filename": "cpo/train.jsonl",
        "hf_contract_filename": "cpo/dataset_contract.json",
    },
    "dpo": {
        "config_source": "dpo_full_response.yaml",
        "config_release": "dpo.yaml",
        "entrypoint": "train_dpo.py",
        "data_source": "prepared/full_response_preference_pairs_final.jsonl",
        "data_release": "dpo.jsonl",
        "contract_source": "dataset_contract_full_response_preference.json",
        "hf_train_filename": "dpo/train.jsonl",
        "hf_contract_filename": "dpo/dataset_contract.json",
    },
}

SFT_MODEL_SOURCE: dict[str, Any] = {
    "repo_id": "alwaysgood/dqs-runs",
    "repo_type": "dataset",
    "revision": "a58b1878988efcecc9a2644f8324bd00131864b5",
    "remote_dir": "gemma4_e2b_it_full_iter_lowqe_sf_on_seed42/checkpoints/final",
    "local_dir": "models/sft_final",
    "expected_sft": {
        "run_id": "gemma4_e2b_it_full_iter_lowqe_sf_on_seed42",
        "subset_idx": 22,
        "global_step": 184,
        "tuning_mode": "full",
    },
    "files": {
        "chat_template.jinja": {
            "size": 16804,
            "sha256": "33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373",
        },
        "config.json": {
            "size": 5111,
            "sha256": "86c9621e8b8512a6e338e2ecce7d66ace31c596f77a7b358786bfad1cdcff353",
        },
        "dqs_stage_model.json": {
            "size": 128,
            "sha256": "cb4e2baa46d19fc0aad3adbb39bb23588947b202ea29caceac6e59292f560876",
        },
        "generation_config.json": {
            "size": 203,
            "sha256": "5439541d3bf0ba9ade7c1122dc14703f85f32a520150351b344b14f48c574cc2",
        },
        "model.safetensors": {
            "size": 10247526494,
            "sha256": "304387c31d762065420035d711ebed0eb6e296d0ee28c8918645ed3943fdaf4e",
        },
        "processor_config.json": {
            "size": 1689,
            "sha256": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
        },
        "tokenizer.json": {
            "size": 32169626,
            "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        },
        "tokenizer_config.json": {
            "size": 6865,
            "sha256": "de3cba60561eb2ee6362e34274bb7196b87d8febf01d54a2356a1b1f5a14284b",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-mode",
        choices=("local", "hf"),
        default="local",
        help="local copies immutable train JSONL files; hf pins one dataset repo commit",
    )
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-revision")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace only an existing directory carrying the release marker",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="also create <output>.tar.gz and its SHA256 sidecar",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _validate_hf_args(args: argparse.Namespace) -> None:
    if args.data_mode != "hf":
        if args.hf_repo_id or args.hf_revision:
            raise ValueError("--hf-repo-id/--hf-revision require --data-mode hf")
        return
    if not args.hf_repo_id:
        raise ValueError("--data-mode hf requires --hf-repo-id")
    revision = str(args.hf_revision or "")
    if len(revision) != 40 or any(char not in "0123456789abcdefABCDEF" for char in revision):
        raise ValueError("--data-mode hf requires an exact 40-hex --hf-revision")


def _ensure_inputs(*, data_mode: str) -> None:
    required_paths = [
        *(RUNTIME_SOURCE_ROOT / name for name in RUNTIME_SOURCE_FILES),
        PACKAGE_ROOT / "requirements-gpu.txt",
        DEPLOYMENT_ASSETS / "Makefile",
        DEPLOYMENT_ASSETS / "README.md",
        DEPLOYMENT_ASSETS / "scripts" / "download_data.py",
        DEPLOYMENT_ASSETS / "scripts" / "download_model.py",
        DEPLOYMENT_ASSETS / "scripts" / "validate_bundle.py",
        DEPLOYMENT_ASSETS / "tests" / "test_download_data.py",
        DEPLOYMENT_ASSETS / "tests" / "test_download_model.py",
        DEPLOYMENT_ASSETS / "tests" / "test_release_contract.py",
        *(
            RESEARCH_ROOT / "configs" / spec["config_source"]
            for spec in OBJECTIVES.values()
        ),
        *(
            RESEARCH_ROOT / "contracts" / spec["contract_source"]
            for spec in OBJECTIVES.values()
        ),
    ]
    if data_mode == "local":
        required_paths.extend(
            RESEARCH_ROOT / spec["data_source"]
            for spec in OBJECTIVES.values()
        )
    missing = [
        str(path)
        for path in required_paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("release inputs are missing: " + ", ".join(missing))


def _release_config(
    objective: str,
    spec: dict[str, str],
    *,
    data_mode: str,
    hf_repo_id: str | None,
    hf_revision: str | None,
) -> dict[str, Any]:
    source_path = RESEARCH_ROOT / "configs" / spec["config_source"]
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config is not a mapping: {source_path}")
    run = payload["run"]
    model = payload["model"]
    data = payload["data"]
    run["output_dir"] = f"outputs/{run['id']}"
    model["name_or_path"] = "models/sft_final"
    data["cache_dir"] = f".cache/datasets/{objective}"
    data["contract_path"] = f"data/contracts/{objective}.json"
    if objective == "mpo":
        # Evaluation remains owned by the main DQS evaluator, not this training bundle.
        payload.pop("evaluation", None)
    if data_mode == "local":
        data["source"] = "local"
        data["path"] = f"data/train/{spec['data_release']}"
        data["hf_repo_id"] = None
        data["hf_revision"] = None
    else:
        # HF is only the explicit staging source. Training is local-only and
        # never performs a network download from inside a trainer process.
        data["source"] = "local"
        data["path"] = f"data/train/{spec['data_release']}"
        data["hf_repo_id"] = None
        data["hf_revision"] = None
        data["hf_train_filename"] = None
        data["hf_contract_filename"] = None
    for key in (
        "hf_config_name",
        "hf_train_filename",
        "hf_eval_filename",
        "hf_train_split",
        "hf_eval_split",
        "hf_contract_filename",
    ):
        if key in data:
            data[key] = None
    return payload


def _copy_assets(staging: Path) -> None:
    shutil.copy2(DEPLOYMENT_ASSETS / "Makefile", staging / "Makefile")
    shutil.copy2(DEPLOYMENT_ASSETS / "README.md", staging / "README.md")
    shutil.copy2(
        DEPLOYMENT_ASSETS / "scripts" / "download_data.py",
        staging / "scripts" / "download_data.py",
    )
    shutil.copy2(
        DEPLOYMENT_ASSETS / "scripts" / "download_model.py",
        staging / "scripts" / "download_model.py",
    )
    shutil.copy2(
        DEPLOYMENT_ASSETS / "scripts" / "validate_bundle.py",
        staging / "scripts" / "validate_bundle.py",
    )
    shutil.copy2(
        DEPLOYMENT_ASSETS / "tests" / "test_download_data.py",
        staging / "tests" / "test_download_data.py",
    )
    shutil.copy2(
        DEPLOYMENT_ASSETS / "tests" / "test_download_model.py",
        staging / "tests" / "test_download_model.py",
    )
    shutil.copy2(
        DEPLOYMENT_ASSETS / "tests" / "test_release_contract.py",
        staging / "tests" / "test_release_contract.py",
    )


def _write_text_files(staging: Path) -> None:
    (staging / RELEASE_MARKER).write_text(
        "generated by post_training/scripts/build_release.py\n", encoding="utf-8"
    )
    (staging / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n.venv/\n.cache/\ndata/train/*.jsonl\n"
        "models/\noutputs/\nwandb/\n",
        encoding="utf-8",
    )
    (staging / "src" / "__init__.py").write_text(
        '"""DQS strict preference-training runtime."""\n', encoding="utf-8"
    )


def _copy_runtime(staging: Path) -> None:
    for filename in RUNTIME_SOURCE_FILES:
        shutil.copy2(RUNTIME_SOURCE_ROOT / filename, staging / "src" / filename)
    shutil.copy2(
        PACKAGE_ROOT / "requirements-gpu.txt",
        staging / "requirements-gpu.txt",
    )


def _normalize_contract_paths(value: Any) -> Any:
    """Remove workstation-specific paths without changing semantic hashes."""

    if isinstance(value, dict):
        return {key: _normalize_contract_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_contract_paths(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.relative_to(POST_TRAINING_ROOT)
            except ValueError:
                return value
            return f"source://post_training/{relative.as_posix()}"
    return value


def _copy_objectives(
    staging: Path,
    *,
    data_mode: str,
    hf_repo_id: str | None,
    hf_revision: str | None,
) -> dict[str, dict[str, Any]]:
    objective_manifest: dict[str, dict[str, Any]] = {}
    for objective, spec in OBJECTIVES.items():
        contract_source = RESEARCH_ROOT / "contracts" / spec["contract_source"]
        contract_payload = json.loads(contract_source.read_text(encoding="utf-8"))
        source_contract_sha256 = sha256_file(contract_source)
        release_contract_payload = _normalize_contract_paths(contract_payload)
        contract_release = staging / "data" / "contracts" / f"{objective}.json"
        contract_release.write_text(
            json.dumps(
                release_contract_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        config_payload = _release_config(
            objective,
            spec,
            data_mode=data_mode,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
        )
        config_release = staging / "configs" / spec["config_release"]
        config_release.write_text(
            yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        expected_hash = str(contract_payload.get("artifact_sha256", ""))
        if not expected_hash:
            raise ValueError(
                f"{objective} contract has no immutable artifact_sha256"
            )
        data_relative = f"data/train/{spec['data_release']}"
        data_bundled = data_mode == "local"
        if data_mode == "local":
            data_source = RESEARCH_ROOT / spec["data_source"]
            observed_hash = sha256_file(data_source)
            if observed_hash != expected_hash:
                raise ValueError(
                    f"{objective} source data does not match its immutable contract: "
                    f"observed={observed_hash} expected={expected_hash!r}"
                )
            data_release = staging / "data" / "train" / spec["data_release"]
            shutil.copy2(data_source, data_release)
            data_relative = data_release.relative_to(staging).as_posix()

        objective_manifest[objective] = {
            "entrypoint": f"src/{spec['entrypoint']}",
            "config": config_release.relative_to(staging).as_posix(),
            "contract": contract_release.relative_to(staging).as_posix(),
            "source_contract_sha256": source_contract_sha256,
            "release_contract_sha256": sha256_file(contract_release),
            "data": data_relative,
            "data_bundled": data_bundled,
            "artifact_sha256": expected_hash,
            "row_count": int(contract_payload["row_count"]),
            "hf_train_filename": spec["hf_train_filename"] if data_mode == "hf" else None,
            "hf_contract_filename": (
                spec["hf_contract_filename"] if data_mode == "hf" else None
            ),
            "run_id": str(config_payload["run"]["id"]),
            "expected_sft": {
                "run_id": str(config_payload["model"]["expected_sft_run_id"]),
                "subset_idx": int(config_payload["model"]["expected_sft_subset_idx"]),
                "global_step": int(config_payload["model"]["expected_sft_global_step"]),
            },
        }
    return objective_manifest


def _content_manifest(staging: Path) -> tuple[dict[str, str], str]:
    hashes: dict[str, str] = {}
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(staging).as_posix()
        hashes[relative] = sha256_file(path)
    digest = hashlib.sha256()
    for relative, file_hash in sorted(hashes.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return hashes, digest.hexdigest()


def _sft_model_manifest(objectives: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_values = {
        (
            value["expected_sft"]["run_id"],
            int(value["expected_sft"]["subset_idx"]),
            int(value["expected_sft"]["global_step"]),
        )
        for value in objectives.values()
    }
    expected = SFT_MODEL_SOURCE["expected_sft"]
    pinned = (
        expected["run_id"],
        int(expected["subset_idx"]),
        int(expected["global_step"]),
    )
    if expected_values != {pinned}:
        raise ValueError(
            "objective configs do not match the pinned downloadable full-SFT model"
        )
    files = SFT_MODEL_SOURCE["files"]
    return {
        **SFT_MODEL_SOURCE,
        "file_count": len(files),
        "total_size_bytes": sum(int(value["size"]) for value in files.values()),
    }


def _write_manifest(
    staging: Path,
    *,
    data_mode: str,
    hf_repo_id: str | None,
    hf_revision: str | None,
    objectives: dict[str, dict[str, Any]],
    source_git_commit: str | None,
    source_post_training_dirty: bool,
) -> dict[str, Any]:
    hashes, content_sha = _content_manifest(staging)
    manifest: dict[str, Any] = {
        "schema_version": "dqs_preference_release.v1",
        "bundle_name": "dqs_preference_training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": source_git_commit,
        "source_post_training_dirty": source_post_training_dirty,
        "generator": "post_training/scripts/build_release.py",
        "data_mode": data_mode,
        "data_access": (
            "bundled_local" if data_mode == "local" else "explicit_download_then_local"
        ),
        "hf_dataset": (
            {"repo_id": hf_repo_id, "revision": hf_revision}
            if data_mode == "hf"
            else None
        ),
        "sft_model": _sft_model_manifest(objectives),
        "objectives": objectives,
        "runtime_requirements": "requirements-gpu.txt",
        "files": hashes,
        "bundle_content_sha256": content_sha,
        "excluded_scopes": [
            "raw_golden_pairs",
            "dataset_synthesis",
            "source_quality_review",
            "analysis_reports",
            "research_caches",
            "evaluation_runtime",
        ],
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _archive(output: Path) -> tuple[Path, str]:
    archive = Path(str(output) + ".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    digest = sha256_file(archive)
    Path(str(archive) + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    return archive, digest


def build(args: argparse.Namespace) -> dict[str, Any]:
    _validate_hf_args(args)
    _ensure_inputs(data_mode=args.data_mode)
    # Capture provenance before creating the temporary staging directory under
    # post_training/, otherwise the generator can mark its own staging files as
    # uncommitted source changes.
    source_git_commit = _git_value("rev-parse", "HEAD")
    source_post_training_dirty = bool(
        _git_value("status", "--short", "--", "post_training")
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"release output already exists; pass --replace: {output}")
        if not (output / RELEASE_MARKER).is_file():
            raise ValueError(f"refusing to replace an unmarked directory: {output}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        for relative in ("src", "configs", "scripts", "tests", "data/contracts", "data/train"):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        _copy_assets(staging)
        _write_text_files(staging)
        _copy_runtime(staging)
        objectives = _copy_objectives(
            staging,
            data_mode=args.data_mode,
            hf_repo_id=args.hf_repo_id,
            hf_revision=args.hf_revision,
        )
        manifest = _write_manifest(
            staging,
            data_mode=args.data_mode,
            hf_repo_id=args.hf_repo_id,
            hf_revision=args.hf_revision,
            objectives=objectives,
            source_git_commit=source_git_commit,
            source_post_training_dirty=source_post_training_dirty,
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    result: dict[str, Any] = {
        "output": str(output),
        "data_mode": args.data_mode,
        "bundle_content_sha256": manifest["bundle_content_sha256"],
        "objectives": {
            name: {"rows": value["row_count"], "artifact_sha256": value["artifact_sha256"]}
            for name, value in manifest["objectives"].items()
        },
    }
    if args.archive:
        archive, digest = _archive(output)
        result["archive"] = str(archive)
        result["archive_sha256"] = digest
    return result


def main() -> None:
    result = build(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
