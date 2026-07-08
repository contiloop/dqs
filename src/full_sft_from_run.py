#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config, config_hash, save_effective_config
from io_utils import read_jsonl
from progress import progress, progress_context
from runtime_logging import configure_runtime_logging
from sft_train import estimate_update_steps_for_rows


configure_runtime_logging()


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _source_sft_path(source_run_dir: Path, subset_idx: int) -> Path:
    return source_run_dir / "subsets" / f"subset_{subset_idx:03d}" / "sft_train.jsonl"


def _target_sft_path(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return Path(str(_get(cfg, "paths.subset_dir"))) / f"subset_{subset_idx:03d}" / "sft_train.jsonl"


def _checkpoint_dir(cfg: Mapping[str, Any]) -> Path:
    return Path(str(_get(cfg, "paths.checkpoint_dir")))


def _stage_state_path(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return _checkpoint_dir(cfg) / f"sft_stage_state_subset_{subset_idx:03d}.json"


def _training_summary_path(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return _checkpoint_dir(cfg) / f"sft_training_summary_subset_{subset_idx:03d}.json"


def _dry_run_summary_path(cfg: Mapping[str, Any], subset_idx: int) -> Path:
    return _checkpoint_dir(cfg) / f"sft_dry_run_summary_subset_{subset_idx:03d}.json"


def _run_summary_path(cfg: Mapping[str, Any]) -> Path:
    return Path(str(_get(cfg, "paths.artifact_root"))) / "full_sft_from_run_summary.json"


def _available_source_subsets(source_run_dir: Path) -> list[int]:
    subset_root = source_run_dir / "subsets"
    if not subset_root.exists():
        raise SystemExit(f"source run has no subsets directory: {subset_root}")

    indices: list[int] = []
    for path in subset_root.glob("subset_*/sft_train.jsonl"):
        subset_name = path.parent.name
        raw_index = subset_name.removeprefix("subset_")
        if raw_index.isdigit():
            indices.append(int(raw_index))
    if not indices:
        raise SystemExit(f"source run has no subset_*/sft_train.jsonl files: {subset_root}")
    return sorted(set(indices))


def _resolve_stage_end(
    *,
    cfg: Mapping[str, Any],
    source_indices: list[int],
    start_subset: int,
    end_subset: int | None,
    max_subsets: int | None,
) -> int:
    if end_subset is not None:
        resolved_end = end_subset
    else:
        configured_end = _get(cfg, "run.subset_end")
        resolved_end = int(configured_end) if configured_end is not None else max(source_indices) + 1

    if max_subsets is not None:
        if max_subsets <= 0:
            raise SystemExit("stage_max_subsets must be > 0 when set")
        resolved_end = min(resolved_end, start_subset + max_subsets)

    if resolved_end <= start_subset:
        raise SystemExit(f"no subsets selected for full SFT: start={start_subset}, end={resolved_end}")
    return resolved_end


def _row_counts_for_range(source_run_dir: Path, start_subset: int, end_subset: int) -> dict[int, int]:
    rows_by_subset: dict[int, int] = {}
    missing: list[str] = []
    empty: list[str] = []
    for subset_idx in range(start_subset, end_subset):
        source_path = _source_sft_path(source_run_dir, subset_idx)
        if not source_path.exists():
            missing.append(str(source_path))
            continue
        row_count = len(read_jsonl(source_path))
        if row_count <= 0:
            empty.append(str(source_path))
            continue
        rows_by_subset[subset_idx] = row_count
    if missing:
        raise SystemExit("missing source SFT datasets:\n" + "\n".join(missing))
    if empty:
        raise SystemExit("empty source SFT datasets:\n" + "\n".join(empty))
    return rows_by_subset


def _total_scheduler_steps(
    *,
    cfg: Mapping[str, Any],
    rows_by_subset: Mapping[int, int],
    world_size: int,
) -> int:
    return sum(
        estimate_update_steps_for_rows(row_count, cfg, world_size=max(1, world_size))
        for row_count in rows_by_subset.values()
    )


def _is_completed_subset(cfg: Mapping[str, Any], subset_idx: int) -> bool:
    state = _read_json_if_exists(_stage_state_path(cfg, subset_idx))
    return bool(state and state.get("status") == "completed")


def _copy_dataset(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _sft_command(
    *,
    args: argparse.Namespace,
    subset_idx: int,
    dataset_path: Path,
    scheduler_total_steps: int,
    overrides: list[str],
) -> list[str]:
    sft_script = Path(__file__).with_name("sft_train.py")
    if args.sft_nproc_per_node > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.sft_nproc_per_node),
            str(sft_script),
        ]
    else:
        cmd = [sys.executable, str(sft_script)]

    cmd.extend(
        [
            "--config",
            args.config,
            "--subset-idx",
            str(subset_idx),
            "--dataset-path",
            str(dataset_path),
            "--stage-scheduler-total-steps",
            str(scheduler_total_steps),
            "--force-save-checkpoint",
        ]
    )
    for override in overrides:
        cmd.extend(["--override", override])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _compose_overrides(args: argparse.Namespace) -> list[str]:
    overrides = ["training=full"]
    if args.final_only_artifacts:
        overrides.extend(
            [
                "training.save_full_model=false",
                "training.save_total_limit=1",
            ]
        )
    overrides.extend(args.override)
    if args.run_id:
        overrides.append(f"run.id={args.run_id}")
    return overrides


def _delete_checkpoint_dirs(cfg: Mapping[str, Any]) -> list[str]:
    deleted: list[str] = []
    checkpoint_dir = _checkpoint_dir(cfg)
    if not checkpoint_dir.exists():
        return deleted
    for path in sorted(checkpoint_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        shutil.rmtree(path)
        deleted.append(str(path))
    return deleted


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_run_dir = Path(args.source_run_dir).expanduser()
    if not source_run_dir.exists():
        raise SystemExit(f"missing source run dir: {source_run_dir}")

    overrides = _compose_overrides(args)
    cfg = compose_config(args.config, overrides=overrides)
    if str(_get(cfg, "training.tuning_mode", "")).strip().lower() != "full":
        raise SystemExit("full-sft-from-run requires training.tuning_mode=full")

    target_run_dir = Path(str(_get(cfg, "paths.artifact_root"))).expanduser()
    if source_run_dir.resolve() == target_run_dir.resolve():
        raise SystemExit(
            "source run and target full SFT run resolve to the same directory; "
            "use a new FULL_SFT_RUN_ID"
        )

    source_indices = _available_source_subsets(source_run_dir)
    cycle_start = int(_get(cfg, "run.subset_start", 0) or 0)
    start_subset = int(args.subset_idx if args.subset_idx is not None else cycle_start)
    stage_end = _resolve_stage_end(
        cfg=cfg,
        source_indices=source_indices,
        start_subset=start_subset,
        end_subset=args.stage_end_subset,
        max_subsets=args.stage_max_subsets,
    )
    if start_subset < cycle_start:
        raise SystemExit(f"start subset {start_subset} is before run.subset_start {cycle_start}")

    scheduler_rows = _row_counts_for_range(source_run_dir, cycle_start, stage_end)
    sft_rows = {idx: scheduler_rows[idx] for idx in range(start_subset, stage_end)}
    scheduler_total_steps = _total_scheduler_steps(
        cfg=cfg,
        rows_by_subset=scheduler_rows,
        world_size=max(1, int(args.sft_nproc_per_node or 1)),
    )

    cfg_hash = config_hash(cfg)
    save_effective_config(_get(cfg, "paths.config_snapshot_path"), cfg)
    Path(str(_get(cfg, "paths.config_hash_path"))).parent.mkdir(parents=True, exist_ok=True)
    Path(str(_get(cfg, "paths.config_hash_path"))).write_text(f"{cfg_hash}\n", encoding="utf-8")

    summary_path = _run_summary_path(cfg)
    summary: dict[str, Any] = {
        "run_id": _get(cfg, "run.id"),
        "source_run_dir": str(source_run_dir),
        "target_run_dir": str(target_run_dir),
        "config_hash": cfg_hash,
        "training_mode": _get(cfg, "training.tuning_mode"),
        "stage_start_subset": start_subset,
        "stage_end_subset": stage_end,
        "stage_end_subset_exclusive": True,
        "cycle_start_subset": cycle_start,
        "copy_datasets": bool(args.copy_datasets),
        "final_only_artifacts": bool(args.final_only_artifacts),
        "delete_checkpoints_on_complete": bool(args.delete_checkpoints_on_complete),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "plan_only": bool(args.plan_only),
        "sft_nproc_per_node": int(args.sft_nproc_per_node),
        "sft_scheduler_total_steps": scheduler_total_steps,
        "source_rows_by_subset": {f"subset_{idx:03d}": count for idx, count in sft_rows.items()},
        "subsets": [],
    }
    _write_json(summary_path, summary)
    progress(
        "full-sft-from-run start",
        run=_get(cfg, "run.id"),
        source=source_run_dir,
        start_subset=start_subset,
        end_subset_exclusive=stage_end,
        sft_nproc_per_node=args.sft_nproc_per_node,
        scheduler_total_steps=scheduler_total_steps,
    )

    if args.plan_only:
        summary["status"] = "planned"
        _write_json(summary_path, summary)
        progress("full-sft-from-run planned", run=_get(cfg, "run.id"), subsets=len(sft_rows))
        return summary

    env = _subprocess_env()
    for subset_idx in range(start_subset, stage_end):
        source_path = _source_sft_path(source_run_dir, subset_idx)
        target_path = _target_sft_path(cfg, subset_idx)
        dataset_path = target_path if args.copy_datasets else source_path
        subset_record: dict[str, Any] = {
            "subset_idx": subset_idx,
            "subset_name": f"subset_{subset_idx:03d}",
            "source_sft_path": str(source_path),
            "target_sft_path": str(target_path) if args.copy_datasets else None,
            "sft_rows": sft_rows[subset_idx],
        }

        if _is_completed_subset(cfg, subset_idx) and not args.force:
            subset_record["status"] = "skipped_completed"
            subset_record["stage_state_path"] = str(_stage_state_path(cfg, subset_idx))
            subset_record["training_summary_path"] = str(_training_summary_path(cfg, subset_idx))
            summary["subsets"].append(subset_record)
            _write_json(summary_path, summary)
            progress("full-sft subset skip", subset=f"subset_{subset_idx:03d}", reason="completed")
            continue

        if args.copy_datasets:
            _copy_dataset(source_path, target_path)

        cmd = _sft_command(
            args=args,
            subset_idx=subset_idx,
            dataset_path=dataset_path,
            scheduler_total_steps=scheduler_total_steps,
            overrides=overrides,
        )
        subset_record["sft_command"] = " ".join(cmd)
        try:
            with progress_context("full-sft subset", subset=f"subset_{subset_idx:03d}", rows=sft_rows[subset_idx]):
                subprocess.run(cmd, check=True, env=env)
        except KeyboardInterrupt as exc:
            subset_record["status"] = "interrupted"
            subset_record["error"] = "interrupted by user"
            summary["subsets"].append(subset_record)
            summary["status"] = "interrupted"
            summary["failed_subset_idx"] = subset_idx
            _write_json(summary_path, summary)
            raise SystemExit(130) from exc
        except subprocess.CalledProcessError as exc:
            subset_record["status"] = "failed"
            subset_record["error"] = f"SFT exited with code {exc.returncode}"
            summary["subsets"].append(subset_record)
            summary["status"] = "failed"
            summary["failed_subset_idx"] = subset_idx
            _write_json(summary_path, summary)
            raise SystemExit(exc.returncode) from exc

        if args.dry_run:
            subset_record["status"] = "dry_run_completed"
            subset_record["training_summary_path"] = str(_dry_run_summary_path(cfg, subset_idx))
            subset_record["training_summary"] = _read_json_if_exists(_dry_run_summary_path(cfg, subset_idx))
        else:
            subset_record["status"] = "completed"
            subset_record["stage_state_path"] = str(_stage_state_path(cfg, subset_idx))
            subset_record["training_summary_path"] = str(_training_summary_path(cfg, subset_idx))
            subset_record["training_summary"] = _read_json_if_exists(_training_summary_path(cfg, subset_idx))
        summary["subsets"].append(subset_record)
        _write_json(summary_path, summary)

    if args.delete_checkpoints_on_complete and not args.dry_run:
        deleted = _delete_checkpoint_dirs(cfg)
        summary["deleted_checkpoint_dirs"] = deleted
        progress("full-sft checkpoints deleted", run=_get(cfg, "run.id"), count=len(deleted))

    summary["status"] = "dry_run_completed" if args.dry_run else "completed"
    _write_json(summary_path, summary)
    progress("full-sft-from-run done", run=_get(cfg, "run.id"), subsets=len(summary["subsets"]))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-parameter SFT over existing subset_*/sft_train.jsonl files from a DQS run."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--run-id", default=None, help="Target full SFT run id. Must differ from the source run.")
    parser.add_argument("--subset-idx", type=int, default=None, help="First subset index to train.")
    parser.add_argument("--stage-end-subset", type=int, default=None, help="Exclusive final subset index.")
    parser.add_argument("--stage-max-subsets", type=int, default=None)
    parser.add_argument("--sft-nproc-per-node", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--no-copy-datasets", dest="copy_datasets", action="store_false")
    parser.set_defaults(copy_datasets=True)
    parser.add_argument(
        "--final-only-artifacts",
        action="store_true",
        help="Avoid duplicate full_model artifacts and keep only the latest Trainer checkpoint.",
    )
    parser.add_argument(
        "--delete-checkpoints-on-complete",
        action="store_true",
        help="After all subsets finish, delete checkpoint-* directories and keep final artifacts only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
