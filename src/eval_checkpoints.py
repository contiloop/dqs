#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from config_loader import compose_config
from io_utils import write_jsonl
from wandb_logging import eval_metric_payload, log_wandb_metrics


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _checkpoint_step(path: Path) -> int | None:
    suffix = path.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


def _checkpoint_dirs(
    *,
    checkpoint_dir: Path,
    start_step: int | None,
    end_step: int | None,
    max_checkpoints: int | None,
) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    if checkpoint_dir.exists():
        for path in checkpoint_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            step = _checkpoint_step(path)
            if step is None:
                continue
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0])
    if max_checkpoints is not None:
        checkpoints = checkpoints[:max(0, max_checkpoints)]
    if not checkpoints:
        raise SystemExit(f"no checkpoint-* directories found under {checkpoint_dir}")
    return checkpoints


def _summary_path(output_root: Path) -> Path:
    return output_root / "summary.jsonl"


def _checkpoint_output_dir(output_root: Path, step: int) -> Path:
    return output_root / f"checkpoint-{step:06d}"


def _eval_cmd(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("eval.py")),
        "--config",
        args.config,
        "--override",
        f"eval={args.profile}",
        "--model-path",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--skip-wandb-log",
    ]
    for override in args.override:
        cmd.extend(["--override", override])
    for override in args.eval_override:
        cmd.extend(["--override", override])
    if args.data_path:
        cmd.extend(["--data-path", args.data_path])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.metrics:
        cmd.extend(["--metrics", args.metrics])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")
    return cmd


def _read_eval_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "eval_summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = compose_config(args.config, overrides=args.override + [f"eval={args.profile}"] + args.eval_override)
    checkpoint_dir = Path(args.checkpoint_dir or str(_get(cfg, "paths.checkpoint_dir")))
    output_root = Path(
        args.output_dir
        or str(Path(str(_get(cfg, "paths.artifact_root"))) / "eval" / f"{args.profile}_by_checkpoint")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    checkpoints = _checkpoint_dirs(
        checkpoint_dir=checkpoint_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        max_checkpoints=args.max_checkpoints,
    )
    for step, checkpoint_path in checkpoints:
        output_dir = _checkpoint_output_dir(output_root, step)
        record: dict[str, Any] = {
            "checkpoint_step": step,
            "checkpoint_path": str(checkpoint_path),
            "eval_profile": args.profile,
            "dataset_path": str(args.data_path or _get(cfg, "eval.dataset_path")),
            "output_dir": str(output_dir),
        }
        cmd = _eval_cmd(args=args, checkpoint_path=checkpoint_path, output_dir=output_dir)
        record["command"] = " ".join(cmd)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["error"] = f"eval exited with code {exc.returncode}"
            records.append(record)
            write_jsonl(_summary_path(output_root), records)
            raise SystemExit(exc.returncode) from exc
        eval_summary = _read_eval_summary(output_dir)
        record["status"] = "completed"
        record["metrics"] = eval_summary.get("metrics", {})
        record["rows"] = eval_summary.get("rows")
        record["generation_ok_rows"] = eval_summary.get("generation_ok_rows")
        record["filter_pass_rows"] = eval_summary.get("filter_pass_rows")
        record["filter_fail_rows"] = eval_summary.get("filter_fail_rows")
        checkpoint_metrics = eval_metric_payload(eval_summary, prefix=f"eval_checkpoint/{args.profile}")
        checkpoint_metrics["checkpoint/step"] = step
        log_wandb_metrics(
            cfg,
            checkpoint_metrics,
            step=step,
            job_type="eval-checkpoints",
            finish=True,
        )
        records.append(record)
        write_jsonl(_summary_path(output_root), records)

    summary = {
        "run_id": _get(cfg, "run.id"),
        "eval_profile": args.profile,
        "checkpoint_dir": str(checkpoint_dir),
        "output_root": str(output_root),
        "summary_path": str(_summary_path(output_root)),
        "checkpoint_count": len(records),
        "status": "completed",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every DQS checkpoint in step order.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--profile", default="final")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
