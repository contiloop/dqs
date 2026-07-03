#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from split_core import split_text
from split_io import chunked_row
from token_counter import QwenTokenCounter


_TOKEN_COUNTER: QwenTokenCounter | None = None


@dataclass(frozen=True, slots=True)
class PreprocessArgs:
    raw_dataset_repo: str
    split: str
    revision: str
    source_field: str
    max_input_tokens: int
    tokenizer_model: str
    tokenizer_revision: str
    output_dir: Path
    workers: int
    rows_per_shard: int
    task_buffer_size: int
    limit: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild prepared data from raw HF corpus.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> PreprocessArgs:
    with path.open(encoding="utf-8") as handle:
        root = yaml.safe_load(handle) or {}
    data = root["data"]
    preprocess = data["preprocess_raw"]
    tokenizer_model = args.tokenizer_model or preprocess["tokenizer_model"]
    if isinstance(tokenizer_model, str) and tokenizer_model.startswith("${"):
        tokenizer_model = default_model_name(path.parent)
    return PreprocessArgs(
        raw_dataset_repo=args.repo or data["raw_dataset_repo"],
        split=args.split or preprocess.get("split", "train"),
        revision=args.revision or preprocess.get("revision", "main"),
        source_field=data["source_field"],
        max_input_tokens=int(data["max_input_tokens"]),
        tokenizer_model=tokenizer_model,
        tokenizer_revision=args.tokenizer_revision or preprocess.get("tokenizer_revision", "main"),
        output_dir=Path(args.output_dir or preprocess["output_dir"]),
        workers=int(args.workers or preprocess.get("workers", 1)),
        rows_per_shard=int(preprocess.get("rows_per_shard", 50000)),
        task_buffer_size=int(preprocess.get("task_buffer_size", 256)),
        limit=args.limit,
    )


def default_model_name(config_dir: Path) -> str:
    root = load_yaml(config_dir / "config.yaml")
    model_group = None
    for item in root.get("defaults", []):
        if isinstance(item, dict) and "model" in item:
            model_group = item["model"]
            break
    if not isinstance(model_group, str):
        raise SystemExit("cannot resolve default model group from configs/config.yaml")
    model_cfg = load_yaml(config_dir / "model" / f"{model_group}.yaml")
    name_or_path = model_cfg.get("model", {}).get("name_or_path")
    if not isinstance(name_or_path, str):
        raise SystemExit(f"cannot resolve model.name_or_path for model group {model_group}")
    return name_or_path


def init_worker(tokenizer_model: str, tokenizer_revision: str) -> None:
    global _TOKEN_COUNTER
    _TOKEN_COUNTER = QwenTokenCounter.from_model_or_path(
        tokenizer_model,
        revision=tokenizer_revision,
    )


def process_row(task: tuple[int, dict[str, Any], str, int]) -> tuple[int, list[dict[str, Any]], bool]:
    row_index, row, source_field, max_input_tokens = task
    if _TOKEN_COUNTER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    text = row.get(source_field)
    if not isinstance(text, str):
        raise ValueError(f"row {row_index} missing string field {source_field!r}")
    chunks = split_text(text, max_input_tokens, _TOKEN_COUNTER)
    if len(chunks) == 1:
        return row_index, [row], False
    rows = [
        chunked_row(row, source_field, chunk, row_index + 1, chunk_idx, len(chunks))
        for chunk_idx, chunk in enumerate(chunks)
    ]
    return row_index, rows, True


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("missing pyarrow; run `make set` first") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"invalid yaml object: {path}")
    return data


def flush_shard(output_dir: Path, rows: list[dict[str, Any]], shard_idx: int) -> Path:
    shard_path = output_dir / f"part-{shard_idx:05d}.parquet"
    write_parquet(shard_path, rows)
    print(f"wrote {shard_path} rows={len(rows)}")
    rows.clear()
    return shard_path


def main() -> None:
    parsed_args = parse_args()
    args = load_config(Path(parsed_args.config), parsed_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("missing datasets; run `make set` first") from exc

    dataset = load_dataset(args.raw_dataset_repo, split=args.split, revision=args.revision)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    input_rows = 0
    output_rows = 0
    split_rows = 0
    chunks_created = 0
    shard_idx = 0
    shard_paths: list[str] = []
    shard_rows: list[dict[str, Any]] = []
    buffer_size = max(1, args.task_buffer_size)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.tokenizer_model, args.tokenizer_revision),
    ) as pool:
        pending: list[Any] = []
        for row_index, row in enumerate(dataset):
            pending.append(
                pool.submit(
                    process_row,
                    (row_index, dict(row), args.source_field, args.max_input_tokens),
                )
            )
            if len(pending) < buffer_size:
                continue
            shard_idx = drain_pending(
                pending,
                args.output_dir,
                shard_rows,
                shard_paths,
                shard_idx,
                args.rows_per_shard,
                counters=(input_rows, output_rows, split_rows, chunks_created),
            )
            input_rows += len(pending)
            for future in pending:
                _, rows, was_split = future.result()
                output_rows += len(rows)
                split_rows += int(was_split)
                if was_split:
                    chunks_created += len(rows)
            pending.clear()

        if pending:
            shard_idx = drain_pending(
                pending,
                args.output_dir,
                shard_rows,
                shard_paths,
                shard_idx,
                args.rows_per_shard,
                counters=(input_rows, output_rows, split_rows, chunks_created),
            )
            input_rows += len(pending)
            for future in pending:
                _, rows, was_split = future.result()
                output_rows += len(rows)
                split_rows += int(was_split)
                if was_split:
                    chunks_created += len(rows)
            pending.clear()

    if shard_rows:
        shard_paths.append(str(flush_shard(args.output_dir, shard_rows, shard_idx)))

    summary = {
        "raw_dataset_repo": args.raw_dataset_repo,
        "split": args.split,
        "revision": args.revision,
        "source_field": args.source_field,
        "max_input_tokens": args.max_input_tokens,
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_revision": args.tokenizer_revision,
        "output_dir": str(args.output_dir),
        "workers": args.workers,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "split_rows": split_rows,
        "chunks_created": chunks_created,
        "shards": shard_paths,
    }
    write_summary(args.output_dir / "preprocess_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def drain_pending(
    pending: list[Any],
    output_dir: Path,
    shard_rows: list[dict[str, Any]],
    shard_paths: list[str],
    shard_idx: int,
    rows_per_shard: int,
    *,
    counters: tuple[int, int, int, int],
) -> int:
    del counters
    for future in pending:
        _, rows, _ = future.result()
        shard_rows.extend(rows)
        while len(shard_rows) >= rows_per_shard:
            to_write = shard_rows[:rows_per_shard]
            del shard_rows[:rows_per_shard]
            shard_paths.append(str(flush_shard(output_dir, to_write, shard_idx)))
            shard_idx += 1
    return shard_idx


if __name__ == "__main__":
    main()
