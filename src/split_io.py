from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import TextIO, TypeAlias

from split_core import SplitError, TokenCounter, split_text


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    input_path: Path
    output_path: Path
    summary_path: Path
    text_field: str
    max_tokens: int
    token_counter: TokenCounter
    tokenizer_name: str


@dataclass(frozen=True, slots=True)
class SplitSummary:
    input_path: str
    output_path: str
    summary_path: str
    text_field: str
    max_tokens: int
    tokenizer_name: str
    input_rows: int
    output_rows: int
    split_rows: int
    chunks_created: int

    def to_dict(self) -> JsonObject:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "text_field": self.text_field,
            "max_tokens": self.max_tokens,
            "tokenizer_name": self.tokenizer_name,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "split_rows": self.split_rows,
            "chunks_created": self.chunks_created,
        }


@contextmanager
def open_text(path: Path, mode: str) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, mode, encoding="utf-8", newline="\n") as handle:
            yield handle
        return
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        yield handle


def split_file(config: SplitConfig) -> SplitSummary:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    output_rows = 0
    split_rows = 0
    chunks_created = 0
    with open_text(config.input_path, "rt") as source:
        with open_text(config.output_path, "wt") as target:
            for line in source:
                input_rows += 1
                row = parse_json_object(line, input_rows)
                text_value = row.get(config.text_field)
                if not isinstance(text_value, str):
                    raise SplitError(
                        detail=f"row {input_rows} missing string field {config.text_field!r}"
                    )
                chunks = split_text(text_value, config.max_tokens, config.token_counter)
                if len(chunks) == 1:
                    write_json_row(target, row)
                    output_rows += 1
                    continue
                split_rows += 1
                chunks_created += len(chunks)
                for chunk_idx, chunk in enumerate(chunks):
                    write_json_row(
                        target,
                        chunked_row(
                            row,
                            config.text_field,
                            chunk,
                            input_rows,
                            chunk_idx,
                            len(chunks),
                        ),
                    )
                    output_rows += 1
    return write_summary(config, input_rows, output_rows, split_rows, chunks_created)


def parse_json_object(line: str, line_number: int) -> JsonObject:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise SplitError(detail=f"line {line_number} is not a JSON object")
    return value


def write_json_row(target: TextIO, row: JsonObject) -> None:
    _ = target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    _ = target.write("\n")


def chunked_row(
    row: JsonObject,
    text_field: str,
    chunk: str,
    row_index: int,
    chunk_idx: int,
    chunk_count: int,
) -> JsonObject:
    out = dict(row)
    out[text_field] = chunk
    identifier = out.get("id")
    if isinstance(identifier, str):
        out["id"] = f"{identifier}__chunk_{chunk_idx}"
    metadata_value = out.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
    metadata["split_parent_id"] = identifier if isinstance(identifier, str) else str(row_index)
    metadata["split_parent_row_index"] = row_index
    metadata["split_chunk_idx"] = chunk_idx
    metadata["split_chunk_count"] = chunk_count
    out["metadata"] = metadata
    return out


def write_summary(
    config: SplitConfig,
    input_rows: int,
    output_rows: int,
    split_rows: int,
    chunks_created: int,
) -> SplitSummary:
    summary = SplitSummary(
        input_path=str(config.input_path),
        output_path=str(config.output_path),
        summary_path=str(config.summary_path),
        text_field=config.text_field,
        max_tokens=config.max_tokens,
        tokenizer_name=config.tokenizer_name,
        input_rows=input_rows,
        output_rows=output_rows,
        split_rows=split_rows,
        chunks_created=chunks_created,
    )
    _ = config.summary_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def default_summary_path(output_path: Path) -> Path:
    suffix = "".join(output_path.suffixes)
    if suffix:
        return output_path.with_name(output_path.name.removesuffix(suffix) + ".summary.json")
    return output_path.with_suffix(".summary.json")
