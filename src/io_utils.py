from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8", newline="\n") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, Mapping):
                raise ValueError(f"{path_obj}:{line_no} must contain a JSON object")
            rows.append(dict(parsed))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path_obj.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            serialized = json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(serialized)
            handle.write("\n")
            count += 1
    return count
