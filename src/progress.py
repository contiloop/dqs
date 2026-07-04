from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Iterator
from typing import Any


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def progress_enabled() -> bool:
    return _env_flag("DQS_PROGRESS", True)


def progress(event: str, **fields: Any) -> None:
    if not progress_enabled():
        return
    parts = [f"[dqs] {event}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", "\\n")
        parts.append(f"{key}={text}")
    print(" ".join(parts), file=sys.stderr, flush=True)


@contextlib.contextmanager
def progress_context(event: str, **fields: Any) -> Iterator[None]:
    started = time.perf_counter()
    progress(f"{event} start", **fields)
    try:
        yield
    except BaseException as exc:
        progress(
            f"{event} failed",
            elapsed_s=f"{time.perf_counter() - started:.1f}",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    progress(f"{event} done", elapsed_s=f"{time.perf_counter() - started:.1f}")
