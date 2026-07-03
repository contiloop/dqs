from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Protocol

from split_boundaries import safe_word_split_bounds, text_units


WORD_RE: Final = re.compile(r"\S+")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class SplitError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def split_text(text: str, max_tokens: int, token_counter: TokenCounter) -> list[str]:
    if max_tokens <= 0:
        raise SplitError(detail="max_tokens must be > 0")
    if token_counter.count(text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current_units: list[str] = []
    current_text = ""
    for unit in text_units(text):
        unit_tokens = token_counter.count(unit.strip())
        if unit_tokens > max_tokens:
            flush_units(chunks, current_units)
            current_text = ""
            chunks.extend(split_oversized_unit_by_tokens(unit, max_tokens, token_counter))
            continue
        candidate = f"{current_text}{unit}".strip()
        if current_units and token_counter.count(candidate) > max_tokens:
            flush_units(chunks, current_units)
            current_text = ""
        current_units.append(unit)
        current_text = f"{current_text}{unit}"
    flush_units(chunks, current_units)
    return [chunk for chunk in chunks if chunk]


def split_oversized_unit_by_tokens(
    text: str,
    max_tokens: int,
    token_counter: TokenCounter,
) -> list[str]:
    chunks: list[str] = []
    chunk_start: int | None = None
    chunk_end = 0
    for match in WORD_RE.finditer(text):
        if chunk_start is None:
            chunk_start = match.start()
        candidate = text[chunk_start : match.end()].strip()
        if token_counter.count(candidate) <= max_tokens:
            chunk_end = match.end()
            continue

        if chunk_start < match.start():
            previous_end, next_start = safe_word_split_bounds(
                text,
                chunk_start,
                chunk_end,
                match.start(),
            )
            previous_chunk = text[chunk_start:previous_end].strip()
            if previous_chunk:
                chunks.append(previous_chunk)
            chunk_start = next_start
            chunk_end = match.end()
            candidate = text[chunk_start:chunk_end].strip()
            if token_counter.count(candidate) <= max_tokens:
                continue

        oversized_span = text[chunk_start : match.end()].strip()
        chunks.extend(split_oversized_span_by_tokens(oversized_span, max_tokens, token_counter))
        chunk_start = None
        chunk_end = 0

    if chunk_start is not None:
        tail = text[chunk_start:chunk_end].strip()
        if tail:
            chunks.append(tail)
    return chunks


def split_oversized_span_by_tokens(
    text: str,
    max_tokens: int,
    token_counter: TokenCounter,
) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if token_counter.count(remaining) <= max_tokens:
            chunks.append(remaining)
            break
        split_end = longest_prefix_within_tokens(remaining, max_tokens, token_counter)
        chunks.append(remaining[:split_end])
        remaining = remaining[split_end:].strip()
    return chunks


def longest_prefix_within_tokens(
    text: str,
    max_tokens: int,
    token_counter: TokenCounter,
) -> int:
    best = 0
    low = 1
    high = len(text)
    while low <= high:
        midpoint = (low + high) // 2
        if token_counter.count(text[:midpoint]) <= max_tokens:
            best = midpoint
            low = midpoint + 1
            continue
        high = midpoint - 1
    while best > 1 and token_counter.count(text[:best]) > max_tokens:
        best -= 1
    if best == 0:
        return 1
    return best


def flush_units(chunks: list[str], current_units: list[str]) -> None:
    if not current_units:
        return
    chunk = "".join(current_units).strip()
    current_units.clear()
    if chunk:
        chunks.append(chunk)
