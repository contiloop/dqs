from __future__ import annotations

from collections.abc import Iterator
import re
from typing import Final


BOUNDARY_CHARS: Final = frozenset(".!?。！？")
CLOSING_CHARS: Final = frozenset("\"'”’)]}»")
OPENING_CHARS: Final = frozenset("\"'“‘([{«")
TOKEN_CHARS: Final = frozenset(".&")
ABBREVIATIONS: Final = frozenset(
    "apr aug co corp dec dr e.g e.u etc feb gov h.r i.e inc jan jr jul jun ltd mar "
    "mr mrs ms no nov oct plc prof rep sen sept sep sr st u.k u.n u.s vs www".split()
)
PARAGRAPH_BREAK_RE: Final = re.compile(r"\n[ \t]*\n+")
WORD_RE: Final = re.compile(r"\S+")
DOTTED_INITIALISM_RE: Final = re.compile(r"(?:[a-z]\.){1,4}")
TRAILING_TOKEN_STRIP_CHARS: Final = "\"'“”‘’)]}»,;:"


def text_units(text: str) -> list[str]:
    ends = boundary_indexes(text)
    units: list[str] = []
    start = 0
    for end in ends:
        if end <= start:
            continue
        units.append(text[start:end])
        start = end
    if start < len(text):
        units.append(text[start:])
    return [unit for unit in units if unit]


def boundary_indexes(text: str) -> list[int]:
    indexes = {len(text)}
    indexes.update(match.end() for match in PARAGRAPH_BREAK_RE.finditer(text))
    indexes.update(sentence_boundary_indexes(text))
    return sorted(indexes)


def sentence_boundary_indexes(text: str) -> Iterator[int]:
    idx = 0
    while idx < len(text):
        if text[idx] in BOUNDARY_CHARS:
            boundary_end = closing_suffix_end(text, idx + 1)
            if is_sentence_boundary(text, idx, boundary_end):
                yield boundary_end
                idx = boundary_end
                continue
        idx += 1


def closing_suffix_end(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and text[idx] in CLOSING_CHARS:
        idx += 1
    return idx


def skip_spaces(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def is_sentence_boundary(text: str, punct_idx: int, boundary_end: int) -> bool:
    punctuation = text[punct_idx]
    if punctuation == "." and is_leading_dot_token(text, punct_idx):
        return False
    if punctuation == "." and is_decimal_point(text, punct_idx):
        return False
    if punctuation == "." and is_ellipsis_point(text, punct_idx):
        return False
    if punctuation == "." and is_embedded_token_period(text, punct_idx):
        return False
    if punctuation == "." and is_abbreviation_or_initial(text, punct_idx):
        return False
    if punctuation == "." and is_dotted_abbreviation_or_initialism(text, punct_idx):
        return False

    next_idx = skip_spaces(text, boundary_end)
    if next_idx >= len(text):
        return True
    next_char = text[next_idx]
    if next_char.islower():
        return False
    return next_char.isupper() or next_char.isdigit() or next_char in OPENING_CHARS


def is_decimal_point(text: str, punct_idx: int) -> bool:
    return (
        punct_idx > 0
        and punct_idx + 1 < len(text)
        and text[punct_idx - 1].isdigit()
        and text[punct_idx + 1].isdigit()
    )


def is_leading_dot_token(text: str, punct_idx: int) -> bool:
    return (
        punct_idx + 1 < len(text)
        and text[punct_idx + 1].isalnum()
        and (punct_idx == 0 or not text[punct_idx - 1].isalnum())
    )


def is_ellipsis_point(text: str, punct_idx: int) -> bool:
    return (
        (punct_idx > 0 and text[punct_idx - 1] == ".")
        or (punct_idx + 1 < len(text) and text[punct_idx + 1] == ".")
    )


def is_embedded_token_period(text: str, punct_idx: int) -> bool:
    return (
        punct_idx > 0
        and punct_idx + 1 < len(text)
        and text[punct_idx - 1].isalnum()
        and text[punct_idx + 1].isalnum()
    )


def is_abbreviation_or_initial(text: str, punct_idx: int) -> bool:
    token = token_before_period(text, punct_idx).lower()
    if len(token) == 1 and token.isalpha():
        return True
    return token in ABBREVIATIONS


def is_dotted_abbreviation_or_initialism(text: str, punct_idx: int) -> bool:
    token = token_before_period(text, punct_idx).strip(".")
    parts = token.split(".")
    return len(parts) > 1 and all(part.isalpha() and 1 <= len(part) <= 3 for part in parts)


def token_before_period(text: str, punct_idx: int) -> str:
    idx = punct_idx - 1
    while idx >= 0 and (text[idx].isalnum() or text[idx] in TOKEN_CHARS):
        idx -= 1
    return text[idx + 1 : punct_idx].strip(".")


def safe_word_split_bounds(
    text: str,
    chunk_start: int,
    chunk_end: int,
    next_start: int,
) -> tuple[int, int]:
    tokens = list(WORD_RE.finditer(text, chunk_start, chunk_end))
    if len(tokens) < 2:
        return chunk_end, next_start
    unsafe_token = tokens[-1]
    if not is_unsafe_trailing_token(unsafe_token.group(0)):
        return chunk_end, next_start
    return tokens[-2].end(), unsafe_token.start()


def is_unsafe_trailing_token(token: str) -> bool:
    normalized = token.strip(TRAILING_TOKEN_STRIP_CHARS).lower()
    if normalized in {".", "www.", "h.r."}:
        return True
    if normalized.endswith("://www."):
        return True
    if normalized.endswith(".") and normalized[:-1] in ABBREVIATIONS:
        return True
    return DOTTED_INITIALISM_RE.fullmatch(normalized) is not None
