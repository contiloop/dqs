from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def text_tokenizer(tokenizer_or_processor: Any) -> Any:
    backend = getattr(tokenizer_or_processor, "tokenizer", None)
    if backend is not None and callable(backend):
        return backend
    return tokenizer_or_processor


def text_token_ids(tokenizer_or_processor: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    tokenizer = text_tokenizer(tokenizer_or_processor)
    try:
        encoded = tokenizer(text=text, add_special_tokens=add_special_tokens)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)

    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
    else:
        input_ids = getattr(encoded, "input_ids", None)
        if input_ids is None and hasattr(encoded, "ids"):
            input_ids = encoded.ids
    if input_ids is None:
        raise TypeError(f"tokenizer output does not contain input_ids: {type(encoded)!r}")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def text_decode(
    tokenizer_or_processor: Any,
    token_ids: list[int],
    *,
    skip_special_tokens: bool = False,
) -> str:
    tokenizer = text_tokenizer(tokenizer_or_processor)
    return str(tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))
