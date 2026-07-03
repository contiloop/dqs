from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


@dataclass(frozen=True, slots=True)
class TokenCounterLoadError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class QwenTokenCounter:
    tokenizer: Tokenizer
    name: str

    @classmethod
    def from_path(cls, path: Path) -> QwenTokenCounter:
        tokenizer_path = path / "tokenizer.json" if path.is_dir() else path
        if not tokenizer_path.exists():
            raise TokenCounterLoadError(detail=f"tokenizer path does not exist: {tokenizer_path}")
        return cls(
            tokenizer=Tokenizer.from_file(str(tokenizer_path)),
            name=f"tokenizer-file:{tokenizer_path}",
        )

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        *,
        revision: str = "main",
        token: str | None = None,
    ) -> QwenTokenCounter:
        return cls(
            tokenizer=Tokenizer.from_pretrained(identifier, revision=revision, token=token),
            name=f"hf-tokenizer:{identifier}@{revision}",
        )

    @classmethod
    def from_model_or_path(
        cls,
        model_or_path: str,
        *,
        revision: str = "main",
        token: str | None = None,
    ) -> QwenTokenCounter:
        local_path = Path(model_or_path).expanduser()
        if local_path.exists():
            return cls.from_path(local_path)
        return cls.from_pretrained(model_or_path, revision=revision, token=token)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
