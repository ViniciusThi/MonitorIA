from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResult:
    raw: str
    provider: str


class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str) -> LLMResult: ...


class LLMUnavailableError(RuntimeError):
    pass

