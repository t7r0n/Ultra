from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import registry


@dataclass
class LlamaCPPCompletion:
    text: str
    logprobs: Optional[List[float]] = None


class LlamaCPPEngine:
    """Placeholder llama.cpp backend."""

    name = "llamacpp"

    def generate(self, prompt: str, **kwargs) -> LlamaCPPCompletion:  # pragma: no cover - optional
        raise NotImplementedError("llama.cpp backend requires llama_cpp_python")

    def supports_structured_decoding(self) -> bool:
        return False


registry.register(LlamaCPPEngine())
