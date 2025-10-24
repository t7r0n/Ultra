from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import registry


@dataclass
class SGLangCompletion:
    text: str
    logprobs: Optional[List[float]] = None


class SGLangEngine:
    """Placeholder SGLang backend."""

    name = "sglang"

    def generate(self, prompt: str, **kwargs) -> SGLangCompletion:  # pragma: no cover - optional
        raise NotImplementedError("SGLang backend not implemented in lightweight build")

    def supports_structured_decoding(self) -> bool:
        return True


registry.register(SGLangEngine())
