from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import registry


@dataclass
class VLLMCompletion:
    text: str
    logprobs: Optional[List[float]] = None


class VLLMEngine:
    """Placeholder for vLLM backend."""

    name = "vllm"

    def generate(self, prompt: str, **kwargs) -> VLLMCompletion:  # pragma: no cover - requires vLLM
        raise NotImplementedError("vLLM backend requires runtime environment")

    def supports_structured_decoding(self) -> bool:
        return True


registry.register(VLLMEngine())
