"""Backend registry for ULTRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol


class CompletionResult(Protocol):
    """Protocol for backend generation results."""

    text: str
    logprobs: Optional[List[float]]


class BackendEngine(Protocol):
    """Protocol describing minimum backend surface required by Ultra Mode."""

    name: str

    def generate(self, prompt: str, **kwargs) -> CompletionResult:
        ...

    def supports_structured_decoding(self) -> bool:
        ...


@dataclass(slots=True)
class SamplingParameters:
    """Backend-agnostic sampling parameters."""

    temperature: float
    top_p: float
    max_tokens: int
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class BackendRegistry:
    """Simple backend registry to allow dependency injection within tests."""

    def __init__(self) -> None:
        self._engines: Dict[str, BackendEngine] = {}

    def register(self, engine: BackendEngine) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> BackendEngine:
        try:
            return self._engines[name]
        except KeyError as exc:
            raise RuntimeError(f"Backend '{name}' is not registered") from exc

    def names(self) -> Iterable[str]:
        return self._engines.keys()


registry = BackendRegistry()

# Import side effects register default engines.
from . import hf_engine as _hf_engine  # noqa: F401  (imported for side effects)
from . import vllm_engine as _vllm_engine  # noqa: F401
from . import sglang_engine as _sglang_engine  # noqa: F401
from . import llama_cpp_engine as _llama_cpp_engine  # noqa: F401

__all__ = ["SamplingParameters", "registry", "BackendEngine", "CompletionResult"]
