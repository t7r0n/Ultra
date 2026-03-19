from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..mcd.inspector import InspectionResult


@dataclass(slots=True)
class SamplingParameters:
    temperature: float
    top_p: float
    top_k: Optional[int] = None
    max_new_tokens: int = 512
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[List[int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Completion:
    text: str
    tokens: Optional[List[str]] = None
    logprobs: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackendEngine(Protocol):
    name: str

    def generate(
        self,
        *,
        prompt: str,
        sampling: SamplingParameters,
        inspection: "InspectionResult",
        structured: Optional[Dict[str, Any]] = None,
    ) -> Completion:
        ...

    def supports_structured_decoding(self) -> bool:
        ...


class BackendRegistry:
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

# Import side effects register engines.
from . import hf_engine as _hf_engine  # noqa: F401
from . import vllm_engine as _vllm_engine  # noqa: F401
from . import sglang_engine as _sglang_engine  # noqa: F401
from . import llama_cpp_engine as _llama_cpp_engine  # noqa: F401

__all__ = ["SamplingParameters", "Completion", "BackendEngine", "registry"]

