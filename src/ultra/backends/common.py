"""Shared backend adapter types and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import BackendName, PrecisionName


@dataclass(slots=True)
class StructuredOutputRequest:
    enabled: bool = False
    json_schema: dict[str, Any] | None = None
    regex: str | None = None
    grammar: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationRequest:
    model_ref: str
    messages: list[dict[str, str]]
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    precision: PrecisionName = "auto"
    structured_output: StructuredOutputRequest = field(default_factory=StructuredOutputRequest)
    enable_thinking: bool = False
    stop_strings: list[str] = field(default_factory=list)
    request_logprobs: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    backend: BackendName
    text: str
    prompt_text: str
    latency_s: float
    generated_tokens: int
    avg_logprob: float | None
    finish_reason: str | None
    model_name: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def backend_description(backend: BackendName, *, available: bool, notes: list[str] | None = None) -> dict[str, Any]:
    return {
        "backend": backend,
        "available": available,
        "notes": notes or [],
    }


def render_plain_prompt(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def stop_strings_for_request(request: GenerationRequest) -> list[str]:
    stop_strings = list(request.stop_strings)
    if request.enable_thinking and "</think>" not in stop_strings:
        stop_strings.append("</think>")
    return stop_strings


def structured_decoder_name(
    structured_output: StructuredOutputRequest,
    *,
    json_name: str = "guided_json",
    regex_name: str = "guided_regex",
    grammar_name: str = "guided_grammar",
) -> str | None:
    if not structured_output.enabled:
        return None
    if structured_output.json_schema is not None:
        return json_name
    if structured_output.regex is not None:
        return regex_name
    if structured_output.grammar is not None:
        return grammar_name
    return None


def normalize_finish_reason(finish_reason: Any, *, generated_tokens: int, max_new_tokens: int) -> str:
    if isinstance(finish_reason, str) and finish_reason.strip():
        return finish_reason
    return "length" if generated_tokens >= max_new_tokens else "stop"
