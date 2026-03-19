"""llama.cpp backend implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib import request as urllib_request

from .common import (
    GenerationRequest,
    GenerationResult,
    backend_description,
    normalize_finish_reason,
    render_plain_prompt,
    stop_strings_for_request,
    structured_decoder_name,
)


BACKEND_NAME = "llamacpp"
_MODEL_CACHE: dict[tuple[str, int], Any] = {}


@dataclass(slots=True)
class LlamaCppSamplingRequest:
    model_ref: str
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int | None = None
    repeat_penalty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_backend() -> dict[str, Any]:
    available, notes = _availability()
    return backend_description(BACKEND_NAME, available=available, notes=notes)


def generate(request: GenerationRequest) -> GenerationResult:
    base_url = os.environ.get("ULTRA_LLAMACPP_BASE_URL")
    if base_url:
        return _generate_via_server(request, base_url=base_url)
    try:
        from llama_cpp import Llama
    except ModuleNotFoundError as exc:
        raise RuntimeError("llama.cpp backend requires `llama-cpp-python`.") from exc

    model_path = _resolve_model_path(request.model_ref)
    llm = _load_model(
        str(model_path),
        request.max_new_tokens,
        max_context_tokens=request.extra.get("max_context_tokens"),
        llama_cls=Llama,
    )
    start = time.perf_counter()
    stop_strings = stop_strings_for_request(request)
    prompt_text = render_plain_prompt(request.messages)
    response = llm.create_chat_completion(
        messages=request.messages,
        max_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repeat_penalty=request.repetition_penalty or 1.0,
        stop=stop_strings or None,
        seed=request.seed,
    )
    latency_s = time.perf_counter() - start
    choice = response["choices"][0]
    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0)) or estimate_completion_tokens(choice["message"]["content"])
    return GenerationResult(
        backend=BACKEND_NAME,
        text=choice["message"]["content"].strip(),
        prompt_text=prompt_text,
        latency_s=latency_s,
        generated_tokens=completion_tokens,
        avg_logprob=None,
        finish_reason=normalize_finish_reason(
            choice.get("finish_reason"),
            generated_tokens=completion_tokens,
            max_new_tokens=request.max_new_tokens,
        ),
        model_name=str(model_path),
        usage=usage,
        metrics={
            "latency_s": latency_s,
            "tokens_per_s": (completion_tokens / latency_s) if latency_s > 0 else None,
            "max_context_tokens": request.extra.get("max_context_tokens"),
            "truncated_prompt_tokens": 0,
            "stop_strings": stop_strings,
            "logprobs_requested": request.request_logprobs,
            "peak_memory_bytes": None,
            "structured_decoder": structured_decoder_name(
                request.structured_output,
                json_name="json_repair",
                regex_name="regex_repair",
                grammar_name="grammar_prompt_only",
            ),
        },
        raw=response,
    )


def _resolve_model_path(model_ref: str) -> Path:
    candidate = Path(model_ref)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for gguf in sorted(candidate.glob("*.gguf")):
            return gguf
    raise RuntimeError(f"Could not resolve GGUF model path from {model_ref!r}")


def _load_model(model_path: str, max_tokens: int, *, max_context_tokens: Any, llama_cls: Any) -> Any:
    resolved_context = max_context_tokens if isinstance(max_context_tokens, int) and max_context_tokens > 0 else max(4096, max_tokens * 4)
    cache_key = (model_path, max_tokens, resolved_context)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    llm = llama_cls(model_path=model_path, n_ctx=resolved_context, verbose=False)
    _MODEL_CACHE[cache_key] = llm
    return llm


def _availability() -> tuple[bool, list[str]]:
    base_url = os.environ.get("ULTRA_LLAMACPP_BASE_URL")
    if base_url:
        return True, [f"using server {base_url}"]
    try:
        import llama_cpp  # noqa: F401

        return True, []
    except ModuleNotFoundError as exc:
        return False, [f"missing dependency: {exc.name}"]


def estimate_completion_tokens(text: str) -> int:
    return max(len(text.split()), 1) if text.strip() else 0


def _generate_via_server(request: GenerationRequest, *, base_url: str) -> GenerationResult:
    stop_strings = stop_strings_for_request(request)
    payload: dict[str, Any] = {
        "model": request.model_ref,
        "messages": request.messages,
        "max_tokens": request.max_new_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
    }
    if request.repetition_penalty is not None:
        payload["repeat_penalty"] = request.repetition_penalty
    if stop_strings:
        payload["stop"] = stop_strings
    prompt_text = render_plain_prompt(request.messages)
    start = time.perf_counter()
    response = _post_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload)
    latency_s = time.perf_counter() - start
    choice = response["choices"][0]
    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0)) or estimate_completion_tokens(choice["message"]["content"])
    return GenerationResult(
        backend=BACKEND_NAME,
        text=choice["message"]["content"].strip(),
        prompt_text=prompt_text,
        latency_s=latency_s,
        generated_tokens=completion_tokens,
        avg_logprob=None,
        finish_reason=normalize_finish_reason(
            choice.get("finish_reason"),
            generated_tokens=completion_tokens,
            max_new_tokens=request.max_new_tokens,
        ),
        model_name=response.get("model", request.model_ref),
        usage=usage,
        metrics={
            "latency_s": latency_s,
            "tokens_per_s": (completion_tokens / latency_s) if latency_s > 0 else None,
            "max_context_tokens": request.extra.get("max_context_tokens"),
            "truncated_prompt_tokens": 0,
            "stop_strings": stop_strings,
            "logprobs_requested": request.request_logprobs,
            "peak_memory_bytes": None,
            "structured_decoder": structured_decoder_name(
                request.structured_output,
                json_name="json_repair",
                regex_name="regex_repair",
                grammar_name="grammar_prompt_only",
            ),
            "transport": "server",
        },
        raw=response,
    )


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib_request.urlopen(req, timeout=600) as handle:
        return json.loads(handle.read().decode("utf-8"))
