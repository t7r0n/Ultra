"""SGLang backend implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
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


BACKEND_NAME = "sglang"


@dataclass(slots=True)
class SGLangSamplingRequest:
    model_ref: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_backend() -> dict[str, Any]:
    available, notes = _availability()
    return backend_description(BACKEND_NAME, available=available, notes=notes)


def generate(request: GenerationRequest) -> GenerationResult:
    base_url = os.environ.get("ULTRA_SGLANG_BASE_URL")
    if not base_url:
        raise RuntimeError("Set ULTRA_SGLANG_BASE_URL to an OpenAI-compatible SGLang server.")
    payload = {
        "model": request.model_ref,
        "messages": request.messages,
        "max_tokens": request.max_new_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
    }
    prompt_text = render_plain_prompt(request.messages)
    if request.top_k is not None:
        payload["top_k"] = request.top_k
    if request.request_logprobs:
        payload["logprobs"] = True
    stop_strings = stop_strings_for_request(request)
    if stop_strings:
        payload["stop"] = stop_strings
    if request.structured_output.enabled:
        if request.structured_output.json_schema is not None:
            payload["guided_json"] = request.structured_output.json_schema
        if request.structured_output.regex is not None:
            payload["guided_regex"] = request.structured_output.regex
        if request.structured_output.grammar is not None:
            payload["guided_grammar"] = request.structured_output.grammar
    start = time.perf_counter()
    response_payload = _post_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload)
    latency_s = time.perf_counter() - start
    choice = response_payload["choices"][0]
    usage = response_payload.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0)) or estimate_completion_tokens(choice["message"]["content"])
    avg_logprob = average_openai_logprobs(choice.get("logprobs"))
    return GenerationResult(
        backend=BACKEND_NAME,
        text=choice["message"]["content"].strip(),
        prompt_text=prompt_text,
        latency_s=latency_s,
        generated_tokens=completion_tokens,
        avg_logprob=avg_logprob,
        finish_reason=normalize_finish_reason(
            choice.get("finish_reason"),
            generated_tokens=completion_tokens,
            max_new_tokens=request.max_new_tokens,
        ),
        model_name=response_payload.get("model", request.model_ref),
        usage=usage,
        metrics={
            "latency_s": latency_s,
            "tokens_per_s": (completion_tokens / latency_s) if latency_s > 0 else None,
            "max_context_tokens": request.extra.get("max_context_tokens"),
            "truncated_prompt_tokens": 0,
            "stop_strings": stop_strings,
            "logprobs_requested": request.request_logprobs,
            "peak_memory_bytes": None,
            "structured_decoder": structured_decoder_name(request.structured_output),
        },
        raw=response_payload,
    )


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib_request.urlopen(req, timeout=600) as handle:
        return json.loads(handle.read().decode("utf-8"))


def _availability() -> tuple[bool, list[str]]:
    base_url = os.environ.get("ULTRA_SGLANG_BASE_URL")
    if base_url:
        return True, [f"using server {base_url}"]
    return False, ["set ULTRA_SGLANG_BASE_URL to an OpenAI-compatible SGLang server"]


def average_openai_logprobs(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    values = [item.get("logprob") for item in content if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float))]
    if not values:
        return None
    return float(sum(values) / len(values))


def estimate_completion_tokens(text: str) -> int:
    return max(len(text.split()), 1) if text.strip() else 0
