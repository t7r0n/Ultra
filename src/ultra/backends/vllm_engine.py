"""vLLM backend implementation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import os
import time
from typing import Any
from urllib import request as urllib_request

from .common import (
    GenerationRequest,
    GenerationResult,
    StructuredOutputRequest,
    backend_description,
    normalize_finish_reason,
    render_plain_prompt,
    stop_strings_for_request,
    structured_decoder_name,
)


BACKEND_NAME = "vllm"
_LLM_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}


@dataclass(slots=True)
class VLLMSamplingRequest:
    model_ref: str
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    guided_json: dict[str, Any] | None = None
    guided_regex: str | None = None
    guided_grammar: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_backend() -> dict[str, Any]:
    available, notes = _availability()
    return backend_description(BACKEND_NAME, available=available, notes=notes)


def generate(request: GenerationRequest) -> GenerationResult:
    base_url = os.environ.get("ULTRA_VLLM_BASE_URL")
    if base_url:
        return _generate_via_server(request, base_url=base_url)
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise RuntimeError("vLLM backend requires `vllm`, `transformers`, and compatible CUDA drivers.") from exc

    tokenizer, llm = _load_runtime(
        request.model_ref,
        auto_tokenizer_cls=AutoTokenizer,
        llm_cls=LLM,
        runtime_extra=request.extra,
    )
    prompt_text = render_messages(tokenizer, request.messages, enable_thinking=request.enable_thinking)
    prompt_text, truncated_prompt_tokens = _apply_context_limit(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=request.max_new_tokens,
        max_context_tokens=request.extra.get("max_context_tokens"),
    )
    prompt_token_count = _token_count(tokenizer, prompt_text)
    sampling_kwargs: dict[str, Any] = {
        "max_tokens": _effective_max_new_tokens(
            prompt_tokens=prompt_token_count,
            requested_max_new_tokens=request.max_new_tokens,
            max_context_tokens=request.extra.get("max_context_tokens"),
        ),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
    }
    if request.top_k is not None:
        sampling_kwargs["top_k"] = request.top_k
    if request.repetition_penalty is not None:
        sampling_kwargs["repetition_penalty"] = request.repetition_penalty
    if request.request_logprobs:
        sampling_kwargs["logprobs"] = 1
    stop_strings = stop_strings_for_request(request)
    if stop_strings:
        sampling_kwargs["stop"] = stop_strings
    _apply_structured_output(sampling_kwargs, request.structured_output)
    params = SamplingParams(**sampling_kwargs)

    start = time.perf_counter()
    outputs = llm.generate([prompt_text], params, use_tqdm=False)
    latency_s = time.perf_counter() - start
    output = outputs[0].outputs[0]
    avg_logprob = None
    cumulative_logprob = getattr(output, "cumulative_logprob", None)
    token_ids = getattr(output, "token_ids", None) or []
    if cumulative_logprob is not None and token_ids:
        avg_logprob = float(cumulative_logprob) / max(len(token_ids), 1)
    return GenerationResult(
        backend=BACKEND_NAME,
        text=output.text.strip(),
        prompt_text=prompt_text,
        latency_s=latency_s,
        generated_tokens=len(token_ids),
        avg_logprob=avg_logprob,
        finish_reason=normalize_finish_reason(
            getattr(output, "finish_reason", None),
            generated_tokens=len(token_ids),
            max_new_tokens=request.max_new_tokens,
        ),
        model_name=request.model_ref,
        usage={
            "prompt_tokens": len(getattr(outputs[0], "prompt_token_ids", []) or []),
            "completion_tokens": len(token_ids),
            "total_tokens": len(getattr(outputs[0], "prompt_token_ids", []) or []) + len(token_ids),
        },
        metrics={
            "latency_s": latency_s,
            "tokens_per_s": (len(token_ids) / latency_s) if latency_s > 0 else None,
            "max_context_tokens": request.extra.get("max_context_tokens"),
            "truncated_prompt_tokens": truncated_prompt_tokens,
            "stop_strings": stop_strings,
            "logprobs_requested": request.request_logprobs,
            "peak_memory_bytes": None,
            "structured_decoder": structured_decoder_name(request.structured_output),
        },
        raw={"output": _safe_jsonable(output)},
    )


def render_messages(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        import inspect

        signature = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in signature.parameters:
            kwargs["enable_thinking"] = enable_thinking
    except Exception:
        pass
    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except Exception:
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def _apply_structured_output(sampling_kwargs: dict[str, Any], structured_output: StructuredOutputRequest) -> None:
    if not structured_output.enabled:
        return
    if structured_output.json_schema is not None:
        sampling_kwargs["guided_json"] = structured_output.json_schema
    if structured_output.regex is not None:
        sampling_kwargs["guided_regex"] = structured_output.regex
    if structured_output.grammar is not None:
        sampling_kwargs["guided_grammar"] = structured_output.grammar


def _load_runtime(
    model_ref: str,
    *,
    auto_tokenizer_cls: Any,
    llm_cls: Any,
    runtime_extra: dict[str, Any],
) -> tuple[Any, Any]:
    rope_scaling_override = runtime_extra.get("rope_scaling_override") if isinstance(runtime_extra.get("rope_scaling_override"), dict) else None
    max_context_tokens = runtime_extra.get("max_context_tokens")
    cache_key = (
        model_ref,
        json.dumps(rope_scaling_override, sort_keys=True) if rope_scaling_override is not None else "",
        str(max_context_tokens) if isinstance(max_context_tokens, int) else "",
    )
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]
    tokenizer = auto_tokenizer_cls.from_pretrained(model_ref, trust_remote_code=True)
    llm_kwargs: dict[str, Any] = {"model": model_ref, "trust_remote_code": True}
    if isinstance(max_context_tokens, int) and max_context_tokens > 0:
        llm_kwargs["max_model_len"] = max_context_tokens
    if rope_scaling_override is not None:
        llm_kwargs["hf_overrides"] = {"rope_scaling": rope_scaling_override}
    llm = llm_cls(**llm_kwargs)
    _LLM_CACHE[cache_key] = (tokenizer, llm)
    return tokenizer, llm


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "__dict__"):
            return {key: _safe_jsonable(item) for key, item in value.__dict__.items()}
        return str(value)


def _availability() -> tuple[bool, list[str]]:
    base_url = os.environ.get("ULTRA_VLLM_BASE_URL")
    if base_url:
        return True, [f"using server {base_url}"]
    try:
        import vllm  # noqa: F401
        import transformers  # noqa: F401

        return True, []
    except ModuleNotFoundError as exc:
        return False, [f"missing dependency: {exc.name}"]


def _effective_max_new_tokens(*, prompt_tokens: int, requested_max_new_tokens: int, max_context_tokens: Any) -> int:
    if not isinstance(max_context_tokens, int) or max_context_tokens <= 0:
        return requested_max_new_tokens
    remaining = max_context_tokens - prompt_tokens
    return max(1, min(requested_max_new_tokens, remaining))


def _apply_context_limit(
    *,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    max_context_tokens: Any,
) -> tuple[str, int]:
    if not isinstance(max_context_tokens, int) or max_context_tokens <= 0:
        return prompt_text, 0
    if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
        return prompt_text, 0
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    max_input_tokens = max(max_context_tokens - max_new_tokens, 1)
    if len(prompt_ids) <= max_input_tokens:
        return prompt_text, 0
    truncated = len(prompt_ids) - max_input_tokens
    trimmed_prompt = tokenizer.decode(prompt_ids[-max_input_tokens:], skip_special_tokens=False)
    return str(trimmed_prompt), truncated


def _token_count(tokenizer: Any, prompt_text: str) -> int:
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(prompt_text, add_special_tokens=False))
        except Exception:
            pass
    return len(prompt_text.split())


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
    if request.top_k is not None:
        payload["top_k"] = request.top_k
    if request.repetition_penalty is not None:
        payload["repetition_penalty"] = request.repetition_penalty
    if request.request_logprobs:
        payload["logprobs"] = True
    if stop_strings:
        payload["stop"] = stop_strings
    _apply_structured_output(payload, request.structured_output)
    prompt_text = render_plain_prompt(request.messages)
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
            "transport": "server",
        },
        raw=response_payload,
    )


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib_request.urlopen(req, timeout=600) as handle:
        return json.loads(handle.read().decode("utf-8"))


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
