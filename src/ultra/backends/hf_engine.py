"""Transformers backend implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
import re
import time
from typing import Any

from .common import (
    GenerationRequest,
    GenerationResult,
    StructuredOutputRequest,
    backend_description,
    normalize_finish_reason,
    stop_strings_for_request,
)
from ..config import PrecisionName


BACKEND_NAME = "hf"
_MODEL_CACHE: dict[tuple[str, PrecisionName, str, str], tuple[Any, Any]] = {}


@dataclass(slots=True)
class HFSamplingRequest:
    model_ref: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    structured_output: StructuredOutputRequest | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_backend() -> dict[str, Any]:
    available, notes = _availability()
    return backend_description(BACKEND_NAME, available=available, notes=notes)


def generate(request: GenerationRequest, *, device: str | None = None) -> GenerationResult:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("Transformers backend requires `torch` and `transformers`.") from exc

    tokenizer, model = _load_model(
        request.model_ref,
        precision=request.precision,
        device=device,
        rope_scaling_override=_rope_scaling_override(request),
        auto_model_cls=AutoModelForCausalLM,
        auto_config_cls=AutoConfig,
        auto_tokenizer_cls=AutoTokenizer,
    )
    prompt_text = render_messages(tokenizer, request.messages, enable_thinking=request.enable_thinking)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs, prompt_text, truncated_prompt_tokens = _apply_context_limit(
        tokenizer=tokenizer,
        inputs=inputs,
        prompt_text=prompt_text,
        max_new_tokens=request.max_new_tokens,
        max_context_tokens=request.extra.get("max_context_tokens"),
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    _set_reproducibility(request.seed)

    effective_max_new_tokens = _effective_max_new_tokens(
        prompt_tokens=int(inputs["input_ids"].shape[1]),
        requested_max_new_tokens=request.max_new_tokens,
        max_context_tokens=request.extra.get("max_context_tokens"),
    )
    requested_stop_strings = stop_strings_for_request(request)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": effective_max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": bool(request.request_logprobs),
        "renormalize_logits": bool(request.request_logprobs),
        "do_sample": request.temperature > 0,
        "temperature": max(request.temperature, 1e-5),
        "top_p": request.top_p,
    }
    if request.top_k is not None:
        generate_kwargs["top_k"] = request.top_k
    if request.min_p is not None:
        generate_kwargs["min_p"] = request.min_p
    if request.repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = request.repetition_penalty
    stop_sequences = encode_stop_strings(tokenizer, requested_stop_strings)
    if stop_sequences:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnTokenSequences(StoppingCriteria):
            def __init__(self, sequences: list[list[int]]) -> None:
                self.sequences = [tuple(sequence) for sequence in sequences if sequence]

            def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
                del scores, kwargs
                for sequence in self.sequences:
                    length = len(sequence)
                    if input_ids.shape[1] < length:
                        continue
                    if input_ids[0, -length:].tolist() == list(sequence):
                        return True
                return False

        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])
    structured_decoder = "repair"
    prefix_allowed_tokens_fn = structured_prefix_allowed_tokens_fn(tokenizer, request.structured_output)
    if prefix_allowed_tokens_fn is not None:
        structured_decoder = "lm-format-enforcer"
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

    outlined = generate_with_outlines(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        request=request,
        effective_max_new_tokens=effective_max_new_tokens,
    )
    if outlined is not None:
        outlined.metrics.update(
            {
                "device": str(model.device),
                "max_context_tokens": request.extra.get("max_context_tokens"),
                "stop_strings": requested_stop_strings,
                "logprobs_requested": request.request_logprobs,
                "structured_decoder": "outlines",
                "truncated_prompt_tokens": truncated_prompt_tokens,
            }
        )
        return outlined

    if inputs["input_ids"].is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(inputs["input_ids"].device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)
    if inputs["input_ids"].is_cuda:
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - start
    peak_memory_bytes = None
    if inputs["input_ids"].is_cuda:
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(inputs["input_ids"].device))

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    avg_logprob = _average_logprob(model, outputs, generated_ids) if request.request_logprobs else None
    metrics = {
        "latency_s": latency_s,
        "tokens_per_s": (float(generated_ids.shape[0]) / latency_s) if latency_s > 0 else None,
        "device": str(model.device),
        "max_context_tokens": request.extra.get("max_context_tokens"),
        "truncated_prompt_tokens": truncated_prompt_tokens,
        "stop_strings": requested_stop_strings,
        "logprobs_requested": request.request_logprobs,
        "peak_memory_bytes": peak_memory_bytes,
        "structured_decoder": structured_decoder,
    }

    result = GenerationResult(
        backend=BACKEND_NAME,
        text=text,
        prompt_text=prompt_text,
        latency_s=latency_s,
        generated_tokens=int(generated_ids.shape[0]),
        avg_logprob=avg_logprob,
        finish_reason=normalize_finish_reason(
            "stop" if generated_ids.numel() < effective_max_new_tokens else "length",
            generated_tokens=int(generated_ids.shape[0]),
            max_new_tokens=effective_max_new_tokens,
        ),
        model_name=request.model_ref,
        usage={
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated_ids.shape[0]),
            "total_tokens": int(inputs["input_ids"].shape[1] + generated_ids.shape[0]),
        },
        metrics=metrics,
        raw={"text": text, "structured_decoder": structured_decoder},
    )
    return maybe_repair_structured_output(result, request, tokenizer=tokenizer, model=model)


def render_messages(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in signature.parameters:
            kwargs["enable_thinking"] = enable_thinking
    except (TypeError, ValueError):
        pass
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        return str(rendered)
    except Exception:
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def maybe_repair_structured_output(result: GenerationResult, request: GenerationRequest, *, tokenizer: Any, model: Any) -> GenerationResult:
    if not request.structured_output.enabled:
        return result
    if request.structured_output.json_schema is not None:
        try:
            payload = json.loads(result.text)
            if isinstance(payload, dict):
                return result
        except Exception:
            pass
    elif request.structured_output.regex is not None:
        if re.fullmatch(request.structured_output.regex, result.text.strip(), flags=re.DOTALL):
            return result
    else:
        return result

    repair_messages = [
        *request.messages,
        {"role": "assistant", "content": result.text},
        {
            "role": "user",
            "content": repair_instruction(request.structured_output),
        },
    ]
    repair_request = GenerationRequest(
        model_ref=request.model_ref,
        messages=repair_messages,
        max_new_tokens=min(256, request.max_new_tokens),
        temperature=0.0,
        top_p=1.0,
        top_k=request.top_k,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
        precision=request.precision,
        structured_output=StructuredOutputRequest(enabled=False),
        enable_thinking=False,
        stop_strings=request.stop_strings,
        request_logprobs=request.request_logprobs,
    )
    repaired = generate(repair_request, device=str(model.device))
    repaired.raw["repaired_from"] = result.text
    return repaired
def structured_prefix_allowed_tokens_fn(tokenizer: Any, structured_output: StructuredOutputRequest) -> Any | None:
    if structured_output.grammar is not None:
        return None
    if structured_output.json_schema is None and structured_output.regex is None:
        return None
    try:
        from lmformatenforcer import JsonSchemaParser, RegexParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    except Exception:
        return None
    try:
        if structured_output.json_schema is not None:
            parser = JsonSchemaParser(structured_output.json_schema)
        else:
            parser = RegexParser(structured_output.regex or "")
        return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    except Exception:
        return None


def generate_with_outlines(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    request: GenerationRequest,
    effective_max_new_tokens: int,
) -> GenerationResult | None:
    if request.structured_output.grammar is None:
        return None
    try:
        import outlines  # type: ignore
        from outlines.types import CFG  # type: ignore
    except Exception:
        return None
    if not hasattr(outlines, "from_transformers") or not hasattr(outlines, "Generator"):
        return None
    try:
        structured_model = outlines.from_transformers(model, tokenizer)
        generator = outlines.Generator(structured_model, CFG(request.structured_output.grammar))
        start = time.perf_counter()
        generated = generator(
            prompt_text,
            max_new_tokens=effective_max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        latency_s = time.perf_counter() - start
    except Exception:
        return None
    text = generated if isinstance(generated, str) else json.dumps(generated, sort_keys=True)
    return GenerationResult(
        backend=BACKEND_NAME,
        text=str(text).strip(),
        prompt_text=prompt_text,
        latency_s=latency_s,
        generated_tokens=len(tokenizer.encode(str(text), add_special_tokens=False)) if hasattr(tokenizer, "encode") else len(str(text).split()),
        avg_logprob=None,
        finish_reason="stop",
        model_name=request.model_ref,
        usage={
            "prompt_tokens": len(tokenizer.encode(prompt_text, add_special_tokens=False)) if hasattr(tokenizer, "encode") else len(prompt_text.split()),
            "completion_tokens": len(tokenizer.encode(str(text), add_special_tokens=False)) if hasattr(tokenizer, "encode") else len(str(text).split()),
            "total_tokens": (
                (len(tokenizer.encode(prompt_text, add_special_tokens=False)) if hasattr(tokenizer, "encode") else len(prompt_text.split()))
                + (len(tokenizer.encode(str(text), add_special_tokens=False)) if hasattr(tokenizer, "encode") else len(str(text).split()))
            ),
        },
        metrics={"latency_s": latency_s, "tokens_per_s": None},
        raw={"text": str(text), "structured_decoder": "outlines"},
    )


def repair_instruction(structured_output: StructuredOutputRequest) -> str:
    if structured_output.json_schema is not None:
        return (
            "Return only valid JSON matching this schema description: "
            f"{structured_output.json_schema}"
        )
    if structured_output.regex is not None:
        return f"Return only text matching this regular expression exactly: {structured_output.regex}"
    if structured_output.grammar is not None:
        return "Return only text that satisfies the requested grammar."
    return "Return only the final answer."


def _load_model(
    model_ref: str,
    *,
    precision: PrecisionName,
    device: str | None,
    rope_scaling_override: dict[str, Any] | None,
    auto_model_cls: Any,
    auto_config_cls: Any,
    auto_tokenizer_cls: Any,
) -> tuple[Any, Any]:
    cache_key = (
        model_ref,
        precision,
        device or "auto",
        json.dumps(rope_scaling_override, sort_keys=True) if rope_scaling_override is not None else "",
    )
    if cache_key in _MODEL_CACHE:
        tokenizer, model = _MODEL_CACHE[cache_key]
        return tokenizer, model

    dtype = _torch_dtype(precision)
    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if device is None:
        model_kwargs["device_map"] = "auto"
    tokenizer = auto_tokenizer_cls.from_pretrained(model_ref, trust_remote_code=True)
    config = None
    if rope_scaling_override is not None:
        config = auto_config_cls.from_pretrained(model_ref, trust_remote_code=True)
        config.rope_scaling = rope_scaling_override
    if config is not None:
        model_kwargs["config"] = config
    model = auto_model_cls.from_pretrained(model_ref, trust_remote_code=True, **model_kwargs)
    if device is not None:
        model = model.to(device)
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _average_logprob(model: Any, outputs: Any, generated_ids: Any) -> float | None:
    if not getattr(outputs, "scores", None):
        return None
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)[0]
    transition_scores = transition_scores[-generated_ids.shape[0] :] if generated_ids.shape[0] else transition_scores[:0]
    return float(transition_scores.mean().item()) if transition_scores.numel() else None


def _torch_dtype(precision: PrecisionName) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp16", "auto"}:
        return torch.float16
    return None


def _set_reproducibility(seed: int | None) -> None:
    if seed is None:
        return
    import torch
    from transformers import set_seed

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def encode_stop_strings(tokenizer: Any, stop_strings: list[str]) -> list[list[int]]:
    sequences: list[list[int]] = []
    for stop_string in stop_strings:
        encoded = tokenizer.encode(stop_string, add_special_tokens=False)
        if encoded and encoded not in sequences:
            sequences.append([int(token_id) for token_id in encoded])
    return sequences


def think_end_stop_sequences(tokenizer: Any) -> list[list[int]]:
    sequences: list[list[int]] = []
    for variant in ("</think>", "\n</think>", "</think>\n", "\n</think>\n"):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded and encoded not in sequences:
            sequences.append([int(token_id) for token_id in encoded])
    return sequences


def _rope_scaling_override(request: GenerationRequest) -> dict[str, Any] | None:
    value = request.extra.get("rope_scaling_override")
    return value if isinstance(value, dict) else None


def _effective_max_new_tokens(*, prompt_tokens: int, requested_max_new_tokens: int, max_context_tokens: Any) -> int:
    if not isinstance(max_context_tokens, int) or max_context_tokens <= 0:
        return requested_max_new_tokens
    remaining = max_context_tokens - prompt_tokens
    return max(1, min(requested_max_new_tokens, remaining))


def _apply_context_limit(
    *,
    tokenizer: Any,
    inputs: dict[str, Any],
    prompt_text: str,
    max_new_tokens: int,
    max_context_tokens: Any,
) -> tuple[dict[str, Any], str, int]:
    if not isinstance(max_context_tokens, int) or max_context_tokens <= 0:
        return inputs, prompt_text, 0
    max_input_tokens = max(max_context_tokens - max_new_tokens, 1)
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.shape[1] <= max_input_tokens:
        return inputs, prompt_text, 0
    truncated = int(input_ids.shape[1] - max_input_tokens)
    trimmed_inputs = {}
    for key, value in inputs.items():
        if getattr(value, "ndim", 0) >= 2:
            trimmed_inputs[key] = value[:, -max_input_tokens:]
        else:
            trimmed_inputs[key] = value
    trimmed_prompt = tokenizer.decode(trimmed_inputs["input_ids"][0], skip_special_tokens=False)
    return trimmed_inputs, str(trimmed_prompt), truncated


def _availability() -> tuple[bool, list[str]]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True, []
    except ModuleNotFoundError as exc:
        return False, [f"missing dependency: {exc.name}"]
