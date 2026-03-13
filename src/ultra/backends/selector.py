"""Backend auto-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os

from ..config import BackendName


@dataclass(slots=True)
class AutoBackendContext:
    gguf: bool
    has_gpu: bool
    ctx: int | None = None
    batch: int | None = None
    structured_output: bool = False
    high_throughput: bool = False


def select_backend(context: AutoBackendContext) -> BackendName:
    """Apply the spec's `--backend auto` rule order."""

    if context.gguf:
        return "llamacpp"
    if context.has_gpu and _prefer_paged_attention(context):
        if runtime_available("vllm"):
            return "vllm"
        if runtime_available("sglang"):
            return "sglang"
    return "hf"


def runtime_available(backend: BackendName) -> bool:
    service_env = {
        "vllm": "ULTRA_VLLM_BASE_URL",
        "sglang": "ULTRA_SGLANG_BASE_URL",
        "llamacpp": "ULTRA_LLAMACPP_BASE_URL",
    }.get(backend)
    if service_env and os.environ.get(service_env):
        return True
    module_name = {
        "hf": "transformers",
        "vllm": "vllm",
        "sglang": "sglang",
        "llamacpp": "llama_cpp",
        "auto": "transformers",
    }[backend]
    return importlib.util.find_spec(module_name) is not None


def _prefer_paged_attention(context: AutoBackendContext) -> bool:
    long_context = context.ctx is not None and context.ctx >= 16384
    large_batch = context.batch is not None and context.batch >= 8
    return bool(context.structured_output or context.high_throughput or long_context or large_batch)
