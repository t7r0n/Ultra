"""Backend adapters declared by the ULTRA spec."""

from .common import GenerationRequest, GenerationResult, StructuredOutputRequest
from .hf_engine import BACKEND_NAME as HF_BACKEND_NAME, HFSamplingRequest, describe_backend as describe_hf_backend, generate as generate_hf
from .llama_cpp_engine import BACKEND_NAME as LLAMACPP_BACKEND_NAME, LlamaCppSamplingRequest, describe_backend as describe_llamacpp_backend, generate as generate_llamacpp
from .selector import AutoBackendContext, runtime_available, select_backend
from .sglang_engine import BACKEND_NAME as SGLANG_BACKEND_NAME, SGLangSamplingRequest, describe_backend as describe_sglang_backend, generate as generate_sglang
from .vllm_engine import BACKEND_NAME as VLLM_BACKEND_NAME, VLLMSamplingRequest, describe_backend as describe_vllm_backend, generate as generate_vllm

__all__ = [
    "AutoBackendContext",
    "HF_BACKEND_NAME",
    "HFSamplingRequest",
    "LLAMACPP_BACKEND_NAME",
    "LlamaCppSamplingRequest",
    "SGLANG_BACKEND_NAME",
    "SGLangSamplingRequest",
    "GenerationRequest",
    "GenerationResult",
    "StructuredOutputRequest",
    "VLLM_BACKEND_NAME",
    "VLLMSamplingRequest",
    "describe_hf_backend",
    "describe_llamacpp_backend",
    "describe_sglang_backend",
    "describe_vllm_backend",
    "generate_hf",
    "generate_llamacpp",
    "generate_sglang",
    "generate_vllm",
    "runtime_available",
    "select_backend",
]
