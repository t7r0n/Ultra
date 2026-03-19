"""Resource and memory estimator helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..backends import AutoBackendContext, select_backend
from ..config import BackendName, PrecisionName
from ..mcd.inspector import fetch_model_reference, inspect_model
from .hwcheck import collect_hardware_report


class EstimateError(RuntimeError):
    """Raised when an estimate cannot be computed from the available metadata."""


@dataclass(slots=True)
class EstimatorInputs:
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    ctx: int
    batch: int = 1
    precision: PrecisionName = "auto"
    params: int | None = None
    gpu_memory_utilization: float = 1.0


@dataclass(slots=True)
class MemoryEstimate:
    model_ref: str
    backend: BackendName
    precision: PrecisionName
    ctx: int
    batch: int
    bytes_per_value: int
    weights_bytes: int | None
    kv_bytes_per_token: int
    total_kv_bytes: int
    estimated_total_bytes: int | None
    device_kind: str
    device_name: str | None
    device_total_bytes: int | None
    device_free_bytes: int | None
    headroom_bytes: int | None
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_memory(inputs: EstimatorInputs, *, model_ref: str = "<manual>", backend: BackendName = "auto") -> MemoryEstimate:
    bytes_per_value = bytes_per_precision(inputs.precision)
    kv_bytes_per_token = (
        2
        * inputs.num_layers
        * inputs.num_key_value_heads
        * inputs.head_dim
        * bytes_per_value
    )
    total_kv_bytes = int(kv_bytes_per_token * inputs.ctx * inputs.batch * inputs.gpu_memory_utilization)
    weights_bytes = inputs.params * bytes_per_value if inputs.params is not None else None

    assumptions = [
        "ctx is treated as prompt_len + avg_generated for the KV estimate.",
        "GQA uses num_key_value_heads when provided.",
    ]
    if inputs.precision == "auto":
        assumptions.append("auto precision is estimated as fp16/bf16 (2 bytes per value).")
    if inputs.gpu_memory_utilization != 1.0:
        assumptions.append("vLLM-style gpu_memory_utilization was applied to the KV pool.")

    return MemoryEstimate(
        model_ref=model_ref,
        backend=backend,
        precision=inputs.precision,
        ctx=inputs.ctx,
        batch=inputs.batch,
        bytes_per_value=bytes_per_value,
        weights_bytes=weights_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        total_kv_bytes=total_kv_bytes,
        estimated_total_bytes=weights_bytes + total_kv_bytes if weights_bytes is not None else None,
        device_kind="unknown",
        device_name=None,
        device_total_bytes=None,
        device_free_bytes=None,
        headroom_bytes=None,
        assumptions=assumptions,
    )


def estimate_model_ref(
    model_ref: str,
    *,
    ctx: int,
    batch: int,
    precision: PrecisionName,
    backend: BackendName,
) -> MemoryEstimate:
    resolved = fetch_model_reference(model_ref)
    selected_backend = backend
    if backend == "auto":
        hardware = collect_hardware_report()
        selected_backend = select_backend(
            AutoBackendContext(
                gguf=resolved.gguf,
                has_gpu=bool(hardware.gpus),
                ctx=ctx,
                batch=batch,
            )
        )
    inspection = inspect_model(str(resolved.resolved_path), backend=selected_backend)
    if inspection.n_layers is None or inspection.n_heads is None or inspection.head_dim is None:
        raise EstimateError("Model inspection did not expose enough attention metadata for estimation.")
    weights_bytes = estimate_weights_bytes(resolved.resolved_path)
    inputs = EstimatorInputs(
        num_layers=inspection.n_layers,
        num_attention_heads=inspection.n_heads,
        num_key_value_heads=inspection.n_kv_heads or inspection.n_heads,
        head_dim=inspection.head_dim,
        ctx=ctx,
        batch=batch,
        precision=precision,
        params=None,
        gpu_memory_utilization=1.0 if selected_backend != "vllm" else 0.9,
    )
    estimate = estimate_memory(inputs, model_ref=model_ref, backend=selected_backend)
    estimate.weights_bytes = weights_bytes
    estimate.estimated_total_bytes = weights_bytes + estimate.total_kv_bytes if weights_bytes is not None else None
    _apply_device_headroom(estimate, preferred_gpu=selected_backend in {"hf", "vllm", "sglang", "llamacpp"})
    if weights_bytes is not None:
        estimate.assumptions.append("weights_bytes uses on-disk model file sizes as a rough inference-only proxy.")
    return estimate


def bytes_per_precision(precision: PrecisionName) -> int:
    if precision in {"fp16", "bf16", "auto"}:
        return 2
    if precision == "fp32":
        return 4
    raise EstimateError(f"Unsupported precision: {precision}")


def estimate_weights_bytes(model_path: Path) -> int | None:
    candidates = list(_weight_files(model_path))
    if not candidates:
        return None
    return sum(candidate.stat().st_size for candidate in candidates)


def _weight_files(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path] if model_path.suffix.lower() in {".gguf", ".safetensors", ".bin", ".pt", ".pth"} else []
    suffixes = {".gguf", ".safetensors", ".bin", ".pt", ".pth"}
    return sorted(
        candidate
        for candidate in model_path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in suffixes
    )


def _apply_device_headroom(estimate: MemoryEstimate, *, preferred_gpu: bool) -> None:
    report = collect_hardware_report()
    if preferred_gpu and report.gpus:
        gpu = max(report.gpus, key=lambda item: item.get("memory_free", 0))
        estimate.device_kind = "gpu"
        estimate.device_name = str(gpu.get("name"))
        estimate.device_total_bytes = int(gpu.get("memory_total")) if gpu.get("memory_total") is not None else None
        estimate.device_free_bytes = int(gpu.get("memory_free")) if gpu.get("memory_free") is not None else None
    else:
        estimate.device_kind = "ram"
        estimate.device_name = report.machine
        estimate.device_total_bytes = report.total_memory_bytes
        estimate.device_free_bytes = report.total_memory_bytes
    if estimate.estimated_total_bytes is not None and estimate.device_free_bytes is not None:
        estimate.headroom_bytes = estimate.device_free_bytes - estimate.estimated_total_bytes
