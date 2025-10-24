from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..mcd.inspector import InspectionResult

PRECISION_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}


@dataclass(slots=True)
class Estimate:
    precision: str
    weights_bytes: int
    kv_cache_bytes: int
    total_bytes: int
    backend: str
    context: int
    batch_size: int

    def to_table(self) -> Dict[str, str]:
        return {
            "precision": self.precision,
            "weights": human_size(self.weights_bytes),
            "kv_cache": human_size(self.kv_cache_bytes),
            "total": human_size(self.total_bytes),
            "backend": self.backend,
            "context": str(self.context),
            "batch": str(self.batch_size),
        }


def estimate_memory(
    *,
    inspection: InspectionResult,
    context: int,
    batch_size: int,
    precision: str,
    backend: str,
) -> Estimate:
    resolved_precision = resolve_precision(inspection, precision)
    bytes_per_value = PRECISION_BYTES[resolved_precision]
    params = approximate_params(inspection)
    weights_bytes = params * bytes_per_value
    kv_bytes_per_token = 2 * inspection.n_layers * inspection.n_kv_heads * inspection.head_dim * bytes_per_value
    kv_cache_bytes = kv_bytes_per_token * context * batch_size
    if backend == "vllm":
        gpu_utilization = float(inspection.metadata.get("generation_config", {}).get("gpu_memory_utilization", 0.9) or 0.9)
        kv_cache_bytes = int(kv_cache_bytes / gpu_utilization)
    total_bytes = weights_bytes + kv_cache_bytes
    return Estimate(
        precision=resolved_precision,
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        total_bytes=total_bytes,
        backend=backend,
        context=context,
        batch_size=batch_size,
    )


def resolve_precision(inspection: InspectionResult, requested: str) -> str:
    if requested != "auto":
        if requested not in PRECISION_BYTES:
            raise ValueError(f"Unsupported precision: {requested}")
        return requested
    preference = ["bf16", "fp16", "fp32"]
    for candidate in preference:
        if candidate in inspection.dtype_candidates:
            return candidate
    return "fp16"


def approximate_params(inspection: InspectionResult) -> int:
    config = inspection.metadata.get("config", {})
    vocab_size = int(config.get("vocab_size", 0))
    hidden = inspection.hidden_size
    layers = inspection.n_layers
    core = hidden * hidden * layers * 12
    embed = vocab_size * hidden
    return core + embed


def format_estimate(estimate: Estimate) -> str:
    table = estimate.to_table()
    lines = ["Resource Estimate"]
    for key, value in table.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def human_size(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    float_value = float(value)
    while float_value >= 1024 and idx < len(units) - 1:
        float_value /= 1024
        idx += 1
    return f"{float_value:.2f} {units[idx]}"
