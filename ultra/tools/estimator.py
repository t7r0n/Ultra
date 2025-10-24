from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..mcd.inspector import InspectionResult
from . import hwcheck


PRECISION_BYTES = {
    "fp8": 1,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}

DEFAULT_VLLM_UTILIZATION = 0.90


@dataclass(slots=True)
class DeviceEstimate:
    name: str
    total_bytes: Optional[int]
    estimated_bytes: Optional[int]
    headroom_bytes: Optional[int]
    fits: Optional[bool]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "total_bytes": self.total_bytes,
            "estimated_bytes": self.estimated_bytes,
            "headroom_bytes": self.headroom_bytes,
            "fits": self.fits,
        }


@dataclass(slots=True)
class EstimateBreakdown:
    precision: str
    bytes_per_value: int
    context: int
    batch: int
    weights_bytes: int
    kv_bytes_total: int
    params_estimated: Optional[int]
    backend: str
    device_estimates: List[DeviceEstimate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "precision": self.precision,
            "bytes_per_value": self.bytes_per_value,
            "context": self.context,
            "batch": self.batch,
            "weights_bytes": self.weights_bytes,
            "kv_bytes_total": self.kv_bytes_total,
            "params_estimated": self.params_estimated,
            "backend": self.backend,
            "devices": [device.to_dict() for device in self.device_estimates],
            "notes": self.notes,
        }


def _resolve_precision(requested: str, inspection: InspectionResult) -> str:
    if requested != "auto":
        return requested
    preferred = [dtype for dtype in inspection.dtype_candidates if dtype in ("bf16", "fp16", "fp32")]
    if preferred:
        return preferred[0]
    return inspection.dtype_candidates[0] if inspection.dtype_candidates else "fp16"


def _bytes_per_value(precision: str) -> int:
    precision_lower = precision.lower()
    if precision_lower not in PRECISION_BYTES:
        raise ValueError(f"Unsupported precision '{precision}'")
    return PRECISION_BYTES[precision_lower]


def _weight_files(inspection: InspectionResult) -> List[int]:
    suffixes = {".safetensors", ".bin", ".gguf"}
    sizes: List[int] = []
    for path_str in inspection.metadata.get("files", []):
        path = inspection.source_path / path_str
        if path.is_file() and path.suffix in suffixes:
            sizes.append(path.stat().st_size)
    return sizes


def _estimate_params(weights_bytes: int, bytes_per_value: int) -> Optional[int]:
    if not weights_bytes or not bytes_per_value:
        return None
    return int(weights_bytes / bytes_per_value)


def estimate_memory(
    *,
    inspection: InspectionResult,
    context: int,
    batch_size: int,
    precision: str,
    backend: str,
) -> EstimateBreakdown:
    resolved_precision = _resolve_precision(precision, inspection)
    bytes_per_value = _bytes_per_value(resolved_precision)
    weight_bytes = sum(_weight_files(inspection))
    params_estimated = _estimate_params(weight_bytes, bytes_per_value)

    kv_bytes_per_token = 2 * inspection.n_layers * inspection.n_kv_heads * inspection.head_dim * bytes_per_value
    kv_total = kv_bytes_per_token * context * batch_size
    notes: List[str] = [
        f"kv_bytes_per_token={kv_bytes_per_token}",
        f"head_dim={inspection.head_dim}",
    ]
    backend_lower = backend.lower()
    if backend_lower == "auto":
        backend_lower = inspection.backends_supported[0] if inspection.backends_supported else "hf"
    if backend_lower == "vllm":
        # Inflate kv_total to account for vLLM's gpu_memory_utilization setting,
        # which limits the fraction of GPU memory available for allocation.
        kv_total = int(math.ceil(kv_total / DEFAULT_VLLM_UTILIZATION))
        notes.append(
            f"kv_total inflated by dividing by vLLM gpu_memory_utilization ({DEFAULT_VLLM_UTILIZATION}) "
            "to account for only a fraction of GPU memory being available for allocation."
        )
    total_estimated = weight_bytes + kv_total

    system_report = hwcheck.collect_system_report()
    device_estimates: List[DeviceEstimate] = []
    for gpu in system_report.get("gpus", []):
        total = gpu.get("total_bytes") if isinstance(gpu, dict) else None
        name = gpu.get("name") if isinstance(gpu, dict) else "gpu"
        headroom = None
        fits = None
        if isinstance(total, int):
            headroom = total - total_estimated
            fits = headroom >= 0
        device_estimates.append(
            DeviceEstimate(
                name=str(name),
                total_bytes=total if isinstance(total, int) else None,
                estimated_bytes=total_estimated,
                headroom_bytes=headroom,
                fits=fits,
            )
        )

    breakdown = EstimateBreakdown(
        precision=resolved_precision,
        bytes_per_value=bytes_per_value,
        context=context,
        batch=batch_size,
        weights_bytes=weight_bytes,
        kv_bytes_total=kv_total,
        params_estimated=params_estimated,
        backend=backend_lower,
        device_estimates=device_estimates,
        notes=notes,
    )
    return breakdown


def _humanize(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def format_estimate(estimate: EstimateBreakdown) -> str:
    lines: List[str] = []
    lines.append("ULTRA Memory Estimate")
    lines.append(f"Precision: {estimate.precision} ({estimate.bytes_per_value} bytes/value)")
    lines.append(f"Context: {estimate.context} tokens | Batch: {estimate.batch}")
    lines.append(f"Weights: {_humanize(estimate.weights_bytes)}")
    lines.append(f"KV Cache: {_humanize(estimate.kv_bytes_total)}")
    total = estimate.weights_bytes + estimate.kv_bytes_total
    lines.append(f"Total: {_humanize(total)}")
    if estimate.params_estimated is not None:
        lines.append(f"Parameters (approx): {estimate.params_estimated:,}")
    lines.append("Devices:")
    for device in estimate.device_estimates:
        lines.append(
            "  - "
            f"{device.name}: total={_humanize(device.total_bytes)} | headroom={_humanize(device.headroom_bytes)} | "
            f"fits={device.fits}"
        )
    if estimate.notes:
        lines.append("Notes:")
        lines.extend(f"  * {note}" for note in estimate.notes)
    return "\n".join(lines)

