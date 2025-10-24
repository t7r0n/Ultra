from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - psutil optional
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency
    snapshot_download = None  # type: ignore


@dataclass(slots=True)
class GPUInfo:
    name: str
    total_memory: Optional[int]


@dataclass(slots=True)
class SystemReport:
    python_version: str
    platform: str
    cuda_visible_devices: Optional[str]
    gpus: Dict[str, GPUInfo]
    total_ram: Optional[int]
    available_ram: Optional[int]
    dependencies: Dict[str, bool]


OPTIONAL_DEPENDENCIES = [
    "torch",
    "transformers",
    "huggingface_hub",
    "vllm",
    "sglang",
    "llama_cpp_python",
]


def collect_system_report() -> Dict[str, object]:
    python_version = platform.python_version()
    system = platform.platform()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = detect_gpus()
    total_ram = None
    available_ram = None
    if psutil is not None:  # pragma: no branch - psutil optional
        total_ram = int(psutil.virtual_memory().total)
        available_ram = int(psutil.virtual_memory().available)
    dependencies = {name: importable(name) for name in OPTIONAL_DEPENDENCIES}
    report = SystemReport(
        python_version=python_version,
        platform=system,
        cuda_visible_devices=cuda_visible,
        gpus=gpus,
        total_ram=total_ram,
        available_ram=available_ram,
        dependencies=dependencies,
    )
    return asdict(report)


def format_report(report: Dict[str, object]) -> str:
    lines = ["ULTRA Doctor Report"]
    lines.append(f"Python: {report['python_version']}")
    lines.append(f"Platform: {report['platform']}")
    lines.append(f"CUDA_VISIBLE_DEVICES: {report['cuda_visible_devices']}")
    gpu_entries = report.get("gpus", {}) or {}
    for gpu_id, gpu_info in gpu_entries.items():
        lines.append(
            f"GPU {gpu_id}: {gpu_info['name']} ({human_size(gpu_info['total_memory']) if gpu_info['total_memory'] else 'unknown'})"
        )
    lines.append(
        f"RAM: {human_size(report.get('available_ram'))} / {human_size(report.get('total_ram'))}"
    )
    deps = report.get("dependencies", {}) or {}
    for name, ok in deps.items():
        lines.append(f"Dependency {name}: {'OK' if ok else 'missing'}")
    return "\n".join(lines)


def detect_gpus() -> Dict[str, GPUInfo]:
    gpus: Dict[str, GPUInfo] = {}
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return gpus
    try:  # pragma: no cover - external command
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return gpus
    for idx, line in enumerate(result.stdout.strip().splitlines()):
        if not line:
            continue
        parts = [chunk.strip() for chunk in line.split(",")]
        name = parts[0]
        total = None
        if len(parts) > 1 and parts[1].endswith(" MiB"):
            total = int(parts[1].split()[0]) * 1024 * 1024
        gpus[str(idx)] = GPUInfo(name=name, total_memory=total)
    return gpus


def human_size(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    float_value = float(value)
    while float_value >= 1024 and idx < len(units) - 1:
        float_value /= 1024
        idx += 1
    return f"{float_value:.2f} {units[idx]}"


def importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def fetch_model_snapshot(
    model_ref: str,
    revision: Optional[str] = None,
    local_dir: Optional[os.PathLike[str]] = None,
    trust_remote_code: bool = False,
) -> str:
    if snapshot_download is None:  # pragma: no cover - network download
        raise RuntimeError("huggingface_hub is required for fetch")
    resolved = snapshot_download(
        repo_id=model_ref,
        revision=revision,
        allow_patterns=None,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        trust_remote_code=trust_remote_code,
    )
    return resolved


def prepare_agent_environment() -> Dict[str, object]:
    """Return sandbox environment description."""
    return {
        "cwd": os.getcwd(),
        "env": {key: os.environ.get(key) for key in ("PATH", "HOME")},
    }
