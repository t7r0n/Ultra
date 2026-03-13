"""Hardware and dependency inspection helpers for `ultra doctor`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from typing import Any


@dataclass(slots=True)
class HardwareReport:
    python_version: str
    python_executable: str
    platform: str
    processor: str
    machine: str
    cpu_count: int
    total_memory_bytes: int | None
    cuda_version: str | None
    rocm_version: str | None
    torch_cuda_available: bool | None
    nvidia_smi_path: str | None
    nvml_available: bool
    gpus: list[dict[str, Any]]
    dependencies: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_hardware_report() -> HardwareReport:
    return HardwareReport(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform=platform.platform(),
        processor=platform.processor(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 0,
        total_memory_bytes=_read_total_memory_bytes(),
        cuda_version=_torch_cuda_version(),
        rocm_version=_torch_rocm_version(),
        torch_cuda_available=_torch_cuda_available(),
        nvidia_smi_path=shutil.which("nvidia-smi"),
        nvml_available=_nvml_available(),
        gpus=_collect_gpus(),
        dependencies=_dependency_versions(),
    )


def _dependency_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "transformers",
        "huggingface_hub",
        "accelerate",
        "tokenizers",
        "vllm",
        "sglang",
        "llama_cpp_python",
        "pynvml",
        "safetensors",
        "numba",
        "uvloop",
        "orjson",
        "typer",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _collect_gpus() -> list[dict[str, Any]]:
    gpus = _collect_gpus_from_nvml()
    return gpus if gpus else _collect_gpus_from_nvidia_smi()


def _read_total_memory_bytes() -> int | None:
    meminfo_path = "/proc/meminfo"
    if not os.path.exists(meminfo_path):
        return None
    with open(meminfo_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    return None


def _collect_gpus_from_nvml() -> list[dict[str, Any]]:
    if not _nvml_available():
        return []
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[dict[str, Any]] = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            gpus.append(
                {
                    "index": index,
                    "name": str(name),
                    "memory_total": int(memory.total),
                    "memory_free": int(memory.free),
                    "driver_version": pynvml.nvmlSystemGetDriverVersion().decode("utf-8"),
                }
            )
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _collect_gpus_from_nvidia_smi() -> list[dict[str, Any]]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return []
    result = subprocess.run(
        [
            command,
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total": int(parts[2]) * 1024 * 1024,
                "memory_free": int(parts[3]) * 1024 * 1024,
                "driver_version": parts[4],
            }
        )
    return gpus


def _nvml_available() -> bool:
    return importlib.util.find_spec("pynvml") is not None


def _torch_cuda_available() -> bool | None:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _torch_cuda_version() -> str | None:
    try:
        import torch

        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


def _torch_rocm_version() -> str | None:
    try:
        import torch

        return getattr(torch.version, "hip", None)
    except Exception:
        return None
