from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency for RAM metrics
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency for GPU metrics
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore

try:  # pragma: no cover - optional dependency for downloads
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency
    snapshot_download = None  # type: ignore


@dataclasses.dataclass(slots=True)
class GPUReport:
    index: int
    name: str
    total_bytes: Optional[int]
    free_bytes: Optional[int]
    driver: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "driver": self.driver,
        }


@dataclasses.dataclass(slots=True)
class DependencyReport:
    name: str
    available: bool
    version: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "available": self.available, "version": self.version}


@dataclasses.dataclass(slots=True)
class SystemReport:
    python: str
    executable: str
    platform: str
    cpu: str
    ram_total: Optional[int]
    ram_available: Optional[int]
    cuda_visible_devices: Optional[str]
    gpus: List[GPUReport]
    dependencies: List[DependencyReport]
    raw_nvidia_smi: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "python": self.python,
            "executable": self.executable,
            "platform": self.platform,
            "cpu": self.cpu,
            "ram_total": self.ram_total,
            "ram_available": self.ram_available,
            "cuda_visible_devices": self.cuda_visible_devices,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "dependencies": [dep.to_dict() for dep in self.dependencies],
            "raw_nvidia_smi": self.raw_nvidia_smi,
        }


DEPENDENCIES: Dict[str, str] = {
    "torch": "__version__",
    "transformers": "__version__",
    "vllm": "__version__",
    "sglang": "__version__",
    "llama_cpp_python": "__version__",
    "huggingface_hub": "__version__",
}


def _query_dependency(name: str, attr: str) -> DependencyReport:
    try:
        module = __import__(name)
    except Exception:  # pragma: no cover - optional dependency
        return DependencyReport(name=name, available=False, version=None)
    version = getattr(module, attr, None)
    if callable(version):  # pragma: no cover - defensive
        version = version()  # type: ignore[assignment]
    return DependencyReport(name=name, available=True, version=str(version) if version else None)


def _query_ram() -> tuple[Optional[int], Optional[int]]:
    if psutil is None:  # pragma: no cover - optional dependency
        return None, None
    virtual = psutil.virtual_memory()
    return int(virtual.total), int(virtual.available)


def _humanize_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def _nvml_query() -> List[GPUReport]:
    if pynvml is None:  # pragma: no cover - optional dependency
        return []
    try:
        pynvml.nvmlInit()
    except Exception:  # pragma: no cover - NVML init failure
        return []
    driver_version: Optional[str]
    try:
        driver_version = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
    except Exception:  # pragma: no cover - optional dependency
        driver_version = None
    gpus: List[GPUReport] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:  # pragma: no cover - optional dependency
        count = 0
    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu = GPUReport(
                index=index,
                name=name,
                total_bytes=int(mem_info.total),
                free_bytes=int(mem_info.free),
                driver=driver_version,
            )
        except Exception:  # pragma: no cover - optional dependency
            gpu = GPUReport(index=index, name="unknown", total_bytes=None, free_bytes=None, driver=driver_version)
        gpus.append(gpu)
    try:
        pynvml.nvmlShutdown()
    except Exception:  # pragma: no cover - optional dependency
        pass
    return gpus


def _nvidia_smi_query() -> Optional[str]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:  # pragma: no cover - external process
        output = subprocess.check_output(
            [
                binary,
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return None
    return output.strip()


def collect_system_report() -> Dict[str, object]:
    cpu_descriptor = platform.processor() or platform.machine()
    ram_total, ram_available = _query_ram()
    dependencies = [_query_dependency(name, attr) for name, attr in DEPENDENCIES.items()]
    gpus = _nvml_query()
    report = SystemReport(
        python=sys.version.split()[0],
        executable=sys.executable,
        platform=platform.platform(),
        cpu=cpu_descriptor,
        ram_total=ram_total,
        ram_available=ram_available,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpus=gpus,
        dependencies=dependencies,
        raw_nvidia_smi=_nvidia_smi_query(),
    )
    return report.to_dict()


def format_report(report: Dict[str, object]) -> str:
    lines: List[str] = ["ULTRA Doctor Report"]
    lines.append(f"Python      : {report['python']} ({report['executable']})")
    lines.append(f"Platform    : {report['platform']}")
    lines.append(f"CPU         : {report['cpu']}")
    lines.append(
        "RAM         : "
        f"{_humanize_bytes(report.get('ram_available') if isinstance(report.get('ram_available'), int) else None)}"
        " available / "
        f"{_humanize_bytes(report.get('ram_total') if isinstance(report.get('ram_total'), int) else None)} total"
    )
    lines.append(f"CUDA_VISIBLE_DEVICES: {report.get('cuda_visible_devices')}")
    gpus = report.get("gpus", []) or []
    for gpu in gpus:
        gpu_dict = dict(gpu)
        index = gpu_dict.get("index")
        name = gpu_dict.get("name")
        driver = gpu_dict.get("driver")
        total = _humanize_bytes(gpu_dict.get("total_bytes") if isinstance(gpu_dict.get("total_bytes"), int) else None)
        free = _humanize_bytes(gpu_dict.get("free_bytes") if isinstance(gpu_dict.get("free_bytes"), int) else None)
        lines.append(f"GPU {index}: {name} | free {free} / total {total} | driver {driver}")
    lines.append("Dependencies:")
    for dependency in report.get("dependencies", []) or []:
        dep = dict(dependency)
        status = "ok" if dep.get("available") else "missing"
        version = dep.get("version")
        suffix = f" (v{version})" if version else ""
        lines.append(f"  - {dep.get('name')}: {status}{suffix}")
    if report.get("raw_nvidia_smi"):
        lines.append("nvidia-smi:")
        lines.extend(f"  {line}" for line in str(report["raw_nvidia_smi"]).splitlines())
    return "\n".join(lines)


_HF_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "*.merges",
    "*.vocab",
    "*.tiktoken",
    "*.safetensors",
    "*.bin",
    "*.gguf",
    "*.json",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model_snapshot(
    *,
    model_ref: str,
    revision: Optional[str] = None,
    local_dir: Optional[Path] = None,
    trust_remote_code: bool = False,
) -> Dict[str, object]:
    if snapshot_download is None:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub is required to download models")
    result_path = snapshot_download(
        repo_id=model_ref,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=_HF_ALLOW_PATTERNS,
        trust_remote_code=trust_remote_code,
    )
    root = Path(result_path)
    manifest: List[Dict[str, object]] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(file_path.relative_to(root))
        checksum = _sha256_file(file_path)
        manifest.append({"path": rel, "sha256": checksum, "size": file_path.stat().st_size})
    backend = "llamacpp" if any(p.suffix == ".gguf" for p in root.rglob("*.gguf")) else "hf"
    return {
        "path": str(root),
        "files": manifest,
        "backend": backend,
    }


def prepare_agent_environment(*, sandbox: str, open_terminal: bool) -> Dict[str, object]:
    return {
        "sandbox": sandbox,
        "open_terminal": open_terminal,
        "cwd": os.getcwd(),
        "env": {"PATH": os.environ.get("PATH"), "HOME": os.environ.get("HOME")},
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

