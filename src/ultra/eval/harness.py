"""LM Evaluation Harness suite integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from ..config import BackendName
from ..mcd.inspector import fetch_model_reference


SUITE_NAME = "harness"


def describe_suite() -> dict[str, Any]:
    return {
        "suite": SUITE_NAME,
        "status": "implemented",
        "notes": "Uses lm-eval with HF, vLLM, SGLang, or GGUF-backed providers when available.",
    }


def run_suite(
    *,
    model_ref: str,
    tasks: list[str],
    backend: BackendName,
    out: Path,
) -> dict[str, Any]:
    executable = shutil.which("lm_eval")
    if executable is None:
        raise RuntimeError("Harness suite requires `lm_eval` on PATH.")
    if not tasks:
        raise RuntimeError("Harness suite requires at least one task.")
    output_dir = out.parent / (out.stem + "_lm_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name, model_args = harness_model_adapter(model_ref=model_ref, backend=backend)
    command = [
        executable,
        "--model",
        model_name,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--output_path",
        str(output_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "suite": SUITE_NAME,
        "command": command,
        "model": model_name,
        "model_args": model_args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": str(output_dir),
        "results": load_harness_results(output_dir),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"lm_eval failed with exit code {result.returncode}")
    return payload


def harness_model_adapter(*, model_ref: str, backend: BackendName) -> tuple[str, str]:
    resolved = fetch_model_reference(model_ref)
    selected_backend = backend
    if backend == "auto":
        selected_backend = resolved.backend
    if selected_backend == "auto":
        selected_backend = "llamacpp" if resolved.gguf else "hf"

    if selected_backend == "hf":
        return "hf", f"pretrained={model_ref}"
    if selected_backend == "vllm":
        parts = [f"pretrained={model_ref}", "gpu_memory_utilization=0.9"]
        return "vllm", ",".join(parts)
    if selected_backend == "sglang":
        base_url = os.environ.get("ULTRA_SGLANG_BASE_URL")
        if base_url:
            return "local-chat-completions", f"model={model_ref},base_url={base_url.rstrip('/')}/v1/chat/completions"
        return "sglang", f"pretrained={model_ref}"
    if selected_backend == "llamacpp":
        model_path = str(resolved.resolved_path)
        return "gguf", f"gguf_file={model_path}"
    raise RuntimeError(f"Unsupported harness backend: {selected_backend}")


def load_harness_results(output_dir: Path) -> dict[str, Any] | None:
    json_candidates = sorted(output_dir.rglob("*.json"))
    for candidate in json_candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and ("results" in payload or "configs" in payload):
            return payload
    return None
