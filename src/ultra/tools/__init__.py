"""Utility module namespace for ULTRA."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "EstimateError",
    "EstimatorInputs",
    "HardwareReport",
    "MemoryEstimate",
    "RunArtifacts",
    "SandboxProfile",
    "VerificationResult",
    "verify_prompt_alignment",
    "build_sandbox_profile",
    "collect_hardware_report",
    "estimate_memory",
    "estimate_model_ref",
    "initialize_run_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in {"EstimateError", "EstimatorInputs", "MemoryEstimate", "estimate_memory", "estimate_model_ref"}:
        module = import_module(".estimator", __name__)
        return getattr(module, name)
    if name in {"HardwareReport", "collect_hardware_report"}:
        module = import_module(".hwcheck", __name__)
        return getattr(module, name)
    if name in {"RunArtifacts", "initialize_run_artifacts"}:
        module = import_module(".logging", __name__)
        return getattr(module, name)
    if name in {"SandboxProfile", "build_sandbox_profile"}:
        module = import_module(".sandbox", __name__)
        return getattr(module, name)
    if name in {"VerificationResult", "verify_prompt_alignment"}:
        module = import_module(".verifiers", __name__)
        return getattr(module, name)
    raise AttributeError(name)
