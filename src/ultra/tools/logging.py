"""Run artifact helpers for scaffolded chat, agent, and eval commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from ..config import BackendName


@dataclass(slots=True)
class RunArtifacts:
    root: Path
    run_json: Path
    prompts_jsonl: Path
    candidates_jsonl: Path
    selection_json: Path
    final_txt: Path
    metrics_json: Path
    final_json: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def initialize_run_artifacts(
    *,
    command: str,
    model_ref: str,
    backend: BackendName,
    config: dict[str, Any],
    hardware: dict[str, Any] | None,
    logdir: Path | None,
    extra: dict[str, Any],
    structured_output: bool,
) -> RunArtifacts:
    root = logdir or _default_run_dir(command, model_ref)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(
        root=root,
        run_json=root / "run.json",
        prompts_jsonl=root / "prompts.jsonl",
        candidates_jsonl=root / "candidates.jsonl",
        selection_json=root / "selection.json",
        final_txt=root / "final.txt",
        metrics_json=root / "metrics.json",
        final_json=root / "final.json" if structured_output else None,
    )
    payload = {
        "artifact_version": 2,
        "command": command,
        "model_ref": model_ref,
        "backend": backend,
        "config": config,
        "hardware": hardware or {},
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **extra,
    }
    _write_json(artifacts.run_json, payload)
    artifacts.prompts_jsonl.write_text("", encoding="utf-8")
    artifacts.candidates_jsonl.write_text("", encoding="utf-8")
    _write_json(
        artifacts.selection_json,
        {
            "initialized": True,
            "scores": [],
            "selected_candidate_index": None,
            "selected_candidate": None,
            "judge": None,
            "long_context_policy": None,
        },
    )
    artifacts.final_txt.write_text("", encoding="utf-8")
    _write_json(
        artifacts.metrics_json,
        {
            "initialized": True,
            "latency_s": None,
            "tokens_per_s": None,
            "peak_kv_blocks": None,
            "peak_memory_bytes": None,
        },
    )
    if artifacts.final_json is not None:
        _write_json(artifacts.final_json, {})
    return artifacts


def _default_run_dir(command: str, model_ref: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = model_ref.replace("/", "_").replace(" ", "_")
    return Path("runs") / f"{command}-{slug}-{timestamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
