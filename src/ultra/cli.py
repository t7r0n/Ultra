"""CLI entrypoints for the ULTRA scaffold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from typer import Argument, Exit, Option, Typer, echo

from .agent_loop import run_agent
from .config import BackendName, EvalSuiteName, PrecisionName, SandboxName, UltraProfileName, default_config_path, load_config
from .eval import arena as arena_eval
from .eval import code as code_eval
from .eval import harness as harness_eval
from .eval import longctx as long_eval
from .eval import rag as rag_eval
from .mcd.inspector import ModelReferenceError, fetch_model_reference, inspect_model
from .runtime import run_ultra_turn
from .tools.estimator import EstimateError, estimate_model_ref
from .tools.hwcheck import collect_hardware_report
from .tools.logging import initialize_run_artifacts
from .tools.sandbox import build_sandbox_profile


app = Typer(help="ULTRA: An Extreme-Mode Inference & Evaluation Orchestrator for Open-Source LLMs.")


@app.command()
def doctor(
    json: bool = Option(False, "--json", help="Emit machine-readable JSON instead of text."),
) -> None:
    """hardware+deps check"""

    report = collect_hardware_report()
    if json:
        echo(json_dumps(report.to_dict()))
        return

    echo(f"Python: {report.python_version} ({report.python_executable})")
    echo(f"Platform: {report.platform}")
    echo(f"CPU: {report.processor or 'unknown'}")
    echo(f"CPUs: {report.cpu_count}")
    echo(f"Memory bytes: {report.total_memory_bytes if report.total_memory_bytes is not None else 'unknown'}")
    echo(f"CUDA version: {report.cuda_version or 'unknown'}")
    echo(f"ROCm version: {report.rocm_version or 'unknown'}")
    echo(f"Torch CUDA available: {report.torch_cuda_available if report.torch_cuda_available is not None else 'unknown'}")
    echo(f"NVML available: {report.nvml_available}")
    echo(f"nvidia-smi: {report.nvidia_smi_path or 'not found'}")
    echo(f"GPU count: {len(report.gpus)}")
    for gpu in report.gpus:
        echo(
            "GPU {index}: {name} total={total} free={free} driver={driver}".format(
                index=gpu["index"],
                name=gpu["name"],
                total=gpu["memory_total"],
                free=gpu["memory_free"],
                driver=gpu["driver_version"],
            )
        )
    for dependency, version in sorted(report.dependencies.items()):
        echo(f"{dependency}: {version or 'not installed'}")


@app.command()
def fetch(
    hf_repo_or_gguf_path: str = Argument(..., metavar="HF_REPO_OR_GGUF_PATH"),
    revision: str | None = Option(None, "--revision", help="Optional Hugging Face revision."),
    local_dir: Path | None = Option(None, "--local-dir", metavar="PATH", help="Optional destination directory."),
    trust_remote_code: bool = Option(False, "--trust-remote-code", help="Recorded for downstream backends."),
) -> None:
    """HF/GGUF download + verify"""

    try:
        resolved = fetch_model_reference(
            hf_repo_or_gguf_path,
            revision=revision,
            local_dir=local_dir,
            trust_remote_code=trust_remote_code,
        )
    except ModelReferenceError as exc:
        echo(str(exc))
        raise Exit(2) from exc

    payload = resolved.to_dict()
    payload["sha256"] = collect_sha256(resolved.resolved_path)
    payload["verification"] = collect_fetch_verification(resolved.resolved_path)
    echo(json_dumps(payload))


@app.command("inspect")
def inspect_command(
    model_ref: str = Argument(..., metavar="MODEL_REF"),
    backend: BackendName = Option("auto", "--backend", help="Backend selector."),
    json: bool = Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """MCD: config, chat template, tokenizer, dtype"""

    try:
        inspection = inspect_model(model_ref, backend=backend)
    except ModelReferenceError as exc:
        echo(str(exc))
        raise Exit(2) from exc

    if json:
        echo(json_dumps(inspection.to_dict()))
        return

    payload = inspection.to_dict()
    for key in (
        "backend",
        "arch",
        "context_window",
        "sliding_window",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "hidden_size",
        "vision_support",
        "system_prompt_hint",
    ):
        echo(f"{key}: {payload[key]}")
    echo(f"backends_supported: {', '.join(payload['backends_supported'])}")
    if payload.get("notes"):
        echo("notes:")
        for note in payload["notes"]:
            echo(f"- {note}")


@app.command()
def estimate(
    model_ref: str = Argument(..., metavar="MODEL_REF"),
    ctx: int = Option(8192, "--ctx", help="Target active sequence length."),
    batch: int = Option(1, "--batch", help="Target batch size."),
    precision: PrecisionName = Option("auto", "--precision", help="Weight and KV precision."),
    backend: BackendName = Option("auto", "--backend", help="Backend selector."),
) -> None:
    """VRAM/RAM/throughput at target settings"""

    try:
        estimate_result = estimate_model_ref(
            model_ref,
            ctx=ctx,
            batch=batch,
            precision=precision,
            backend=backend,
        )
    except (EstimateError, ModelReferenceError) as exc:
        echo(str(exc))
        raise Exit(2) from exc

    payload = estimate_result.to_dict()
    echo(f"model_ref: {payload['model_ref']}")
    echo(f"backend: {payload['backend']}")
    echo(f"precision: {payload['precision']}")
    echo(f"device_kind: {payload['device_kind']}")
    echo(f"device_name: {payload['device_name']}")
    echo(f"device_total_bytes: {payload['device_total_bytes']}")
    echo(f"device_free_bytes: {payload['device_free_bytes']}")
    echo(f"kv_bytes_per_token: {payload['kv_bytes_per_token']}")
    echo(f"total_kv_bytes: {payload['total_kv_bytes']}")
    echo(f"weights_bytes: {payload['weights_bytes']}")
    echo(f"estimated_total_bytes: {payload['estimated_total_bytes']}")
    echo(f"headroom_bytes: {payload['headroom_bytes']}")
    if payload["assumptions"]:
        echo("assumptions:")
        for assumption in payload["assumptions"]:
            echo(f"- {assumption}")


@app.command()
def chat(
    model_ref: str = Argument(..., metavar="MODEL_REF"),
    backend: BackendName = Option("auto", "--backend", help="Backend selector."),
    ultra_profile: UltraProfileName = Option("default", "--ultra-profile", help="Ultra profile preset."),
    config: Path | None = Option(None, "--config", metavar="PATH", help="Optional ultra.yaml override."),
    logdir: Path | None = Option(None, "--logdir", metavar="RUN_DIR", help="Run artifact directory."),
) -> None:
    """interactive; Ultra pipeline"""

    effective_config = load_config(config or default_config_path())
    hardware = collect_hardware_report()
    artifacts = initialize_run_artifacts(
        command="chat",
        model_ref=model_ref,
        backend=backend,
        config=effective_config.to_dict(),
        hardware=hardware.to_dict(),
        logdir=logdir,
        extra={"ultra_profile": ultra_profile, "status": "active"},
        structured_output=effective_config.structured_output.enabled,
    )
    conversation: list[dict[str, str]] = []
    if not sys.stdin.isatty():
        initial_prompt = sys.stdin.read().strip()
        if initial_prompt:
            conversation.append({"role": "user", "content": initial_prompt})
            turn = run_ultra_turn(
                model_ref=model_ref,
                backend=backend,
                config=effective_config,
                ultra_profile=ultra_profile,
                conversation=conversation,
                artifacts=artifacts,
            )
            echo(turn.final_text)
            return
    echo(f"Run directory: {artifacts.root}")
    echo("Enter `exit` or press Ctrl-D to stop.")
    while True:
        try:
            prompt = input("chat> ").strip()
        except EOFError:
            echo("")
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        conversation.append({"role": "user", "content": prompt})
        turn = run_ultra_turn(
            model_ref=model_ref,
            backend=backend,
            config=effective_config,
            ultra_profile=ultra_profile,
            conversation=conversation,
            artifacts=artifacts,
        )
        echo(turn.final_text)
        conversation.append({"role": "assistant", "content": turn.final_text})


@app.command()
def agent(
    model_ref: str = Argument(..., metavar="MODEL_REF"),
    backend: BackendName = Option("auto", "--backend", help="Backend selector."),
    open_terminal: bool = Option(False, "--open-terminal", help="Stream terminal output live and save terminal.jsonl."),
    sandbox: SandboxName = Option("docker", "--sandbox", help="Sandbox backend."),
) -> None:
    """agentic coding loop + tool sandbox"""

    effective_config = load_config(default_config_path())
    sandbox_profile = build_sandbox_profile(sandbox)
    hardware = collect_hardware_report()
    artifacts = initialize_run_artifacts(
        command="agent",
        model_ref=model_ref,
        backend=backend,
        config=effective_config.to_dict(),
        hardware=hardware.to_dict(),
        logdir=None,
        extra={
            "open_terminal": open_terminal,
            "sandbox": sandbox_profile.to_dict(),
            "status": "active",
        },
        structured_output=effective_config.structured_output.enabled,
    )
    if not sys.stdin.isatty():
        goal = sys.stdin.read().strip()
    else:
        goal = input("agent> ").strip()
    if not goal:
        echo("No agent goal provided.")
        raise Exit(2)
    result = run_agent(
        model_ref=model_ref,
        backend=backend,
        config=effective_config,
        goal=goal,
        artifacts=artifacts,
        sandbox_profile=sandbox_profile,
        open_terminal=open_terminal,
    )
    echo(result.final)


@app.command("eval")
def eval_command(
    model_ref: str = Argument(..., metavar="MODEL_REF"),
    suite: EvalSuiteName = Option("harness", "--suite", help="Evaluation suite."),
    tasks: list[str] | None = Option(None, "--tasks", help="Task identifiers."),
    backend: BackendName = Option("auto", "--backend", help="Backend selector."),
    out: Path = Option(Path("results.json"), "--out", metavar="PATH", help="Evaluation output JSON."),
) -> None:
    """bench harnesses + Ultra inference wrapper"""

    effective_config = load_config(default_config_path())
    hardware = collect_hardware_report()
    artifacts = initialize_run_artifacts(
        command="eval",
        model_ref=model_ref,
        backend=backend,
        config=effective_config.to_dict(),
        hardware=hardware.to_dict(),
        logdir=None,
        extra={"suite": suite, "tasks": tasks or [], "status": "active"},
        structured_output=effective_config.structured_output.enabled,
    )
    if suite == "harness":
        payload = harness_eval.run_suite(model_ref=model_ref, tasks=tasks or [], backend=backend, out=out)
    elif suite == "code":
        payload = code_eval.run_suite(model_ref=model_ref, tasks=tasks or [], backend=backend, out=out, config=effective_config)
    elif suite == "long":
        payload = long_eval.run_suite(model_ref=model_ref, tasks=tasks or [], backend=backend, out=out, config=effective_config)
    elif suite == "arena":
        payload = arena_eval.run_suite(model_ref=model_ref, tasks=tasks or [], backend=backend, out=out, config=effective_config)
    elif suite == "rag":
        payload = rag_eval.run_suite(model_ref=model_ref, tasks=tasks or [], backend=backend, out=out, config=effective_config)
    else:
        raise Exit(2)
    payload = {
        **payload,
        "model_ref": model_ref,
        "suite": suite,
        "tasks": tasks or [],
        "backend": backend,
        "run_dir": str(artifacts.root),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_dumps(payload), encoding="utf-8")
    echo(f"Wrote eval output to {out}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by both `python -m ultra` and the console script."""

    try:
        app(argv)
    except Exit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 0
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 0
    return 0


def collect_sha256(path: Path) -> dict[str, str]:
    """Return SHA256 digests for files under a fetched model path."""

    if path.is_file():
        return {path.name: sha256sum(path)}

    digests: dict[str, str] = {}
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digests[str(file_path.relative_to(path))] = sha256sum(file_path)
    return digests


def collect_fetch_verification(path: Path) -> dict[str, Any]:
    if path.is_file():
        is_gguf = path.suffix.lower() == ".gguf"
        return {
            "model_class": "gguf" if is_gguf else "artifact",
            "gguf_files": [path.name] if is_gguf else [],
            "transformer_weight_files": [path.name] if path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"} else [],
            "config_present": False,
            "tokenizer_files_present": False,
            "processor_files_present": False,
            "file_count": 1,
            "total_size_bytes": path.stat().st_size,
            "required_files_ok": is_gguf,
        }

    gguf_files = sorted(str(file.relative_to(path)) for file in path.rglob("*.gguf"))
    transformer_weight_files = sorted(
        str(file.relative_to(path))
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
    )
    config_present = (path / "config.json").exists()
    tokenizer_files_present = any(
        (path / filename).exists()
        for filename in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    processor_files_present = any(
        (path / filename).exists()
        for filename in ("processor_config.json", "preprocessor_config.json")
    )
    files = [file for file in path.rglob("*") if file.is_file()]
    if gguf_files and not transformer_weight_files:
        model_class = "gguf"
    elif transformer_weight_files and not gguf_files:
        model_class = "transformers"
    elif gguf_files or transformer_weight_files:
        model_class = "hybrid"
    else:
        model_class = "artifact"
    required_files_ok = bool(gguf_files) or bool(transformer_weight_files and config_present and tokenizer_files_present)
    return {
        "model_class": model_class,
        "gguf_files": gguf_files,
        "transformer_weight_files": transformer_weight_files,
        "config_present": config_present,
        "tokenizer_files_present": tokenizer_files_present,
        "processor_files_present": processor_files_present,
        "file_count": len(files),
        "total_size_bytes": sum(file.stat().st_size for file in files),
        "required_files_ok": required_files_ok,
    }


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
