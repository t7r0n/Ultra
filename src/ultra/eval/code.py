"""Code benchmark suite integration."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ..config import BackendName, GlobalConfig
from ..runtime import run_ultra_turn
from ..tools.logging import initialize_run_artifacts
from ..tools.verifiers import verify_code


SUITE_NAME = "code"
EVALPLUS_TASKS = {"humaneval_plus", "mbpp_plus"}
LIVECODEBENCH_TASKS = {"livecodebench", "live_code_bench", "lcb"}
SWEBENCH_TASKS = {"swebench", "swebench_lite", "swebench_verified"}


def describe_suite() -> dict[str, Any]:
    return {
        "suite": SUITE_NAME,
        "status": "implemented",
        "notes": "Supports EvalPlus, LiveCodeBench, SWE-bench wrappers, and local JSON code tasks.",
    }


def run_suite(
    *,
    model_ref: str,
    tasks: list[str],
    backend: BackendName,
    out: Path,
    config: GlobalConfig,
) -> dict[str, Any]:
    benchmark, options = parse_code_suite_request(tasks)
    if benchmark in EVALPLUS_TASKS:
        payload = run_evalplus_suite(model_ref=model_ref, benchmark=benchmark)
    elif benchmark in LIVECODEBENCH_TASKS:
        payload = run_livecodebench_suite(model_ref=model_ref, benchmark=benchmark, options=options)
    elif benchmark in SWEBENCH_TASKS:
        payload = run_swebench_suite(
            model_ref=model_ref,
            benchmark=benchmark,
            options=options,
            backend=backend,
            out=out,
            config=config,
        )
    elif benchmark == "local":
        payload = evaluate_local_code_tasks(
            model_ref=model_ref,
            backend=backend,
            config=config,
            task_file=Path(options["task_file"]),
        )
    else:
        raise RuntimeError(f"Unsupported code suite task: {benchmark}")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_code_suite_request(tasks: list[str]) -> tuple[str, dict[str, str]]:
    if not tasks:
        raise RuntimeError(
            "Code suite requires an EvalPlus/LiveCodeBench/SWE-bench task name or a path to a local JSON task file."
        )
    first = tasks[0]
    if Path(first).exists():
        return "local", {"task_file": first}
    benchmark = normalize_task_name(first)
    options: dict[str, str] = {}
    for item in tasks[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            options[key.strip()] = value.strip()
        elif Path(item).exists():
            options.setdefault("task_file", item)
        else:
            options[item] = "true"
    return benchmark, options


def run_evalplus_suite(*, model_ref: str, benchmark: str) -> dict[str, Any]:
    executable = shutil.which("evalplus.evaluate")
    if executable is None:
        raise RuntimeError("EvalPlus task requested but `evalplus.evaluate` is not available on PATH.")
    command = [executable, "--dataset", benchmark, "--model", model_ref]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "suite": SUITE_NAME,
        "benchmark": benchmark,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "pass@1": parse_pass_at_1(result.stdout, result.stderr),
    }
    if result.returncode != 0:
        raise RuntimeError(f"EvalPlus failed with exit code {result.returncode}")
    return payload


def run_livecodebench_suite(*, model_ref: str, benchmark: str, options: dict[str, str]) -> dict[str, Any]:
    if not module_available("lcb_runner.runner.main"):
        raise RuntimeError("LiveCodeBench requested but `lcb_runner.runner.main` is not installed.")
    command = [
        sys.executable,
        "-m",
        "lcb_runner.runner.main",
        "--model",
        model_ref,
        "--scenario",
        options.get("scenario", "codegeneration"),
    ]
    release_version = options.get("release_version")
    if release_version:
        command.extend(["--release_version", release_version])
    if options.get("evaluate", "true").lower() != "false":
        command.append("--evaluate")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "suite": SUITE_NAME,
        "benchmark": benchmark,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "pass@1": parse_pass_at_1(result.stdout, result.stderr),
    }
    if result.returncode != 0:
        raise RuntimeError(f"LiveCodeBench failed with exit code {result.returncode}")
    return payload


def run_swebench_suite(
    *,
    model_ref: str,
    benchmark: str,
    options: dict[str, str],
    backend: BackendName,
    out: Path,
    config: GlobalConfig,
) -> dict[str, Any]:
    if not module_available("swebench.harness.run_evaluation"):
        raise RuntimeError("SWE-bench requested but `swebench.harness.run_evaluation` is not installed.")
    predictions_path = resolve_swebench_predictions_path(
        model_ref=model_ref,
        benchmark=benchmark,
        options=options,
        backend=backend,
        out=out,
        config=config,
    )
    run_id = options.get("run_id", out.stem)
    dataset_name = options.get("dataset_name", benchmark)
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
    ]
    if "max_workers" in options:
        command.extend(["--max_workers", options["max_workers"]])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "suite": SUITE_NAME,
        "benchmark": benchmark,
        "command": command,
        "predictions_path": str(predictions_path),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "pass@1": parse_pass_at_1(result.stdout, result.stderr),
    }
    if result.returncode != 0:
        raise RuntimeError(f"SWE-bench failed with exit code {result.returncode}")
    return payload


def resolve_swebench_predictions_path(
    *,
    model_ref: str,
    benchmark: str,
    options: dict[str, str],
    backend: BackendName,
    out: Path,
    config: GlobalConfig,
) -> Path:
    if "predictions" in options:
        predictions_path = Path(options["predictions"])
        if not predictions_path.exists():
            raise RuntimeError(f"SWE-bench predictions file not found: {predictions_path}")
        return predictions_path
    if "task_file" not in options:
        raise RuntimeError(
            "SWE-bench requires either `predictions=<path>` or a local task file argument for prediction generation."
        )
    task_file = Path(options["task_file"])
    predictions_path = out.parent / f"{out.stem}_{benchmark}_predictions.json"
    build_swebench_predictions_from_local_tasks(
        model_ref=model_ref,
        backend=backend,
        config=config,
        task_file=task_file,
        predictions_path=predictions_path,
    )
    return predictions_path


def build_swebench_predictions_from_local_tasks(
    *,
    model_ref: str,
    backend: BackendName,
    config: GlobalConfig,
    task_file: Path,
    predictions_path: Path,
) -> Path:
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    predictions: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_config = deepcopy(config)
        task_config.structured_output.enabled = False
        artifacts = initialize_run_artifacts(
            command="eval-swebench",
            model_ref=model_ref,
            backend=backend,
            config=task_config.to_dict(),
            hardware=None,
            logdir=predictions_path.parent / f"runs/swebench-task-{index}",
            extra={"task_id": task.get("instance_id", index)},
            structured_output=False,
        )
        turn = run_ultra_turn(
            model_ref=model_ref,
            backend=backend,
            config=task_config,
            ultra_profile="coding",
            conversation=[{"role": "user", "content": task["prompt"]}],
            artifacts=artifacts,
            seed_base=index * 1000,
        )
        predictions.append(
            {
                "instance_id": task.get("instance_id", str(index)),
                "model_name_or_path": model_ref,
                "model_patch": turn.final_text,
            }
        )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(predictions, indent=2, sort_keys=True), encoding="utf-8")
    return predictions_path


def evaluate_local_code_tasks(*, model_ref: str, backend: BackendName, config: GlobalConfig, task_file: Path) -> dict[str, Any]:
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    passed = 0
    for index, task in enumerate(tasks):
        task_config = deepcopy(config)
        task_config.structured_output.enabled = False
        artifacts = initialize_run_artifacts(
            command="eval-code",
            model_ref=model_ref,
            backend=backend,
            config=task_config.to_dict(),
            hardware=None,
            logdir=task_file.parent / f"runs/code-task-{index}",
            extra={"task_id": task.get("task_id", index)},
            structured_output=False,
        )
        turn = run_ultra_turn(
            model_ref=model_ref,
            backend=backend,
            config=task_config,
            ultra_profile="coding",
            conversation=[{"role": "user", "content": task["prompt"]}],
            artifacts=artifacts,
            seed_base=index * 1000,
        )
        verification = verify_code(
            turn.final_text,
            function_name=task.get("function_name"),
            tests=list(task.get("tests", [])),
        )
        if verification.passed:
            passed += 1
        results.append(
            {
                "task_id": task.get("task_id", index),
                "verification": verification.to_dict(),
                "final_text": turn.final_text,
            }
        )
    total = max(len(results), 1)
    return {
        "suite": SUITE_NAME,
        "benchmark": "local",
        "tasks": results,
        "pass@1": passed / total,
    }


def parse_pass_at_1(stdout: str, stderr: str) -> float | None:
    for stream in (stdout, stderr):
        try:
            payload = json.loads(stream)
            if isinstance(payload, dict) and "pass@1" in payload:
                return float(payload["pass@1"])
        except Exception:
            pass
        for line in stream.splitlines():
            normalized = line.strip().lower()
            if "pass@1" not in normalized:
                continue
            _, _, tail = normalized.partition("pass@1")
            tail = tail.lstrip(" :=\t")
            try:
                return float(tail.split()[0])
            except Exception:
                continue
    return None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def normalize_task_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
