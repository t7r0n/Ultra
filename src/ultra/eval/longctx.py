"""Long-context benchmark suite integration."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

from ..config import BackendName, GlobalConfig
from ..runtime import run_ultra_turn
from ..tools.logging import initialize_run_artifacts


SUITE_NAME = "long"
LONGBENCH_TASKS = {"longbench_v2", "longbench"}
BABILONG_TASKS = {"babilong"}
NIAH_TASKS = {"niah", "needle_in_a_haystack", "needle_in_haystack"}


def describe_suite() -> dict[str, Any]:
    return {
        "suite": SUITE_NAME,
        "status": "implemented",
        "notes": "Supports LongBench v2, BABILong, NIAH utilities, and local JSON long-context tasks.",
    }


def run_suite(
    *,
    model_ref: str,
    tasks: list[str],
    backend: BackendName,
    out: Path,
    config: GlobalConfig,
) -> dict[str, Any]:
    benchmark, options = parse_long_suite_request(tasks)
    if benchmark == "local":
        payload = evaluate_long_tasks(
            model_ref=model_ref,
            backend=backend,
            config=config,
            task_payload=json.loads(Path(options["task_file"]).read_text(encoding="utf-8")),
            run_root=Path(options["task_file"]).parent / "runs",
            benchmark="local",
        )
    elif benchmark in LONGBENCH_TASKS:
        payload = run_longbench_suite(model_ref=model_ref, backend=backend, config=config, options=options)
    elif benchmark in BABILONG_TASKS:
        payload = run_babilong_suite(model_ref=model_ref, backend=backend, config=config, options=options)
    elif benchmark in NIAH_TASKS:
        payload = run_niah_suite(model_ref=model_ref, backend=backend, config=config, options=options, out=out)
    else:
        raise RuntimeError(f"Unsupported long-context suite task: {benchmark}")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_long_suite_request(tasks: list[str]) -> tuple[str, dict[str, str]]:
    if not tasks:
        raise RuntimeError("Long-context suite requires a benchmark name or a path to a local JSON task file.")
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


def run_longbench_suite(
    *,
    model_ref: str,
    backend: BackendName,
    config: GlobalConfig,
    options: dict[str, str],
) -> dict[str, Any]:
    if not module_available("datasets"):
        raise RuntimeError("LongBench requested but `datasets` is not installed.")
    from datasets import load_dataset  # type: ignore

    dataset_name = options.get("dataset_name", "THUDM/LongBench-v2")
    split = options.get("split", "train")
    limit = int(options.get("limit", "10"))
    dataset = load_dataset(dataset_name, split=split)
    subset = options.get("subset")
    rows: list[dict[str, Any]] = []
    for row in dataset:
        row_dict = dict(row)
        if subset and str(row_dict.get("task") or row_dict.get("dataset") or "") != subset:
            continue
        rows.append(row_dict)
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError("LongBench dataset selection returned no rows.")
    payload = evaluate_long_tasks(
        model_ref=model_ref,
        backend=backend,
        config=config,
        task_payload=[normalize_long_task_row(row, benchmark="longbench_v2") for row in rows],
        run_root=Path("runs") / "longbench_v2",
        benchmark="longbench_v2",
    )
    payload["dataset_name"] = dataset_name
    payload["split"] = split
    payload["subset"] = subset
    return payload


def run_babilong_suite(
    *,
    model_ref: str,
    backend: BackendName,
    config: GlobalConfig,
    options: dict[str, str],
) -> dict[str, Any]:
    if not module_available("datasets"):
        raise RuntimeError("BABILong requested but `datasets` is not installed.")
    from datasets import load_dataset  # type: ignore

    dataset_name = options.get("dataset_name", "RMT-team/babilong")
    split = options.get("split", "train")
    limit = int(options.get("limit", "10"))
    subset = options.get("subset")
    dataset = load_dataset(dataset_name, split=split)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        row_dict = dict(row)
        if subset and str(row_dict.get("task") or row_dict.get("dataset") or row_dict.get("name") or "") != subset:
            continue
        rows.append(row_dict)
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError("BABILong dataset selection returned no rows.")
    payload = evaluate_long_tasks(
        model_ref=model_ref,
        backend=backend,
        config=config,
        task_payload=[normalize_long_task_row(row, benchmark="babilong") for row in rows],
        run_root=Path("runs") / "babilong",
        benchmark="babilong",
    )
    payload["dataset_name"] = dataset_name
    payload["split"] = split
    payload["subset"] = subset
    return payload


def run_niah_suite(
    *,
    model_ref: str,
    backend: BackendName,
    config: GlobalConfig,
    options: dict[str, str],
    out: Path,
) -> dict[str, Any]:
    haystack = options.get("haystack")
    haystack_path = options.get("haystack_path")
    if haystack is None and haystack_path is not None:
        haystack = Path(haystack_path).read_text(encoding="utf-8")
    if haystack is None:
        haystack = ("This is filler context. " * 512).strip()
    needle = options.get("needle", "The launch code is CORAL-47.")
    question = options.get("question", "What is the launch code?")
    answer = options.get("answer", needle)
    position = float(options.get("needle_position", "0.5"))
    task_payload = [build_niah_task(haystack=haystack, needle=needle, question=question, answer=answer, position=position)]
    payload = evaluate_long_tasks(
        model_ref=model_ref,
        backend=backend,
        config=config,
        task_payload=task_payload,
        run_root=out.parent / "runs" / "niah",
        benchmark="niah",
    )
    payload["needle_position"] = position
    return payload


def evaluate_long_tasks(
    *,
    model_ref: str,
    backend: BackendName,
    config: GlobalConfig,
    task_payload: list[dict[str, Any]],
    run_root: Path,
    benchmark: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    correct = 0.0
    for index, task in enumerate(task_payload):
        task_config = deepcopy(config)
        task_config.context_window = max(
            task_config.context_window if isinstance(task_config.context_window, int) else 0,
            estimated_context_window(task["context"]),
        )
        prompt = f"{task['context']}\n\nQuestion: {task['question']}"
        artifacts = initialize_run_artifacts(
            command=f"eval-{benchmark}",
            model_ref=model_ref,
            backend=backend,
            config=task_config.to_dict(),
            hardware=None,
            logdir=run_root / f"{benchmark}-task-{index}",
            extra={"task_id": task.get("task_id", index)},
            structured_output=False,
        )
        turn = run_ultra_turn(
            model_ref=model_ref,
            backend=backend,
            config=task_config,
            ultra_profile="reasoning",
            conversation=[{"role": "user", "content": prompt}],
            artifacts=artifacts,
            seed_base=index * 1000,
        )
        expected_values = normalize_expected_answers(task.get("answer"))
        actual = turn.final_text.strip().lower()
        score = 1.0 if any(expected and expected in actual for expected in expected_values) else 0.0
        correct += score
        results.append(
            {
                "task_id": task.get("task_id", index),
                "score": score,
                "expected": expected_values,
                "final_text": turn.final_text,
            }
        )
    return {
        "suite": SUITE_NAME,
        "benchmark": benchmark,
        "accuracy": correct / max(len(results), 1),
        "tasks": results,
    }


def normalize_long_task_row(row: dict[str, Any], *, benchmark: str) -> dict[str, Any]:
    context = first_present(row, "context", "input", "document", "story", default="")
    question = first_present(row, "question", "query", "instruction", default="")
    answer = row.get("answer")
    if answer is None:
        answer = row.get("answers")
    if answer is None:
        answer = row.get("target")
    task_id = row.get("task_id") or row.get("id") or row.get("instance_id") or row.get("_id")
    return {
        "task_id": task_id or f"{benchmark}-{abs(hash(json.dumps(row, sort_keys=True, default=str)))}",
        "context": str(context),
        "question": str(question),
        "answer": answer,
    }


def build_niah_task(*, haystack: str, needle: str, question: str, answer: str, position: float) -> dict[str, Any]:
    words = haystack.split()
    insert_at = min(max(int(len(words) * position), 0), len(words))
    context_words = words[:insert_at] + [needle] + words[insert_at:]
    return {
        "task_id": "niah-0",
        "context": " ".join(context_words),
        "question": question,
        "answer": answer,
    }


def estimated_context_window(context: str) -> int:
    return max(len(context) // 4, 1024)


def normalize_expected_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(item).strip().lower() for item in value]
    return [str(value).strip().lower()]


def first_present(payload: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def normalize_task_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
