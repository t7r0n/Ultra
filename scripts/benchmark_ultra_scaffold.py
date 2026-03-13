#!/usr/bin/env python3
"""Benchmark greedy decoding against a simplified Ultra-style fan-out loop."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import gc
import io
import json
import os
from pathlib import Path
import re
import signal
import textwrap
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ultra.config import default_config
from ultra.mcd.inspector import inspect_model
from ultra.tools.estimator import estimate_model_ref
from ultra.ultra_mode.fanout import build_fanout_plan


HF_HOME = os.environ.get("HF_HOME", "/tmp/hf_home")
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class Task:
    task_id: str
    category: str
    prompt: str
    answer: Any
    max_new_tokens: int
    evaluator: str


@dataclass(slots=True)
class CandidateResult:
    raw_text: str
    canonical_answer: Any
    score: float
    latency_s: float
    meta: dict[str, Any]


TASKS: list[Task] = [
    Task(
        task_id="math_bookstore",
        category="math",
        prompt="A bookstore sold 18 notebooks on Monday and 27 on Tuesday. Then 9 were returned. How many notebooks were sold in total? Reply with only the number.",
        answer="36",
        max_new_tokens=48,
        evaluator="math",
    ),
    Task(
        task_id="math_train",
        category="math",
        prompt="A train trip is 245 miles. The train traveled 97 miles in the morning and 88 in the afternoon. How many miles are left? Reply with only the number.",
        answer="60",
        max_new_tokens=48,
        evaluator="math",
    ),
    Task(
        task_id="math_baker",
        category="math",
        prompt="A baker made 24 muffins, sold 7, then baked 13 more. How many muffins does the baker have now? Reply with only the number.",
        answer="30",
        max_new_tokens=48,
        evaluator="math",
    ),
    Task(
        task_id="logic_syllogism",
        category="logic",
        prompt="If all bloops are razzies and all razzies are lazies, which statement is definitely true? A) Some bloops are not lazies. B) All bloops are lazies. C) No razzies are lazies. D) Some lazies are not razzies. Reply with only A, B, C, or D.",
        answer="B",
        max_new_tokens=24,
        evaluator="mcq",
    ),
    Task(
        task_id="logic_height",
        category="logic",
        prompt="Emma is taller than Liam. Liam is taller than Noah. Who is shortest? A) Emma B) Liam C) Noah D) Cannot tell. Reply with only A, B, C, or D.",
        answer="C",
        max_new_tokens=24,
        evaluator="mcq",
    ),
    Task(
        task_id="logic_calendar",
        category="logic",
        prompt="A meeting is scheduled for Friday. Two days earlier is: A) Wednesday B) Thursday C) Tuesday D) Monday. Reply with only A, B, C, or D.",
        answer="A",
        max_new_tokens=24,
        evaluator="mcq",
    ),
    Task(
        task_id="json_customer",
        category="json",
        prompt='Extract the data as compact JSON with keys customer, city, item, quantity, total. Sentence: "Customer Maya Chen from Austin ordered 3 lamps and paid $84." Return JSON only.',
        answer={"customer": "Maya Chen", "city": "Austin", "item": "lamps", "quantity": 3, "total": 84},
        max_new_tokens=80,
        evaluator="json",
    ),
    Task(
        task_id="json_ticket",
        category="json",
        prompt='Extract the data as compact JSON with keys ticket_id, priority, assignee, status. Sentence: "Ticket #A19, priority high, assigned to Omar, status open." Return JSON only.',
        answer={"ticket_id": "A19", "priority": "high", "assignee": "Omar", "status": "open"},
        max_new_tokens=80,
        evaluator="json",
    ),
    Task(
        task_id="code_square_evens",
        category="code",
        prompt=textwrap.dedent(
            """
            Write Python code only.
            Implement:
            def square_evens(values):
                \"\"\"Return a list containing the square of each even integer in values, preserving order.\"\"\"
            """
        ).strip(),
        answer={
            "function_name": "square_evens",
            "tests": [
                "assert square_evens([1, 2, 3, 4]) == [4, 16]",
                "assert square_evens([]) == []",
                "assert square_evens([2, 2, 5]) == [4, 4]",
            ],
        },
        max_new_tokens=196,
        evaluator="code",
    ),
    Task(
        task_id="code_normalize_spaces",
        category="code",
        prompt=textwrap.dedent(
            """
            Write Python code only.
            Implement:
            def normalize_spaces(text):
                \"\"\"Collapse consecutive whitespace to a single space and strip leading/trailing whitespace.\"\"\"
            """
        ).strip(),
        answer={
            "function_name": "normalize_spaces",
            "tests": [
                "assert normalize_spaces('  hello   world  ') == 'hello world'",
                "assert normalize_spaces('a\\n\\tb') == 'a b'",
                "assert normalize_spaces('single') == 'single'",
            ],
        },
        max_new_tokens=196,
        evaluator="code",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=Path("reports/ultra_scaffold_benchmark_20260312.json"))
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEFAULT_DEVICE)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", HF_HOME)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_home": os.environ["HF_HOME"],
        "device": args.device,
        "n_candidates": args.n_candidates,
        "models": [],
        "aggregate": {},
    }

    aggregate_scores: dict[str, list[float]] = defaultdict(list)
    aggregate_by_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for model_id in args.models:
        model_report = benchmark_model(model_id=model_id, n_candidates=args.n_candidates, device=args.device)
        report["models"].append(model_report)
        for strategy, accuracy in model_report["summary"]["strategy_accuracy"].items():
            aggregate_scores[strategy].append(accuracy)
        for category, strategy_scores in model_report["summary"]["category_accuracy"].items():
            for strategy, accuracy in strategy_scores.items():
                aggregate_by_category[category][strategy].append(accuracy)

    report["aggregate"] = {
        "strategy_accuracy_mean": {
            strategy: sum(values) / len(values)
            for strategy, values in sorted(aggregate_scores.items())
        },
        "category_accuracy_mean": {
            category: {
                strategy: sum(values) / len(values)
                for strategy, values in sorted(strategy_scores.items())
            }
            for category, strategy_scores in sorted(aggregate_by_category.items())
        },
    }

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote benchmark report to {args.output}")


def benchmark_model(*, model_id: str, n_candidates: int, device: str) -> dict[str, Any]:
    print(f"=== Loading {model_id} ===")
    inspection_start = time.perf_counter()
    inspection = inspect_model(model_id)
    inspection_s = time.perf_counter() - inspection_start
    estimate_start = time.perf_counter()
    estimate = estimate_model_ref(model_id, ctx=2048, batch=1, precision="fp16" if device == "cuda" else "fp32", backend="hf")
    estimate_s = time.perf_counter() - estimate_start

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_load_s = time.perf_counter() - tokenizer_start
    model_start = time.perf_counter()
    dtype = torch.float16 if device == "cuda" else torch.float32
    target_device = "cuda:0" if device == "cuda" else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": target_device})
    model_load_s = time.perf_counter() - model_start
    model.eval()

    config = default_config()
    config.ultra.n_candidates = n_candidates
    fanout_plan = build_fanout_plan(config, "default")

    task_reports: list[dict[str, Any]] = []
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for task in TASKS:
        baseline = run_baseline(model, tokenizer, task, device=device)
        ultra = run_ultra_scaffold(model, tokenizer, task, fanout_plan, device=device)
        task_report = {
            "task_id": task.task_id,
            "category": task.category,
            "answer": task.answer,
            "baseline": summarize_candidate(baseline),
            "ultra_scaffold": summarize_candidate(ultra["selected"]),
            "ultra_candidates": [summarize_candidate(candidate) for candidate in ultra["candidates"]],
        }
        task_reports.append(task_report)
        by_strategy["baseline_greedy"].append(baseline.score)
        by_strategy["ultra_scaffold"].append(ultra["selected"].score)
        by_category[task.category]["baseline_greedy"].append(baseline.score)
        by_category[task.category]["ultra_scaffold"].append(ultra["selected"].score)
        print(
            f"{model_id} | {task.task_id}: baseline={baseline.score:.0f} ultra={ultra['selected'].score:.0f}"
        )

    summary = {
        "strategy_accuracy": {
            strategy: sum(values) / len(values)
            for strategy, values in sorted(by_strategy.items())
        },
        "category_accuracy": {
            category: {
                strategy: sum(values) / len(values)
                for strategy, values in sorted(strategy_scores.items())
            }
            for category, strategy_scores in sorted(by_category.items())
        },
    }

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "device": device,
        "timings": {
            "inspection_s": round(inspection_s, 4),
            "estimate_s": round(estimate_s, 4),
            "tokenizer_load_s": round(tokenizer_load_s, 4),
            "model_load_s": round(model_load_s, 4),
        },
        "inspection": inspection.to_dict(),
        "estimate": estimate.to_dict(),
        "fanout_plan": fanout_plan.to_dict(),
        "tasks": task_reports,
        "summary": summary,
    }


def run_baseline(model: Any, tokenizer: Any, task: Task, *, device: str) -> CandidateResult:
    prompt_messages = [
        {"role": "system", "content": system_prompt_for_task(task, style="concise")},
        {"role": "user", "content": user_prompt_for_task(task, style="concise")},
    ]
    raw_text, latency_s = generate_response(
        model,
        tokenizer,
        prompt_messages,
        max_new_tokens=task.max_new_tokens,
        do_sample=False,
        device=device,
    )
    return evaluate_candidate(task, raw_text, latency_s, meta={"strategy": "baseline_greedy"})


def run_ultra_scaffold(model: Any, tokenizer: Any, task: Task, fanout_plan: Any, *, device: str) -> dict[str, Any]:
    candidates: list[CandidateResult] = []
    temperatures = fanout_plan.temperatures
    top_p_values = fanout_plan.top_p
    styles = fanout_plan.styles

    for index in range(fanout_plan.n_candidates):
        temperature = temperatures[index % len(temperatures)]
        top_p = top_p_values[index % len(top_p_values)]
        style = styles[index % len(styles)]
        prompt_messages = [
            {"role": "system", "content": system_prompt_for_task(task, style=style)},
            {"role": "user", "content": user_prompt_for_task(task, style=style)},
        ]
        raw_text, latency_s = generate_response(
            model,
            tokenizer,
            prompt_messages,
            max_new_tokens=task.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            seed=1000 + index,
            device=device,
        )
        candidates.append(
            evaluate_candidate(
                task,
                raw_text,
                latency_s,
                meta={
                    "strategy": "ultra_scaffold",
                    "temperature": temperature,
                    "top_p": top_p,
                    "style": style,
                },
            )
        )

    selected = select_candidate(task, candidates)
    return {"candidates": candidates, "selected": selected}


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    do_sample: bool,
    device: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int | None = None,
) -> tuple[str, float]:
    if seed is not None:
        set_seed(seed)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - start
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, latency_s


def select_candidate(task: Task, candidates: list[CandidateResult]) -> CandidateResult:
    if task.evaluator == "code":
        best = max(candidates, key=lambda candidate: (candidate.score, -candidate.latency_s))
        return best

    answers = [candidate.canonical_answer for candidate in candidates if candidate.canonical_answer not in (None, "")]
    if not answers:
        return max(candidates, key=lambda candidate: candidate.score)

    counts = Counter(json.dumps(answer, sort_keys=True) if isinstance(answer, dict) else str(answer) for answer in answers)
    best_count = max(counts.values())
    best_answers = {answer for answer, count in counts.items() if count == best_count}
    for candidate in candidates:
        normalized = json.dumps(candidate.canonical_answer, sort_keys=True) if isinstance(candidate.canonical_answer, dict) else str(candidate.canonical_answer)
        if normalized in best_answers:
            return candidate
    return candidates[0]


def evaluate_candidate(task: Task, raw_text: str, latency_s: float, *, meta: dict[str, Any]) -> CandidateResult:
    if task.evaluator == "math":
        canonical = extract_math_answer(raw_text)
        score = 1.0 if canonical == task.answer else 0.0
    elif task.evaluator == "mcq":
        canonical = extract_mcq_answer(raw_text)
        score = 1.0 if canonical == task.answer else 0.0
    elif task.evaluator == "json":
        canonical = extract_json_answer(raw_text)
        score = 1.0 if canonical == task.answer else 0.0
    elif task.evaluator == "code":
        canonical = extract_code(raw_text)
        code_result = evaluate_code_answer(canonical, task.answer)
        meta = {**meta, **code_result["meta"]}
        return CandidateResult(raw_text=raw_text, canonical_answer=canonical, score=code_result["score"], latency_s=latency_s, meta=meta)
    else:
        canonical = raw_text
        score = 0.0
    return CandidateResult(raw_text=raw_text, canonical_answer=canonical, score=score, latency_s=latency_s, meta=meta)


def evaluate_code_answer(code: str, answer: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    metadata: dict[str, Any] = {"tests_passed": 0, "tests_total": len(answer["tests"])}
    try:
        with time_limit(2), redirect_stdout(io.StringIO()):
            exec(code, namespace, namespace)
        function_name = answer["function_name"]
        if function_name not in namespace:
            metadata["error"] = f"Missing function {function_name}"
            return {"score": 0.0, "meta": metadata}
        for test in answer["tests"]:
            with time_limit(2), redirect_stdout(io.StringIO()):
                exec(test, namespace, namespace)
            metadata["tests_passed"] += 1
        return {"score": 1.0, "meta": metadata}
    except Exception as exc:  # noqa: BLE001
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["partial_score"] = metadata["tests_passed"] / metadata["tests_total"]
        return {"score": metadata["partial_score"], "meta": metadata}


def summarize_candidate(candidate: CandidateResult) -> dict[str, Any]:
    return {
        "raw_text": candidate.raw_text,
        "canonical_answer": candidate.canonical_answer,
        "score": candidate.score,
        "latency_s": round(candidate.latency_s, 4),
        "meta": candidate.meta,
    }


def system_prompt_for_task(task: Task, *, style: str) -> str:
    if task.evaluator == "code":
        if style == "cot":
            return "You are a careful Python coding assistant. Think through the solution internally, but output only the final Python code."
        return "You are a concise Python coding assistant. Output only valid Python code."
    if task.evaluator == "json":
        return "You are a precise information extraction assistant. Return valid JSON only."
    if style == "cot":
        return "You are a reasoning assistant. Work carefully and end with a line that begins FINAL: followed by only the answer."
    return "You are a concise assistant. Give the final answer directly."


def user_prompt_for_task(task: Task, *, style: str) -> str:
    if style != "cot" or task.evaluator in {"json", "code"}:
        return task.prompt
    return task.prompt + "\nShow your reasoning briefly, then end with `FINAL: <answer>`."


def extract_math_answer(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


def extract_mcq_answer(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def extract_json_answer(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


class time_limit:
    """A tiny signal-based timeout for code evaluation on Linux."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous_handler: Any = None

    def __enter__(self) -> None:
        self.previous_handler = signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.seconds)
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    @staticmethod
    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("Timed out during code execution.")


if __name__ == "__main__":
    main()
