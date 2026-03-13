#!/usr/bin/env python3
"""Benchmark Qwen3/Qwen3.5 thinking and non-thinking modes on GPU."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import gc
import io
import json
import os
from pathlib import Path
import re
import signal
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed


HF_HOME = os.environ.get("HF_HOME", "/tmp/hf_home")
JSON_DECODER = json.JSONDecoder()

SOURCE_PRIORITY = {
    "json": 5,
    "boxed": 4,
    "final": 3,
    "fallback_last_number": 2,
    "fallback_choice": 2,
    "fallback_text": 1,
    "none": 0,
}

STRUCTURED_SOURCES = {"json", "boxed", "final"}


@dataclass(slots=True)
class Task:
    task_id: str
    category: str
    prompt: str
    answer: Any
    evaluator: str
    aliases: list[str] = field(default_factory=list)
    max_new_tokens_single: int = 128
    max_new_tokens_thinking: int = 320


@dataclass(slots=True)
class Condition:
    label: str
    model_id: str
    thinking: bool
    family: str
    text_temperature: float
    text_top_p: float
    text_top_k: int
    text_min_p: float
    text_repetition_penalty: float = 1.0
    code_temperature: float | None = None
    code_top_p: float | None = None
    code_top_k: int | None = None
    code_min_p: float | None = None
    code_repetition_penalty: float | None = None

    def params_for_task(self, task: Task) -> dict[str, Any]:
        if task.category == "code" and self.code_temperature is not None:
            return {
                "temperature": self.code_temperature,
                "top_p": self.code_top_p,
                "top_k": self.code_top_k,
                "min_p": self.code_min_p,
                "repetition_penalty": self.code_repetition_penalty,
            }
        return {
            "temperature": self.text_temperature,
            "top_p": self.text_top_p,
            "top_k": self.text_top_k,
            "min_p": self.text_min_p,
            "repetition_penalty": self.text_repetition_penalty,
        }


@dataclass(slots=True)
class CandidateResult:
    raw_text: str
    cleaned_text: str
    canonical_answer: Any
    score: float
    latency_s: float
    generated_tokens: int
    think_block_present: bool
    meta: dict[str, Any]


@dataclass(slots=True)
class GenerationResult:
    raw_text: str
    generated_tokens: int
    latency_s: float
    avg_logprob: float
    sequence_ids: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = [tuple(sequence) for sequence in sequences if sequence]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
        del scores, kwargs
        if not self.sequences:
            return False
        for sequence in self.sequences:
            length = len(sequence)
            if input_ids.shape[1] < length:
                continue
            target = torch.tensor(sequence, device=input_ids.device, dtype=input_ids.dtype)
            if bool(torch.all(input_ids[:, -length:] == target, dim=1).any()):
                return True
        return False


TASKS: list[Task] = [
    Task(
        task_id="math_tickets",
        category="math",
        prompt=(
            "A theater sold adult tickets for $17 and child tickets for $9. "
            "It sold 52 tickets total and collected $708. "
            "How many child tickets were sold?"
        ),
        answer="22",
        evaluator="number",
        max_new_tokens_single=96,
        max_new_tokens_thinking=256,
    ),
    Task(
        task_id="math_pipes",
        category="math",
        prompt=(
            "Pipe A fills a tank in 6 hours. Pipe B fills it in 9 hours. "
            "A drain empties it in 18 hours. If all three run together for 3 hours starting from empty, "
            "what fraction of the tank is still unfilled?"
        ),
        answer="1/3",
        evaluator="fraction",
        aliases=["0.3333333333", "0.333333333", "0.3333333333333333"],
        max_new_tokens_single=96,
        max_new_tokens_thinking=256,
    ),
    Task(
        task_id="math_divisible_by_5",
        category="math",
        prompt=(
            "How many 4-digit numbers have distinct digits, a nonzero first digit, and are divisible by 5? "
            "Give the count."
        ),
        answer="952",
        evaluator="number",
        max_new_tokens_single=128,
        max_new_tokens_thinking=320,
    ),
    Task(
        task_id="math_mixture",
        category="math",
        prompt=(
            "A chemist mixes 30 liters of a 20% acid solution with some amount of a 50% acid solution "
            "to obtain a 32% acid mixture. How many liters of the 50% solution are needed? "
            "Give the number of liters needed."
        ),
        answer="20",
        evaluator="number",
        max_new_tokens_single=96,
        max_new_tokens_thinking=256,
    ),
    Task(
        task_id="logic_knights",
        category="logic",
        prompt=(
            "On an island, knights always tell the truth and knaves always lie. "
            "A says: `B is a knave.` "
            "B says: `A and I are of different types.` "
            "Who is the knight? Reply with only `A` or `B`."
        ),
        answer="B",
        evaluator="choice",
        max_new_tokens_single=96,
        max_new_tokens_thinking=256,
    ),
    Task(
        task_id="logic_lineup",
        category="logic",
        prompt=(
            "Five students Ana, Bo, Cy, Di, and Ez stand in a line. "
            "Ana is somewhere before Bo. Cy is somewhere after Bo. Di is before Ana. Ez is after Cy. "
            "Who must be in the middle?"
        ),
        answer="Bo",
        evaluator="text",
        aliases=["bo"],
        max_new_tokens_single=96,
        max_new_tokens_thinking=256,
    ),
    Task(
        task_id="logic_mixed_box",
        category="logic",
        prompt=(
            "You have three boxes labeled `Apples`, `Oranges`, and `Mixed`, but every label is wrong. "
            "You may draw one fruit from exactly one box to determine all contents. "
            "Which labeled box should you draw from first?"
        ),
        answer="Mixed",
        evaluator="text",
        aliases=["mixed box", "the box labeled mixed"],
        max_new_tokens_single=96,
        max_new_tokens_thinking=256,
    ),
    Task(
        task_id="code_normalize_spaces",
        category="code",
        prompt=(
            "Write Python code only. Implement:\n"
            "def normalize_spaces(text):\n"
            "    \"\"\"Collapse consecutive whitespace to a single space and strip leading/trailing whitespace.\"\"\"\n"
        ),
        answer={
            "function_name": "normalize_spaces",
            "tests": [
                "assert normalize_spaces('  hello   world  ') == 'hello world'",
                "assert normalize_spaces('a\\n\\tb') == 'a b'",
                "assert normalize_spaces('single') == 'single'",
            ],
        },
        evaluator="code",
        max_new_tokens_single=160,
        max_new_tokens_thinking=320,
    ),
    Task(
        task_id="code_merge_intervals",
        category="code",
        prompt=(
            "Write Python code only. Implement:\n"
            "def merge_intervals(intervals):\n"
            "    \"\"\"Given a list of [start, end] inclusive integer intervals, merge overlaps and return them sorted by start.\"\"\"\n"
        ),
        answer={
            "function_name": "merge_intervals",
            "tests": [
                "assert merge_intervals([[1,3],[2,4],[7,8]]) == [[1,4],[7,8]]",
                "assert merge_intervals([]) == []",
                "assert merge_intervals([[5,5],[1,2],[2,3]]) == [[1,3],[5,5]]",
            ],
        },
        evaluator="code",
        max_new_tokens_single=192,
        max_new_tokens_thinking=384,
    ),
    Task(
        task_id="code_first_unique_char",
        category="code",
        prompt=(
            "Write Python code only. Implement:\n"
            "def first_unique_char(text):\n"
            "    \"\"\"Return the first character that appears exactly once in text, or None if no such character exists.\"\"\"\n"
        ),
        answer={
            "function_name": "first_unique_char",
            "tests": [
                "assert first_unique_char('swiss') == 'w'",
                "assert first_unique_char('aabb') is None",
                "assert first_unique_char('leetcode') == 'l'",
            ],
        },
        evaluator="code",
        max_new_tokens_single=192,
        max_new_tokens_thinking=384,
    ),
]


CONDITIONS: list[Condition] = [
    Condition(
        label="qwen3.5_0.8b_nonthinking",
        model_id="Qwen/Qwen3.5-0.8B",
        thinking=False,
        family="qwen3.5",
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=20,
        text_min_p=0.0,
        text_repetition_penalty=1.0,
    ),
    Condition(
        label="qwen3.5_0.8b_thinking",
        model_id="Qwen/Qwen3.5-0.8B",
        thinking=True,
        family="qwen3.5",
        text_temperature=1.0,
        text_top_p=0.95,
        text_top_k=20,
        text_min_p=0.0,
        text_repetition_penalty=1.0,
        code_temperature=0.6,
        code_top_p=0.95,
        code_top_k=20,
        code_min_p=0.0,
        code_repetition_penalty=1.0,
    ),
    Condition(
        label="qwen3_1.7b_nonthinking",
        model_id="Qwen/Qwen3-1.7B",
        thinking=False,
        family="qwen3",
        text_temperature=0.7,
        text_top_p=0.8,
        text_top_k=20,
        text_min_p=0.0,
        text_repetition_penalty=1.0,
    ),
    Condition(
        label="qwen3_1.7b_thinking",
        model_id="Qwen/Qwen3-1.7B",
        thinking=True,
        family="qwen3",
        text_temperature=0.6,
        text_top_p=0.95,
        text_top_k=20,
        text_min_p=0.0,
        text_repetition_penalty=1.0,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument(
        "--single-token-multiplier",
        type=float,
        default=1.0,
        help="Multiply max_new_tokens for non-thinking generations.",
    )
    parser.add_argument(
        "--thinking-token-multiplier",
        type=float,
        default=1.0,
        help="Multiply max_new_tokens for thinking generations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qwen_reasoning_modes_gpu_20260312.json"),
    )
    parser.add_argument("--condition", action="append", help="Run only the named condition label.")
    parser.add_argument("--task", action="append", help="Run only the named task id.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU-only benchmark requested, but CUDA is not available.")

    selected_conditions = [condition for condition in CONDITIONS if not args.condition or condition.label in args.condition]
    if not selected_conditions:
        raise SystemExit("No matching conditions selected.")

    os.environ.setdefault("HF_HOME", HF_HOME)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    selected_tasks = [task for task in TASKS if not args.task or task.task_id in args.task]
    if not selected_tasks:
        raise SystemExit("No matching tasks selected.")

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cuda",
        "dtype": "fp16",
        "n_candidates": args.n_candidates,
        "single_token_multiplier": args.single_token_multiplier,
        "thinking_token_multiplier": args.thinking_token_multiplier,
        "tasks_run": [task.task_id for task in selected_tasks],
        "conditions": [],
    }

    for condition in selected_conditions:
        condition_report = run_condition(
            condition,
            tasks=selected_tasks,
            n_candidates=args.n_candidates,
            single_token_multiplier=args.single_token_multiplier,
            thinking_token_multiplier=args.thinking_token_multiplier,
        )
        report["conditions"].append(condition_report)

    report["aggregate"] = aggregate_report(report["conditions"])
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote benchmark report to {args.output}", flush=True)


def run_condition(
    condition: Condition,
    *,
    tasks: list[Task],
    n_candidates: int,
    single_token_multiplier: float,
    thinking_token_multiplier: float,
) -> dict[str, Any]:
    print(f"=== Loading {condition.label} ({condition.model_id}) ===", flush=True)
    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(condition.model_id)
    tokenizer_load_s = time.perf_counter() - tokenizer_start

    model_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        condition.model_id,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
    )
    model.eval()
    torch.cuda.empty_cache()
    model_load_s = time.perf_counter() - model_start

    task_reports: list[dict[str, Any]] = []
    scores: dict[str, list[float]] = defaultdict(list)
    category_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    category_times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    selected_candidate_total_tokens = 0
    selected_candidate_total_think = 0
    selected_candidate_total_think_closed = 0
    selected_candidate_total_answer_stage = 0
    selected_candidate_total_structured = 0
    selected_candidate_total_fallback = 0
    selected_candidate_total_truncated_inside_think = 0

    for task in tasks:
        single_pass = run_single_pass(
            model,
            tokenizer,
            condition,
            task,
            single_token_multiplier=single_token_multiplier,
            thinking_token_multiplier=thinking_token_multiplier,
        )
        ultra = run_ultra_fanout(
            model,
            tokenizer,
            condition,
            task,
            n_candidates=n_candidates,
            single_token_multiplier=single_token_multiplier,
            thinking_token_multiplier=thinking_token_multiplier,
        )
        selected_candidate = ultra["selected"]
        selected_candidate_total_tokens += selected_candidate.generated_tokens
        selected_candidate_total_think += int(selected_candidate.think_block_present)
        selected_candidate_total_think_closed += int(selected_candidate.meta.get("think_closed", False))
        selected_candidate_total_answer_stage += int(selected_candidate.meta.get("answer_stage_reached", False))
        selected_candidate_total_structured += int(selected_candidate.meta.get("canonical_source") in STRUCTURED_SOURCES)
        selected_candidate_total_fallback += int(
            str(selected_candidate.meta.get("canonical_source", "")).startswith("fallback_")
        )
        selected_candidate_total_truncated_inside_think += int(
            selected_candidate.meta.get("truncated_inside_think", False)
        )

        task_reports.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "answer": task.answer,
                "single_pass": summarize_candidate(single_pass),
                "ultra_selected": summarize_candidate(selected_candidate),
                "ultra_oracle_best": summarize_candidate(ultra["oracle_best"]),
                "ultra_candidates": [summarize_candidate(candidate) for candidate in ultra["candidates"]],
            }
        )

        scores["single_pass"].append(single_pass.score)
        scores["ultra_fanout"].append(selected_candidate.score)
        scores["oracle_best_of_k"].append(ultra["oracle_best"].score)
        category_scores[task.category]["single_pass"].append(single_pass.score)
        category_scores[task.category]["ultra_fanout"].append(selected_candidate.score)
        category_scores[task.category]["oracle_best_of_k"].append(ultra["oracle_best"].score)
        category_times[task.category]["single_pass"].append(single_pass.latency_s)
        category_times[task.category]["ultra_fanout"].append(sum(candidate.latency_s for candidate in ultra["candidates"]))

        print(
            f"{condition.label} | {task.task_id}: single={single_pass.score:.2f} "
            f"ultra={selected_candidate.score:.2f}",
            flush=True,
        )

    summary = {
        "accuracy": {name: sum(values) / len(values) for name, values in sorted(scores.items())},
        "category_accuracy": {
            category: {name: sum(values) / len(values) for name, values in sorted(strategy.items())}
            for category, strategy in sorted(category_scores.items())
        },
        "category_generation_time_s": {
            category: {name: sum(values) for name, values in sorted(strategy.items())}
            for category, strategy in sorted(category_times.items())
        },
        "avg_selected_generated_tokens": selected_candidate_total_tokens / len(tasks),
        "selected_think_block_rate": selected_candidate_total_think / len(tasks),
        "selected_think_closed_rate": selected_candidate_total_think_closed / len(tasks),
        "selected_answer_stage_reached_rate": selected_candidate_total_answer_stage / len(tasks),
        "selected_structured_answer_rate": selected_candidate_total_structured / len(tasks),
        "selected_fallback_answer_rate": selected_candidate_total_fallback / len(tasks),
        "selected_truncated_inside_think_rate": selected_candidate_total_truncated_inside_think / len(tasks),
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "condition": asdict(condition),
        "timings": {
            "tokenizer_load_s": round(tokenizer_load_s, 4),
            "model_load_s": round(model_load_s, 4),
        },
        "tasks": task_reports,
        "summary": summary,
    }


def run_single_pass(
    model: Any,
    tokenizer: Any,
    condition: Condition,
    task: Task,
    *,
    single_token_multiplier: float,
    thinking_token_multiplier: float,
) -> CandidateResult:
    messages = [
        {"role": "system", "content": system_prompt(task, concise=True)},
        {"role": "user", "content": single_user_prompt(task)},
    ]
    generation = generate_once(
        model,
        tokenizer,
        messages,
        condition=condition,
        task=task,
        strategy="single_pass",
        candidate_index=0,
        single_token_multiplier=single_token_multiplier,
        thinking_token_multiplier=thinking_token_multiplier,
    )
    return evaluate_candidate(
        task,
        generation.raw_text,
        generation.generated_tokens,
        generation.latency_s,
        condition=condition,
        meta={"strategy": "single_pass", "avg_logprob": generation.avg_logprob, **generation.meta},
    )


def run_ultra_fanout(
    model: Any,
    tokenizer: Any,
    condition: Condition,
    task: Task,
    *,
    n_candidates: int,
    single_token_multiplier: float,
    thinking_token_multiplier: float,
) -> dict[str, Any]:
    candidates: list[CandidateResult] = []
    for candidate_index in range(n_candidates):
        style = ultra_style(candidate_index)
        messages = [
            {"role": "system", "content": system_prompt(task, concise=False, style=style)},
            {"role": "user", "content": ultra_user_prompt(task, style=style)},
        ]
        generation = generate_once(
            model,
            tokenizer,
            messages,
            condition=condition,
            task=task,
            strategy="ultra_fanout",
            candidate_index=candidate_index,
            single_token_multiplier=single_token_multiplier,
            thinking_token_multiplier=thinking_token_multiplier,
        )
        candidates.append(
            evaluate_candidate(
                task,
                generation.raw_text,
                generation.generated_tokens,
                generation.latency_s,
                condition=condition,
                meta={
                    "strategy": "ultra_fanout",
                    "candidate_index": candidate_index,
                    "style": style,
                    "avg_logprob": generation.avg_logprob,
                    **generation.meta,
                },
            )
        )
    oracle_best = max(candidates, key=lambda candidate: (candidate.score, -candidate.latency_s))
    return {"candidates": candidates, "selected": select_candidate(task, candidates), "oracle_best": oracle_best}


def generate_once(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    condition: Condition,
    task: Task,
    strategy: str,
    candidate_index: int,
    single_token_multiplier: float,
    thinking_token_multiplier: float,
) -> GenerationResult:
    params = condition.params_for_task(task)
    params = varied_params(params, strategy=strategy, condition=condition, candidate_index=candidate_index)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=condition.thinking,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    base_max_new_tokens = task.max_new_tokens_thinking if condition.thinking else task.max_new_tokens_single
    multiplier = thinking_token_multiplier if condition.thinking else single_token_multiplier
    total_max_new_tokens = max(1, int(round(base_max_new_tokens * multiplier)))
    set_seed(seed_for(condition.label, task.task_id, strategy, candidate_index))
    generation_kwargs = {
        "do_sample": True,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "min_p": params["min_p"],
        "repetition_penalty": params["repetition_penalty"],
    }
    if condition.thinking:
        return generate_think_then_answer(
            model,
            tokenizer,
            inputs,
            task=task,
            condition=condition,
            generation_kwargs=generation_kwargs,
            total_max_new_tokens=total_max_new_tokens,
        )
    return generate_with_scores(
        model,
        tokenizer,
        inputs,
        max_new_tokens=total_max_new_tokens,
        generation_kwargs=generation_kwargs,
        extra_meta={"generation_stage_count": 1, "reasoning_budget": total_max_new_tokens, "answer_budget": 0},
    )


def generate_think_then_answer(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    *,
    task: Task,
    condition: Condition,
    generation_kwargs: dict[str, Any],
    total_max_new_tokens: int,
) -> GenerationResult:
    reasoning_budget, answer_budget = thinking_stage_budgets(task, total_max_new_tokens)
    stop_sequences = think_end_stop_sequences(tokenizer)
    stage_one = generate_with_scores(
        model,
        tokenizer,
        inputs,
        max_new_tokens=reasoning_budget,
        generation_kwargs=generation_kwargs,
        stop_sequences=stop_sequences,
        extra_meta={"generation_stage_count": 1, "reasoning_budget": reasoning_budget, "answer_budget": answer_budget},
    )
    parsed = split_reasoning_and_answer(stage_one.raw_text, family=condition.family, thinking=True)
    if parsed["answer_stage_reached"] or not parsed["think_closed"] or answer_budget <= 0:
        stage_one.meta.update(
            {
                "answer_stage_invoked": False,
                "stage_one_generated_tokens": stage_one.generated_tokens,
                "stage_two_generated_tokens": 0,
            }
        )
        return stage_one

    continuation_inputs = {
        "input_ids": stage_one.sequence_ids,
        "attention_mask": torch.ones_like(stage_one.sequence_ids),
    }
    stage_two = generate_with_scores(
        model,
        tokenizer,
        continuation_inputs,
        max_new_tokens=answer_budget,
        generation_kwargs={"do_sample": False},
        extra_meta={"generation_stage_count": 2, "reasoning_budget": reasoning_budget, "answer_budget": answer_budget},
    )
    total_generated_tokens = stage_one.generated_tokens + stage_two.generated_tokens
    total_latency = stage_one.latency_s + stage_two.latency_s
    if total_generated_tokens:
        avg_logprob = (
            (stage_one.avg_logprob * stage_one.generated_tokens) + (stage_two.avg_logprob * stage_two.generated_tokens)
        ) / total_generated_tokens
    else:
        avg_logprob = float("-inf")
    return GenerationResult(
        raw_text=stage_one.raw_text + stage_two.raw_text,
        generated_tokens=total_generated_tokens,
        latency_s=total_latency,
        avg_logprob=avg_logprob,
        sequence_ids=stage_two.sequence_ids,
        meta={
            "stop_sequence_count": stage_one.meta.get("stop_sequence_count", 0),
            "stopped_on_stop_sequence": stage_one.meta.get("stopped_on_stop_sequence", False),
            "generation_stage_count": 2,
            "reasoning_budget": reasoning_budget,
            "answer_budget": answer_budget,
            "answer_stage_invoked": True,
            "stage_one_generated_tokens": stage_one.generated_tokens,
            "stage_two_generated_tokens": stage_two.generated_tokens,
        },
    )


def generate_with_scores(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    generation_kwargs: dict[str, Any],
    stop_sequences: list[list[int]] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> GenerationResult:
    effective_generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
        "renormalize_logits": True,
        **generation_kwargs,
    }
    if stop_sequences:
        effective_generation_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])
    using_cuda = any(value.is_cuda for value in inputs.values() if isinstance(value, torch.Tensor))
    if using_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **effective_generation_kwargs)
    if using_cuda:
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - start
    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True,
    )[0]
    transition_scores = transition_scores[-generated_ids.shape[0] :] if generated_ids.shape[0] else transition_scores[:0]
    avg_logprob = float(transition_scores.mean().item()) if transition_scores.numel() else float("-inf")
    stop_sequence_matched = generated_ids_end_with_any(generated_ids, stop_sequences or [])
    return GenerationResult(
        raw_text=raw_text,
        generated_tokens=int(generated_ids.shape[0]),
        latency_s=latency_s,
        avg_logprob=avg_logprob,
        sequence_ids=outputs.sequences,
        meta={
            **(extra_meta or {}),
            "stop_sequence_count": len(stop_sequences or []),
            "stopped_on_stop_sequence": stop_sequence_matched,
        },
    )


def thinking_stage_budgets(task: Task, total_max_new_tokens: int) -> tuple[int, int]:
    if total_max_new_tokens <= 1:
        return 1, 0
    if task.category == "code":
        answer_budget = min(task.max_new_tokens_single, total_max_new_tokens - 1)
    else:
        answer_budget = min(48, max(16, total_max_new_tokens // 5), total_max_new_tokens - 1)
    reasoning_budget = max(1, total_max_new_tokens - answer_budget)
    return reasoning_budget, answer_budget


def think_end_stop_sequences(tokenizer: Any) -> list[list[int]]:
    sequences: list[list[int]] = []
    for variant in ("</think>", "\n</think>", "</think>\n", "\n</think>\n"):
        encoded = encode_text_for_stopping(tokenizer, variant)
        if encoded and encoded not in sequences:
            sequences.append(encoded)
    return sequences


def encode_text_for_stopping(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        payload = tokenizer(text, add_special_tokens=False)
        encoded = payload["input_ids"] if isinstance(payload, dict) else payload.input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def generated_ids_end_with_any(generated_ids: torch.Tensor, stop_sequences: list[list[int]]) -> bool:
    if generated_ids.ndim != 1 or generated_ids.numel() == 0:
        return False
    generated = generated_ids.tolist()
    for sequence in stop_sequences:
        if sequence and generated[-len(sequence) :] == sequence:
            return True
    return False


def varied_params(base: dict[str, Any], *, strategy: str, condition: Condition, candidate_index: int) -> dict[str, Any]:
    if strategy == "single_pass":
        return base

    temp_offsets = [-0.15, -0.05, 0.0, 0.05, 0.15]
    top_p_offsets = [-0.08, -0.03, 0.0, 0.02, 0.05]
    idx = candidate_index % len(temp_offsets)
    varied = dict(base)
    varied["temperature"] = clamp(base["temperature"] + temp_offsets[idx], 0.2, 1.2)
    varied["top_p"] = clamp(base["top_p"] + top_p_offsets[idx], 0.6, 1.0)
    if condition.family == "qwen3.5" and condition.thinking and base["temperature"] <= 0.6:
        varied["temperature"] = clamp(base["temperature"] + [0.0, 0.05, -0.05, 0.1, -0.1][idx], 0.35, 0.8)
    return varied


def select_candidate(task: Task, candidates: list[CandidateResult]) -> CandidateResult:
    if task.evaluator == "code":
        return max(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.meta.get("tests_passed", 0),
                candidate.meta.get("avg_logprob", float("-inf")),
                -candidate.latency_s,
            ),
        )

    normalized_values = [normalize_selection_value(candidate.canonical_answer) for candidate in candidates]
    counts = Counter(value for value in normalized_values if value)
    if not counts:
        shortlisted = candidates
    else:
        best_count = max(counts.values())
        winners = {value for value, count in counts.items() if count == best_count}
        shortlisted = [
            candidate
            for candidate, normalized in zip(candidates, normalized_values, strict=True)
            if normalized in winners
        ]

    return max(
        shortlisted,
        key=lambda candidate: (
            SOURCE_PRIORITY.get(str(candidate.meta.get("canonical_source", "none")), 0),
            int(candidate.meta.get("answer_stage_reached", False)),
            int(candidate.meta.get("think_closed", False)),
            candidate.meta.get("avg_logprob", float("-inf")),
            -candidate.latency_s,
        ),
    )


def evaluate_candidate(
    task: Task,
    raw_text: str,
    generated_tokens: int,
    latency_s: float,
    *,
    condition: Condition,
    meta: dict[str, Any],
) -> CandidateResult:
    parsed = split_reasoning_and_answer(raw_text, family=condition.family, thinking=condition.thinking)
    cleaned_text = parsed["answer_text"] or parsed["reasoning_text"]
    think_present = bool(parsed["reasoning_text"]) if condition.thinking else ("<think>" in raw_text or "</think>" in raw_text)

    if task.evaluator == "number":
        canonical, source = extract_canonical_answer(task, answer_text=parsed["answer_text"], fallback_text=cleaned_text)
        score = 1.0 if canonical == task.answer else 0.0
    elif task.evaluator == "fraction":
        canonical, source = extract_canonical_answer(task, answer_text=parsed["answer_text"], fallback_text=cleaned_text)
        score = 1.0 if equivalent_fraction_or_decimal(canonical, task.answer, task.aliases) else 0.0
    elif task.evaluator == "choice":
        canonical, source = extract_canonical_answer(task, answer_text=parsed["answer_text"], fallback_text=cleaned_text)
        score = 1.0 if canonical == task.answer else 0.0
    elif task.evaluator == "text":
        canonical, source = extract_canonical_answer(task, answer_text=parsed["answer_text"], fallback_text=cleaned_text)
        valid_answers = {normalize_text_answer(task.answer), *(normalize_text_answer(alias) for alias in task.aliases)}
        score = score_short_text(canonical, valid_answers)
    elif task.evaluator == "code":
        canonical = extract_code(cleaned_text or raw_text)
        code_eval = evaluate_code(canonical, task.answer)
        source = "code"
        meta = {**meta, **parsed, **code_eval["meta"]}
        score = code_eval["score"]
    else:
        canonical = cleaned_text
        source = "none"
        score = 0.0

    return CandidateResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        canonical_answer=canonical,
        score=score,
        latency_s=latency_s,
        generated_tokens=generated_tokens,
        think_block_present=think_present,
        meta={**meta, **parsed, "canonical_source": source},
    )


def strip_model_control_tokens(text: str) -> str:
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    return text.strip()


def split_reasoning_and_answer(raw_text: str, *, family: str, thinking: bool) -> dict[str, Any]:
    del family  # reserved for future family-specific parsing tweaks
    text = strip_model_control_tokens(raw_text)
    if not thinking:
        cleaned = text.replace("<think>\n\n</think>\n\n", "")
        cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
        return {
            "reasoning_text": "",
            "answer_text": cleaned,
            "normalized_text": cleaned,
            "think_closed": False,
            "truncated_inside_think": False,
            "answer_stage_reached": bool(cleaned),
        }

    if "</think>" in text:
        reasoning_text, answer_text = text.split("</think>", 1)
        reasoning_text = reasoning_text.replace("<think>", "").strip()
        answer_text = answer_text.strip()
        normalized_text = answer_text or reasoning_text
        return {
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
            "normalized_text": normalized_text,
            "think_closed": True,
            "truncated_inside_think": False,
            "answer_stage_reached": bool(answer_text),
        }

    reasoning_text = text.replace("<think>", "").strip()
    return {
        "reasoning_text": reasoning_text,
        "answer_text": "",
        "normalized_text": reasoning_text,
        "think_closed": False,
        "truncated_inside_think": bool(reasoning_text),
        "answer_stage_reached": False,
    }


def extract_canonical_answer(task: Task, *, answer_text: str, fallback_text: str) -> tuple[str | None, str]:
    texts = [answer_text, fallback_text]

    if task.evaluator in {"choice", "text"}:
        for text in texts:
            payload = extract_first_json_object(text)
            if isinstance(payload, dict) and "answer" in payload:
                return str(payload["answer"]).strip(), "json"

    if task.evaluator in {"number", "fraction"}:
        for text in texts:
            boxed = extract_boxed_answer(text)
            if boxed is not None:
                return boxed, "boxed"

    for text in texts:
        final = extract_final_field(text)
        if final is not None:
            return final, "final"

    if task.evaluator in {"number", "fraction"}:
        return extract_last_number(fallback_text), "fallback_last_number"
    if task.evaluator == "choice":
        if is_concise_answer_text(fallback_text):
            return extract_choice(fallback_text), "fallback_choice"
        return None, "none"
    if task.evaluator == "text":
        if is_concise_answer_text(fallback_text):
            return fallback_text.strip(), "fallback_text"
        return None, "none"
    return None, "none"


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    for match in re.finditer(r"\{", text):
        try:
            payload, end = JSON_DECODER.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        if end:
            continue
    return None


def extract_boxed_answer(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_final_field(text: str) -> str | None:
    match = re.search(r"FINAL\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip("`")
    return None


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return matches[-1] if matches else None


def extract_choice(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def normalize_text_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(".")


def score_short_text(candidate: str | None, valid_answers: set[str]) -> float:
    if not candidate:
        return 0.0
    normalized_candidate = normalize_text_answer(candidate)
    if normalized_candidate in valid_answers:
        return 1.0
    if not is_concise_answer_text(candidate):
        return 0.0
    for answer in valid_answers:
        if re.search(rf"\b{re.escape(answer)}\b", normalized_candidate):
            return 1.0
    return 0.0


def is_concise_answer_text(text: str, *, max_words: int = 8, max_chars: int = 80) -> bool:
    normalized = normalize_text_answer(text)
    if not normalized:
        return False
    word_count = len(re.findall(r"\w+", normalized))
    return word_count <= max_words and len(normalized) <= max_chars


def equivalent_fraction_or_decimal(candidate: str | None, answer: str, aliases: list[str]) -> bool:
    if candidate is None:
        return False
    values = [answer, *aliases]
    candidate_value = numeric_value(candidate)
    if candidate_value is None:
        return candidate in values
    for value in values:
        reference = numeric_value(value)
        if reference is not None and abs(candidate_value - reference) <= 1e-6:
            return True
    return False


def numeric_value(text: str) -> float | None:
    text = text.strip()
    try:
        if "/" in text and not text.startswith("http"):
            return float(Fraction(text))
        return float(Decimal(text))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def evaluate_code(code: str, answer: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    metadata = {"tests_total": len(answer["tests"]), "tests_passed": 0}
    try:
        with time_limit(3), redirect_stdout(io.StringIO()):
            exec(code, namespace, namespace)
        function_name = answer["function_name"]
        if function_name not in namespace:
            metadata["error"] = f"Missing function {function_name}"
            return {"score": 0.0, "meta": metadata}
        for test in answer["tests"]:
            with time_limit(3), redirect_stdout(io.StringIO()):
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
        "cleaned_text": candidate.cleaned_text,
        "canonical_answer": candidate.canonical_answer,
        "score": candidate.score,
        "latency_s": round(candidate.latency_s, 4),
        "generated_tokens": candidate.generated_tokens,
        "think_block_present": candidate.think_block_present,
        "meta": candidate.meta,
    }


def system_prompt(task: Task, *, concise: bool, style: str = "default") -> str:
    if task.category == "code":
        if concise:
            return "You are a careful Python coding assistant. Output only final Python code."
        if style == "plan":
            return "You are a meticulous Python coding assistant. Think through edge cases, then output only final Python code."
        if style == "tests":
            return "You are a Python coding assistant. Internally verify against likely edge cases, then output only final Python code."
        return "You are a strong Python coding assistant. Output only final Python code."
    output_instruction = response_format_instruction(task)
    if concise:
        return f"You are a precise reasoning assistant. Solve carefully. {output_instruction}"
    if style == "plan":
        return f"You are a precise reasoning assistant. Plan briefly, solve carefully. {output_instruction}"
    if style == "verify":
        return f"You are a precise reasoning assistant. Double-check the result before answering. {output_instruction}"
    return f"You are a precise reasoning assistant. Solve carefully. {output_instruction}"


def ultra_user_prompt(task: Task, *, style: str) -> str:
    prompt = task.prompt
    if task.category == "code":
        if style == "tests":
            return prompt + "\nMake sure the function handles the implied edge cases in the docstring."
        return prompt
    extras = [response_format_instruction(task)]
    if style == "plan":
        extras.append("Use a short plan before answering.")
    if style == "verify":
        extras.append("Check your work once before answering.")
    return prompt + "\n" + " ".join(extras)


def single_user_prompt(task: Task) -> str:
    if task.category == "code":
        return task.prompt
    return task.prompt + "\n" + response_format_instruction(task)


def response_format_instruction(task: Task) -> str:
    if task.evaluator in {"number", "fraction"}:
        return r"Put the final answer in \boxed{...}. Do not use JSON."
    if task.evaluator in {"choice", "text"}:
        return 'Return valid JSON only with the shape {"answer": "<final_answer>"}.'
    if task.evaluator == "code":
        return "Output only final Python code."
    return "Give the final answer directly."


def ultra_style(candidate_index: int) -> str:
    return ["default", "plan", "verify", "tests", "default"][candidate_index % 5]


def normalize_selection_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value).strip().lower()


def aggregate_report(condition_reports: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_by_strategy: dict[str, list[float]] = defaultdict(list)
    category_accuracy: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    timing_ratio: dict[str, list[float]] = defaultdict(list)

    for report in condition_reports:
        summary = report["summary"]
        for strategy, accuracy in summary["accuracy"].items():
            accuracy_by_strategy[strategy].append(accuracy)
        for category, strategies in summary["category_accuracy"].items():
            for strategy, accuracy in strategies.items():
                category_accuracy[category][strategy].append(accuracy)
        single_total = sum(task["single_pass"]["latency_s"] for task in report["tasks"])
        ultra_total = sum(
            candidate["latency_s"]
            for task in report["tasks"]
            for candidate in task["ultra_candidates"]
        )
        timing_ratio["ultra_vs_single_generation_time"].append(ultra_total / single_total if single_total else 0.0)

    return {
        "mean_accuracy": {
            strategy: sum(values) / len(values)
            for strategy, values in sorted(accuracy_by_strategy.items())
        },
        "mean_category_accuracy": {
            category: {
                strategy: sum(values) / len(values)
                for strategy, values in sorted(strategies.items())
            }
            for category, strategies in sorted(category_accuracy.items())
        },
        "mean_timing_ratio": {
            name: sum(values) / len(values)
            for name, values in sorted(timing_ratio.items())
        },
    }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def seed_for(*parts: Any) -> int:
    value = 0
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % (2**31 - 1)
    return value


class time_limit:
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
