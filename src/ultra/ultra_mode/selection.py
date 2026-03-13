"""Selection helpers for ULTRA candidate ranking."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
import re
from typing import Any

from ..config import GlobalConfig


@dataclass(slots=True)
class CandidateEvaluation:
    index: int
    text: str
    canonical_answer: str | None
    verifier_score: float
    judge_score: float
    passed_verifiers: int
    self_consistency_votes: int
    judge_votes: float
    mbr_score: float
    avg_logprob: float | None
    latency_s: float
    generated_tokens: int
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectionPlan:
    self_consistency: bool
    mbr: bool
    logprobs: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectionResult:
    selected_index: int
    scores: list[CandidateEvaluation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "scores": [score.to_dict() for score in self.scores],
        }


def build_selection_plan(config: GlobalConfig) -> SelectionPlan:
    return SelectionPlan(
        self_consistency=config.ultra.selection.self_consistency,
        mbr=config.ultra.selection.mbr,
        logprobs=config.ultra.selection.logprobs,
    )


def select_candidates(
    texts: list[str],
    *,
    verifier_scores: list[float],
    judge_scores: list[float],
    passed_verifiers: list[int],
    avg_logprobs: list[float | None],
    latencies_s: list[float],
    generated_tokens: list[int],
    candidate_metrics: list[dict[str, Any]],
    task_type: str,
    structured_output: bool,
    selection_plan: SelectionPlan,
) -> SelectionResult:
    canonical_answers = [canonical_answer(text, task_type=task_type, structured_output=structured_output) for text in texts]
    votes = self_consistency_votes(canonical_answers) if selection_plan.self_consistency else {}
    mbr_scores = mbr_scores_for_answers(canonical_answers) if selection_plan.mbr else [0.0 for _ in texts]

    evaluations: list[CandidateEvaluation] = []
    for index, text in enumerate(texts):
        evaluations.append(
            CandidateEvaluation(
                index=index,
                text=text,
                canonical_answer=canonical_answers[index],
                verifier_score=verifier_scores[index],
                judge_score=judge_scores[index],
                passed_verifiers=passed_verifiers[index],
                self_consistency_votes=votes.get(normalize_selection_value(canonical_answers[index]), 0),
                judge_votes=float(candidate_metrics[index].get("judge_votes", 0.0) or 0.0),
                mbr_score=mbr_scores[index],
                avg_logprob=avg_logprobs[index],
                latency_s=latencies_s[index],
                generated_tokens=generated_tokens[index],
                metrics=candidate_metrics[index],
            )
        )

    selected = max(
        evaluations,
        key=ranking_tuple,
    )
    return SelectionResult(selected_index=selected.index, scores=evaluations)


def ranking_tuple(item: CandidateEvaluation) -> tuple[float, float, int, int, int, float, float, float]:
    return (
        item.verifier_score,
        item.judge_score,
        item.passed_verifiers,
        int(item.judge_votes * 1000),
        item.self_consistency_votes,
        item.mbr_score,
        item.avg_logprob if item.avg_logprob is not None else float("-inf"),
        -item.latency_s,
    )


def canonical_answer(text: str, *, task_type: str, structured_output: bool) -> str | None:
    if structured_output:
        payload = extract_json_object(text)
        if isinstance(payload, dict):
            if "answer" in payload:
                return normalize_selection_value(payload["answer"])
            return normalize_selection_value(payload)
    if task_type == "code":
        return text.strip()
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return normalize_selection_value(boxed)
    final = extract_final_answer(text)
    if final is not None:
        return normalize_selection_value(final)
    return normalize_selection_value(text) or None


def self_consistency_votes(values: list[str | None]) -> dict[str, int]:
    counts = Counter(normalize_selection_value(value) for value in values if normalize_selection_value(value))
    return dict(counts)


def mbr_scores_for_answers(values: list[str | None]) -> list[float]:
    normalized = [normalize_selection_value(value) for value in values]
    scores: list[float] = []
    for value in normalized:
        if not value:
            scores.append(0.0)
            continue
        utility = 0.0
        for other in normalized:
            if not other:
                continue
            utility += SequenceMatcher(None, value, other).ratio()
        scores.append(utility / max(len(normalized), 1))
    return scores


def extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def extract_boxed_answer(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    return match.group(1).strip() if match else None


def extract_final_answer(text: str) -> str | None:
    match = re.search(r"FINAL\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def normalize_selection_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return re.sub(r"\s+", " ", str(value).strip().lower())
