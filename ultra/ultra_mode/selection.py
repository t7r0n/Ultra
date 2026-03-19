from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..mcd.inspector import InspectionResult
from ..tools.verifiers import VerificationResult, VerifierSuite, build_verifier_suite
from .fanout import Candidate, UltraProfile


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _compute_mbr_scores(candidates: List[Candidate]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    normalized = {candidate.index: _normalize_text(candidate.completion.text) for candidate in candidates}
    for candidate in candidates:
        score = 0.0
        for other in candidates:
            if candidate.index == other.index:
                continue
            score += _jaccard_similarity(normalized[candidate.index], normalized[other.index])
        scores[candidate.index] = score
    return scores


@dataclass(slots=True)
class SelectionResult:
    candidate: Candidate
    votes: Dict[int, int]
    mbr_scores: Dict[int, float]
    verifier_results: Dict[int, List[VerificationResult]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.candidate.to_dict(),
            "votes": self.votes,
            "mbr_scores": self.mbr_scores,
            "verifier_results": {
                str(index): [result.__dict__ for result in results]
                for index, results in self.verifier_results.items()
            },
        }


def _build_suite(profile: UltraProfile, sandbox_dir: Optional[str]) -> VerifierSuite:
    if profile.structured_output.enabled:
        kind = "json"
    elif profile.name.lower() == "coding":
        kind = "code"
    else:
        kind = "text"
    path = None if sandbox_dir is None else Path(sandbox_dir)
    return build_verifier_suite(kind=kind, sandbox_dir=path)


def select_candidate(
    *,
    candidates: List[Candidate],
    profile: UltraProfile,
    inspection: InspectionResult,
    sandbox_dir: Optional[str] = None,
) -> SelectionResult:
    if not candidates:
        raise ValueError("No candidates generated for selection")
    normalized = {candidate.index: _normalize_text(candidate.completion.text) for candidate in candidates}
    vote_counts = Counter(normalized.values())
    votes: Dict[int, int] = {
        candidate.index: vote_counts[normalized[candidate.index]] for candidate in candidates
    }
    mbr_scores: Dict[int, float] = {}
    if profile.ultra.selection.mbr:
        mbr_scores = _compute_mbr_scores(candidates)

    def _score(candidate: Candidate) -> tuple:
        vote = votes.get(candidate.index, 0) if profile.ultra.selection.self_consistency else 0
        mbr = mbr_scores.get(candidate.index, 0.0)
        return (-vote, -mbr, candidate.index)

    winner = min(candidates, key=_score)
    suite = _build_suite(profile, sandbox_dir)
    verifier_results: Dict[int, List[VerificationResult]] = {}
    for candidate in candidates:
        if profile.ultra.selection.logprobs:
            verifier_results[candidate.index] = suite.run(candidate.completion.text)
        else:
            verifier_results[candidate.index] = []
    return SelectionResult(candidate=winner, votes=votes, mbr_scores=mbr_scores, verifier_results=verifier_results)


@dataclass(slots=True)
class EvaluationSummary:
    suite: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"suite": self.suite, "metrics": self.metrics}


def build_evaluation_runner(suite: str) -> Callable[..., Dict[str, Any]]:
    def runner(*, inspection: InspectionResult, tasks: Optional[Iterable[str]], backend: str) -> Dict[str, Any]:
        task_list = list(tasks) if tasks else []
        return {
            "suite": suite,
            "backend": backend,
            "model": inspection.arch,
            "tasks": task_list,
            "status": "scheduled",
        }

    return runner


def bootstrap_agentic_session(
    *,
    inspection: InspectionResult,
    profile: UltraProfile,
    sandbox: Dict[str, Any],
) -> str:
    return (
        f"Agent session initialized for {inspection.arch} "
        f"with sandbox={sandbox['sandbox']} open_terminal={sandbox['open_terminal']}"
    )

