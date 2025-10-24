from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..tools.verifiers import VerifierSuite


@dataclass(slots=True)
class SelectionResult:
    best_candidate: Any
    votes: Dict[str, int]
    verifier_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_candidate": self.best_candidate,
            "votes": self.votes,
            "verifier_results": self.verifier_results,
        }


def select_candidate(
    candidates: List[Dict[str, Any]],
    profile: Dict[str, Any],
    inspection: Any,
) -> SelectionResult:
    vote_counter: Dict[str, int] = {}
    verifier_suite = VerifierSuite([])
    for candidate in candidates:
        key = candidate["content"]
        vote_counter[key] = vote_counter.get(key, 0) + 1
    best_key = max(vote_counter, key=vote_counter.get)
    best_candidate = next(candidate for candidate in candidates if candidate["content"] == best_key)
    verifier_results = [result.__dict__ for result in verifier_suite.run(best_candidate["content"])]
    return SelectionResult(best_candidate=best_candidate, votes=vote_counter, verifier_results=verifier_results)


@dataclass(slots=True)
class EvaluationSummary:
    suite: str
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return self.metrics


def build_evaluation_runner(suite: str) -> Callable[..., Dict[str, Any]]:
    def runner(*, inspection: Any, tasks: Optional[Iterable[str]], backend: str) -> Dict[str, Any]:
        tasks_list = list(tasks) if tasks else ["default"]
        metrics = {task: 0.0 for task in tasks_list}
        return {
            "suite": suite,
            "backend": backend,
            "tasks": tasks_list,
            "metrics": metrics,
            "model": inspection.arch,
        }

    return runner


def bootstrap_agentic_session(*, inspection: Any, profile: Dict[str, Any], sandbox: Dict[str, Any]) -> str:
    return (
        f"Agent session for {inspection.arch} using sandbox={sandbox['sandbox']} "
        f"open_terminal={sandbox['open_terminal']}"
    )
