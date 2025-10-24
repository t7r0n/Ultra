"""Code evaluation adapters (EvalPlus, LiveCodeBench, SWE-bench)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class CodeEvalResult:
    task: str
    pass_k: float


def run_code(tasks: Iterable[str]) -> List[CodeEvalResult]:
    return [CodeEvalResult(task=task, pass_k=0.0) for task in tasks]
