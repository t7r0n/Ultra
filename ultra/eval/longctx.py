"""Long-context benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class LongContextResult:
    benchmark: str
    score: float


def run_long_context(tasks: Iterable[str]) -> List[LongContextResult]:
    return [LongContextResult(benchmark=task, score=0.0) for task in tasks]
