"""RAG evaluation via Ragas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class RagMetric:
    dataset: str
    metrics: Dict[str, float]


def run_rag(tasks: Iterable[str]) -> List[RagMetric]:
    return [RagMetric(dataset=task, metrics={"faithfulness": 0.0, "answer_relevancy": 0.0}) for task in tasks]
