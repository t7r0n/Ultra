"""Arena-Hard-Auto integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class ArenaMatch:
    prompt_id: str
    score: float


def run_arena(tasks: Iterable[str]) -> List[ArenaMatch]:
    return [ArenaMatch(prompt_id=task, score=0.0) for task in tasks]
