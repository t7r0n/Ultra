from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List


@dataclass(slots=True)
class VerificationResult:
    name: str
    passed: bool
    details: Dict[str, object]


class Verifier:
    """Callable wrapper for verifier hooks."""

    def __init__(self, name: str, func: Callable[[str], VerificationResult]) -> None:
        self.name = name
        self._func = func

    def __call__(self, candidate: str) -> VerificationResult:
        return self._func(candidate)


class VerifierSuite:
    def __init__(self, verifiers: Iterable[Verifier]) -> None:
        self._verifiers = list(verifiers)

    def run(self, candidate: str) -> List[VerificationResult]:
        return [verifier(candidate) for verifier in self._verifiers]
