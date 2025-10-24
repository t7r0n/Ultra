from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(slots=True)
class SandboxCommand:
    command: List[str]
    timeout: int = 60


@dataclass(slots=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str


class Sandbox:
    """A minimal sandbox abstraction that can be mocked in tests."""

    def __init__(self, profile: str) -> None:
        self.profile = profile

    def run(self, command: Iterable[str]) -> SandboxResult:
        raise NotImplementedError("Sandbox execution requires integration with docker or firejail")
