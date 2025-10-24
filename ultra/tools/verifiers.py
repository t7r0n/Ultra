from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


@dataclass(slots=True)
class VerificationResult:
    name: str
    passed: bool
    details: Dict[str, object]


class Verifier:
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


def _non_empty(candidate: str) -> VerificationResult:
    stripped = candidate.strip()
    return VerificationResult(
        name="non_empty",
        passed=bool(stripped),
        details={"length": len(stripped)},
    )


def _json_verifier(candidate: str) -> VerificationResult:
    try:
        json.loads(candidate)
        return VerificationResult(name="json_parse", passed=True, details={})
    except Exception as exc:  # pragma: no cover - best effort
        return VerificationResult(name="json_parse", passed=False, details={"error": str(exc)})


def _run_pytest(candidate: str, *, workdir: Path, timeout: int = 30) -> VerificationResult:
    # Validate that workdir is an absolute path and is within a safe sandbox directory
    safe_sandbox_root = Path("/tmp")  # You may want to make this configurable
    try:
        workdir_abs = workdir.resolve()
        if not str(workdir_abs).startswith(str(safe_sandbox_root.resolve())):
            return VerificationResult(
                name="pytest",
                passed=False,
                details={"error": f"Unsafe workdir: {workdir_abs} is not within {safe_sandbox_root}"},
            )
    except Exception as exc:
        return VerificationResult(
            name="pytest",
            passed=False,
            details={"error": f"Failed to resolve workdir: {exc}"},
        )

    # Basic candidate content check (optional, can be extended)
    dangerous_keywords = ["os.system", "subprocess", "open(", "__import__", "eval(", "exec("]
    for keyword in dangerous_keywords:
        if keyword in candidate:
            return VerificationResult(
                name="pytest",
                passed=False,
                details={"error": f"Candidate contains dangerous keyword: {keyword}"},
            )

    test_file = workdir / "candidate_solution.py"
    test_file.write_text(candidate)
        result = subprocess.run(  # pragma: no cover - external process
            ["pytest", str(test_file)],
            cwd=str(workdir),
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - external process
        return VerificationResult(name="pytest", passed=False, details={"error": str(exc)})
    return VerificationResult(
        name="pytest",
        passed=result.returncode == 0,
        details={"stdout": result.stdout, "stderr": result.stderr},
    )


def build_verifier_suite(*, kind: str, sandbox_dir: Optional[Path] = None) -> VerifierSuite:
    verifiers: List[Verifier] = [Verifier("non_empty", _non_empty)]
    if kind == "json":
        verifiers.append(Verifier("json_parse", _json_verifier))
    if kind == "code" and sandbox_dir is not None:
        verifiers.append(Verifier("pytest", lambda candidate: _run_pytest(candidate, workdir=sandbox_dir)))
    return VerifierSuite(verifiers)

