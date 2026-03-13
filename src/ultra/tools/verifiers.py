"""Verifiers used by Ultra-mode selection and agent loops."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import io
import json
import re
from typing import Any


@dataclass(slots=True)
class VerificationResult:
    name: str
    passed: bool
    score: float = 0.0
    details: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_math(expected: str, candidate: str) -> VerificationResult:
    candidate_value = numeric_value(extract_last_number(candidate) or candidate)
    expected_value = numeric_value(expected)
    passed = candidate_value is not None and expected_value is not None and abs(candidate_value - expected_value) <= 1e-6
    return VerificationResult(
        name="math",
        passed=passed,
        score=1.0 if passed else 0.0,
        details="numeric comparison" if passed else f"expected {expected}, candidate {candidate!r}",
    )


def verify_math_response(candidate: str, *, prompt: str) -> VerificationResult:
    if not prompt_looks_mathematical(prompt):
        return VerificationResult(name="math_format", passed=True, score=0.0, details="prompt not classified as mathematical")
    boxed = extract_boxed_answer(candidate)
    final = extract_final_answer(candidate)
    number = extract_last_number(candidate)
    if boxed is not None and numeric_value(boxed) is not None:
        return VerificationResult(name="math_format", passed=True, score=1.0, details="boxed numeric answer detected")
    if final is not None and numeric_value(final) is not None:
        return VerificationResult(name="math_format", passed=True, score=0.85, details="FINAL numeric answer detected")
    if number is not None:
        return VerificationResult(name="math_format", passed=True, score=0.5, details="numeric answer detected without explicit final formatting")
    return VerificationResult(name="math_format", passed=False, score=0.0, details="no numeric answer detected for a mathematical prompt")


def verify_structured_json(candidate: str, *, required_keys: list[str]) -> VerificationResult:
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return VerificationResult(name="structured_json", passed=False, score=0.0, details=str(exc))
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return VerificationResult(
            name="structured_json",
            passed=False,
            score=0.0,
            details=f"missing keys: {', '.join(missing)}",
            metadata={"payload": payload},
        )
    return VerificationResult(name="structured_json", passed=True, score=1.0, metadata={"payload": payload})


def verify_code(code: str, *, function_name: str | None = None, tests: list[str] | None = None) -> VerificationResult:
    namespace: dict[str, Any] = {}
    passed = 0
    tests = tests or []
    try:
        with redirect_stdout(io.StringIO()):
            exec(code, namespace, namespace)
        if function_name and function_name not in namespace:
            return VerificationResult(name="code", passed=False, score=0.0, details=f"missing function {function_name}")
        for test in tests:
            with redirect_stdout(io.StringIO()):
                exec(test, namespace, namespace)
            passed += 1
        total = max(len(tests), 1)
        return VerificationResult(
            name="code",
            passed=passed == len(tests),
            score=passed / total,
            details=f"passed {passed}/{len(tests)} tests",
            metadata={"tests_total": len(tests), "tests_passed": passed},
        )
    except Exception as exc:  # noqa: BLE001
        total = max(len(tests), 1)
        return VerificationResult(
            name="code",
            passed=False,
            score=passed / total,
            details=f"{type(exc).__name__}: {exc}",
            metadata={"tests_total": len(tests), "tests_passed": passed},
        )


def verify_text_contains(candidate: str, expected_variants: list[str]) -> VerificationResult:
    normalized_candidate = normalize_text(candidate)
    normalized_variants = [normalize_text(item) for item in expected_variants]
    passed = any(variant and re.search(rf"\b{re.escape(variant)}\b", normalized_candidate) for variant in normalized_variants)
    return VerificationResult(
        name="text_match",
        passed=passed,
        score=1.0 if passed else 0.0,
        details="substring match" if passed else f"expected one of {expected_variants!r}",
    )


def verify_prompt_alignment(candidate: str, *, prompt: str) -> VerificationResult:
    prompt_terms = content_terms(prompt)
    candidate_terms = content_terms(candidate)
    if not candidate_terms:
        return VerificationResult(name="prompt_alignment", passed=False, score=0.0, details="empty candidate response")
    if not prompt_terms:
        return VerificationResult(name="prompt_alignment", passed=True, score=0.0, details="no content terms extracted from prompt")
    overlap = len(prompt_terms & candidate_terms) / max(len(prompt_terms), 1)
    passed = overlap >= 0.1 or len(candidate_terms) >= 4
    return VerificationResult(
        name="prompt_alignment",
        passed=passed,
        score=min(1.0, overlap * 2.0),
        details=f"content-term overlap={overlap:.3f}",
        metadata={"overlap_terms": sorted(prompt_terms & candidate_terms)},
    )


def verify_citation_style(candidate: str, *, prompt: str) -> VerificationResult:
    if not prompt_requests_citations(prompt):
        return VerificationResult(name="citation_style", passed=True, score=0.0, details="prompt did not request citations")
    if contains_citation_markers(candidate):
        return VerificationResult(name="citation_style", passed=True, score=1.0, details="citation markers detected")
    return VerificationResult(name="citation_style", passed=False, score=0.0, details="citations were requested but none were detected")


def verify_no_thought_leak(candidate: str) -> VerificationResult:
    leaked = "<think>" in candidate.lower() or "</think>" in candidate.lower()
    return VerificationResult(
        name="surface_cleanliness",
        passed=not leaked,
        score=0.0 if leaked else 1.0,
        details="no hidden-thought markers detected" if not leaked else "hidden-thought markers leaked into final output",
    )


def numeric_value(text: str) -> float | None:
    text = text.strip()
    try:
        if "/" in text and not text.startswith("http"):
            return float(Fraction(text))
        return float(Decimal(text))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return matches[-1] if matches else None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(".")


def extract_final_answer(text: str) -> str | None:
    match = re.search(r"FINAL\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_boxed_answer(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    return match.group(1).strip() if match else None


def prompt_looks_mathematical(prompt: str) -> bool:
    lowered = prompt.lower()
    math_markers = ("solve", "calculate", "equation", "probability", "integer", "fraction", "sum", "total", "how many")
    symbol_markers = any(symbol in prompt for symbol in ("=", "+", "-", "*", "/", "%"))
    digit_markers = sum(character.isdigit() for character in prompt) >= 2
    return any(marker in lowered for marker in math_markers) or (digit_markers and symbol_markers)


def prompt_requests_citations(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(marker in lowered for marker in ("cite", "citation", "source", "sources", "references", "url"))


def contains_citation_markers(text: str) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in (
            r"\[\d+\]",
            r"\(source",
            r"https?://",
            r"\bsource:\b",
            r"\breferences?:\b",
        )
    )


def content_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    terms = {
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        if token not in stopwords
    }
    return terms
