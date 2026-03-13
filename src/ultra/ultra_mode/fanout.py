"""Fan-out planning and prompt diversification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..config import GlobalConfig, UltraProfileName


@dataclass(slots=True)
class CandidatePlan:
    index: int
    style: str
    temperature: float
    top_p: float
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    system_prompt: str
    user_prompt: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FanoutPlan:
    ultra_profile: UltraProfileName
    n_candidates: int
    temperatures: list[float]
    top_p: list[float]
    styles: list[str]
    candidates: list[CandidatePlan]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ultra_profile": self.ultra_profile,
            "n_candidates": self.n_candidates,
            "temperatures": self.temperatures,
            "top_p": self.top_p,
            "styles": self.styles,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def build_fanout_plan(
    config: GlobalConfig,
    ultra_profile: UltraProfileName,
    *,
    prompt: str,
    structured_output: bool = False,
    seed_base: int = 0,
) -> FanoutPlan:
    n_candidates = adaptive_candidate_count(config, ultra_profile=ultra_profile, prompt=prompt)
    temperatures = list(config.ultra.diversity.temperatures)
    top_p_values = list(config.ultra.diversity.top_p)
    styles = candidate_styles(config, ultra_profile=ultra_profile)
    candidates: list[CandidatePlan] = []
    for index in range(n_candidates):
        style = styles[index % len(styles)]
        temperature = temperatures[index % len(temperatures)]
        top_p = top_p_values[index % len(top_p_values)]
        candidate_prompt = style_prompt(prompt, style=style, ultra_profile=ultra_profile, structured_output=structured_output)
        candidates.append(
            CandidatePlan(
                index=index,
                style=style,
                temperature=temperature,
                top_p=top_p,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.0,
                system_prompt=system_prompt_for_profile(ultra_profile, style=style, structured_output=structured_output),
                user_prompt=candidate_prompt,
                seed=seed_base + index,
            )
        )
    return FanoutPlan(
        ultra_profile=ultra_profile,
        n_candidates=n_candidates,
        temperatures=temperatures,
        top_p=top_p_values,
        styles=styles,
        candidates=candidates,
    )


def adaptive_candidate_count(config: GlobalConfig, *, ultra_profile: UltraProfileName, prompt: str) -> int:
    base = config.ultra.n_candidates
    max_candidates = config.ultra.max_n_candidates
    complexity = prompt_complexity_score(prompt, ultra_profile=ultra_profile)
    if ultra_profile == "creative":
        target = max(base, 10 + complexity)
    elif ultra_profile in {"reasoning", "coding"}:
        target = max(base, 8 + complexity)
    else:
        target = max(base, 6 + complexity)
    return min(max_candidates, target)


def candidate_styles(config: GlobalConfig, *, ultra_profile: UltraProfileName) -> list[str]:
    styles = dedupe_styles(list(config.ultra.diversity.styles))
    if ultra_profile == "reasoning":
        return dedupe_styles(["concise", "cot", "verify", "plan", "formal", "debate", *styles])
    if ultra_profile == "coding":
        return dedupe_styles(["concise", "tests", "edge_cases", "repair", *styles])
    if ultra_profile == "creative":
        return dedupe_styles(["concise", "expansive", "voice_shift", "formal", *styles])
    return dedupe_styles(["concise", "plan", "citation", *styles])


def style_prompt(prompt: str, *, style: str, ultra_profile: UltraProfileName, structured_output: bool) -> str:
    instructions: list[str] = []
    if style == "cot":
        instructions.append("Think step by step, but keep the final answer clearly separated.")
    elif style == "verify":
        instructions.append("Double-check the answer before responding.")
    elif style == "plan":
        instructions.append("Create a short plan, then answer.")
    elif style == "formal":
        instructions.append("Use a careful, audit-friendly structure with an explicit final answer.")
    elif style == "debate":
        instructions.append("Consider at least two plausible answers briefly before deciding.")
    elif style == "citation":
        instructions.append("Support concrete claims with citations or source markers when possible.")
    elif style == "tests":
        instructions.append("Check likely edge cases before producing the final code.")
    elif style == "edge_cases":
        instructions.append("Handle corner cases explicitly.")
    elif style == "repair":
        instructions.append("Inspect the likely failure modes, then provide the corrected solution directly.")
    elif style == "expansive":
        instructions.append("Lean into stronger detail and richer language.")
    elif style == "voice_shift":
        instructions.append("Vary voice and rhythm, but keep the answer coherent.")
    if ultra_profile == "coding":
        instructions.append("Return only final Python code.")
    if structured_output:
        instructions.append("The final answer must satisfy the requested structured output format.")
    return prompt if not instructions else prompt + "\n" + " ".join(instructions)


def system_prompt_for_profile(ultra_profile: UltraProfileName, *, style: str, structured_output: bool) -> str:
    if ultra_profile == "coding":
        base = "You are a meticulous Python coding assistant."
    elif ultra_profile == "creative":
        base = "You are a deliberate creative writing assistant."
    else:
        base = "You are a precise reasoning assistant."
    if style == "verify":
        base += " Verify your answer before finalizing it."
    if style == "plan":
        base += " Plan briefly before answering."
    if structured_output:
        base += " Obey the requested output format exactly."
    return base


def prompt_complexity_score(prompt: str, *, ultra_profile: UltraProfileName) -> int:
    prompt_words = len(prompt.split())
    score = 0
    if prompt_words >= 64:
        score += 2
    if prompt_words >= 160:
        score += 3
    if "```" in prompt:
        score += 2
    if any(token in prompt for token in ("=", "+", "-", "/", "%", "\\boxed")):
        score += 2 if ultra_profile == "reasoning" else 1
    lowered = prompt.lower()
    if any(marker in lowered for marker in ("cite", "source", "reference", "url")):
        score += 1
    if ultra_profile == "coding" and any(marker in lowered for marker in ("test", "bug", "patch", "refactor")):
        score += 2
    return score


def dedupe_styles(styles: list[str]) -> list[str]:
    ordered: list[str] = []
    for style in styles:
        if style not in ordered:
            ordered.append(style)
    return ordered
