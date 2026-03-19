"""Refinement helpers for ULTRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from ..config import GlobalConfig
from ..backends.common import GenerationRequest, StructuredOutputRequest


@dataclass(slots=True)
class RefinePlan:
    self_refine_passes: int
    chain_of_verification: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_refine_plan(config: GlobalConfig) -> RefinePlan:
    return RefinePlan(
        self_refine_passes=config.ultra.refine.self_refine_passes,
        chain_of_verification=config.ultra.refine.chain_of_verification,
    )


def refine_response(
    *,
    generate_fn: Callable[[GenerationRequest], Any],
    model_ref: str,
    original_messages: list[dict[str, str]],
    candidate_texts: list[str],
    refine_plan: RefinePlan,
    structured_output: StructuredOutputRequest,
    seed: int | None,
    precision: str,
) -> Any | None:
    if refine_plan.self_refine_passes <= 0 or not candidate_texts:
        return None
    best_candidates = candidate_texts[: min(5, len(candidate_texts))]
    current_messages = list(original_messages)
    latest_result: Any | None = None
    current_seed = seed
    for iteration in range(refine_plan.self_refine_passes):
        critique_prompt = build_refine_prompt(
            best_candidates,
            chain_of_verification=False,
            iteration=iteration,
        )
        current_messages = [
            *current_messages,
            {"role": "user", "content": critique_prompt},
        ]
        latest_result = generate_fn(
            GenerationRequest(
                model_ref=model_ref,
                messages=current_messages,
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
                seed=current_seed,
                precision=precision,  # type: ignore[arg-type]
                structured_output=structured_output,
                request_logprobs=True,
            )
        )
        best_candidates = [latest_result.text, *best_candidates[:2]]
        current_messages.append({"role": "assistant", "content": latest_result.text})
        current_seed = None if current_seed is None else current_seed + 1
    if refine_plan.chain_of_verification and latest_result is not None:
        verification_messages = [
            *current_messages,
            {"role": "user", "content": build_cov_prompt(latest_result.text)},
        ]
        latest_result = generate_fn(
            GenerationRequest(
                model_ref=model_ref,
                messages=verification_messages,
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
                seed=current_seed,
                precision=precision,  # type: ignore[arg-type]
                structured_output=structured_output,
                request_logprobs=True,
            )
        )
    return latest_result


def build_refine_prompt(candidate_texts: list[str], *, chain_of_verification: bool, iteration: int = 0) -> str:
    numbered = "\n\n".join(f"Candidate {index + 1}:\n{text}" for index, text in enumerate(candidate_texts))
    if chain_of_verification:
        return (
            "Critique these candidate answers, resolve disagreements, verify the strongest claims, "
            "and return the single best final answer.\n\n"
            f"{numbered}"
        )
    if iteration > 0:
        return (
            "Improve the current best answer again. Remove unsupported claims, tighten reasoning, and keep only the strongest final answer.\n\n"
            f"{numbered}"
        )
    return "Merge these candidate answers into the single best final answer.\n\n" + numbered


def build_cov_prompt(answer: str) -> str:
    return (
        "Review the draft answer below. Check the strongest claims, remove weak or unsupported claims, "
        "and return the corrected final answer only.\n\n"
        f"{answer}"
    )
