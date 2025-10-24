from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..backends import SamplingParameters, registry
from ..mcd.inspector import InspectionResult
from .fanout import Candidate, UltraProfile


@dataclass(slots=True)
class RefinedCandidate:
    content: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    verification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "history": self.history,
            "verification": self.verification,
            "metadata": self.metadata,
        }


def _self_refine(
    *,
    candidate: Candidate,
    inspection: InspectionResult,
    profile: UltraProfile,
) -> RefinedCandidate:
    passes = profile.ultra.refine.self_refine_passes
    if passes <= 0:
        return RefinedCandidate(content=candidate.completion.text, metadata={"passes": 0})

    engine = registry.get(candidate.engine)
    content = candidate.completion.text
    history: List[Dict[str, Any]] = []
    prompt = candidate.prompt
    for iteration in range(1, passes + 1):
        refinement_prompt = (
            f"{prompt}\n\n[Self-Refine] Review the previous answer and produce an improved final response.\n"
            f"Previous answer:\n{content}\nImproved answer:"
        )
        sampling = SamplingParameters(
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=profile.ultra.max_new_tokens,
            seed=candidate.sampling.seed,
        )
        completion = engine.generate(
            prompt=refinement_prompt,
            sampling=sampling,
            inspection=inspection,
            structured=None,
        )
        refined_text = completion.text.strip()
        if not refined_text:
            refined_text = content
        history.append({"iteration": iteration, "prompt": refinement_prompt, "response": refined_text})
        content = refined_text
    return RefinedCandidate(content=content, history=history, metadata={"passes": passes})


def _chain_of_verification(
    *,
    refined: RefinedCandidate,
    candidate: Candidate,
    inspection: InspectionResult,
    profile: UltraProfile,
) -> Optional[str]:
    if not profile.ultra.refine.chain_of_verification:
        return None
    engine = registry.get(candidate.engine)
    verification_prompt = (
        f"{candidate.prompt}\n\n[Verify] Provide a numbered checklist validating each critical step in the answer:"
        f"\nAnswer:\n{refined.content}\nChecklist:"
    )
    sampling = SamplingParameters(
        temperature=0.1,
        top_p=0.8,
        max_new_tokens=256,
        seed=(candidate.sampling.seed or 0) + 7,
    )
    completion = engine.generate(
        prompt=verification_prompt,
        sampling=sampling,
        inspection=inspection,
        structured=None,
    )
    return completion.text.strip()


def refine_candidate(
    candidate: Candidate,
    inspection: InspectionResult,
    profile: UltraProfile,
) -> RefinedCandidate:
    refined = _self_refine(candidate=candidate, inspection=inspection, profile=profile)
    verification = _chain_of_verification(
        refined=refined,
        candidate=candidate,
        inspection=inspection,
        profile=profile,
    )
    refined.verification = verification
    refined.metadata.setdefault("chain_of_verification", bool(verification))
    return refined

