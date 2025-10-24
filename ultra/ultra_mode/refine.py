from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class RefinedCandidate:
    content: Any
    metadata: Dict[str, Any]


def refine_candidate(candidate: Dict[str, Any], inspection: Any, profile: Dict[str, Any]) -> RefinedCandidate:
    passes = profile.get("ultra", {}).get("refine", {}).get("self_refine_passes", 0)
    refined_content = candidate["content"]
    if passes:
        refined_content = refined_content + " [refined]"
    return RefinedCandidate(content=refined_content, metadata={"passes": passes})
