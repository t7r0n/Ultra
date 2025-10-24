from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - yaml optional
    yaml = None  # type: ignore

from ..backends import SamplingParameters
from ..mcd.inspector import InspectionResult

DEFAULT_PROFILE = {
    "backend": "auto",
    "precision": "auto",
    "context_window": "auto",
    "ultra": {
        "n_candidates": 4,
        "max_n_candidates": 32,
        "diversity": {
            "temperatures": [0.2, 0.6, 0.9],
            "top_p": [0.85, 0.95],
            "styles": ["concise", "cot"],
        },
        "selection": {
            "self_consistency": True,
            "mbr": True,
            "logprobs": True,
        },
        "refine": {
            "self_refine_passes": 1,
            "chain_of_verification": True,
        },
    },
    "structured_output": {
        "enabled": False,
        "json_schema": None,
    },
}


@dataclass(slots=True)
class Candidate:
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {"content": self.content}
        payload.update(self.metadata)
        return payload


def load_ultra_profile(*, config_path: Optional[pathlib.Path], profile_name: str) -> Dict[str, Any]:
    if config_path is None:
        profile = json.loads(json.dumps(DEFAULT_PROFILE))
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse custom Ultra profiles")
        data = yaml.safe_load(config_path.read_text())
        profile = data
    profile.setdefault("ultra", {}).setdefault("profile", profile_name)
    return profile


def generate_candidates(
    inspection: InspectionResult,
    profile: Dict[str, Any],
    backend: str,
) -> List[Dict[str, Any]]:
    settings = profile.get("ultra", {})
    n_candidates = int(settings.get("n_candidates", 1))
    temps = settings.get("diversity", {}).get("temperatures", [0.7])
    top_ps = settings.get("diversity", {}).get("top_p", [0.95])
    candidates: List[Dict[str, Any]] = []
    for idx in range(n_candidates):
        temperature = temps[idx % len(temps)]
        top_p = top_ps[idx % len(top_ps)]
        sampling = SamplingParameters(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=512,
            seed=random.randint(0, 1_000_000),
        )
        content = (
            f"[{inspection.arch}] Candidate {idx+1} via {backend} with T={sampling.temperature} top_p={sampling.top_p}"
        )
        candidate = Candidate(
            content=content,
            metadata={
                "sampling": sampling.__dict__,
                "backend": backend,
            },
        )
        candidates.append(candidate.to_dict())
    return candidates
