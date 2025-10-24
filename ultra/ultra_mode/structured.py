from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..mcd.inspector import InspectionResult


def collect_interactive_messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are ULTRA mode assistant."},
        {"role": "user", "content": "Hello"},
    ]


def enforce_structure(
    *,
    inspection: InspectionResult,
    profile: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    structured_cfg = profile.get("structured_output", {})
    if not structured_cfg.get("enabled"):
        return candidate
    schema = structured_cfg.get("json_schema")
    payload = {
        "content": candidate["content"],
        "schema": schema,
    }
    return payload
