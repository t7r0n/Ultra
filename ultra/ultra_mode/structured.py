from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from ..mcd.inspector import InspectionResult
from .fanout import Candidate, UltraProfile


def collect_interactive_messages(initial: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    if initial is not None:
        return [dict(message) for message in initial]
    messages: List[Dict[str, str]] = []
    system_prompt = "You are ULTRA mode assistant maximizing answer quality."
    messages.append({"role": "system", "content": system_prompt})
    if sys.stdin.isatty():  # pragma: no cover - interactive
        try:
            user_prompt = input("User prompt: ")
        except EOFError:
            user_prompt = "Hello"
    else:
        user_prompt = "Hello"
    messages.append({"role": "user", "content": user_prompt})
    return messages


def enforce_structure(
    *,
    inspection: InspectionResult,
    profile: UltraProfile,
    candidate: Candidate,
) -> Dict[str, Any]:
    if not profile.structured_output.enabled:
        return {"content": candidate.completion.text}
    schema = profile.structured_output.json_schema
    text = candidate.completion.text
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"raw": text}
    return {"content": parsed, "schema": schema}

