"""Structured output helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from ..config import GlobalConfig


@dataclass(slots=True)
class StructuredOutputPlan:
    enabled: bool
    json_schema: dict[str, Any] | None
    regex: str | None
    grammar: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_structured_output_plan(config: GlobalConfig) -> StructuredOutputPlan:
    return StructuredOutputPlan(
        enabled=config.structured_output.enabled,
        json_schema=config.structured_output.json_schema,
        regex=config.structured_output.regex,
        grammar=config.structured_output.grammar,
    )


def validate_structured_output(text: str, plan: StructuredOutputPlan) -> tuple[bool, Any]:
    if not plan.enabled:
        return True, text
    if plan.json_schema is None:
        if plan.regex is not None:
            if re.fullmatch(plan.regex, text.strip(), flags=re.DOTALL):
                return True, text
            return False, "output does not match structured regex"
        if plan.grammar is not None:
            return (bool(text.strip()), text if text.strip() else "grammar-constrained output was empty")
        return False, "structured output enabled but no json_schema, regex, or grammar was configured"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return False, "structured payload must be a JSON object"
    required = plan.json_schema.get("required", [])
    missing = [field for field in required if field not in payload]
    if missing:
        return False, f"missing required fields: {', '.join(missing)}"
    return True, payload
