from __future__ import annotations

from ultra.mcd.inspector import inspect_model
from ultra.ultra_mode import fanout, structured


def test_chat_template_and_profile_loading() -> None:
    inspection = inspect_model("ultra/assets/examples/qwen")
    profile = fanout.load_ultra_profile(config_path=None, profile_name="default")
    messages = structured.collect_interactive_messages()
    rendered = inspection.apply_chat_template(messages)
    assert rendered.startswith("<<SYS>>")
    assert rendered.endswith("<|assistant|>")
    assert profile["ultra"]["n_candidates"] == 4


def test_structured_output_enforcement() -> None:
    inspection = inspect_model("ultra/assets/examples/qwen")
    profile = fanout.load_ultra_profile(config_path=None, profile_name="default")
    profile["structured_output"]["enabled"] = True
    profile["structured_output"]["json_schema"] = {"type": "object"}
    candidate = {"content": "answer"}
    payload = structured.enforce_structure(inspection=inspection, profile=profile, candidate=candidate)
    assert payload["schema"] == {"type": "object"}
