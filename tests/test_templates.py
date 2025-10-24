from __future__ import annotations

from ultra.backends import Completion, SamplingParameters
from ultra.mcd.inspector import inspect_model
from ultra.ultra_mode import fanout, structured
from ultra.ultra_mode.fanout import Candidate


def test_chat_template_and_profile_loading() -> None:
    inspection = inspect_model("ultra/assets/examples/qwen")
    profile = fanout.load_ultra_profile(config_path=None, profile_name="default")
    messages = structured.collect_interactive_messages()
    rendered = inspection.apply_chat_template(messages)
    assert rendered.startswith("<<SYS>>")
    assert rendered.endswith("<|assistant|>")
    assert profile.ultra.n_candidates == 8


def test_structured_output_enforcement() -> None:
    inspection = inspect_model("ultra/assets/examples/qwen")
    profile = fanout.load_ultra_profile(config_path=None, profile_name="default")
    profile.structured_output.enabled = True
    profile.structured_output.json_schema = {"type": "object"}
    candidate = Candidate(
        index=0,
        messages=[],
        prompt="",
        sampling=SamplingParameters(temperature=0.2, top_p=0.9),
        completion=Completion(text="{}"),
        style=None,
        engine="hf",
    )
    payload = structured.enforce_structure(inspection=inspection, profile=profile, candidate=candidate)
    assert payload["schema"] == {"type": "object"}
