from __future__ import annotations

from ultra.mcd.inspector import inspect_model


def test_inspect_qwen_assets() -> None:
    result = inspect_model("ultra/assets/examples/qwen")
    assert result.arch == "qwen"
    assert result.n_kv_heads == 8
    assert result.rope_scaling == {"type": "linear", "factor": 2.0}
    assert result.context_window == 32768
    assert "llamacpp" in result.backends_supported
    assert "fp16" in result.dtype_candidates
    assert result.metadata["rope_scaling_note"] == "gguf_auto"
    rendered = result.apply_chat_template(
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Prompt"},
        ]
    )
    assert "<<SYS>>System<</SYS>>" in rendered
    assert "<|user|>\nPrompt\n" in rendered


def test_pretty_serialization() -> None:
    result = inspect_model("ultra/assets/examples/qwen")
    pretty = result.pretty()
    assert "\"arch\": \"qwen\"" in pretty
