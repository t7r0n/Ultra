from __future__ import annotations

from ultra.mcd.inspector import inspect_model
from ultra.tools.estimator import estimate_memory


def test_estimator_matches_formula() -> None:
    inspection = inspect_model("ultra/assets/examples/qwen")
    estimate = estimate_memory(
        inspection=inspection,
        context=8192,
        batch_size=1,
        precision="fp16",
        backend="hf",
    )
    bytes_per_value = 2
    expected_kv = 2 * inspection.n_layers * inspection.n_kv_heads * inspection.head_dim * bytes_per_value * 8192
    assert abs(estimate.kv_cache_bytes - expected_kv) / expected_kv < 0.03
    assert estimate.precision == "fp16"
