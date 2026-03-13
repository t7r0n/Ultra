from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultra.backends.selector import AutoBackendContext, select_backend
from ultra.mcd.inspector import ModelInspection
from ultra.tools.estimator import EstimatorInputs, estimate_memory
from ultra.tools.hwcheck import HardwareReport


class EstimatorTests(unittest.TestCase):
    def test_kv_formula_matches_spec(self) -> None:
        estimate = estimate_memory(
            EstimatorInputs(
                num_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                ctx=8192,
                batch=1,
                precision="fp16",
            ),
            model_ref="spec-example",
            backend="hf",
        )

        self.assertEqual(estimate.kv_bytes_per_token, 2 * 32 * 8 * 128 * 2)
        self.assertEqual(estimate.total_kv_bytes, (2 * 32 * 8 * 128 * 2) * 8192)

    def test_auto_backend_prefers_vllm_for_long_context_when_available(self) -> None:
        with patch("ultra.backends.selector.runtime_available", side_effect=lambda name: name == "vllm"):
            backend = select_backend(
                AutoBackendContext(
                    gguf=False,
                    has_gpu=True,
                    ctx=32768,
                    batch=1,
                )
            )

        self.assertEqual(backend, "vllm")

    def test_estimate_model_ref_uses_weight_sizes_and_headroom(self) -> None:
        from ultra.tools import estimator as estimator_module

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen2ForCausalLM"],
                        "torch_dtype": "float16",
                        "max_position_embeddings": 8192,
                        "num_hidden_layers": 32,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 8,
                        "hidden_size": 4096,
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "chat"}), encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"x" * 1024)

            fake_report = HardwareReport(
                python_version="3.11.0",
                python_executable="/usr/bin/python3",
                platform="test-platform",
                processor="test-cpu",
                machine="x86_64",
                cpu_count=8,
                total_memory_bytes=64 * 1024 * 1024 * 1024,
                cuda_version="12.8",
                rocm_version=None,
                torch_cuda_available=True,
                nvidia_smi_path="/usr/bin/nvidia-smi",
                nvml_available=False,
                gpus=[
                    {
                        "index": 0,
                        "name": "Test GPU",
                        "memory_total": 8 * 1024 * 1024 * 1024,
                        "memory_free": 6 * 1024 * 1024 * 1024,
                        "driver_version": "555.00",
                    }
                ],
                dependencies={},
            )

            with patch.object(estimator_module, "collect_hardware_report", return_value=fake_report):
                estimate = estimator_module.estimate_model_ref(
                    str(model_dir),
                    ctx=8192,
                    batch=1,
                    precision="fp16",
                    backend="auto",
                )

        self.assertEqual(estimate.weights_bytes, 1024)
        self.assertEqual(estimate.backend, "hf")
        self.assertEqual(estimate.device_kind, "gpu")
        self.assertEqual(estimate.device_name, "Test GPU")
        self.assertIsNotNone(estimate.headroom_bytes)
        self.assertGreater(estimate.headroom_bytes, 0)


if __name__ == "__main__":
    unittest.main()
