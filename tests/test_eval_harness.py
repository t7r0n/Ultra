from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultra.eval import harness as harness_eval


class HarnessEvalTests(unittest.TestCase):
    def test_harness_model_adapter_supports_hf_vllm_sglang_and_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"x")

            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_bytes(b"gguf")

            hf_model, hf_args = harness_eval.harness_model_adapter(model_ref=str(model_dir), backend="hf")
            vllm_model, vllm_args = harness_eval.harness_model_adapter(model_ref=str(model_dir), backend="vllm")
            with patch.dict(os.environ, {"ULTRA_SGLANG_BASE_URL": "http://localhost:30000"}, clear=False):
                sglang_model, sglang_args = harness_eval.harness_model_adapter(model_ref=str(model_dir), backend="sglang")
            gguf_model, gguf_args = harness_eval.harness_model_adapter(model_ref=str(gguf_path), backend="llamacpp")

        self.assertEqual(hf_model, "hf")
        self.assertEqual(hf_args, f"pretrained={model_dir}")
        self.assertEqual(vllm_model, "vllm")
        self.assertIn("gpu_memory_utilization=0.9", vllm_args)
        self.assertEqual(sglang_model, "local-chat-completions")
        self.assertIn("base_url=http://localhost:30000/v1/chat/completions", sglang_args)
        self.assertEqual(gguf_model, "gguf")
        self.assertIn(f"gguf_file={gguf_path}", gguf_args)

    def test_run_suite_records_command_and_results_payload(self) -> None:
        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(command, capture_output, text, check):
            del capture_output, text, check
            output_dir = Path(command[command.index("--output_path") + 1])
            (output_dir / "results.json").write_text(json.dumps({"results": {"mmlu": {"acc,none": 0.5}}}), encoding="utf-8")
            return FakeCompleted()

        with tempfile.TemporaryDirectory() as tmpdir, patch("ultra.eval.harness.shutil.which", return_value="/usr/bin/lm_eval"), patch(
            "ultra.eval.harness.subprocess.run",
            side_effect=fake_run,
        ):
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"x")
            payload = harness_eval.run_suite(
                model_ref=str(model_dir),
                tasks=["mmlu"],
                backend="vllm",
                out=Path(tmpdir) / "harness.json",
            )

        self.assertEqual(payload["model"], "vllm")
        self.assertIn("mmlu", payload["results"]["results"])


if __name__ == "__main__":
    unittest.main()
