from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultra.cli import main
from ultra.cli import collect_fetch_verification
from ultra.config import load_config


class TemplateTests(unittest.TestCase):
    def test_cli_help_contains_spec_commands(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(["--help"])
        help_text = buffer.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("usage: ultra", help_text)
        for command in ("doctor", "fetch", "inspect", "estimate", "chat", "agent", "eval"):
            self.assertIn(command, help_text)

    def test_example_config_loads_as_nested_dataclasses(self) -> None:
        example = ROOT / "src" / "ultra" / "assets" / "examples" / "ultra.yaml"
        config = load_config(example)

        self.assertEqual(config.backend, "auto")
        self.assertEqual(config.ultra.n_candidates, 8)
        self.assertEqual(config.ultra.diversity.temperatures, [0.2, 0.6, 0.9])
        self.assertTrue(config.ultra.selection.self_consistency)
        self.assertTrue(config.long_context.allow_rope_scaling)
        self.assertIsNone(config.structured_output.regex)
        self.assertIsNone(config.structured_output.grammar)
        self.assertEqual(config.judge.backend, "auto")
        self.assertEqual(config.judge.top_k, 4)
        self.assertTrue(config.judge.pairwise)

    def test_chat_scaffold_creates_spec_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "chat-run"
            with redirect_stdout(io.StringIO()):
                exit_code = main(["chat", "Qwen/Qwen2.5-7B-Instruct", "--logdir", str(run_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "run.json").exists())
            self.assertTrue((run_dir / "prompts.jsonl").exists())
            self.assertTrue((run_dir / "candidates.jsonl").exists())
            self.assertTrue((run_dir / "selection.json").exists())
            self.assertTrue((run_dir / "final.txt").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertIn("hardware", run_payload)
            self.assertIn("config", run_payload)
            selection_payload = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
            self.assertTrue(selection_payload["initialized"])
            self.assertIn("scores", selection_payload)
            metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics_payload["initialized"])
            self.assertIn("latency_s", metrics_payload)

    def test_spec_named_modules_exist(self) -> None:
        required_files = [
            ROOT / "src" / "ultra" / "cli.py",
            ROOT / "src" / "ultra" / "backends" / "hf_engine.py",
            ROOT / "src" / "ultra" / "backends" / "vllm_engine.py",
            ROOT / "src" / "ultra" / "backends" / "sglang_engine.py",
            ROOT / "src" / "ultra" / "backends" / "llama_cpp_engine.py",
            ROOT / "src" / "ultra" / "mcd" / "inspector.py",
            ROOT / "src" / "ultra" / "ultra_mode" / "fanout.py",
            ROOT / "src" / "ultra" / "ultra_mode" / "selection.py",
            ROOT / "src" / "ultra" / "ultra_mode" / "refine.py",
            ROOT / "src" / "ultra" / "ultra_mode" / "structured.py",
            ROOT / "src" / "ultra" / "eval" / "harness.py",
            ROOT / "src" / "ultra" / "eval" / "code.py",
            ROOT / "src" / "ultra" / "eval" / "longctx.py",
            ROOT / "src" / "ultra" / "eval" / "arena.py",
            ROOT / "src" / "ultra" / "eval" / "rag.py",
            ROOT / "src" / "ultra" / "tools" / "sandbox.py",
            ROOT / "src" / "ultra" / "tools" / "verifiers.py",
            ROOT / "src" / "ultra" / "tools" / "estimator.py",
            ROOT / "src" / "ultra" / "tools" / "hwcheck.py",
            ROOT / "src" / "ultra" / "tools" / "logging.py",
            ROOT / "tests" / "test_mcd.py",
            ROOT / "tests" / "test_estimator.py",
            ROOT / "tests" / "test_templates.py",
        ]
        for required_file in required_files:
            self.assertTrue(required_file.exists(), required_file)

    def test_pyproject_console_entry_matches_scaffold(self) -> None:
        pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('ultra = "ultra.cli:app"', pyproject_text)

    def test_fetch_verification_detects_transformers_and_gguf_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"x")
            verification = collect_fetch_verification(model_dir)
            self.assertEqual(verification["model_class"], "transformers")
            self.assertTrue(verification["required_files_ok"])

            gguf_dir = Path(tmpdir) / "gguf"
            gguf_dir.mkdir()
            (gguf_dir / "model.gguf").write_bytes(b"gguf")
            gguf_verification = collect_fetch_verification(gguf_dir)
            self.assertEqual(gguf_verification["model_class"], "gguf")
            self.assertTrue(gguf_verification["required_files_ok"])


if __name__ == "__main__":
    unittest.main()
