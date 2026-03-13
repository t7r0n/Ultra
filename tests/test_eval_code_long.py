from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultra.config import GlobalConfig
from ultra.eval import code as code_eval
from ultra.eval import longctx as long_eval


class CodeAndLongEvalTests(unittest.TestCase):
    def test_livecodebench_wrapper_builds_command_and_parses_pass_at_1(self) -> None:
        config = GlobalConfig()

        class FakeCompleted:
            returncode = 0
            stdout = '{"pass@1": 0.6}'
            stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "ultra.eval.code.module_available",
            side_effect=lambda name: name == "lcb_runner.runner.main",
        ), patch("ultra.eval.code.subprocess.run", return_value=FakeCompleted()):
            payload = code_eval.run_suite(
                model_ref="fake/model",
                tasks=["livecodebench", "release_version=release_v5", "scenario=codegeneration"],
                backend="hf",
                out=Path(tmpdir) / "livecodebench.json",
                config=config,
            )

        self.assertEqual(payload["benchmark"], "livecodebench")
        self.assertEqual(payload["pass@1"], 0.6)
        self.assertIn("--release_version", payload["command"])

    def test_swebench_wrapper_generates_predictions_from_local_tasks(self) -> None:
        config = GlobalConfig()

        class FakeCompleted:
            returncode = 0
            stdout = "pass@1: 0.2"
            stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "swebench_tasks.json"
            task_file.write_text(
                json.dumps([{"instance_id": "repo__issue-1", "prompt": "Write a patch"}]),
                encoding="utf-8",
            )
            fake_turn = types.SimpleNamespace(final_text="diff --git a/x b/x\n+fix\n")
            with patch(
                "ultra.eval.code.module_available",
                side_effect=lambda name: name == "swebench.harness.run_evaluation",
            ), patch("ultra.eval.code.subprocess.run", return_value=FakeCompleted()), patch(
                "ultra.eval.code.run_ultra_turn",
                return_value=fake_turn,
            ):
                payload = code_eval.run_suite(
                    model_ref="fake/model",
                    tasks=["swebench_lite", str(task_file)],
                    backend="hf",
                    out=Path(tmpdir) / "swebench.json",
                    config=config,
                )

            predictions = json.loads(Path(payload["predictions_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["benchmark"], "swebench_lite")
        self.assertEqual(payload["pass@1"], 0.2)
        self.assertEqual(predictions[0]["instance_id"], "repo__issue-1")
        self.assertIn("model_patch", predictions[0])

    def test_niah_suite_runs_synthetic_task(self) -> None:
        config = GlobalConfig()
        fake_turn = types.SimpleNamespace(final_text="The launch code is CORAL-47.")
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "ultra.eval.longctx.run_ultra_turn",
            return_value=fake_turn,
        ):
            payload = long_eval.run_suite(
                model_ref="fake/model",
                tasks=["niah", "needle=CORAL-47", "answer=CORAL-47", "question=What is the launch code?"],
                backend="hf",
                out=Path(tmpdir) / "niah.json",
                config=config,
            )

        self.assertEqual(payload["benchmark"], "niah")
        self.assertEqual(payload["accuracy"], 1.0)

    def test_longbench_wrapper_uses_dataset_rows(self) -> None:
        config = GlobalConfig()
        fake_turn = types.SimpleNamespace(final_text="final answer: blue")
        fake_datasets = types.SimpleNamespace(
            load_dataset=lambda dataset_name, split: [
                {
                    "id": "row-1",
                    "input": "Very long context",
                    "question": "What color?",
                    "answer": "blue",
                    "task": "qa",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "ultra.eval.longctx.module_available",
            side_effect=lambda name: name == "datasets",
        ), patch.dict(sys.modules, {"datasets": fake_datasets}), patch(
            "ultra.eval.longctx.run_ultra_turn",
            return_value=fake_turn,
        ):
            payload = long_eval.run_suite(
                model_ref="fake/model",
                tasks=["longbench_v2", "subset=qa", "limit=1"],
                backend="hf",
                out=Path(tmpdir) / "longbench.json",
                config=config,
            )

        self.assertEqual(payload["benchmark"], "longbench_v2")
        self.assertEqual(payload["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
