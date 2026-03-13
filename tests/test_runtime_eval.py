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

from ultra.backends.common import GenerationRequest, GenerationResult
from ultra.config import GlobalConfig
from ultra.eval import arena as arena_eval
from ultra.eval import code as code_eval
from ultra.eval import rag as rag_eval
from ultra.mcd.inspector import ModelInspection
from ultra.runtime import run_ultra_turn
from ultra.tools.logging import initialize_run_artifacts


class RuntimeEvalTests(unittest.TestCase):
    def test_ultra_turn_can_beat_single_candidate_via_self_consistency(self) -> None:
        config_single = GlobalConfig()
        config_single.ultra.n_candidates = 1
        config_single.ultra.max_n_candidates = 1
        config_single.ultra.refine.self_refine_passes = 0
        config_ultra = GlobalConfig()
        config_ultra.ultra.n_candidates = 4
        config_ultra.ultra.refine.self_refine_passes = 0

        def fake_generate(request: GenerationRequest) -> GenerationResult:
            if request.seed is None:
                raise AssertionError("seed should be set")
            if request.seed % 10 == 0:
                text = "wrong answer"
            else:
                text = "correct answer"
            return GenerationResult(
                backend="hf",
                text=text,
                prompt_text="prompt",
                latency_s=0.1,
                generated_tokens=4,
                avg_logprob=-0.1,
                finish_reason="stop",
                model_name=request.model_ref,
            )

        with patch("ultra.runtime.BACKEND_GENERATORS", {"hf": fake_generate}):
            single = run_ultra_turn(
                model_ref="fake/model",
                backend="hf",
                config=config_single,
                ultra_profile="default",
                conversation=[{"role": "user", "content": "answer the question"}],
                artifacts=None,
                seed_base=10,
            )
            ultra = run_ultra_turn(
                model_ref="fake/model",
                backend="hf",
                config=config_ultra,
                ultra_profile="default",
                conversation=[{"role": "user", "content": "answer the question"}],
                artifacts=None,
                seed_base=10,
            )

        self.assertEqual(single.final_text, "wrong answer")
        self.assertEqual(ultra.final_text, "correct answer")

    def test_ultra_turn_writes_structured_final_json(self) -> None:
        config = GlobalConfig()
        config.ultra.n_candidates = 2
        config.ultra.refine.self_refine_passes = 0
        config.structured_output.enabled = True
        config.structured_output.json_schema = {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        }

        def fake_generate(request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                backend="hf",
                text='{"answer": "ok"}',
                prompt_text="prompt",
                latency_s=0.1,
                generated_tokens=4,
                avg_logprob=-0.1,
                finish_reason="stop",
                model_name=request.model_ref,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "chat-run"
            artifacts = initialize_run_artifacts(
                command="chat",
                model_ref="fake/model",
                backend="hf",
                config=config.to_dict(),
                hardware=None,
                logdir=run_dir,
                extra={},
                structured_output=True,
            )
            with patch("ultra.runtime.BACKEND_GENERATORS", {"hf": fake_generate}):
                run_ultra_turn(
                    model_ref="fake/model",
                    backend="hf",
                    config=config,
                    ultra_profile="default",
                    conversation=[{"role": "user", "content": "answer in json"}],
                    artifacts=artifacts,
                    seed_base=1,
                )
            payload = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
            prompt_entry = json.loads((run_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
            candidate_entry = json.loads((run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
            selection_payload = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
            metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["answer"], "ok")
        self.assertIn("request", prompt_entry)
        self.assertIn("backend_metrics", candidate_entry)
        self.assertIn("selected_candidate", selection_payload)
        self.assertIn("selected_candidate_metrics", metrics_payload)

    def test_code_eval_local_suite_reports_pass_at_1(self) -> None:
        config = GlobalConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "code_tasks.json"
            task_file.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "trim",
                            "prompt": "Write trim(x) that strips spaces.",
                            "function_name": "trim",
                            "tests": ["assert trim('  hi  ') == 'hi'"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            fake_turn = types.SimpleNamespace(final_text="def trim(x):\n    return x.strip()\n")
            with patch("ultra.eval.code.run_ultra_turn", return_value=fake_turn):
                payload = code_eval.run_suite(
                    model_ref="fake/model",
                    tasks=[str(task_file)],
                    backend="hf",
                    out=Path(tmpdir) / "results.json",
                    config=config,
                )

        self.assertEqual(payload["pass@1"], 1.0)

    def test_code_eval_evalplus_payload_parses_pass_at_1(self) -> None:
        config = GlobalConfig()

        class FakeCompleted:
            returncode = 0
            stdout = '{"pass@1": 0.75}'
            stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir, patch("ultra.eval.code.shutil.which", return_value="/usr/bin/evalplus.evaluate"), patch(
            "ultra.eval.code.subprocess.run",
            return_value=FakeCompleted(),
        ):
            payload = code_eval.run_suite(
                model_ref="fake/model",
                tasks=["humaneval_plus"],
                backend="hf",
                out=Path(tmpdir) / "evalplus.json",
                config=config,
            )

        self.assertEqual(payload["pass@1"], 0.75)

    def test_arena_suite_blinds_candidate_ids(self) -> None:
        config = GlobalConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "arena_tasks.json"
            task_file.write_text(json.dumps([{"task_id": "1", "prompt": "Say hi."}]), encoding="utf-8")
            turns = [
                types.SimpleNamespace(final_text="ultra answer"),
                types.SimpleNamespace(final_text="greedy"),
            ]
            with patch("ultra.eval.arena.run_ultra_turn", side_effect=turns):
                payload = arena_eval.run_suite(
                    model_ref="fake/model",
                    tasks=[str(task_file)],
                    backend="hf",
                    out=Path(tmpdir) / "arena.json",
                    config=config,
                )

        candidates = payload["judge_logs"][0]["candidates"]
        blind_ids = {candidate["blind_id"] for candidate in candidates}
        self.assertEqual(blind_ids, {"candidate_a", "candidate_b"})

    def test_rag_suite_computes_metrics(self) -> None:
        config = GlobalConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "rag_tasks.json"
            task_file.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "rag1",
                            "question": "What animal barks?",
                            "contexts": ["Dogs bark loudly.", "Cats purr."],
                            "answer": "dog",
                            "response": "Dog",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            payload = rag_eval.run_suite(
                model_ref="fake/model",
                tasks=[str(task_file)],
                backend="hf",
                out=Path(tmpdir) / "rag.json",
                config=config,
            )

        self.assertEqual(payload["suite"], "rag")
        self.assertIn("faithfulness", payload["metrics"][0])
        self.assertIn("answer_relevancy", payload["metrics"][0])

    def test_rag_suite_uses_ragas_when_available(self) -> None:
        config = GlobalConfig()

        class FakeDataset:
            @staticmethod
            def from_list(rows):
                return rows

        class FakeResult:
            def to_dict(self):
                return {
                    "faithfulness": [0.9],
                    "answer_relevancy": [0.8],
                    "context_precision": [0.7],
                    "context_recall": [0.6],
                }

        fake_ragas = types.SimpleNamespace(evaluate=lambda dataset, metrics: FakeResult())
        fake_ragas_metrics = types.SimpleNamespace(
            faithfulness=object(),
            answer_relevancy=object(),
            context_precision=object(),
            context_recall=object(),
        )
        fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "ultra.eval.rag.ragas_available",
            return_value=True,
        ), patch.dict(
            sys.modules,
            {
                "ragas": fake_ragas,
                "ragas.metrics": fake_ragas_metrics,
                "datasets": fake_datasets,
            },
        ):
            task_file = Path(tmpdir) / "rag_tasks.json"
            task_file.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "rag1",
                            "question": "What animal barks?",
                            "contexts": ["Dogs bark loudly.", "Cats purr."],
                            "answer": "dog",
                            "response": "Dog",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            payload = rag_eval.run_suite(
                model_ref="fake/model",
                tasks=[str(task_file)],
                backend="hf",
                out=Path(tmpdir) / "rag_ragas.json",
                config=config,
            )

        self.assertEqual(payload["backend"], "ragas")
        self.assertEqual(payload["metrics"][0]["faithfulness"], 0.9)

    def test_long_context_policy_caps_mistral_sliding_window(self) -> None:
        config = GlobalConfig()
        config.context_window = 16384
        config.ultra.n_candidates = 1
        config.ultra.max_n_candidates = 1
        config.ultra.refine.self_refine_passes = 0
        seen_extras: list[dict[str, object]] = []

        def fake_generate(request: GenerationRequest) -> GenerationResult:
            seen_extras.append(dict(request.extra))
            return GenerationResult(
                backend="hf",
                text="ok",
                prompt_text="prompt",
                latency_s=0.1,
                generated_tokens=2,
                avg_logprob=-0.1,
                finish_reason="stop",
                model_name=request.model_ref,
            )

        inspection = ModelInspection(
            backend="hf",
            arch="MistralForCausalLM",
            dtype_candidates=["float16"],
            context_window=32768,
            sliding_window=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            hidden_size=4096,
            chat_template="chat",
            rope_scaling=None,
            tokenizer_info={},
            vision_support=False,
            stop_tokens=[],
            eos_tokens=[],
            system_prompt_hint=None,
            backends_supported=["hf", "vllm"],
            notes=[],
        )

        with patch("ultra.runtime.BACKEND_GENERATORS", {"hf": fake_generate}), patch(
            "ultra.runtime.inspect_model",
            return_value=inspection,
        ):
            turn = run_ultra_turn(
                model_ref="fake/model",
                backend="hf",
                config=config,
                ultra_profile="reasoning",
                conversation=[{"role": "user", "content": "summarize the document"}],
                artifacts=None,
                seed_base=1,
            )

        self.assertEqual(seen_extras[0]["max_context_tokens"], 4096)
        self.assertEqual(turn.metrics["long_context_policy"]["effective_context_window"], 4096)

    def test_ultra_turn_judge_can_overrule_logprob_ranking(self) -> None:
        config = GlobalConfig()
        config.ultra.n_candidates = 2
        config.ultra.max_n_candidates = 2
        config.ultra.refine.self_refine_passes = 0
        config.judge.enabled = True
        config.judge.top_k = 2
        config.judge.weight = 0.8
        config.judge.bias_mitigation.shuffle = False

        def fake_generate(request: GenerationRequest) -> GenerationResult:
            if request.extra.get("ultra_role") == "judge":
                text = json.dumps(
                    {
                        "winner": "candidate_b",
                        "score_a": 0.2,
                        "score_b": 0.95,
                        "reason": "candidate_b is clearly more correct",
                    }
                )
                return GenerationResult(
                    backend="hf",
                    text=text,
                    prompt_text="judge prompt",
                    latency_s=0.05,
                    generated_tokens=16,
                    avg_logprob=None,
                    finish_reason="stop",
                    model_name=request.model_ref,
                )
            if request.seed is None:
                raise AssertionError("seed should be set")
            if request.seed % 10 == 0:
                text = "candidate zero"
                avg_logprob = -0.05
            else:
                text = "candidate one"
                avg_logprob = -0.5
            return GenerationResult(
                backend="hf",
                text=text,
                prompt_text="prompt",
                latency_s=0.1,
                generated_tokens=4,
                avg_logprob=avg_logprob,
                finish_reason="stop",
                model_name=request.model_ref,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "judge-run"
            artifacts = initialize_run_artifacts(
                command="chat",
                model_ref="fake/model",
                backend="hf",
                config=config.to_dict(),
                hardware=None,
                logdir=run_dir,
                extra={},
                structured_output=False,
            )
            with patch("ultra.runtime.BACKEND_GENERATORS", {"hf": fake_generate}):
                turn = run_ultra_turn(
                    model_ref="fake/model",
                    backend="hf",
                    config=config,
                    ultra_profile="default",
                    conversation=[{"role": "user", "content": "answer the question"}],
                    artifacts=artifacts,
                    seed_base=10,
                )

            selection_payload = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
            metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            candidate_entries = [
                json.loads(line)
                for line in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(turn.final_text, "candidate one")
        self.assertEqual(selection_payload["selected_index"], 1)
        self.assertTrue(selection_payload["judge"]["enabled"])
        self.assertEqual(metrics_payload["judge_comparison_count"], 1)
        self.assertEqual(candidate_entries[1]["judge_votes"], 1.0)
