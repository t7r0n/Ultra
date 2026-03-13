from __future__ import annotations

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

from ultra.backends.common import GenerationRequest, StructuredOutputRequest
from ultra.backends import hf_engine, llama_cpp_engine, sglang_engine, vllm_engine
from ultra.tools.sandbox import build_sandbox_profile, run_in_sandbox


class BackendRuntimeTests(unittest.TestCase):
    def test_vllm_generate_passes_guided_json(self) -> None:
        calls: list[dict] = []

        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                del messages, tokenize, add_generation_prompt, enable_thinking
                return "prompt"

        class FakeOutput:
            text = '{"answer": "ok"}'
            token_ids = [1, 2, 3]
            cumulative_logprob = -0.6
            finish_reason = "stop"

        class FakeResult:
            outputs = [FakeOutput()]
            prompt_token_ids = [10, 11]

        class FakeLLM:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def generate(self, prompts, params, use_tqdm=False):
                del prompts, use_tqdm
                calls.append(params.kwargs)
                return [FakeResult()]

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
        fake_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()))

        with patch.dict(sys.modules, {"vllm": fake_vllm, "transformers": fake_transformers}):
            result = vllm_engine.generate(
                GenerationRequest(
                    model_ref="fake/model",
                    messages=[{"role": "user", "content": "hi"}],
                    structured_output=StructuredOutputRequest(enabled=True, json_schema={"type": "object"}),
                )
            )

        self.assertEqual(result.text, '{"answer": "ok"}')
        self.assertIn("guided_json", calls[0])

    def test_vllm_generate_can_use_openai_compatible_server(self) -> None:
        response_payload = {
            "model": "fake/model",
            "choices": [
                {
                    "message": {"content": '{"answer": "ok"}'},
                    "finish_reason": None,
                    "logprobs": {"content": [{"logprob": -0.1}]},
                }
            ],
            "usage": {"completion_tokens": 3},
        }
        with patch.dict("os.environ", {"ULTRA_VLLM_BASE_URL": "http://localhost:8000"}, clear=False), patch(
            "ultra.backends.vllm_engine._post_json",
            return_value=response_payload,
        ):
            result = vllm_engine.generate(
                GenerationRequest(
                    model_ref="fake/model",
                    messages=[{"role": "user", "content": "hi"}],
                    structured_output=StructuredOutputRequest(enabled=True, json_schema={"type": "object"}),
                )
            )

        self.assertEqual(result.text, '{"answer": "ok"}')
        self.assertEqual(result.metrics["transport"], "server")
        self.assertEqual(result.metrics["structured_decoder"], "guided_json")

    def test_docker_sandbox_adds_network_none(self) -> None:
        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        with patch("ultra.tools.sandbox.shutil.which", return_value="/usr/bin/docker"), patch(
            "ultra.tools.sandbox.subprocess.run",
            return_value=FakeCompleted(),
        ) as run_mock:
            run_in_sandbox(
                ["python -m pytest -q"],
                profile=build_sandbox_profile("docker", allow_network=False),
                cwd=Path.cwd(),
            )

        command = run_mock.call_args.args[0]
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("--cap-drop", command)
        self.assertIn("ALL", command)
        self.assertIn("no-new-privileges", command)

    def test_hf_structured_prefix_uses_lm_format_enforcer_when_available(self) -> None:
        fake_lmfe = types.SimpleNamespace(
            JsonSchemaParser=lambda schema: ("json", schema),
            RegexParser=lambda regex: ("regex", regex),
        )
        fake_integration = types.SimpleNamespace(
            build_transformers_prefix_allowed_tokens_fn=lambda tokenizer, parser: ("prefix", tokenizer, parser)
        )
        with patch.dict(
            sys.modules,
            {
                "lmformatenforcer": fake_lmfe,
                "lmformatenforcer.integrations.transformers": fake_integration,
            },
        ):
            prefix = hf_engine.structured_prefix_allowed_tokens_fn(
                object(),
                StructuredOutputRequest(enabled=True, json_schema={"type": "object"}),
            )

        self.assertEqual(prefix[0], "prefix")

    def test_hf_generate_with_outlines_for_grammar_when_available(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [1 for _ in text.split()] or [1]

        fake_outlines = types.SimpleNamespace(
            from_transformers=lambda model, tokenizer: ("wrapped", model, tokenizer),
            Generator=lambda wrapped, cfg: (lambda prompt_text, **kwargs: "GRAMMAR_OK"),
        )
        fake_types = types.SimpleNamespace(CFG=lambda grammar: grammar)
        result = None
        with patch.dict(sys.modules, {"outlines": fake_outlines, "outlines.types": fake_types}):
            result = hf_engine.generate_with_outlines(
                model=object(),
                tokenizer=FakeTokenizer(),
                prompt_text="hello world",
                request=GenerationRequest(
                    model_ref="fake/model",
                    messages=[{"role": "user", "content": "hi"}],
                    structured_output=StructuredOutputRequest(enabled=True, grammar="root ::= 'x'"),
                ),
                effective_max_new_tokens=16,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.text, "GRAMMAR_OK")

    def test_sglang_generate_records_prompt_and_stop_metadata(self) -> None:
        response_payload = {
            "model": "fake/model",
            "choices": [
                {
                    "message": {"content": "hello world"},
                    "finish_reason": None,
                    "logprobs": {"content": [{"logprob": -0.2}, {"logprob": -0.4}]},
                }
            ],
            "usage": {"completion_tokens": 2},
        }
        with patch.dict("os.environ", {"ULTRA_SGLANG_BASE_URL": "http://localhost:30000"}, clear=False), patch(
            "ultra.backends.sglang_engine._post_json",
            return_value=response_payload,
        ):
            result = sglang_engine.generate(
                GenerationRequest(
                    model_ref="fake/model",
                    messages=[{"role": "user", "content": "hi"}],
                    enable_thinking=True,
                    structured_output=StructuredOutputRequest(enabled=True, regex="hello world"),
                )
            )

        self.assertIn("user: hi", result.prompt_text)
        self.assertEqual(result.finish_reason, "stop")
        self.assertIn("</think>", result.metrics["stop_strings"])
        self.assertEqual(result.metrics["structured_decoder"], "guided_regex")
        self.assertAlmostEqual(result.avg_logprob or 0.0, -0.3, places=4)

    def test_llamacpp_generate_records_prompt_and_stop_metadata(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeLlama:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def create_chat_completion(self, **kwargs):
                calls.append(kwargs)
                return {
                    "choices": [{"message": {"content": "hello world"}, "finish_reason": None}],
                    "usage": {"completion_tokens": 0},
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf = Path(tmpdir) / "model.gguf"
            gguf.write_bytes(b"gguf")
            with patch.dict(sys.modules, {"llama_cpp": types.SimpleNamespace(Llama=FakeLlama)}):
                result = llama_cpp_engine.generate(
                    GenerationRequest(
                        model_ref=str(gguf),
                        messages=[{"role": "user", "content": "hi"}],
                        enable_thinking=True,
                        structured_output=StructuredOutputRequest(enabled=True, json_schema={"type": "object"}),
                    )
                )

        self.assertIn("user: hi", result.prompt_text)
        self.assertIn("</think>", calls[0]["stop"])
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.metrics["structured_decoder"], "json_repair")
        self.assertEqual(result.generated_tokens, 2)

    def test_llamacpp_generate_can_use_openai_compatible_server(self) -> None:
        response_payload = {
            "model": "fake/model",
            "choices": [{"message": {"content": "hello world"}, "finish_reason": None}],
            "usage": {"completion_tokens": 2},
        }
        with patch.dict("os.environ", {"ULTRA_LLAMACPP_BASE_URL": "http://localhost:8080"}, clear=False), patch(
            "ultra.backends.llama_cpp_engine._post_json",
            return_value=response_payload,
        ):
            result = llama_cpp_engine.generate(
                GenerationRequest(
                    model_ref="fake/model",
                    messages=[{"role": "user", "content": "hi"}],
                    enable_thinking=True,
                )
            )

        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.metrics["transport"], "server")
        self.assertIn("</think>", result.metrics["stop_strings"])
