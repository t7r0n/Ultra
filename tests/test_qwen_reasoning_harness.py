from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_qwen_reasoning_modes.py"


def load_harness_module():
    spec = importlib.util.spec_from_file_location("benchmark_qwen_reasoning_modes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = load_harness_module()


class FakeGenerateOutput:
    def __init__(self, sequences: torch.Tensor, scores: list[torch.Tensor]) -> None:
        self.sequences = sequences
        self.scores = scores


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.decodes = {
            (101, 102, 103): "<think>\nplan carefully\n</think>",
            (201, 202): '{"answer": "B"}',
        }
        self.encodes = {
            "</think>": [103],
            "\n</think>": [103],
            "</think>\n": [103],
            "\n</think>\n": [103],
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        del messages, tokenize, add_generation_prompt, enable_thinking
        return "prompt"

    def __call__(self, text: str, return_tensors: str = "pt"):
        del text, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return self.encodes[text]

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return self.decodes[tuple(ids.tolist())]


class FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        call_index = len(self.generate_calls)
        if call_index == 1:
            return FakeGenerateOutput(
                sequences=torch.tensor([[1, 2, 101, 102, 103]], dtype=torch.long),
                scores=[torch.tensor([[0.0]]), torch.tensor([[0.0]]), torch.tensor([[0.0]])],
            )
        if call_index == 2:
            return FakeGenerateOutput(
                sequences=torch.tensor([[1, 2, 101, 102, 103, 201, 202]], dtype=torch.long),
                scores=[torch.tensor([[0.0]]), torch.tensor([[0.0]])],
            )
        raise AssertionError("Unexpected extra generate call")

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        del sequences, normalize_logits
        if len(scores) == 3:
            return torch.tensor([[-0.4, -0.4, -0.4]], dtype=torch.float32)
        if len(scores) == 2:
            return torch.tensor([[-0.2, -0.2]], dtype=torch.float32)
        raise AssertionError("Unexpected score length")


class QwenReasoningHarnessTests(unittest.TestCase):
    def make_condition(self, *, thinking: bool, family: str = "qwen3"):
        return HARNESS.Condition(
            label="test",
            model_id="test/model",
            thinking=thinking,
            family=family,
            text_temperature=0.7,
            text_top_p=0.9,
            text_top_k=20,
            text_min_p=0.0,
        )

    def make_task(self, *, evaluator: str, answer, aliases: list[str] | None = None):
        return HARNESS.Task(
            task_id="test_task",
            category="logic" if evaluator in {"choice", "text"} else "math",
            prompt="test prompt",
            answer=answer,
            evaluator=evaluator,
            aliases=aliases or [],
        )

    def test_split_reasoning_and_answer_preserves_unclosed_think_text(self) -> None:
        parsed = HARNESS.split_reasoning_and_answer(
            "<think>\nreasoning without close",
            family="qwen3",
            thinking=True,
        )

        self.assertEqual(parsed["reasoning_text"], "reasoning without close")
        self.assertEqual(parsed["answer_text"], "")
        self.assertTrue(parsed["truncated_inside_think"])
        self.assertFalse(parsed["think_closed"])

    def test_split_reasoning_and_answer_extracts_post_think_answer(self) -> None:
        parsed = HARNESS.split_reasoning_and_answer(
            "<think>\nreasoning\n</think>\nFINAL: 22",
            family="qwen3",
            thinking=True,
        )

        self.assertEqual(parsed["reasoning_text"], "reasoning")
        self.assertEqual(parsed["answer_text"], "FINAL: 22")
        self.assertTrue(parsed["think_closed"])
        self.assertTrue(parsed["answer_stage_reached"])

    def test_extract_canonical_answer_prefers_boxed_number(self) -> None:
        task = self.make_task(evaluator="number", answer="22")

        canonical, source = HARNESS.extract_canonical_answer(
            task,
            answer_text="Final result: \\boxed{22}",
            fallback_text="22",
        )

        self.assertEqual(canonical, "22")
        self.assertEqual(source, "boxed")

    def test_evaluate_candidate_matches_text_alias_inside_sentence(self) -> None:
        task = self.make_task(
            evaluator="text",
            answer="Mixed",
            aliases=["mixed box", "the box labeled mixed"],
        )
        condition = self.make_condition(thinking=False)

        candidate = HARNESS.evaluate_candidate(
            task,
            "The correct box is the Mixed box.",
            generated_tokens=12,
            latency_s=0.1,
            condition=condition,
            meta={"strategy": "single_pass"},
        )

        self.assertEqual(candidate.score, 1.0)
        self.assertEqual(candidate.meta["canonical_source"], "fallback_text")

    def test_evaluate_candidate_keeps_reasoning_text_when_think_never_closes(self) -> None:
        task = self.make_task(evaluator="number", answer="22")
        condition = self.make_condition(thinking=True, family="qwen3")

        candidate = HARNESS.evaluate_candidate(
            task,
            "<think>\nI will solve this carefully but never finish",
            generated_tokens=32,
            latency_s=0.1,
            condition=condition,
            meta={"strategy": "single_pass"},
        )

        self.assertTrue(candidate.cleaned_text.startswith("I will solve this carefully"))
        self.assertTrue(candidate.meta["truncated_inside_think"])
        self.assertFalse(candidate.meta["answer_stage_reached"])

    def test_long_reasoning_trace_does_not_count_as_text_answer(self) -> None:
        task = self.make_task(
            evaluator="text",
            answer="Mixed",
            aliases=["mixed box", "the box labeled mixed"],
        )
        condition = self.make_condition(thinking=True, family="qwen3")
        raw = (
            "<think>\n"
            "Let me reason this through carefully. The labels Apples, Oranges, and Mixed are all wrong. "
            "The Mixed box cannot be mixed, and the box labeled Mixed might actually contain apples or oranges. "
            "I should consider what happens if I draw from the box labeled Mixed and compare the possibilities."
        )

        candidate = HARNESS.evaluate_candidate(
            task,
            raw,
            generated_tokens=64,
            latency_s=0.1,
            condition=condition,
            meta={"strategy": "single_pass"},
        )

        self.assertEqual(candidate.score, 0.0)
        self.assertIsNone(candidate.canonical_answer)

    def test_single_user_prompt_uses_boxed_schema_for_math(self) -> None:
        task = self.make_task(evaluator="number", answer="22")

        prompt = HARNESS.single_user_prompt(task)

        self.assertIn(r"\boxed{...}", prompt)
        self.assertNotIn("FINAL:", prompt)

    def test_single_user_prompt_uses_json_schema_for_text(self) -> None:
        task = self.make_task(evaluator="text", answer="Mixed")

        prompt = HARNESS.single_user_prompt(task)

        self.assertIn('{"answer": "<final_answer>"}', prompt)
        self.assertNotIn("FINAL:", prompt)

    def test_stop_on_token_sequences_matches_suffix_only(self) -> None:
        stopping = HARNESS.StopOnTokenSequences([[7, 8]])

        self.assertFalse(stopping(torch.tensor([[1, 7]], dtype=torch.long), torch.tensor([[0.0]])))
        self.assertTrue(stopping(torch.tensor([[1, 7, 8]], dtype=torch.long), torch.tensor([[0.0]])))

    def test_select_candidate_does_not_use_ground_truth_score_for_non_code(self) -> None:
        task = self.make_task(evaluator="text", answer="beta")
        candidate_with_score = HARNESS.CandidateResult(
            raw_text="wrong",
            cleaned_text="wrong",
            canonical_answer="alpha",
            score=1.0,
            latency_s=0.1,
            generated_tokens=8,
            think_block_present=False,
            meta={
                "canonical_source": "fallback_text",
                "answer_stage_reached": False,
                "think_closed": False,
                "avg_logprob": -9.0,
            },
        )
        candidate_without_score = HARNESS.CandidateResult(
            raw_text="FINAL: beta",
            cleaned_text="FINAL: beta",
            canonical_answer="beta",
            score=0.0,
            latency_s=0.2,
            generated_tokens=8,
            think_block_present=False,
            meta={
                "canonical_source": "final",
                "answer_stage_reached": True,
                "think_closed": True,
                "avg_logprob": -1.0,
            },
        )

        selected = HARNESS.select_candidate(task, [candidate_with_score, candidate_without_score])

        self.assertIs(selected, candidate_without_score)

    def test_generate_once_uses_two_stage_answer_continuation_for_thinking(self) -> None:
        task = self.make_task(evaluator="choice", answer="B")
        condition = self.make_condition(thinking=True, family="qwen3")
        model = FakeModel()
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "system", "content": HARNESS.system_prompt(task, concise=True)},
            {"role": "user", "content": HARNESS.single_user_prompt(task)},
        ]

        generation = HARNESS.generate_once(
            model,
            tokenizer,
            messages,
            condition=condition,
            task=task,
            strategy="single_pass",
            candidate_index=0,
            single_token_multiplier=1.0,
            thinking_token_multiplier=1.0,
        )

        self.assertEqual(len(model.generate_calls), 2)
        self.assertTrue(model.generate_calls[0]["do_sample"])
        self.assertFalse(model.generate_calls[1]["do_sample"])
        self.assertIn("stopping_criteria", model.generate_calls[0])
        self.assertNotIn("stopping_criteria", model.generate_calls[1])
        self.assertEqual(generation.meta["generation_stage_count"], 2)
        self.assertTrue(generation.meta["answer_stage_invoked"])
        self.assertEqual(generation.meta["stop_sequence_count"], 1)
        self.assertIn("</think>", generation.raw_text)
        self.assertIn('{"answer": "B"}', generation.raw_text)


if __name__ == "__main__":
    unittest.main()
