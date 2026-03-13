# Qwen3 / Qwen3.5 GPU Benchmark Report

Date: 2026-03-12

## Setup

- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design, 8 GB VRAM
- Driver: 595.79
- Precision: fp16
- Runtime: `torch 2.10.0+cu128`, `transformers 5.3.0`, `accelerate 1.13.0`
- Benchmark harness: `scripts/benchmark_qwen_reasoning_modes.py`
- Candidate count for Ultra fan-out: 3
- Model/mode matrix:
  - `Qwen/Qwen3.5-0.8B` non-thinking
  - `Qwen/Qwen3.5-0.8B` thinking
  - `Qwen/Qwen3-1.7B` non-thinking
  - `Qwen/Qwen3-1.7B` thinking

## What Was Tested

Two GPU-only passes were run:

1. Full suite
   - 10 tasks
   - 4 math, 3 logic, 3 code
   - default token budgets from the harness
   - raw data: `reports/qwen_reasoning_modes_gpu_20260312.json`
2. Long-budget reasoning rerun
   - 7 tasks
   - math + logic only
   - doubled token budgets for both thinking and non-thinking
   - raw data: `reports/qwen_reasoning_modes_gpu_longreason_20260312.json`

Scoring:

- Single-pass: one sampled generation using the mode-specific parameters.
- Ultra fan-out: 3 sampled candidates, then selection.
- Code tasks: selected by unit-test score.
- Math/logic/text tasks: selected by normalized answer voting.

## Full Suite Results

| Condition | Single | Ultra | Math Single / Ultra | Logic Single / Ultra | Code Single / Ultra | Single Gen Time | Ultra Gen Time | Ultra Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B non-thinking | 10.0% | 10.0% | 25.0% / 25.0% | 0.0% / 0.0% | 0.0% / 0.0% | 77.8s | 234.5s | 3.01x |
| Qwen3.5-0.8B thinking | 20.0% | 30.0% | 0.0% / 0.0% | 0.0% / 0.0% | 66.7% / 100.0% | 204.4s | 605.7s | 2.96x |
| Qwen3-1.7B non-thinking | 30.0% | 30.0% | 0.0% / 0.0% | 0.0% / 0.0% | 100.0% / 100.0% | 68.7s | 192.2s | 2.80x |
| Qwen3-1.7B thinking | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 191.1s | 585.3s | 3.06x |

Aggregate across all four conditions:

- Mean single-pass accuracy: 15.0%
- Mean Ultra accuracy: 17.5%
- Mean Ultra time multiplier: 2.96x
- Mean category accuracy:
  - Math: 6.25% -> 6.25%
  - Logic: 0.0% -> 0.0%
  - Code: 41.7% -> 50.0%

Interpretation:

- The small aggregate gain came entirely from code.
- There was no measured reasoning lift on math or logic.
- `Qwen3-1.7B` non-thinking was the strongest overall condition, but only because it solved all 3 code tasks.

## Long-Budget Reasoning Rerun

This rerun was added to check whether the weak reasoning scores were mostly caused by truncated outputs.

Configuration changes:

- tasks restricted to the 7 math/logic items
- `--single-token-multiplier 2`
- `--thinking-token-multiplier 2`

Results:

| Condition | Single | Ultra | Single Gen Time | Ultra Gen Time | Ultra Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B non-thinking | 0.0% | 0.0% | 93.7s | 281.3s | 3.00x |
| Qwen3.5-0.8B thinking | 0.0% | 0.0% | 245.4s | 731.9s | 2.98x |
| Qwen3-1.7B non-thinking | 0.0% | 0.0% | 84.1s | 261.1s | 3.10x |
| Qwen3-1.7B thinking | 0.0% | 0.0% | 248.8s | 727.7s | 2.92x |

Interpretation:

- Doubling the token budget did not recover a single reasoning task.
- That rules out "short decode budget" as the main explanation for the reasoning failure pattern.
- Ultra fan-out still added nearly 3x generation cost with no reasoning gain.

## Failure Modes Observed

### 1. Qwen3-1.7B thinking never surfaced a usable final answer

In the full suite:

- `empty_cleaned_text` after stripping think blocks: 20 / 20 scored outputs
- selected think-block rate: 1.0

Meaning:

- the model stayed inside `<think>` and did not emit a post-`</think>` answer before the token budget ended
- this happened in both single-pass and Ultra selection

In the long-budget rerun:

- `empty_cleaned_text`: 12 / 14 scored outputs
- only 2 / 14 outputs even contained `</think>`

### 2. Qwen3.5-0.8B thinking produced visible reasoning text, but not strong answers

- It did not use explicit `<think>` tags in the captured outputs.
- It often produced long chain-of-thought style text and still failed to end with a correct answer.
- Its only clear advantage over non-thinking was on code generation, not reasoning.

### 3. Non-thinking modes were faster and better behaved

- Both non-thinking conditions were much more likely to emit directly usable code.
- `Qwen3-1.7B` non-thinking was the best practical mode in this benchmark.
- Even so, it still scored 0 on every math and logic task.

## Representative Examples

### Ultra-only win

Condition: `Qwen3.5-0.8B` thinking

- Task: `code_merge_intervals`
- Single-pass: failed
- Ultra fan-out: passed

This is the only clear case where the scaffold itself improved performance in the full 10-task suite.

### Thinking-mode failure

Condition: `Qwen3-1.7B` thinking

- Task: `math_tickets`
- Behavior: spent the full generation budget inside `<think>` and stopped mid-solution
- Result: no usable final answer, score `0`

This pattern repeated across all tasks in the full suite.

## Conclusion

Under this scaffold, on this GPU, with these fp16 Qwen models:

- Ultra fan-out does not improve reasoning performance in a meaningful way.
- The only measurable gain was a narrow code-selection benefit for `Qwen3.5-0.8B` thinking.
- For math and logic, the answer is no: the scaffold did not improve performance.
- For code, the answer is weakly yes in one condition, but the gain is small relative to the nearly 3x generation cost.

Best practical mode from this test:

- `Qwen/Qwen3-1.7B` non-thinking if the target is lightweight code generation.

Worst mode from this test:

- `Qwen/Qwen3-1.7B` thinking under this scaffold, because it consumed large budgets inside the think block and never surfaced answers consistently.

## Artifacts

- Harness: `scripts/benchmark_qwen_reasoning_modes.py`
- Full suite raw JSON: `reports/qwen_reasoning_modes_gpu_20260312.json`
- Long-budget reasoning raw JSON: `reports/qwen_reasoning_modes_gpu_longreason_20260312.json`
