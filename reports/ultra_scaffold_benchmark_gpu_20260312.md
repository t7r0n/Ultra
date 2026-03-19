# ULTRA Scaffold GPU Benchmark Report

Date: 2026-03-12

## Scope

This benchmark answers a narrow question honestly:

- Does the current repository, as implemented today, show a measurable quality gain from an Ultra-style fan-out loop?
- The repository still does not contain a full model-backed `ultra chat` pipeline.
- To test the core idea anyway, `scripts/benchmark_ultra_scaffold.py` was used to compare:
  - `baseline_greedy`: one deterministic generation
  - `ultra_scaffold`: 5 sampled candidates using the repo's default fan-out temperatures/styles, then simple selection:
    - majority vote for math / logic / JSON
    - unit-test-first selection for code

This is a benchmark of the scaffolded Ultra idea, not proof that the full spec is implemented.

## Environment

- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design, 8 GB VRAM
- Runtime: CUDA via `torch 2.10.0+cu128`
- Precision: fp16 on GPU
- Models:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
- Tasks: 10 exact-match tasks
  - 3 math
  - 3 logic multiple choice
  - 2 JSON extraction
  - 2 code-with-tests

Raw results: `reports/ultra_scaffold_benchmark_gpu_20260312.json`

## Accuracy

Per-model strategy accuracy:

| Model | Greedy | Ultra Scaffold | Delta |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 20% | 30% | +10 points |
| Qwen2.5-1.5B-Instruct | 60% | 60% | 0 |

Aggregate mean across both models:

| Strategy | Accuracy |
| --- | ---: |
| Greedy | 40% |
| Ultra Scaffold | 45% |

Category means across both models:

| Category | Greedy | Ultra Scaffold | Delta |
| --- | ---: | ---: | ---: |
| Code | 75% | 100% | +25 points |
| JSON | 50% | 50% | 0 |
| Logic | 33.3% | 33.3% | 0 |
| Math | 16.7% | 16.7% | 0 |

## Latency / Compute Cost

Total generation time per model:

| Model | Greedy Total | Ultra Total | Cost Multiplier |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 16.0 s | 97.9 s | 6.12x |
| Qwen2.5-1.5B-Instruct | 8.5 s | 91.0 s | 10.69x |

GPU versus the earlier CPU runs:

| Model | Greedy Speedup | Ultra Speedup |
| --- | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 1.73x | 1.60x |
| Qwen2.5-1.5B-Instruct | 5.01x | 4.72x |

The GPU makes the benchmark practical, but it does not change the quality pattern. The task scores on GPU matched the earlier CPU run.

## Representative Outcomes

One real win:

- Task: `code_normalize_spaces`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Greedy failed by returning a line-by-line strip implementation.
- Ultra Scaffold succeeded because one sampled candidate produced the correct implementation:
  - `def normalize_spaces(text): return ' '.join(text.split())`

Representative failures that fan-out did not fix:

- `math_bookstore`
  - 0.5B baseline and Ultra both answered `46`
  - 1.5B baseline answered `46`, Ultra selected `48`
  - Correct answer was `36`
- `json_customer`
  - 0.5B hallucinated nested objects and total `252`
  - 1.5B kept the right structure but returned `"$84"` instead of numeric `84`
  - Fan-out did not correct either case
- `logic_calendar`
  - both models stayed wrong under both strategies

## Conclusion

Short answer: no, not in a meaningful general way.

What the data supports:

- The simplified Ultra-style loop did help one weak code task by letting a better sampled candidate survive selection.
- It did not improve math, logic, or JSON extraction on this benchmark.
- On the stronger 1.5B model, it produced no net gain at all.
- The overall +5 point lift is fragile and comes entirely from a single task on the 0.5B model.
- The compute cost increase was large: roughly 6x to 11x more generation time.

Practical reading:

- If the task has a hard verifier, especially code tests, fan-out plus selection can help even in a scaffolded form.
- Without stronger selectors, judges, or refiners, this scaffold does not yet justify its compute cost for general reasoning or extraction work.
- The current repository should not be described as "performance-improving" yet. It shows a narrow code-task upside, not a broad quality win.

## Reproduction

GPU benchmark command:

```bash
HF_HOME=/tmp/hf_home PYTHONPATH=src /tmp/ultra_gpu_env/bin/python \
  scripts/benchmark_ultra_scaffold.py \
  --device cuda \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --n-candidates 5 \
  --output reports/ultra_scaffold_benchmark_gpu_20260312.json
```
