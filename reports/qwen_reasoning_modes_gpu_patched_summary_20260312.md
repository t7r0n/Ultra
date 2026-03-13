# Qwen Reasoning Modes GPU Benchmark Summary (Patched Harness)

Date: 2026-03-12

## Scope

This summary covers:

- the completed patched full GPU benchmark:
  - `reports/qwen_reasoning_modes_gpu_patched_20260312.json`
- the earlier pre-patch benchmark for comparison:
  - `reports/qwen_reasoning_modes_gpu_20260312.json`
- a partial long-budget stress run that was intentionally stopped after it became clear the extra token budget was mostly adding latency rather than signal

## Hardware and Software

- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design
- VRAM at benchmark start: about 5696 MiB free out of 8192 MiB
- Driver: 595.79
- Precision: fp16
- Runtime: GPU only
- `torch`: 2.10.0+cu128
- `transformers`: 5.3.0

## Models and Conditions

- `qwen3.5_0.8b_nonthinking`
- `qwen3.5_0.8b_thinking`
- `qwen3_1.7b_nonthinking`
- `qwen3_1.7b_thinking`

## Patched Harness Changes Being Evaluated

The patched benchmark differs from the earlier run in several important ways:

- non-code Ultra selection no longer leaks benchmark labels into candidate choice
- unclosed `<think>` traces are preserved instead of being erased
- math prompts now target `\\boxed{...}`
- text and choice prompts now target JSON answers
- thinking mode uses a two-stage decode path
- explicit `</think>` stop handling is present
- answer extraction and short-text scoring are stricter and more deployable
- oracle best-of-k is now reported separately

Because of these harness changes, the patched and pre-patch numbers are useful to compare, but they are not perfectly apples-to-apples.

## Commands

Completed full patched benchmark:

```bash
HF_HOME=/tmp/hf_home PYTHONPATH=src /tmp/ultra_gpu_env/bin/python scripts/benchmark_qwen_reasoning_modes.py --device cuda --output reports/qwen_reasoning_modes_gpu_patched_20260312.json
```

Partial long-budget stress run:

```bash
HF_HOME=/tmp/hf_home PYTHONPATH=src /tmp/ultra_gpu_env/bin/python scripts/benchmark_qwen_reasoning_modes.py --device cuda --single-token-multiplier 2 --thinking-token-multiplier 2 --task math_tickets --task math_pipes --task math_divisible_by_5 --task math_mixture --task logic_knights --task logic_lineup --task logic_mixed_box --output reports/qwen_reasoning_modes_gpu_patched_longreason_20260312.json
```

The long-budget run was interrupted manually because the 0.8B thinking condition was consuming extreme time per task without producing any gains on completed math items.

## Main Results From the Completed Patched Full Benchmark

### Top-line accuracy

| Condition | Single Pass | Ultra Fan-out | Oracle Best-of-k |
| --- | ---: | ---: | ---: |
| `qwen3.5_0.8b_nonthinking` | 0.0 | 0.3 | 0.5 |
| `qwen3.5_0.8b_thinking` | 0.1 | 0.2 | 0.2 |
| `qwen3_1.7b_nonthinking` | 0.4 | 0.4 | 0.5 |
| `qwen3_1.7b_thinking` | 0.0 | 0.0 | 0.0 |

### Category accuracy

#### `qwen3.5_0.8b_nonthinking`

- math: `0.00 -> 0.00`
- logic: `0.00 -> 0.67`
- code: `0.00 -> 0.33`

Ultra-only recoveries:

- `logic_knights`
- `logic_mixed_box`
- `code_first_unique_char`

#### `qwen3.5_0.8b_thinking`

- math: `0.00 -> 0.00`
- logic: `0.00 -> 0.00`
- code: `0.33 -> 0.67`

Ultra-only recovery:

- `code_normalize_spaces`

Also solved in both modes:

- `code_first_unique_char`

#### `qwen3_1.7b_nonthinking`

- math: `0.00 -> 0.00`
- logic: `0.33 -> 0.33`
- code: `1.00 -> 1.00`

Solved in both modes:

- `logic_lineup`
- `code_normalize_spaces`
- `code_merge_intervals`
- `code_first_unique_char`

No Ultra-only gains.

#### `qwen3_1.7b_thinking`

- math: `0.00 -> 0.00`
- logic: `0.00 -> 0.00`
- code: `0.00 -> 0.00`

No gains anywhere.

## Thinking-Mode Diagnostics

### Selected think closure and answer-stage rates

| Condition | Think Closed Rate | Answer Stage Reached Rate | Truncated Inside Think Rate |
| --- | ---: | ---: | ---: |
| `qwen3.5_0.8b_nonthinking` | 0.0 | 1.0 | 0.0 |
| `qwen3.5_0.8b_thinking` | 0.0 | 0.0 | 1.0 |
| `qwen3_1.7b_nonthinking` | 0.0 | 1.0 | 0.0 |
| `qwen3_1.7b_thinking` | 0.0 | 0.0 | 1.0 |

Interpretation:

- both non-thinking conditions produce normal direct answers
- both thinking conditions still fail to close the think block in selected outputs
- both thinking conditions still fail to reach the answer stage at all in selected outputs
- the patched harness confirms the failure mode instead of hiding it

This is an important result: the explicit stop handling is now present and correct, but the models often do not emit `</think>` within budget, so the answer stage never begins.

## Generation Cost

The Ultra fan-out path still costs about five times as much total generation time as single-pass decoding.

| Condition | Single Total Generation Time (s) | Ultra Total Generation Time (s) | Ultra / Single |
| --- | ---: | ---: | ---: |
| `qwen3.5_0.8b_nonthinking` | 56.02 | 303.92 | 5.42x |
| `qwen3.5_0.8b_thinking` | 143.82 | 733.13 | 5.10x |
| `qwen3_1.7b_nonthinking` | 49.95 | 276.60 | 5.54x |
| `qwen3_1.7b_thinking` | 143.68 | 706.10 | 4.91x |

Practical reading:

- Ultra fan-out remains expensive
- thinking mode remains dramatically slower than non-thinking mode
- long internal traces are the main reason the thinking conditions are poor fits for this 8 GB GPU

## Comparison Against the Earlier Pre-Patch Benchmark

Earlier pre-patch top-line results:

| Condition | Old Single | Old Ultra |
| --- | ---: | ---: |
| `qwen3.5_0.8b_nonthinking` | 0.1 | 0.1 |
| `qwen3.5_0.8b_thinking` | 0.2 | 0.3 |
| `qwen3_1.7b_nonthinking` | 0.3 | 0.3 |
| `qwen3_1.7b_thinking` | 0.0 | 0.0 |

Patched top-line results:

| Condition | New Single | New Ultra |
| --- | ---: | ---: |
| `qwen3.5_0.8b_nonthinking` | 0.0 | 0.3 |
| `qwen3.5_0.8b_thinking` | 0.1 | 0.2 |
| `qwen3_1.7b_nonthinking` | 0.4 | 0.4 |
| `qwen3_1.7b_thinking` | 0.0 | 0.0 |

Important caveat:

- the patched numbers are more trustworthy
- the pre-patch run had label leakage in non-code selection and weaker handling of broken thinking traces
- the patched decline in some single-pass numbers is partly the result of stricter scoring and more honest extraction

What changed in practice:

- `qwen3.5_0.8b_nonthinking` now shows real Ultra upside on short logic and one code task
- `qwen3.5_0.8b_thinking` looks weaker after the harness became stricter
- `qwen3_1.7b_nonthinking` remains the best balanced condition overall, but Ultra still adds no top-line gain
- `qwen3_1.7b_thinking` remains a complete failure case

## Partial Long-Budget Stress Observations

The long-budget stress run doubled both single-pass and thinking token budgets on the 7 reasoning tasks only.

Completed observations before interruption:

### `qwen3.5_0.8b_nonthinking`

- `math_tickets`: `0.00 -> 0.00`
- `math_pipes`: `0.00 -> 0.00`
- `math_divisible_by_5`: `0.00 -> 0.00`
- `math_mixture`: `0.00 -> 0.00`
- `logic_knights`: `0.00 -> 1.00`
- `logic_lineup`: `0.00 -> 0.00`
- `logic_mixed_box`: `0.00 -> 1.00`

This matches the completed full benchmark pattern: logic-only upside, no math improvement.

### `qwen3.5_0.8b_thinking` partial

- `math_tickets`: `0.00 -> 0.00`
- `math_pipes`: `0.00 -> 0.00`
- `math_divisible_by_5`: `0.00 -> 0.00`
- `math_mixture`: `0.00 -> 0.00`
- `logic_knights`: `0.00 -> 0.00`

The run was interrupted at this point because time per task had become extreme and there was still no improvement.

Conclusion from the partial long-budget stress run:

- more tokens did not rescue math
- more tokens did not rescue 0.8B thinking
- more tokens appear to mostly amplify latency on this GPU

## Bottom Line

The patched harness is better and the results are more trustworthy now. It is doing a better job of measuring real behavior.

However:

- the patch did not unlock broad reasoning gains
- `qwen3_1.7b_thinking` is still unusable in this setup
- `qwen3.5_0.8b_thinking` is still too slow and too weak to justify on 8 GB hardware
- the only meaningful Ultra gains are narrow:
  - short logic recoveries for `qwen3.5_0.8b_nonthinking`
  - occasional code recoveries for the 0.8B models

Best practical mode from the patched benchmark:

- `qwen3_1.7b_nonthinking` for the strongest baseline quality
- `qwen3.5_0.8b_nonthinking` if you specifically want cheaper logic fan-out experiments

Worst practical mode:

- `qwen3_1.7b_thinking`

## Recommendation

If the goal is maximum real performance on RTX 20-series 8 GB hardware, the next work should focus on:

1. non-thinking strong baselines first
2. task-specific structured outputs plus verification
3. smarter candidate generation or verifier design instead of simply giving thinking mode more tokens
4. avoiding long-budget thinking mode on 8 GB cards unless a different decoding/control strategy proves it can actually reach answer stage
