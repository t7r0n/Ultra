# ULTRA

ULTRA is a high-performance orchestration layer for open-source LLMs.  
It is built to turn raw OSS models into production-grade systems through backend portability, task-aware generation policies, verifier-based selection, and deep run telemetry.

## Why ULTRA

- Model-agnostic runtime for Qwen, Mistral, Llama, and related OSS families
- Backend portability across HF Transformers, vLLM, SGLang, and llama.cpp
- Task-specific quality controls for code, logic, structured output, and evaluation
- Operational visibility with prompts, candidates, selection traces, and metrics artifacts

## Performance Highlights

Measured benchmark results (RTX 2070 8GB, fp16, GPU runs):

- **Code category accuracy improved from 75% to 100%** on the ULTRA scaffold benchmark (`+25 points`)
- **Qwen2.5-0.5B overall accuracy improved from 20% to 30%** with fan-out + selection
- **Logic recovery improved from 0% to 67%** for Qwen3.5-0.8B non-thinking in patched reasoning benchmarks
- **Code accuracy improved from 33% to 67%** for Qwen3.5-0.8B thinking mode

Reference artifacts:
- `reports/ultra_scaffold_benchmark_gpu_20260312.json`
- `reports/qwen_reasoning_modes_gpu_patched_20260312.json`
- `reports/qwen_reasoning_modes_gpu_patched_summary_20260312.md`

## Product Positioning

ULTRA is positioned as a quality-per-compute amplifier:
- push open-source models closer to frontier-grade outcomes on verifiable workloads
- standardize quality policy across GPU generations (RTX 20/30/40/50 and beyond)
- give teams one professional control plane for performance, reliability, and scale

## Current Repository Scope

- `doctor`, `inspect`, and `estimate` are operational
- `fetch` supports local paths and HF snapshot workflows
- `chat`, `agent`, and `eval` implement spec-aligned runtime scaffolding and artifacts
- Product architecture and implementation plan are documented in `Plan.md`

## Quick Start

```bash
python -m ultra --help
```

or, after installation:

```bash
ultra --help
```
