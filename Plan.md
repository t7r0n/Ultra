Below is a production‑grade, **implementation‑ready** Markdown specification for the CLI you described. It defines the exact behavior, file layout, configuration schema, algorithms, and acceptance tests needed for an engineering agent to implement **Ultra Mode** end‑to‑end without guesswork.

---

# ULTRA: An Extreme‑Mode Inference & Evaluation Orchestrator for Open‑Source LLMs

**Goal:** Max out *quality* (not throughput) for any open‑source LLM—text‑only or multimodal—*without training*. ULTRA performs model introspection, selects the right backend and inference params, fans out diverse candidates, *selects* and *verifies* the best outputs, and supports deep evals and agentic coding.

**Primary constraints**

* No model training or finetuning.
* Unlimited test‑time compute allowed.
* Highest precision feasible is preferred over speed.
* Works with Hugging Face Hub models (transformers safetensors) and GGUF/`llama.cpp`.
* Supports Transformer backends (Transformers), **vLLM**, **SGLang** (optional), and **llama.cpp** (optional).

---

## Table of Contents

1. [Architecture](#architecture)
2. [CLI Overview](#cli-overview)
3. [Install & Runtime Requirements](#install--runtime-requirements)
4. [Model Discovery & Introspection (MCD)](#model-discovery--introspection-mcd)
5. [Backend Selection](#backend-selection)
6. [Ultra Mode Inference Pipeline](#ultra-mode-inference-pipeline)
7. [Long‑Context Handling](#longcontext-handling)
8. [Structured Outputs (JSON/Grammar)](#structured-outputs-jsongrammar)
9. [Agentic Coding Mode](#agentic-coding-mode)
10. [Evaluation Mode](#evaluation-mode)
11. [Resource & Memory Estimator](#resource--memory-estimator)
12. [Determinism & Reproducibility](#determinism--reproducibility)
13. [Config Files](#config-files)
14. [Telemetry & Artifacts](#telemetry--artifacts)
15. [Security & Sandboxing](#security--sandboxing)
16. [Directory Layout](#directory-layout)
17. [Acceptance Tests](#acceptance-tests)
18. [Appendix: Param Mappings](#appendix-param-mappings)
19. [Appendix: References](#appendix-references)

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ ULTRA CLI (Python 3.10+ / Typer)                                   │
│  ├─ doctor        (hardware+deps check)                            │
│  ├─ fetch         (HF/GGUF download + verify)                      │
│  ├─ inspect       (MCD: config, chat template, tokenizer, dtype)   │
│  ├─ chat          (interactive; Ultra pipeline)                    │
│  ├─ agent         (agentic coding loop + tool sandbox)             │
│  ├─ eval          (bench harnesses + Ultra inference wrapper)      │
│  └─ estimate      (VRAM/RAM/throughput at target settings)         │
└────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐   ┌─────────────────────────────────┐
│  Backends                    │   │  Utilities                      │
│  • Transformers (HF)         │   │  • KV cache estimator           │
│  • vLLM (OpenAI‑compat)      │   │  • Prompt chat templater        │
│  • SGLang (optional)         │   │  • Verifiers (tests/linters)    │
│  • llama.cpp (GGUF)          │   │  • Judge / Self‑certainty       │
└──────────────────────────────┘   │  • Artifact logger (JSONL, Parq)│
                                   └─────────────────────────────────┘
```

---

## CLI Overview

### Commands & flags

```bash
ultra doctor [--json]
ultra fetch <hf_repo_or_gguf_path> [--revision REV] [--local-dir PATH] [--trust-remote-code]
ultra inspect <model_ref> [--backend auto|vllm|hf|sglang|llamacpp] [--json]
ultra estimate <model_ref> [--ctx N] [--batch B] [--precision auto|fp16|bf16|fp32] [--backend ...]
ultra chat <model_ref> [--backend ...] [--ultra-profile default|reasoning|coding|creative]
                  [--config ultra.yaml] [--logdir RUN_DIR]
ultra agent <model_ref> [--backend ...] [--open-terminal] [--sandbox docker|firejail]
ultra eval  <model_ref> [--suite harness|code|long|arena|rag]
                  [--tasks ...] [--backend ...] [--out results.json]
```

* **`model_ref`**: HF repo id (`Qwen/Qwen2.5-7B-Instruct`) or local path. GGUF paths auto‑route to `llama.cpp`.
* **`--backend auto`**: Uses rules in [Backend Selection](#backend-selection).
* **`--ultra-profile`**: Preset decoders & selectors (see [Ultra Mode](#ultra-mode-inference-pipeline)).

---

## Install & Runtime Requirements

* Python **3.10–3.12**.
* CUDA 12.x + PyTorch (if NVIDIA) or ROCm (AMD) where relevant.
* Packages (subset): `torch`, `transformers`, `huggingface_hub`, `accelerate`, `tokenizers`,
  `vllm`, `sglang` (optional), `llama_cpp_python` (optional), `flash-attn` (optional for Mistral),
  `pynvml` for GPU checks, `numba`, `uvloop`, `orjson`, `typer`.

**Why specific libs?**

* HF `GenerationConfig` & `apply_chat_template` ensure model‑native formatting; vLLM exposes OpenAI‑compatible API, SamplingParams, and GPU pre‑allocation knobs for KV cache; both are well‑documented and stable. ([Hugging Face][1])

---

## Model Discovery & Introspection (MCD)

**Inputs and outputs**

* **Input**: `model_ref` (HF id or local dir).
* **Output (JSON)**: { `arch`, `dtype_candidates`, `context_window`, `sliding_window` (if any), `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`, `hidden_size`, `chat_template`, `rope_scaling` (if any), `tokenizer_info`, `vision_support`, `stop_tokens`, `eos_tokens`, `system_prompt_hint`, `backends_supported` }

**Procedure**

1. **Fetch** (if remote): Use `huggingface_hub.snapshot_download()`; acquire `config.json`, tokenizer files, and `generation_config.json`. ([Hugging Face][2])
2. **Parse config**: Read `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` (GQA), `hidden_size` → derive `head_dim = hidden_size / num_attention_heads`. (GQA uses `num_key_value_heads` in KV sizing.) ([Hugging Face][3])
3. **Chat template**: Load tokenizer, call `tokenizer.apply_chat_template` with `add_generation_prompt=True` to extract canonical chat format. ([Hugging Face][4])
4. **RoPE/long context**:

   * If **Qwen**/**Qwen3**: read `rope_scaling` fields (`type`, `factor`, `original_max_position_embeddings`); warn if extending beyond pretrain window. ([Qwen][5])
   * If **Mistral**: record `sliding_window` (SWA). ([Hugging Face][6])
   * If **GGUF (llama.cpp)**: read GGUF metadata; note that many extended‑context GGUFs include RoPE scale in file and are auto‑applied by llama.cpp. ([Hugging Face][7])
5. **Vision**: Detect processor files (e.g., LLaVA) and mark multimodal support. ([Hugging Face][8])

---

## Backend Selection

**Rule order (`auto`)**

1. **GGUF path** → `llama.cpp` backend (best GGUF support). ([Llama CPP Python][9])
2. **Very large batch or heavy long‑context** with GPU → prefer **vLLM** (PagedAttention, KV pre‑allocation control). ([VLLM Documentation][10])
3. **Single‑process local** or special layers/tools → **Transformers**; enable Flash‑Attn for Mistral when available. ([Hugging Face][11])
4. **High‑throughput serving or structured decoding at scale** → **vLLM** or **SGLang**. ([VLLM Documentation][12])

---

## Ultra Mode Inference Pipeline

> Applied by `ultra chat` and internally by `ultra eval` (unless a benchmark requires fixed decoding).

### 0) Precision policy (quality first)

* Prefer **bf16/fp16 weights** (per model default). For math‑sensitive tasks or if VRAM permits, allow **fp32 weights**. Set `torch.set_float32_matmul_precision("highest")` (deterministic + max numeric fidelity). ([docs.pytorch.org][13])

### 1) **Fan‑out** diversified candidates (Best‑of‑N)

* Generate **N** candidates (adaptive 8→32) with diversified *temperatures*, *top‑p*, and *prompt scaffolds* (concise vs CoT), optionally ToT rollouts for puzzles. Self‑consistency is the baseline. ([arxiv.org][14])

### 2) **Selection** (rank → shortlist)

* **Hard verifiers first** (unit tests for code; math checks).
* Else **self‑consistency vote** on final answers.
* Tie‑breakers: **self‑certainty** (requires logprobs) and/or **Minimum Bayes Risk (MBR) reranking** over N hypotheses with task‑specific utility. ([arxiv.org][14])

### 3) **Synthesis / Editing**

* Run **Self‑Refine** (draft → critique → revise, 1–2 passes) on the *top‑K* candidates to merge and clean, optionally **Chain‑of‑Verification** to reduce hallucinations for factual tasks. ([arxiv.org][15])

### 4) Structured output (if requested)

* Enforce JSON/regex/grammar via backend support (see [Structured Outputs](#structured-outputs-jsongrammar)). ([VLLM Documentation][12])

**Rationale:** All three pillars—fan‑out, judge/verify, refine—are robust, widely reproduced at test time. ([arxiv.org][14])

---

## Long‑Context Handling

* **Mistral**: honor `config.sliding_window`; do *not* exceed SWA replay window when expecting precise attention. ([Hugging Face][6])
* **Qwen*/Qwen3**: apply `rope_scaling` carefully (YaRN/longrope) only when truly needed; set `factor` ≈ target_ctx / original_ctx. ([Qwen][5])
* **GGUF**: rely on file‑embedded RoPE factors where present. ([Hugging Face][7])

**Long‑context evals:** provide **LongBench v2**, **BABILong**, and **NIAH** utilities. ([github.com][16])

---

## Structured Outputs (JSON/Grammar)

* With **vLLM**, use `guided_json`, `guided_regex`, or `guided_grammar` (xgrammar) via SamplingParams/OpenAI‑compatible extra fields. ([VLLM Documentation][12])
* With Transformers or other backends, integrate **Outlines** / **LM‑Format‑Enforcer** as adapters. ([dottxt-ai.github.io][17])

---

## Agentic Coding Mode

* **Plan–Act–Check** loop with sandboxed execution:

  1. Propose tool plan (filesystem, tests, REPL).
  2. Execute in sandbox; capture logs.
  3. Verify (unit tests, static analysis).
  4. Reflexive retry (1 pass) using failure logs.
* Ultra pipeline handles code generation with strong verifiers and pass@k aggregation.

(For benchmarks, see **EvalPlus**, **LiveCodeBench**, **SWE‑bench** below.) ([github.com][18])

---

## Evaluation Mode

`ultra eval --suite ...` encapsulates **inference‑time scaling** while respecting each benchmark’s rules.

### Suites & tasks

1. **harness** (EleutherAI LM Evaluation Harness): MMLU/BBH/GPQA/IFEval etc.

   * Integrate through Harness’ provider adapter (HF/vLLM/SGLang). ([github.com][19])
2. **code** (LLM‑for‑Code): **HumanEval+ / MBPP+ (EvalPlus)**, **LiveCodeBench**, **SWE‑bench**.

   * Enforce pass@k with real execution. ([github.com][18])
3. **long**: **LongBench v2**, **BABILong**, **NIAH** generators. ([github.com][16])
4. **arena**: **Arena‑Hard‑Auto** (LLM‑as‑judge with bias mitigations—randomized order, blind IDs). ([github.com][20])
5. **rag**: **Ragas** metrics (faithfulness, answer relevancy, context precision/recall). ([docs.ragas.io][21])

**LLM‑as‑Judge caveats**: implement position‑bias defenses (shuffle, pairwise). ([arxiv.org][22])

---

## Resource & Memory Estimator

**Purpose:** Before running, show **VRAM/RAM** needed for chosen `ctx`, `batch`, precision, and backend.

**Formulas**

* **Weights (rough)**: `params * bytes_per_param` (no optimizer, inference only).
* **KV cache per token (bytes)**:

  ```
  kv_bytes_per_token = 2 (K,V) * num_layers * num_kv_heads * head_dim * bytes_per_value
  ```

  (Use **`num_key_value_heads`** for GQA; else use `num_attention_heads`.) ([Hugging Face][3])
* **Total KV**: `kv_bytes_per_token * (prompt_len + avg_generated) * batch_size`.
* **vLLM pre‑allocation**: multiply KV pool by `gpu_memory_utilization`. ([VLLM Documentation][10])

*Note*: Some vendor docs write the formula with `num_heads`; our estimator prefers `num_kv_heads` when present (GQA). Use HF config values. ([Hugging Face][3])

---

## Determinism & Reproducibility

* For **vLLM**, set per‑request `seed` and note reproducibility caveats (identical schedule). ([VLLM Documentation][23])
* For **PyTorch** backends, set:

  ```python
  torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.set_float32_matmul_precision("highest")
  ```

  (CUDNN determinism tradeoffs apply.) ([docs.pytorch.org][24])

---

## Config Files

### `ultra.yaml` (global run config)

```yaml
backend: auto            # auto|vllm|hf|sglang|llamacpp
precision: auto          # auto|fp16|bf16|fp32
context_window: auto     # int or auto
ultra:
  n_candidates: 8
  max_n_candidates: 32
  diversity:
    temperatures: [0.2, 0.6, 0.9]
    top_p: [0.85, 0.95]
    styles: ["concise", "cot"]
  selection:
    self_consistency: true
    mbr: true
    logprobs: true
  refine:
    self_refine_passes: 1
    chain_of_verification: true
structured_output:
  enabled: false
  json_schema: null
long_context:
  allow_rope_scaling: true
  mistral_respect_swa: true
judge:
  enabled: false
  bias_mitigation:
    shuffle: true
    blind_ids: true
logging:
  save_prompts: true
  save_candidates: true
  save_logits: false
```

---

## Telemetry & Artifacts

* Every `ultra chat/agent/eval` run produces:

  * `run.json`: full config and hardware snapshot.
  * `prompts.jsonl`: prompt + chat template + seed.
  * `candidates.jsonl`: all N candidates + logprobs (if available).
  * `selection.json`: scores (verifier, vote, MBR).
  * `final.txt` and (if structured) `final.json`.
  * `metrics.json`: latency, peak KV blocks, tokens/s (backend‑specific metrics visible in vLLM). ([VLLM Documentation][25])

---

## Security & Sandboxing

* Agentic code execution runs in `docker` (default) or `firejail`. No outbound network unless explicitly allowed.
* File writes constrained to a run‑scoped working dir.
* Secrets are never echoed into prompts.

---

## Directory Layout

```
ultra/
  __main__.py
  cli.py
  backends/
    hf_engine.py
    vllm_engine.py
    sglang_engine.py
    llama_cpp_engine.py
  mcd/
    inspector.py
  ultra_mode/
    fanout.py
    selection.py
    refine.py
    structured.py
  eval/
    harness.py
    code.py
    longctx.py
    arena.py
    rag.py
  tools/
    sandbox.py
    verifiers.py
    estimator.py
    hwcheck.py
    logging.py
  assets/examples/
tests/
  test_mcd.py
  test_estimator.py
  test_templates.py
```

---

## Acceptance Tests

1. **MCD correctness (HF)**

   * Given `Qwen/Qwen2.5-7B-Instruct`, `inspect` returns `num_kv_heads` and `rope_scaling` fields when present; `apply_chat_template` produces nonempty system+user template. ([Hugging Face][4])
2. **MCD correctness (GGUF)**

   * Given GGUF with extended context, `inspect` reports `context_window >= 16k` and `backend=llama.cpp` with auto RoPE noted. ([Hugging Face][7])
3. **Estimator sanity**

   * For model config (`L=32, H=32, kv_heads=8, head_dim=128, fp16`), `estimate --ctx 8192 --batch 1` matches formula within ±3%. ([Hugging Face][3])
4. **vLLM structured output**

   * With `--structured_output.enabled true` and a JSON schema, OpenAI‑compatible call injects `guided_json` and output `final.json` parses. ([VLLM Documentation][12])
5. **Ultra pipeline win**

   * On a reasoning task set (e.g., GSM8K subset), `n_candidates=8` + self‑consistency beats greedy baseline by ≥X% (configurable). ([arxiv.org][14])
6. **Long‑context limits respected**

   * For Mistral models, do not exceed `sliding_window` in attention; observed answer quality does not degrade due to unintended truncation. ([Hugging Face][6])
7. **Eval integration (code)**

   * `ultra eval --suite code --tasks humaneval_plus` runs tests with **EvalPlus** harness and writes `pass@1`. ([github.com][18])
8. **Arena‑Hard‑Auto**

   * `ultra eval --suite arena` produces judge logs with shuffled candidate order and blind IDs. ([github.com][20])
9. **RAG metrics**

   * `ultra eval --suite rag` computes faithfulness and context precision/recall via **Ragas** and writes `metrics.json`. ([docs.ragas.io][21])

---

## Appendix: Param Mappings

### A) Transformers (HF) → `GenerationConfig`

* Load model/tokenizer; merge `generation_config.json` with CLI overrides (temperature, top_p, top_k, repetition_penalty, max_new_tokens, stop). ([Hugging Face][1])
* Use `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` to format chat. ([Hugging Face][4])

### B) vLLM `SamplingParams` (Python or OpenAI‑compat)

* Map: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `repetition_penalty`, `seed`, plus **Structured Output**: `guided_json`/`guided_regex`/`guided_grammar`. Also expose `gpu_memory_utilization` as engine arg for KV pool. ([VLLM Documentation][26])
* For serving via OpenAI‑compat, ULTRA sets `base_url` to backend server and forwards extras. ([VLLM Documentation][27])

### C) SGLang (optional)

* Map generation params as per SGLang docs; where naming diverges, include adapter (temperature/top_p/top_k/max_new_tokens/repetition penalties). ([docs.sglang.ai][28])

### D) `llama.cpp` (GGUF)

* Limited param set: `temperature`, `top_p`, `top_k`, `repeat_penalty`, etc.; rely on GGUF metadata for RoPE/ctx. ([tfwol.github.io][29])

---

## Appendix: References

* **HF chat templates & generation config** (apply_chat_template; GenerationConfig; generation docs). ([Hugging Face][4])
* **HF Hub APIs** (`snapshot_download`, metadata). ([Hugging Face][2])
* **vLLM** (SamplingParams; OpenAI‑compat; GPU memory utilization; metrics; structured outputs). ([VLLM Documentation][26])
* **SGLang** docs (sampling parameters). ([docs.sglang.ai][28])
* **llama.cpp & GGUF** (auto RoPE factors). ([Hugging Face][7])
* **RoPE scaling & SWA** (Qwen guidance; Mistral sliding window). ([Qwen][5])
* **KV cache formula (with GQA)** (HF blog). ([Hugging Face][3])
* **Ultra‑mode components**: self‑consistency; ToT; Self‑Refine; CoVe; MBR. ([arxiv.org][14])
* **Evaluations**: LM Harness; EvalPlus (HumanEval+, MBPP+); LiveCodeBench; SWE‑bench; LongBench v2; NIAH; Arena‑Hard‑Auto; Ragas. ([github.com][19])
* **Determinism**: vLLM reproducibility; PyTorch randomness notes. ([VLLM Documentation][23])

---

# Implementation Details (for the agent)

Below are precise, code‑level steps and invariants. No heuristics, no TODOs.

## 1) Doctor

* Detect GPUs via **NVML** (`pynvml`), enumerate `name`, `total_mem`, driver version.
* Print CPU, RAM, OS, Python, CUDA/ROCm, PyTorch/Transformers/vLLM versions.
* If NVIDIA: `nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv`. ([docs.nvidia.com][30])
* Emit JSON if `--json`.

## 2) Fetch

* If `model_ref` matches `*/.*` or `org/model`: use `snapshot_download()` with `allow_patterns` for weights/config/tokenizer. Verify SHA256. ([Hugging Face][2])
* If path ends with `.gguf` or dir contains `.gguf`: mark backend `llama.cpp`.

## 3) Inspect

* Parse `config.json`; compute `head_dim`, detect `num_key_value_heads` (fallback to `num_attention_heads`).
* Load tokenizer, extract `chat_template`. ([Hugging Face][4])
* For Qwen*/Qwen3: report `rope_scaling`. For Mistral: `sliding_window`. For GGUF: check presence of embedded RoPE scale (log note). ([Qwen][5])

## 4) Estimate

* Inputs: `ctx`, `batch`, `precision` bytes (`fp16/bf16=2`, `fp32=4`), config fields.
* Compute KV bytes as formula in [Resource Estimator](#resource--memory-estimator). If backend is vLLM, inflate by `gpu_memory_utilization`. ([VLLM Documentation][10])
* Print a table: Weights, KV total, Headroom (device_total – reserved – est).

## 5) Chat

* Build **ULTRA profile** from `ultra.yaml`.
* Apply **chat template** and system prompt. ([Hugging Face][4])
* **Fan‑out**: Spawn workers (thread or async) to call backend with **N** diversified SamplingParams.

  * Transformers: `generate()` with `GenerationConfig` overrides. ([Hugging Face][1])
  * vLLM: `SamplingParams` / OpenAI‑compat `extra_body`. ([VLLM Documentation][26])
* **Selection**:

  * If code: run unit tests in sandbox; compute pass@k.
  * Else: self‑consistency vote by normalized majority; optional MBR score with task‑specific utility. ([arxiv.org][31])
* **Refine**: run one **Self‑Refine** pass on top‑K, optionally **CoVe** verify. ([arxiv.org][15])
* **Output**: final answer; optionally structured JSON via guided decoding. ([VLLM Documentation][12])

## 6) Agent

* Enable tools: FS, shell, python exec. If `--open-terminal`, render shell stream.
* Enforce sandbox profile (no network unless `--allow-net`).
* Same Ultra pipeline, but verifiers include tests/linters/formatters.

## 7) Eval

* **Harness**: Use official **LM Evaluation Harness** runner; plug ULTRA as a provider (HF or OpenAI‑compat for vLLM/SGLang). ([github.com][19])
* **Code**: **EvalPlus** (HumanEval+/MBPP+); **LiveCodeBench**; **SWE‑bench** wrappers:

  * Must run tests in isolation per task with timeouts.
  * Store pass@k, logs, patches (SWE‑bench). ([github.com][18])
* **Long**: runners for **LongBench v2**, **BABILong**, **NIAH**. ([github.com][16])
* **Arena**: **Arena‑Hard‑Auto**: produce pairwise judged scores; randomize candidate order; blind IDs. ([github.com][20])
* **RAG**: **Ragas**: compute faithfulness / answer‑relevancy / context precision/recall. ([docs.ragas.io][21])

---

## Notes & Guardrails

* **Precision**: default to model‑native (`auto`) and raise only when VRAM allows; keep `torch.set_float32_matmul_precision("highest")` for numerics. ([docs.pytorch.org][13])
* **Long context**:

  * For **Mistral**, exceeding `sliding_window` silently truncates effective attention—keep it within bounds. ([Hugging Face][6])
  * For **Qwen** families, prefer `rope_scaling` only when contexts significantly exceed `original_max_position_embeddings`. ([Qwen][5])
* **Structured decoding** may have backend‑specific edge cases; prefer vLLM xgrammar path. ([VLLM Documentation][12])

---

## Why these design choices?

* **Chat template fidelity** prevents formatting errors (system/assistant tokens differ by model). ([Hugging Face][4])
* **PagedAttention & pre‑allocation control (vLLM)** give predictable KV capacity for long contexts & large fans‑out. ([VLLM Documentation][10])
* **Self‑consistency / ToT / Self‑Refine / CoVe / MBR** are the most consistent test‑time quality levers without retraining. ([arxiv.org][14])
* **Bench coverage** mirrors current community standards (Harness, EvalPlus, LiveCodeBench, SWE‑bench, LongBench v2, Arena‑Hard‑Auto, Ragas). ([github.com][19])

---

### End of spec

If you want, I can also generate the initial project scaffold (files/folders, Typer CLI, and minimal backend adapters) directly from this spec.

[1]: https://huggingface.co/docs/transformers/en/main_classes/text_generation?utm_source=chatgpt.com "Generation"
[2]: https://huggingface.co/docs/huggingface_hub/en/guides/download?utm_source=chatgpt.com "Download files from the Hub"
[3]: https://huggingface.co/blog/kv-cache-quantization?utm_source=chatgpt.com "Unlocking Longer Generation with Key-Value Cache ..."
[4]: https://huggingface.co/docs/transformers/en/chat_templating?utm_source=chatgpt.com "Chat templates"
[5]: https://qwen.readthedocs.io/en/latest/inference/transformers.html?utm_source=chatgpt.com "Transformers - Qwen"
[6]: https://huggingface.co/docs/transformers/en/model_doc/mistral?utm_source=chatgpt.com "Mistral"
[7]: https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF?utm_source=chatgpt.com "TheBloke/openchat-3.5-1210-GGUF"
[8]: https://huggingface.co/docs/transformers/en/model_doc/llava?utm_source=chatgpt.com "LLaVa"
[9]: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/?utm_source=chatgpt.com "API Reference"
[10]: https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=chatgpt.com "Paged Attention - vLLM"
[11]: https://huggingface.co/docs/transformers/v4.34.0/en/model_doc/mistral?utm_source=chatgpt.com "Mistral"
[12]: https://docs.vllm.ai/en/v0.9.2/features/structured_outputs.html?utm_source=chatgpt.com "Structured Outputs - vLLM"
[13]: https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html?utm_source=chatgpt.com "torch.set_float32_matmul_precision"
[14]: https://arxiv.org/abs/2203.11171?utm_source=chatgpt.com "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
[15]: https://arxiv.org/abs/2303.17651?utm_source=chatgpt.com "Self-Refine: Iterative Refinement with Self-Feedback"
[16]: https://github.com/THUDM/LongBench?utm_source=chatgpt.com "LongBench v2 and LongBench (ACL 25'&24')"
[17]: https://dottxt-ai.github.io/outlines/?utm_source=chatgpt.com "Outlines"
[18]: https://github.com/evalplus/evalplus?utm_source=chatgpt.com "evalplus/evalplus: Rigourous evaluation of LLM- ..."
[19]: https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com "EleutherAI/lm-evaluation-harness: A framework for few- ..."
[20]: https://github.com/lmarena/arena-hard-auto?utm_source=chatgpt.com "Arena-Hard-Auto: An automatic LLM benchmark."
[21]: https://docs.ragas.io/en/v0.1.21/concepts/metrics/?utm_source=chatgpt.com "Metrics"
[22]: https://arxiv.org/abs/2306.05685?utm_source=chatgpt.com "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
[23]: https://docs.vllm.ai/en/v0.9.1/usage/reproducibility.html?utm_source=chatgpt.com "Reproducibility - vLLM"
[24]: https://docs.pytorch.org/docs/stable/notes/randomness.html?utm_source=chatgpt.com "Reproducibility — PyTorch 2.9 documentation"
[25]: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com "Metrics - vLLM"
[26]: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html?utm_source=chatgpt.com "Sampling Parameters - vLLM"
[27]: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?utm_source=chatgpt.com "OpenAI-Compatible Server - vLLM"
[28]: https://docs.sglang.ai/?utm_source=chatgpt.com "SGLang Documentation — SGLang"
[29]: https://tfwol.github.io/text-generation-webui/Generation-parameters.html?utm_source=chatgpt.com "Generation parameters | text-generation-webui - GitHub Pages"
[30]: https://docs.nvidia.com/deploy/nvidia-smi/index.html?utm_source=chatgpt.com "Nvidia-smi Manual"
[31]: https://arxiv.org/abs/2311.05263?utm_source=chatgpt.com "Model-Based Minimum Bayes Risk Decoding for Text ..."

