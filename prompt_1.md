# Request For GPT-5.4 Pro: Diagnose ULTRA Scaffold Performance And Fix The Current Benchmark / Inference Problems

Important:

- You will not have access to the repo, terminal, or any files besides this markdown content.
- Treat this markdown as the full authoritative handoff.
- If you need to reason about the implementation, use the code excerpts included below rather than assuming hidden code exists elsewhere.
- This markdown includes the benchmark setup, commands, scoring logic, observed behavior, and measured results; do not assume separate hidden benchmark context exists outside this file.

I need a deep technical diagnosis and a concrete solution plan for the current ULTRA scaffold and the benchmark results below. Please assume the goal is to make the scaffold actually improve quality, especially reasoning, on a small 8 GB GPU machine.

I want you to do all of the following:

1. Diagnose whether the current benchmark methodology is sound or flawed.
2. Diagnose whether the current prompt / decoding / answer-extraction logic is mismatched to Qwen3 and Qwen3.5 thinking-mode models.
3. Identify the most likely causes of the current poor reasoning results.
4. Propose exact code-level changes to the benchmark harness and, if appropriate, to the ULTRA scaffold design.
5. Propose an improved experiment plan that can realistically run on an RTX 2070 Max-Q 8 GB GPU.
6. Be explicit about which observed failures are likely model limitations versus harness / prompt / parsing mistakes.

Everything below is the full context from the experiments already run.

## 1. Repository State

This repository is an early scaffold for a Python CLI called `ultra`.

Important reality check:

- The repository does not yet contain a full model-backed `ultra chat` pipeline.
- The `chat`, `agent`, and `eval` CLI commands are scaffold-only and mainly initialize run artifacts with `"status": "scaffold_only"`.
- The working pieces that actually exist are mostly:
  - config loading
  - a fan-out plan builder
  - model inspection
  - a simple memory estimator
  - benchmark scripts that call Hugging Face models directly

So the benchmark results below are not from a complete production ULTRA implementation. They are from purpose-built benchmark harnesses that test the core Ultra-style idea.

Current relevant code areas:

- `src/ultra/config.py`
- `src/ultra/ultra_mode/fanout.py`
- `src/ultra/mcd/inspector.py`
- `src/ultra/tools/estimator.py`
- `src/ultra/cli.py`
- `scripts/benchmark_ultra_scaffold.py`
- `scripts/benchmark_qwen_reasoning_modes.py`

## 2. Hardware / Software Stack

Machine:

- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design
- VRAM: 8 GB
- Driver: `595.79`

GPU benchmark runtime:

- Python env: `/tmp/ultra_gpu_env`
- `torch 2.10.0+cu128`
- `transformers 5.3.0`
- `accelerate 1.13.0`
- CUDA: `12.8`
- HF cache dir: `/tmp/hf_home`

Notes:

- All benchmark runs requested here used GPU only, not CPU.
- All benchmark runs requested here used fp16, not bf16.
- PyTorch reported `torch.cuda.is_bf16_supported() == True`, but I still used fp16 for these runs.
- Hugging Face downloads were unauthenticated because no `HF_TOKEN` was set.

Qwen3.5-specific runtime warning seen repeatedly:

- "The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d"

This likely affected speed, not raw accuracy, but I want your opinion on whether it could affect generation behavior in practice.

## 2.1 Hard constraints and optimization target

These constraints are real and important:

- Target machine is an RTX 2070 Max-Q with 8 GB VRAM.
- User explicitly wants GPU-only execution, not CPU fallback.
- User explicitly requested fp16 for these experiments.
- Practical comfortable dense-model range on this machine, with current environment and HF runtime, appears to be around `0.8B` to `1.7B`.
- A dense `4B` fp16 model is likely too tight once runtime overhead and KV cache are included.
- The actual goal is not "good enough." The goal is maximum practical quality on this hardware.
- The optimization target is not limited to this exact laptop GPU. I want a solution strategy that maximizes practical performance on:
  - current machine: RTX 2070 Max-Q 8 GB
  - other RTX 20-series consumer GPUs
  - RTX 30-series consumer GPUs
  - RTX 40-series consumer GPUs
  - RTX 50-series consumer GPUs
- If the best design should adapt by VRAM tier, architecture generation, CUDA capability, or kernel support, say so explicitly and provide a tiered strategy.
- If you think the best path requires changing kernels, install deps, decode strategy, prompting, answer extraction, or even using quantization or a different serving stack, say so explicitly.
- Do not assume we can just "use a bigger model."

Observed practical runtime constraints during testing:

- Qwen3.5 used a slower fallback path because fast-path dependencies were missing.
- Long reasoning-mode decodes were expensive enough that even `n_candidates=3` already had roughly `3x` generation cost.
- The current harness does not batch candidates; it generates them sequentially.
- The current harness does not use an external judge model.
- The current harness does not use a second answer-only generation stage after thinking.

## 3. Existing ULTRA Config / Scaffold Behavior

### 3.1 Default ULTRA config relevant to fan-out

From `src/ultra/config.py`:

- `ultra.n_candidates = 8`
- `ultra.max_n_candidates = 32`
- `ultra.diversity.temperatures = [0.2, 0.6, 0.9]`
- `ultra.diversity.top_p = [0.85, 0.95]`
- `ultra.diversity.styles = ["concise", "cot"]`
- `ultra.selection.self_consistency = True`
- `ultra.selection.mbr = True`
- `ultra.selection.logprobs = True`
- `ultra.refine.self_refine_passes = 1`
- `ultra.refine.chain_of_verification = True`

Important limitation:

- The benchmark harnesses do not implement the full intended selection / judge / refine stack implied by the config.
- They mostly implement simple sampling plus simple selection.

### 3.2 What the scaffold currently does in practice

`src/ultra/ultra_mode/fanout.py` only builds a `FanoutPlan`:

- `ultra_profile`
- `n_candidates`
- `temperatures`
- `top_p`
- `styles`

It does not perform full reasoning orchestration, judging, or refinement.

### 3.3 CLI reality

From `src/ultra/cli.py`:

- `doctor`, `fetch`, `inspect`, `estimate` are functional enough to run.
- `chat`, `agent`, and `eval` are scaffold-only and do not run a real inference pipeline.

This matters because all performance experiments below bypassed the CLI and used custom benchmark scripts directly.

## 4. Benchmark History

I ran two major benchmark families, plus some small smoke checks.

### 4.1 Earlier ULTRA scaffold benchmark using Qwen2.5

Purpose:

- Test whether a simplified Ultra-style fan-out loop helps at all on small instruct models.

Models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

Precision / device:

- GPU
- fp16

Script:

- `scripts/benchmark_ultra_scaffold.py`

Report files:

- `reports/ultra_scaffold_benchmark_gpu_20260312.json`
- `reports/ultra_scaffold_benchmark_gpu_20260312.md`

Exact command:

```bash
HF_HOME=/tmp/hf_home PYTHONPATH=src /tmp/ultra_gpu_env/bin/python \
  scripts/benchmark_ultra_scaffold.py \
  --device cuda \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --n-candidates 5 \
  --output reports/ultra_scaffold_benchmark_gpu_20260312.json
```

### 4.2 New Qwen3 / Qwen3.5 benchmark requested later

Purpose:

- Test newer Qwen family models in both non-thinking and thinking modes.
- Check whether Ultra fan-out helps once thinking-mode models are used.

Models / conditions:

- `Qwen/Qwen3.5-0.8B` non-thinking
- `Qwen/Qwen3.5-0.8B` thinking
- `Qwen/Qwen3-1.7B` non-thinking
- `Qwen/Qwen3-1.7B` thinking

Precision / device:

- GPU
- fp16

Script:

- `scripts/benchmark_qwen_reasoning_modes.py`

Report files:

- `reports/qwen_reasoning_modes_gpu_20260312.json`
- `reports/qwen_reasoning_modes_gpu_longreason_20260312.json`
- `reports/qwen_reasoning_modes_gpu_qwen3_qwen3p5_20260312.md`

Main full-suite command:

```bash
HF_HOME=/tmp/hf_home PYTHONPATH=src /tmp/ultra_gpu_env/bin/python \
  scripts/benchmark_qwen_reasoning_modes.py \
  --n-candidates 3 \
  --output reports/qwen_reasoning_modes_gpu_20260312.json
```

Long-budget reasoning rerun command:

```bash
HF_HOME=/tmp/hf_home PYTHONPATH=src /tmp/ultra_gpu_env/bin/python \
  scripts/benchmark_qwen_reasoning_modes.py \
  --n-candidates 3 \
  --single-token-multiplier 2 \
  --thinking-token-multiplier 2 \
  --task math_tickets \
  --task math_pipes \
  --task math_divisible_by_5 \
  --task math_mixture \
  --task logic_knights \
  --task logic_lineup \
  --task logic_mixed_box \
  --output reports/qwen_reasoning_modes_gpu_longreason_20260312.json
```

### 4.3 Smoke checks before the full Qwen3/Qwen3.5 run

Before running the full Qwen3/Qwen3.5 benchmark, I did small smoke runs on reduced subsets to confirm:

- `Qwen3.5-0.8B` non-thinking loaded correctly on GPU
- `Qwen3.5-0.8B` thinking loaded correctly on GPU
- `enable_thinking` path was functional at least at the Transformers API level

I did not preserve separate report files for those small smoke subsets.

## 5. Exact Behavior Of `scripts/benchmark_ultra_scaffold.py`

This is the earlier Qwen2.5 benchmark harness.

### 5.1 Comparison strategies

- `baseline_greedy`
  - single deterministic generation
  - `do_sample=False`
- `ultra_scaffold`
  - multiple sampled candidates
  - simple selection only

### 5.2 Fan-out configuration actually used

This harness pulls a `FanoutPlan` from the scaffold.

For the actual GPU run:

- `n_candidates = 5`
- temperatures = `[0.2, 0.6, 0.9]`
- top_p = `[0.85, 0.95]`
- styles = `["concise", "cot"]`

Candidate loop behavior:

- candidate temperatures and top_p are repeated cyclically from the plan
- code / reasoning style depends on the plan style
- sample seed = `1000 + index`

### 5.3 Prompting behavior

For code tasks:

- concise system prompt:
  - "You are a concise Python coding assistant. Output only valid Python code."
- cot system prompt:
  - "You are a careful Python coding assistant. Think through the solution internally, but output only the final Python code."

For JSON tasks:

- "You are a precise information extraction assistant. Return valid JSON only."

For reasoning tasks:

- concise:
  - "You are a concise assistant. Give the final answer directly."
- cot:
  - "You are a reasoning assistant. Work carefully and end with a line that begins FINAL: followed by only the answer."

For cot user prompts on non-code / non-JSON tasks:

- the harness appends:
  - "Show your reasoning briefly, then end with `FINAL: <answer>`."

### 5.4 Selection logic

For code:

- choose candidate with max `(score, -latency)`
- `score` is unit-test-based

For non-code:

- majority vote over normalized canonical answers
- if multiple answers tie for majority, the first matching candidate in candidate order is returned
- this is not a strong judge

### 5.5 Evaluation logic

- math: last number in output
- multiple choice: first `[A-D]`
- JSON: first parseable JSON object found in output
- code: execute extracted code, run tests, allow partial score if some tests pass

### 5.6 Tasks in this harness

10 exact-match tasks:

- Math:
  - `math_bookstore` -> answer `36`
  - `math_train` -> answer `60`
  - `math_baker` -> answer `30`
- Logic MCQ:
  - `logic_syllogism` -> answer `B`
  - `logic_height` -> answer `C`
  - `logic_calendar` -> answer `A`
- JSON:
  - `json_customer` -> `{"customer":"Maya Chen","city":"Austin","item":"lamps","quantity":3,"total":84}`
  - `json_ticket` -> `{"ticket_id":"A19","priority":"high","assignee":"Omar","status":"open"}`
- Code:
  - `code_square_evens`
  - `code_normalize_spaces`

## 6. Results From `benchmark_ultra_scaffold.py` (Qwen2.5 GPU Run)

Per-model results:

- `Qwen/Qwen2.5-0.5B-Instruct`
  - greedy: `20%`
  - ultra_scaffold: `30%`
  - delta: `+10 points`
- `Qwen/Qwen2.5-1.5B-Instruct`
  - greedy: `60%`
  - ultra_scaffold: `60%`
  - delta: `0`

Aggregate mean across both models:

- greedy: `40%`
- ultra_scaffold: `45%`

Category means across both models:

- code: `75% -> 100%`
- JSON: `50% -> 50%`
- logic: `33.3% -> 33.3%`
- math: `16.7% -> 16.7%`

Generation-time totals:

- `Qwen2.5-0.5B-Instruct`
  - greedy total: `16.0 s`
  - ultra total: `97.9 s`
  - cost multiplier: `6.12x`
- `Qwen2.5-1.5B-Instruct`
  - greedy total: `8.5 s`
  - ultra total: `91.0 s`
  - cost multiplier: `10.69x`

Key representative outcomes:

- One real win:
  - task: `code_normalize_spaces`
  - model: `Qwen/Qwen2.5-0.5B-Instruct`
  - greedy failed
  - Ultra succeeded because one sampled candidate produced the correct implementation
- Representative failures not fixed by fan-out:
  - `math_bookstore`
  - `json_customer`
  - `logic_calendar`

Interpretation from this first benchmark:

- Ultra-style fan-out seemed to help only on a weak code task with a hard verifier.
- It did not improve math, logic, or JSON extraction.
- On the stronger 1.5B model, it produced no net gain.

## 7. Exact Behavior Of `scripts/benchmark_qwen_reasoning_modes.py`

This is the newer Qwen3 / Qwen3.5 benchmark harness.

Important difference from the earlier Qwen2.5 harness:

- `single_pass` here is not greedy.
- `single_pass` is one sampled generation using mode-specific parameters.
- `ultra_fanout` is multiple sampled generations with parameter variation plus selection.

So the Qwen3/Qwen3.5 results are not strictly apples-to-apples with the earlier greedy-vs-fanout benchmark.

### 7.1 Conditions and decoding parameters

Condition 1:

- label: `qwen3.5_0.8b_nonthinking`
- model: `Qwen/Qwen3.5-0.8B`
- `enable_thinking=False`
- text parameters:
  - temperature `1.0`
  - top_p `1.0`
  - top_k `20`
  - min_p `0.0`
  - repetition_penalty `1.0`

Condition 2:

- label: `qwen3.5_0.8b_thinking`
- model: `Qwen/Qwen3.5-0.8B`
- `enable_thinking=True`
- text parameters:
  - temperature `1.0`
  - top_p `0.95`
  - top_k `20`
  - min_p `0.0`
  - repetition_penalty `1.0`
- code override parameters:
  - temperature `0.6`
  - top_p `0.95`
  - top_k `20`
  - min_p `0.0`
  - repetition_penalty `1.0`

Condition 3:

- label: `qwen3_1.7b_nonthinking`
- model: `Qwen/Qwen3-1.7B`
- `enable_thinking=False`
- text parameters:
  - temperature `0.7`
  - top_p `0.8`
  - top_k `20`
  - min_p `0.0`
  - repetition_penalty `1.0`

Condition 4:

- label: `qwen3_1.7b_thinking`
- model: `Qwen/Qwen3-1.7B`
- `enable_thinking=True`
- text parameters:
  - temperature `0.6`
  - top_p `0.95`
  - top_k `20`
  - min_p `0.0`
  - repetition_penalty `1.0`

### 7.2 Candidate variation logic for Ultra fan-out

The `single_pass` strategy uses the base parameters above.

The `ultra_fanout` strategy varies them.

General offsets:

- temperature offsets: `[-0.15, -0.05, 0.0, 0.05, 0.15]`
- top_p offsets: `[-0.08, -0.03, 0.0, 0.02, 0.05]`
- clipped temperature range: `[0.2, 1.2]`
- clipped top_p range: `[0.6, 1.0]`

Special case:

- for Qwen3.5 thinking when base temperature `<= 0.6`
- special temperature offsets: `[0.0, 0.05, -0.05, 0.1, -0.1]`
- clipped temperature range becomes `[0.35, 0.8]`

Important:

- The actual runs used `n_candidates=3`, not 5.
- So only the first 3 offsets were actually used.

### 7.3 Prompting behavior

#### Single-pass system prompts

For code:

- "You are a careful Python coding assistant. Output only final Python code."

For non-code:

- "You are a precise reasoning assistant. Solve carefully and end with `FINAL: <answer>`."

#### Ultra fan-out styles

Possible styles:

- `default`
- `plan`
- `verify`
- `tests`
- `default`

For code tasks:

- concise single-pass:
  - "You are a careful Python coding assistant. Output only final Python code."
- Ultra style `plan`:
  - "You are a meticulous Python coding assistant. Think through edge cases, then output only final Python code."
- Ultra style `tests`:
  - "You are a Python coding assistant. Internally verify against likely edge cases, then output only final Python code."
- default Ultra code:
  - "You are a strong Python coding assistant. Output only final Python code."

For non-code reasoning:

- concise single-pass:
  - "You are a precise reasoning assistant. Solve carefully and end with `FINAL: <answer>`."
- Ultra style `plan`:
  - "You are a precise reasoning assistant. Plan briefly, solve carefully, and end with `FINAL: <answer>`."
- Ultra style `verify`:
  - "You are a precise reasoning assistant. Double-check the result before ending with `FINAL: <answer>`."
- default Ultra reasoning:
  - same as concise single-pass

Ultra user-prompt modifications:

- code style `tests`:
  - append "Make sure the function handles the implied edge cases in the docstring."
- reasoning style `plan`:
  - append "Use a short plan before the final answer."
- reasoning style `verify`:
  - append "Check your work once before the final answer."

### 7.4 `apply_chat_template` behavior

The harness uses:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=condition.thinking,
)
```

This is the central mechanism used to switch Qwen thinking mode on or off.

### 7.5 Generation behavior

Important generation settings:

- `do_sample=True` for both `single_pass` and `ultra_fanout`
- `pad_token_id=tokenizer.eos_token_id`
- no explicit stop strings
- no explicit stop-on-`</think>` logic
- no explicit separate answer-generation phase after thinking

This may be a major problem for Qwen thinking models.

### 7.6 Cleaning / parsing behavior

The harness defines:

```python
def clean_model_text(text: str) -> str:
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()
```

This is extremely important.

Implication:

- If the model emits `<think>` but never emits `</think>` before `max_new_tokens` is reached, then `clean_model_text()` strips the entire output to empty string.
- This happened heavily for `Qwen3-1.7B` thinking.

### 7.7 Answer extraction logic

For number tasks:

- `extract_final_field(cleaned_text)` or fallback `extract_last_number(cleaned_text)`

For fraction tasks:

- `extract_final_field(cleaned_text)` or fallback `extract_last_number(cleaned_text)`
- compared numerically with alias tolerance

For choice tasks:

- `extract_final_field(cleaned_text)` or fallback `extract_choice(cleaned_text)`

For text tasks:

- `normalize_text_answer(extract_final_field(cleaned_text) or cleaned_text)`
- exact normalized-string comparison against expected answer / aliases

Important implication for text tasks:

- if the model does not emit `FINAL: Mixed` and instead emits something like "The correct box is the Mixed box.", then the entire normalized sentence is compared and will fail unless it exactly matches an alias
- this may make text scoring too strict

For code tasks:

- extract fenced code or whole text
- execute code
- run unit tests
- allow partial score based on tests passed

### 7.8 Selection logic

For code:

- choose candidate with max `(score, -latency)`

For non-code:

- normalize candidate answers
- majority vote over normalized values
- among tied winners, choose max `(score, -latency)`

This is still a weak selector.

### 7.9 Tasks in this Qwen3/Qwen3.5 harness

10 tasks total:

- Math:
  - `math_tickets`
    - answer `22`
    - single tokens `96`
    - thinking tokens `256`
  - `math_pipes`
    - answer `1/3`
    - aliases `0.3333333333`, `0.333333333`, `0.3333333333333333`
    - single tokens `96`
    - thinking tokens `256`
  - `math_divisible_by_5`
    - answer `952`
    - single tokens `128`
    - thinking tokens `320`
  - `math_mixture`
    - answer `20`
    - single tokens `96`
    - thinking tokens `256`
- Logic:
  - `logic_knights`
    - answer `B`
    - single tokens `96`
    - thinking tokens `256`
  - `logic_lineup`
    - answer `Bo`
    - alias `bo`
    - single tokens `96`
    - thinking tokens `256`
  - `logic_mixed_box`
    - answer `Mixed`
    - aliases `mixed box`, `the box labeled mixed`
    - single tokens `96`
    - thinking tokens `256`
- Code:
  - `code_normalize_spaces`
    - single tokens `160`
    - thinking tokens `320`
  - `code_merge_intervals`
    - single tokens `192`
    - thinking tokens `384`
  - `code_first_unique_char`
    - single tokens `192`
    - thinking tokens `384`

## 7.10 Critical code excerpts from `scripts/benchmark_qwen_reasoning_modes.py`

These are the exact functions most likely to matter for diagnosing the current failures.

### Generation loop

```python
def generate_once(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    condition: Condition,
    task: Task,
    strategy: str,
    candidate_index: int,
    single_token_multiplier: float,
    thinking_token_multiplier: float,
) -> tuple[str, int, float]:
    params = condition.params_for_task(task)
    params = varied_params(params, strategy=strategy, condition=condition, candidate_index=candidate_index)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=condition.thinking,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    base_max_new_tokens = task.max_new_tokens_thinking if condition.thinking else task.max_new_tokens_single
    multiplier = thinking_token_multiplier if condition.thinking else single_token_multiplier
    max_new_tokens = max(1, int(round(base_max_new_tokens * multiplier)))
    set_seed(seed_for(condition.label, task.task_id, strategy, candidate_index))
    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "min_p": params["min_p"],
        "repetition_penalty": params["repetition_penalty"],
        "pad_token_id": tokenizer.eos_token_id,
    }
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    torch.cuda.synchronize()
    latency_s = time.perf_counter() - start
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    return raw_text, int(generated_ids.shape[0]), latency_s
```

### Cleaning logic

```python
def clean_model_text(text: str) -> str:
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()
```

### Scoring logic

```python
def evaluate_candidate(task: Task, raw_text: str, generated_tokens: int, latency_s: float, *, meta: dict[str, Any]) -> CandidateResult:
    cleaned_text = clean_model_text(raw_text)
    think_present = "<think>" in raw_text or "</think>" in raw_text

    if task.evaluator == "number":
        canonical = extract_final_field(cleaned_text) or extract_last_number(cleaned_text)
        score = 1.0 if canonical == task.answer else 0.0
    elif task.evaluator == "fraction":
        canonical = extract_final_field(cleaned_text) or extract_last_number(cleaned_text)
        score = 1.0 if equivalent_fraction_or_decimal(canonical, task.answer, task.aliases) else 0.0
    elif task.evaluator == "choice":
        canonical = extract_final_field(cleaned_text) or extract_choice(cleaned_text)
        score = 1.0 if canonical == task.answer else 0.0
    elif task.evaluator == "text":
        canonical = normalize_text_answer(extract_final_field(cleaned_text) or cleaned_text)
        valid_answers = {normalize_text_answer(task.answer), *(normalize_text_answer(alias) for alias in task.aliases)}
        score = 1.0 if canonical in valid_answers else 0.0
    elif task.evaluator == "code":
        canonical = extract_code(cleaned_text)
        code_eval = evaluate_code(canonical, task.answer)
        meta = {**meta, **code_eval["meta"]}
        score = code_eval["score"]
    else:
        canonical = cleaned_text
        score = 0.0

    return CandidateResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        canonical_answer=canonical,
        score=score,
        latency_s=latency_s,
        generated_tokens=generated_tokens,
        think_block_present=think_present,
        meta=meta,
    )
```

### Selection logic

```python
def select_candidate(task: Task, candidates: list[CandidateResult]) -> CandidateResult:
    if task.evaluator == "code":
        return max(candidates, key=lambda candidate: (candidate.score, -candidate.latency_s))

    normalized_values = [normalize_selection_value(candidate.canonical_answer) for candidate in candidates]
    counts = Counter(value for value in normalized_values if value)
    if not counts:
        return max(candidates, key=lambda candidate: candidate.score)
    best_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == best_count}
    shortlisted = [
        candidate
        for candidate, normalized in zip(candidates, normalized_values, strict=True)
        if normalized in winners
    ]
    return max(shortlisted, key=lambda candidate: (candidate.score, -candidate.latency_s))
```

### Prompt templates

```python
def system_prompt(task: Task, *, concise: bool, style: str = "default") -> str:
    if task.category == "code":
        if concise:
            return "You are a careful Python coding assistant. Output only final Python code."
        if style == "plan":
            return "You are a meticulous Python coding assistant. Think through edge cases, then output only final Python code."
        if style == "tests":
            return "You are a Python coding assistant. Internally verify against likely edge cases, then output only final Python code."
        return "You are a strong Python coding assistant. Output only final Python code."
    if concise:
        return "You are a precise reasoning assistant. Solve carefully and end with `FINAL: <answer>`."
    if style == "plan":
        return "You are a precise reasoning assistant. Plan briefly, solve carefully, and end with `FINAL: <answer>`."
    if style == "verify":
        return "You are a precise reasoning assistant. Double-check the result before ending with `FINAL: <answer>`."
    return "You are a precise reasoning assistant. Solve carefully and end with `FINAL: <answer>`."


def ultra_user_prompt(task: Task, *, style: str) -> str:
    if task.category == "code":
        if style == "tests":
            return task.prompt + "\nMake sure the function handles the implied edge cases in the docstring."
        return task.prompt
    if style == "plan":
        return task.prompt + "\nUse a short plan before the final answer."
    if style == "verify":
        return task.prompt + "\nCheck your work once before the final answer."
    return task.prompt
```

### Parameter variation

```python
def varied_params(base: dict[str, Any], *, strategy: str, condition: Condition, candidate_index: int) -> dict[str, Any]:
    if strategy == "single_pass":
        return base

    temp_offsets = [-0.15, -0.05, 0.0, 0.05, 0.15]
    top_p_offsets = [-0.08, -0.03, 0.0, 0.02, 0.05]
    idx = candidate_index % len(temp_offsets)
    varied = dict(base)
    varied["temperature"] = clamp(base["temperature"] + temp_offsets[idx], 0.2, 1.2)
    varied["top_p"] = clamp(base["top_p"] + top_p_offsets[idx], 0.6, 1.0)
    if condition.family == "qwen3.5" and condition.thinking and base["temperature"] <= 0.6:
        varied["temperature"] = clamp(base["temperature"] + [0.0, 0.05, -0.05, 0.1, -0.1][idx], 0.35, 0.8)
    return varied
```

### Code evaluator

```python
def evaluate_code(code: str, answer: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    metadata = {"tests_total": len(answer["tests"]), "tests_passed": 0}
    try:
        with time_limit(3), redirect_stdout(io.StringIO()):
            exec(code, namespace, namespace)
        function_name = answer["function_name"]
        if function_name not in namespace:
            metadata["error"] = f"Missing function {function_name}"
            return {"score": 0.0, "meta": metadata}
        for test in answer["tests"]:
            with time_limit(3), redirect_stdout(io.StringIO()):
                exec(test, namespace, namespace)
            metadata["tests_passed"] += 1
        return {"score": 1.0, "meta": metadata}
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["partial_score"] = metadata["tests_passed"] / metadata["tests_total"]
        return {"score": metadata["partial_score"], "meta": metadata}
```

## 7.11 Critical code excerpts from `scripts/benchmark_ultra_scaffold.py`

These matter because they were used in the earlier Qwen2.5 benchmark.

### Baseline vs Ultra scaffold generation

```python
def run_baseline(model: Any, tokenizer: Any, task: Task, *, device: str) -> CandidateResult:
    prompt_messages = [
        {"role": "system", "content": system_prompt_for_task(task, style="concise")},
        {"role": "user", "content": user_prompt_for_task(task, style="concise")},
    ]
    raw_text, latency_s = generate_response(
        model,
        tokenizer,
        prompt_messages,
        max_new_tokens=task.max_new_tokens,
        do_sample=False,
        device=device,
    )
    return evaluate_candidate(task, raw_text, latency_s, meta={"strategy": "baseline_greedy"})


def run_ultra_scaffold(model: Any, tokenizer: Any, task: Task, fanout_plan: Any, *, device: str) -> dict[str, Any]:
    candidates: list[CandidateResult] = []
    temperatures = fanout_plan.temperatures
    top_p_values = fanout_plan.top_p
    styles = fanout_plan.styles

    for index in range(fanout_plan.n_candidates):
        temperature = temperatures[index % len(temperatures)]
        top_p = top_p_values[index % len(top_p_values)]
        style = styles[index % len(styles)]
        prompt_messages = [
            {"role": "system", "content": system_prompt_for_task(task, style=style)},
            {"role": "user", "content": user_prompt_for_task(task, style=style)},
        ]
        raw_text, latency_s = generate_response(
            model,
            tokenizer,
            prompt_messages,
            max_new_tokens=task.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            seed=1000 + index,
            device=device,
        )
        candidates.append(
            evaluate_candidate(
                task,
                raw_text,
                latency_s,
                meta={
                    "strategy": "ultra_scaffold",
                    "temperature": temperature,
                    "top_p": top_p,
                    "style": style,
                },
            )
        )

    selected = select_candidate(task, candidates)
    return {"candidates": candidates, "selected": selected}
```

### Qwen2.5 harness prompt templates

```python
def system_prompt_for_task(task: Task, *, style: str) -> str:
    if task.evaluator == "code":
        if style == "cot":
            return "You are a careful Python coding assistant. Think through the solution internally, but output only the final Python code."
        return "You are a concise Python coding assistant. Output only valid Python code."
    if task.evaluator == "json":
        return "You are a precise information extraction assistant. Return valid JSON only."
    if style == "cot":
        return "You are a reasoning assistant. Work carefully and end with a line that begins FINAL: followed by only the answer."
    return "You are a concise assistant. Give the final answer directly."


def user_prompt_for_task(task: Task, *, style: str) -> str:
    if style != "cot" or task.evaluator in {"json", "code"}:
        return task.prompt
    return task.prompt + "\nShow your reasoning briefly, then end with `FINAL: <answer>`."
```

### Qwen2.5 harness selection logic

```python
def select_candidate(task: Task, candidates: list[CandidateResult]) -> CandidateResult:
    if task.evaluator == "code":
        best = max(candidates, key=lambda candidate: (candidate.score, -candidate.latency_s))
        return best

    answers = [candidate.canonical_answer for candidate in candidates if candidate.canonical_answer not in (None, "")]
    if not answers:
        return max(candidates, key=lambda candidate: candidate.score)

    counts = Counter(json.dumps(answer, sort_keys=True) if isinstance(answer, dict) else str(answer) for answer in answers)
    best_count = max(counts.values())
    best_answers = {answer for answer, count in counts.items() if count == best_count}
    for candidate in candidates:
        normalized = json.dumps(candidate.canonical_answer, sort_keys=True) if isinstance(candidate.canonical_answer, dict) else str(candidate.canonical_answer)
        if normalized in best_answers:
            return candidate
    return candidates[0]
```

## 7.12 Critical scaffold code excerpts

These are the core ULTRA scaffold pieces the benchmarks relied on.

### Fan-out plan builder

```python
@dataclass(slots=True)
class FanoutPlan:
    ultra_profile: UltraProfileName
    n_candidates: int
    temperatures: list[float]
    top_p: list[float]
    styles: list[str]


def build_fanout_plan(config: GlobalConfig, ultra_profile: UltraProfileName) -> FanoutPlan:
    return FanoutPlan(
        ultra_profile=ultra_profile,
        n_candidates=config.ultra.n_candidates,
        temperatures=list(config.ultra.diversity.temperatures),
        top_p=list(config.ultra.diversity.top_p),
        styles=list(config.ultra.diversity.styles),
    )
```

### Relevant config defaults

```python
@dataclass
class UltraDiversityConfig:
    temperatures: List[float] = field(default_factory=lambda: [0.2, 0.6, 0.9])
    top_p: List[float] = field(default_factory=lambda: [0.85, 0.95])
    styles: List[str] = field(default_factory=lambda: ["concise", "cot"])


@dataclass
class UltraSelectionConfig:
    self_consistency: bool = True
    mbr: bool = True
    logprobs: bool = True


@dataclass
class UltraRefineConfig:
    self_refine_passes: int = 1
    chain_of_verification: bool = True


@dataclass
class UltraSection:
    n_candidates: int = 8
    max_n_candidates: int = 32
    diversity: UltraDiversityConfig = field(default_factory=UltraDiversityConfig)
    selection: UltraSelectionConfig = field(default_factory=UltraSelectionConfig)
    refine: UltraRefineConfig = field(default_factory=UltraRefineConfig)
```

## 8. Results From `benchmark_qwen_reasoning_modes.py` Full 10-Task Run

### 8.1 Accuracy

`Qwen3.5-0.8B` non-thinking:

- single-pass: `10.0%`
- Ultra fan-out: `10.0%`
- math: `25.0% -> 25.0%`
- logic: `0.0% -> 0.0%`
- code: `0.0% -> 0.0%`

`Qwen3.5-0.8B` thinking:

- single-pass: `20.0%`
- Ultra fan-out: `30.0%`
- math: `0.0% -> 0.0%`
- logic: `0.0% -> 0.0%`
- code: `66.7% -> 100.0%`

`Qwen3-1.7B` non-thinking:

- single-pass: `30.0%`
- Ultra fan-out: `30.0%`
- math: `0.0% -> 0.0%`
- logic: `0.0% -> 0.0%`
- code: `100.0% -> 100.0%`

`Qwen3-1.7B` thinking:

- single-pass: `0.0%`
- Ultra fan-out: `0.0%`
- math: `0.0% -> 0.0%`
- logic: `0.0% -> 0.0%`
- code: `0.0% -> 0.0%`

Aggregate across all four conditions:

- mean single-pass accuracy: `15.0%`
- mean Ultra accuracy: `17.5%`
- mean Ultra time multiplier: `2.9599339963317592x`

Mean category accuracy across all four conditions:

- math: `6.25% -> 6.25%`
- logic: `0.0% -> 0.0%`
- code: `41.666666666666664% -> 50.0%`

### 8.2 Generation-time totals

`Qwen3.5-0.8B` non-thinking:

- total single generation time: `77.8 s`
- total Ultra generation time: `234.5 s`
- ratio: `3.01x`

`Qwen3.5-0.8B` thinking:

- total single generation time: `204.4 s`
- total Ultra generation time: `605.7 s`
- ratio: `2.96x`

`Qwen3-1.7B` non-thinking:

- total single generation time: `68.7 s`
- total Ultra generation time: `192.2 s`
- ratio: `2.80x`

`Qwen3-1.7B` thinking:

- total single generation time: `191.1 s`
- total Ultra generation time: `585.3 s`
- ratio: `3.06x`

### 8.3 Model/tokenizer load timings

Full run:

- `qwen3.5_0.8b_nonthinking`
  - tokenizer load `2.3436 s`
  - model load `1.7317 s`
- `qwen3.5_0.8b_thinking`
  - tokenizer load `2.2265 s`
  - model load `1.1305 s`
- `qwen3_1.7b_nonthinking`
  - tokenizer load `1.2958 s`
  - model load `142.3006 s`
  - this long load included initial fetch/cache behavior
- `qwen3_1.7b_thinking`
  - tokenizer load `1.3402 s`
  - model load `2.0659 s`
  - after cache was present

### 8.4 Average selected generated tokens

From the report summaries:

- `qwen3.5_0.8b_nonthinking`: `101.0`
- `qwen3.5_0.8b_thinking`: `294.4`
- `qwen3_1.7b_nonthinking`: `92.2`
- `qwen3_1.7b_thinking`: `294.4`

### 8.5 Representative wins and failures

One clear Ultra-only win in the full Qwen run:

- condition: `Qwen3.5-0.8B` thinking
- task: `code_merge_intervals`
- single-pass: failed
- Ultra fan-out: passed

Other notable observations:

- `Qwen3-1.7B` non-thinking solved all 3 code tasks in both single-pass and Ultra modes.
- `Qwen3-1.7B` thinking solved nothing at all.
- No condition showed any measured improvement on logic.
- No condition showed any measured improvement on math.

## 9. Why I Added The Long-Budget Rerun

After the first full Qwen3/Qwen3.5 run, I noticed that many failures ended exactly at the token cap and never reached `FINAL:`.

That made me worry that some failures were not true reasoning failures, but simply decode-budget truncation.

So I patched `scripts/benchmark_qwen_reasoning_modes.py` to add:

- `--single-token-multiplier`
- `--thinking-token-multiplier`

Then I reran only the 7 math/logic tasks with both multipliers set to `2.0`.

## 10. Results From The Long-Budget Reasoning Rerun

Tasks in rerun:

- `math_tickets`
- `math_pipes`
- `math_divisible_by_5`
- `math_mixture`
- `logic_knights`
- `logic_lineup`
- `logic_mixed_box`

All 4 conditions:

- single-pass: `0.0%`
- Ultra fan-out: `0.0%`

Per-condition generation times:

`Qwen3.5-0.8B` non-thinking:

- single total `93.7 s`
- Ultra total `281.3 s`
- ratio `3.00x`

`Qwen3.5-0.8B` thinking:

- single total `245.4 s`
- Ultra total `731.9 s`
- ratio `2.98x`

`Qwen3-1.7B` non-thinking:

- single total `84.1 s`
- Ultra total `261.1 s`
- ratio `3.10x`

`Qwen3-1.7B` thinking:

- single total `248.8 s`
- Ultra total `727.7 s`
- ratio `2.92x`

Interpretation:

- Doubling the token budget did not recover even one reasoning task.
- That strongly suggests the reasoning failure pattern is not primarily a short-token-budget problem.

## 11. Diagnostic Counts That Matter

I computed extra diagnostics from the JSON reports.

### 11.1 Full Qwen3/Qwen3.5 run diagnostics

`qwen3.5_0.8b_nonthinking`:

- `no_final`
  - single-pass: `9 / 10`
  - Ultra selected: `9 / 10`
- cap-like failures:
  - single-pass: `7 / 10`
  - Ultra selected: `7 / 10`
- `empty_cleaned_text`
  - single-pass: `0 / 10`
  - Ultra selected: `0 / 10`

`qwen3.5_0.8b_thinking`:

- `no_final`
  - single-pass: `5 / 10`
  - Ultra selected: `5 / 10`
- cap-like failures:
  - single-pass: `8 / 10`
  - Ultra selected: `7 / 10`
- `empty_cleaned_text`
  - single-pass: `0 / 10`
  - Ultra selected: `0 / 10`

`qwen3_1.7b_nonthinking`:

- `no_final`
  - single-pass: `7 / 10`
  - Ultra selected: `7 / 10`
- cap-like failures:
  - single-pass: `7 / 10`
  - Ultra selected: `7 / 10`
- `empty_cleaned_text`
  - single-pass: `0 / 10`
  - Ultra selected: `0 / 10`

`qwen3_1.7b_thinking`:

- `no_final`
  - single-pass: `10 / 10`
  - Ultra selected: `10 / 10`
- cap-like failures:
  - single-pass: `10 / 10`
  - Ultra selected: `10 / 10`
- `empty_cleaned_text`
  - single-pass: `10 / 10`
  - Ultra selected: `10 / 10`

### 11.2 Long-budget rerun diagnostics

`qwen3.5_0.8b_nonthinking`:

- `no_final`
  - single-pass: `7 / 7`
  - Ultra selected: `7 / 7`
- cap-like failures:
  - single-pass: `7 / 7`
  - Ultra selected: `7 / 7`
- `empty_cleaned_text`
  - single-pass: `0 / 7`
  - Ultra selected: `0 / 7`

`qwen3.5_0.8b_thinking`:

- `no_final`
  - single-pass: `4 / 7`
  - Ultra selected: `5 / 7`
- cap-like failures:
  - single-pass: `7 / 7`
  - Ultra selected: `7 / 7`
- `empty_cleaned_text`
  - single-pass: `0 / 7`
  - Ultra selected: `0 / 7`

`qwen3_1.7b_nonthinking`:

- `no_final`
  - single-pass: `7 / 7`
  - Ultra selected: `7 / 7`
- cap-like failures:
  - single-pass: `7 / 7`
  - Ultra selected: `7 / 7`
- `empty_cleaned_text`
  - single-pass: `0 / 7`
  - Ultra selected: `0 / 7`

`qwen3_1.7b_thinking`:

- `no_final`
  - single-pass: `6 / 7`
  - Ultra selected: `6 / 7`
- cap-like failures:
  - single-pass: `6 / 7`
  - Ultra selected: `6 / 7`
- `empty_cleaned_text`
  - single-pass: `6 / 7`
  - Ultra selected: `6 / 7`

## 12. Very Important Observed Failure Modes

### 12.1 Qwen3-1.7B thinking appears fundamentally mismatched to the current harness

Observed behavior:

- raw outputs often started with `<think>`
- many outputs never emitted `</think>` before `max_new_tokens`
- because of current cleaning logic, this often produced `cleaned_text == ""`
- scores then became `0` even if the raw reasoning text might have contained the correct answer later or was still on the way to it

In the full run:

- selected think-block rate was `1.0`
- `empty_cleaned_text` happened for `20 / 20` scored outputs

In the long-budget rerun:

- `empty_cleaned_text` still happened for `12 / 14` scored outputs
- only `2 / 14` outputs even contained `</think>`

This seems like one of the most important current issues.

### 12.2 Qwen3.5-0.8B thinking behaved differently from Qwen3-1.7B thinking

Observed behavior:

- `Qwen3.5-0.8B` thinking did not show explicit `<think>` tags in the captured outputs
- `selected_think_block_rate` was `0.0`
- outputs often looked like visible reasoning text, but still failed to produce correct final answers

So the two Qwen families did not behave the same way under the current harness.

### 12.3 The harness may be too strict or misaligned for text-answer tasks

For `logic_lineup` and `logic_mixed_box`:

- if the model did not emit `FINAL: Bo` or `FINAL: Mixed`
- then the full cleaned text was normalized and compared as one string
- so outputs like "The answer is Mixed." would fail unless they exactly matched an alias

This may be undercounting actual correctness.

### 12.4 The harness may still be too weak to show any genuine ULTRA advantage

Current "Ultra" behavior in the Qwen harness is only:

- multiple sampled candidates
- light prompt variation
- light parameter variation
- simple majority-vote or unit-test-based selection

It does not have:

- a real judge model
- logprob-based reranking
- self-refine loop
- chain-of-verification loop
- explicit post-thinking answer extraction / continuation
- task-adaptive candidate budgeting
- batching / parallel generation

So one possibility is that the "ULTRA" side is currently too shallow to produce real gains.

### 12.5 Non-thinking modes were more usable than thinking modes in this setup

This was especially true for:

- code generation
- directly parseable answers

The best practical condition in the experiments was:

- `Qwen/Qwen3-1.7B` non-thinking

But even that condition still scored:

- `0` on all math
- `0` on all logic

## 12.6 Representative raw failure examples

These are real snippets from the captured outputs.

### Raw `Qwen3-1.7B` thinking failure on `math_tickets`

Single-pass raw text started like this and ended before `</think>`:

```text
<think>
Okay, let's try to figure out how many child tickets were sold. So, the theater sold adult tickets for $17 and child tickets for $9. They sold a total of 52 tickets and collected $708. Hmm, I need to find the number of child tickets.

Let me start by setting up some variables. Let's let a be the number of adult tickets and c be the number of child tickets. Then, according to the problem, the total number of tickets is 52. So, that gives me the equation:

a + c = 52

And the total amount of money collected is $708. Since adult tickets are $17 each and child tickets are $9 each, the equation for the total revenue would be:

17a + 9c = 708

So now I have a system of two equations:

1. a + c = 52
2. 17a + 9c = 708

I need to solve this system to find the value of c, which is the number of child tickets. Let me think about how to solve this. The first equation can be used to express one variable in terms of the other
```

Because there was no closing `</think>`, the current cleaner reduced this to:

```text
""
```

### Raw `Qwen3-1.7B` thinking failure on `code_normalize_spaces`

Single-pass raw text started like this and ended before any final code answer:

```text
<think>
Okay, I need to write a Python function called normalize_spaces that takes a text string and returns a new string where consecutive whitespace is collapsed into a single space, and leading and trailing whitespace is stripped. Let me think about how to approach this.

First, the problem says to collapse consecutive whitespaces. So, any sequence of spaces, tabs, newlines, etc., should become a single space. But how to detect those sequences?

In Python, the string method rsplit can be used, but maybe a better approach is to iterate through the text and check for whitespace characters. Alternatively, I can use the re module to find all whitespace sequences and replace them with a single space.
...
```

Again, with no `</think>`, the cleaner reduced this to empty text for scoring.

### Raw `Qwen3.5-0.8B` thinking failure on `math_tickets`

This model did not show explicit `<think>` tags in the captured output, but it still burned most of the budget on visible reasoning text and did not finish with a clean final answer:

```text
Thinking Process:

1.  Analyze the Request:
    *   Ticket price for adult: $17
    *   Ticket price for child: $9
    *   Total tickets sold: 52
    *   Total collected: $708
    *   Goal: Find the number of child tickets sold.
    *   Constraint: End the response with `FINAL: <answer>`.

2.  Set up the variables:
    *   Let $a$ = number of adult tickets sold.
    *   Let $b$ = number of child tickets sold.
    *   Total tickets: $a + b = 52$.
    *   Total revenue (Collected): $(a × 17) + (b × 9) = 708$.

3.  Formulate the system of equations:
    1.  $a + b = 52$
    2.  $17a + 9b = 708$

4.  Solve for variables:
...
```

## 13. What I Suspect Might Be Going Wrong

I want your judgment on these hypotheses.

Hypothesis A:

- We are using Qwen thinking mode incorrectly.
- Maybe `enable_thinking=True` is not enough by itself.
- Maybe we need a proper two-stage decode:
  - let the model think
  - detect `</think>`
  - continue generation for final answer
- Or maybe we need a different prompt format altogether.

Hypothesis B:

- The generic `FINAL: <answer>` instruction is not well aligned to Qwen thinking-mode expectations.
- The models may be spending budget on long reasoning traces but not transitioning cleanly to the final answer format.

Hypothesis C:

- `clean_model_text()` is too destructive for unclosed think blocks.
- For Qwen3-1.7B thinking, this may be the single biggest harness bug.

Hypothesis D:

- The text-answer evaluation for logic tasks is too strict and is suppressing real signal.

Hypothesis E:

- The tasks may be too exact-match / short-answer-heavy for these model sizes, especially on math / logic.
- But the long-budget rerun still being all-zero suggests something more than just task difficulty.

Hypothesis F:

- Using `do_sample=True` even for single-pass in the Qwen harness may be adding noise and making it harder to compare against Ultra fan-out.

Hypothesis G:

- The current Ultra fan-out is too shallow and too weakly selected to produce real reasoning gains.

Hypothesis H:

- The missing Qwen3.5 fast-path dependencies might matter more than just latency.
- I am less convinced of this, but I want your view.

## 14. What I Need You To Tell Me

Please answer all of these concretely.

### 14.1 Benchmark validity

- Is this benchmark methodology fair?
- Which parts are valid?
- Which parts are flawed enough that the current conclusions are partly unreliable?

### 14.2 Qwen thinking-mode handling

- Are we using Qwen3 / Qwen3.5 thinking mode correctly through `apply_chat_template(... enable_thinking=...)`?
- Is the current prompt shape wrong for these models?
- Should we explicitly stop on `</think>` and then force a second generation stage for the final answer?
- Should we parse the answer differently for Qwen thinking models?

### 14.3 Exact code changes you would make

Please propose exact changes to:

- `clean_model_text()`
- the generation loop
- stopping criteria
- prompt templates
- answer extraction
- text-answer scoring
- selection logic
- candidate count policy

If possible, suggest code snippets or specific pseudocode.

### 14.4 Better evaluation design

How would you redesign the benchmark so it more fairly measures:

- raw single-pass quality
- real benefit of fan-out
- benefit of thinking mode
- benefit of verification / judging / refinement

Please be explicit about:

- which baseline should be greedy vs sampled
- how many candidates are worth trying on 8 GB VRAM
- whether to use self-consistency, majority vote, verifier-first, or judge models
- what task mix should be used

### 14.5 Best next experiments

Given this exact hardware limit, what should the next experiments be?

Examples:

- Qwen3-1.7B non-thinking vs thinking with corrected post-think handling
- Qwen3.5-0.8B with different prompt format
- a stronger 3B-ish model that still fits
- code-only verifier experiments
- structured output / JSON-focused benchmarks

Please prioritize experiments that will produce the most diagnostic value per GPU-hour.

### 14.6 Cross-hardware performance strategy

Please also answer this explicitly:

- What architecture and inference strategy would you recommend if the goal is to maximize quality not just on this RTX 2070 Max-Q 8 GB machine, but across RTX 20/30/40/50-series consumer GPUs more broadly?
- Which recommendations should stay constant across generations?
- Which recommendations should change by VRAM tier, tensor-core generation, kernel availability, or framework backend?
- If the optimal approach differs for 8 GB, 12 GB, 16 GB, and 24 GB cards, provide that matrix explicitly.

## 15. My Current Bottom-Line Reading

This is my current interpretation, but I want you to challenge it if wrong.

- The current scaffold does not meaningfully improve reasoning.
- The only clear positive signal so far is narrow code-task recovery when fan-out plus a hard verifier is used.
- Qwen3 thinking-mode behavior is probably being mishandled by the harness.
- The current answer extraction and post-processing are likely suppressing real signal, especially for Qwen3-1.7B thinking.
- Even after accounting for that, the current Ultra implementation is still too shallow to justify its compute cost.
- Any final solution should aim for maximum practical performance on the current RTX 2070 Max-Q 8 GB setup and also generalize into a sensible strategy for RTX 20/30/40/50-series consumer GPUs.

## 15.1 What I want from your answer format

Please structure your answer in this order:

1. `Root cause analysis`
2. `What is benchmark artifact vs true model weakness`
3. `Exact code changes to make immediately`
4. `Improved decoding / thinking-mode handling design`
5. `Improved benchmark design`
6. `Best next experiments on this 8 GB GPU`
7. `What would actually maximize performance on this hardware`

I do not want generic advice. I want an engineering-grade diagnosis and a concrete path to maximum practical performance.

## 16. Relevant File Paths

These are the local files that contain the current implementation and results:

- `scripts/benchmark_ultra_scaffold.py`
- `scripts/benchmark_qwen_reasoning_modes.py`
- `reports/ultra_scaffold_benchmark_gpu_20260312.json`
- `reports/ultra_scaffold_benchmark_gpu_20260312.md`
- `reports/qwen_reasoning_modes_gpu_20260312.json`
- `reports/qwen_reasoning_modes_gpu_longreason_20260312.json`
- `reports/qwen_reasoning_modes_gpu_qwen3_qwen3p5_20260312.md`
- `src/ultra/config.py`
- `src/ultra/ultra_mode/fanout.py`
- `src/ultra/mcd/inspector.py`
- `src/ultra/tools/estimator.py`
- `src/ultra/cli.py`

Please treat the information in this prompt as authoritative context from the actual runs already completed, and use it to recommend the next code changes and experiment plan.
