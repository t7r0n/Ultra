"""Shared runtime for ULTRA chat, agent, and eval flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import random
from typing import Any

from .backends import AutoBackendContext, select_backend
from .backends.common import GenerationRequest, GenerationResult, StructuredOutputRequest
from .backends import hf_engine, llama_cpp_engine, sglang_engine, vllm_engine
from .config import BackendName, GlobalConfig, UltraProfileName
from .mcd.inspector import ModelInspection, ModelReferenceError, fetch_model_reference, inspect_model
from .tools.hwcheck import collect_hardware_report
from .tools.logging import RunArtifacts, append_jsonl
from .tools.verifiers import (
    VerificationResult,
    verify_citation_style,
    verify_code,
    verify_math_response,
    verify_no_thought_leak,
    verify_prompt_alignment,
)
from .ultra_mode.fanout import build_fanout_plan
from .ultra_mode.refine import build_refine_plan, refine_response
from .ultra_mode.selection import SelectionResult, build_selection_plan, ranking_tuple, select_candidates
from .ultra_mode.structured import StructuredOutputPlan, build_structured_output_plan, validate_structured_output


BACKEND_GENERATORS = {
    "hf": hf_engine.generate,
    "vllm": vllm_engine.generate,
    "sglang": sglang_engine.generate,
    "llamacpp": llama_cpp_engine.generate,
}


@dataclass(slots=True)
class CandidateRuntimeRecord:
    index: int
    style: str
    text: str
    verification: list[VerificationResult]
    generation: GenerationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "style": self.style,
            "text": self.text,
            "verification": [result.to_dict() for result in self.verification],
            "generation": self.generation.to_dict(),
        }


@dataclass(slots=True)
class UltraTurnResult:
    backend: BackendName
    prompt_messages: list[dict[str, str]]
    fanout: dict[str, Any]
    candidates: list[CandidateRuntimeRecord]
    selection: SelectionResult
    final_text: str
    final_payload: Any | None
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "prompt_messages": self.prompt_messages,
            "fanout": self.fanout,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection": self.selection.to_dict(),
            "final_text": self.final_text,
            "final_payload": self.final_payload,
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class LongContextPolicy:
    requested_context_window: int | None
    effective_context_window: int | None
    rope_scaling_override: dict[str, Any] | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_context_window": self.requested_context_window,
            "effective_context_window": self.effective_context_window,
            "rope_scaling_override": self.rope_scaling_override,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class JudgeOutcome:
    candidate_index: int
    weighted_score: float
    raw_score: float
    vote_points: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "weighted_score": self.weighted_score,
            "raw_score": self.raw_score,
            "vote_points": self.vote_points,
            "details": self.details,
        }


def choose_backend(
    model_ref: str,
    *,
    requested_backend: BackendName,
    ctx: int | None,
    batch: int,
    structured_output: bool,
    high_throughput: bool = False,
) -> BackendName:
    if requested_backend != "auto":
        return requested_backend
    resolved = fetch_model_reference(model_ref)
    hardware = collect_hardware_report()
    return select_backend(
        AutoBackendContext(
            gguf=resolved.gguf,
            has_gpu=bool(hardware.gpus),
            ctx=ctx,
            batch=batch,
            structured_output=structured_output,
            high_throughput=high_throughput,
        )
    )


def run_ultra_turn(
    *,
    model_ref: str,
    backend: BackendName,
    config: GlobalConfig,
    ultra_profile: UltraProfileName,
    conversation: list[dict[str, str]],
    artifacts: RunArtifacts | None = None,
    seed_base: int | None = None,
) -> UltraTurnResult:
    structured_plan = build_structured_output_plan(config)
    selected_backend = choose_backend(
        model_ref,
        requested_backend=backend,
        ctx=config.context_window if isinstance(config.context_window, int) else None,
        batch=1,
        structured_output=structured_plan.enabled,
    )
    inspection: ModelInspection | None = None
    try:
        inspection = inspect_model(model_ref, backend=selected_backend)
    except ModelReferenceError:
        pass
    long_context_policy = build_long_context_policy(config=config, inspection=inspection, backend=selected_backend)
    fanout_plan = build_fanout_plan(
        config,
        ultra_profile,
        prompt=conversation[-1]["content"],
        structured_output=structured_plan.enabled,
        seed_base=seed_base or stable_seed(model_ref, conversation[-1]["content"]),
    )
    selection_plan = build_selection_plan(config)
    generator = BACKEND_GENERATORS[selected_backend]
    candidate_records: list[CandidateRuntimeRecord] = []
    verifier_scores: list[float] = []
    passed_verifiers: list[int] = []
    avg_logprobs: list[float | None] = []
    latencies_s: list[float] = []
    generated_tokens: list[int] = []
    candidate_metrics: list[dict[str, Any]] = []
    candidate_log_payloads: list[dict[str, Any]] = []
    texts: list[str] = []

    for candidate in fanout_plan.candidates:
        messages = [
            {"role": "system", "content": candidate.system_prompt},
            *conversation[:-1],
            {"role": "user", "content": candidate.user_prompt},
        ]
        request = GenerationRequest(
            model_ref=model_ref,
            messages=messages,
            max_new_tokens=max_new_tokens_for_profile(ultra_profile),
            temperature=candidate.temperature,
            top_p=candidate.top_p,
            top_k=candidate.top_k,
            min_p=candidate.min_p,
            repetition_penalty=candidate.repetition_penalty,
            seed=candidate.seed,
            precision=config.precision,
            structured_output=StructuredOutputRequest(
                enabled=structured_plan.enabled,
                json_schema=structured_plan.json_schema,
                regex=structured_plan.regex,
                grammar=structured_plan.grammar,
            ),
            enable_thinking=ultra_profile == "reasoning",
            request_logprobs=config.ultra.selection.logprobs,
            extra={
                "inspection": {
                    "arch": inspection.arch if inspection is not None else None,
                    "context_window": inspection.context_window if inspection is not None else None,
                    "sliding_window": inspection.sliding_window if inspection is not None else None,
                    "rope_scaling": inspection.rope_scaling if inspection is not None else None,
                },
                "max_context_tokens": long_context_policy.effective_context_window,
                "rope_scaling_override": long_context_policy.rope_scaling_override,
                "long_context_notes": list(long_context_policy.notes),
            },
        )
        generation = generator(request)
        verification = verify_candidate(
            generation.text,
            prompt=conversation[-1]["content"],
            structured_plan=structured_plan,
            ultra_profile=ultra_profile,
        )
        score = sum(item.score for item in verification) / max(len(verification), 1)
        passed = sum(1 for item in verification if item.passed)
        runtime_metrics = {
            **generation.metrics,
            "verification_names": [item.name for item in verification],
            "verification_details": [item.to_dict() for item in verification],
            "passed_verifiers": passed,
            "verifier_score": score,
        }
        candidate_record = CandidateRuntimeRecord(
            index=candidate.index,
            style=candidate.style,
            text=generation.text,
            verification=verification,
            generation=generation,
        )
        candidate_records.append(candidate_record)
        texts.append(generation.text)
        verifier_scores.append(score)
        passed_verifiers.append(passed)
        avg_logprobs.append(generation.avg_logprob)
        latencies_s.append(generation.latency_s)
        generated_tokens.append(generation.generated_tokens)
        candidate_metrics.append(runtime_metrics)
        candidate_log_payloads.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "candidate_index": candidate.index,
                "style": candidate.style,
                "text": generation.text,
                "avg_logprob": generation.avg_logprob,
                "latency_s": generation.latency_s,
                "generated_tokens": generation.generated_tokens,
                "finish_reason": generation.finish_reason,
                "prompt_text": generation.prompt_text,
                "backend_metrics": generation.metrics,
                "verification": [item.to_dict() for item in verification],
                "verifier_score": score,
                "passed_verifiers": passed,
            }
        )
        if artifacts is not None:
            if config.logging.save_prompts:
                append_jsonl(
                    artifacts.prompts_jsonl,
                    {
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "candidate_index": candidate.index,
                        "style": candidate.style,
                        "messages": messages,
                        "seed": candidate.seed,
                        "backend": selected_backend,
                        "request": request.to_dict(),
                    },
                )

    judge_outcomes, judge_metadata = run_candidate_judge(
        model_ref=model_ref,
        prompt=conversation[-1]["content"],
        backend=selected_backend,
        config=config,
        ultra_profile=ultra_profile,
        texts=texts,
        verifier_scores=verifier_scores,
        passed_verifiers=passed_verifiers,
        avg_logprobs=avg_logprobs,
        seed_base=seed_base or stable_seed(model_ref, conversation[-1]["content"]),
    )
    judge_scores = [outcome.weighted_score for outcome in judge_outcomes]
    for index, outcome in enumerate(judge_outcomes):
        candidate_metrics[index].update(
            {
                "judge_score_raw": outcome.raw_score,
                "judge_score_weighted": outcome.weighted_score,
                "judge_votes": outcome.vote_points,
                "judge_details": outcome.details,
            }
        )
        candidate_log_payloads[index].update(
            {
                "judge_score_raw": outcome.raw_score,
                "judge_score_weighted": outcome.weighted_score,
                "judge_votes": outcome.vote_points,
                "judge_details": outcome.details,
            }
        )
    if artifacts is not None and config.logging.save_candidates:
        for payload in candidate_log_payloads:
            append_jsonl(artifacts.candidates_jsonl, payload)

    selection = select_candidates(
        texts,
        verifier_scores=verifier_scores,
        judge_scores=judge_scores,
        passed_verifiers=passed_verifiers,
        avg_logprobs=avg_logprobs,
        latencies_s=latencies_s,
        generated_tokens=generated_tokens,
        candidate_metrics=candidate_metrics,
        task_type=task_type_for_profile(ultra_profile),
        structured_output=structured_plan.enabled,
        selection_plan=selection_plan,
    )
    final_text = texts[selection.selected_index]
    refine_plan = build_refine_plan(config)
    refined = refine_response(
        generate_fn=generator,
        model_ref=model_ref,
        original_messages=conversation,
        candidate_texts=[texts[item.index] for item in sorted(selection.scores, key=ranking_tuple, reverse=True)],
        refine_plan=refine_plan,
        structured_output=StructuredOutputRequest(
            enabled=structured_plan.enabled,
            json_schema=structured_plan.json_schema,
            regex=structured_plan.regex,
            grammar=structured_plan.grammar,
        ),
        seed=(seed_base or stable_seed(model_ref, conversation[-1]["content"])) + 10_000,
        precision=config.precision,
    )
    if refined is not None:
        final_text = refined.text

    valid_payload: Any | None = None
    valid, payload = validate_structured_output(final_text, structured_plan)
    if valid:
        valid_payload = payload if structured_plan.enabled else None

    metrics = {
        "latency_s": sum(latencies_s),
        "tokens_per_s": (
            sum(generated_tokens) / sum(latencies_s)
            if sum(latencies_s) > 0
            else None
        ),
        "peak_kv_blocks": None,
        "peak_memory_bytes": max(
            (
                int(metric["peak_memory_bytes"])
                for metric in candidate_metrics
                if metric.get("peak_memory_bytes") is not None
            ),
            default=None,
        ),
        "selected_candidate_index": selection.selected_index,
        "selected_backend": selected_backend,
        "selected_candidate_metrics": selection.scores[selection.selected_index].metrics,
        "judge_enabled": config.judge.enabled,
        "judge_backend": judge_metadata["judge_backend"],
        "judge_model_ref": judge_metadata["judge_model_ref"],
        "judge_shortlist": judge_metadata["shortlist"],
        "judge_pairwise": judge_metadata["pairwise"],
        "judge_comparison_count": judge_metadata["comparison_count"],
        "long_context_policy": long_context_policy.to_dict(),
    }
    if artifacts is not None:
        artifacts.selection_json.write_text(
            json.dumps(
                {
                    **selection.to_dict(),
                    "selected_candidate": selection.scores[selection.selected_index].to_dict(),
                    "judge": judge_metadata,
                    "long_context_policy": long_context_policy.to_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        artifacts.final_txt.write_text(final_text + ("\n" if not final_text.endswith("\n") else ""), encoding="utf-8")
        artifacts.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        if artifacts.final_json is not None and valid_payload is not None:
            artifacts.final_json.write_text(json.dumps(valid_payload, indent=2, sort_keys=True), encoding="utf-8")
    return UltraTurnResult(
        backend=selected_backend,
        prompt_messages=conversation,
        fanout=fanout_plan.to_dict(),
        candidates=candidate_records,
        selection=selection,
        final_text=final_text,
        final_payload=valid_payload,
        metrics=metrics,
    )


def verify_candidate(
    text: str,
    *,
    prompt: str,
    structured_plan: StructuredOutputPlan,
    ultra_profile: UltraProfileName,
) -> list[VerificationResult]:
    results: list[VerificationResult] = []
    if structured_plan.enabled:
        valid, payload = validate_structured_output(text, structured_plan)
        details = payload if isinstance(payload, str) else "structured output valid"
        results.append(
            VerificationResult(
                name="structured_output",
                passed=valid,
                score=1.0 if valid else 0.0,
                details=details,
                metadata=payload if isinstance(payload, dict) else None,
            )
        )
    if ultra_profile == "coding":
        results.append(verify_code(text))
    else:
        results.append(verify_no_thought_leak(text))
        results.append(verify_prompt_alignment(text, prompt=prompt))
        citation_result = verify_citation_style(text, prompt=prompt)
        if citation_result.score > 0 or not citation_result.passed:
            results.append(citation_result)
        if ultra_profile == "reasoning":
            results.append(verify_math_response(text, prompt=prompt))
    if not results:
        results.append(VerificationResult(name="neutral", passed=True, score=0.0, details="no task-specific verifier"))
    return results


def build_long_context_policy(
    *,
    config: GlobalConfig,
    inspection: ModelInspection | None,
    backend: BackendName,
) -> LongContextPolicy:
    requested = config.context_window if isinstance(config.context_window, int) else (inspection.context_window if inspection is not None else None)
    effective = requested
    notes: list[str] = []
    rope_scaling_override: dict[str, Any] | None = None
    if inspection is None:
        return LongContextPolicy(
            requested_context_window=requested,
            effective_context_window=effective,
            rope_scaling_override=None,
            notes=["Model inspection unavailable; long-context policy fell back to the configured context window."],
        )
    arch = (inspection.arch or "").lower()
    base_context = inspection.context_window or requested

    if "mistral" in arch and config.long_context.mistral_respect_swa and inspection.sliding_window:
        if effective is None or effective > inspection.sliding_window:
            effective = inspection.sliding_window
            notes.append(
                f"Mistral sliding_window={inspection.sliding_window} enforced for precise attention behavior."
            )

    if "qwen" in arch and requested is not None and base_context is not None and requested > base_context:
        if inspection.rope_scaling is not None:
            notes.append("Qwen rope_scaling metadata detected; using the model-provided long-context configuration.")
        elif config.long_context.allow_rope_scaling:
            rope_scaling_override = {
                "type": "yarn",
                "factor": round(requested / max(base_context, 1), 6),
                "original_max_position_embeddings": base_context,
            }
            notes.append(
                "Qwen requested context exceeds the pretrain window; applying a runtime rope_scaling override."
            )
        else:
            effective = base_context
            notes.append("Qwen context was capped at the pretrain window because rope scaling is disabled.")

    if backend == "llamacpp" and inspection.rope_scaling is not None:
        notes.append("GGUF embedded RoPE metadata detected; relying on llama.cpp file-embedded scaling.")

    return LongContextPolicy(
        requested_context_window=requested,
        effective_context_window=effective,
        rope_scaling_override=rope_scaling_override,
        notes=notes,
    )


def task_type_for_profile(profile: UltraProfileName) -> str:
    if profile == "coding":
        return "code"
    if profile == "reasoning":
        return "math"
    return "text"


def max_new_tokens_for_profile(profile: UltraProfileName) -> int:
    if profile == "coding":
        return 768
    if profile == "reasoning":
        return 512
    if profile == "creative":
        return 640
    return 384


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_candidate_judge(
    *,
    model_ref: str,
    prompt: str,
    backend: BackendName,
    config: GlobalConfig,
    ultra_profile: UltraProfileName,
    texts: list[str],
    verifier_scores: list[float],
    passed_verifiers: list[int],
    avg_logprobs: list[float | None],
    seed_base: int,
) -> tuple[list[JudgeOutcome], dict[str, Any]]:
    neutral = [
        JudgeOutcome(
            candidate_index=index,
            weighted_score=0.0,
            raw_score=0.0,
            vote_points=0.0,
            details={"enabled": False},
        )
        for index in range(len(texts))
    ]
    metadata = {
        "enabled": config.judge.enabled,
        "judge_backend": None,
        "judge_model_ref": None,
        "shortlist": [],
        "pairwise": config.judge.pairwise,
        "comparison_count": 0,
        "comparisons": [],
    }
    if not config.judge.enabled or len(texts) < 2:
        return neutral, metadata

    judge_model_ref = config.judge.model_ref or model_ref
    if config.judge.backend == "auto" and judge_model_ref == model_ref:
        judge_backend = backend
    else:
        try:
            judge_backend = choose_backend(
                judge_model_ref,
                requested_backend=config.judge.backend,
                ctx=config.context_window if isinstance(config.context_window, int) else None,
                batch=1,
                structured_output=True,
            )
        except ModelReferenceError:
            judge_backend = backend
    shortlist = shortlist_candidate_indices(
        verifier_scores=verifier_scores,
        passed_verifiers=passed_verifiers,
        avg_logprobs=avg_logprobs,
        limit=max(2, min(len(texts), config.judge.top_k)),
    )
    metadata.update(
        {
            "judge_backend": judge_backend,
            "judge_model_ref": judge_model_ref,
            "shortlist": shortlist,
        }
    )
    if len(shortlist) < 2:
        return neutral, metadata

    generator = BACKEND_GENERATORS[judge_backend]
    weight = max(0.0, min(float(config.judge.weight), 1.0))
    raw_totals = {index: 0.0 for index in shortlist}
    vote_totals = {index: 0.0 for index in shortlist}
    appearances = {index: 0 for index in shortlist}
    comparison_logs: list[dict[str, Any]] = []

    if config.judge.pairwise:
        for offset, left_index in enumerate(shortlist[:-1]):
            for right_index in shortlist[offset + 1 :]:
                comparison_seed = stable_seed(
                    judge_model_ref,
                    prompt,
                    str(seed_base),
                    f"{left_index}:{right_index}",
                )
                ordered_indices = [left_index, right_index]
                if config.judge.bias_mitigation.shuffle:
                    random.Random(comparison_seed).shuffle(ordered_indices)
                slot_to_candidate = {
                    "candidate_a": ordered_indices[0],
                    "candidate_b": ordered_indices[1],
                }
                judge_request = GenerationRequest(
                    model_ref=judge_model_ref,
                    messages=build_judge_messages(
                        prompt=prompt,
                        task_type=task_type_for_profile(ultra_profile),
                        candidates={
                            "candidate_a": texts[ordered_indices[0]],
                            "candidate_b": texts[ordered_indices[1]],
                        },
                        blind_ids=config.judge.bias_mitigation.blind_ids,
                        pairwise=True,
                    ),
                    max_new_tokens=192,
                    temperature=0.0,
                    top_p=1.0,
                    seed=comparison_seed,
                    precision=config.precision,
                    structured_output=StructuredOutputRequest(
                        enabled=True,
                        json_schema=judge_schema(pairwise=True),
                    ),
                    enable_thinking=False,
                    request_logprobs=False,
                    extra={
                        "ultra_role": "judge",
                        "pair": [left_index, right_index],
                        "source_backend": backend,
                    },
                )
                generation = generator(judge_request)
                payload = parse_judge_payload(generation.text, pairwise=True)
                comparison_log = {
                    "pair": [left_index, right_index],
                    "slot_to_candidate": slot_to_candidate,
                    "raw_text": generation.text,
                    "parsed": payload,
                }
                comparison_logs.append(comparison_log)
                if payload is None:
                    continue
                for slot_name, candidate_index in slot_to_candidate.items():
                    score_key = "score_a" if slot_name == "candidate_a" else "score_b"
                    raw_totals[candidate_index] += payload[score_key]
                    appearances[candidate_index] += 1
                winner = payload["winner"]
                if winner in slot_to_candidate:
                    vote_totals[slot_to_candidate[winner]] += 1.0
                elif winner == "tie":
                    for candidate_index in slot_to_candidate.values():
                        vote_totals[candidate_index] += 0.5
    else:
        for index in shortlist:
            scoring_seed = stable_seed(judge_model_ref, prompt, str(seed_base), f"candidate:{index}")
            judge_request = GenerationRequest(
                model_ref=judge_model_ref,
                messages=build_judge_messages(
                    prompt=prompt,
                    task_type=task_type_for_profile(ultra_profile),
                    candidates={"candidate_a": texts[index]},
                    blind_ids=config.judge.bias_mitigation.blind_ids,
                    pairwise=False,
                ),
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
                seed=scoring_seed,
                precision=config.precision,
                structured_output=StructuredOutputRequest(
                    enabled=True,
                    json_schema=judge_schema(pairwise=False),
                ),
                enable_thinking=False,
                request_logprobs=False,
                extra={
                    "ultra_role": "judge",
                    "candidate_index": index,
                    "source_backend": backend,
                },
            )
            generation = generator(judge_request)
            payload = parse_judge_payload(generation.text, pairwise=False)
            comparison_logs.append({"candidate_index": index, "raw_text": generation.text, "parsed": payload})
            if payload is None:
                continue
            raw_totals[index] += payload["score"]
            appearances[index] += 1

    metadata["comparison_count"] = len(comparison_logs)
    metadata["comparisons"] = comparison_logs
    outcomes: list[JudgeOutcome] = []
    for index in range(len(texts)):
        if index not in raw_totals or appearances[index] == 0:
            outcomes.append(
                JudgeOutcome(
                    candidate_index=index,
                    weighted_score=0.0,
                    raw_score=0.0,
                    vote_points=0.0,
                    details={
                        "enabled": config.judge.enabled,
                        "shortlisted": index in raw_totals,
                        "appearance_count": appearances.get(index, 0),
                    },
                )
            )
            continue
        raw_average = raw_totals[index] / appearances[index]
        vote_average = vote_totals[index] / appearances[index] if config.judge.pairwise else 0.0
        blended = raw_average if not config.judge.pairwise else ((raw_average * 0.7) + (vote_average * 0.3))
        outcomes.append(
            JudgeOutcome(
                candidate_index=index,
                weighted_score=round(blended * weight, 6),
                raw_score=round(raw_average, 6),
                vote_points=round(vote_totals[index], 6),
                details={
                    "enabled": True,
                    "shortlisted": True,
                    "appearance_count": appearances[index],
                    "judge_backend": judge_backend,
                    "judge_model_ref": judge_model_ref,
                },
            )
        )
    return outcomes, metadata


def shortlist_candidate_indices(
    *,
    verifier_scores: list[float],
    passed_verifiers: list[int],
    avg_logprobs: list[float | None],
    limit: int,
) -> list[int]:
    ordered = sorted(
        range(len(verifier_scores)),
        key=lambda index: (
            verifier_scores[index],
            passed_verifiers[index],
            avg_logprobs[index] if avg_logprobs[index] is not None else float("-inf"),
            -index,
        ),
        reverse=True,
    )
    return ordered[:limit]


def build_judge_messages(
    *,
    prompt: str,
    task_type: str,
    candidates: dict[str, str],
    blind_ids: bool,
    pairwise: bool,
) -> list[dict[str, str]]:
    rubric = (
        "Judge the answer quality for correctness, instruction following, completeness, faithfulness, "
        "citation compliance when requested, and clean final-answer style. Do not reward hidden reasoning markers."
    )
    if pairwise:
        candidate_lines: list[str] = []
        for slot_name, text in candidates.items():
            heading = slot_name.upper() if blind_ids else f"{slot_name.upper()} (source preserved)"
            candidate_lines.append(f"{heading}:\n{text}")
        user_content = (
            f"User prompt:\n{prompt}\n\n"
            f"Task type: {task_type}\n\n"
            f"{chr(10).join(candidate_lines)}\n\n"
            'Return JSON with keys "winner", "score_a", "score_b", and "reason". '
            'Winner must be "candidate_a", "candidate_b", or "tie". '
            "Scores must be between 0.0 and 1.0."
        )
    else:
        only_text = candidates["candidate_a"]
        label = "CANDIDATE_A" if blind_ids else "CANDIDATE_A (source preserved)"
        user_content = (
            f"User prompt:\n{prompt}\n\n"
            f"Task type: {task_type}\n\n"
            f"{label}:\n{only_text}\n\n"
            'Return JSON with keys "score" and "reason". Score must be between 0.0 and 1.0.'
        )
    return [
        {"role": "system", "content": rubric},
        {"role": "user", "content": user_content},
    ]


def judge_schema(*, pairwise: bool) -> dict[str, Any]:
    if pairwise:
        return {
            "type": "object",
            "required": ["winner", "score_a", "score_b", "reason"],
            "properties": {
                "winner": {"type": "string", "enum": ["candidate_a", "candidate_b", "tie"]},
                "score_a": {"type": "number"},
                "score_b": {"type": "number"},
                "reason": {"type": "string"},
            },
        }
    return {
        "type": "object",
        "required": ["score", "reason"],
        "properties": {
            "score": {"type": "number"},
            "reason": {"type": "string"},
        },
    }


def parse_judge_payload(text: str, *, pairwise: bool) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    if pairwise:
        winner = str(payload.get("winner", "tie")).strip().lower()
        if winner not in {"candidate_a", "candidate_b", "tie"}:
            winner = "tie"
        return {
            "winner": winner,
            "score_a": clamp_judge_score(payload.get("score_a")),
            "score_b": clamp_judge_score(payload.get("score_b")),
            "reason": str(payload.get("reason", "")).strip(),
        }
    return {
        "score": clamp_judge_score(payload.get("score")),
        "reason": str(payload.get("reason", "")).strip(),
    }


def clamp_judge_score(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
