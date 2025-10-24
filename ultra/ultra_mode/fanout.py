from __future__ import annotations

import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from ..backends import Completion, SamplingParameters, registry
from ..mcd.inspector import InspectionResult


DEFAULT_PROFILE = {
    "backend": "auto",
    "precision": "auto",
    "context_window": "auto",
    "ultra": {
        "n_candidates": 8,
        "max_n_candidates": 32,
        "diversity": {
            "temperatures": [0.2, 0.6, 0.9],
            "top_p": [0.85, 0.95],
            "styles": ["concise", "cot"],
        },
        "selection": {
            "self_consistency": True,
            "mbr": True,
            "logprobs": True,
        },
        "refine": {
            "self_refine_passes": 1,
            "chain_of_verification": True,
        },
        "max_new_tokens": 1024,
    },
    "structured_output": {"enabled": False, "json_schema": None},
    "long_context": {"allow_rope_scaling": True, "mistral_respect_swa": True},
    "judge": {
        "enabled": False,
        "bias_mitigation": {"shuffle": True, "blind_ids": True},
    },
    "logging": {
        "save_prompts": True,
        "save_candidates": True,
        "save_logits": False,
    },
}


STYLE_HINTS = {
    "concise": "Answer concisely while preserving correctness.",
    "cot": "Reason step-by-step before delivering the final answer.",
    "creative": "Use vivid language and examples to enhance clarity.",
    "coding": "When coding, provide runnable code and explain test strategy.",
}


@dataclass(slots=True)
class SelectionConfig:
    self_consistency: bool
    mbr: bool
    logprobs: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionConfig":
        return cls(
            self_consistency=bool(data.get("self_consistency", True)),
            mbr=bool(data.get("mbr", True)),
            logprobs=bool(data.get("logprobs", True)),
        )


@dataclass(slots=True)
class RefineConfig:
    self_refine_passes: int
    chain_of_verification: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefineConfig":
        return cls(
            self_refine_passes=int(data.get("self_refine_passes", 0)),
            chain_of_verification=bool(data.get("chain_of_verification", False)),
        )


@dataclass(slots=True)
class StructuredOutputConfig:
    enabled: bool
    json_schema: Optional[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredOutputConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            json_schema=data.get("json_schema"),
        )


@dataclass(slots=True)
class UltraSettings:
    n_candidates: int
    max_n_candidates: int
    temperatures: List[float]
    top_p: List[float]
    styles: List[str]
    selection: SelectionConfig
    refine: RefineConfig
    max_new_tokens: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UltraSettings":
        diversity = data.get("diversity", {})
        return cls(
            n_candidates=int(data.get("n_candidates", 1)),
            max_n_candidates=int(data.get("max_n_candidates", data.get("n_candidates", 1))),
            temperatures=[float(value) for value in diversity.get("temperatures", [0.7])],
            top_p=[float(value) for value in diversity.get("top_p", [0.95])],
            styles=list(diversity.get("styles", [])),
            selection=SelectionConfig.from_dict(data.get("selection", {})),
            refine=RefineConfig.from_dict(data.get("refine", {})),
            max_new_tokens=int(data.get("max_new_tokens", 512)),
        )


@dataclass(slots=True)
class UltraProfile:
    name: str
    backend: str
    precision: str
    context_window: Optional[int]
    ultra: UltraSettings
    structured_output: StructuredOutputConfig
    long_context: Dict[str, Any]
    judge: Dict[str, Any]
    logging: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "precision": self.precision,
            "context_window": self.context_window,
            "ultra": {
                "n_candidates": self.ultra.n_candidates,
                "max_n_candidates": self.ultra.max_n_candidates,
                "temperatures": self.ultra.temperatures,
                "top_p": self.ultra.top_p,
                "styles": self.ultra.styles,
                "selection": self.ultra.selection.__dict__,
                "refine": self.ultra.refine.__dict__,
                "max_new_tokens": self.ultra.max_new_tokens,
            },
            "structured_output": {
                "enabled": self.structured_output.enabled,
                "json_schema": self.structured_output.json_schema,
            },
            "long_context": self.long_context,
            "judge": self.judge,
            "logging": self.logging,
        }


@dataclass(slots=True)
class Candidate:
    index: int
    messages: List[Dict[str, str]]
    prompt: str
    sampling: SamplingParameters
    completion: Completion
    style: Optional[str]
    engine: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "messages": self.messages,
            "prompt": self.prompt,
            "sampling": {
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "top_k": self.sampling.top_k,
                "max_new_tokens": self.sampling.max_new_tokens,
                "seed": self.sampling.seed,
            },
            "completion": {
                "text": self.completion.text,
                "logprobs": self.completion.logprobs,
                "metadata": self.completion.metadata,
            },
            "style": self.style,
            "engine": self.engine,
            "metadata": self.metadata,
        }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_ultra_profile(*, config_path: Optional[Path], profile_name: str) -> UltraProfile:
    config_data = json.loads(json.dumps(DEFAULT_PROFILE))
    if config_path is not None:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load custom Ultra profiles")
        loaded = yaml.safe_load(config_path.read_text())
        if "profiles" in loaded:
            try:
                loaded = loaded["profiles"][profile_name]
            except KeyError as exc:
                raise KeyError(f"Profile '{profile_name}' not found in {config_path}") from exc
        config_data = _deep_merge(config_data, loaded)
    ultra_settings = UltraSettings.from_dict(config_data["ultra"])
    profile = UltraProfile(
        name=profile_name,
        backend=config_data.get("backend", "auto"),
        precision=config_data.get("precision", "auto"),
        context_window=(None if config_data.get("context_window") == "auto" else config_data.get("context_window")),
        ultra=ultra_settings,
        structured_output=StructuredOutputConfig.from_dict(config_data.get("structured_output", {})),
        long_context=config_data.get("long_context", {}),
        judge=config_data.get("judge", {}),
        logging=config_data.get("logging", {}),
    )
    return profile


def _resolve_engine(backend: str, profile: UltraProfile, inspection: InspectionResult) -> str:
    if backend != "auto":
        return backend
    if profile.backend != "auto":
        return profile.backend
    if inspection.backends_supported:
        return inspection.backends_supported[0]
    return "hf"


def _cycle(values: Iterable[Any]) -> Iterable[Any]:
    while True:
        for item in values:
            yield item


def _augment_messages(messages: List[Dict[str, str]], style: Optional[str]) -> List[Dict[str, str]]:
    if not style:
        return [dict(message) for message in messages]
    hint = STYLE_HINTS.get(style, "")
    if not hint:
        return [dict(message) for message in messages]
    styled = [dict(message) for message in messages]
    for message in styled:
        if message.get("role") == "system":
            message["content"] = f"{message['content']}\n{hint}" if message["content"] else hint
            break
    else:
        styled.insert(0, {"role": "system", "content": hint})
    return styled


def generate_candidates(
    *,
    inspection: InspectionResult,
    profile: UltraProfile,
    backend: str,
    messages: List[Dict[str, str]],
) -> List[Candidate]:
    engine_name = _resolve_engine(backend, profile, inspection)
    engine = registry.get(engine_name)
    n_candidates = min(profile.ultra.n_candidates, profile.ultra.max_n_candidates)
    temperature_iter = _cycle(profile.ultra.temperatures or [0.7])
    top_p_iter = _cycle(profile.ultra.top_p or [0.95])
    style_iter = _cycle(profile.ultra.styles or [None])

    structured_payload: Optional[Dict[str, Any]] = None
    if profile.structured_output.enabled:
        structured_payload = {"json_schema": profile.structured_output.json_schema}

    def worker(index: int) -> Candidate:
        temperature = next(temperature_iter)
        top_p = next(top_p_iter)
        style = next(style_iter)
        sampling = SamplingParameters(
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=profile.ultra.max_new_tokens,
            seed=random.randint(0, 2**31 - 1),
            stop=inspection.stop_tokens or None,
        )
        styled_messages = _augment_messages(messages, style)
        prompt = inspection.apply_chat_template(styled_messages)
        completion = engine.generate(
            prompt=prompt,
            sampling=sampling,
            inspection=inspection,
            structured=structured_payload,
        )
        metadata = {
            "temperature": temperature,
            "top_p": top_p,
            "style": style,
        }
        return Candidate(
            index=index,
            messages=styled_messages,
            prompt=prompt,
            sampling=sampling,
            completion=completion,
            style=style,
            engine=engine_name,
            metadata=metadata,
        )

    candidates: List[Candidate] = []
    with ThreadPoolExecutor(max_workers=min(4, n_candidates)) as executor:
        futures = {executor.submit(worker, idx): idx for idx in range(n_candidates)}
        for future in as_completed(futures):
            candidates.append(future.result())
    candidates.sort(key=lambda candidate: candidate.index)
    return candidates

