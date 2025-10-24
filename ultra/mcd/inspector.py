from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..tools import hwcheck

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency for remote fetches
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency
    snapshot_download = None  # type: ignore

try:  # pragma: no cover - optional dependency for tokenizer rendering
    import jinja2
except Exception:  # pragma: no cover - optional dependency
    jinja2 = None  # type: ignore


class ModelDiscoveryError(RuntimeError):
    """Raised when inspection cannot complete."""


def _load_json(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise ModelDiscoveryError(f"Missing required file: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_snapshot(model_ref: str) -> Path:
    candidate = Path(model_ref)
    if candidate.exists():
        return candidate
    if "/" not in model_ref:
        raise ModelDiscoveryError(f"Model reference '{model_ref}' does not exist locally")
    if snapshot_download is None:  # pragma: no cover - optional dependency
        raise ModelDiscoveryError("huggingface_hub is required to download remote models")
    payload = hwcheck.fetch_model_snapshot(model_ref=model_ref)
    return Path(payload["path"])


def _detect_backends(path: Path, preferred_backend: str) -> List[str]:
    has_gguf = any(file.suffix == ".gguf" for file in path.rglob("*.gguf"))
    candidates: List[str] = []
    if has_gguf:
        candidates.append("llamacpp")
    else:
        candidates.extend(["hf", "vllm", "sglang"])
    if preferred_backend != "auto" and preferred_backend not in candidates:
        candidates.insert(0, preferred_backend)
    return list(dict.fromkeys(candidates))


def _vision_support(config: Dict[str, Any]) -> bool:
    return bool(
        config.get("vision_config")
        or config.get("mm_projector")
        or config.get("mm_vision_tower")
        or config.get("vision_tower")
    )


def _dtype_candidates(config: Dict[str, Any]) -> List[str]:
    base = ["fp16", "bf16", "fp32"]
    dtype = config.get("torch_dtype")
    if dtype:
        base.append(str(dtype))
    return sorted(dict.fromkeys(base))


def _context_window(config: Dict[str, Any], tokenizer_config: Dict[str, Any]) -> int:
    values: Sequence[int] = []
    for key in (
        "rope_scaling.original_max_position_embeddings",
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
    ):
        current = config
        for chunk in key.split("."):
            if isinstance(current, dict) and chunk in current:
                current = current[chunk]
            else:
                current = None
                break
        if isinstance(current, int):
            values.append(int(current))
    if tokenizer_config.get("model_max_length"):
        values.append(int(tokenizer_config["model_max_length"]))
    if not values:
        return 0
    return max(values)


def _stop_tokens(generation_config: Dict[str, Any]) -> List[int]:
    tokens: List[int] = []
    if "stop" in generation_config and isinstance(generation_config["stop"], list):
        for entry in generation_config["stop"]:
            if isinstance(entry, int):
                tokens.append(entry)
    if "stop_token_ids" in generation_config and isinstance(generation_config["stop_token_ids"], list):
        tokens.extend(int(token) for token in generation_config["stop_token_ids"])
    return list(dict.fromkeys(tokens))


def _system_prompt_hint(config: Dict[str, Any], generation_config: Dict[str, Any]) -> Optional[str]:
    for candidate in (generation_config.get("system_prompt"), config.get("system_prompt")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


class _ChatTemplateAdapter:
    def __init__(self, template: str, *, eos_token: Optional[str]) -> None:
        self.template = template
        self._eos_token = eos_token
        if jinja2 is not None and "{%" in template:
            env = jinja2.Environment(autoescape=False)  # type: ignore[call-arg]
            self._compiled = env.from_string(template)
        else:
            self._compiled = None

    def apply(self, messages: Iterable[Dict[str, str]], add_generation_prompt: bool) -> str:
        if self._compiled is not None:
            return self._compiled.render(messages=list(messages), add_generation_prompt=add_generation_prompt)
        rendered: List[str] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                rendered.append(f"<<SYS>>{content}<</SYS>>")
            elif role == "user":
                rendered.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                rendered.append(f"<|assistant|>\n{content}\n")
            else:
                rendered.append(f"<{role}>\n{content}\n")
        if add_generation_prompt:
            rendered.append("<|assistant|>")
        return "".join(rendered)


@dataclass(slots=True)
class InspectionResult:
    arch: str
    dtype_candidates: List[str]
    context_window: int
    sliding_window: Optional[int]
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    chat_template: str
    rope_scaling: Optional[Dict[str, Any]]
    tokenizer_info: Dict[str, Any]
    vision_support: bool
    stop_tokens: List[int]
    eos_tokens: List[int]
    system_prompt_hint: Optional[str]
    backends_supported: List[str]
    source_path: Path
    _template_adapter: _ChatTemplateAdapter = field(repr=False, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arch": self.arch,
            "dtype_candidates": self.dtype_candidates,
            "context_window": self.context_window,
            "sliding_window": self.sliding_window,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "chat_template": self.chat_template,
            "rope_scaling": self.rope_scaling,
            "tokenizer_info": self.tokenizer_info,
            "vision_support": self.vision_support,
            "stop_tokens": self.stop_tokens,
            "eos_tokens": self.eos_tokens,
            "system_prompt_hint": self.system_prompt_hint,
            "backends_supported": self.backends_supported,
            "source_path": str(self.source_path),
            "metadata": self.metadata,
        }

    def pretty(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def apply_chat_template(self, messages: Iterable[Dict[str, str]]) -> str:
        return self._template_adapter.apply(list(messages), add_generation_prompt=True)


def inspect_model(model_ref: str, preferred_backend: str = "auto") -> InspectionResult:
    path = _maybe_snapshot(model_ref)
    config = _load_json(path / "config.json")
    tokenizer_config = _load_json(path / "tokenizer_config.json", required=False)
    generation_config = _load_json(path / "generation_config.json", required=False)

    arch = str(config.get("model_type", "unknown"))
    n_layers = int(config.get("num_hidden_layers", 0))
    n_heads = int(config.get("num_attention_heads", 0))
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    hidden_size = int(config.get("hidden_size", 0))
    head_dim = int(config.get("head_dim", hidden_size // n_heads if n_heads else 0))
    context_window = _context_window(config, tokenizer_config)
    sliding_window = config.get("sliding_window") or config.get("sliding_window_size")
    rope_scaling = config.get("rope_scaling")
    tokenizer_info = {
        "model_max_length": tokenizer_config.get("model_max_length", context_window),
        "padding_side": tokenizer_config.get("padding_side", "right"),
        "special_tokens_map": {
            "eos_token": tokenizer_config.get("eos_token"),
            "pad_token": tokenizer_config.get("pad_token"),
            "bos_token": tokenizer_config.get("bos_token"),
            "additional_special_tokens": tokenizer_config.get("additional_special_tokens", []),
        },
    }
    stop_tokens = _stop_tokens(generation_config)
    eos_tokens: List[int] = list(stop_tokens)
    eos_token_id = generation_config.get("eos_token_id") or config.get("eos_token_id")
    if isinstance(eos_token_id, int) and eos_token_id not in eos_tokens:
        eos_tokens.append(eos_token_id)
    system_prompt_hint = _system_prompt_hint(config, generation_config)
    dtype_candidates = _dtype_candidates(config)
    backends_supported = _detect_backends(path, preferred_backend)
    metadata = {
        "files": sorted(str(p.relative_to(path)) for p in path.rglob("*")),
        "rope_scaling_note": "gguf_auto" if any(p.suffix == ".gguf" for p in path.rglob("*.gguf")) else None,
    }

    chat_template = tokenizer_config.get("chat_template", "")
    adapter = _ChatTemplateAdapter(chat_template, eos_token=tokenizer_config.get("eos_token"))

    return InspectionResult(
        arch=arch,
        dtype_candidates=dtype_candidates,
        context_window=int(context_window),
        sliding_window=int(sliding_window) if sliding_window is not None else None,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        chat_template=chat_template,
        rope_scaling=rope_scaling,
        tokenizer_info=tokenizer_info,
        vision_support=_vision_support(config),
        stop_tokens=stop_tokens,
        eos_tokens=eos_tokens,
        system_prompt_hint=system_prompt_hint,
        backends_supported=backends_supported,
        source_path=path,
        _template_adapter=adapter,
        metadata=metadata,
    )

