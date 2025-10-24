from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency
    snapshot_download = None  # type: ignore


@dataclass(slots=True)
class InspectionResult:
    """Structured representation of model discovery output."""

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
    source_path: pathlib.Path
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
        payload = self.to_dict()
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    def apply_chat_template(self, messages: Iterable[Dict[str, str]]) -> str:
        template = self.chat_template
        if "{%" not in template:
            # Static template, return as-is with appended assistant tag.
            return template
        rendered: List[str] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                rendered.append(f"<<SYS>>{content}<</SYS>>")
            elif role == "user":
                rendered.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                rendered.append(f"<|assistant|>\n{content}\n")
        if not rendered or not rendered[-1].endswith("<|assistant|>\n"):
            rendered.append("<|assistant|>")
        return "".join(rendered)


class ModelDiscoveryError(RuntimeError):
    """Raised when inspection fails."""


def inspect_model(model_ref: str, preferred_backend: str = "auto") -> InspectionResult:
    path = resolve_model_path(model_ref)
    config = load_json(path / "config.json")
    tokenizer_config = load_optional_json(path / "tokenizer_config.json")
    generation_config = load_optional_json(path / "generation_config.json")

    arch = str(config.get("model_type", "unknown"))
    dtype_candidates = sorted({
        str(config.get("torch_dtype", "auto")),
        "fp16",
        "bf16",
        "fp32",
    })
    context_window = int(
        config.get(
            "max_position_embeddings",
            tokenizer_config.get("model_max_length", 0) if tokenizer_config else 0,
        )
    )
    sliding_window = config.get("sliding_window")
    n_layers = int(config.get("num_hidden_layers", 0))
    n_heads = int(config.get("num_attention_heads", 0))
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    hidden_size = int(config.get("hidden_size", 0))
    head_dim = int(hidden_size / n_heads) if n_heads else 0
    chat_template = (tokenizer_config or {}).get("chat_template", "")
    rope_scaling = config.get("rope_scaling")
    tokenizer_info = {
        "model_max_length": (tokenizer_config or {}).get("model_max_length", context_window),
        "padding_side": (tokenizer_config or {}).get("padding_side", "right"),
    }
    vision_support = "vision_config" in config
    stop_tokens = list((generation_config or {}).get("stop_token_ids", []))
    eos_token_id = (generation_config or {}).get("eos_token_id")
    eos_tokens = stop_tokens.copy()
    if eos_token_id is not None and eos_token_id not in eos_tokens:
        eos_tokens.append(eos_token_id)
    system_prompt_hint = (generation_config or {}).get("system_prompt")
    backends_supported = resolve_backends(path, preferred_backend)
    metadata = {
        "files": sorted(p.name for p in path.iterdir()),
        "config": config,
        "generation_config": generation_config,
        "tokenizer_config": tokenizer_config,
    }

    return InspectionResult(
        arch=arch,
        dtype_candidates=dtype_candidates,
        context_window=context_window,
        sliding_window=int(sliding_window) if sliding_window is not None else None,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        chat_template=chat_template,
        rope_scaling=rope_scaling,
        tokenizer_info=tokenizer_info,
        vision_support=vision_support,
        stop_tokens=stop_tokens,
        eos_tokens=eos_tokens,
        system_prompt_hint=system_prompt_hint,
        backends_supported=backends_supported,
        source_path=path,
        metadata=metadata,
    )


def resolve_model_path(model_ref: str) -> pathlib.Path:
    path = pathlib.Path(model_ref)
    if path.is_file() and path.suffix == ".gguf":
        return path.parent
    if path.exists():
        return path
    if snapshot_download is None:  # pragma: no cover - network download
        raise ModelDiscoveryError("huggingface_hub is required to download remote models")
    downloaded = snapshot_download(model_ref)
    return pathlib.Path(downloaded)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        raise ModelDiscoveryError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_backends(path: pathlib.Path, preferred_backend: str) -> List[str]:
    backends: List[str] = []
    if any(p.suffix == ".gguf" for p in path.iterdir()):
        backends.append("llamacpp")
    else:
        backends.extend(["hf", "vllm"])
    if preferred_backend != "auto" and preferred_backend not in backends:
        backends.insert(0, preferred_backend)
    return sorted(set(backends), key=backends.index)
