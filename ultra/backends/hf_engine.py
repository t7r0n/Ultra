from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import Completion, SamplingParameters, registry

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore


def _resolve_model_identifier(path: Path) -> str:
    return str(path)


@dataclass(slots=True)
class _ModelCacheEntry:
    tokenizer: object
    model: object


class HFEngine:
    name = "hf"

    def __init__(self) -> None:
        self._cache: Dict[str, _ModelCacheEntry] = {}

    def supports_structured_decoding(self) -> bool:
        return False

    def _load(self, model_path: Path) -> Tuple[object, object]:  # pragma: no cover - heavy dependency
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("transformers is required for the HF backend")
        model_id = _resolve_model_identifier(model_path)
        if model_id not in self._cache:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            if torch is not None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
            self._cache[model_id] = _ModelCacheEntry(tokenizer=tokenizer, model=model)
        entry = self._cache[model_id]
        return entry.tokenizer, entry.model

    def generate(  # pragma: no cover - heavy dependency
        self,
        *,
        prompt: str,
        sampling: SamplingParameters,
        inspection,
        structured: Optional[Dict[str, object]] = None,
    ) -> Completion:
        tokenizer, model = self._load(inspection.source_path)
        encoded = tokenizer(prompt, return_tensors="pt")
        if torch is not None:
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generation_kwargs: Dict[str, object] = {
            "max_new_tokens": sampling.max_new_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
        }
        if sampling.top_k is not None:
            generation_kwargs["top_k"] = sampling.top_k
        if sampling.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = sampling.repetition_penalty
        if sampling.seed is not None and torch is not None:
            torch.manual_seed(sampling.seed)
            torch.cuda.manual_seed_all(sampling.seed)
        output = model.generate(**encoded, **generation_kwargs)
        generated_tokens = output[0]
        prompt_length = encoded["input_ids"].shape[-1]
        completion_tokens = generated_tokens[prompt_length:]
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return Completion(text=text, metadata={"backend": self.name})


registry.register(HFEngine())

