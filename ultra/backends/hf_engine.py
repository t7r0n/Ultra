from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

from . import registry

LOGGER = logging.getLogger(__name__)


@dataclass
class HFCompletion:
    text: str
    logprobs: Optional[List[float]] = None


class HFEngine:
    """Transformers backend."""

    name = "hf"

    def __init__(self) -> None:
        self._cache = {}

    def _load(self, model_ref: str) -> tuple:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("transformers is required for HF backend")
        if model_ref not in self._cache:
            tokenizer = AutoTokenizer.from_pretrained(model_ref)
            model = AutoModelForCausalLM.from_pretrained(model_ref)
            if torch is not None:
                model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self._cache[model_ref] = (tokenizer, model)
        return self._cache[model_ref]

    def generate(self, prompt: str, **kwargs) -> HFCompletion:  # pragma: no cover - heavy dependency
        model_ref = kwargs.get("model_ref")
        if model_ref is None:
            raise ValueError("model_ref must be provided")
        tokenizer, model = self._load(model_ref)
        encoded = tokenizer(prompt, return_tensors="pt")
        if torch is not None:
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
        output = model.generate(**encoded, max_new_tokens=kwargs.get("max_tokens", 512))
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return HFCompletion(text=text)

    def supports_structured_decoding(self) -> bool:
        return False


registry.register(HFEngine())
