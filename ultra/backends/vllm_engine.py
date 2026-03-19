from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from . import Completion, SamplingParameters, registry

try:  # pragma: no cover - optional dependency
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional dependency
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


@dataclass(slots=True)
class _VLLMInstance:
    llm: object


class VLLMEngine:
    name = "vllm"

    def __init__(self) -> None:
        self._instances: Dict[str, _VLLMInstance] = {}

    def supports_structured_decoding(self) -> bool:
        return True

    def _load(self, model_path: str) -> object:  # pragma: no cover - heavy dependency
        if LLM is None:
            raise RuntimeError("vllm is required for the vLLM backend")
        if model_path not in self._instances:
            self._instances[model_path] = _VLLMInstance(LLM(model=model_path))
        return self._instances[model_path].llm

    def generate(  # pragma: no cover - heavy dependency
        self,
        *,
        prompt: str,
        sampling: SamplingParameters,
        inspection,
        structured: Optional[Dict[str, object]] = None,
    ) -> Completion:
        llm = self._load(str(inspection.source_path))
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            max_tokens=sampling.max_new_tokens,
            repetition_penalty=sampling.repetition_penalty,
            stop=sampling.stop,
            seed=sampling.seed,
        )
        if structured and structured.get("json_schema") is not None:
            params.guided_json = structured.get("json_schema")
        outputs = llm.generate([prompt], params)
        completion = outputs[0].outputs[0]
        logprobs = None
        if completion.logprobs is not None:
            logprobs = [entry.logprob for entry in completion.logprobs]
        return Completion(text=completion.text, logprobs=logprobs, metadata={"backend": self.name})


registry.register(VLLMEngine())

