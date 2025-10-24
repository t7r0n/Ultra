from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from . import Completion, SamplingParameters, registry

try:  # pragma: no cover - optional dependency
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore


class LlamaCPPEngine:
    name = "llamacpp"

    def __init__(self) -> None:
        self._instances: Dict[str, object] = {}

    def supports_structured_decoding(self) -> bool:
        return False

    def _resolve_model(self, inspection) -> Path:
        for file_name in inspection.metadata.get("files", []):
            path = inspection.source_path / file_name
            if path.suffix == ".gguf" and path.is_file():
                return path
        raise RuntimeError("No GGUF file found for llama.cpp backend")

    def _load(self, model_path: Path) -> object:  # pragma: no cover - heavy dependency
        if Llama is None:
            raise RuntimeError("llama_cpp_python is required for llama.cpp backend")
        key = str(model_path)
        if key not in self._instances:
            self._instances[key] = Llama(model_path=str(model_path))
        return self._instances[key]

    def generate(  # pragma: no cover - heavy dependency
        self,
        *,
        prompt: str,
        sampling: SamplingParameters,
        inspection,
        structured: Optional[Dict[str, object]] = None,
    ) -> Completion:
        model_path = self._resolve_model(inspection)
        llm = self._load(model_path)
        response = llm(
            prompt,
            max_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
        )
        choice = response["choices"][0]
        text = choice["text"]
        return Completion(text=text, metadata={"backend": self.name, "model_path": str(model_path)})


registry.register(LlamaCPPEngine())

