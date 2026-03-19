from __future__ import annotations

from typing import Dict, Optional

from . import Completion, SamplingParameters, registry


class SGLangEngine:
    name = "sglang"

    def supports_structured_decoding(self) -> bool:
        return True

    def generate(
        self,
        *,
        prompt: str,
        sampling: SamplingParameters,
        inspection,
        structured: Optional[Dict[str, object]] = None,
    ) -> Completion:
        raise RuntimeError("SGLang backend requires the sglang package and runtime server")


registry.register(SGLangEngine())

