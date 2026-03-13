"""Configuration helpers for ULTRA."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, get_args, get_origin, get_type_hints


DEFAULT_CONFIG_FILENAME = "ultra.yaml"
BackendName = Literal["auto", "vllm", "hf", "sglang", "llamacpp"]
PrecisionName = Literal["auto", "fp16", "bf16", "fp32"]
UltraProfileName = Literal["default", "reasoning", "coding", "creative"]
EvalSuiteName = Literal["harness", "code", "long", "arena", "rag"]
SandboxName = Literal["docker", "firejail"]


class ConfigError(RuntimeError):
    """Raised when the user-facing config file cannot be parsed."""


def _deep_update(base: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            if not isinstance(node, MutableMapping):
                raise ConfigError(f"Cannot merge mapping into scalar at key '{key}'")
            _deep_update(node, value)
        else:
            base[key] = value
    return base


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [part.strip() for part in inner.split(",")]
        return [_parse_scalar(part) for part in parts]
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if data is None:
            return {}
        if isinstance(data, Mapping):
            return dict(data)
        raise ConfigError("YAML root must be a mapping")
    except ModuleNotFoundError:
        pass

    root: Dict[str, Any] = {}
    stack: List[tuple[int, MutableMapping[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        stripped = line.strip()
        if stripped.endswith(":") and ":" not in stripped[:-1]:
            key = stripped[:-1].strip()
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
            continue
        if ":" not in stripped:
            raise ConfigError(f"Invalid line in YAML: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        parent[key.strip()] = _parse_scalar(raw_value.strip())
    return root


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
class StructuredOutputConfig:
    enabled: bool = False
    json_schema: Dict[str, Any] | None = None
    regex: str | None = None
    grammar: str | None = None


@dataclass
class LongContextConfig:
    allow_rope_scaling: bool = True
    mistral_respect_swa: bool = True


@dataclass
class JudgeBiasMitigation:
    shuffle: bool = True
    blind_ids: bool = True


@dataclass
class JudgeConfig:
    enabled: bool = False
    model_ref: str | None = None
    backend: BackendName = "auto"
    top_k: int = 4
    pairwise: bool = True
    weight: float = 0.35
    bias_mitigation: JudgeBiasMitigation = field(default_factory=JudgeBiasMitigation)


@dataclass
class LoggingConfig:
    save_prompts: bool = True
    save_candidates: bool = True
    save_logits: bool = False


@dataclass
class UltraSection:
    n_candidates: int = 8
    max_n_candidates: int = 32
    diversity: UltraDiversityConfig = field(default_factory=UltraDiversityConfig)
    selection: UltraSelectionConfig = field(default_factory=UltraSelectionConfig)
    refine: UltraRefineConfig = field(default_factory=UltraRefineConfig)


@dataclass
class GlobalConfig:
    backend: BackendName = "auto"
    precision: PrecisionName = "auto"
    context_window: int | str = "auto"
    ultra: UltraSection = field(default_factory=UltraSection)
    structured_output: StructuredOutputConfig = field(default_factory=StructuredOutputConfig)
    long_context: LongContextConfig = field(default_factory=LongContextConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_config() -> GlobalConfig:
    return GlobalConfig()


def default_config_path(cwd: Path | None = None) -> Path | None:
    base_dir = cwd or Path.cwd()
    candidate = base_dir / DEFAULT_CONFIG_FILENAME
    return candidate if candidate.exists() else None


def load_config(path: Path | None) -> GlobalConfig:
    config = default_config()
    if not path:
        return config
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    data = _load_yaml(path)
    merged = _deep_update(config.to_dict(), data)
    return _dict_to_config(merged)


def _dict_to_config(data: Mapping[str, Any]) -> GlobalConfig:
    return _dict_to_dataclass(GlobalConfig, data)


def _dict_to_dataclass(cls: type, payload: Mapping[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field_info in fields(cls):
        key = field_info.name
        annotation = type_hints.get(key, field_info.type)
        if key in payload:
            value = payload[key]
        elif field_info.default is not MISSING:
            value = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = field_info.default_factory()  # type: ignore[attr-defined]
        else:
            value = None
        kwargs[key] = _coerce_value(annotation, value)
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    dataclass_type = _resolve_dataclass_type(annotation)
    if dataclass_type is not None and isinstance(value, Mapping):
        return _dict_to_dataclass(dataclass_type, value)
    origin = get_origin(annotation)
    if origin in {list, List} and isinstance(value, list):
        item_type = get_args(annotation)[0]
        return [_coerce_value(item_type, item) for item in value]
    return value


def _resolve_dataclass_type(annotation: Any) -> type | None:
    if isinstance(annotation, str):
        return None
    if is_dataclass(annotation):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return annotation if is_dataclass(annotation) else None
    for candidate in get_args(annotation):
        resolved = _resolve_dataclass_type(candidate)
        if resolved is not None:
            return resolved
    return None
