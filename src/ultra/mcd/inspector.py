"""Model discovery and introspection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
import struct
from typing import Any, BinaryIO

from ..backends import AutoBackendContext, select_backend
from ..config import BackendName
from ..tools.hwcheck import collect_hardware_report


class ModelReferenceError(RuntimeError):
    """Raised when a model reference cannot be resolved or inspected."""


@dataclass(slots=True)
class ResolvedModelReference:
    requested: str
    resolved_path: Path
    source: str
    backend: BackendName
    gguf: bool
    trust_remote_code: bool = False
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_path"] = str(self.resolved_path)
        return payload


@dataclass(slots=True)
class ModelInspection:
    backend: BackendName
    arch: str | None
    dtype_candidates: list[str]
    context_window: int | None
    sliding_window: int | None
    n_layers: int | None
    n_heads: int | None
    n_kv_heads: int | None
    head_dim: int | None
    hidden_size: int | None
    chat_template: str
    rope_scaling: dict[str, Any] | None
    tokenizer_info: dict[str, Any]
    vision_support: bool
    stop_tokens: list[str]
    eos_tokens: list[str]
    system_prompt_hint: str | None
    backends_supported: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["num_kv_heads"] = self.n_kv_heads
        return payload


def fetch_model_reference(
    model_ref: str,
    *,
    revision: str | None = None,
    local_dir: Path | None = None,
    trust_remote_code: bool = False,
) -> ResolvedModelReference:
    candidate = Path(model_ref)
    if candidate.exists():
        resolved_path = candidate.resolve()
        return ResolvedModelReference(
            requested=model_ref,
            resolved_path=resolved_path,
            source="local",
            backend=_auto_backend_for_path(resolved_path),
            gguf=_contains_gguf(resolved_path),
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModelReferenceError(
            "Remote Hugging Face fetch requires `huggingface_hub`; install dependencies or use a local path."
        ) from exc

    allow_patterns = [
        "*.json",
        "*.model",
        "*.txt",
        "*.tiktoken",
        "*.safetensors",
        "*.bin",
        "*.gguf",
    ]
    try:
        resolved = snapshot_download(
            repo_id=model_ref,
            revision=revision,
            local_dir=str(local_dir) if local_dir else None,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelReferenceError(f"Could not resolve remote model reference {model_ref!r}: {exc}") from exc
    resolved_path = Path(resolved).resolve()
    return ResolvedModelReference(
        requested=model_ref,
        resolved_path=resolved_path,
        source="huggingface",
        backend=_auto_backend_for_path(resolved_path),
        gguf=_contains_gguf(resolved_path),
        trust_remote_code=trust_remote_code,
        revision=revision,
    )


def inspect_model(model_ref: str, *, backend: BackendName = "auto") -> ModelInspection:
    resolved = fetch_model_reference(model_ref)
    selected_backend = _select_backend_for_inspection(resolved, requested_backend=backend)
    if resolved.gguf:
        return _inspect_gguf(resolved, selected_backend)
    return _inspect_transformers_model(resolved, selected_backend)


def _inspect_transformers_model(resolved: ResolvedModelReference, backend: BackendName) -> ModelInspection:
    config_path = resolved.resolved_path / "config.json"
    if not config_path.exists():
        raise ModelReferenceError(f"Could not find config.json under {resolved.resolved_path}")

    config = _load_json(config_path)
    tokenizer_config = _load_json_if_exists(resolved.resolved_path / "tokenizer_config.json")
    generation_config = _load_json_if_exists(resolved.resolved_path / "generation_config.json")

    n_heads = _first_int(config, "num_attention_heads")
    hidden_size = _first_int(config, "hidden_size")
    n_kv_heads = _first_int(config, "num_key_value_heads", default=n_heads)
    head_dim = int(hidden_size / n_heads) if hidden_size and n_heads else None

    chat_template, tokenizer_info, tokenizer_eos_tokens, notes = _load_tokenizer_metadata(
        resolved.resolved_path,
        tokenizer_config=tokenizer_config,
        trust_remote_code=resolved.trust_remote_code,
    )
    eos_tokens = _normalize_tokens(
        generation_config.get("eos_token_id"),
        generation_config.get("eos_token_ids"),
        tokenizer_eos_tokens,
        tokenizer_config.get("eos_token"),
    )
    stop_tokens = _normalize_tokens(generation_config.get("stop_strings"))
    system_prompt_hint = "system role detected in canonical chat template" if "system" in chat_template.lower() else None

    return ModelInspection(
        backend=backend,
        arch=_resolve_architecture(config),
        dtype_candidates=_dtype_candidates(config),
        context_window=_context_window(config, tokenizer_config),
        sliding_window=_first_int(config, "sliding_window"),
        n_layers=_first_int(config, "num_hidden_layers", "n_layer", "num_layers"),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        chat_template=chat_template,
        rope_scaling=config.get("rope_scaling") if isinstance(config.get("rope_scaling"), dict) else None,
        tokenizer_info=tokenizer_info,
        vision_support=_vision_support(resolved.resolved_path, config),
        stop_tokens=stop_tokens,
        eos_tokens=eos_tokens,
        system_prompt_hint=system_prompt_hint,
        backends_supported=["hf", "vllm", "sglang"],
        notes=notes,
    )


def _inspect_gguf(resolved: ResolvedModelReference, backend: BackendName) -> ModelInspection:
    gguf_path = resolved.resolved_path if resolved.resolved_path.is_file() else _first_gguf_file(resolved.resolved_path)
    if gguf_path is None:
        raise ModelReferenceError(f"Could not locate a .gguf file under {resolved.resolved_path}")

    metadata = _read_gguf_metadata(gguf_path)
    arch = str(metadata.get("general.architecture") or "gguf")
    hidden_size = _gguf_int(metadata, f"{arch}.embedding_length", "llama.embedding_length")
    n_heads = _gguf_int(metadata, f"{arch}.attention.head_count", "llama.attention.head_count")
    n_kv_heads = _gguf_int(metadata, f"{arch}.attention.head_count_kv", "llama.attention.head_count_kv", default=n_heads)
    head_dim = int(hidden_size / n_heads) if hidden_size and n_heads else None
    rope_scaling = _gguf_rope_scaling(metadata, arch)
    notes: list[str] = []
    if rope_scaling is not None:
        notes.append("GGUF embedded RoPE metadata detected; llama.cpp should apply file-embedded scaling.")

    chat_template = str(
        metadata.get("tokenizer.chat_template")
        or metadata.get("tokenizer.ggml.chat_template")
        or ""
    )
    tokenizer_info = {
        "tokenizer_class": metadata.get("tokenizer.ggml.model"),
        "model_max_length": _gguf_int(metadata, f"{arch}.context_length", "llama.context_length"),
        "bos_token": metadata.get("tokenizer.ggml.bos_token_id"),
        "eos_token": metadata.get("tokenizer.ggml.eos_token_id"),
    }
    system_prompt_hint = "system role detected in canonical chat template" if "system" in chat_template.lower() else None

    return ModelInspection(
        backend=backend,
        arch=arch,
        dtype_candidates=_gguf_dtype_candidates(metadata),
        context_window=_gguf_int(metadata, f"{arch}.context_length", "llama.context_length"),
        sliding_window=_gguf_int(metadata, f"{arch}.attention.sliding_window", "llama.attention.sliding_window"),
        n_layers=_gguf_int(metadata, f"{arch}.block_count", "llama.block_count"),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        chat_template=chat_template,
        rope_scaling=rope_scaling,
        tokenizer_info=tokenizer_info,
        vision_support=arch.lower() in {"llava", "qwen2vl", "qwen2_5_vl"},
        stop_tokens=[],
        eos_tokens=_normalize_tokens(metadata.get("tokenizer.ggml.eos_token_id")),
        system_prompt_hint=system_prompt_hint,
        backends_supported=["llamacpp"],
        notes=notes,
    )


def _load_tokenizer_metadata(
    model_dir: Path,
    *,
    tokenizer_config: dict[str, Any],
    trust_remote_code: bool,
) -> tuple[str, dict[str, Any], list[str], list[str]]:
    processor_config = _load_json_if_exists(model_dir / "processor_config.json")
    preprocessor_config = _load_json_if_exists(model_dir / "preprocessor_config.json")
    raw_chat_template = str(tokenizer_config.get("chat_template") or "")
    tokenizer_info = {
        "tokenizer_class": tokenizer_config.get("tokenizer_class"),
        "model_max_length": tokenizer_config.get("model_max_length"),
        "bos_token": tokenizer_config.get("bos_token"),
        "eos_token": tokenizer_config.get("eos_token"),
        "raw_chat_template": raw_chat_template,
        "processor_class": processor_config.get("processor_class") or preprocessor_config.get("processor_class"),
        "has_processor_config": bool(processor_config),
        "has_preprocessor_config": bool(preprocessor_config),
        "image_token": processor_config.get("image_token") or preprocessor_config.get("image_token"),
    }
    eos_tokens = _normalize_tokens(tokenizer_config.get("eos_token"))
    notes: list[str] = []
    if processor_config or preprocessor_config:
        notes.append("Processor metadata detected; multimodal preprocessing files are present.")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_info.update(
            {
                "tokenizer_class": tokenizer.__class__.__name__,
                "model_max_length": getattr(tokenizer, "model_max_length", tokenizer_info["model_max_length"]),
                "bos_token": getattr(tokenizer, "bos_token", tokenizer_info["bos_token"]),
                "eos_token": getattr(tokenizer, "eos_token", tokenizer_info["eos_token"]),
                "raw_chat_template": getattr(tokenizer, "chat_template", raw_chat_template),
            }
        )
        eos_tokens = _normalize_tokens(getattr(tokenizer, "eos_token", None), getattr(tokenizer, "eos_token_id", None))
        rendered = _render_chat_template(tokenizer)
        return rendered or raw_chat_template, tokenizer_info, eos_tokens, notes
    except Exception:
        if raw_chat_template:
            notes.append("Tokenizer could not be loaded; using tokenizer_config chat_template fallback.")
        return raw_chat_template, tokenizer_info, eos_tokens, notes


def _render_chat_template(tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": "You are ULTRA."},
        {"role": "user", "content": "Summarize the current plan in one sentence."},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in signature.parameters:
            kwargs["enable_thinking"] = False
    except (TypeError, ValueError):
        pass
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(rendered, list):
        return " ".join(str(item) for item in rendered)
    return str(rendered)


def _select_backend_for_inspection(resolved: ResolvedModelReference, *, requested_backend: BackendName) -> BackendName:
    if requested_backend != "auto":
        return requested_backend
    hardware = collect_hardware_report()
    return select_backend(
        AutoBackendContext(
            gguf=resolved.gguf,
            has_gpu=bool(hardware.gpus),
        )
    )


def _resolve_architecture(config: dict[str, Any]) -> str | None:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    model_type = config.get("model_type")
    return str(model_type) if model_type is not None else None


def _dtype_candidates(config: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("torch_dtype", "dtype"):
        value = config.get(key)
        if isinstance(value, str) and value not in candidates:
            candidates.append(value)
    if not candidates:
        candidates.extend(["bf16", "fp16", "fp32"])
    return candidates


def _gguf_dtype_candidates(metadata: dict[str, Any]) -> list[str]:
    file_type = metadata.get("general.file_type")
    if file_type is None:
        return ["gguf"]
    return [f"gguf_file_type_{file_type}"]


def _context_window(config: dict[str, Any], tokenizer_config: dict[str, Any]) -> int | None:
    for key in ("max_position_embeddings", "n_positions", "seq_length"):
        value = _first_int(config, key)
        if value is not None:
            return value
    model_max_length = tokenizer_config.get("model_max_length")
    return int(model_max_length) if isinstance(model_max_length, int) and model_max_length > 0 else None


def _vision_support(model_dir: Path, config: dict[str, Any]) -> bool:
    if config.get("vision_config") is not None:
        return True
    architectures = config.get("architectures")
    if isinstance(architectures, list) and any(
        isinstance(item, str) and any(marker in item.lower() for marker in ("vl", "vision", "llava", "idefics"))
        for item in architectures
    ):
        return True
    if config.get("vision_tower") is not None or config.get("mm_projector_type") is not None:
        return True
    for file_name in ("processor_config.json", "preprocessor_config.json"):
        if (model_dir / file_name).exists():
            return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ModelReferenceError(f"{path} did not contain a JSON object.")
    return payload


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    return _load_json(path) if path.exists() else {}


def _normalize_tokens(*values: Any) -> list[str]:
    tokens: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            tokens.extend(str(item) for item in value if item is not None)
            continue
        tokens.append(str(value))
    return tokens


def _first_int(config: dict[str, Any], *keys: str, default: int | None = None) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int):
            return value
    return default


def _gguf_int(metadata: dict[str, Any], *keys: str, default: int | None = None) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return default


def _gguf_rope_scaling(metadata: dict[str, Any], arch: str) -> dict[str, Any] | None:
    rope_type = metadata.get(f"{arch}.rope.scaling.type") or metadata.get("llama.rope.scaling.type")
    rope_factor = metadata.get(f"{arch}.rope.scaling.factor") or metadata.get("llama.rope.scaling.factor")
    if rope_type is None and rope_factor is None:
        return None
    payload: dict[str, Any] = {}
    if rope_type is not None:
        payload["type"] = rope_type
    if rope_factor is not None:
        payload["factor"] = rope_factor
    return payload


def _is_gguf_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".gguf"


def _contains_gguf(path: Path) -> bool:
    if _is_gguf_path(path):
        return True
    if not path.is_dir():
        return False
    return any(candidate.suffix.lower() == ".gguf" for candidate in path.iterdir() if candidate.is_file())


def _auto_backend_for_path(path: Path) -> BackendName:
    return "llamacpp" if _contains_gguf(path) else "hf"


def _first_gguf_file(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path
    if not path.is_dir():
        return None
    for candidate in sorted(path.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() == ".gguf":
            return candidate
    return None


def _read_gguf_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"GGUF":
            raise ModelReferenceError(f"{path} is not a valid GGUF file.")
        version = _read_u32(handle)
        if version not in {2, 3}:
            raise ModelReferenceError(f"Unsupported GGUF version {version} in {path}.")
        _tensor_count = _read_u64(handle)
        kv_count = _read_u64(handle)
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            metadata[key] = _read_gguf_value(handle, value_type)
        return metadata


def _read_gguf_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == 0:
        return _read_u8(handle)
    if value_type == 1:
        return _read_i8(handle)
    if value_type == 2:
        return _read_u16(handle)
    if value_type == 3:
        return _read_i16(handle)
    if value_type == 4:
        return _read_u32(handle)
    if value_type == 5:
        return _read_i32(handle)
    if value_type == 6:
        return _read_f32(handle)
    if value_type == 7:
        return bool(_read_u8(handle))
    if value_type == 8:
        return _read_string(handle)
    if value_type == 9:
        item_type = _read_u32(handle)
        length = _read_u64(handle)
        return [_read_gguf_value(handle, item_type) for _ in range(length)]
    if value_type == 10:
        return _read_u64(handle)
    if value_type == 11:
        return _read_i64(handle)
    if value_type == 12:
        return _read_f64(handle)
    raise ModelReferenceError(f"Unsupported GGUF metadata value type: {value_type}")


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    data = handle.read(length)
    if len(data) != length:
        raise ModelReferenceError("Unexpected end of file while reading GGUF string.")
    return data.decode("utf-8")


def _read_u8(handle: BinaryIO) -> int:
    return _unpack("<B", handle, 1)


def _read_i8(handle: BinaryIO) -> int:
    return _unpack("<b", handle, 1)


def _read_u16(handle: BinaryIO) -> int:
    return _unpack("<H", handle, 2)


def _read_i16(handle: BinaryIO) -> int:
    return _unpack("<h", handle, 2)


def _read_u32(handle: BinaryIO) -> int:
    return _unpack("<I", handle, 4)


def _read_i32(handle: BinaryIO) -> int:
    return _unpack("<i", handle, 4)


def _read_u64(handle: BinaryIO) -> int:
    return _unpack("<Q", handle, 8)


def _read_i64(handle: BinaryIO) -> int:
    return _unpack("<q", handle, 8)


def _read_f32(handle: BinaryIO) -> float:
    return _unpack("<f", handle, 4)


def _read_f64(handle: BinaryIO) -> float:
    return _unpack("<d", handle, 8)


def _unpack(fmt: str, handle: BinaryIO, size: int) -> Any:
    data = handle.read(size)
    if len(data) != size:
        raise ModelReferenceError("Unexpected end of file while reading GGUF metadata.")
    return struct.unpack(fmt, data)[0]
