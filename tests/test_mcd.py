from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultra.mcd.inspector import inspect_model


class MCDTests(unittest.TestCase):
    def test_inspect_local_hf_config_extracts_attention_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen2ForCausalLM"],
                        "torch_dtype": "bfloat16",
                        "max_position_embeddings": 32768,
                        "num_hidden_layers": 32,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 8,
                        "hidden_size": 4096,
                        "rope_scaling": {
                            "type": "yarn",
                            "factor": 4.0,
                            "original_max_position_embeddings": 8192,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "Qwen2TokenizerFast",
                        "chat_template": "{% for message in messages %}{{ message['role'] }}{% endfor %}system",
                        "model_max_length": 32768,
                        "eos_token": "</s>",
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "processor_config.json").write_text(
                json.dumps({"processor_class": "AutoProcessor", "image_token": "<image>"}),
                encoding="utf-8",
            )
            (model_dir / "generation_config.json").write_text(
                json.dumps({"eos_token_id": 151645, "stop_strings": ["</s>"]}),
                encoding="utf-8",
            )

            inspection = inspect_model(str(model_dir))

        self.assertEqual(inspection.n_kv_heads, 8)
        self.assertEqual(inspection.head_dim, 128)
        self.assertEqual(inspection.context_window, 32768)
        self.assertEqual(inspection.rope_scaling["type"], "yarn")
        self.assertTrue(inspection.chat_template)
        self.assertIn("hf", inspection.backends_supported)
        self.assertEqual(inspection.backend, "hf")
        self.assertTrue(inspection.vision_support)
        self.assertEqual(inspection.tokenizer_info["processor_class"], "AutoProcessor")

    def test_inspect_uses_apply_chat_template_when_tokenizer_is_available(self) -> None:
        class FakeTokenizer:
            bos_token = "<bos>"
            eos_token = "<eos>"
            eos_token_id = 9
            model_max_length = 1234
            chat_template = "raw-template"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return "<system>You are ULTRA.</system><user>Summarize the current plan in one sentence.</user>"

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args, kwargs
                return FakeTokenizer()

        fake_transformers = type("FakeTransformers", (), {"AutoTokenizer": FakeAutoTokenizer})

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen2ForCausalLM"],
                        "torch_dtype": "bfloat16",
                        "max_position_embeddings": 32768,
                        "num_hidden_layers": 32,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 8,
                        "hidden_size": 4096,
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "fallback-template"}),
                encoding="utf-8",
            )
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                inspection = inspect_model(str(model_dir))

        self.assertIn("<system>You are ULTRA.</system>", inspection.chat_template)
        self.assertEqual(inspection.tokenizer_info["tokenizer_class"], "FakeTokenizer")
        self.assertEqual(inspection.eos_tokens, ["<eos>", "9"])

    def test_inspect_local_gguf_extracts_backend_context_and_rope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            self._write_test_gguf(
                gguf_path,
                {
                    "general.architecture": "llama",
                    "llama.context_length": 16384,
                    "llama.block_count": 32,
                    "llama.attention.head_count": 32,
                    "llama.attention.head_count_kv": 8,
                    "llama.embedding_length": 4096,
                    "llama.rope.scaling.type": "yarn",
                    "llama.rope.scaling.factor": 4.0,
                },
            )

            inspection = inspect_model(str(gguf_path))

        self.assertEqual(inspection.backend, "llamacpp")
        self.assertEqual(inspection.context_window, 16384)
        self.assertEqual(inspection.n_kv_heads, 8)
        self.assertEqual(inspection.head_dim, 128)
        self.assertEqual(inspection.rope_scaling["type"], "yarn")
        self.assertIn("RoPE", inspection.notes[0])

    def _write_test_gguf(self, path: Path, metadata: dict[str, object]) -> None:
        type_map = {
            str: 8,
            int: 11,
            float: 12,
            bool: 7,
        }

        def write_string(handle, value: str) -> None:
            encoded = value.encode("utf-8")
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)

        def write_value(handle, value: object) -> None:
            value_type = type_map[type(value)]
            handle.write(struct.pack("<I", value_type))
            if isinstance(value, str):
                write_string(handle, value)
            elif isinstance(value, bool):
                handle.write(struct.pack("<B", int(value)))
            elif isinstance(value, int):
                handle.write(struct.pack("<q", value))
            elif isinstance(value, float):
                handle.write(struct.pack("<d", value))
            else:
                raise AssertionError(f"Unsupported GGUF test value: {value!r}")

        with path.open("wb") as handle:
            handle.write(b"GGUF")
            handle.write(struct.pack("<I", 3))
            handle.write(struct.pack("<Q", 0))
            handle.write(struct.pack("<Q", len(metadata)))
            for key, value in metadata.items():
                write_string(handle, key)
                write_value(handle, value)


if __name__ == "__main__":
    unittest.main()
