ULTRA is a spec-driven scaffold for an extreme-mode inference and evaluation CLI for open-source LLMs.

`Plan.md` is the authoritative product specification. The code in `src/ultra/` now mirrors that spec at the scaffold level: the package imports cleanly, the command surface exists, the config schema matches `ultra.yaml`, and the named backend/eval/ultra-mode modules are present.

Current scope:
- `doctor`, `inspect`, and `estimate` have working scaffold implementations.
- `fetch` works for local paths and can use `huggingface_hub` when installed.
- `chat`, `agent`, and `eval` initialize spec-shaped run artifacts and report scaffold status rather than pretending the full pipeline exists.

Run the CLI with `python -m ultra --help` or `ultra --help` once the package is on `PYTHONPATH` or installed in editable mode.
