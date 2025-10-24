from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional

from ..mcd.inspector import InspectionResult

DEFAULT_RUN_ROOT = pathlib.Path.home() / "ultra_runs"


def ensure_logdir(logdir: Optional[pathlib.Path]) -> pathlib.Path:
    if logdir is not None:
        logdir.mkdir(parents=True, exist_ok=True)
        return logdir
    DEFAULT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = DEFAULT_RUN_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def persist_chat_run(
    *,
    logdir: Optional[pathlib.Path],
    inspection: InspectionResult,
    profile: Dict[str, Any],
    prompts: List[Dict[str, str]],
    candidates: List[Dict[str, Any]],
    selection: Any,
    final: Any,
) -> pathlib.Path:
    run_dir = ensure_logdir(logdir)
    (run_dir / "run.json").write_text(json.dumps({
        "profile": profile,
        "inspection": inspection.to_dict(),
    }, indent=2, sort_keys=True))
    (run_dir / "prompts.jsonl").write_text("\n".join(json.dumps(item) for item in prompts))
    (run_dir / "candidates.jsonl").write_text("\n".join(json.dumps(item) for item in candidates))
    if isinstance(selection, dict):
        selection_payload = selection
    else:
        to_dict = getattr(selection, "to_dict", None)
        selection_payload = to_dict() if callable(to_dict) else {}
    (run_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2))
    final_payload = getattr(final, "content", final)
    (run_dir / "final.txt").write_text(str(final_payload))
    if isinstance(final_payload, dict):
        (run_dir / "final.json").write_text(json.dumps(final_payload, indent=2, sort_keys=True))
    return run_dir


def prepare_agent_run(*, open_terminal: bool, sandbox: str) -> Dict[str, Any]:
    return {
        "open_terminal": open_terminal,
        "sandbox": sandbox,
        "env": {
            "cwd": os.getcwd(),
        },
    }
