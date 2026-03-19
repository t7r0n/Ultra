from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..mcd.inspector import InspectionResult
from ..tools import hwcheck
from ..ultra_mode.fanout import Candidate, UltraProfile
from ..ultra_mode.refine import RefinedCandidate
from ..ultra_mode.selection import SelectionResult

DEFAULT_RUN_ROOT = Path.home() / "ultra_runs"


def ensure_logdir(logdir: Optional[Path]) -> Path:
    if logdir is not None:
        logdir.mkdir(parents=True, exist_ok=True)
        return logdir
    DEFAULT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = DEFAULT_RUN_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_jsonl(path: Path, items: Iterable[Dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items))


def persist_chat_run(
    *,
    logdir: Optional[Path],
    inspection: InspectionResult,
    profile: UltraProfile,
    prompts: List[Dict[str, str]],
    candidates: List[Candidate],
    selection: SelectionResult,
    final: RefinedCandidate,
) -> Path:
    run_dir = ensure_logdir(logdir)
    hardware = hwcheck.collect_system_report()
    run_payload = {
        "inspection": inspection.to_dict(),
        "profile": profile.to_dict(),
        "hardware": hardware,
    }
    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))
    _write_jsonl(run_dir / "prompts.jsonl", prompts)
    _write_jsonl(run_dir / "candidates.jsonl", [candidate.to_dict() for candidate in candidates])
    (run_dir / "selection.json").write_text(json.dumps(selection.to_dict(), indent=2, sort_keys=True))
    final_text = final.content
    (run_dir / "final.txt").write_text(final_text)
    if profile.structured_output.enabled:
        (run_dir / "final.json").write_text(json.dumps({"content": final_text}, indent=2, ensure_ascii=False))
    metrics = {
        "n_candidates": len(candidates),
        "winner_index": selection.candidate.index,
        "tokens": len(final_text.split()),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return run_dir


def prepare_agent_run(*, open_terminal: bool, sandbox: str) -> Dict[str, object]:
    return {
        "open_terminal": open_terminal,
        "sandbox": sandbox,
    }

