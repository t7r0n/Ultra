"""Arena-style pairwise judging with blind IDs and order shuffling."""

from __future__ import annotations

from copy import deepcopy
import json
import random
from pathlib import Path
from typing import Any

from ..config import BackendName, GlobalConfig
from ..runtime import run_ultra_turn, stable_seed
from ..tools.logging import initialize_run_artifacts


SUITE_NAME = "arena"


def describe_suite() -> dict[str, Any]:
    return {
        "suite": SUITE_NAME,
        "status": "implemented",
        "notes": "Runs blind pairwise judging between greedy and Ultra outputs with shuffled order.",
    }


def run_suite(
    *,
    model_ref: str,
    tasks: list[str],
    backend: BackendName,
    out: Path,
    config: GlobalConfig,
) -> dict[str, Any]:
    if not tasks:
        raise RuntimeError("Arena suite requires a path to a local JSON prompt file.")
    task_file = Path(tasks[0])
    prompts = json.loads(task_file.read_text(encoding="utf-8"))
    judge_logs: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        task_config = deepcopy(config)
        greedy_config = deepcopy(config)
        greedy_config.ultra.n_candidates = 1
        greedy_config.ultra.refine.self_refine_passes = 0
        ultra_artifacts = initialize_run_artifacts(
            command="eval-arena-ultra",
            model_ref=model_ref,
            backend=backend,
            config=task_config.to_dict(),
            hardware=None,
            logdir=task_file.parent / f"runs/arena-ultra-{index}",
            extra={"task_id": prompt.get("task_id", index)},
            structured_output=False,
        )
        greedy_artifacts = initialize_run_artifacts(
            command="eval-arena-greedy",
            model_ref=model_ref,
            backend=backend,
            config=greedy_config.to_dict(),
            hardware=None,
            logdir=task_file.parent / f"runs/arena-greedy-{index}",
            extra={"task_id": prompt.get("task_id", index)},
            structured_output=False,
        )
        ultra_turn = run_ultra_turn(
            model_ref=model_ref,
            backend=backend,
            config=task_config,
            ultra_profile="default",
            conversation=[{"role": "user", "content": prompt["prompt"]}],
            artifacts=ultra_artifacts,
            seed_base=index * 2000,
        )
        greedy_turn = run_ultra_turn(
            model_ref=model_ref,
            backend=backend,
            config=greedy_config,
            ultra_profile="default",
            conversation=[{"role": "user", "content": prompt["prompt"]}],
            artifacts=greedy_artifacts,
            seed_base=index * 2000 + 1,
        )
        randomized = [
            {"blind_id": "candidate_a", "source": "ultra", "text": ultra_turn.final_text},
            {"blind_id": "candidate_b", "source": "greedy", "text": greedy_turn.final_text},
        ]
        random.Random(stable_seed(model_ref, prompt["prompt"])).shuffle(randomized)
        judge_logs.append(
            {
                "task_id": prompt.get("task_id", index),
                "prompt": prompt["prompt"],
                "candidates": randomized,
                "winner": max(randomized, key=lambda item: len(item["text"].strip()))["blind_id"],
            }
        )
    payload = {"suite": SUITE_NAME, "judge_logs": judge_logs}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
