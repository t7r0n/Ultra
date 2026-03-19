"""RAG metrics integration."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

from ..config import BackendName, GlobalConfig
from ..runtime import run_ultra_turn
from ..tools.logging import initialize_run_artifacts


SUITE_NAME = "rag"


def describe_suite() -> dict[str, Any]:
    return {
        "suite": SUITE_NAME,
        "status": "implemented",
        "notes": "Uses Ragas when available, otherwise falls back to local heuristic faithfulness, answer relevancy, context precision, and context recall.",
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
        raise RuntimeError("RAG suite requires a path to a local JSON task file.")
    task_file = Path(tasks[0])
    rows = json.loads(task_file.read_text(encoding="utf-8"))
    materialized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        task_config = deepcopy(config)
        prompt = row.get("question", "") + "\n\nContext:\n" + "\n".join(row.get("contexts", []))
        artifacts = initialize_run_artifacts(
            command="eval-rag",
            model_ref=model_ref,
            backend=backend,
            config=task_config.to_dict(),
            hardware=None,
            logdir=task_file.parent / f"runs/rag-task-{index}",
            extra={"task_id": row.get("task_id", index)},
            structured_output=False,
        )
        response = row.get("response")
        if not response:
            turn = run_ultra_turn(
                model_ref=model_ref,
                backend=backend,
                config=task_config,
                ultra_profile="reasoning",
                conversation=[{"role": "user", "content": prompt}],
                artifacts=artifacts,
                seed_base=index * 1000,
            )
            response = turn.final_text
        answer = row.get("answer", "")
        contexts = row.get("contexts", [])
        materialized_rows.append(
            {
                "task_id": row.get("task_id", index),
                "question": row.get("question", ""),
                "ground_truth": answer,
                "contexts": contexts,
                "answer": response,
            }
        )
    metrics = compute_rag_metrics(materialized_rows)
    payload = {
        "suite": SUITE_NAME,
        "metrics": metrics,
        "backend": "ragas" if ragas_available() else "heuristic",
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def compute_rag_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if ragas_available():
        ragas_result = compute_ragas_metrics(rows)
        if ragas_result is not None:
            return ragas_result
    metrics: list[dict[str, Any]] = []
    for row in rows:
        response = str(row.get("answer", ""))
        answer = str(row.get("ground_truth", ""))
        contexts = list(row.get("contexts", []))
        metrics.append(
            {
                "task_id": row.get("task_id"),
                "faithfulness": lexical_overlap(response, answer),
                "answer_relevancy": answer_relevancy(str(row.get("question", "")), response),
                "context_precision": context_precision(response, contexts),
                "context_recall": lexical_overlap(answer, " ".join(contexts)),
                "response": response,
            }
        )
    return metrics


def compute_ragas_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import answer_relevancy as ragas_answer_relevancy  # type: ignore
        from ragas.metrics import context_precision as ragas_context_precision  # type: ignore
        from ragas.metrics import context_recall as ragas_context_recall  # type: ignore
        from ragas.metrics import faithfulness as ragas_faithfulness  # type: ignore
    except Exception:
        return None

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[
            ragas_faithfulness,
            ragas_answer_relevancy,
            ragas_context_precision,
            ragas_context_recall,
        ],
    )
    records = ragas_result_records(result)
    merged: list[dict[str, Any]] = []
    for row, record in zip(rows, records, strict=False):
        merged.append(
            {
                "task_id": row.get("task_id"),
                "faithfulness": record.get("faithfulness"),
                "answer_relevancy": record.get("answer_relevancy"),
                "context_precision": record.get("context_precision"),
                "context_recall": record.get("context_recall"),
                "response": row.get("answer"),
            }
        )
    return merged


def ragas_result_records(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if hasattr(frame, "to_dict"):
            return frame.to_dict(orient="records")
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict):
            keys = list(payload.keys())
            if not keys:
                return []
            row_count = len(payload[keys[0]])
            return [
                {key: payload[key][index] for key in keys}
                for index in range(row_count)
            ]
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def ragas_available() -> bool:
    return importlib.util.find_spec("ragas") is not None and importlib.util.find_spec("datasets") is not None


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.lower().split() if token}
    right_tokens = {token for token in right.lower().split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def answer_relevancy(question: str, response: str) -> float:
    return lexical_overlap(response, question)


def context_precision(response: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    context_tokens = set(" ".join(contexts).lower().split())
    response_tokens = {token for token in response.lower().split() if token}
    if not response_tokens:
        return 0.0
    return len(response_tokens & context_tokens) / len(response_tokens)
