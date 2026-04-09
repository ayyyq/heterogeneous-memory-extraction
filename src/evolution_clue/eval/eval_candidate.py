"""Evaluate one extraction prompt candidate on a batch and write pair-level logs."""

from __future__ import annotations

import json
from pathlib import Path

from src.evolution_gepa.memory_extraction_adapter import (
    MemoryExtractionAdapter,
    MemoryExtractionDataInst,
)
from src.evolution_clue.schemas import EvalSummary


def _log(msg: str) -> None:
    print(f"[memevolve] {msg}", flush=True)


def evaluate_prompt_on_batch(
    batch: list[MemoryExtractionDataInst],
    prompt_id: str,
    system_prompt: str,
    extraction_model: str,
    inference_model: str,
    logs_dir: str,
    num_workers: int = 1,
    use_async: bool = False,
    async_max_concurrency: int = 8,
    progress_callback: bool = False,
) -> EvalSummary:
    """
    Evaluate a prompt candidate on a batch using MemoryExtractionAdapter.

    Notes:
    - This path does not perform any online memory update.
    - Pair-level logs are written for analysis tooling.
    """
    out_dir = Path(logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(
        f"eval_candidate start: prompt_id={prompt_id} batch_size={len(batch)} logs_dir={out_dir}"
    )

    adapter = MemoryExtractionAdapter(
        extraction_model=extraction_model,
        inference_model=inference_model,
        num_workers=num_workers,
        use_async=use_async,
        async_max_concurrency=async_max_concurrency,
        verbose=False,
    )
    candidate = {"system_prompt": system_prompt}
    eval_batch = adapter.evaluate(
        batch,
        candidate,
        capture_traces=True,
        progress_callback=progress_callback,
    )

    scores = [float(s) for s in eval_batch.scores]

    trajectories = eval_batch.trajectories or []
    for i, data_inst in enumerate(batch):
        source = data_inst["source"]
        score = scores[i]
        trace = trajectories[i] if i < len(trajectories) else {}

        rec = {
            "source_conversation": source.get_conversation_text() or "",
            "extracted_memory": trace.get("extraction_output", ""),
            "target_conversation": trace.get("target_inference_trajectory", ""),
            "target_reward": score,
        }
        with open(out_dir / f"{i:04d}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

    num_samples = len(scores)
    accuracy = (sum(scores) / num_samples) if num_samples else 0.0

    summary = {
        "prompt_id": prompt_id,
        "logs_dir": str(out_dir),
        "num_samples": num_samples,
        "accuracy": accuracy,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _log(
        f"eval_candidate end: prompt_id={prompt_id} accuracy={accuracy:.6f} summary_path={out_dir / 'summary.json'}"
    )
    return summary
