"""CLI entrypoint for MemEvolve-style prompt tournament on memory extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evolution_clue.core.auto_prompt_evolver import AutoPromptEvolver
from src.extraction.extraction_prompts import (
    FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT,
    MEM0_SYSTEM_PROMPT,
    SURVEY_SYSTEM_PROMPT,
    SIMPLE_SYSTEM_PROMPT,
)


def _log(msg: str) -> None:
    print(f"[run_evolution_memevolve] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MemEvolve-style tournament evolution for extraction system prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Path to global split JSON (e.g. data_collection/train/global_split_per20.json).",
    )

    p.add_argument("--num_rounds", type=int, default=3)
    p.add_argument("--num_systems", type=int, default=3)
    p.add_argument("--task_batch_x", type=int, default=40)
    p.add_argument("--top_t", type=int, default=2)
    p.add_argument("--extra_sample_y", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--analysis_model", type=str, default="openai/Qwen3-32B")
    p.add_argument("--generation_model", type=str, default="openai/Qwen3-32B")
    p.add_argument("--extraction_model", type=str, default="openai/Qwen3-32B")
    p.add_argument("--inference_model", type=str, default="openai/Qwen3-32B")

    p.add_argument("--work_dir", type=str, default="evolution_runs/myevolution_02")
    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument("--extraction_use_async", action="store_true")
    p.add_argument("--extraction_num_workers", type=int, default=1)
    p.add_argument("--extraction_async_max_concurrency", type=int, default=8)
    p.add_argument("--use_pareto_selection", action="store_true")

    p.add_argument("--clustering_model", type=str, default=None,
                    help="Model for clustering (defaults to analysis_model).")
    p.add_argument("--max_clusters", type=int, default=7,
                    help="Maximum number of extraction-strategy clusters.")

    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--prompt_type",
        type=str,
        choices=["factual-experiential", "survey", "simple", "mem0"],
        default="factual-experiential",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    _log(
        "start: "
        f"split_file={args.split_file} work_dir={args.work_dir} num_rounds={args.num_rounds} "
        f"num_systems={args.num_systems} task_batch_x={args.task_batch_x} top_t={args.top_t} "
        f"extra_sample_y={args.extra_sample_y} max_workers={args.max_workers} "
        f"extraction_use_async={args.extraction_use_async} extraction_num_workers={args.extraction_num_workers} "
        f"extraction_async_max_concurrency={args.extraction_async_max_concurrency} "
        f"prompt_type={args.prompt_type}"
    )

    _PROMPT_MAP = {
        "factual-experiential": FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT,
        "mem0": MEM0_SYSTEM_PROMPT,
        "survey": SURVEY_SYSTEM_PROMPT,
        "simple": SIMPLE_SYSTEM_PROMPT,
    }
    base_system_prompt = _PROMPT_MAP[args.prompt_type]

    evolver = AutoPromptEvolver(
        split_file=args.split_file,
        work_dir=args.work_dir,
        base_system_prompt=base_system_prompt,
        analysis_model=args.analysis_model,
        generation_model=args.generation_model,
        extraction_model=args.extraction_model,
        inference_model=args.inference_model,
        num_systems=args.num_systems,
        task_batch_x=args.task_batch_x,
        top_t=args.top_t,
        extra_sample_y=args.extra_sample_y,
        seed=args.seed,
        max_workers=args.max_workers,
        extraction_use_async=args.extraction_use_async,
        extraction_num_workers=args.extraction_num_workers,
        extraction_async_max_concurrency=args.extraction_async_max_concurrency,
        use_pareto_selection=args.use_pareto_selection,
        resume=args.resume,
        refresh_split=False,
        clustering_model=args.clustering_model,
        max_clusters=args.max_clusters,
    )
    result = evolver.run(num_rounds=args.num_rounds)
    _log("completed.")
    _log(f"rounds={result['rounds']}")
    _log(f"history_path={result['history_path']}")
    _log(f"best_prompt_path={result['best_prompt_path']}")


if __name__ == "__main__":
    main()
