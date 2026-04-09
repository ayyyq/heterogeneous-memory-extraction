"""
GEPA-based evolution CLI for memory extraction prompts.

This script wires together:
- data_processing generators' outputs (source.jsonl / target.jsonl),
- the unified MyMem extraction / inference / evaluation stack,
- and the GEPA MemoryExtractionAdapter.

Inspecting results (candidates, scores, prompt changes):
- State is saved under --run_dir (e.g. gepa_runs/factual-experiential-0210) as gepa_state.bin.
- Use: python -m src.evolution_gepa.inspect_evolution --run_dir <run_dir>
- Options: --dump_dir to write each candidate's components to files; --diff_seed_best to diff
  seed vs best prompt; --diff_parent_child to diff each candidate vs parent; --trace to print
  iteration trace (selected candidate, subsample scores per iteration).

Usage (example):

    python -m src.evolution_gepa.run_evolution \
        --datasets membench-high,prefeval-mcq \
        --source_files /path/to/membench-high_source.jsonl,/path/to/prefeval-mcq_source.jsonl \
        --target_files /path/to/membench-high_target.jsonl,/path/to/prefeval-mcq_target.jsonl \
        --extraction_model qwen3:32b \
        --inference_model qwen3:32b \
        --reflection_lm openai/gpt-5-mini \
        --max_metric_calls 200 \
        --num_workers 1 \
        --run_dir output/gepa_runs/memory_evolution

    # Or use --num_epochs 5 instead of --max_metric_calls (mutually exclusive).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import gepa

from src.evolution_gepa.memory_extraction_adapter import MemoryExtractionAdapter
from src.evolution_gepa.prepare_data import prepare_gepa_data, save_prepared_data
from src.evolution_gepa.utils import extract_seed_candidate

SUPPORTED_DATASETS = {
    "membench-high",
    "membench-low",
    "personamem-v2-val",
    "prefeval-mcq",
    "alfworld",
    "fever",
    "pddl",
    "babyai",
    "scienceworld",
    "GameOf24",
    "AIME_2024",
    "AIME_2025",
    "AIME_2020_2024",
    "MMLU_Pro_Physics",
    "MMLU_Pro_Engineering",
    "MathEquationBalancer",
    "bigcodebench",
}


class MaxEpochsStopper:
    """Stop after num_epochs full passes over the training set."""

    def __init__(self, num_epochs: int, trainset_size: int, minibatch_size: int):
        iters_per_epoch = max(1, math.ceil(trainset_size / minibatch_size))
        self.max_iterations = num_epochs * iters_per_epoch
        self._last_reported_iteration = -1

    def __call__(self, gepa_state) -> bool:
        # Print progress once per completed iteration so users can track
        # completed vs remaining work, especially in single-epoch runs.
        curr_iter = int(getattr(gepa_state, "i", 0))
        if curr_iter != self._last_reported_iteration:
            completed = min(curr_iter + 1, self.max_iterations)
            remaining = max(0, self.max_iterations - completed)
            print(
                f"[run_evolution][progress] iteration {completed}/{self.max_iterations} "
                f"(remaining: {remaining})"
            )
            self._last_reported_iteration = curr_iter
        return gepa_state.i >= self.max_iterations


def _parse_comma_separated(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run GEPA to evolve memory-extraction prompts on MyMem datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Comma-separated dataset names. Must be within the supported GEPA evolution set. "
            "Required when generating or refreshing a split file."
        ),
    )
    p.add_argument(
        "--source_files",
        type=str,
        default="",
        help=(
            "Comma-separated paths to source.jsonl files (one per dataset). "
            "Required when generating or refreshing a split file."
        ),
    )
    p.add_argument(
        "--target_files",
        type=str,
        default="",
        help=(
            "Comma-separated paths to target.jsonl files (one per dataset). "
            "Required when generating or refreshing a split file."
        ),
    )
    p.add_argument(
        "--extraction_model",
        type=str,
        default="openai/Qwen3-32B",
        help="Model name for memory extraction (src.llm / LiteLLM).",
    )
    p.add_argument(
        "--inference_model",
        type=str,
        default="openai/Qwen3-32B",
        help="Model name for target inference (BaseInferenceEngine).",
    )
    p.add_argument(
        "--reflection_lm",
        type=str,
        default="openai/Qwen3-32B",
        help="GEPA reflection LLM identifier (passed to gepa.optimize).",
    )
    p.add_argument(
        "--val_size",
        type=float,
        default=0.5,
        help="When < 1: validation proportion; when >= 1: validation sample count.",
    )
    p.add_argument(
        "--num_sources_per_dataset",
        type=int,
        default=None,
        help="If set, limit number of sources per dataset for faster debugging.",
    )
    p.add_argument(
        "--max_metric_calls",
        type=int,
        default=None,
        help="Budget for GEPA metric calls. Mutually exclusive with --num_epochs.",
    )
    p.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs. Mutually exclusive with --max_metric_calls.",
    )
    p.add_argument(
        "--reflection_minibatch_size",
        type=int,
        default=5,
        help="GEPA reflection minibatch size.",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default="output/gepa_memory_evolution",
        help="Directory where GEPA will save its state and logs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GEPA data preparation.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Extraction worker count used in MemoryExtractionAdapter (inference remains serial).",
    )
    p.add_argument(
        "--use_async",
        action="store_true",
        help="Enable async evolution path (cannot be combined with --num_workers > 1).",
    )
    p.add_argument(
        "--async_max_concurrency",
        type=int,
        default=8,
        help="Max concurrent extraction requests in async mode.",
    )
    p.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=(
            "Optional path to the global GEPA data split file. If not provided, "
            "defaults to <run_dir>/gepa_global_split.json."
        ),
    )
    p.add_argument(
        "--refresh_split",
        action="store_true",
        help=(
            "Regenerate the global split file even if it already exists. "
            "When set, --datasets/--source_files/--target_files must be provided."
        ),
    )
    p.add_argument(
        "--prompt_type",
        type=str,
        default="factual-experiential",
        help="Survey prompt type for GEPA evolution (e.g. 'factual-experiential', 'survey', 'simple').",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.num_epochs is not None and args.max_metric_calls is not None:
        raise ValueError(
            "[run_evolution] --num_epochs and --max_metric_calls are mutually exclusive. "
            "Provide exactly one of them."
        )
    if args.use_async and args.num_workers > 1:
        raise ValueError(
            "[run_evolution] --use_async cannot be combined with --num_workers > 1."
        )

    # Determine effective split file path.
    split_file_path = args.split_file
    if split_file_path is None:
        split_file_path = str(Path(args.run_dir) / "gepa_global_split.json")

    # Parse comma-separated lists (may be empty when using an existing split).
    datasets = _parse_comma_separated(args.datasets) if args.datasets else []
    source_files = _parse_comma_separated(args.source_files) if args.source_files else []
    target_files = _parse_comma_separated(args.target_files) if args.target_files else []

    using_existing_split = (
        (not args.refresh_split) and split_file_path is not None and os.path.isfile(split_file_path)
    )

    if using_existing_split:
        # Load dataset configuration from the existing global split file.
        print(f"[run_evolution] Using existing split file: {split_file_path}")
        with open(split_file_path, "r", encoding="utf-8") as f:
            split_cfg = json.load(f)
        split_datasets = split_cfg.get("datasets", [])
        if not isinstance(split_datasets, list) or not split_datasets:
            raise ValueError(
                f"[run_evolution] Invalid split file {split_file_path}: "
                "field 'datasets' must be a non-empty list."
            )
        datasets = [d["name"] for d in split_datasets]
        source_files = [d["source_file_path"] for d in split_datasets]
        target_files = [d["target_file_path"] for d in split_datasets]
    else:
        # When generating or refreshing a split, require full dataset/file specs.
        if not (datasets and source_files and target_files):
            raise ValueError(
                "[run_evolution] When no existing split file is available or "
                "--refresh_split is set, --datasets, --source_files and "
                "--target_files must all be provided."
            )
        if not (len(datasets) == len(source_files) == len(target_files)):
            raise ValueError(
                "[run_evolution] datasets, source_files and target_files must "
                "have the same length.\n"
                f"  datasets     = {datasets}\n"
                f"  source_files = {source_files}\n"
                f"  target_files = {target_files}"
            )

    print("[run_evolution] Datasets:", datasets)
    for ds, s, t in zip(datasets, source_files, target_files):
        print(f"[run_evolution]  - {ds}: source={s}, target={t}")

    unsupported = sorted(set(datasets) - SUPPORTED_DATASETS)
    if unsupported:
        raise ValueError(
            "[run_evolution] Unsupported dataset(s): "
            f"{unsupported}. Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )

    # 1) Prepare GEPA train/val sets from existing source/target JSONL and a global split.
    trainset, valset = prepare_gepa_data(
        datasets=datasets,
        source_file_paths=source_files,
        target_file_paths=target_files,
        num_sources_per_dataset=args.num_sources_per_dataset,
        val_size=args.val_size,
        seed=args.seed,
        split_file_path=split_file_path,
        refresh=args.refresh_split,
    )

    print(f"[run_evolution] Train examples: {len(trainset)}, Val examples: {len(valset)}")

    # 2) Build initial candidate from the baseline factual-experiential prompt.
    seed_candidate = extract_seed_candidate(prompt_type=args.prompt_type)
    print("[run_evolution] Seed candidate components:", list(seed_candidate.keys()))

    # 3) Instantiate adapter that plugs GEPA into MyMem's pipelines.
    adapter = MemoryExtractionAdapter(
        extraction_model=args.extraction_model,
        inference_model=args.inference_model,
        num_workers=args.num_workers,
        use_async=args.use_async,
        async_max_concurrency=args.async_max_concurrency,
        verbose=False,
    )

    os.makedirs(args.run_dir, exist_ok=True)

    # Persist train/val metadata for inspection and per-dataset accuracy.
    save_prepared_data(trainset, valset, args.run_dir)

    # Persist data config to allow reconstruction if val_meta.json is missing.
    data_config = {
        "datasets": datasets,
        "source_files": source_files,
        "target_files": target_files,
        "val_size": args.val_size,
        "seed": args.seed,
        "num_sources_per_dataset": args.num_sources_per_dataset,
        "split_file_path": split_file_path,
        "refresh_split": args.refresh_split,
        "num_epochs": args.num_epochs,
        "max_metric_calls": args.max_metric_calls,
        "num_workers": args.num_workers,
        "use_async": args.use_async,
        "async_max_concurrency": args.async_max_concurrency,
        "cache_evaluation": True,
        "skip_perfect_score": True,
    }
    with open(os.path.join(args.run_dir, "gepa_data_config.json"), "w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)

    # 4) Run GEPA optimization.
    optimize_kwargs: dict = {
        "seed_candidate": seed_candidate,
        "trainset": trainset,
        "valset": valset,
        "adapter": adapter,
        "reflection_lm": args.reflection_lm,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "run_dir": args.run_dir,
        "display_progress_bar": True,
        "cache_evaluation": True,
        "skip_perfect_score": True,
    }
    if args.num_epochs is not None:
        optimize_kwargs["max_metric_calls"] = None
        optimize_kwargs["stop_callbacks"] = MaxEpochsStopper(
            args.num_epochs, len(trainset), args.reflection_minibatch_size
        )
    elif args.max_metric_calls is not None:
        optimize_kwargs["max_metric_calls"] = args.max_metric_calls
        optimize_kwargs["stop_callbacks"] = None
    else:
        optimize_kwargs["max_metric_calls"] = None
        optimize_kwargs["stop_callbacks"] = None

    result = gepa.optimize(**optimize_kwargs)

    best_idx = result.best_idx
    best_candidate = result.candidates[best_idx]
    best_score = result.val_aggregate_scores[best_idx] if result.val_aggregate_scores else None

    print("\n[run_evolution] Optimization completed.")
    print(f"[run_evolution] Best candidate index: {best_idx}")
    if best_score is not None:
        print(f"[run_evolution] Best validation score: {best_score:.4f}")
    print("[run_evolution] Best candidate components:")
    for k, v in best_candidate.items():
        print(f"  - {k}: {len(v)} chars")
    print(f"[run_evolution] GEPA state saved under: {args.run_dir}")
    print("[run_evolution] To inspect candidates, scores, and prompt diffs, run:")
    print(f"  python -m src.evolution_gepa.inspect_evolution --run_dir {args.run_dir} --dump_dir {args.run_dir}/candidates --diff_seed_best --trace")


if __name__ == "__main__":
    main()

