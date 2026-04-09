"""
ACE-based evolution CLI for memory extraction prompts.

Uses ACE's three-role architecture (Generator/Extractor, Reflector, Curator)
to evolve the memory extraction playbook. Offline-only: train on full trainset,
optionally using a validation set for best checkpoint selection.

Usage:
    python -m src.evolution_ace.run_evolution_ace \
        --split_file data_collection/train/global_split.json \
        --save_path results
"""

import argparse
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evolution_gepa.prepare_data import prepare_gepa_data
from src.evolution_ace import ACE, MemoryExtractionDataProcessor


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ACE evolution for memory extraction prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Path to global split JSON (e.g. data_collection/train/global_split.json).",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Base directory for saving results.",
    )
    p.add_argument(
        "--prompt_type",
        type=str,
        default="factual-experiential",
        choices=["factual-experiential", "survey", "simple"],
        help="Seed prompt type for ACE evolution.",
    )
    p.add_argument(
        "--extraction_model",
        type=str,
        default="openai/Qwen3-32B",
        help="Model for memory extraction.",
    )
    p.add_argument(
        "--inference_model",
        type=str,
        default="openai/Qwen3-32B",
        help="Model for target inference.",
    )
    p.add_argument(
        "--reflector_model",
        type=str,
        default="openai/Qwen3-32B",
        help="Model for Reflector.",
    )
    p.add_argument(
        "--curator_model",
        type=str,
        default="openai/Qwen3-32B",
        help="Model for Curator.",
    )
    p.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--max_num_rounds",
        type=int,
        default=3,
        help="Max reflection rounds per sample when incorrect.",
    )
    p.add_argument(
        "--curator_frequency",
        type=int,
        default=1,
        help="Run Curator every N steps.",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Validation evaluation frequency (every N steps). Active only when val_size > 0.",
    )
    p.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save intermediate playbooks every N steps.",
    )
    p.add_argument(
        "--val_size",
        type=float,
        default=0.0,
        help="Validation split ratio in [0,1). 0 = no validation; >0 enables best-checkpoint selection.",
    )
    p.add_argument(
        "--test_workers",
        type=int,
        default=1,
        help="Parallel workers for validation evaluation.",
    )
    p.add_argument(
        "--playbook_token_budget",
        type=int,
        default=80000,
        help="Token budget for playbook.",
    )
    p.add_argument(
        "--use_async",
        action="store_true",
        help="Enable async evolution path (safe mode: extraction concurrency only).",
    )
    p.add_argument(
        "--async_max_concurrency",
        type=int,
        default=8,
        help="Max concurrent extraction requests in async mode.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if not (0.0 <= args.val_size < 1.0):
        raise ValueError(f"val_size must be in [0, 1), got {args.val_size}")
    if args.curator_frequency <= 0:
        raise ValueError(f"curator_frequency must be > 0, got {args.curator_frequency}")
    if args.save_steps <= 0:
        raise ValueError(f"save_steps must be > 0, got {args.save_steps}")
    if args.eval_steps <= 0:
        raise ValueError(f"eval_steps must be > 0, got {args.eval_steps}")
    if args.use_async and args.test_workers > 1:
        raise ValueError(
            "run_evolution_ace: --use_async cannot be combined with --test_workers > 1 "
            "(async mode uses safe extraction-only concurrency)."
        )

    # Load data from prepare_gepa_data using split file
    trainset, valset = prepare_gepa_data(
        datasets=[],
        source_file_paths=[],
        target_file_paths=[],
        split_file_path=args.split_file,
        refresh=False,
        val_size=args.val_size,
    )

    # Convert to task_dict format
    processor = MemoryExtractionDataProcessor()
    train_task_dicts = processor.process_task_data(trainset)
    val_task_dicts = processor.process_task_data(valset)
    val_samples = val_task_dicts if val_task_dicts else None

    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": "memory_extraction",
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "use_async": args.use_async,
        "async_max_concurrency": args.async_max_concurrency,
    }

    ace = ACE(
        extraction_model=args.extraction_model,
        inference_model=args.inference_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=4096,
        use_async=args.use_async,
        async_max_concurrency=args.async_max_concurrency,
        prompt_type=args.prompt_type,
    )
    if args.use_async:
        asyncio.run(
            ace.arun(
                train_samples=train_task_dicts,
                val_samples=val_samples,
                data_processor=processor,
                config=config,
            )
        )
    else:
        ace.run(
            train_samples=train_task_dicts,
            val_samples=val_samples,
            data_processor=processor,
            config=config,
        )


if __name__ == "__main__":
    main()
