"""
Pipelines for MyMem.

Currently includes:
- run_collection: raw_data collection / no-memory baseline inference (CLI + function)
- run_generation: memory extraction from generated sources (CLI + function)
- run_evaluation: load source_output + target, run inference (optional memory), then evaluation (CLI + function)
- run_gepa_evaluation: extraction by prompt (or no_memory/gepa) then inference + evaluation, multi-dataset (CLI + function)
- run_llm_judge_evaluation: LLM-as-judge evaluation for datasets like LongMemEval (CLI + function)
- print_evaluation: single (re-print n output files) or pair (compare no_memory vs with_memory) (CLI + functions)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_collection",
    "run_generation",
    "run_evaluation_pipeline",
    "run_gepa_evaluation",
    "run_llm_judge_evaluation",
    "run_single_mode",
    "run_pair_mode",
]


def __getattr__(name: str) -> Any:
    # Keep package imports lazy to avoid importing CLI modules during `python -m`.
    if name == "run_collection":
        from .run_collection import run_collection as fn
        return fn
    if name == "run_generation":
        from .run_generation import run_generation as fn
        return fn
    if name == "run_evaluation_pipeline":
        from .run_evaluation import run_evaluation_pipeline as fn
        return fn
    if name == "run_gepa_evaluation":
        from .run_gepa_evaluation import run_gepa_evaluation as fn
        return fn
    if name == "run_llm_judge_evaluation":
        from .run_llm_judge_evaluation import run_llm_judge_evaluation as fn
        return fn
    if name == "run_single_mode":
        from .print_evaluation import run_single_mode as fn
        return fn
    if name == "run_pair_mode":
        from .print_evaluation import run_pair_mode as fn
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

