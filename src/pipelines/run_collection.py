"""
Collection pipeline (no-memory baseline) CLI.

Per src-structure-refactor.plan.md (90-93):
- Generate InferenceInputExample targets from raw data via generator
- Select inference engine by dataset
- Load generated targets
- Run inference for each target with memory=""
- Save outputs (InferenceOutputExample jsonl)

Only supports:
- GMemory-style datasets: alfworld / fever / pddl
- Dynamic-cheatsheet tasks: DC_TASKS (GameOf24, AIME_*, etc.)

LongMemEval and MemBench are excluded by design.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.data_processing.generators.dynamic_cheatsheet_generator import (
    DC_TASKS,
)
from src.data_processing.generators.agentboard_generator import AGENTBOARD_DATASETS
from src.data_processing.generators.factory import create_generator
from src.data_processing.loader import DataLoader
from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.inference import get_inference_engine
from src.evaluation.factory import run_evaluation
from src.utils import print_example


def _normalize_dataset_for_engine(dataset_name: str) -> str:
    """
    Map dataset_name to the inference-engine dataset key.

    - GMemory-style: unchanged (alfworld/fever/pddl)
    - AgentBoard: unchanged (babyai/scienceworld/jericho)
    - Dynamic-cheatsheet tasks:
        - GameOf24 -> gameof24 (engine factory expects lowercase)
        - other DC_TASKS -> dynamic_cheatsheet (shared adapter)
    """
    if dataset_name in ("alfworld", "fever", "pddl"):
        return dataset_name
    if dataset_name in AGENTBOARD_DATASETS:
        return dataset_name
    if dataset_name == "bigcodebench":
        return "bigcodebench"
    if dataset_name == "GameOf24":
        return "gameof24"
    if dataset_name in DC_TASKS:
        return "dynamic_cheatsheet"
    return dataset_name


@dataclass
class CollectionRunResult:
    work_dir: str
    target_file: str
    output_file: str


def run_collection(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    max_targets: int = -1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    generator_kwargs: Optional[Dict[str, Any]] = None,
    debug: int = -1,
    refresh: bool = False,
    num_workers: int = 1,
) -> CollectionRunResult:
    """
    Run the collection pipeline for one dataset.

    Args:
        dataset_name: dataset key for generator (e.g. 'alfworld', 'fever', 'pddl', 'GameOf24', 'AIME_2024', ...)
        model_name: LLM model name for src.llm
        output_dir: directory to write artifacts
        max_targets: max number of targets to generate (-1 = all)
        engine_kwargs: forwarded to get_inference_engine / BaseInferenceEngine
        generator_kwargs: forwarded to generator.generate_targets
    """
    engine_kwargs = dict(engine_kwargs or {})
    generator_kwargs = dict(generator_kwargs or {})

    work_dir = os.path.join(output_dir, dataset_name, model_name)
    os.makedirs(work_dir, exist_ok=True)

    # 1) generate_target
    target_file = os.path.join(work_dir, "target.jsonl")
    if not os.path.exists(target_file) or refresh:
        gen = create_generator(dataset_name)
        gen.generate_targets(
            output_data_dir=work_dir,
            max_targets=max_targets,
            **generator_kwargs,
        )
    else:
        print(f"target.jsonl already exists at {target_file}")

    # 2) load targets
    loader = DataLoader(target_file=target_file)
    targets_dict = loader.load_targets()
    targets: List[InferenceInputExample] = list(targets_dict.values())
    if debug > 0:
        targets = targets[:debug]
    
    # 3) select engine (thread-local factory to avoid adapter state clashes)
    engine_dataset = _normalize_dataset_for_engine(dataset_name)

    # Probe few_shots_num via a temporary engine instance to determine output subdir
    _probe_engine = get_inference_engine(
        dataset_name=engine_dataset,
        model_name=model_name,
        **engine_kwargs,
    )
    _probe_adapter = getattr(_probe_engine, "adapter", None)
    few_shots_num = getattr(_probe_adapter, "few_shots_num", 0) or 0
    del _probe_engine

    thread_local = threading.local()

    def _get_engine():
        if not hasattr(thread_local, "engine"):
            thread_local.engine = get_inference_engine(
                dataset_name=engine_dataset,
                model_name=model_name,
                **engine_kwargs,
            )
        return thread_local.engine

    # 4) run inference (memory="") and save
    output_file = os.path.join(work_dir, f"zero-shot" if few_shots_num == 0 else f"{few_shots_num}-shot", "target_output.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    tasks = list(enumerate(targets))
    ordered_outputs: List[Optional[InferenceOutputExample]] = [None] * len(tasks)

    if num_workers <= 1:
        for idx, t in tasks:
            print(f"[run_collection] [{idx + 1}/{len(tasks)}] target_id={t.id}")
            ordered_outputs[idx] = _get_engine().run_one(t, memory="", use_memory=False)
    else:
        def _run_task(item):
            idx, t = item
            return idx, _get_engine().run_one(t, memory="", use_memory=False)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {executor.submit(_run_task, item): item for item in tasks}
            done = 0
            for future in as_completed(future_map):
                idx, out_ex = future.result()
                ordered_outputs[idx] = out_ex
                done += 1
                print(f"[run_collection] [{done}/{len(tasks)}] done target_id={out_ex.id}")

    outputs: List[InferenceOutputExample] = [o for o in ordered_outputs if o is not None]
    assert len(outputs) == len(targets)

    with open(output_file, "w", encoding="utf-8") as f:
        for out_ex in outputs:
            f.write(json.dumps(out_ex.to_dict(), ensure_ascii=False) + "\n")

    print_example("First Output Example", outputs[0].to_dict())

    # 5) evaluation
    run_evaluation(
        dataset_name=dataset_name,
        output_file=output_file,
    )

    return CollectionRunResult(
        work_dir=work_dir,
        target_file=target_file,
        output_file=output_file,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MyMem collection (no-memory baseline) pipeline.")
    p.add_argument("--dataset_name", required=True, choices=[
        "alfworld", "fever", "pddl",
        "babyai", "scienceworld",
        "toolbench",
        "bigcodebench",
        "GameOf24", "AIME_2024", "AIME_2025", "AIME_2020_2024",
        "GPQA_Diamond", "MMLU_Pro_Physics", "MMLU_Pro_Engineering",
        "MathEquationBalancer",
    ])
    p.add_argument("--model_name", required=True, help="Model name for src.llm (litellm)")
    p.add_argument("--output_dir", required=True, help="Directory to write run artifacts")
    p.add_argument("--max_targets", type=int, default=-1, help="Max number of targets to generate (-1 = all)")

    # Engine args
    p.add_argument("--few_shots_num", type=int, default=None, help="Few-shots number (gmemory)")
    p.add_argument("--max_tokens", type=int, default=512, help="LLM max_tokens")
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode if supported (decoding hyperparameters are selected by inference factory).",
    )

    p.add_argument("--num_workers", type=int, default=1,
                   help="Parallel workers for inference (each worker uses its own engine).")
    p.add_argument("--debug", type=int, default=-1)
    p.add_argument("--refresh", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.dataset_name == 'fever':
        args.max_targets = 250

    engine_kwargs: Dict[str, Any] = {
        "few_shots_num": args.few_shots_num,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }
    
    print(args)

    result = run_collection(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_targets=args.max_targets,
        engine_kwargs=engine_kwargs,
        generator_kwargs={},
        debug=args.debug,
        refresh=args.refresh,
        num_workers=args.num_workers,
    )

    print("[Collection] Done.")
    print(f"[Collection] work_dir={result.work_dir}")
    print(f"[Collection] target_file={result.target_file}")
    print(f"[Collection] output_file={result.output_file}")


if __name__ == "__main__":
    main()
