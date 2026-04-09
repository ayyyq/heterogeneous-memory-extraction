"""
Evaluation pipeline CLI.

Similar to run.py inference flow:
1. Load source_output.jsonl (SourceExample with memory and target_ids) and target.jsonl (InferenceInputExample).
2. Select inference engine by dataset.
3. Run inference per (source, target) with optional memory.
4. Run evaluation on the output file.

Supports the same datasets as run_collection / run_extraction:
- GMemory-style: alfworld / fever / pddl
- Dynamic-cheatsheet: DC_TASKS
- LongMemEval: longmemeval
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.data_processing.generators.dynamic_cheatsheet_generator import DC_TASKS
from src.data_processing.loader import DataLoader
from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample, SourceExample
from src.evaluation.factory import run_evaluation
from src.inference import get_inference_engine
from src.utils import print_example


def _normalize_dataset_for_engine(dataset_name: str) -> str:
    """
    Map dataset_name to the inference-engine dataset key.
    Must match run_collection / memory_extraction_adapter.
    """
    if dataset_name in ("alfworld", "fever", "pddl"):
        return dataset_name
    if dataset_name == "longmemeval":
        return "longmemeval"
    if dataset_name in ("membench", "membench-high", "membench-low"):
        return "membench"
    if dataset_name == "GameOf24":
        return "gameof24"
    if dataset_name in ("personamem-v2-val",):
        return "personamem-v2"
    if dataset_name == "prefeval-mcq":
        return "prefeval-mcq"
    if dataset_name == "bigcodebench":
        return "bigcodebench"
    if dataset_name in DC_TASKS:
        return "dynamic_cheatsheet"
    return dataset_name


@dataclass
class EvaluationRunResult:
    work_dir: str
    source_file: str
    target_file: str
    output_file: str
    use_memory: bool


def run_evaluation_pipeline(
    dataset_name: str,
    model_name: str,
    work_dir: str,
    use_memory: bool = True,
    max_targets_per_source: Optional[int] = None,
    source_file: Optional[str] = None,
    target_file: Optional[str] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    debug: int = -1,
    num_workers: int = 1,
) -> EvaluationRunResult:
    """
    Run the evaluation pipeline: load sources (with memory) + targets, run inference, evaluate.

    Steps:
    1) Load source_output.jsonl (SourceExample with memory, target_ids) and target.jsonl.
    2) Select engine via get_inference_engine(normalized_dataset, model_name).
    3) For each source, for each target_id in source.target_ids, run engine.run_one(target, memory).
    4) Write InferenceOutputExample jsonl and call run_evaluation.

    num_workers > 1 enables parallel inference across targets (each worker uses its own engine).

    If work_dir is provided, it is used as the working directory and output is written to
    work_dir/source_target_output.jsonl (no use_memory/no_memory subdir). Otherwise
    work_dir = output_dir/data/dataset_name/model_name and output goes to work_dir/mode_dir/.
    """
    engine_kwargs = dict(engine_kwargs or {})

    work_dir = os.path.normpath(work_dir)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    source_path = source_file or os.path.join(work_dir, "source.jsonl")
    target_path = target_file or os.path.join(work_dir, "target.jsonl")
    output_file = os.path.join(work_dir, "source_target_output.jsonl")

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Source file not found: {source_path}. Run extraction pipeline first."
        )
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Target file not found: {target_path}. Run collection pipeline first."
        )

    # 1) Load sources and targets
    loader = DataLoader(source_file=source_path, target_file=target_path)
    sources_dict = loader.load_sources()
    targets_dict = loader.load_targets()
    sources: List[SourceExample] = list(sources_dict.values())
    targets: List[InferenceInputExample] = list(targets_dict.values())
    target_index = {t.id: t for t in targets}

    if debug > 0:
        sources = sources[:debug]

    def _iter_target_ids(source: SourceExample) -> List[str]:
        ids = list(source.target_ids or [])
        if max_targets_per_source is not None and max_targets_per_source >= 0:
            ids = ids[: max_targets_per_source]
        return ids

    total_pairs = sum(len(_iter_target_ids(s)) for s in sources)
    if total_pairs == 0:
        raise ValueError("No source->target mappings (target_ids empty). Check source_output.jsonl.")

    print_example("First Source Example", sources[0].to_dict())
    print_example("First Target Example", targets[0].to_dict())

    # 2) Select engine factory (per-thread to avoid adapter state clashes)
    engine_dataset = _normalize_dataset_for_engine(dataset_name)
    thread_local = threading.local()

    def _get_engine():
        engine = getattr(thread_local, "engine", None)
        if engine is None:
            engine = get_inference_engine(
                dataset_name=engine_dataset,
                model_name=model_name,
                **engine_kwargs,
            )
            thread_local.engine = engine
        return engine

    # 3) Run inference for each (source, target_id)
    outputs: List[InferenceOutputExample] = []
    processed = 0
    tasks: List[tuple[int, str, InferenceInputExample, str]] = []
    task_index = 0
    for source in sources:
        memory = source.memory if use_memory else ""
        for target_id in _iter_target_ids(source):
            target = target_index.get(target_id)
            if not target:
                print(f"[Warning] Target id {target_id} not found, skipping")
                continue
            tasks.append((task_index, source.id, target, memory))
            task_index += 1

    if num_workers is None or num_workers < 1:
        num_workers = 1

    if num_workers == 1:
        for idx, source_id, target, memory in tasks:
            processed += 1
            print(f"[run_evaluation] [{processed}/{total_pairs}] source_id={source_id} -> target_id={target.id}")
            out_ex = _get_engine().run_one(target, memory=memory, use_memory=use_memory)
            out_ex.metadata["source_id"] = source_id
            outputs.append(out_ex)
    else:
        ordered_outputs: List[Optional[InferenceOutputExample]] = [None] * len(tasks)

        def _run_task(item: tuple[int, str, InferenceInputExample, str]):
            idx, source_id, target, memory = item
            out_ex = _get_engine().run_one(target, memory=memory, use_memory=use_memory)
            out_ex.metadata["source_id"] = source_id
            return idx, out_ex

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {executor.submit(_run_task, item): item for item in tasks}
            for future in as_completed(future_map):
                idx, out_ex = future.result()
                ordered_outputs[idx] = out_ex
                processed += 1
                source_id = out_ex.metadata.get("source_id", "unknown")
                print(f"[run_evaluation] [{processed}/{total_pairs}] done source_id={source_id}")

        outputs = [out for out in ordered_outputs if out is not None]

    with open(output_file, "w", encoding="utf-8") as f:
        for out_ex in outputs:
            f.write(json.dumps(out_ex.to_dict(), ensure_ascii=False) + "\n")

    print_example("First Output Example", outputs[0].to_dict())
    print(f"[run_evaluation] Wrote {len(outputs)} outputs to {output_file}")

    # 4) Evaluation
    run_evaluation(dataset_name=dataset_name, output_file=output_file)

    return EvaluationRunResult(
        work_dir=work_dir,
        source_file=source_path,
        target_file=target_path,
        output_file=output_file,
        use_memory=use_memory,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MyMem evaluation pipeline (inference with optional memory + evaluation)."
    )
    p.add_argument(
        "--dataset_name",
        required=True,
        choices=[
            "alfworld",
            "fever",
            "pddl",
            "babyai",
            "scienceworld",
            "longmemeval",
            "membench",
            "membench-high",
            "membench-low",
            "GameOf24",
            "AIME_2024",
            "AIME_2025",
            "AIME_2020_2024",
            "GPQA_Diamond",
            "MMLU_Pro_Physics",
            "MMLU_Pro_Engineering",
            "MathEquationBalancer",
            "personamem-v2-val",
            "prefeval-mcq",
            "bigcodebench",
            "toolbench",
        ],
    )
    p.add_argument("--model_name", required=True, help="Model name for src.llm (litellm)")
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory where collection/extraction outputs are stored (same as run_collection/run_extraction).",
    )
    p.add_argument(
        "--not_use_memory",
        action="store_true",
        help="Not use extracted memory for inference.",
    )
    p.add_argument(
        "--source_file",
        default=None,
        help="Path to source.jsonl (default: output_dir/data/dataset_name/model_name/source.jsonl)",
    )
    p.add_argument(
        "--target_file",
        default=None,
        help="Path to target.jsonl (default: output_dir/data/dataset_name/model_name/target.jsonl)",
    )
    # Engine args (same as run_collection)
    p.add_argument("--max_tokens", type=int, default=512, help="LLM max_tokens")
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode if supported (decoding hyperparameters are selected by inference factory).",
    )
    p.add_argument("--debug", type=int, default=-1, help="If > 0, process only first N sources")
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel workers for inference (each worker uses its own engine).",
    )
    p.add_argument(
        "--max_targets_per_source",
        type=int,
        default=None,
        help="If set, limit number of target_ids processed per source example.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    args.use_memory = not args.not_use_memory

    engine_kwargs: Dict[str, Any] = {
        "few_shots_num": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }

    print(args)

    result = run_evaluation_pipeline(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        work_dir=args.output_dir,
        use_memory=args.use_memory,
        max_targets_per_source=args.max_targets_per_source,
        source_file=args.source_file,
        target_file=args.target_file,
        engine_kwargs=engine_kwargs,
        debug=args.debug,
        num_workers=args.num_workers,
    )

    print("[Evaluation] Done.")
    print(f"[Evaluation] work_dir={result.work_dir}")
    print(f"[Evaluation] source_file={result.source_file}")
    print(f"[Evaluation] target_file={result.target_file}")
    print(f"[Evaluation] output_file={result.output_file}")
    print(f"[Evaluation] use_memory={result.use_memory}")


if __name__ == "__main__":
    main()
