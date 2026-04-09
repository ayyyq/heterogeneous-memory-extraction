"""
Evaluation factory for collection pipeline.

- GMemory datasets (alfworld/fever/pddl): print summary only, no new files.
- Dynamic-cheatsheet tasks: evaluate outputs, write eval_outputs.jsonl, print summary.
- LongMemEval: evaluate via LLM-as-judge, write back same file, print per-task/overall/abstention metrics.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.data_processing.generators.gmemory_generator import GMM_DATASETS
from src.data_processing.generators.dynamic_cheatsheet_generator import DC_TASKS
from src.data_processing.generators.membench_generator import MEMBENCH_DATASETS
from src.data_processing.generators.agentboard_generator import AGENTBOARD_DATASETS
from src.data_processing.generators.bigcodebench_generator import BIGCODEBENCH_DATASETS
from src.data_processing.generators.toolbench_generator import TOOLBENCH_DATASETS
from src.evaluation import evaluate_dynamic_cheatsheet as dc_eval
from src.evaluation.evaluate_longmemeval import (
    evaluate_longmemeval,
    print_longmemeval_metrics,
)
from src.evaluation.evaluate_personamemv2 import evaluate_personamemv2
from src.evaluation.evaluate_prefeval import evaluate_prefeval_mcq
from src.evaluation.evaluate_agentboard import evaluate_agentboard
from src.evaluation.evaluate_bigcodebench import evaluate_bigcodebench
from src.evaluation.evaluate_toolbench import evaluate_toolbench, print_toolbench_metrics
from src.utils import print_example

# Import dataset lists from generators (single source of truth)
# DC_DATASETS is a set version of DC_TASKS for backward compatibility and type consistency
DC_DATASETS = set(DC_TASKS)
LONGMEMEVAL_SOURCE = "longmemeval"


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_outputs(path: str) -> List[InferenceOutputExample]:
    return [InferenceOutputExample.from_dict(obj) for obj in _load_jsonl(path)]


def print_evaluation(
    title: str,
    total: int,
    correct: int,
    *,
    correct_label: str = "Correct",
    by_group: Optional[Dict[str, List[bool]]] = None,
    extra_lines: Optional[List[str]] = None,
) -> None:
    """
    Print evaluation summary: total, correct/success count, accuracy; optional per-group breakdown and extra lines.
    """
    accuracy = (correct / total) if total else 0.0
    sep = "=" * max(20, len(title) + 6)
    print(f"\n=== {title} ===")
    print(f"Total: {total}, {correct_label}: {correct}, Accuracy: {accuracy:.4f}")
    if extra_lines:
        for line in extra_lines:
            print(line)
    if by_group:
        print("Per env/task_type:")
        for env, vals in by_group.items():
            acc = sum(vals) / len(vals) if vals else 0.0
            print(f"  {env}: {acc:.4f} ({len(vals)} examples)")
    print(f"{sep}\n")


def evaluate_gmemory(outputs: Iterable[InferenceOutputExample]) -> None:
    total = 0
    success = 0
    by_env: Dict[str, List[bool]] = {}
    for out in outputs:
        total += 1
        label = out.label
        ok = bool(label)
        success += 1 if ok else 0
        env = out.metadata.get("env_name")
        by_env.setdefault(env, []).append(ok)

    print_evaluation(
        "GMemory Evaluation",
        total,
        success,
        correct_label="Success",
        by_group=by_env,
    )


def evaluate_dynamic_cheatsheet(
    dataset_name: str,
    output_file: str,
) -> None:
    """
    Evaluate Dynamic-Cheatsheet outputs.

    说明：
    - 不再依赖 target_file；所有需要的信息都已经被写入 InferenceOutputExample.metadata 中：
      * "input": 发送给模型的格式化输入
      * "target": 数据集 ground-truth
      * "final_answer": 由 adapter / extractor 提取的最终答案
    - 逻辑与 data/dynamic-cheatsheet/run_benchmark.py 中一致：
      * GameOf24: eval_for_GameOf24(original_input, final_answer)
      * AIME_*:   eval_for_exact_matching_with_no_punctuation(final_answer, target)
      * GPQA/MMLU:eval_for_multiple_choice(formatted_input, final_answer, target)
      * MathEquationBalancer: eval_equation_balancer(None, final_answer, target)
    """
    outputs = _load_outputs(output_file)

    total = 0
    correct = 0

    new_outputs = []
    for out in outputs:
        total += 1
        meta = out.metadata
        final_answer = (meta.get("final_answer") or "").strip()
        formatted_input = meta.get("input", "") or ""
        target_gt = meta.get("target", "") or ""
        raw_input = out.task_main or ""  # DynamicCheatsheetGenerator 将 raw_input 放在 task_main

        if dataset_name == "GameOf24":
            # 与原 run_benchmark.py 一致：用 raw_input（四个数字）和 final_answer 评估
            label = dc_eval.eval_for_GameOf24(raw_input, final_answer)
        elif dataset_name in ("AIME_2024", "AIME_2025", "AIME_2020_2024"):
            label = dc_eval.eval_for_exact_matching_with_no_punctuation(
                final_answer.lower(), target_gt.lower()
            )
        elif dataset_name in ("GPQA_Diamond", "MMLU_Pro_Physics", "MMLU_Pro_Engineering"):
            label = dc_eval.eval_for_multiple_choice(formatted_input, final_answer, target_gt)
        elif dataset_name == "MathEquationBalancer":
            label = dc_eval.eval_equation_balancer(None, final_answer, target_gt)
        else:
            label = False

        correct += 1 if label else 0
        out_dict = out.to_dict()
        out_dict["label"] = bool(label)
        new_outputs.append(out_dict)

    assert len(new_outputs) == len(outputs)
    print_example("First Evaluated Output Example", new_outputs[0])

    with open(output_file, "w", encoding="utf-8") as f:
        for out in new_outputs:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print_evaluation(
        "Dynamic-Cheatsheet Evaluation",
        total,
        correct,
        extra_lines=[
            f"Task: {dataset_name}",
            f"Eval outputs written to: {output_file}",
        ],
    )


def _stats_gmemory(outputs: List[InferenceOutputExample]) -> tuple[int, int, Dict[str, List[bool]]]:
    """Compute total, success, by_env from GMemory-style outputs."""
    total = 0
    success = 0
    by_env: Dict[str, List[bool]] = {}
    for out in outputs:
        total += 1
        ok = bool(out.label)
        success += 1 if ok else 0
        env = out.metadata.get("env_name")
        by_env.setdefault(env, []).append(ok)
    return total, success, by_env


def _stats_from_labels(outputs: List[InferenceOutputExample]) -> tuple[int, int]:
    """Compute total and correct from outputs using .label (for DC or generic)."""
    total = len(outputs)
    correct = sum(1 for out in outputs if out.label is True)
    return total, correct


def print_membench_metrics(output_file: str) -> None:
    """
    Load MemBench InferenceOutputExample jsonl (labels already set by adapter), aggregate and print.
    No file write; print only (like GMemory).
    """
    from collections import defaultdict

    outputs = _load_outputs(output_file)
    if not outputs:
        print("[MemBench] No outputs to aggregate.")
        return

    total = len(outputs)
    correct = sum(1 for out in outputs if out.label is True)
    accuracy = (correct / total) if total else 0.0

    category_stats: Dict[str, List[bool]] = defaultdict(list)
    level_stats: Dict[str, List[bool]] = defaultdict(list)
    for out in outputs:
        ok = bool(out.label)
        cat = out.metadata.get("category", "unknown")
        level = out.metadata.get("data_level", "unknown")
        category_stats[cat].append(ok)
        level_stats[level].append(ok)

    print("\n" + "=" * 60)
    print("MemBench Evaluation Results")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct} / {total}")
    if category_stats:
        print("\n" + "-" * 60)
        print("Accuracy by Category:")
        print("-" * 60)
        for cat in sorted(category_stats.keys()):
            vals = category_stats[cat]
            acc = sum(vals) / len(vals) if vals else 0.0
            print(f"  {cat:30s}: {acc:.4f} ({len(vals)} samples)")
    if level_stats:
        print("\n" + "-" * 60)
        print("Accuracy by Data Level:")
        print("-" * 60)
        for level in sorted(level_stats.keys()):
            vals = level_stats[level]
            acc = sum(vals) / len(vals) if vals else 0.0
            print(f"  {level:30s}: {acc:.4f} ({len(vals)} samples)")
    print("=" * 60 + "\n")


def print_evaluation_from_files(*paths: str, num_workers: int = 1) -> None:
    """
    Read one or more InferenceOutputExample jsonl files and print the same evaluation summary
    as run_collection / run_evaluation, so you can re-print results without re-running.

    Uses InferenceOutputExample.source (from the first example in each file) to decide format:
    - source in GMM_DATASETS (alfworld/fever/pddl): GMemory summary + per env/task_type.
    - source in DC_DATASETS: Dynamic-Cheatsheet summary with Task: source.
    - source == longmemeval: LongMemEval per-task / task-averaged / overall / abstention accuracy.
    - source == membench: MemBench overall + by category / data_level (print only).
    - otherwise: generic Total / Correct / Accuracy.

    Example:
        print_evaluation_from_files(
            "output/data/alfworld/model/zero-shot/target_output.jsonl",
            "output/data/GameOf24/model/zero-shot/target_output.jsonl",
        )
    """
    for path in paths:
        if not os.path.isfile(path):
            print(f"[skip] Not a file: {path}")
            continue
        try:
            outputs = _load_outputs(path)
        except Exception as e:
            print(f"[skip] Failed to load {path}: {e}")
            continue
        if not outputs:
            print(f"[skip] Empty file: {path}")
            continue

        source = outputs[0].source
        extra_lines = [f"File: {path}", f"Source: {source}"]

        if source in AGENTBOARD_DATASETS:
            evaluate_agentboard(source, path)
        elif source in GMM_DATASETS:
            total, success, by_env = _stats_gmemory(outputs)
            print_evaluation(
                f"GMemory Evaluation ({source})",
                total,
                success,
                correct_label="Success",
                by_group=by_env,
                extra_lines=extra_lines,
            )
        elif source in DC_DATASETS:
            total, correct = _stats_from_labels(outputs)
            print_evaluation(
                f"Dynamic-Cheatsheet Evaluation ({source})",
                total,
                correct,
                extra_lines=[f"Task: {source}", *extra_lines],
            )
        elif source == LONGMEMEVAL_SOURCE:
            print(f"\n=== LongMemEval Evaluation ({path}) ===")
            print_longmemeval_metrics(path)
            print("=" * 60 + "\n")
        elif source in MEMBENCH_DATASETS:
            print(f"\n=== MemBench Evaluation ({path}) ===")
            print_membench_metrics(path)
        elif source in BIGCODEBENCH_DATASETS:
            print(f"\n=== BigCodeBench Evaluation ({path}) ===")
            evaluate_bigcodebench(path, num_workers=num_workers)
        elif source in TOOLBENCH_DATASETS:
            print_toolbench_metrics(path)
        elif source == "personamem-v2":
            print(f"\n=== PersonaMem-v2 Evaluation ({path}) ===")
            evaluate_personamemv2(path)
        else:
            total, correct = _stats_from_labels(outputs)
            print_evaluation(
                f"Evaluation ({source})",
                total,
                correct,
                extra_lines=extra_lines,
            )


def print_evaluation_from_files_readonly(*paths: str) -> None:
    """
    Read-only variant of print_evaluation_from_files.
    Same output format, but never re-runs evaluation or writes to files.
    All results are read from existing labels in the jsonl.
    """
    for path in paths:
        if not os.path.isfile(path):
            print(f"[skip] Not a file: {path}")
            continue
        try:
            outputs = _load_outputs(path)
        except Exception as e:
            print(f"[skip] Failed to load {path}: {e}")
            continue
        if not outputs:
            print(f"[skip] Empty file: {path}")
            continue

        source = outputs[0].source
        extra_lines = [f"File: {path}", f"Source: {source}"]

        if source in AGENTBOARD_DATASETS:
            evaluate_agentboard(source, path)
        elif source in GMM_DATASETS:
            total, success, by_env = _stats_gmemory(outputs)
            print_evaluation(
                f"GMemory Evaluation ({source})",
                total,
                success,
                correct_label="Success",
                by_group=by_env,
                extra_lines=extra_lines,
            )
        elif source in DC_DATASETS:
            total, correct = _stats_from_labels(outputs)
            print_evaluation(
                f"Dynamic-Cheatsheet Evaluation ({source})",
                total,
                correct,
                extra_lines=[f"Task: {source}", *extra_lines],
            )
        elif source == LONGMEMEVAL_SOURCE:
            print(f"\n=== LongMemEval Evaluation ({path}) ===")
            print_longmemeval_metrics(path)
            print("=" * 60 + "\n")
        elif source in MEMBENCH_DATASETS:
            print(f"\n=== MemBench Evaluation ({path}) ===")
            print_membench_metrics(path)
        elif source in BIGCODEBENCH_DATASETS:
            # Read-only: read existing labels instead of re-executing code
            subset = outputs[0].metadata.get("subset", "unknown")
            prompt_split = outputs[0].metadata.get("prompt_split", "unknown")
            total = len(outputs)
            pass_count = sum(1 for o in outputs if o.label)
            pass_at_1 = pass_count / total if total else 0.0
            print("\n" + "=" * 60)
            print("BigCodeBench Evaluation")
            print("=" * 60)
            print(f"Subset      : {subset}")
            print(f"Prompt split: {prompt_split}")
            print(f"Total       : {total}")
            print(f"Pass        : {pass_count}")
            print(f"Pass@1      : {pass_at_1:.4f}")
            print(f"Output file : {path}")
            print("=" * 60 + "\n")
        elif source in TOOLBENCH_DATASETS:
            print_toolbench_metrics(path)
        elif source == "personamem-v2":
            print(f"\n=== PersonaMem-v2 Evaluation ({path}) ===")
            evaluate_personamemv2(path)
        else:
            total, correct = _stats_from_labels(outputs)
            print_evaluation(
                f"Evaluation ({source})",
                total,
                correct,
                extra_lines=extra_lines,
            )


def run_evaluation(
    dataset_name: str,
    output_file: str,
) -> None:
    if dataset_name in AGENTBOARD_DATASETS:
        evaluate_agentboard(dataset_name, output_file)
        return

    if dataset_name in GMM_DATASETS:
        evaluate_gmemory(_load_outputs(output_file))
        return

    if dataset_name in DC_DATASETS:
        evaluate_dynamic_cheatsheet(dataset_name, output_file)
        return

    if dataset_name == LONGMEMEVAL_SOURCE:
        print(
            "[longmemeval] Collection-phase outputs saved. "
            "Run `run_llm_judge_evaluation.py` for LLM evaluation."
        )
        # evaluate_longmemeval(output_file)
        return

    if dataset_name in MEMBENCH_DATASETS:
        print_membench_metrics(output_file)
        return

    if dataset_name in BIGCODEBENCH_DATASETS:
        evaluate_bigcodebench(output_file, num_workers=1)
        return

    if dataset_name in TOOLBENCH_DATASETS:
        evaluate_toolbench(dataset_name, output_file)
        print(
            "[toolbench] Heuristic evaluation done. "
            "Run `python -m src.pipelines.run_llm_judge_evaluation --dataset_name toolbench` "
            "for LLM-judge evaluation (SoPR/SoWR)."
        )
        return

    if dataset_name in ("personamem-v2-val", "personamem-v2"):
        evaluate_personamemv2(output_file)
        return

    if dataset_name == "prefeval-mcq":
        evaluate_prefeval_mcq(output_file)
        return

    raise ValueError(f"Unsupported dataset_name for evaluation: {dataset_name}")
