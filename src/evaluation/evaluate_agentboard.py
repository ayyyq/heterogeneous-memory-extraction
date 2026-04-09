"""
Evaluation for AgentBoard-style datasets: babyai, scienceworld, jericho.

Metrics:
- Success Rate (SR): fraction of episodes where label == True
- Progress Rate (PR): mean reward (0-1, fraction of subgoals completed)
Both are reported overall and broken down by difficulty (easy / hard).
"""

import json
from collections import defaultdict
from typing import Dict, List

from src.data_processing.schemas import InferenceOutputExample


def _load_outputs(path: str) -> List[InferenceOutputExample]:
    outputs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(InferenceOutputExample.from_dict(json.loads(line)))
    return outputs


def evaluate_agentboard(dataset_name: str, output_file: str) -> None:
    """
    Print SR / PR evaluation summary for an AgentBoard dataset.

    Args:
        dataset_name: one of 'babyai', 'scienceworld', 'jericho'
        output_file:  path to InferenceOutputExample jsonl produced by run_collection
    """
    outputs = _load_outputs(output_file)
    if not outputs:
        print(f"[AgentBoard Eval] No outputs found in {output_file}")
        return

    total = len(outputs)
    sr_list: List[float] = []
    pr_list: List[float] = []
    by_difficulty_sr: Dict[str, List[float]] = defaultdict(list)
    by_difficulty_pr: Dict[str, List[float]] = defaultdict(list)

    for out in outputs:
        success = bool(out.label)
        reward = float(out.metadata.get("reward", 1.0 if success else 0.0))
        difficulty = out.metadata.get("difficulty", "unknown")

        sr_list.append(1.0 if success else 0.0)
        pr_list.append(reward)
        by_difficulty_sr[difficulty].append(1.0 if success else 0.0)
        by_difficulty_pr[difficulty].append(reward)

    overall_sr = sum(sr_list) / total if total else 0.0
    overall_pr = sum(pr_list) / total if total else 0.0

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"AgentBoard Evaluation — {dataset_name}")
    print(sep)
    print(f"Total episodes : {total}")
    print(f"Success Rate   : {overall_sr:.4f}  ({int(sum(sr_list))}/{total})")
    print(f"Progress Rate  : {overall_pr:.4f}")

    for diff in sorted(by_difficulty_sr.keys()):
        d_sr = by_difficulty_sr[diff]
        d_pr = by_difficulty_pr[diff]
        n = len(d_sr)
        print(
            f"  [{diff:>6}]  SR={sum(d_sr)/n:.4f}  PR={sum(d_pr)/n:.4f}  (n={n})"
        )

    print(f"Output file    : {output_file}")
    print(f"{sep}\n")
