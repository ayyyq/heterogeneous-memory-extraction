"""
PrefEval MCQ evaluation.

Loads source_target_output.jsonl (InferenceOutputExample with label already set by
PrefEvalAdapter.build_output) and prints overall + per-topic accuracy.
No LLM calls required — labels are determined during inference.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

from src.data_processing.schemas import InferenceOutputExample


def _load_outputs(path: str) -> List[InferenceOutputExample]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            InferenceOutputExample.from_dict(json.loads(line))
            for line in f
            if line.strip()
        ]


def evaluate_prefeval_mcq(output_file: str) -> Dict[str, float]:
    """
    Compute PrefEval MCQ accuracy from *output_file*.

    Prints overall accuracy and per-topic breakdown to stdout.
    Returns a result dict with keys: accuracy, total, correct.
    """
    if not os.path.isfile(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    outputs = _load_outputs(output_file)
    if not outputs:
        print("[PrefEval MCQ] No outputs to evaluate.")
        return {"accuracy": 0.0, "total": 0, "correct": 0}

    total = len(outputs)
    correct = sum(1 for o in outputs if o.label is True)
    accuracy = correct / total if total else 0.0

    # Per-topic breakdown
    topic_labels: Dict[str, List[bool]] = defaultdict(list)
    for out in outputs:
        topic = out.metadata.get("topic", "unknown")
        topic_labels[topic].append(bool(out.label))

    sep = "=" * 64
    print(f"\n{sep}")
    print("PrefEval MCQ Evaluation Results")
    print(sep)
    print(f"  Total:    {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    if topic_labels:
        print(f"\n  {'Topic':<46} {'Acc':>6}  {'N':>5}")
        print("  " + "-" * 60)
        for topic in sorted(topic_labels.keys()):
            vals = topic_labels[topic]
            acc = sum(vals) / len(vals) if vals else 0.0
            print(f"  {topic:<46} {acc:>6.4f}  {len(vals):>5}")

    print(f"{sep}\n")

    return {"accuracy": accuracy, "total": total, "correct": correct}
