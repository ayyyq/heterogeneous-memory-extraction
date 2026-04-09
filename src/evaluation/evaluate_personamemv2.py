"""
PersonaMem-v2 evaluation.

Loads source_target_output.jsonl (InferenceOutputExample with label already set by
PersonaMemV2Adapter.build_output) and prints accuracy.

Labels are set during inference (predicted_letter == correct_letter), so this
module only aggregates and reports — no re-evaluation needed.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

from src.data_processing.schemas import InferenceOutputExample


def _load_outputs(path: str) -> List[InferenceOutputExample]:
    with open(path, "r", encoding="utf-8") as f:
        return [InferenceOutputExample.from_dict(json.loads(line)) for line in f if line.strip()]


def evaluate_personamemv2(output_file: str) -> Dict[str, float]:
    """
    Aggregate and print PersonaMem-v2 MCQ accuracy from output_file.

    Returns a dict with at least {"accuracy": float}.
    """
    if not os.path.isfile(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    outputs = _load_outputs(output_file)
    if not outputs:
        print("[PersonaMem-v2] No outputs to evaluate.")
        return {"accuracy": 0.0}

    total = len(outputs)
    correct = sum(1 for o in outputs if o.label is True)
    accuracy = correct / total if total else 0.0

    # Optional per-category breakdown (if question_type stored in metadata)
    category_stats: Dict[str, List[bool]] = defaultdict(list)
    for out in outputs:
        qtype = (
            out.metadata.get("question_type")
            or out.metadata.get("qa_type")
            or "unknown"
        )
        category_stats[qtype].append(bool(out.label))

    sep = "=" * 60
    print(f"\n{sep}")
    print("PersonaMem-v2 Evaluation Results")
    print(sep)
    print(f"Total:    {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    if len(category_stats) > 1:
        print("\nPer question_type:")
        for qtype in sorted(category_stats):
            vals = category_stats[qtype]
            acc = sum(vals) / len(vals) if vals else 0.0
            print(f"  {qtype:40s}: {acc:.4f}  ({len(vals)} samples)")

    print(f"{sep}\n")

    return {"accuracy": accuracy, "total": total, "correct": correct}
