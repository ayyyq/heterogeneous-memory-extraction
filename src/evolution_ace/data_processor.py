"""
Data processor for evolution_ace.

Mirrors ACE DataProcessor interface: process_task_data, answer_is_correct, evaluate_accuracy.
For memory extraction: correctness = target inference success (SUCCESS/FAILED).
"""

import sys
import os
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class MemoryExtractionDataProcessor:
    """
    DataProcessor for ACE-compatible evaluation.
    predicted/target are "SUCCESS" or "FAILED" (target inference outcome).
    """

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert raw MemoryExtractionDataInst list to task_dict format for ACE.
        Each item: {source, target, source_dataset} -> {question, context, target, source, target_example, source_dataset}
        """
        task_dicts = []
        for item in raw_data:
            source = item["source"]
            target_ex = item["target"]
            dataset = item["source_dataset"]
            conversation = source.get_conversation_text()
            target_desc = (target_ex.task_main or "")[:500]
            task_dicts.append({
                "question": conversation,
                "context": target_desc,
                "target": "SUCCESS",
                "source": source,
                "target_example": target_ex,
                "source_dataset": dataset,
            })
        return task_dicts

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        For memory extraction: predicted is "SUCCESS" or "FAILED" from target inference.
        ground_truth is always "SUCCESS" (we want success).
        """
        return str(predicted).strip().upper() == "SUCCESS"

    def evaluate_accuracy(
        self, predictions: List[str], ground_truths: List[str]
    ) -> float:
        """Standard accuracy."""
        if not predictions or len(predictions) != len(ground_truths):
            return 0.0
        correct = sum(
            1 for p, g in zip(predictions, ground_truths)
            if self.answer_is_correct(p, g)
        )
        return correct / len(predictions)
