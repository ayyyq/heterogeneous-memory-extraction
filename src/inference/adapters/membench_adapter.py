"""
MemBench single-turn multiple-choice adapter.

No GMemory. Prompt from inference_prompts/membench_prompt (INSTRUCTION_FIRST only).
build_output: raw_response in metadata; rule-based extract_answer + label in adapter.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.inference.inference_prompts.membench_prompt import (
    extract_answer,
    get_question_choices_instruction,
)
from .base import BaseInferenceAdapter


class MemBenchAdapter(BaseInferenceAdapter):
    """Single-turn multiple-choice: one LLM call; evaluation in build_output (extract_answer + label)."""

    def __init__(self, dataset_name: str = "membench"):
        self.dataset_name = dataset_name
        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False

    def set_target(self, target: InferenceInputExample, memory: str = "", use_memory: bool = False) -> None:
        self._target = target
        self._memory = memory or ""
        self._use_memory = use_memory

    def get_system_prompt(
        self,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return ""

    def get_first_user_message(self, task_config: Optional[Dict[str, Any]] = None) -> str:
        if self._target is None:
            return ""
        meta = self._target.metadata
        question = (meta.get("question") or "").strip()
        time = (meta.get("time") or "").strip()
        choices = meta.get("choices") or {}
        return get_question_choices_instruction(
            question, time, choices, memory=self._memory.strip(), use_memory=self._use_memory
        )

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        assistant_msg = {"role": "assistant", "content": raw_llm_output or ""}
        return False, [assistant_msg], {"raw_response": (raw_llm_output or "").strip(), "done": True}

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        raw_response = record.get("raw_response")
        if raw_response is None or raw_response == "":
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    raw_response = (m.get("content") or "").strip()
                    break
        raw_response = raw_response or ""

        metadata = dict(target.metadata)
        metadata["raw_response"] = raw_response

        choices = target.metadata.get("choices") or {}
        ground_truth = (target.metadata.get("ground_truth") or "").strip()
        extracted = extract_answer(raw_response, choices)
        label = extracted.upper() == ground_truth.upper()

        return InferenceOutputExample(
            source=target.source,
            id=target.id,
            task_main=target.task_main,
            messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
            label=label,
            metadata=metadata,
        )
