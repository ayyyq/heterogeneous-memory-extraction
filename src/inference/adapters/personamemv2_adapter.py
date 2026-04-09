"""
PersonaMem-v2 single-turn MCQ adapter.

target.metadata fields (set by PersonaMemV2Generator):
  - user_query: question text (already parsed, same as task_main)
  - correct_answer: ground-truth answer text
  - choices: list of answer strings (incorrect + correct)
  - related_conversation_snippet: list of messages (for reference only)

Inference flow:
  1. assign_choice_letters(choices, correct_answer) → letters_dict, correct_letter
  2. build_prompt(question, letters_dict, memory, use_memory) → user message
  3. LLM produces raw_response
  4. extract_answer(raw_response, letters_dict) → predicted_letter
  5. label = predicted_letter == correct_letter
"""

from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.inference.inference_prompts.personamemv2_prompt import (
    assign_choice_letters,
    build_prompt,
    build_system_prompt,
    extract_answer,
)
from .base import BaseInferenceAdapter


class PersonaMemV2Adapter(BaseInferenceAdapter):
    """Single-turn MCQ adapter for PersonaMem-v2."""

    def __init__(self, dataset_name: str = "personamem-v2"):
        self.dataset_name = dataset_name
        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False
        self._letters_dict: Dict[str, str] = {}
        self._correct_letter: str = ""

    def set_target(
        self, target: InferenceInputExample, memory: str = "", use_memory: bool = False
    ) -> None:
        self._target = target
        self._memory = memory or ""
        self._use_memory = use_memory

        choices_list: List[str] = target.metadata.get("choices") or []
        correct_answer: str = (target.metadata.get("correct_answer") or "").strip()
        self._letters_dict, self._correct_letter = assign_choice_letters(
            choices_list, correct_answer
        )

    def get_system_prompt(
        self, few_shots: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        return build_system_prompt(self._memory, self._use_memory)

    def get_first_user_message(
        self, task_config: Optional[Dict[str, Any]] = None
    ) -> str:
        if self._target is None:
            return ""
        question = (self._target.task_main or "").strip()
        if not question and task_config:
            question = (task_config.get("task") or task_config.get("user_query") or "").strip()
        return build_prompt(
            question=question,
            letters_dict=self._letters_dict,
        )

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        assistant_msg = {"role": "assistant", "content": raw_llm_output or ""}
        return False, [assistant_msg], {"raw_response": (raw_llm_output or "").strip()}

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        raw_response = record.get("raw_response", "")
        if not raw_response:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    raw_response = (m.get("content") or "").strip()
                    break
        raw_response = raw_response or ""

        predicted_letter = extract_answer(raw_response, self._letters_dict)
        correct_letter = self._correct_letter
        label = bool(predicted_letter and predicted_letter == correct_letter)

        metadata = dict(target.metadata)
        metadata.update(
            {
                "raw_response": raw_response,
                "predicted_letter": predicted_letter,
                "correct_letter": correct_letter,
                "letters_dict": self._letters_dict,
            }
        )

        return InferenceOutputExample(
            source=target.source,
            id=target.id,
            task_main=target.task_main,
            messages=[
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ],
            label=label,
            metadata=metadata,
        )
