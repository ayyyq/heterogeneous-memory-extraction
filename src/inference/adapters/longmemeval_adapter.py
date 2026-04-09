"""
LongMemEval single-turn adapter.

No GMemory. Prompt from inference_prompts/longmemeval_prompt (no-retrieval, cot=True).
Output metadata = target.metadata + hypothesis only; label set by evaluate_longmemeval.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.inference.inference_prompts.longmemeval_prompt import get_first_user_message_content
from .base import BaseInferenceAdapter


class LongMemEvalAdapter(BaseInferenceAdapter):
    """Single-turn QA: one LLM call; evaluation via evaluate_longmemeval (LLM-as-judge)."""

    def __init__(self, dataset_name: str = "longmemeval"):
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
        question = ""
        if self._target is not None:
            question = self._target.task_main or ""
        if not question and task_config is not None:
            question = task_config.get("task", "") or (
                task_config.get("question", "") if isinstance(task_config.get("question"), str) else ""
            )
        return get_first_user_message_content(question, memory=self._memory.strip(), use_memory=self._use_memory) if question else ""

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        assistant_msg = {"role": "assistant", "content": raw_llm_output or ""}
        hypothesis = (raw_llm_output or "").strip()
        return False, [assistant_msg], {"hypothesis": hypothesis}

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        hypothesis = record.get("hypothesis", "")
        if not hypothesis and messages:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    hypothesis = (m.get("content") or "").strip()
                    break
        metadata = dict(target.metadata)
        metadata["hypothesis"] = hypothesis
        msg_list = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        return InferenceOutputExample(
            source=target.source,
            id=target.id,
            task_main=target.task_main,
            messages=msg_list,
            label=None,
            metadata=metadata,
        )
