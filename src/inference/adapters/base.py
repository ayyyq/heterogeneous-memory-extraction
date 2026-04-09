"""
Base inference adapter.

Unified interface: get_system_prompt, get_first_user_message, continue_conversation, build_output.
No set_env / process_action / step / feedback in base (only GMemory uses set_env; round logic in continue_conversation).

Adapters that support per-target state may implement set_target(target, memory="", use_memory=False)
to receive target, memory, and use_memory before get_system_prompt / get_first_user_message are called.
use_memory=True: use the with_memory template even when memory="". Memory handling is adapter-specific.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample


class BaseInferenceAdapter(ABC):
    """
    Adapter for dataset-specific inference: prompts, round control, output shape.
    Engine calls continue_conversation each round; adapter returns (has_next, next_messages, record).
    """

    @abstractmethod
    def get_system_prompt(
        self,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt (optionally with few_shots).
        Memory is not passed here; adapters that need memory get it via set_target(target, memory=...).
        """
        pass

    @abstractmethod
    def get_first_user_message(self, task_config: Optional[Dict[str, Any]] = None) -> str:
        """
        First user message to start the conversation.
        
        Adapters should prioritize reading from target (via set_target) when available.
        task_config parameter is kept for backward compatibility but should be treated as fallback.
        For GMemory adapters: use target.metadata.get('task_description') or target.task_main.
        For other adapters: use target.task_main or target.metadata fields (e.g., 'question', 'task').
        """
        pass

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        """
        Process this round's LLM output and decide whether to continue.

        Returns:
            has_next_round: True if another LLM round is needed.
            next_messages: Messages to append for the next round (e.g. [{"role": "user", "content": observation}]).
            record: Dict to merge into accumulated record for build_output (e.g. reward, done, trajectory item).
        """
        # Default: single-turn, no next round
        return False, [], {}

    @abstractmethod
    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        """Build and return InferenceOutputExample from messages and accumulated record."""
        pass
