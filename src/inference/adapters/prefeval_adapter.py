"""
PrefEval Implicit Persona MCQ inference adapter.

Single-turn MCQ:
  - System prompt: base instruction + memory block when memory is provided.
  - First user message: question + shuffled ABCD options + format instruction.
  - build_output: extract <choice>X</choice>, compare to correct_idx, set label.

All MCQ helpers (_format_options, _extract_choice) are self-contained;
no imports from the PrefEval/ directory.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from .base import BaseInferenceAdapter


_SYSTEM_PROMPT = "You are a helpful AI assistant."

_MEMORY_BLOCK = """\
# Retrieved Memory
The following are retrieved memories from past interactions.
<memory>
{memory}
</memory>
Instruction: Treat the above memories as potentially useful context. Consider them when \
responding to the current request, and use them when relevant; ignore them if they are \
unrelated to the current request."""

_MCQ_INSTRUCTION = (
    "I'm trying to decide on this and here are 4 options for my query:\n"
    "{options}\n\n"
    "Now, I'd like you to pick one of them as your top recommendation for me.\n"
    "Important instructions for your response:\n"
    "1. Choose only one option (A, B, C, or D) that best matches my preferences.\n"
    "2. Your answer must be one of these options.\n"
    "3. Don't say things like \"I can't choose\" or suggest alternatives not listed.\n"
    "4. Answer example: <choice>B</choice>. Give me your answer in this exact format, "
    "without any additional explanation:\n"
    "   <choice>[A/B/C/D]</choice>"
)


def _format_options(options: List[str]) -> str:
    """Format list of options as 'A. ...\nB. ...\nC. ...\nD. ...'"""
    return "\n".join(f"{letter}. {opt}" for letter, opt in zip("ABCD", options))


def _extract_choice(text: str) -> Optional[str]:
    """
    Extract the letter inside <choice>X</choice>.
    Falls back to the last standalone A/B/C/D if the tag is absent.
    Returns None if nothing found.
    """
    match = re.search(r"<choice>\s*([ABCD])\s*</choice>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    fallback = re.findall(r"\b([ABCD])\b", text)
    return fallback[-1].upper() if fallback else None


class PrefEvalAdapter(BaseInferenceAdapter):
    """
    PrefEval Implicit Persona + MCQ adapter.

    Lifecycle:
        engine calls set_target(target, memory, use_memory)
        then get_system_prompt()  → memory-aware system prompt
        then get_first_user_message()  → question + shuffled ABCD
        then continue_conversation()  → always False (single turn)
        then build_output()  → extract choice, set label
    """

    def __init__(self, dataset_name: str = "prefeval-mcq"):
        self.dataset_name = dataset_name
        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def set_target(
        self,
        target: InferenceInputExample,
        memory: str = "",
        use_memory: bool = False,
    ) -> None:
        self._target = target
        self._memory = memory or ""
        self._use_memory = use_memory

    # ── Prompt construction ──────────────────────────────────────────────────

    def get_system_prompt(self, few_shots=None) -> str:
        if self._memory or self._use_memory:
            block = _MEMORY_BLOCK.format(memory=self._memory.strip())
            return f"{_SYSTEM_PROMPT}\n\n{block}"
        return _SYSTEM_PROMPT

    def get_first_user_message(self, task_config=None) -> str:
        if self._target is None:
            return ""
        question = self._target.task_main or self._target.metadata.get("question", "")
        shuffled_options: List[str] = self._target.metadata.get("shuffled_options", [])
        if not shuffled_options:
            return question
        formatted = _format_options(shuffled_options)
        mcq_block = _MCQ_INSTRUCTION.format(options=formatted)
        return f"{question}\n\n{mcq_block}"

    # ── Conversation control ─────────────────────────────────────────────────

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        new_messages = [{"role": "assistant", "content": raw_llm_output or ""}]
        return False, new_messages, {}

    # ── Output construction ──────────────────────────────────────────────────

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        # Find the last assistant reply to extract the choice
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        choice = _extract_choice(last_assistant)
        correct_idx: Optional[str] = target.metadata.get("correct_idx")

        label: Optional[bool] = None
        if choice is not None and correct_idx is not None:
            label = choice == correct_idx

        msg_list = [{"role": m["role"], "content": m["content"]} for m in messages]
        out_dict: Dict[str, Any] = {
            "source": target.source,
            "id": target.id,
            "task_main": target.task_main,
            "messages": msg_list,
            "label": label,
        }
        # Preserve all target metadata (preference, persona, topic, shuffled_options, etc.)
        out_dict.update(target.metadata)
        out_dict["extracted_choice"] = choice
        return InferenceOutputExample.from_dict(out_dict)
