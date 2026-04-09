"""
BigCodeBench single-turn adapter.

Strict interaction:
- system: BCB system prompt (+ optional memory block)
- user: one task prompt
- assistant: one code response
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample

from .base import BaseInferenceAdapter


# DEFAULT_SYSTEM_PROMPT = """You are an expert Python programmer solving BigCodeBench coding tasks.

# You may receive a [Retrieved Memory Context] block with past experiences from similar problems.
# These are references for learning, not guaranteed solutions:
# - [MEMORY TYPE] SUCCESS_PROCEDURE: A successful approach from a similar task, learn the implementation pattern.
# - [MEMORY TYPE] FAILURE_REFLECTION: A failed attempt with lessons, avoid similar mistakes.

# Use the memories as inspiration, but always analyze your current task independently and
# adapt your approach based on its specific requirements. Generate clean, correct Python code.

# Hard constraints for BigCodeBench:
# - Do NOT change the required function signature, return type, or required exception types/messages.
# - Do NOT wrap specific exceptions into generic ones; keep the exact exception class and message if specified.
# - Import every module you use; remove unused imports; do not rely on implicit imports.
# - Avoid broad try/except (e.g., except Exception) unless the task explicitly requires it.
# - Avoid any network calls or extra file I/O beyond what the task specifies.
# - Keep code deterministic: no randomness, time-based logic, or unnecessary logging.
# """


DEFAULT_SYSTEM_PROMPT = """You are an expert Python programmer solving BigCodeBench coding tasks.

Hard constraints for BigCodeBench:
- Do NOT change the required function signature, return type, or required exception types/messages.
- Do NOT wrap specific exceptions into generic ones; keep the exact exception class and message if specified.
- Import every module you use; remove unused imports; do not rely on implicit imports.
- Avoid broad try/except (e.g., except Exception) unless the task explicitly requires it.
- Avoid any network calls or extra file I/O beyond what the task specifies.
- Keep code deterministic: no randomness, time-based logic, or unnecessary logging.
"""

WITH_MEMORY_SYSTEM_PROMPT = """You are an expert Python programmer solving BigCodeBench coding tasks.

You may receive a [Retrieved Memory Context] block with past experiences from similar problems.

Use the memories as inspiration, but always analyze your current task independently and adapt your approach based on its specific requirements. Generate clean, correct Python code.

Hard constraints for BigCodeBench:
- Do NOT change the required function signature, return type, or required exception types/messages.
- Do NOT wrap specific exceptions into generic ones; keep the exact exception class and message if specified.
- Import every module you use; remove unused imports; do not rely on implicit imports.
- Avoid broad try/except (e.g., except Exception) unless the task explicitly requires it.
- Avoid any network calls or extra file I/O beyond what the task specifies.
- Keep code deterministic: no randomness, time-based logic, or unnecessary logging.
"""


def extract_code_from_response(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return text.strip()


class BigCodeBenchAdapter(BaseInferenceAdapter):
    def __init__(
        self,
        dataset_name: str = "bigcodebench",
        memory_budget_tokens: int = 2000,
    ):
        self.dataset_name = dataset_name

        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False

    def set_target(self, target: InferenceInputExample, memory: str = "", use_memory: bool = False) -> None:
        self._target = target
        self._memory = (memory or "").strip()
        self._use_memory = bool(use_memory)

    # @staticmethod
    # def _coerce_bcb_memory_content(*, raw_content: str, outcome: str, task_description: str) -> str:
    #     text = str(raw_content or "").strip()
    #     if not text:
    #         return ""
    #     if "[MEMORY TYPE]" in text.upper():
    #         return text

    #     out = str(outcome or "unknown").strip().lower()
    #     is_failure = out in {"failure", "fail", "failed", "0", "false", "no"}
    #     if is_failure:
    #         m = re.search(r"(?is)(?:^|\n)reflection\s*:\s*(.*)$", text)
    #         reflection = (m.group(1) if m else text).strip()
    #         td = task_description.strip() if task_description else ""
    #         return (
    #             "[MEMORY TYPE] FAILURE_REFLECTION\n"
    #             "[TASK]\n"
    #             f"{td}\n\n"
    #             "[REFLECTION]\n"
    #             f"{reflection}"
    #         ).strip()

    #     body = text
    #     m = re.match(r"(?is)^task\s*:\s*.*?\n\n(.*)$", text)
    #     if m:
    #         body = (m.group(1) or "").strip()
    #     td = task_description.strip() if task_description else ""
    #     return (
    #         "[MEMORY TYPE] SUCCESS_PROCEDURE\n"
    #         "[TASK]\n"
    #         f"{td}\n\n"
    #         "[EXECUTION TRAJECTORY]\n"
    #         f"{body}"
    #     ).strip()

    def _format_memory_context(self, memory_text: str) -> str:
        # text = (memory_text or "").strip()
        # if not text:
        #     return ""
        # # If source memory already concatenates multiple examples, keep it as body.
        # coerced = self._coerce_bcb_memory_content(
        #     raw_content=text,
        #     outcome="unknown",
        #     task_description=(self._target.task_main if self._target else ""),
        # )
        # if len(coerced) > self.memory_budget_tokens:
        #     coerced = coerced[: self.memory_budget_tokens] + "..."
        return "# Retrieved Memory\n" + memory_text

    def get_system_prompt(self, few_shots: Optional[List[Dict[str, Any]]] = None) -> str:
        # Two explicit templates:
        # - use_memory=False: DEFAULT_SYSTEM_PROMPT / self.system_prompt (no memory channel).
        # - use_memory=True:  WITH_MEMORY_SYSTEM_PROMPT plus optional retrieved memory block.
        base = WITH_MEMORY_SYSTEM_PROMPT.strip() if self._use_memory else DEFAULT_SYSTEM_PROMPT.strip()
        if self._use_memory or self._memory:
            mem_context = self._format_memory_context(self._memory)
            if mem_context:
                return f"{base}\n\n{mem_context}"
        return base

    def get_first_user_message(self, task_config: Optional[Dict[str, Any]] = None) -> str:
        if self._target is None:
            return ""
        return self._target.task_main or ""

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
        raw_response = record.get("raw_response")
        if not raw_response:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    raw_response = (m.get("content") or "").strip()
                    break
        raw_response = raw_response or ""
        code = extract_code_from_response(raw_response)

        flat_meta: Dict[str, Any] = dict(target.metadata)
        flat_meta["raw_response"] = raw_response
        flat_meta["code"] = code

        out: Dict[str, Any] = {
            "source": target.source,
            "id": target.id,
            "task_main": target.task_main,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "label": None,
        }
        out.update(flat_meta)
        return InferenceOutputExample.from_dict(out)
