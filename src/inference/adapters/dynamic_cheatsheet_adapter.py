"""
Dynamic cheatsheet adapter (e.g. GameOf24).

Aligned to dynamic_cheatsheet.language_model.advanced_generate(approach_name="default"):
- First user prompt = inference_prompts/dc_prompt.txt with [[QUESTION]] / [[CHEATSHEET]] filled.
- Code execution trigger: "EXECUTE CODE!" with ```python ... ```
- final_answer extracted via extract_answer().
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.inference.dc_utils.extractor import extract_answer
from src.inference.dc_utils.execute_code import extract_and_run_python_code
from .base import BaseInferenceAdapter

# Follow-up user message text (same semantics as DC generate())
_FOLLOW_UP_TEMPLATE = (
    "Proceed with any additional steps required and provide the completed solution. "
    "If everything is already complete, type FINAL ANSWER and submit it in the expected format. "
    "If you are stuck, please try alternative methods to solve the problem and provide the final solution.{warning}"
)


class DynamicCheatsheetAdapter(BaseInferenceAdapter):
    """Adapter for GameOf24-style tasks: optional code execution and multi-round via continue_conversation."""

    def __init__(
        self,
        dataset_name: str = "GameOf24",
        max_depth_num_rounds: int = 3,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
    ):
        self.dataset_name = dataset_name
        self.max_depth_num_rounds = max_depth_num_rounds
        self.allow_code_execution = allow_code_execution
        self.code_execution_flag = code_execution_flag
        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False
        self._depth = 0
        self._final_output_so_far = ""
        self._steps: List[Dict[str, Any]] = []
        self._generator_prompt: Optional[str] = None

    def set_target(self, target: InferenceInputExample, memory: str = "", use_memory: bool = False) -> None:
        self._target = target
        self._memory = memory or ""
        self._use_memory = use_memory
        self._depth = 0
        self._final_output_so_far = ""
        self._steps = []

    def get_system_prompt(
        self,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        DC keeps system prompt minimal; memory is handled in first user message via dc_with_memory_prompt when present.
        """
        return "You are a helpful assistant."

    def get_first_user_message(self, task_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Build first user message:
        - If memory present or use_memory: use dc_with_memory_prompt.txt with [[MEMORY]] and [[QUESTION]] filled.
        - Else: use dc_prompt.txt with [[QUESTION]] and [[CHEATSHEET]] filled.
        """
        formatted_input = ""
        if self._target is not None:
            formatted_input = self._target.metadata.get("input", "") or self._target.task_main
        elif task_config is not None:
            formatted_input = task_config.get("input", "") or task_config.get("task", "")

        prompt_dir = os.path.join(os.path.dirname(__file__), "..", "inference_prompts")
        memory_template = (
            self._target.metadata.get("memory_template")
            if self._target is not None
            else None
        )
        if self._memory or self._use_memory:
            if memory_template == "rb":
                prompt_path = os.path.join(prompt_dir, "dc_rb_memory_prompt.txt")
            else:
                prompt_path = os.path.join(prompt_dir, "dc_with_memory_prompt.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            prompt = (
                template.replace("[[MEMORY]]", self._memory.strip())
                .replace("[[QUESTION]]", formatted_input or "")
            )
        else:
            prompt_path = os.path.join(prompt_dir, "dc_prompt.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            prompt = (
                template.replace("[[QUESTION]]", formatted_input or "")
                .replace("[[CHEATSHEET]]", "(empty)")
            )
        self._generator_prompt = prompt
        self._final_output_so_far = ""
        self._steps = []
        self._depth = 0
        return prompt

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        new_messages: List[Dict[str, str]] = []
        output = raw_llm_output or ""
        pre_flag = output.split(self.code_execution_flag)[0].strip() if self.code_execution_flag in output else ""
        should_execute = (
            self.allow_code_execution
            and self.code_execution_flag in output
            and len(pre_flag) >= 3
            and pre_flag[-3:] == "```"
        )

        if should_execute:
            output_prefix = output.split(self.code_execution_flag)[0].strip()
            executed = extract_and_run_python_code(output_prefix)
            executed = (executed or "").strip()
            current_output = f"{output_prefix}\n{self.code_execution_flag}\n\n{executed}"
            self._final_output_so_far = f"{self._final_output_so_far}\n\n{current_output}".strip()
            self._steps.append(
                {
                    "round": self._depth,
                    "generator_prompt": self._generator_prompt if self._depth == 0 else None,
                    "generator_output": current_output,
                    "generator_answer": extract_answer(current_output),
                }
            )
            # 与原 dynamic-cheatsheet 行为对齐：assistant 历史中看到的是 current_output（含执行结果）
            assistant_msg = {"role": "assistant", "content": current_output}
            new_messages.append(assistant_msg)

            if self._depth < self.max_depth_num_rounds:
                warning = ""
                if self._depth == self.max_depth_num_rounds - 1:
                    warning = " (This is the last round. No more code execution will be allowed. Please present your final solution now.)"
                follow_up = _FOLLOW_UP_TEMPLATE.format(warning=warning)
                self._depth += 1
                user_msg = {"role": "user", "content": follow_up}
                new_messages.append(user_msg)
                return True, new_messages, {}
            else:
                return False, new_messages, {
                    "final_answer": extract_answer(self._final_output_so_far),
                    "final_output": self._final_output_so_far,
                    "steps": self._steps,
                }
        else:
            # 无代码执行：assistant 历史中看到原始输出
            self._final_output_so_far = f"{self._final_output_so_far}\n\n{output}".strip()
            self._steps.append(
                {
                    "round": self._depth,
                    "generator_prompt": self._generator_prompt if self._depth == 0 else None,
                    "generator_output": output,
                    "generator_answer": extract_answer(output),
                }
            )
            assistant_msg = {"role": "assistant", "content": output}
            new_messages.append(assistant_msg)
            return False, new_messages, {
                "final_answer": extract_answer(self._final_output_so_far),
                "final_output": self._final_output_so_far,
                "steps": self._steps,
            }

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        """
        Build InferenceOutputExample for dynamic-cheatsheet tasks.

        与 GMemoryAdapter 一致：不再嵌套 "metadata" 字段，而是平铺所有 metadata。
        InferenceOutputExample.from_dict 会自动将除
        source/id/task_main/messages/label 以外的键收集到 .metadata。
        """
        msg_list = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]
        steps = record.get("steps", [])
        final_answer = record.get("final_answer", "")
        final_output = record.get("final_output", "")

        # 先复制原始 target.metadata（包含 input/target 等），再追加 dc 相关字段
        flat_meta: Dict[str, Any] = dict(target.metadata)
        flat_meta["steps"] = steps
        flat_meta["final_answer"] = final_answer
        flat_meta["final_output"] = final_output

        out: Dict[str, Any] = {
            "source": target.source,
            "id": target.id,
            "task_main": target.task_main,
            "messages": msg_list,
            "label": None,
        }
        out.update(flat_meta)

        return InferenceOutputExample.from_dict(out)
