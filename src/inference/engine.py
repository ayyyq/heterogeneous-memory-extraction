"""
Unified inference engine.

Runs inference only: round loop, build InferenceOutputExample, append to output file.
Logs system prompt, first user prompt, and conversation/response. No evaluation or metrics.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

from src.llm import LLMChatCompletion, Message
from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.utils import print_messages

from .adapters.base import BaseInferenceAdapter


class BaseInferenceEngine:
    """
    run_one(target, memory, ...) runs inference and returns one InferenceOutputExample.
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        adapter: BaseInferenceAdapter,
        max_steps: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        stop_strs: List[str] = None,
        print_adapter_messages: bool = True,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.adapter = adapter
        self._max_steps = max_steps
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.stop_strs = stop_strs
        self.print_adapter_messages = print_adapter_messages
        self.logger = logger
        self.llm = LLMChatCompletion(model_name=model_name)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _merge_record(self, accumulated: Dict[str, Any], new: Dict[str, Any]) -> None:
        for k, v in new.items():
            if k == "trajectory" and isinstance(v, list):
                accumulated.setdefault("trajectory", []).extend(v)
            elif k == "trajectory_item" and v is not None:
                accumulated.setdefault("trajectory", []).append(v)
            else:
                accumulated[k] = v

    def run_one(
        self,
        target: InferenceInputExample,
        memory: str = "",
        few_shots: Optional[List[Dict[str, Any]]] = None,
        use_memory: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputExample:
        """
        Run inference for one target. Build InferenceOutputExample,
        and return the output example.

        use_memory: If True, use the with_memory template even when memory="".
        """
        task_config = target.to_task_config()
        if hasattr(self.adapter, "set_target"):
            self.adapter.set_target(target, memory=memory, use_memory=use_memory)
        if hasattr(self.adapter, "set_env"):
            self.adapter.set_env(task_config)

        system_prompt = self.adapter.get_system_prompt(few_shots=few_shots)
        # Adapters should read first user message from target (via set_target) when available.
        # task_config is kept for backward compatibility but adapters prioritize target.
        first_user = self.adapter.get_first_user_message(task_config=task_config)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]
        if self.print_adapter_messages:
            print("=" * 60)
            print_messages(messages)
        accumulated_record: Dict[str, Any] = {}

        # 由 adapter 完全控制何时结束对话（包括 max_steps / code execution 逻辑）。
        try:
            while True:
                llm_messages = [Message(m["role"], m["content"]) for m in messages]
                result = self.llm(
                    messages=llm_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    enable_thinking=self.enable_thinking,
                    thinking_budget=self.thinking_budget,
                    stop_strs=self.stop_strs,
                )
                raw_response = result.raw_response

                # 将当前 LLM 输出交给 adapter；adapter 负责把需要的新消息（assistant/user）追加到 messages_so_far，
                # 并返回本轮新增的消息列表，便于统一打印。
                has_next, new_messages, record = self.adapter.continue_conversation(messages, raw_response)
                if new_messages:
                    if self.print_adapter_messages:
                        print_messages(new_messages)
                    messages.extend(new_messages)

                self._merge_record(accumulated_record, record)

                if not has_next:
                    break
        except (RuntimeError, ValueError) as e:
            err_lower = str(e).lower()
            if "context" in err_lower and ("length" in err_lower or "limit" in err_lower):
                print(f"[engine] Context length overflow, marking task as failed: {e}")
                accumulated_record["reward"] = 0.0
                accumulated_record["success"] = False
                accumulated_record["feedback"] = f"Context length overflow: {e}"
                return self.adapter.build_output(target, messages, **accumulated_record)
            raise

        output_example = self.adapter.build_output(target, messages, **accumulated_record)

        return output_example

    async def run_one_async(
        self,
        target: InferenceInputExample,
        memory: str = "",
        few_shots: Optional[List[Dict[str, Any]]] = None,
        use_memory: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputExample:
        """
        Async variant of `run_one`.
        """
        task_config = target.to_task_config()
        if hasattr(self.adapter, "set_target"):
            self.adapter.set_target(target, memory=memory, use_memory=use_memory)
        if hasattr(self.adapter, "set_env"):
            self.adapter.set_env(task_config)

        system_prompt = self.adapter.get_system_prompt(few_shots=few_shots)
        first_user = self.adapter.get_first_user_message(task_config=task_config)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]
        if self.print_adapter_messages:
            print("=" * 60)
            print_messages(messages)
        accumulated_record: Dict[str, Any] = {}

        try:
            while True:
                llm_messages = [Message(m["role"], m["content"]) for m in messages]
                result = await self.llm.acall(
                    messages=llm_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    enable_thinking=self.enable_thinking,
                    thinking_budget=self.thinking_budget,
                    stop_strs=self.stop_strs,
                )
                raw_response = result.raw_response

                has_next, new_messages, record = self.adapter.continue_conversation(messages, raw_response)
                if new_messages:
                    if self.print_adapter_messages:
                        print_messages(new_messages)
                    messages.extend(new_messages)

                self._merge_record(accumulated_record, record)

                if not has_next:
                    break
        except (RuntimeError, ValueError) as e:
            err_lower = str(e).lower()
            if "context" in err_lower and ("length" in err_lower or "limit" in err_lower):
                print(f"[engine] Context length overflow, marking task as failed: {e}")
                accumulated_record["reward"] = 0.0
                accumulated_record["success"] = False
                accumulated_record["feedback"] = f"Context length overflow: {e}"
                return self.adapter.build_output(target, messages, **accumulated_record)
            raise

        output_example = self.adapter.build_output(target, messages, **accumulated_record)
        return output_example
