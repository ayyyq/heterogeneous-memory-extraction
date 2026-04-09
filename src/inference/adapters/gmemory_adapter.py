"""
GMemory-style multi-turn adapter (alfworld / fever / pddl).

Only adapter with set_env. continue_conversation internally uses env.process_action, env.step, env.feedback.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from .base import BaseInferenceAdapter

_ENV_CONFIG_PATHS = {
    "alfworld": "GMemory/tasks/env_configs/alfworld_config.yaml",
    "fever": "GMemory/tasks/env_configs/fever_config.yaml",
    "pddl": "GMemory/tasks/env_configs/pddl_config.yaml",
}
_MME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

FEW_SHOT_TEMPLATE = """\

Here are some examples:

{few_shots}
"""


def _load_env_config(dataset_name: str, max_trials: int) -> Tuple[Any, Any]:
    from GMemory.tasks.envs import get_env

    env_config_path = _ENV_CONFIG_PATHS.get(dataset_name)
    env_config = {}
    if env_config_path:
        full_path = os.path.join(_MME_ROOT, env_config_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                env_config = yaml.safe_load(f) or {}
    env = get_env(dataset_name, env_config, max_trials)
    return env, env_config


class GMemoryAdapter(BaseInferenceAdapter):
    """Multi-turn adapter for alfworld, fever, pddl. Only adapter that implements set_env."""

    def __init__(
        self,
        dataset_name: str,
        max_steps: int = 30,
        few_shots_num: int = 0,
    ):
        self.dataset_name = dataset_name
        self.max_steps = max_steps
        self.few_shots_num = few_shots_num
        self.env, _ = _load_env_config(dataset_name, max_steps)
        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False
        self._last_task_config: Optional[Dict[str, Any]] = None
        self._step: int = 0

    def set_target(self, target: InferenceInputExample, memory: str = "", use_memory: bool = False) -> None:
        """Set the current target and memory. Prompt content will be read from target; memory is added to system prompt."""
        self._target = target
        self._memory = memory.strip() or ""
        self._use_memory = use_memory

    def set_env(self, task_config: dict) -> Tuple[str, str]:
        """
        Initialize environment state only (alfworld: load gamefile; fever: store config; pddl: create env).
        Does not return prompt content - use target.task_main / target.metadata['task_description'] for prompts.
        """
        self._last_task_config = task_config
        # 新任务开始时重置 step 计数
        self._step = 0
        task_main, task_description = self.env.set_env(task_config)
        # Note: task_description is stored by generator in target.metadata['task_description']
        # We don't use it here for prompts, only for env initialization
        return task_main, task_description

    def get_system_prompt(
        self,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt for the current target.
        Memory is taken from set_target(target, memory=...); added to system prompt when present.

        Strict mode:
        - Requires `set_target(target, memory=...)` to have been called.
        - Requires `set_env(target.to_task_config())` to have been called (so env is initialized).
        - Does NOT fallback to recomputing or guessing prompt inputs.
        """
        from GMemory.tasks.prompts import get_dataset_system_prompt, get_task_few_shots

        if self._target is None:
            raise ValueError("GMemoryAdapter.get_system_prompt() requires set_target(target) before calling.")
        if not self._last_task_config:
            raise ValueError(
                "GMemoryAdapter.get_system_prompt() requires set_env(target.to_task_config()) before calling."
            )

        # get_dataset_system_prompt expects a dict; use the exact task_config used to initialize env.
        task_config = self._last_task_config
        
        base_prompt = get_dataset_system_prompt(self.dataset_name, task_config=task_config)
        if self.few_shots_num and few_shots is None:
            few_shots = get_task_few_shots(
                self.dataset_name, task_config, self.few_shots_num
            )
        if few_shots:
            shots_text = "\n\n".join(few_shots[: self.few_shots_num]).strip()
            base_prompt = base_prompt + FEW_SHOT_TEMPLATE.format(few_shots=shots_text)
        if self._memory or self._use_memory:
            memory_template = (
                self._target.metadata.get("memory_template")
                if self._target is not None
                else None
            )
            if memory_template == "rb":
                # ReasoningBank style: direct concatenation without extra wrapping.
                # The memory string already contains the RB prefix + deliberation.
                return f"{base_prompt.strip()}\n\n{self._memory}"
            memory_instruction = f"""# Retrieved Memory
The following are retrieved memories from past interactions.
<memory>
{self._memory}
</memory>
Instruction: Treat the above memories as potentially useful context. Consider them when responding to the current request, and use them when relevant; ignore them if they are unrelated to the current request.
"""
            return f"{base_prompt.strip()}\n\n{memory_instruction}"
        return base_prompt

    def get_first_user_message(self, task_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get first user message from target (task_description stored by generator).

        Strict mode:
        - Requires `set_target(target)` to have been called.
        - Requires target.metadata['task_description'] to exist (no fallback).
        """
        if self._target is None:
            raise ValueError(
                "GMemoryAdapter.get_first_user_message() requires set_target(target) before calling."
            )
        task_description = self._target.metadata.get("task_description")
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError(
                "Missing required field target.metadata['task_description'] for GMemoryAdapter. "
                "Ensure gmemory_generator.py populated it."
            )
        return task_description

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        # 先追加本轮 assistant 消息，保持完整对话历史
        new_messages: List[Dict[str, str]] = []
        assistant_msg = {"role": "assistant", "content": raw_llm_output}
        new_messages.append(assistant_msg)

        # 每次 LLM 输出视为一个环境步骤
        self._step += 1

        action = self.env.process_action(raw_llm_output or "")
        observation, reward, done = self.env.step(action)
        reached_max_steps = self._step >= self.max_steps

        trajectory_item = {
            # "raw_action": raw_llm_output or "",
            "action": action,
            # "observation": observation,
            "reward": reward,
            "done": done,
        }
        record: Dict[str, Any] = {"trajectory_item": trajectory_item}
        # 终止条件：环境完成或达到最大步数
        if done or reached_max_steps:
            final_reward, final_success, feedback_msg = self.env.feedback()
            record["reward"] = final_reward
            record["success"] = final_success
            record["feedback"] = feedback_msg
            user_msgs = [
                {"role": "user", "content": observation},
                {"role": "user", "content": feedback_msg}
            ]
            new_messages.extend(user_msgs)
            return False, new_messages, record

        user_msg = {"role": "user", "content": observation}
        new_messages.append(user_msg)
        return True, new_messages, record

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        """
        Build InferenceOutputExample.

        注意：InferenceOutputExample.from_dict 期望所有 metadata 字段是“平铺在顶层”的，
        除了 source/id/task_main/messages/label 以外的键都会被收集进 .metadata。
        因此这里不要再嵌一层 "metadata" 字段，否则 env_name 会变成 metadata['metadata']['env_name']。
        """
        trajectory = record.get("trajectory", [])
        msg_list = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        # 从 target.metadata 拿到原始字段（包含 env_name 等），再追加评估相关字段
        flat_meta: Dict[str, Any] = dict(target.metadata)
        flat_meta["num_steps"] = len(trajectory)
        flat_meta["trajectory"] = trajectory
        flat_meta["reward"] = record["reward"]
        flat_meta["success"] = record["success"]
        flat_meta["feedback"] = record["feedback"]

        out: Dict[str, Any] = {
            "source": target.source,
            "id": target.id,
            "task_main": target.task_main,
            "messages": msg_list,
            "label": record["success"],
        }
        # 平铺 metadata 到顶层，保持与 schemas.InferenceOutputExample 一致
        out.update(flat_meta)

        return InferenceOutputExample.from_dict(out)
