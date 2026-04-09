"""
ToolBench multi-turn ReACT adapter.

Multi-turn interaction:
- system: tool/API documentation + optional memory
- user (1): task instruction
- assistant: Thought/Action/ActionInput
- user (N): Observation (API response)
- ... repeat until "Finish" action or max steps

Uses StableToolBench cached API responses when available,
falls back to a placeholder error response for uncached calls.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from src.inference.inference_prompts.toolbench_prompt import (
    build_system_prompt,
    build_first_user_message,
    FINISH_ACTION,
)
from .base import BaseInferenceAdapter


MAX_OBSERVATION_LENGTH = 1024

_MME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_CACHE_DIR = os.path.join(_MME_ROOT, "data", "toolbench", "api_cache", "tool_response_cache")


# ---------------------------------------------------------------------------
# StableToolBench name normalization (mirrors server/utils.py)
# ---------------------------------------------------------------------------

def _standardize(string: str) -> str:
    """Normalize names to match StableToolBench cache paths."""
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    string = string.strip("_")
    if string and string[0].isdigit():
        string = "get_" + string
    return string


def _standardize_category(category: str) -> str:
    """Normalize category name to match cache directory names."""
    s = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in s or "," in s:
        s = s.replace(" ", "_").replace(",", "_")
    return s.replace("__", "_")


def _change_name(name: str) -> str:
    """Avoid Python keyword collisions."""
    reserved = {"from", "class", "return", "false", "true", "id", "and"}
    return "is_" + name if name in reserved else name


# ---------------------------------------------------------------------------
# ReACT output parsing
# ---------------------------------------------------------------------------

def _parse_react_output(raw: str) -> Tuple[str, str, str]:
    """
    Parse ReACT-format LLM output into (thought, action, action_input).

    Expected format:
        Thought: ...
        Action: api_name
        Action Input: {"param": "value"}
    """
    thought = ""
    action = ""
    action_input = ""

    # Extract Thought
    m = re.search(r"Thought\s*:\s*(.*?)(?=\nAction\s*:|$)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        thought = m.group(1).strip()

    # Extract Action
    m = re.search(r"Action\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if m:
        action = m.group(1).strip()

    # Extract Action Input
    m = re.search(r"Action\s+Input\s*:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)
    if m:
        action_input = m.group(1).strip()

    return thought, action, action_input


def _parse_action_input_json(action_input: str) -> dict:
    """Try to parse action input as JSON."""
    if not action_input:
        return {}
    try:
        return json.loads(action_input)
    except json.JSONDecodeError:
        # Try to extract JSON from the string
        m = re.search(r"\{.*\}", action_input, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw_input": action_input}


def _is_finish_action(action: str) -> bool:
    """Check if the action is a Finish action."""
    return action.lower().strip() in ("finish", "final answer", "give_answer")


def _extract_final_answer(action_input_dict: dict) -> str:
    """Extract final answer from Finish action input."""
    if isinstance(action_input_dict, dict):
        return str(action_input_dict.get("final_answer", ""))
    return str(action_input_dict)


class ToolBenchAPIExecutor:
    """
    Execute ToolBench API calls using StableToolBench's hierarchical cached responses.

    Cache structure:
        {cache_dir}/{category}/{tool_name}_for_{category}/{api_name}.json
    Each JSON file maps str(tool_input) -> {"error": "", "response": "..."}
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._file_cache: Dict[str, dict] = {}

    def _load_cache_file(self, path: str) -> dict:
        """Load and cache a single API cache file."""
        if path in self._file_cache:
            return self._file_cache[path]
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._file_cache[path] = data
            return data
        except Exception:
            return {}

    def _resolve_api_info(self, action: str, api_list: list) -> Optional[Dict[str, str]]:
        """
        Resolve action name to cache path components using target's api_list.

        The LLM may produce action in formats like:
        - "tool_name.api_name" (e.g. "TheClique.songkick_concert")
        - just "api_name" (e.g. "songkick_concert")
        - standardized form (e.g. "songkick_concert_for_theclique")
        """
        action_std = _standardize(action)
        action_lower = action.lower().strip()

        # Handle "ToolName.api_name" format: split and standardize the api part separately
        action_api_only_std = None
        if "." in action:
            parts = action.split(".", 1)
            action_api_only_std = _standardize(parts[1])

        for api in api_list:
            if not isinstance(api, dict):
                continue
            tool_name = api.get("tool_name", "")
            api_name = api.get("api_name", "")
            category = api.get("category_name", "")

            std_api = _change_name(_standardize(api_name))
            std_tool = _standardize(tool_name)
            std_category = _standardize_category(category)
            tool_folder = f"{std_tool}_for_{std_category}"

            # Split off the _for_{tool} suffix that StableToolBench appends
            api_base = std_api.split(f"_for_{std_tool}")[0]

            # Match variants the LLM might produce
            if (action_std in (std_api, api_base, f"{std_tool}_{api_base}",
                               f"{std_tool}.{api_base}", f"{std_tool}.{std_api}")
                or action_lower == api_name.lower()
                or (action_api_only_std and action_api_only_std in (std_api, api_base))):
                return {
                    "category": std_category,
                    "tool_folder": tool_folder,
                    "api_file": api_base,
                }

        return None

    def execute(self, action: str, action_input: dict, api_list: list = None) -> str:
        """
        Execute an API call and return the observation string.

        Looks up the StableToolBench hierarchical cache:
          {cache_dir}/{category}/{tool_folder}/{api_file}.json
        Key: str(action_input)
        """
        api_list = api_list or []
        _FALLBACK = json.dumps({
            "error": "",
            "response": (
                "This API is not available in the cache. "
                "The API call was attempted but no cached response was found. "
                "Please try a different approach or API."
            ),
        })

        if not os.path.isdir(self._cache_dir):
            return _FALLBACK

        # 1. Resolve action to cache path via target's api_list
        info = self._resolve_api_info(action, api_list)
        if info:
            cache_path = os.path.join(
                self._cache_dir,
                info["category"],
                info["tool_folder"],
                info["api_file"] + ".json",
            )
            cache_data = self._load_cache_file(cache_path)
            if cache_data:
                key = str(action_input)
                if key in cache_data:
                    resp = cache_data[key]
                    return json.dumps(resp, ensure_ascii=False) if isinstance(resp, dict) else str(resp)
                # Use first cached response as approximate result for unseen inputs
                first_key = next(iter(cache_data))
                resp = cache_data[first_key]
                return json.dumps(resp, ensure_ascii=False) if isinstance(resp, dict) else str(resp)

        # 2. Try flat cache files at root level (older cache format)
        action_std = _standardize(action)
        flat_path = os.path.join(self._cache_dir, action_std + ".json")
        flat_data = self._load_cache_file(flat_path)
        if flat_data:
            key = str(action_input)
            if key in flat_data:
                resp = flat_data[key]
                return json.dumps(resp, ensure_ascii=False) if isinstance(resp, dict) else str(resp)

        return _FALLBACK


class ToolBenchAdapter(BaseInferenceAdapter):
    """
    Multi-turn ReACT adapter for ToolBench tool-calling tasks.

    Usage (per episode):
        adapter.set_target(target, memory="", use_memory=False)
        sys_prompt = adapter.get_system_prompt()
        first_msg = adapter.get_first_user_message()
        # engine loop: LLM -> continue_conversation -> ...
        output = adapter.build_output(target, messages, **record)
    """

    def __init__(
        self,
        dataset_name: str = "toolbench",
        max_steps: int = 20,
        api_cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.max_steps = max_steps

        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False
        self._step: int = 0
        self._api_executor = ToolBenchAPIExecutor(cache_dir=api_cache_dir)

    def set_target(
        self,
        target: InferenceInputExample,
        memory: str = "",
        use_memory: bool = False,
    ) -> None:
        self._target = target
        self._memory = (memory or "").strip()
        self._use_memory = use_memory
        self._step = 0

    def get_system_prompt(self, few_shots=None) -> str:
        api_list = []
        if self._target:
            api_list = self._target.metadata.get("api_list", [])
        return build_system_prompt(
            api_list=api_list,
            memory=self._memory,
            use_memory=self._use_memory,
        )

    def get_first_user_message(self, task_config=None) -> str:
        if self._target is None:
            return ""
        query = self._target.task_main or ""
        return build_first_user_message(query)

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        new_messages: List[Dict[str, str]] = []
        assistant_msg = {"role": "assistant", "content": raw_llm_output or ""}
        new_messages.append(assistant_msg)

        self._step += 1

        # Parse the ReACT output
        thought, action, action_input_str = _parse_react_output(raw_llm_output or "")
        action_input_dict = _parse_action_input_json(action_input_str)

        trajectory_item = {
            "step": self._step,
            "thought": thought,
            "action": action,
            "action_input": action_input_str,
        }
        record: Dict[str, Any] = {"trajectory_item": trajectory_item}

        # Check if done (Finish action or max steps)
        reached_max = self._step >= self.max_steps

        if _is_finish_action(action):
            final_answer = _extract_final_answer(action_input_dict)
            give_up = action_input_dict.get("return_type", "") == "give_up_and_restart"
            record["final_answer"] = final_answer
            record["give_up"] = give_up
            record["success"] = bool(final_answer and not give_up)
            return False, new_messages, record

        if reached_max:
            record["final_answer"] = ""
            record["give_up"] = False
            record["success"] = False
            observation = "You have exhausted the maximum number of steps. Please provide your final answer."
            new_messages.append({"role": "user", "content": f"Observation: {observation}"})
            return False, new_messages, record

        # Execute API call and get observation
        api_list = self._target.metadata.get("api_list", []) if self._target else []
        if not action:
            observation = "No action was specified. Please provide a valid action."
        else:
            observation = self._api_executor.execute(action, action_input_dict, api_list=api_list)

        if len(observation) > MAX_OBSERVATION_LENGTH:
            observation = observation[:MAX_OBSERVATION_LENGTH] + "...(truncated)"
        new_messages.append({"role": "user", "content": f"Observation: {observation}"})
        return True, new_messages, record

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        msg_list = [{"role": m["role"], "content": m["content"]} for m in messages]

        final_answer = record.get("final_answer", "")
        give_up = record.get("give_up", False)
        success = record.get("success", False)
        trajectory = record.get("trajectory", [])

        flat_meta: Dict[str, Any] = dict(target.metadata)
        flat_meta["final_answer"] = final_answer
        flat_meta["give_up"] = give_up
        flat_meta["num_steps"] = len(trajectory)
        flat_meta["trajectory"] = trajectory

        out: Dict[str, Any] = {
            "source": target.source,
            "id": target.id,
            "task_main": target.task_main,
            "messages": msg_list,
            "label": None,  # Evaluation done separately (pass rate + win rate)
        }
        out.update(flat_meta)

        return InferenceOutputExample.from_dict(out)
