"""
AgentBoard multi-turn adapter for babyai, scienceworld, jericho.

Pattern mirrors GMemoryAdapter but uses AgentBoard environments instead of GMemory.
Key differences:
- set_env() initializes the AgentBoard env and captures initial obs/goal at runtime.
- get_first_user_message() returns the live initial observation (not stored in target metadata).
- continue_conversation() calls env.step(action) per LLM turn.

Prompt structure (aligned with AgentBoard VanillaAgent):
  system:    system_msg only (1-line role description)
  user (1):  instruction + examples + [memory] + goal + check hints + initial observation
  user (N):  observation from env step   ← subsequent turns stay multi-turn
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.data_processing.schemas import InferenceInputExample, InferenceOutputExample
from .base import BaseInferenceAdapter

_MME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Prompt files (relative to MME root)
_PROMPT_PATHS = {
    "babyai": "AgentBoard/agentboard/prompts/VanillaAgent/babyai_vanilla_prompt.json",
    "scienceworld": "AgentBoard/agentboard/prompts/VanillaAgent/scienceworld_base.json",
    "jericho": "AgentBoard/agentboard/prompts/VanillaAgent/jericho_vanilla_prompt.json",
}

# ScienceWorld simplification string (matching AgentBoard defaults)
_SW_SIMPLIFICATION = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"


def _load_prompt(dataset_name: str) -> dict:
    rel = _PROMPT_PATHS.get(dataset_name, "")
    full = os.path.join(_MME_ROOT, rel)
    if os.path.exists(full):
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Example resolution helper
# ---------------------------------------------------------------------------

def _resolve_examples(few_shots, prompt_data: dict, few_shots_num: int = 1) -> str:
    """
    Return examples as a single string, capped to few_shots_num items.
    few_shots_num=0 always returns "" (no examples).
    Priority: few_shots argument (if not None) > prompt_data["examples"].
    Handles list-of-str, list-of-dict (converted via str()), and plain str.
    """
    if few_shots_num == 0:
        return ""

    if few_shots is not None:
        items = few_shots if isinstance(few_shots, list) else [few_shots]
        return "\n".join(str(e) for e in items[:few_shots_num]).strip()

    raw = prompt_data.get("examples", "")
    if isinstance(raw, list):
        return "\n".join(str(e) for e in raw[:few_shots_num]).strip()
    if isinstance(raw, str):
        # A single string counts as 1 example; return it only if few_shots_num >= 1.
        return raw.strip()
    return ""


# ---------------------------------------------------------------------------
# Action parsing helpers
# ---------------------------------------------------------------------------

def _extract_action_line(raw: str) -> str:
    """
    Return the last line that starts with 'Action:' (case-insensitive).
    Falls back to raw if no such line is found.
    Handles multi-line LLM outputs like "Think: ...\nAction: turn right".
    """
    for line in reversed(raw.strip().splitlines()):
        if line.strip().lower().startswith("action:"):
            return line.strip()
    return raw.strip()


def _strip_action_prefix(s: str) -> str:
    """Strip leading 'Action:' prefix (case-insensitive) if present."""
    if s.lower().startswith("action:"):
        s = s[len("action:"):].strip()
    return s


def _parse_action_babyai(raw: str) -> str:
    s = _strip_action_prefix(_extract_action_line(raw))
    return s.rstrip(".")


def _parse_action_jericho(raw: str) -> str:
    s = _strip_action_prefix(_extract_action_line(raw))
    action = s.split(",")[0].split(".")[0]
    return " ".join(action.split()[:20]).strip()


def _parse_action_scienceworld(raw: str) -> str:
    return _strip_action_prefix(_extract_action_line(raw))


_ACTION_PARSERS = {
    "babyai": _parse_action_babyai,
    "scienceworld": _parse_action_scienceworld,
    "jericho": _parse_action_jericho,
}

_MEMORY_TEMPLATE = """\

# Retrieved Memory
The following are retrieved memories from past interactions.
<memory>
{memory}
</memory>
Instruction: Treat the above memories as potentially useful context. Consider them when responding to the current request, and use them when relevant; ignore them if they are unrelated.
"""


class AgentBoardAdapter(BaseInferenceAdapter):
    """
    Multi-turn adapter for AgentBoard interactive environments: babyai, scienceworld, jericho.

    Usage (per episode):
        adapter.set_target(target, memory="", use_memory=False)
        adapter.set_env(target.to_task_config())   # inits env, captures init obs/goal
        sys_prompt = adapter.get_system_prompt()
        first_msg  = adapter.get_first_user_message()
        # engine loop: LLM -> continue_conversation -> ...
        output = adapter.build_output(target, messages, **record)
    """

    def __init__(self, dataset_name: str, max_steps: int = 20, few_shots_num: int = 0):
        if dataset_name not in ("babyai", "scienceworld", "jericho"):
            raise ValueError(f"AgentBoardAdapter: unsupported dataset '{dataset_name}'")
        self.dataset_name = dataset_name
        self.max_steps = max_steps
        self.few_shots_num = few_shots_num  # 0 = no examples; >0 = use up to N examples from JSON

        self._prompt_data: dict = _load_prompt(dataset_name)
        self._target: Optional[InferenceInputExample] = None
        self._memory: str = ""
        self._use_memory: bool = False
        self._env = None
        self._initial_obs: str = ""
        self._goal: str = ""
        self._step: int = 0
        self._few_shots = None  # stashed by get_system_prompt for use in get_first_user_message

    # ------------------------------------------------------------------
    # set_target / set_env
    # ------------------------------------------------------------------

    def set_target(
        self,
        target: InferenceInputExample,
        memory: str = "",
        use_memory: bool = False,
    ) -> None:
        self._target = target
        self._memory = (memory or "").strip()
        self._use_memory = use_memory

    def set_env(self, task_config: dict) -> Tuple[str, str]:
        """
        Initialize the AgentBoard environment for the given task config.

        Returns:
            (task_main, initial_obs) — same signature as GMemory env.set_env for
            compatibility with BaseInferenceEngine.
        """
        self._step = 0
        if self.dataset_name == "babyai":
            self._env, self._goal, self._initial_obs = self._init_babyai(task_config)
        elif self.dataset_name == "scienceworld":
            self._env, self._goal, self._initial_obs = self._init_scienceworld(task_config)
        elif self.dataset_name == "jericho":
            self._env, self._goal, self._initial_obs = self._init_jericho(task_config)

        task_main = self._target.task_main if self._target else self.dataset_name
        return task_main, self._initial_obs

    def _init_babyai(self, cfg: dict):
        import sys
        agentboard_path = os.path.join(_MME_ROOT, "AgentBoard", "agentboard")
        if agentboard_path not in sys.path:
            sys.path.insert(0, agentboard_path)
        from environment.babyai_env import BabyAI
        env = BabyAI(
            game_name=cfg["game_name"],
            seed=cfg.get("seed", 1234),
            obs_to_reward=cfg.get("obs_to_reward"),
            difficulty=cfg.get("difficulty"),
        )
        goal = env._get_goal()
        init_obs = env._get_obs()
        return env, goal, init_obs

    def _init_scienceworld(self, cfg: dict):
        import sys
        agentboard_path = os.path.join(_MME_ROOT, "AgentBoard", "agentboard")
        if agentboard_path not in sys.path:
            sys.path.insert(0, agentboard_path)
        from scienceworld import ScienceWorldEnv
        server_path = cfg.get("serverPath", None)
        env_step_limit = cfg.get("envStepLimit", 50)
        env = ScienceWorldEnv("", server_path, envStepLimit=env_step_limit)
        task_name = cfg["task_name"]
        var = cfg["var"]
        env.load(task_name, var, simplificationStr=_SW_SIMPLIFICATION)
        obs, _ = env.reset()
        inventory = env.inventory()
        init_obs = obs + f"\n{inventory}"
        goal = cfg.get("modified_goal", task_name)
        # wrap env so we can track subgoal reward state
        wrapped = _ScienceWorldWrapper(env, cfg.get("subgoals", []))
        return wrapped, goal, init_obs

    def _init_jericho(self, cfg: dict):
        import sys
        agentboard_path = os.path.join(_MME_ROOT, "AgentBoard", "agentboard")
        if agentboard_path not in sys.path:
            sys.path.insert(0, agentboard_path)
        from environment.jericho_env import Jericho
        env = Jericho(
            game_file=cfg["game_file"],
            game_name=cfg["game_name"],
            game_config={"goal": cfg["goal"], "init_obs": cfg["init_obs"]},
            obs_to_reward=cfg.get("obs_to_reward"),
            difficulty=cfg.get("difficulty"),
        )
        goal = env._get_goal()
        init_obs = env._get_obs()
        return env, goal, init_obs

    # ------------------------------------------------------------------
    # Prompt methods
    # ------------------------------------------------------------------

    def get_system_prompt(self, few_shots=None) -> str:
        # Stash few_shots so get_first_user_message can use them.
        self._few_shots = few_shots
        # System prompt is only the 1-line role description, matching AgentBoard VanillaAgent.
        return self._prompt_data.get("system_msg", "").strip()

    def get_first_user_message(self, task_config=None) -> str:
        goal = self._goal or (self._target.metadata.get("goal", "") if self._target else "")
        obs = self._initial_obs or (self._target.metadata.get("init_obs", "") if self._target else "")
        if not goal and not obs:
            raise ValueError(
                "AgentBoardAdapter: set_env() must be called before get_first_user_message()."
            )

        parts: List[str] = []

        # 1. Instruction (action rules)
        instruction = self._prompt_data.get("instruction", "").strip()
        if instruction:
            parts.append(instruction)

        # 2. Few-shot examples — aligned with AgentBoard "Here are examples:\n"
        examples = _resolve_examples(self._few_shots, self._prompt_data, self.few_shots_num)
        if examples:
            parts.append("Here are examples:\n" + examples)

        # 3. Memory block (positioned between examples and current task, per Evo-Memory layout)
        if self._memory or self._use_memory:
            parts.append(_MEMORY_TEMPLATE.format(memory=self._memory).strip())

        # 4. Goal — aligned with AgentBoard need_goal=True
        parts.append(f"You should perform actions to accomplish the goal: {goal}")

        # 5. check_actions / check_inventory hints — all three datasets use both
        parts.append(
            "You should use the following commands for help when your action cannot be understood: "
            "check valid actions\n"
            "You should use the following commands for help when your action cannot be understood: "
            "inventory"
        )

        # 6. Initial observation
        parts.append(f"Observation: {obs}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------

    def continue_conversation(
        self,
        messages_so_far: List[Dict[str, str]],
        raw_llm_output: str,
    ) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        new_messages: List[Dict[str, str]] = []

        assistant_msg = {"role": "assistant", "content": raw_llm_output}
        new_messages.append(assistant_msg)

        self._step += 1

        # Parse action: first extract the relevant Action line, then apply dataset parser.
        parser = _ACTION_PARSERS.get(self.dataset_name, lambda x: _strip_action_prefix(_extract_action_line(x)))
        action = parser(raw_llm_output or "")

        # Step environment
        observation, reward, done, infos = self._env.step(action)
        action_is_valid = (infos or {}).get("action_is_valid", True) if isinstance(infos, dict) else True

        reached_max = self._step >= self.max_steps

        trajectory_item = {
            "action": action,
            "reward": reward,
            "done": done,
        }
        record: Dict[str, Any] = {"trajectory_item": trajectory_item}

        if done or reached_max:
            success = self._compute_success(reward, done)
            record["reward"] = float(reward)
            record["success"] = success
            record["feedback"] = "Task completed." if success else "Task not completed."
            obs_str = str(observation)
            # Mirror babyai_env.py: append a terminal status line so the agent sees outcome.
            if self.dataset_name == "babyai" and not success:
                obs_str += "\n The task is not completed."
            new_messages.append({"role": "user", "content": obs_str})
            # Mirror GMemoryAdapter: append an explicit feedback message for datasets that
            # don't embed terminal status in the observation (scienceworld, jericho).
            if self.dataset_name in ("scienceworld", "jericho"):
                feedback_msg = "You successfully finished this task!" if success else "You failed the task."
                new_messages.append({"role": "user", "content": feedback_msg})
            return False, new_messages, record

        new_messages.append({"role": "user", "content": str(observation)})
        return True, new_messages, record

    def _compute_success(self, reward: float, done: bool) -> bool:
        if self.dataset_name == "jericho":
            # Jericho: check env.won if available
            won = getattr(self._env, "won", None)
            if won is not None:
                return bool(won)
        if self.dataset_name == "babyai":
            return float(reward) >= 1.0
        # ScienceWorld and general: done or reward == 1
        return bool(done) or float(reward) >= 1.0

    # ------------------------------------------------------------------
    # build_output
    # ------------------------------------------------------------------

    def build_output(
        self,
        target: InferenceInputExample,
        messages: List[Dict[str, str]],
        **record: Any,
    ) -> InferenceOutputExample:
        msg_list = [{"role": m["role"], "content": m["content"]} for m in messages]

        flat_meta: Dict[str, Any] = dict(target.metadata)
        flat_meta["reward"] = record.get("reward", 0.0)
        flat_meta["success"] = record.get("success", False)
        flat_meta["feedback"] = record.get("feedback", "")

        trajectory = record.get("trajectory", [])
        flat_meta["num_steps"] = len(trajectory)
        flat_meta["trajectory"] = trajectory

        out: Dict[str, Any] = {
            "source": target.source,
            "id": target.id,
            "task_main": target.task_main,
            "messages": msg_list,
            "label": record.get("success", False),
        }
        out.update(flat_meta)

        return InferenceOutputExample.from_dict(out)


# ---------------------------------------------------------------------------
# ScienceWorld wrapper to track subgoal-based reward (mirrors scienceworld_env.py)
# ---------------------------------------------------------------------------

class _ScienceWorldWrapper:
    """
    Thin wrapper around ScienceWorldEnv that tracks subgoal completion and
    exposes the same step() interface as BabyAI / Jericho envs used in this adapter.
    """

    def __init__(self, env, subgoals: list):
        import re
        self._env = env
        self._subgoals = subgoals
        self._finished = [0] * len(subgoals)
        self._reward = 0.0
        self._done = False
        self._re = re

    def step(self, action: str):
        action = action.strip()
        if action == "check valid actions":
            valid = ", ".join(self._get_valid_actions())
            obs = f"Choose an action from these valid actions: {valid}"
            return obs, self._reward, self._done, {"action_is_valid": True}

        obs, _, _, info = self._env.step(action)
        self._check_subgoals(obs)
        self._reward = self._get_reward()
        self._done = self._check_done()
        action_is_valid = action in self._get_valid_actions()
        return obs, self._reward, self._done, {"action_is_valid": action_is_valid}

    def _check_subgoals(self, obs: str):
        for i, pattern in enumerate(self._subgoals):
            if self._re.search(pattern, obs):
                self._finished[i] = 1.0

    def _get_reward(self) -> float:
        if not self._subgoals:
            return 0.0
        return sum(self._finished) / len(self._subgoals)

    def _check_done(self) -> bool:
        return sum(self._finished) >= len(self._subgoals)

    def _get_valid_actions(self):
        return [a for a in self._env.get_possible_actions() if "reset" not in a]

    # Forward attribute access for compatibility
    def __getattr__(self, name):
        return getattr(self._env, name)
