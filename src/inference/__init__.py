"""
Unified inference layer.

- BaseInferenceEngine: unified engine that runs multi-step conversations via adapters.
- get_inference_engine(...): factory by dataset.
  Decoding hyperparameters are selected internally from (model_name, enable_thinking).
  Callers should not pass temperature/top_p/max_steps.
"""
from typing import Optional, List

from src.inference.engine import BaseInferenceEngine

from .adapters import (
    BaseInferenceAdapter,
    GMemoryAdapter,
    LongMemEvalAdapter,
    MemBenchAdapter,
    DynamicCheatsheetAdapter,
    PersonaMemV2Adapter,
    PrefEvalAdapter,
    AgentBoardAdapter,
    BigCodeBenchAdapter,
    ToolBenchAdapter,
)


def get_inference_engine(
    dataset_name: str,
    model_name: str,
    few_shots_num: Optional[int] = None,
    max_tokens: int = 512,
    enable_thinking: bool = False,
    stop_strs: List[str] = None,
    logger=None,
    **kwargs,
) -> BaseInferenceEngine:
    """
    Return a BaseInferenceEngine with the adapter for the given dataset.

    Notes:
    - temperature/top_p are selected internally from model_name + enable_thinking.
    - max_steps is determined by dataset defaults in this factory.
    - callers should pass task-level controls only (e.g., few_shots_num/max_tokens/enable_thinking).
    """
    thinking_budget = None  # NOTE: vLLM does not yet support thinking_budget; kept for future compatibility
    
    if 'qwen3' in model_name.lower():
        if enable_thinking:
            temperature = 0.6
            top_p = 0.95
        else:
            temperature = 0.7
            top_p = 0.8
    else:
        raise NotImplementedError(f"Inference engine for model {model_name} is not implemented")

    default_max_steps = {
        "alfworld": 20,
        "fever": 7,
        "pddl": 30,
        "gameof24": 1,
        "dynamic_cheatsheet": 1,
        "babyai": 20,
        "scienceworld": 30,
        "jericho": 20,
        "bigcodebench": 1,
        "toolbench": 12,
    }
    steps = default_max_steps.get(dataset_name, 1)

    default_few_shots_num = {
        "alfworld": 1,
        "fever": 1,
        "pddl": 1,
        "babyai": 1,
        "scienceworld": 1,
        "jericho": 1,
    }
    few_shots_num = few_shots_num if few_shots_num is not None else default_few_shots_num.get(dataset_name, 0)

    if dataset_name in ("alfworld", "fever", "pddl"):
        adapter = GMemoryAdapter(
            dataset_name=dataset_name,
            max_steps=steps,
            few_shots_num=few_shots_num,
        )
        max_tokens = 512
        stop_strs = ["\n"]
    elif dataset_name == "longmemeval":
        adapter = LongMemEvalAdapter(dataset_name=dataset_name)
    elif dataset_name == "membench":
        adapter = MemBenchAdapter(dataset_name=dataset_name)
    elif dataset_name in ("gameof24", "dynamic_cheatsheet"):
        adapter = DynamicCheatsheetAdapter(dataset_name=dataset_name)
        max_tokens = 2048
    elif dataset_name == "personamem-v2":
        adapter = PersonaMemV2Adapter(dataset_name=dataset_name)
        max_tokens = 512
    elif dataset_name == "prefeval-mcq":
        adapter = PrefEvalAdapter(dataset_name=dataset_name)
        max_tokens = 128
    elif dataset_name in ("babyai", "scienceworld", "jericho"):
        adapter = AgentBoardAdapter(
            dataset_name=dataset_name,
            max_steps=steps,
            few_shots_num=few_shots_num,
        )
        max_tokens = 512
        stop_strs = ["\n"]
    elif dataset_name == "bigcodebench":
        adapter = BigCodeBenchAdapter(dataset_name=dataset_name)
        max_tokens = 1280
    elif dataset_name == "toolbench":
        adapter = ToolBenchAdapter(dataset_name=dataset_name, max_steps=steps)
        max_tokens = 2048
    else:
        raise ValueError(f"Unknown dataset_name for inference engine: {dataset_name}")

    return BaseInferenceEngine(
        dataset_name=dataset_name,
        model_name=model_name,
        adapter=adapter,
        max_steps=steps,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        stop_strs=stop_strs,
        logger=logger,
        **kwargs,
    )


__all__ = [
    "BaseInferenceEngine",
    "get_inference_engine",
    "BaseInferenceAdapter",
    "GMemoryAdapter",
    "LongMemEvalAdapter",
    "MemBenchAdapter",
    "DynamicCheatsheetAdapter",
    "PersonaMemV2Adapter",
    "PrefEvalAdapter",
    "AgentBoardAdapter",
    "BigCodeBenchAdapter",
    "ToolBenchAdapter",
]
