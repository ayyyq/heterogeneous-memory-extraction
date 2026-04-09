"""
Dataset-specific inference adapters.

Each adapter encapsulates: env.set_env, prompt construction, env.step/feedback,
and building the output dict for InferenceOutputExample.
"""

from .base import BaseInferenceAdapter
from .gmemory_adapter import GMemoryAdapter
from .longmemeval_adapter import LongMemEvalAdapter
from .membench_adapter import MemBenchAdapter
from .dynamic_cheatsheet_adapter import DynamicCheatsheetAdapter
from .personamemv2_adapter import PersonaMemV2Adapter
from .prefeval_adapter import PrefEvalAdapter
from .agentboard_adapter import AgentBoardAdapter
from .bigcodebench_adapter import BigCodeBenchAdapter
from .toolbench_adapter import ToolBenchAdapter

__all__ = [
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
