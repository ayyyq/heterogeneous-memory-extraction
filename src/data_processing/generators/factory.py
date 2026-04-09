"""
Factory for creating generators based on data source type.
"""

from typing import Optional

from src.data_processing.generators.base import BaseGenerator
from src.data_processing.generators.gmemory_generator import GMemoryGenerator, GMM_DATASETS
from src.data_processing.generators.dynamic_cheatsheet_generator import (
    DynamicCheatsheetGenerator,
    DC_TASKS,
)
from src.data_processing.generators.longmemeval_generator import LongMemEvalGenerator
from src.data_processing.generators.membench_generator import MemBenchGenerator, MEMBENCH_DATASETS
from src.data_processing.generators.personamemv2_generator import PersonaMemV2Generator
from src.data_processing.generators.prefeval_generator import PrefEvalGenerator
from src.data_processing.generators.agentboard_generator import AgentBoardGenerator, AGENTBOARD_DATASETS
from src.data_processing.generators.bigcodebench_generator import (
    BigCodeBenchGenerator,
    BIGCODEBENCH_DATASETS,
)
from src.data_processing.generators.toolbench_generator import (
    ToolBenchGenerator,
    TOOLBENCH_DATASETS,
)


def create_generator(
    dataset_name: str,
) -> BaseGenerator:
    """
    Create a generator based on dataset name.

    Mapping logic:
    - GMemory-style datasets: `alfworld`, `fever`, `pddl` -> `GMemoryGenerator`
    - Dynamic-cheatsheet tasks: any name in `DC_TASKS` -> `DynamicCheatsheetGenerator`
    - LongMemEval: `longmemeval` -> `LongMemEvalGenerator`
    - MemBench: `membench` -> `MemBenchGenerator`
    """
    # AgentBoard interactive environments
    if dataset_name in AGENTBOARD_DATASETS:
        return AgentBoardGenerator(dataset_name=dataset_name)

    # BigCodeBench
    if dataset_name in BIGCODEBENCH_DATASETS:
        return BigCodeBenchGenerator(dataset_name=dataset_name)

    # ToolBench
    if dataset_name in TOOLBENCH_DATASETS:
        return ToolBenchGenerator(dataset_name=dataset_name)

    # Dynamic-cheatsheet datasets
    if dataset_name in DC_TASKS:
        return DynamicCheatsheetGenerator(dataset_name=dataset_name)

    # GMemory-style datasets
    if dataset_name in GMM_DATASETS:
        return GMemoryGenerator(dataset_name=dataset_name)

    # LongMemEval
    if dataset_name == "longmemeval":
        return LongMemEvalGenerator(dataset_name=dataset_name)

    # MemBench
    if dataset_name in MEMBENCH_DATASETS:
        return MemBenchGenerator(dataset_name=dataset_name)

    # PersonaMem-v2
    if dataset_name in ["personamem-v2-val"]:
        return PersonaMemV2Generator(dataset_name=dataset_name)

    # PrefEval Implicit Persona + MCQ
    if dataset_name == "prefeval-mcq":
        return PrefEvalGenerator(dataset_name=dataset_name)

    # Unsupported dataset
    raise ValueError(f"Dataset {dataset_name} not supported")


def detect_generator_type(trajectories_file: str, clusters_file: Optional[str] = None) -> str:
    """
    Try to auto-detect the generator type from file structure.
    
    Args:
        trajectories_file: Path to trajectories.json
        clusters_file: Optional path to clusters.json
        
    Returns:
        Detected generator type
    """
    import json
    
    # Try to detect from trajectories.json structure
    try:
        with open(trajectories_file, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)
        
        # Check for CollectorMemory format
        if isinstance(traj_data, dict) and 'trajectories' in traj_data:
            traj = traj_data['trajectories'][0] if traj_data['trajectories'] else {}
            if 'messages' in traj:
                return "collector"
        
        # Check for GMemory format (may differ)
        # TODO: Add more detection logic based on actual GMemory format
        
    except Exception as e:
        print(f"[Warning] Could not auto-detect generator type: {e}")
    
    # Default to collector
    return "collector"
    
