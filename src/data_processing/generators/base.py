"""
基础Generator接口，同时生成sources和targets。

重要原则:
    生成 source.jsonl 和 target.jsonl 后，后续步骤不再使用原始数据文件。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from src.data_processing.schemas import SourceExample, InferenceInputExample
from src.utils import format_example_value, print_example


class BaseGenerator(ABC):
    """
    基础Generator，同时生成sources和targets。
    
    每个Generator处理特定的数据源格式：
    - GMemoryGenerator: CollectorMemory格式（alfworld/pddl/fever）
    - LongMemEvalGenerator: LongMemEval格式
    - MemBenchGenerator: MemBench格式
    """
    
    def __init__(self, dataset_name: str = "alfworld"):
        """
        初始化Generator。
        
        Args:
            dataset_name: 数据集名称（如'alfworld', 'fever', 'pddl', 'longmemeval', 'membench'）
        """
        self.dataset_name = dataset_name
    
    @abstractmethod
    def generate_sources(
        self,
        source_file: str,
        target_file: Optional[str] = None,
        output_data_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        生成 sources（从 InferenceOutputExample 文件或原始轨迹文件）。
        
        Args:
            source_file: 原始数据文件路径（如trajectories.json, InferenceOutputExample的jsonl等）
            target_file: 可选的 target 数据文件（InferenceInputExample 的 jsonl）。
                若提供，则用相似度（similarity）做 source->target 映射并写回 target_ids。
            output_data_dir: 输出目录（保存 source_input.jsonl）
            **kwargs: 其他数据集特定参数（如 filter_success_only, max_targets_per_source, threshold）
        
        Returns:
            None（结果保存到文件）
        """
        pass

    def generate_targets(
        self,
        output_data_dir: Optional[str] = None,
        max_targets: int = -1,
        **kwargs
    ) -> None:
        """
        从原始 raw 文件生成 InferenceInputExample（用于 collection 阶段的 inference）。
        
        Args:
            output_data_dir: 输出目录（保存target.jsonl，可选）
            max_targets: 最多生成多少个 target（-1 表示全部）
            **kwargs: 数据集特定参数（如 input_file、data_dir 等，由具体 Generator 决定）
        
        Returns:
            None（结果保存到文件）
        """
        pass
    
    def save_outputs(
        self,
        sources: List[SourceExample],
        targets: List[InferenceInputExample],
        output_data_dir: str
    ):
        """统一保存sources和targets"""
        self.save_sources(sources, output_data_dir)
        self.save_targets(targets, output_data_dir)
    
    def save_sources(
        self,
        sources: List[SourceExample],
        output_data_dir: str
    ):
        """
        保存sources到source_input.jsonl
        
        Args:
            sources: List of SourceExample objects (with target_ids set)
            output_data_dir: Directory to save source_input.jsonl
        """
        import os
        import json
        
        os.makedirs(output_data_dir, exist_ok=True)
        
        source_file = os.path.join(output_data_dir, 'source_input.jsonl')
        
        # Save sources (target_ids are included in each source)
        with open(source_file, 'w', encoding='utf-8') as f:
            for source in sources:
                f.write(json.dumps(source.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"[{self.__class__.__name__}] Saved {len(sources)} sources to {source_file}")
    
    def save_targets(
        self,
        targets: List[InferenceInputExample],
        output_data_dir: str
    ):
        """
        保存targets到target.jsonl
        
        Args:
            targets: List of InferenceInputExample objects
            output_data_dir: Directory to save target.jsonl
        """
        import os
        import json
        
        os.makedirs(output_data_dir, exist_ok=True)
        
        target_file = os.path.join(output_data_dir, 'target.jsonl')
        
        # Save targets
        with open(target_file, 'w', encoding='utf-8') as f:
            for target in targets:
                f.write(json.dumps(target.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"[{self.__class__.__name__}] Saved {len(targets)} targets to {target_file}")
