"""
Generator for CollectorMemory data format.

Handles: trajectories.json → source.jsonl + target.jsonl
支持从target_file加载或从trajectories提取targets。
"""

import os
import json
import random
import re
import yaml
import numpy as np
from typing import Optional, List, Tuple, Dict

random.seed(42)

from src.data_processing.schemas import SourceExample, InferenceInputExample, InferenceOutputExample
from src.data_processing.generators.base import BaseGenerator
from src.data_processing.mapping_strategies import map_by_similarity
from src.utils import print_example

# 各 dataset 的 env 配置文件路径（相对项目 MME 根目录），与 GMemory/tasks/configs.yaml 一致
DATASETS_PATH = {
    'alfworld': 'data/alfworld/alfworld_tasks_suffix.json',
    'fever': 'data/fever/fever_dev.jsonl',
    'pddl': 'data/pddl/test.jsonl'
}
# GMemory-style datasets (exported for use in other modules)
GMM_DATASETS = {"alfworld", "fever", "pddl"}
ENV_CONFIG_PATHS: Dict[str, Optional[str]] = {
    "alfworld": "GMemory/tasks/env_configs/alfworld_config.yaml",
    "fever": "GMemory/tasks/env_configs/fever_config.yaml",
    "pddl": "GMemory/tasks/env_configs/pddl_config.yaml",
}

# 项目根目录（src/data_processing/generators/ -> 上三级为 MME）
_MME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# PDDL 与 pddl_env.get_all_environment_configs 一致（GMemory 不可用时 fallback 用）
PDDL_TASK_NAMES = ["barman", "blockworld", "gripper", "tyreworld"]
PDDL_NUM_PROBLEMS = {"barman": 20, "blockworld": 10, "gripper": 20, "tyreworld": 10}


class GMemoryGenerator(BaseGenerator):
    """
    从 CollectorMemory 格式生成 sources 和 targets。
    当提供 target_file 时，使用相似度（similarity）做 source->target 映射。
    """

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        from GMemory.mas.utils import EmbeddingFunc
        self.embedding_func = EmbeddingFunc(model_type="sentence-transformers/all-MiniLM-L6-v2")

    def _get_task_env(self, max_trials: int = 50):
        """获取当前 dataset 对应的 task_env，用于调用 set_env 得到 task_main / task_description。"""
        from GMemory.tasks.envs import get_env
        env_config_path = ENV_CONFIG_PATHS.get(self.dataset_name)
        if not env_config_path:
            raise ValueError(f"No env config path for dataset {self.dataset_name}")
        full_path = os.path.join(_MME_ROOT, env_config_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Env config file not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            env_config = yaml.safe_load(f) or {}
        return get_env(self.dataset_name, env_config, max_trials)
    
    def generate_sources(
        self,
        source_file: str,  # InferenceOutputExample的jsonl或trajectories.json
        target_file: Optional[str] = None,  # InferenceInputExample的jsonl（用于mapping）
        filter_success_only: bool = False,
        max_targets_per_source: int = 5,
        threshold: float = 0.0,
        max_sources: int = -1,
        output_data_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        生成 sources（从 InferenceOutputExample 文件）。
        若提供 target_file，则用相似度做映射并写回 target_ids。
        """
        sources = []
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                output_example = InferenceOutputExample.from_dict(data)
                if filter_success_only and not output_example.label:
                    continue
                source = SourceExample(
                    source=output_example.source,
                    id=output_example.id,
                    task_main=output_example.task_main,
                    messages=output_example.messages,
                    target_ids=[],
                    memory="",
                    metadata=output_example.metadata.copy()
                )
                if output_example.label is not None:
                    source.metadata['label'] = output_example.label
                sources.append(source)

        print(f"[GMemoryGenerator] Loaded {len(sources)} sources from {source_file}")

        targets = []
        assert target_file and os.path.exists(target_file)
        with open(target_file, 'r', encoding='utf-8') as f:
            for line in f:
                targets.append(InferenceInputExample.from_dict(json.loads(line)))
        print(f"[GMemoryGenerator] Loaded {len(targets)} targets, running similarity mapping")
        
        assert len(sources) == len(targets)
        print_example("First Source Example", sources[0].to_dict())
        print_example("First Target Example", targets[0].to_dict())
        
        map_by_similarity(
            sources=sources,
            targets=targets,
            embedding_func=self.embedding_func,
            max_targets_per_source=max_targets_per_source,
            threshold=threshold
        )

        if max_sources > 0 and len(sources) > max_sources:
            sources = random.sample(sources, max_sources)

        if sources:
            print_example("First Processed Source Example", sources[0].to_dict())

        if output_data_dir:
            self.save_sources(sources, output_data_dir)

    def generate_targets(
        self, 
        output_data_dir: Optional[str] = None, 
        max_targets: int = -1,
        **kwargs
    ) -> None:
        """
        从原始 raw 文件生成 InferenceInputExample（用于 collection 阶段的 inference）。
        数据路径由 DATASETS_PATH 固定，无需传入 input_file。
        """
        if self.dataset_name not in DATASETS_PATH:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        input_file = DATASETS_PATH[self.dataset_name]
        
        if self.dataset_name == "pddl":
            # 与 GMemory/tasks/envs/__init__.py 和 pddl_env.get_all_environment_configs 一致
            try:
                from GMemory.tasks.envs import TASK_NAMES
                from GMemory.tasks.envs.pddl_env.pddl_env import get_all_environment_configs
                data = get_all_environment_configs(TASK_NAMES, input_file)
            except ImportError:
                # GMemory 不可用时复用相同逻辑：按 TASK_NAMES / Num_Problems 顺序解析 label 文件
                difficulties = []
                with open(input_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line.strip())
                        if "difficulty" not in obj:
                            raise ValueError("No difficulty in annotation file")
                        difficulties.append(obj["difficulty"])
                iter_num = 0
                data = []
                for game_name in PDDL_TASK_NAMES:
                    for i in range(PDDL_NUM_PROBLEMS[game_name]):
                        data.append({
                            "game_name": game_name,
                            "problem_index": i,
                            "difficulty": difficulties[iter_num],
                        })
                        iter_num += 1
        elif input_file.endswith(".jsonl"):
            with open(input_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        else:
            data = json.load(open(input_file, "r", encoding="utf-8"))

        if max_targets > 0 and len(data) > max_targets:
            print(f"[GMemoryGenerator] Truncated {len(data)} targets to {max_targets}")
            data = data[:max_targets]

        # 仅在 pddl 时通过对应 task_env 的 set_env 获取 task_main / task_description；
        # alfworld / fever 为避免环境初始化开销，直接在此处复现原有逻辑。
        task_env = self._get_task_env() if self.dataset_name == "pddl" else None
        
        targets = []
        for idx, item in enumerate(data):
            metadata = self._convert_target_data(item, idx)
            # dataset-specific construction of task_main / task_description
            if self.dataset_name == "alfworld":
                task_main, task_description = self._build_alfworld_task_fields(metadata)
            elif self.dataset_name == "fever":
                task_main, task_description = self._build_fever_task_fields(metadata)
            elif task_env is not None:
                # pddl: 仍然依赖环境来生成描述（需要 goal / 初始观测）
                task_main, task_description = task_env.set_env(metadata)
            else:
                raise ValueError(f"No task env or formatting rule for dataset {self.dataset_name}")
            metadata['task_description'] = task_description
            
            target = InferenceInputExample(
                source=self.dataset_name,
                id=idx,
                task_main=task_main,
                metadata=metadata
            )
            targets.append(target)
        
        print(f"[GMemoryGenerator] Generated {len(targets)} targets from {input_file}")

        if targets:
            print_example("First Target Example", targets[0].to_dict())

        # 保存到文件
        if output_data_dir:
            self.save_targets(targets, output_data_dir)
    
    def _build_alfworld_task_fields(self, metadata: dict) -> Tuple[str, str]:
        """
        为 alfworld 复现原 GMemory 中 AlfworldEnv._parse_task_main/_parse_task_description 的行为，
        在不实际初始化环境的前提下直接构造 task_main / task_description。
        """
        task: str = metadata.get("task", "")
        env_name: str = metadata.get("env_name", "")

        # 对应 AlfworldEnv._parse_task_main:
        #   return self.env_name + '-' + re.search(r'Your task is to:\\s*(.+)', task, re.DOTALL).group(1).strip()
        m = re.search(r"Your task is to:\s*(.+)", task, re.DOTALL)
        if m:
            suffix = m.group(1).strip()
        else:
            # 保底：如果格式不符合预期，就直接用整个 task 作为后缀
            suffix = task.strip()
        task_main = f"{env_name}-{suffix}" if env_name else suffix

        # 对应 AlfworldEnv._parse_task_description:
        #   return task.split('___')[0]
        task_description = task.split("___")[0]
        return task_main, task_description

    def _build_fever_task_fields(self, metadata: dict) -> Tuple[str, str]:
        """
        为 fever 复现 FeverEnv.set_env 中的行为：
            task = f"Claim: {configs['task']}"
            return task, task
        """
        claim: str = metadata.get("task", "")
        text = f"Claim: {claim}"
        return text, text
    
    def _convert_target_data(self, item: dict, idx: int) -> dict:
        """
        将原始target数据转换为metadata。
        保留原字段命名，不统一为goal。
        """
        metadata = {}
        
        if self.dataset_name == 'alfworld':
            goal = item.get('goal', '')
            gamefile = item.get('gamefile', '')
            metadata['task'] = goal
            metadata['env_kwargs'] = {
                'config': 'alfworld',
                'gamefile': gamefile,
            }
            from GMemory.tasks.envs.alfworld_env import get_env_name_from_gamefile, prefixes
            env_name = get_env_name_from_gamefile(gamefile)
            metadata['task_type'] = prefixes[env_name]
            metadata['env_name'] = env_name
        
        elif self.dataset_name == 'fever':
            claim = item.get('claim', '')
            label = item.get('label', item.get('answer', ''))
            metadata['task'] = claim
            metadata['answer'] = label
            metadata['env_name'] = 'fever'
        
        elif self.dataset_name == "pddl":
            # 与 pddl_env.get_all_environment_configs 输出一致，只保留这些字段
            metadata["game_name"] = item["game_name"]
            metadata["problem_index"] = item["problem_index"]
            if "difficulty" in item:
                metadata["difficulty"] = item["difficulty"]
            metadata["env_name"] = "pddl"

        return metadata
