"""
Generator for AgentBoard-style datasets: babyai, scienceworld, jericho.

Collection flow (similar to GMemory):
  generate_targets() -> target.jsonl (InferenceInputExample per episode)
  run_collection -> AgentBoardAdapter runs multi-turn env interaction
  generate_sources(collection_output.jsonl) -> source_input.jsonl (SourceExample)
"""

import json
import os
import random
from typing import Optional, List

random.seed(42)

from src.data_processing.schemas import SourceExample, InferenceInputExample, InferenceOutputExample
from src.data_processing.generators.base import BaseGenerator
from src.data_processing.mapping_strategies import map_by_similarity
from src.utils import print_example

AGENTBOARD_DATASETS = {"babyai", "scienceworld", "jericho"}

_MME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

LABEL_PATHS = {
    "babyai": "data/babyai/test.jsonl",
    "scienceworld": "data/scienceworld/test.jsonl",
    "jericho": "data/jericho/test.jsonl",
}

JERICHO_GAME_DIR = "data/jericho/z-machine-games-master/jericho-game-suite"

# Games that use .z8 instead of .z5
_JERICHO_Z8_GAMES = {"afflicted", "anchor", "snacktime", "partyfoul"}

# Mapping from test.jsonl subtask names to actual filenames in the game suite
_JERICHO_NAME_FIXES = {
    "darkhunt": "huntdark",
}


def _get_label_path(dataset_name: str) -> str:
    rel = LABEL_PATHS[dataset_name]
    return os.path.join(_MME_ROOT, rel)


def _get_jericho_game_file(game_name: str) -> str:
    file_name = _JERICHO_NAME_FIXES.get(game_name, game_name)
    ext = ".z8" if file_name in _JERICHO_Z8_GAMES else ".z5"
    return os.path.join(_MME_ROOT, JERICHO_GAME_DIR, file_name + ext)


class AgentBoardGenerator(BaseGenerator):
    """
    Generator for AgentBoard interactive environments: babyai, scienceworld, jericho.

    generate_targets(): reads label JSONL -> InferenceInputExample per episode.
    generate_sources(): loads collection outputs -> SourceExample with similarity mapping.
    """

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        if dataset_name not in AGENTBOARD_DATASETS:
            raise ValueError(f"AgentBoardGenerator: unsupported dataset '{dataset_name}'")
        from GMemory.mas.utils import EmbeddingFunc
        self.embedding_func = EmbeddingFunc(model_type="BAAI/bge-base-en-v1.5")

    # ------------------------------------------------------------------
    # generate_targets
    # ------------------------------------------------------------------

    def generate_targets(
        self,
        output_data_dir: Optional[str] = None,
        max_targets: int = -1,
        **kwargs,
    ) -> None:
        """
        Read label JSONL -> one InferenceInputExample per episode.
        Env is NOT initialized here; metadata stores all fields needed by AgentBoardAdapter.set_env.
        """
        label_path = _get_label_path(self.dataset_name)
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"[AgentBoardGenerator] Label file not found: {label_path}. "
                f"Place the AgentBoard test.jsonl at data/{self.dataset_name}/test.jsonl."
            )

        raw_data = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

        if max_targets > 0 and len(raw_data) > max_targets:
            print(f"[AgentBoardGenerator] Truncated {len(raw_data)} items to {max_targets}")
            raw_data = raw_data[:max_targets]

        targets: List[InferenceInputExample] = []
        for idx, item in enumerate(raw_data):
            metadata = self._extract_metadata(item, idx)
            task_main = self._build_task_main(metadata, idx)
            target = InferenceInputExample(
                source=self.dataset_name,
                id=idx,
                task_main=task_main,
                metadata=metadata,
            )
            targets.append(target)

        print(f"[AgentBoardGenerator] Generated {len(targets)} targets from {label_path}")
        if targets:
            print_example("First Target Example", targets[0].to_dict())

        if output_data_dir:
            self.save_targets(targets, output_data_dir)

    def _extract_metadata(self, item: dict, idx: int) -> dict:
        """Extract dataset-specific metadata from a label file entry."""
        if self.dataset_name == "babyai":
            additional = item.get("additional_info", {})
            return {
                "game_name": additional.get("subtask", "BabyAI-GoToRedBallGrey-v0"),
                "seed": item.get("seed", 1234),
                "obs_to_reward": item.get("subgoals", None),
                "difficulty": item.get("difficulty", "easy"),
                "env_name": "babyai",
                "goal": item["goal"],
                "init_obs": additional.get("init_obs", ""),
            }
        elif self.dataset_name == "scienceworld":
            additional = item.get("additional_info", {})
            return {
                "task_name": additional.get("env_name", ""),
                "var": additional.get("var", 0),
                "modified_goal": item.get("goal", ""),
                "subgoals": item.get("subgoals", []),
                "difficulty": item.get("difficulty", "easy"),
                "env_name": "scienceworld",
                "goal": item["goal"],
            }
        elif self.dataset_name == "jericho":
            additional = item.get("additional_info", {})
            game_name = additional.get("subtask", "")
            return {
                "game_name": game_name,
                "game_file": _get_jericho_game_file(game_name),
                "obs_to_reward": item.get("subgoals", None),
                "difficulty": item.get("difficulty", "easy"),
                "env_name": "jericho",
                "goal": item["goal"],
                "init_obs": additional.get("init_obs", ""),
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _build_task_main(self, metadata: dict, idx: int) -> str:
        return metadata["goal"]

    # ------------------------------------------------------------------
    # generate_sources
    # ------------------------------------------------------------------

    def generate_sources(
        self,
        source_file: str,
        target_file: Optional[str] = None,
        filter_success_only: bool = False,
        max_targets_per_source: int = 5,
        threshold: float = 0.0,
        max_sources: int = -1,
        output_data_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Load InferenceOutputExample jsonl (collection output) -> SourceExample list.
        If target_file is provided, run similarity mapping to populate target_ids.
        Mirrors GMemoryGenerator.generate_sources.
        """
        sources: List[SourceExample] = []
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
                    metadata=output_example.metadata.copy(),
                )
                if output_example.label is not None:
                    source.metadata["label"] = output_example.label
                sources.append(source)

        print(f"[AgentBoardGenerator] Loaded {len(sources)} sources from {source_file}")

        if target_file and os.path.exists(target_file):
            targets: List[InferenceInputExample] = []
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        targets.append(InferenceInputExample.from_dict(json.loads(line)))
            print(f"[AgentBoardGenerator] Loaded {len(targets)} targets, running similarity mapping")
            assert len(sources) == len(targets), (
                f"source/target count mismatch: {len(sources)} vs {len(targets)}"
            )
            if sources:
                print_example("First Source Example", sources[0].to_dict())
            if targets:
                print_example("First Target Example", targets[0].to_dict())

            map_by_similarity(
                sources=sources,
                targets=targets,
                embedding_func=self.embedding_func,
                max_targets_per_source=max_targets_per_source,
                threshold=threshold,
            )

        if max_sources > 0 and len(sources) > max_sources:
            sources = random.sample(sources, max_sources)

        if sources:
            print_example("First Processed Source Example", sources[0].to_dict())

        if output_data_dir:
            self.save_sources(sources, output_data_dir)
