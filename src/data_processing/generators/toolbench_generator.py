"""
Generator for ToolBench.

Collection flow:
  generate_targets() -> target.jsonl  (from test instructions G1/G2/G3)
  run_collection -> target_output.jsonl  (multi-turn ReACT inference)
  generate_sources(target_output.jsonl, target.jsonl) -> source_input.jsonl

Data format:
  ToolBench test instructions are JSON files with fields:
    - query_id, query, api_list, relevant_apis, etc.
  Answer trajectories are JSONL with multi-turn conversations.
"""

from __future__ import annotations

import json
import os
import random
import hashlib
from typing import List, Optional

import litellm

from src.data_processing.generators.base import BaseGenerator
from src.data_processing.mapping_strategies import map_by_similarity
from src.data_processing.schemas import (
    InferenceInputExample,
    InferenceOutputExample,
    SourceExample,
)
from src.utils import print_example

TOOLBENCH_DATASETS = {"toolbench"}

_MME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_DATA_DIR = os.path.join(_MME_ROOT, "data", "toolbench")


def _normalize_query_id(query_id: str) -> int:
    """Convert ToolBench query_id to integer id."""
    if isinstance(query_id, int):
        return query_id
    s = str(query_id).strip()
    if s.isdigit():
        return int(s)
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _load_toolbench_instructions(data_dir: str) -> List[dict]:
    """
    Load ToolBench test instructions from data_dir.

    Searches for:
    - test_instructions.json or test_instructions.jsonl (single file)
    - G1_instruction.json, G2_instruction.json, G3_instruction.json (split files)
    """
    # Try single file first
    for fname in ("test_instructions.json", "test_instructions.jsonl"):
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            items = []
            with open(path, "r", encoding="utf-8") as f:
                if fname.endswith(".jsonl"):
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        items = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
            print(f"[ToolBenchGenerator] Loaded {len(items)} instructions from {path}")
            return items

    # Try G1/G2/G3 split files (instruction splits only for OOD test)
    items = []
    for split in ("G1_instruction", "G2_instruction", "G3_instruction"):
        path = os.path.join(data_dir, f"{split}.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    item.setdefault("split", split)
                items.extend(data)
            elif isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, dict):
                        val.setdefault("split", split)
                        val.setdefault("query_id", key)
                        items.append(val)
            print(f"[ToolBenchGenerator] Loaded {len(data) if isinstance(data, (list, dict)) else 0} from {path}")

    if not items:
        raise FileNotFoundError(
            f"No ToolBench instruction files found in {data_dir}. "
            "Expected test_instructions.json/.jsonl or G1_instruction.json, G2_instruction.json, G3_instruction.json"
        )
    print(f"[ToolBenchGenerator] Total instructions loaded: {len(items)}")
    return items


class ToolBenchGenerator(BaseGenerator):
    """
    Generator for ToolBench tool-calling benchmark.

    generate_targets(): load test instructions -> InferenceInputExample per query.
    generate_sources(): load collection output -> SourceExample; similarity mapping.
    """

    def __init__(self, dataset_name: str = "toolbench"):
        super().__init__(dataset_name)

    @staticmethod
    def _build_embedding_func(
        *,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
    ):
        class _EmbeddingFunc:
            def __init__(self, model_name: str, base_url: Optional[str], key: Optional[str]):
                self.model_name = model_name
                self.base_url = base_url
                self.api_key = key

            def embed_query(self, text: str):
                kwargs = {"model": self.model_name, "input": [text]}
                if self.base_url:
                    kwargs["api_base"] = self.base_url
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                resp = litellm.embedding(**kwargs)
                data = resp.data[0]
                return data["embedding"] if isinstance(data, dict) else data.embedding

        return _EmbeddingFunc(model_name=model, base_url=api_base, key=api_key)

    def _resolve_embedding_func(self, **kwargs):
        api_base = kwargs.get("openai_api_base")
        api_key = kwargs.get("openai_api_key")
        if api_key is not None:
            model = "text-embedding-3-large"
            print(f"[ToolBenchGenerator] Using embedding: OpenAI ({model})")
            return self._build_embedding_func(model=model, api_base=api_base, api_key=api_key)
        from GMemory.mas.utils import EmbeddingFunc
        print("[ToolBenchGenerator] Using embedding: BGE (BAAI/bge-base-en-v1.5)")
        return EmbeddingFunc(model_type="BAAI/bge-base-en-v1.5")

    def generate_targets(
        self,
        output_data_dir: Optional[str] = None,
        max_targets: int = -1,
        data_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Load ToolBench test instructions and produce target.jsonl.

        Each instruction becomes one InferenceInputExample with:
        - task_main: the user query/instruction
        - metadata: api_list, query_id, split, relevant_apis, tool docs
        """
        data_dir = data_path or _DEFAULT_DATA_DIR
        items = _load_toolbench_instructions(data_dir)

        if max_targets > 0:
            items = items[:max_targets]

        targets: List[InferenceInputExample] = []
        for idx, item in enumerate(items):
            query = item.get("query", "") or item.get("instruction", "")
            query_id = item.get("query_id", idx)
            example_id = _normalize_query_id(query_id)

            # api_list may be a list of API descriptions
            api_list = item.get("api_list", [])
            if isinstance(api_list, str):
                try:
                    api_list = json.loads(api_list)
                except json.JSONDecodeError:
                    api_list = [{"description": api_list}]

            metadata = {
                "query_id": str(query_id),
                "split": item.get("split", ""),
                "api_list": api_list,
                "relevant_apis": item.get("relevant_apis", []),
            }

            target = InferenceInputExample(
                source="toolbench",
                id=example_id,
                task_main=query,
                metadata=metadata,
            )
            targets.append(target)

        print(f"[ToolBenchGenerator] Generated {len(targets)} targets")
        if targets:
            print_example("First Target Example", targets[0].to_dict())

        if output_data_dir:
            self.save_targets(targets, output_data_dir)

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
        Load collection output (target_output.jsonl) -> SourceExample.
        Run similarity mapping to assign target_ids (one-to-many).
        """
        sources: List[SourceExample] = []
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                out = InferenceOutputExample.from_dict(data)
                if filter_success_only and not out.label:
                    continue
                source = SourceExample(
                    source=out.source,
                    id=out.id,
                    task_main=out.task_main,
                    messages=out.messages.copy(),
                    target_ids=[],
                    memory="",
                    metadata=out.metadata.copy(),
                )
                if out.label is not None:
                    source.metadata["label"] = out.label
                sources.append(source)

        print(f"[ToolBenchGenerator] Loaded {len(sources)} sources from {source_file}")

        if target_file and os.path.exists(target_file):
            targets: List[InferenceInputExample] = []
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        targets.append(InferenceInputExample.from_dict(json.loads(line)))
            print(f"[ToolBenchGenerator] Loaded {len(targets)} targets, running similarity mapping")
            if sources:
                print_example("First Source Example", sources[0].to_dict())
            if targets:
                print_example("First Target Example", targets[0].to_dict())

            embedding_func = self._resolve_embedding_func(**kwargs)
            map_by_similarity(
                sources=sources,
                targets=targets,
                embedding_func=embedding_func,
                max_targets_per_source=max_targets_per_source,
                threshold=threshold,
            )

        if max_sources > 0 and len(sources) > max_sources:
            random.seed(42)
            sources = random.sample(sources, max_sources)

        if sources:
            print_example("First Processed Source Example", sources[0].to_dict())

        if output_data_dir:
            self.save_sources(sources, output_data_dir)
