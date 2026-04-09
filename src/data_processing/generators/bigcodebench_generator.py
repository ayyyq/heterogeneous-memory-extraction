"""
Generator for BigCodeBench.

Collection flow:
  generate_targets() -> target.jsonl
  run_collection -> target_output.jsonl
  generate_sources(target_output.jsonl, target.jsonl) -> source_input.jsonl
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

BIGCODEBENCH_DATASETS = {"bigcodebench"}

_MME_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_DATA_DIR = os.path.join(_MME_ROOT, "data", "bigcodebench")
_DEFAULT_SUBSET = "full"
_DEFAULT_PROMPT_SPLIT = "instruct"


def _default_data_path(subset: str = _DEFAULT_SUBSET) -> str:
    return os.path.join(_DEFAULT_DATA_DIR, f"bigcodebench_{subset}.jsonl")


def _normalize_task_id(task_id: str) -> int:
    # BigCodeBench task ids are usually like "BigCodeBench/123".
    if "/" in task_id:
        suffix = task_id.rsplit("/", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    if task_id.isdigit():
        return int(task_id)
    # Deterministic fallback for unexpected ids.
    digest = hashlib.md5(task_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


class BigCodeBenchGenerator(BaseGenerator):
    def __init__(self, dataset_name: str = "bigcodebench"):
        super().__init__(dataset_name)

    @staticmethod
    def _build_openai_embedding_func(
        *,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
    ):
        class _OpenAIEmbeddingFunc:
            def __init__(self, model_name: str, base_url: Optional[str], key: Optional[str]):
                self.model_name = model_name
                self.base_url = base_url
                self.api_key = key

            def embed_query(self, text: str):
                kwargs = {
                    "model": self.model_name,
                    "input": [text],
                }
                if self.base_url:
                    kwargs["api_base"] = self.base_url
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                resp = litellm.embedding(**kwargs)
                data = resp.data[0]
                if isinstance(data, dict):
                    return data["embedding"]
                return data.embedding

        return _OpenAIEmbeddingFunc(model_name=model, base_url=api_base, key=api_key)

    def _resolve_embedding_func(self, **kwargs):
        api_base = kwargs.get("openai_api_base")
        api_key = kwargs.get("openai_api_key")
        if api_key is not None:
            model = "text-embedding-3-large"
            print(f"[BigCodeBenchGenerator] Using embedding: OpenAI ({model})")
            return self._build_openai_embedding_func(
                model=model,
                api_base=api_base,
                api_key=api_key,
            )
        from GMemory.mas.utils import EmbeddingFunc
        print("[BigCodeBenchGenerator] Using embedding: BGE (BAAI/bge-base-en-v1.5)")
        return EmbeddingFunc(model_type="BAAI/bge-base-en-v1.5")
        
    def generate_targets(
        self,
        output_data_dir: Optional[str] = None,
        max_targets: int = -1,
        data_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        subset = _DEFAULT_SUBSET
        prompt_split = _DEFAULT_PROMPT_SPLIT

        path = data_path or _default_data_path(subset)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BigCodeBench data file not found: {path}. "
                "Expected data/bigcodebench/bigcodebench_full.jsonl."
            )

        targets: List[InferenceInputExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                task_id = str(item.get("task_id", ""))
                task_main = str(item.get(f"{prompt_split}_prompt", "") or "")
                if not task_main:
                    continue
                example_id = _normalize_task_id(task_id)
                target = InferenceInputExample(
                    source="bigcodebench",
                    id=example_id,
                    task_main=task_main,
                    metadata={
                        "task_id": task_id,
                        "entry_point": item.get("entry_point", ""),
                        "test": item.get("test", ""),
                        "libs": item.get("libs", ""),
                        "prompt_split": prompt_split,
                        "subset": subset,
                        "code_prompt": item.get("code_prompt", ""),
                        "instruct_prompt": item.get("instruct_prompt", ""),
                        "complete_prompt": item.get("complete_prompt", ""),
                    },
                )
                targets.append(target)

        if max_targets > 0:
            targets = targets[:max_targets]

        print(f"[BigCodeBenchGenerator] Generated {len(targets)} targets from {path}")
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
                # Append evaluation feedback as final user message when available.
                status = source.metadata.get("eval_status") or source.metadata.get("status")
                error = source.metadata.get("eval_error")
                if status is not None or error:
                    feedback_text = json.dumps(
                        {"status": status or "unknown", "error": str(error) if error else ""},
                        ensure_ascii=False,
                    )
                    source.messages.append({"role": "user", "content": "[Eval] " + feedback_text})
                sources.append(source)

        print(f"[BigCodeBenchGenerator] Loaded {len(sources)} sources from {source_file}")

        if target_file and os.path.exists(target_file):
            targets: List[InferenceInputExample] = []
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        targets.append(InferenceInputExample.from_dict(json.loads(line)))
            print(
                f"[BigCodeBenchGenerator] Loaded {len(targets)} targets, running similarity mapping"
            )
            if sources:
                print_example("First Source Example", sources[0].to_dict())
                print("First Source Example Messages\n", sources[0].get_conversation_text())
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
