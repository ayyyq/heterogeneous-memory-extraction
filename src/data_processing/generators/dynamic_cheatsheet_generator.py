"""
Generator for Dynamic Cheatsheet (DC) datasets.

Data source is fixed (no input_file parameter):
- GameOf24: load_dataset("turingmachine/meta-prompting")[task]
- Others: load_from_disk("data/dynamic-cheatsheet/data/{task}") (hardcoded path)

Produces InferenceInputExample with task_main = raw_input (for similarity/mapping)
and metadata = {input: formatted_prompt, target}.
"""

import json
import os
import re
from typing import Optional, List, Dict, Tuple

import numpy as np

from src.data_processing.schemas import SourceExample, InferenceInputExample, InferenceOutputExample
from src.data_processing.generators.base import BaseGenerator
from src.data_processing.mapping_strategies import map_by_similarity
from src.utils import print_example


# DC task names; same as run_benchmark.py
DC_TASKS = [
    "GameOf24",
    "AIME_2024",
    "AIME_2025",
    "AIME_2020_2024",
    "GPQA_Diamond",
    "MMLU_Pro_Physics",
    "MMLU_Pro_Engineering",
    "MathEquationBalancer",
]

# GameOf24 prompt; copy from data/dynamic-cheatsheet/run_benchmark.py
PREDEFINED_PROMPTS = {
    "GameOf24": (
        "Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, "
        "combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, "
        "to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. "
        "Please present a single expression that evaluates to 24."
    ),
}

# AIME instruction suffix (exact string from run_benchmark.py)
AIME_SUFFIX = (
    " (Please provide your answer in the form of an integer, e.g., 1234, with no Markdown formatting or additional text; "
    "make sure to pay attention to the desired format of the final answer though.)"
)

# MathEquationBalancer wrap (instruction + "Equation: " + base)
MATH_EQUATION_INSTRUCTION = (
    "Below is an equation with missing operators. Your task is to fill in the blanks with the correct mathematical "
    "operators: +, -, *, or /. Ensure that the equation is correct once the operators are added. The operators should "
    "be placed in the sequence they appear from left to right. Include the full equation with the operators filled in. "
    "For instance, for the equation 1 ? 2 ? 3 = 6, the correct answer is 1 + 2 + 3 = 6.\n\nEquation: "
)


def format_input_for_task(task_name: str, raw_input: str, question_id: int) -> str:
    """
    Build the formatted input string sent to the model (same logic as run_benchmark.py).

    - GameOf24: PREDEFINED_PROMPTS["GameOf24"] + "\\n\\nQuestion #N:\\n" + raw_input
    - AIME_*: "Question #N:\\n" + raw_input + AIME_SUFFIX
    - MathEquationBalancer: MATH_EQUATION_INSTRUCTION + "Question #N:\\n" + raw_input
    - Else (GPQA_Diamond, MMLU_Pro_*): "Question #N:\\n" + raw_input
    """
    q_prefix = f"Question #{question_id}:\n{raw_input}"
    if task_name in PREDEFINED_PROMPTS:
        return f"{PREDEFINED_PROMPTS[task_name]}\n\n{q_prefix}"
    if task_name in ("AIME_2020_2024", "AIME_2024", "AIME_2025"):
        return q_prefix + AIME_SUFFIX
    if task_name == "MathEquationBalancer":
        return MATH_EQUATION_INSTRUCTION + q_prefix
    return q_prefix


def _load_dc_dataset(dataset_name: str):
    """Load dataset exactly as run_benchmark.py: GameOf24 via load_dataset, others via load_from_disk."""
    from datasets import load_dataset, load_from_disk

    if dataset_name in PREDEFINED_PROMPTS and dataset_name != "P3_Test":
        ds = load_dataset("turingmachine/meta-prompting")
        return ds[dataset_name]
    if dataset_name in [
        "GPQA_Diamond", "AIME_2020_2024", "AIME_2024", "AIME_2025",
        "MMLU_Pro_Physics", "MMLU_Pro_Engineering", "MathEquationBalancer",
    ]:
        path = f"data/dynamic-cheatsheet/data/{dataset_name}"
        return load_from_disk(path)
    raise ValueError(
        f"Task {dataset_name} is not recognized. "
        f"Supported: {DC_TASKS}"
    )


def _normalize(s: str) -> str:
    """Normalize string for embedding lookup: newlines, trailing space, LaTeX backslash escape."""
    if not isinstance(s, str):
        return str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Collapse escaped backslashes before LaTeX commands: \\triangle -> \triangle
    # (dataset may store \\ from JSON/Arrow escape, CSV has single \)
    s = re.sub(r"\\\\([a-zA-Z])", r"\\\1", s)
    return s


def _load_dc_embeddings_from_csv(
    csv_path: str,
    sources: List[SourceExample],
    targets: List[InferenceInputExample],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Load precomputed embeddings from DC-style CSV (columns: input, embedding).
    Align by task_main (raw input): first occurrence per input.
    Returns (source_embeddings, target_embeddings) with keys source.id / target.id.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    input_to_emb: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        inp = _normalize(row["input"])
        if inp not in input_to_emb:
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = eval(emb)
            input_to_emb[inp] = np.array(emb, dtype=float)
    source_emb: Dict[int, np.ndarray] = {
        s.id: input_to_emb[_normalize(s.task_main)]
        for s in sources
    }
    target_emb: Dict[int, np.ndarray] = {
        t.id: input_to_emb[_normalize(t.task_main)]
        for t in targets
    }
    return source_emb, target_emb


class DynamicCheatsheetGenerator(BaseGenerator):
    """
    Generator for Dynamic Cheatsheet tasks.

    - generate_targets(...): Does not accept input_file. Loads data from fixed logic:
      GameOf24 from HF, others from load_from_disk("data/dynamic-cheatsheet/data/{task}").
      Emits InferenceInputExample with task_main=raw_input, metadata={input, target}.
    - generate_sources(source_file, target_file, ...): reads DC output jsonl, builds SourceExample;
      若提供 target_file，则用预计算 embedding 文件（CSV）做相似度映射。
    """

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        if dataset_name not in DC_TASKS:
            raise ValueError(f"dataset_name must be one of {DC_TASKS}, got {dataset_name!r}")

    def generate_targets(
        self,
        output_data_dir: Optional[str] = None,
        max_targets: int = -1,
        **kwargs
    ) -> None:
        """
        Generate target.jsonl from DC raw data.

        Data is loaded from fixed logic: GameOf24 from HF, others from
        load_from_disk("data/dynamic-cheatsheet/data/{task}"). No input_file parameter.
        """
        dataset = _load_dc_dataset(self.dataset_name)

        targets: List[InferenceInputExample] = []
        n = len(dataset)
        if max_targets > 0 and n > max_targets:
            n = max_targets
            print(f"[DynamicCheatsheetGenerator] Truncating to {max_targets} targets")

        for idx in range(n):
            example = dataset[idx]
            raw_input = example["input"]
            target = example["target"]
            formatted = format_input_for_task(self.dataset_name, raw_input, idx + 1)
            metadata = {"input": formatted, "target": target}
            ex = InferenceInputExample(
                source=self.dataset_name,
                id=idx,
                task_main=raw_input,
                metadata=metadata,
            )
            targets.append(ex)

        print(f"[DynamicCheatsheetGenerator] Generated {len(targets)} targets from {self.dataset_name}")

        if targets:
            print_example("First Target Example", targets[0].to_dict())

        if output_data_dir:
            self.save_targets(targets, output_data_dir)

    def generate_sources(
        self,
        source_file: str,
        target_file: Optional[str] = None,
        output_data_dir: Optional[str] = None,
        max_targets_per_source: int = 5,
        threshold: float = 0.0,
        **kwargs
    ) -> None:
        """
        Build SourceExample list from InferenceOutputExample 的 jsonl（source_file 每行一个 InferenceOutputExample）。
        若提供 target_file，则从 data/dynamic-cheatsheet/embeddings/{dataset_name}.csv 读取预计算 embedding，做相似度映射。
        """
        sources: List[SourceExample] = []
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                output_example = InferenceOutputExample.from_dict(data)
                metadata = dict(output_example.metadata)
                if output_example.label is not None:
                    metadata["label"] = output_example.label
                source = SourceExample(
                    source=output_example.source,
                    id=output_example.id,
                    task_main=output_example.task_main,
                    messages=output_example.messages,
                    target_ids=[],
                    memory="",
                    metadata=metadata,
                )
                sources.append(source)

        print(f"[DynamicCheatsheetGenerator] Loaded {len(sources)} sources from {source_file}")

        assert target_file and os.path.exists(target_file)
        targets_list: List[InferenceInputExample] = []
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                targets_list.append(InferenceInputExample.from_dict(json.loads(line)))
        print(f"[DynamicCheatsheetGenerator] Loaded {len(targets_list)} targets, running similarity mapping (embedding file)")

        assert len(sources) == len(targets_list)
        print_example("First Source Example", sources[0].to_dict())
        print_example("First Target Example", targets_list[0].to_dict())

        csv_path = os.path.join("data", "dynamic-cheatsheet", "embeddings", f"{self.dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Precomputed embeddings CSV required but not found: {csv_path}")
        source_emb, target_emb = _load_dc_embeddings_from_csv(csv_path, sources, targets_list)
        map_by_similarity(
            sources=sources,
            targets=targets_list,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
            max_targets_per_source=max_targets_per_source,
            threshold=threshold,
        )

        if sources:
            print_example("First Processed Source Example", sources[0].to_dict())

        if output_data_dir:
            self.save_sources(sources, output_data_dir)

