"""
Extraction pipeline CLI.

Per src-structure-refactor.plan.md (extraction section):
- Take sources produced by generators (GMemory / Dynamic-cheatsheet)
- Run LLMExtractor to extract memories
- Save enriched sources to source.jsonl

This script handles:
- GMemory-style datasets (alfworld / fever / pddl)
- Dynamic-cheatsheet tasks (DC_TASKS)
- LongMemEval: input only source_file (raw longmemeval_oracle.json); generator produces source_input.jsonl and target.jsonl.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from src.data_processing.generators.factory import create_generator


def _normalize_dataset_for_generator(dataset_name: str) -> str:
    """
    Normalize dataset name for generator / directory layout reuse.

    For now we simply return the dataset_name unchanged; this helper is left
    here for future extension (mirrors collection pipeline style).
    """
    return dataset_name


def run_generation(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    generator_kwargs: Optional[dict] = None,
    refresh: bool = False,
    source_file: Optional[str] = None,
) -> None:
    """
    Run the extraction pipeline for one dataset.

    Steps:
    1) Use generator.generate_sources(...) to build source_input.jsonl from
       inference outputs and target.jsonl (if needed).
    2) Load SourceExample list from source_input.jsonl.
    3) Run LLMExtractor over each source to produce memory.
    4) Save enriched sources to source.jsonl.
    """
    generator_kwargs = dict(generator_kwargs or {})

    dataset_key = _normalize_dataset_for_generator(dataset_name)
    work_dir = os.path.join(output_dir, dataset_key, model_name)
    os.makedirs(work_dir, exist_ok=True)

    # 1) generate_sources -> source_input.jsonl (and for longmemeval also target.jsonl)
    source_input_file = os.path.join(work_dir, "source_input.jsonl")
    if not os.path.exists(source_input_file) or refresh:
        gen = create_generator(dataset_name)

        if dataset_name == "longmemeval":
            # LongMemEval: only raw source_file (e.g. longmemeval_oracle.json); no target_file.
            # Generator produces both source_input.jsonl and target.jsonl in work_dir.
            raw_source = source_file
            if raw_source is None or not os.path.exists(raw_source):
                raise FileNotFoundError(
                    f"LongMemEval raw data not found at {raw_source}. "
                    f"Provide path via --source_file or place longmemeval_oracle.json in {work_dir}."
                )
            gen.generate_sources(
                source_file=raw_source,
                target_file=None,
                output_data_dir=work_dir,
                **generator_kwargs,
            )
        elif dataset_name in ["personamem-v2-val"]:
            # PersonaMem-v2: requires local CSV (val.csv).
            raw_source = source_file
            if raw_source is None or not os.path.exists(raw_source):
                raise FileNotFoundError(
                    f"PersonaMem-v2 CSV not found at {raw_source}. "
                    f"Provide path via --source_file."
                )
            gen.generate_sources(
                source_file=raw_source,
                target_file=None,
                output_data_dir=work_dir,
                **generator_kwargs,
            )
        elif dataset_name == "prefeval-mcq":
            # PrefEval: source_file is the persona-driven directory.
            raw_source = source_file
            if raw_source is None or not os.path.isdir(raw_source):
                raise FileNotFoundError(
                    f"PrefEval persona-driven directory not found at {raw_source}. "
                    f"Provide path via --source_file "
                    f"(e.g. data/PrefEval/implicit_preference/persona-driven)."
                )
            gen.generate_sources(
                source_file=raw_source,
                target_file=None,
                output_data_dir=work_dir,
                **generator_kwargs,
            )
        elif dataset_name in ["membench-high", "membench-low"]:
            # MemBench: only raw source_file (e.g. FirstAgentDataHighLevel_multiple_0.json); no target_file.
            raw_source = source_file or os.path.join(work_dir, "membench_data.json")
            if not os.path.exists(raw_source):
                raise FileNotFoundError(
                    f"MemBench raw data not found at {raw_source}. "
                    f"Provide path via --source_file or place membench_data.json in {work_dir}."
                )
            gen.generate_sources(
                source_file=raw_source,
                target_file=None,
                output_data_dir=work_dir,
                **generator_kwargs,
            )
        else:
            # GMemory / DC: expect target.jsonl + target_output.jsonl from collection pipeline.
            target_file = os.path.join(work_dir, "target.jsonl")
            output_file = os.path.join(
                work_dir,
                "1-shot"
                if dataset_key in ["alfworld", "pddl", "fever", "babyai", "scienceworld"]
                else "zero-shot",
                "target_output.jsonl",
            )
            if not os.path.exists(target_file):
                raise FileNotFoundError(
                    f"target.jsonl not found at {target_file}. "
                    f"Please run the collection pipeline first."
                )
            if not os.path.exists(output_file):
                raise FileNotFoundError(
                    f"target_output.jsonl not found at {output_file}. "
                    f"Please run the collection pipeline first to generate inference outputs."
                )
            effective_target_file = target_file
            gen.generate_sources(
                source_file=output_file,
                target_file=effective_target_file,
                output_data_dir=work_dir,
                **generator_kwargs,
            )

        assert os.path.exists(source_input_file), (
            f"Expected generator to create {source_input_file} but file not found."
        )
    else:
        print(f"source_input.jsonl already exists at {source_input_file}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MyMem memory extraction pipeline.")
    p.add_argument(
        "--dataset_name",
        required=True,
        choices=[
            "alfworld",
            "fever",
            "pddl",
            "babyai",
            "scienceworld",
            "longmemeval",
            "membench-high",
            "membench-low",
            "personamem-v2-val",
            "prefeval-mcq",
            "GameOf24",
            "AIME_2024",
            "AIME_2025",
            "AIME_2020_2024",
            "GPQA_Diamond",
            "MMLU_Pro_Physics",
            "MMLU_Pro_Engineering",
            "MathEquationBalancer",
            "bigcodebench",
            "toolbench",
        ],
    )
    p.add_argument(
        "--model_name",
        required=True,
        help="Model name for src.llm (litellm)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory where collection outputs are stored (same as run_collection).",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate source_input.jsonl even if it already exists.",
    )
    p.add_argument(
        "--source_file",
        default=None,
        help="For longmemeval: path to raw longmemeval_oracle.json. Default: output_dir/data/longmemeval/model_name/longmemeval_oracle.json",
    )
    p.add_argument("--openai_api_base", type=str, default=None)
    p.add_argument("--openai_api_key", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    print(args)

    result = run_generation(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        generator_kwargs={
            "openai_api_base": args.openai_api_base,
            "openai_api_key": args.openai_api_key,
        },
        refresh=args.refresh,
        source_file=args.source_file,
    )

    print("[Generation] Done.")


if __name__ == "__main__":
    main()

