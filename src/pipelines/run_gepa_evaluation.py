"""
GEPA/evolution evaluation pipeline: extraction (by prompt) then inference + evaluation.

Flow:
1. Extraction: from source_input.jsonl, extract memory using the selected prompt source -> source.jsonl.
   For no_memory: skip extraction; set memory="" and write source.jsonl.
2. Evaluation: load source.jsonl, run inference with memory (use_memory=True),
   evaluate -> source_target_output.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.data_processing.loader import DataLoader
from src.data_processing.schemas import SourceExample
from src.extraction import LLMExtractor
from src.extraction.extraction_prompts import GENERAL_USER_PROMPT
from src.pipelines.run_evaluation import run_evaluation_pipeline


BUILTIN_PROMPT_TYPES = {
    "no_memory",
    "mem0",
    "success-failure",
    "factual-experiential",
    "survey",
    "simple",
    "reasoningbank",
    "openmemory",
}
EVOLUTION_PROMPT_TYPES = {"gepa", "ace", "memevolve"}
PROMPT_TYPES = BUILTIN_PROMPT_TYPES | EVOLUTION_PROMPT_TYPES


def _parse_comma_separated(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _path_safe(value: str) -> str:
    return value.replace(os.sep, "_").replace("/", "_").replace("\\", "_")


def _load_nonempty_text(path: str, label: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{label} is empty: {path}")
    return text


def _load_split_entries(split_file: str) -> List[dict]:
    if not os.path.isfile(split_file):
        raise ValueError(f"split_file not found: {split_file}")
    try:
        with open(split_file, "r", encoding="utf-8") as f:
            split_cfg = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse split_file {split_file}: {e}") from e

    if not isinstance(split_cfg, dict) or "datasets" not in split_cfg:
        raise ValueError(
            f"Invalid split_file format at {split_file}: expected a dict with key 'datasets'."
        )
    entries = split_cfg.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Invalid split_file {split_file}: field 'datasets' must be a non-empty list."
        )
    return entries


def _resolve_datasets_and_files(
    datasets_arg: str,
    source_files_arg: str,
    target_files_arg: str,
    split_file: Optional[str],
) -> Tuple[List[str], List[str], List[str], Dict[str, Set[str]]]:
    if split_file is not None:
        entries = _load_split_entries(split_file)
        datasets: List[str] = []
        source_files: List[str] = []
        target_files: List[str] = []
        split_train_ids: Dict[str, Set[str]] = {}

        for entry in entries:
            dataset_name = entry.get("name")
            source_file_path = entry.get("source_file_path")
            target_file_path = entry.get("target_file_path")
            train_source_ids = entry.get("train_source_ids")

            if (
                not isinstance(dataset_name, str)
                or not isinstance(source_file_path, str)
                or not isinstance(target_file_path, str)
            ):
                raise ValueError(
                    "Each split dataset entry must contain string fields: "
                    "'name', 'source_file_path', 'target_file_path'."
                )
            if not isinstance(train_source_ids, list):
                raise ValueError(
                    f"split_file dataset '{dataset_name}' has invalid 'train_source_ids' "
                    "(must be a list)."
                )
            if dataset_name in split_train_ids:
                raise ValueError(
                    f"Duplicate dataset name '{dataset_name}' in split_file is not supported."
                )

            datasets.append(dataset_name)
            source_files.append(source_file_path)
            target_files.append(target_file_path)
            split_train_ids[dataset_name] = set(train_source_ids)

        return datasets, source_files, target_files, split_train_ids

    datasets = _parse_comma_separated(datasets_arg)
    source_files = _parse_comma_separated(source_files_arg)
    target_files = _parse_comma_separated(target_files_arg)
    if not (datasets and source_files and target_files):
        raise ValueError(
            "When --split_file is not provided, --datasets, --source_files and "
            "--target_files must all be provided and non-empty."
        )
    if not (len(datasets) == len(source_files) == len(target_files)):
        raise ValueError(
            "datasets, source_files and target_files must have the same length.\n"
            f"  datasets={datasets}\n  source_files={source_files}\n  target_files={target_files}"
        )
    return datasets, source_files, target_files, {}


def _evolution_dir_suffix(prompt_type: str, evolution_run_dir: str) -> str:
    """Use last 2 path components if prompt_type not found in the path, else just basename."""
    norm = os.path.normpath(evolution_run_dir)
    if prompt_type in norm:
        return os.path.basename(norm)
    parts = norm.split(os.sep)
    return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _prompt_type_dir(
    prompt_type: str,
    evolution_run_dir: Optional[str],
    evolution_prompt_file: Optional[str],
    gepa_prompt_index: Optional[int],
) -> str:
    if prompt_type in BUILTIN_PROMPT_TYPES:
        return prompt_type
    if prompt_type == "gepa":
        if evolution_run_dir is None or gepa_prompt_index is None:
            raise ValueError("prompt_type=gepa requires --evolution_run_dir and --gepa_prompt_index")
        return _path_safe(f"gepa_{evolution_run_dir}_{gepa_prompt_index}")
    if prompt_type == "ace":
        if evolution_prompt_file:
            return _path_safe(f"ace_file_{os.path.basename(evolution_prompt_file)}")
        if evolution_run_dir is None:
            raise ValueError("prompt_type=ace requires --evolution_run_dir or --evolution_prompt_file")
        return _path_safe(f"ace_{_evolution_dir_suffix('ace', evolution_run_dir)}")
    if prompt_type == "memevolve":
        if evolution_prompt_file:
            return _path_safe(f"memevolve_file_{os.path.basename(evolution_prompt_file)}")
        if evolution_run_dir is None:
            raise ValueError("prompt_type=memevolve requires --evolution_run_dir or --evolution_prompt_file")
        return _path_safe(f"memevolve_{_evolution_dir_suffix('memevolve', evolution_run_dir)}")
    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def _use_all_targets(dataset_name: str) -> bool:
    """Datasets that should use all targets (max_targets_per_source=None)."""
    lowered = dataset_name.lower()
    return (
        lowered in {
            "membench",
            "membench-high",
            "membench-low",
            "longmemeval",
            "personamem-v2",
            "personamem-v2-val",
            "prefeval",
            "prefeval-mcq",
        }
        or lowered.startswith("personamem-v2")
        or lowered.startswith("prefeval")
    )


def _count_test_sources(
    source_file: str,
    split_train_source_ids: Optional[Set[str]],
    num_sources_per_dataset: Optional[int],
) -> int:
    loader = DataLoader(source_file=source_file)
    sources = list(loader.load_sources().values())
    if split_train_source_ids is not None:
        sources = [s for s in sources if s.id not in split_train_source_ids]
    if num_sources_per_dataset is not None and num_sources_per_dataset > 0:
        sources = sources[:num_sources_per_dataset]
    return len(sources)


def _build_custom_prompt_builder(
    prompt_type: str,
    evolution_run_dir: Optional[str],
    evolution_prompt_file: Optional[str],
    gepa_prompt_index: Optional[int],
) -> Callable[[SourceExample], Tuple[str, str]]:
    if prompt_type == "gepa":
        from gepa.core.state import GEPAState
        from src.evolution_gepa.utils import build_extraction_prompt

        if evolution_run_dir is None or gepa_prompt_index is None:
            raise ValueError("prompt_type=gepa requires --evolution_run_dir and --gepa_prompt_index")
        state_path = os.path.join(evolution_run_dir, "gepa_state.bin")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"GEPA state not found: {state_path}")
        state = GEPAState.load(evolution_run_dir)
        candidate = state.program_candidates[gepa_prompt_index]

        def _builder(src: SourceExample) -> Tuple[str, str]:
            return build_extraction_prompt(candidate, src.get_conversation_text())

        return _builder

    if prompt_type == "memevolve":
        if evolution_prompt_file:
            prompt_path = evolution_prompt_file
        else:
            if evolution_run_dir is None:
                raise ValueError("prompt_type=memevolve requires --evolution_run_dir or --evolution_prompt_file")
            prompt_path = os.path.join(evolution_run_dir, "best_prompt.txt")
        system_prompt = _load_nonempty_text(prompt_path, "memevolve prompt file")

        def _builder(src: SourceExample) -> Tuple[str, str]:
            user_prompt = GENERAL_USER_PROMPT.format(conversation=src.get_conversation_text())
            return system_prompt, user_prompt

        return _builder

    if prompt_type == "ace":
        from src.evolution_ace.playbook import render_playbook_to_system_prompt

        if evolution_prompt_file:
            playbook_path = evolution_prompt_file
        else:
            if evolution_run_dir is None:
                raise ValueError("prompt_type=ace requires --evolution_run_dir or --evolution_prompt_file")
            best_path = os.path.join(evolution_run_dir, "best_playbook.txt")
            final_path = os.path.join(evolution_run_dir, "final_playbook.txt")
            if os.path.isfile(best_path):
                playbook_path = best_path
            elif os.path.isfile(final_path):
                playbook_path = final_path
            else:
                raise FileNotFoundError(
                    "ACE playbook not found. Expected one of:\n"
                    f"  {best_path}\n"
                    f"  {final_path}"
                )
        playbook_text = _load_nonempty_text(playbook_path, "ace playbook file")
        system_prompt = render_playbook_to_system_prompt(playbook_text)

        def _builder(src: SourceExample) -> Tuple[str, str]:
            user_prompt = GENERAL_USER_PROMPT.format(conversation=src.get_conversation_text())
            return system_prompt, user_prompt

        return _builder

    raise ValueError(f"Unsupported prompt_type for custom builder: {prompt_type}")


def run_extraction_step(
    dataset_name: str,
    source_file: str,
    target_file: str,
    work_dir: str,
    prompt_type: str,
    extraction_model: str,
    num_sources_per_dataset: Optional[int] = None,
    evolution_run_dir: Optional[str] = None,
    evolution_prompt_file: Optional[str] = None,
    gepa_prompt_index: Optional[int] = None,
    split_train_source_ids: Optional[Set[str]] = None,
    use_async: bool = False,
    async_max_concurrency: int = 8,
) -> str:
    loader = DataLoader(source_file=source_file, target_file=target_file)
    sources_dict = loader.load_sources()
    sources: List[SourceExample] = list(sources_dict.values())

    if split_train_source_ids is not None:
        total_before = len(sources)
        sources = [s for s in sources if s.id not in split_train_source_ids]
        print(
            f"[run_gepa_evaluation] dataset={dataset_name} split-filtered test sources: "
            f"{len(sources)}/{total_before}"
        )

    if num_sources_per_dataset is not None and num_sources_per_dataset > 0:
        sources = sources[:num_sources_per_dataset]
    if not sources:
        raise ValueError(f"No sources loaded from {source_file}")
    if async_max_concurrency < 1:
        raise ValueError("--async_max_concurrency must be >= 1")

    if prompt_type == "no_memory":
        for s in sources:
            s.memory = ""
        out_path = os.path.join(work_dir, "source.jsonl")
        DataLoader().save_sources(sources, out_path)
        print(f"[run_gepa_evaluation] no_memory: wrote {len(sources)} sources (memory='') to {out_path}")
        return out_path

    if prompt_type in BUILTIN_PROMPT_TYPES:
        if not use_async:
            extractor = LLMExtractor(
                prompt_type=prompt_type,
                model_name=extraction_model,
                enable_thinking=True,
            )
            for i, src in enumerate(sources):
                src.memory = extractor.extract(src)
                print(f"[run_gepa_evaluation] extraction {i+1}/{len(sources)} (source_id={src.id})")
        else:

            async def _extract_all_builtin() -> List[str]:
                semaphore = asyncio.Semaphore(async_max_concurrency)
                ordered_memories: List[Optional[str]] = [None] * len(sources)
                done = 0

                async def _run_item(idx: int, src: SourceExample) -> None:
                    nonlocal done
                    extractor = LLMExtractor(
                        prompt_type=prompt_type,
                        model_name=extraction_model,
                        enable_thinking=True,
                    )
                    async with semaphore:
                        memory = await extractor.aextract(src)
                    ordered_memories[idx] = memory
                    done += 1
                    print(f"[run_gepa_evaluation] extraction {done}/{len(sources)} (source_id={src.id})")

                await asyncio.gather(*[_run_item(i, src) for i, src in enumerate(sources)])
                missing: List[str] = []
                for idx, memory in enumerate(ordered_memories):
                    if memory is None:
                        src = sources[idx]
                        missing.append(f"idx={idx},source_id={src.id}")
                if missing:
                    raise RuntimeError(
                        "Async extraction returned None memory entries for "
                        f"dataset={dataset_name}: {', '.join(missing)}"
                    )
                return [str(m) for m in ordered_memories]

            ordered = asyncio.run(_extract_all_builtin())
            for src, memory in zip(sources, ordered):
                src.memory = memory
    else:
        prompt_builder = _build_custom_prompt_builder(
            prompt_type=prompt_type,
            evolution_run_dir=evolution_run_dir,
            evolution_prompt_file=evolution_prompt_file,
            gepa_prompt_index=gepa_prompt_index,
        )
        if not use_async:
            extractor = LLMExtractor(
                prompt_type="general",
                model_name=extraction_model,
                enable_thinking=True,
            )
            for i, src in enumerate(sources):
                system_prompt, user_prompt = prompt_builder(src)
                src.memory = extractor.extract_with_prompts(src, system_prompt, user_prompt)
                print(f"[run_gepa_evaluation] extraction {i+1}/{len(sources)} (source_id={src.id})")
        else:

            async def _extract_all_custom() -> List[str]:
                semaphore = asyncio.Semaphore(async_max_concurrency)
                ordered_memories: List[Optional[str]] = [None] * len(sources)
                done = 0

                async def _run_item(idx: int, src: SourceExample) -> None:
                    nonlocal done
                    extractor = LLMExtractor(
                        prompt_type="general",
                        model_name=extraction_model,
                        enable_thinking=True,
                    )
                    system_prompt, user_prompt = prompt_builder(src)
                    async with semaphore:
                        memory = await extractor.aextract_with_prompts(src, system_prompt, user_prompt)
                    ordered_memories[idx] = memory
                    done += 1
                    print(f"[run_gepa_evaluation] extraction {done}/{len(sources)} (source_id={src.id})")

                await asyncio.gather(*[_run_item(i, src) for i, src in enumerate(sources)])
                missing: List[str] = []
                for idx, memory in enumerate(ordered_memories):
                    if memory is None:
                        src = sources[idx]
                        missing.append(f"idx={idx},source_id={src.id}")
                if missing:
                    raise RuntimeError(
                        "Async extraction returned None memory entries for "
                        f"dataset={dataset_name}: {', '.join(missing)}"
                    )
                return [str(m) for m in ordered_memories]

            ordered = asyncio.run(_extract_all_custom())
            for src, memory in zip(sources, ordered):
                src.memory = memory

    out_path = os.path.join(work_dir, "source.jsonl")
    DataLoader().save_sources(sources, out_path)
    print(f"[run_gepa_evaluation] wrote {len(sources)} sources to {out_path}")
    return out_path


def run_gepa_evaluation(
    datasets: List[str],
    source_files: List[str],
    target_files: List[str],
    output_dir: str,
    model_name: str,
    prompt_type: str,
    extraction_model: Optional[str] = None,
    inference_model: Optional[str] = None,
    num_sources_per_dataset: Optional[int] = None,
    evolution_run_dir: Optional[str] = None,
    evolution_prompt_file: Optional[str] = None,
    gepa_prompt_index: Optional[int] = None,
    split_train_source_ids_by_dataset: Optional[Dict[str, Set[str]]] = None,
    use_async: bool = False,
    async_max_concurrency: int = 8,
    max_targets_per_source: Optional[int] = None,
    run_times: int = 1,
    resume: bool = False,
    dataset_workers: int = 1,
) -> None:
    extraction_model = extraction_model or model_name
    inference_model = inference_model or model_name
    prompt_type_dir = _prompt_type_dir(
        prompt_type=prompt_type,
        evolution_run_dir=evolution_run_dir,
        evolution_prompt_file=evolution_prompt_file,
        gepa_prompt_index=gepa_prompt_index,
    )
    split_train_source_ids_by_dataset = split_train_source_ids_by_dataset or {}

    if run_times < 1:
        raise ValueError("--run_times must be >= 1")
    if dataset_workers < 1:
        raise ValueError("--dataset_workers must be >= 1")

    # Precompute test source counts to decide single-run for large datasets
    test_source_counts: Dict[str, int] = {}
    skipped_datasets: Set[str] = set()
    for dataset_name, source_file in zip(datasets, source_files):
        if not os.path.isfile(source_file):
            print(
                f"\033[93m[run_gepa_evaluation] WARNING: Source file not found for dataset {dataset_name}: "
                f"{source_file} — skipping this dataset.\033[0m"
            )
            skipped_datasets.add(dataset_name)
            continue
        test_source_counts[dataset_name] = _count_test_sources(
            source_file,
            split_train_source_ids_by_dataset.get(dataset_name),
            num_sources_per_dataset,
        )
        print(f"[run_gepa_evaluation] dataset={dataset_name} test_sources={test_source_counts[dataset_name]}")

    # Check target files existence
    for dataset_name, target_file in zip(datasets, target_files):
        if dataset_name in skipped_datasets:
            continue
        if not os.path.isfile(target_file):
            print(
                f"\033[93m[run_gepa_evaluation] WARNING: Target file not found for dataset {dataset_name}: "
                f"{target_file} — skipping this dataset.\033[0m"
            )
            skipped_datasets.add(dataset_name)

    for run_idx in range(run_times):
        run_no = run_idx + 1

        # Build task list for this run, prepare work_dirs
        run_tasks: List[Tuple[str, str, str, str, str, bool]] = []  # (ds, src, tgt, work_dir, src_jsonl, use_all)
        for dataset_name, source_file, target_file in zip(datasets, source_files, target_files):
            if dataset_name in skipped_datasets:
                continue
            single_run_only = (
                test_source_counts.get(dataset_name, 0) >= 200
                or dataset_name.lower() in {"longmemeval"}
            )
            max_3_run = dataset_name.lower() in {"gpqa_diamond"}
            effective_runs = 1 if single_run_only else (min(run_times, 3) if max_3_run else run_times)
            if run_no > effective_runs:
                continue
            work_dir = os.path.join(output_dir, model_name, prompt_type_dir, dataset_name, f"run_{run_no}")
            output_jsonl = os.path.join(work_dir, "source_target_output.jsonl")
            if resume and os.path.exists(output_jsonl):
                print(
                    f"[run_gepa_evaluation][resume] skip completed dataset={dataset_name} "
                    f"run={run_no} output={output_jsonl}"
                )
                continue
            if not resume and os.path.exists(work_dir):
                print(f"[run_gepa_evaluation][overwrite] remove existing work_dir={work_dir}")
                shutil.rmtree(work_dir)
            os.makedirs(work_dir, exist_ok=True)
            source_jsonl = os.path.join(work_dir, "source.jsonl")
            use_all_targets = _use_all_targets(dataset_name)
            run_tasks.append((dataset_name, source_file, target_file, work_dir, source_jsonl, use_all_targets))

        if not run_tasks:
            print(f"[run_gepa_evaluation] run {run_no}: no datasets to process")
            continue

        # ── Phase 1: Sequential extraction (per dataset, async within each) ──
        print(f"[run_gepa_evaluation] run {run_no} Phase 1: extraction for {len(run_tasks)} datasets")
        for dataset_name, source_file, target_file, work_dir, source_jsonl, _ in run_tasks:
            if os.path.exists(source_jsonl):
                print(f"[run_gepa_evaluation][resume] skip extraction for {dataset_name} (source.jsonl exists)")
                continue
            print(f"[run_gepa_evaluation] extracting dataset={dataset_name} work_dir={work_dir}")
            run_extraction_step(
                dataset_name=dataset_name,
                source_file=source_file,
                target_file=target_file,
                work_dir=work_dir,
                prompt_type=prompt_type,
                extraction_model=extraction_model,
                num_sources_per_dataset=num_sources_per_dataset,
                evolution_run_dir=evolution_run_dir,
                evolution_prompt_file=evolution_prompt_file,
                gepa_prompt_index=gepa_prompt_index,
                split_train_source_ids=split_train_source_ids_by_dataset.get(dataset_name),
                use_async=use_async,
                async_max_concurrency=async_max_concurrency,
            )

        # ── Phase 2: Inference + evaluation (parallel across datasets) ──
        def _run_inference_eval(task: Tuple[str, str, str, str, str, bool]) -> None:
            ds_name, _, tgt_file, wd, src_jsonl, use_all = task
            few_shots_num = 1 if ds_name.lower() in {"babyai", "scienceworld"} else 0
            print(f"[run_gepa_evaluation] inference+eval dataset={ds_name} work_dir={wd}")
            run_evaluation_pipeline(
                dataset_name=ds_name,
                model_name=inference_model,
                use_memory=True,
                max_targets_per_source=None if use_all else max_targets_per_source,
                source_file=src_jsonl,
                target_file=tgt_file,
                work_dir=wd,
                engine_kwargs={
                    "few_shots_num": few_shots_num,
                    "enable_thinking": False,
                    "print_adapter_messages": False,
                },
                num_workers=1,
            )
            print(f"[run_gepa_evaluation] done dataset={ds_name}")

        print(
            f"[run_gepa_evaluation] run {run_no} Phase 2: inference+eval for "
            f"{len(run_tasks)} datasets (dataset_workers={dataset_workers})"
        )
        if dataset_workers == 1:
            for task in run_tasks:
                _run_inference_eval(task)
        else:
            with ThreadPoolExecutor(max_workers=dataset_workers) as executor:
                futures = {
                    executor.submit(_run_inference_eval, task): task[0]
                    for task in run_tasks
                }
                for future in as_completed(futures):
                    ds_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(
                            f"\033[91m[run_gepa_evaluation] ERROR: dataset {ds_name} "
                            f"inference+eval failed: {e}\033[0m"
                        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluation pipeline: extraction by prompt source then inference + evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. Optional when --split_file is provided.",
    )
    p.add_argument(
        "--source_files",
        type=str,
        default="",
        help="Comma-separated paths to source_input.jsonl (one per dataset). Optional when --split_file is provided.",
    )
    p.add_argument(
        "--target_files",
        type=str,
        default="",
        help="Comma-separated paths to target.jsonl (one per dataset). Optional when --split_file is provided.",
    )
    p.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=(
            "Optional global split JSON (prepare_data.py format). When provided, "
            "datasets/source_files/target_files are loaded from this file and evaluation "
            "uses each dataset's test split (source ids not in train_source_ids)."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory; outputs go to output_dir/model_name/prompt_type/dataset_name/.",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="openai/Qwen3-32B",
        help="Model name for extraction and inference (used for both unless overridden).",
    )
    p.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=sorted(PROMPT_TYPES),
        help=(
            "Extraction prompt type. Builtin types: "
            + ", ".join(sorted(BUILTIN_PROMPT_TYPES))
            + ". Evolution types: "
            + ", ".join(sorted(EVOLUTION_PROMPT_TYPES))
            + "."
        ),
    )
    p.add_argument(
        "--evolution_run_dir",
        type=str,
        default=None,
        help="Unified evolution run directory. Used by prompt_type=gepa/ace/memevolve.",
    )
    p.add_argument(
        "--evolution_prompt_file",
        type=str,
        default=None,
        help="Optional explicit prompt/playbook file path. If provided, it overrides auto lookup from --evolution_run_dir.",
    )
    p.add_argument(
        "--gepa_prompt_index",
        type=int,
        default=None,
        help="GEPA candidate index (required for prompt_type=gepa).",
    )
    p.add_argument(
        "--extraction_model",
        type=str,
        default=None,
        help="Model for extraction; default same as --model_name.",
    )
    p.add_argument(
        "--inference_model",
        type=str,
        default=None,
        help="Model for inference; default same as --model_name.",
    )
    p.add_argument(
        "--num_sources_per_dataset",
        type=int,
        default=None,
        help="If set, limit number of sources per dataset after split filtering (take first N without shuffling).",
    )
    p.add_argument(
        "--max_targets_per_source",
        type=int,
        default=None,
        help="If set, limit number of target_ids processed per source example.",
    )
    p.add_argument(
        "--use_async",
        action="store_true",
        help="Enable async extraction.",
    )
    p.add_argument(
        "--async_max_concurrency",
        type=int,
        default=8,
        help="Max concurrent extraction requests when --use_async is enabled.",
    )
    p.add_argument(
        "--run_times",
        type=int,
        default=1,
        help="Number of repeated full runs for extraction + inference + evaluation.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume from unfinished dataset/run; otherwise overwrite existing outputs.",
    )
    p.add_argument(
        "--dataset_workers",
        type=int,
        default=4,
        help="Number of datasets to process in parallel for inference+evaluation (Phase 2).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.async_max_concurrency < 1:
        raise ValueError("--async_max_concurrency must be >= 1")
    if args.run_times < 1:
        raise ValueError("--run_times must be >= 1")
    if args.dataset_workers < 1:
        raise ValueError("--dataset_workers must be >= 1")

    if args.prompt_type == "gepa":
        if args.evolution_run_dir is None:
            raise ValueError("prompt_type=gepa requires --evolution_run_dir")
        if args.gepa_prompt_index is None:
            raise ValueError("prompt_type=gepa requires --gepa_prompt_index")
    if args.prompt_type in {"ace", "memevolve"}:
        if args.evolution_prompt_file is None and args.evolution_run_dir is None:
            raise ValueError(
                f"prompt_type={args.prompt_type} requires --evolution_run_dir or --evolution_prompt_file"
            )

    datasets, source_files, target_files, split_train_source_ids_by_dataset = _resolve_datasets_and_files(
        datasets_arg=args.datasets,
        source_files_arg=args.source_files,
        target_files_arg=args.target_files,
        split_file=args.split_file,
    )

    print("[run_gepa_evaluation] datasets:", datasets)
    for ds, sf, tf in zip(datasets, source_files, target_files):
        print(f"  {ds}: source={sf}, target={tf}")
    print(f"[run_gepa_evaluation] prompt_type={args.prompt_type} model_name={args.model_name}")

    run_gepa_evaluation(
        datasets=datasets,
        source_files=source_files,
        target_files=target_files,
        output_dir=args.output_dir,
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        extraction_model=args.extraction_model,
        inference_model=args.inference_model,
        num_sources_per_dataset=args.num_sources_per_dataset,
        evolution_run_dir=args.evolution_run_dir,
        evolution_prompt_file=args.evolution_prompt_file,
        gepa_prompt_index=args.gepa_prompt_index,
        split_train_source_ids_by_dataset=split_train_source_ids_by_dataset,
        use_async=args.use_async,
        async_max_concurrency=args.async_max_concurrency,
        max_targets_per_source=args.max_targets_per_source,
        run_times=args.run_times,
        resume=args.resume,
        dataset_workers=args.dataset_workers,
    )
    print("[run_gepa_evaluation] Done.")


if __name__ == "__main__":
    main()
