"""
Print evaluation pipeline CLI.

Two modes:
1. single: accept n InferenceOutputExample files, print result for each (re-print for recording).
   Supports GMemory, Dynamic-Cheatsheet, LongMemEval, and MemBench; format is chosen by InferenceOutputExample.source.
2. pair: accept no_memory and with_memory two files, print comparison (by-source acc change, improved/decreased source proportion).
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from typing import List, Optional, Tuple

from src.data_processing.generators.toolbench_generator import TOOLBENCH_DATASETS
from src.data_processing.schemas import InferenceOutputExample
from src.evaluation.factory import (
    _load_outputs,
    print_evaluation_from_files,
    print_evaluation_from_files_readonly,
)


def _key(out: InferenceOutputExample) -> Tuple[str, int]:
    return (out.source, out.id)


def _get_source_id(out: InferenceOutputExample) -> int:
    """Require metadata['source_id']; raise if missing."""
    if "source_id" not in out.metadata:
        raise ValueError(
            f"InferenceOutputExample (source={out.source}, id={out.id}) has no metadata['source_id']. "
            "Only outputs from run_evaluation pipeline are supported for pair mode."
        )
    return out.metadata["source_id"]


def _collect_run_files(folder: str) -> List[Tuple[str, str]]:
    """Return [(run_name, jsonl_path), ...] sorted by run number."""
    import re
    results = []
    try:
        entries = os.listdir(folder)
    except OSError:
        return results
    for entry in entries:
        m = re.fullmatch(r"run_(\d+)", entry)
        if m:
            jsonl_path = os.path.join(folder, entry, "source_target_output.jsonl")
            if os.path.isfile(jsonl_path):
                results.append((int(m.group(1)), entry, jsonl_path))
    results.sort(key=lambda x: x[0])
    return [(name, path) for _, name, path in results]


def _run_acc(jsonl_path: str) -> Tuple[float, int]:
    """Return (accuracy, n) for a jsonl file."""
    outputs = _load_outputs(jsonl_path)
    n = len(outputs)
    if n == 0:
        return 0.0, 0
    acc = sum(1 for o in outputs if o.label) / n
    return acc, n


def _run_sopr(jsonl_path: str) -> Tuple[float, int]:
    """Return (SoPR, n) for a toolbench jsonl file. Uses autoeval_label score if available, else label bool."""
    outputs = _load_outputs(jsonl_path)
    n = len(outputs)
    if n == 0:
        return 0.0, 0
    scores = []
    for o in outputs:
        autoeval = o.metadata.get("autoeval_label", {})
        if autoeval and "score" in autoeval:
            scores.append(autoeval["score"])
        else:
            scores.append(1.0 if o.label else 0.0)
    return sum(scores) / n, n


AIME_DATASETS = {"AIME_2024", "AIME_2025", "AIME_2020_2024"}


def _run_correct_and_total(jsonl_path: str) -> Tuple[int, int]:
    """Return (correct, total) for a jsonl file."""
    outputs = _load_outputs(jsonl_path)
    total = len(outputs)
    correct = sum(1 for o in outputs if o.label)
    return correct, total


def _print_merged_aime(aime_folders: dict, verbose: bool = False) -> None:
    """Print combined AIME results across all three AIME dataset folders."""
    # Collect run files for each AIME dataset
    all_run_files = {}
    for dataset_name, folder in aime_folders.items():
        run_files = _collect_run_files(folder)
        all_run_files[dataset_name] = {name: path for name, path in run_files}

    # Find runs that exist in ALL three datasets
    common_runs = None
    for run_dict in all_run_files.values():
        run_names = set(run_dict.keys())
        common_runs = run_names if common_runs is None else common_runs & run_names
    if not common_runs:
        return

    sorted_runs = sorted(common_runs, key=lambda r: int(r.split("_")[1]))

    # Compute merged acc per run
    accs = []
    for run_name in sorted_runs:
        total_correct = 0
        total_n = 0
        for dataset_name in sorted(aime_folders.keys()):
            jsonl_path = all_run_files[dataset_name][run_name]
            correct, n = _run_correct_and_total(jsonl_path)
            total_correct += correct
            total_n += n
        acc = total_correct / total_n if total_n else 0.0
        accs.append((run_name, acc, total_n))

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Summary (AIME combined): {', '.join(sorted(aime_folders.keys()))}")
    print(sep)
    if verbose:
        for run_name, acc, n in accs:
            print(f"  {run_name}: acc={acc:.4f}  (n={n})")
        print()
    avg = sum(a for _, a, _ in accs) / len(accs)
    print(f"  avg acc:  {avg:.4f}  ({len(accs)} runs)")
    if len(accs) > 1:
        std = math.sqrt(sum((a - avg) ** 2 for _, a, _ in accs) / len(accs))
        print(f"  std:      {std:.4f}")
    print(sep + "\n")


def _handle_folder(folder: str, verbose: bool = False, reeval: bool = False) -> None:
    """Discover run_X subdirs, print each, then print multi-run summary."""
    run_files = _collect_run_files(folder)
    if not run_files:
        print(f"[print_evaluation] No run_*/source_target_output.jsonl found in: {folder}")
        return

    # Filter single-run-only datasets (>= 200 unique source_ids, or specific datasets) to run_1
    dataset_name = os.path.basename(folder)
    run_1_files = [(name, path) for name, path in run_files if name == "run_1"]
    if run_1_files:
        single_run_only = dataset_name.lower() in {"longmemeval"}
        if not single_run_only:
            outputs = _load_outputs(run_1_files[0][1])
            source_ids = {o.metadata.get("source_id") for o in outputs if "source_id" in o.metadata}
            single_run_only = len(source_ids) >= 200
        if single_run_only:
            run_files = run_1_files
        elif dataset_name.lower() in {"gpqa_diamond"}:
            max_run_files = [(name, path) for name, path in run_files if int(name.split("_")[1]) <= 3]
            if max_run_files:
                run_files = max_run_files

    # Re-run evaluation (e.g. bigcodebench code execution) to write labels
    if reeval:
        for run_name, jsonl_path in run_files:
            print_evaluation_from_files(jsonl_path)

    if verbose:
        for run_name, jsonl_path in run_files:
            print(f"\n{'=' * 60}")
            print(f"  {run_name}: {jsonl_path}")
            print(f"{'=' * 60}")
            print_evaluation_from_files_readonly(jsonl_path)

    # Multi-run summary
    is_toolbench = dataset_name.lower() in {d.lower() for d in TOOLBENCH_DATASETS}
    metric_name = "SoPR" if is_toolbench else "acc"
    score_fn = _run_sopr if is_toolbench else _run_acc

    accs = []
    for run_name, jsonl_path in run_files:
        acc, n = score_fn(jsonl_path)
        accs.append((run_name, acc, n))

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Summary: {folder}")
    print(sep)
    for run_name, acc, n in accs:
        print(f"  {run_name}: {metric_name}={acc:.4f}  (n={n})")
    print()
    avg = sum(a for _, a, _ in accs) / len(accs)
    print(f"  avg {metric_name}:  {avg:.4f}  ({len(accs)} runs)")
    if len(accs) > 1:
        std = math.sqrt(sum((a - avg) ** 2 for _, a, _ in accs) / len(accs))
        print(f"  std:      {std:.4f}")
    print(sep + "\n")


def run_single_mode(paths: List[str], verbose: bool = False, reeval: bool = False) -> None:
    """Print evaluation result for each given InferenceOutputExample file or directory (GMemory / Dynamic-Cheatsheet / LongMemEval by source)."""
    if not paths:
        print("[print_evaluation] No files given for single mode.")
        return
    for path in paths:
        if os.path.isdir(path):
            _handle_folder(path, verbose=verbose, reeval=reeval)
        else:
            if reeval:
                print_evaluation_from_files(path)
            else:
                print_evaluation_from_files_readonly(path)

    # Merge AIME datasets if all three are present
    aime_folders = {}
    for path in paths:
        if os.path.isdir(path):
            dataset_name = os.path.basename(path)
            if dataset_name in AIME_DATASETS:
                aime_folders[dataset_name] = path
    if len(aime_folders) == len(AIME_DATASETS):
        _print_merged_aime(aime_folders, verbose=verbose)


def run_pair_mode(
    no_memory_file: str,
    with_memory_file: str,
) -> None:
    """
    Compare no_memory vs with_memory InferenceOutputExample files (from run_evaluation).
    Requires: (1) every example has metadata['source_id']; (2) the two files are one-to-one
    by position: same length, and for each index i, no_mem[i].id == with_mem[i].id and
    no_mem[i].metadata['source_id'] == with_mem[i].metadata['source_id'].
    Reports: total source_ids, total examples, overall acc change, average acc-by-source change,
    and improved/decreased/unchanged source counts. Does not print per-source_id rows.
    """
    if not os.path.isfile(no_memory_file):
        print(f"[print_evaluation] Not a file: {no_memory_file}")
        return
    if not os.path.isfile(with_memory_file):
        print(f"[print_evaluation] Not a file: {with_memory_file}")
        return

    try:
        no_mem = _load_outputs(no_memory_file)
        with_mem = _load_outputs(with_memory_file)
    except Exception as e:
        print(f"[print_evaluation] Failed to load: {e}")
        return

    if len(no_mem) != len(with_mem):
        print(
            f"[print_evaluation] no_memory and with_memory must have the same length (one-to-one). "
            f"Got {len(no_mem)} vs {len(with_mem)}."
        )
        return

    try:
        for i, (o_no, o_with) in enumerate(zip(no_mem, with_mem)):
            _get_source_id(o_no)
            _get_source_id(o_with)
            if o_no.id != o_with.id:
                print(f"[print_evaluation] At index {i}: id mismatch (no_mem={o_no.id}, with_mem={o_with.id}). Files must be one-to-one with same id.")
                return
            if o_no.metadata["source_id"] != o_with.metadata["source_id"]:
                print(f"[print_evaluation] At index {i}: source_id mismatch (no_mem={o_no.metadata['source_id']}, with_mem={o_with.metadata['source_id']}). Files must be one-to-one with same source_id.")
                return
    except ValueError as e:
        print(f"[print_evaluation] {e}")
        return

    # Pair by index: (no_mem[i], with_mem[i]) have same id and source_id
    by_source: dict[int, List[Tuple[bool, bool]]] = defaultdict(list)
    for o_no, o_with in zip(no_mem, with_mem):
        sid = o_no.metadata["source_id"]
        by_source[sid].append((bool(o_no.label), bool(o_with.label)))

    num_sources = len(by_source)
    num_examples = len(no_mem)

    # Per-source acc and change
    changes: List[float] = []
    improved = 0
    decreased = 0
    unchanged = 0
    for sid, pairs in by_source.items():
        n = len(pairs)
        acc_no = sum(1 for a, _ in pairs if a) / n if n else 0.0
        acc_with = sum(1 for _, b in pairs if b) / n if n else 0.0
        ch = acc_with - acc_no
        changes.append(ch)
        if ch > 0:
            improved += 1
        elif ch < 0:
            decreased += 1
        else:
            unchanged += 1

    avg_change_by_source = sum(changes) / len(changes) if changes else 0.0
    acc_no_overall = sum(1 for o in no_mem if o.label) / num_examples
    acc_with_overall = sum(1 for o in with_mem if o.label) / num_examples

    sep = "=" * 60
    print("\n" + sep)
    print("Pair comparison (no_memory vs with_memory)")
    print(sep)
    print(f"  no_memory file:   {no_memory_file}")
    print(f"  with_memory file: {with_memory_file}")
    print()
    print(f"  Total source_ids:     {num_sources}")
    print(f"  Total examples:       {num_examples}")
    print()
    print(f"  Overall acc (no_mem): {acc_no_overall:.4f}")
    print(f"  Overall acc (memory): {acc_with_overall:.4f}")
    print(f"  Overall change:       {acc_with_overall - acc_no_overall:+.4f}")
    print()
    print(f"  Average acc-by-source change: {avg_change_by_source:+.4f}")
    print()
    print("  By-source (improved / decreased / unchanged):")
    print(f"    improved:  {improved}/{num_sources} ({100.0 * improved / num_sources:.1f}%)")
    print(f"    decreased: {decreased}/{num_sources} ({100.0 * decreased / num_sources:.1f}%)")
    print(f"    unchanged: {unchanged}/{num_sources} ({100.0 * unchanged / num_sources:.1f}%)")
    print(sep + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Print evaluation results: single (re-print n files) or pair (compare no_memory vs with_memory)."
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["single", "pair"],
        help="single: print each file's result; pair: compare two files (no_memory vs with_memory).",
    )
    p.add_argument(
        "files",
        nargs="*",
        help="For single: one or more InferenceOutputExample jsonl paths. For pair: exactly two paths (no_memory, with_memory).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-run results in addition to summary.",
    )
    p.add_argument(
        "--reeval",
        action="store_true",
        help="Re-run evaluation for datasets that require code execution (e.g. bigcodebench). Without this flag, only existing labels are read.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.mode == "single":
        if not args.files:
            print("Usage: python -m src.pipelines.print_evaluation --mode single <file1.jsonl> [file2.jsonl ...]")
            return
        run_single_mode(args.files, verbose=args.verbose, reeval=args.reeval)
        return

    if args.mode == "pair":
        if len(args.files) != 2:
            print("Usage: python -m src.pipelines.print_evaluation --mode pair <no_memory.jsonl> <with_memory.jsonl>")
            return
        run_pair_mode(no_memory_file=args.files[0], with_memory_file=args.files[1])


if __name__ == "__main__":
    main()
