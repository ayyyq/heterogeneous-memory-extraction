"""
Inspect a completed GEPA evolution run: view all candidates, scores, iteration trace,
and dump prompt components for diffing.

GEPA saves under run_dir:
- gepa_state.bin: full state (candidates, parents, val subscores, pareto front, trace, etc.)
- generated_best_outputs_valset/: per val task best outputs (when track_best_outputs=True)
- gepa.stop: created when run_dir is set (used for graceful stop)

Usage:
    python -m src.evolution_gepa.inspect_evolution --run_dir gepa_runs/factual-experiential-0210
    python -m src.evolution_gepa.inspect_evolution --run_dir gepa_runs/factual-experiential-0210 --dump_dir gepa_runs/factual-experiential-0210/candidates
    python -m src.evolution_gepa.inspect_evolution --run_dir gepa_runs/factual-experiential-0210 --diff_parent_child
    python -m src.evolution_gepa.inspect_evolution --run_dir gepa_runs/factual-experiential-0210 --pareto --valset_scores
"""

from __future__ import annotations

import argparse
import os
import difflib
from pathlib import Path

import gepa
from gepa.core.state import GEPAState

from src.evolution_gepa.prepare_data import prepare_gepa_data


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect GEPA evolution run: candidates, scores, trace, and prompt diffs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the GEPA run directory (e.g. gepa_runs/factual-experiential-0210).",
    )
    p.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="If set, write each candidate's components to this dir as candidate_<i>_<name>.txt.",
    )
    p.add_argument(
        "--diff_parent_child",
        action="store_true",
        help="Print component-wise diff between each candidate and its parent.",
    )
    p.add_argument(
        "--diff_seed_best",
        action="store_true",
        help="Print component-wise diff between seed (0) and best candidate.",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="Print full_program_trace (each iteration: selected candidate, subsample scores).",
    )
    p.add_argument(
        "--pareto",
        action="store_true",
        help="Print Pareto front: per val instance best score and which candidate(s) achieve it.",
    )
    p.add_argument(
        "--valset_scores",
        action="store_true",
        help="Print per-candidate per-val-instance scores (which val examples each candidate is good/bad on).",
    )
    p.add_argument(
        "--val_meta",
        type=str,
        default=None,
        help="Optional path to val_meta.json (maps val_id -> source_dataset). Defaults to run_dir/val_meta.json if present.",
    )
    p.add_argument(
        "--best_outputs",
        action="store_true",
        help="Summarize best_outputs_valset if present (in-memory best outputs per val task).",
    )
    p.add_argument(
        "--list_run_dir",
        action="store_true",
        help="List files under run_dir (gepa_state.bin, generated_best_outputs_valset/, etc.).",
    )
    return p.parse_args()


def _text_diff(label: str, old: str, new: str) -> None:
    print(f"\n--- {label} ---")
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    for line in difflib.unified_diff(a, b, fromfile="old", tofile="new", lineterm=""):
        print(line, end="")


def _load_gepa_data_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "gepa_data_config.json")
    assert os.path.exists(path)
    print(f"Loading gepa_data_config.json from {path}")
    # if not os.path.isfile(path):
    #     return {}
    # try:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
    # return data if isinstance(data, dict) else {}
    # except Exception:
    #     return {}


def _reconstruct_val_meta_from_raw(config: dict) -> dict[int, str]:
    try:
        val_size = config.get("val_size")
        if val_size is None and "train_ratio" in config:
            val_size = 1.0 - config["train_ratio"]
        if val_size is None:
            val_size = 0.5

        split_file_path = config.get("split_file_path")
        if split_file_path:
            trainset, valset = prepare_gepa_data(
                datasets=config.get("datasets", []),
                source_file_paths=config.get("source_files", []),
                target_file_paths=config.get("target_files", []),
                num_sources_per_dataset=config.get("num_sources_per_dataset"),
                val_size=val_size,
                seed=config.get("seed", 42),
                target_selection=config.get("target_selection", "random"),
                split_file_path=split_file_path,
                refresh=False,
            )
        else:
            trainset, valset = prepare_gepa_data(
                datasets=config.get("datasets", []),
                source_file_paths=config.get("source_files", []),
                target_file_paths=config.get("target_files", []),
                num_sources_per_dataset=config.get("num_sources_per_dataset"),
                val_size=val_size,
                seed=config.get("seed", 42),
                target_selection=config.get("target_selection", "random"),
            )
    except Exception as e:
        print(f"[inspect_evolution] Warning: failed to reconstruct val_meta.json: {e}")
        return {}

    val_id_to_dataset: dict[int, str] = {}
    for idx, item in enumerate(valset):
        ds = item.get("source_dataset")
        if isinstance(ds, str):
            val_id_to_dataset[idx] = ds
    return val_id_to_dataset


def _load_val_meta(run_dir: str, val_meta_path: str | None) -> dict[int, str]:
    path = val_meta_path or os.path.join(run_dir, "val_meta.json")
    if not os.path.isfile(path):
        config = _load_gepa_data_config(run_dir)
        if not config:
            print("[inspect_evolution] Warning: val_meta.json and gepa_data_config.json not found; per-dataset accuracy disabled.")
            return {}
        print("[inspect_evolution] Warning: val_meta.json not found; reconstructing from raw data (may be slow).")
        return _reconstruct_val_meta_from_raw(config)
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        print("[inspect_evolution] Warning: failed to read val_meta.json; attempting reconstruction from raw data.")
        config = _load_gepa_data_config(run_dir)
        if not config:
            print("[inspect_evolution] Warning: gepa_data_config.json not found; per-dataset accuracy disabled.")
            return {}
        return _reconstruct_val_meta_from_raw(config)

    val_id_to_dataset: dict[int, str] = {}
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                ds = item.get("source_dataset")
                if isinstance(ds, str):
                    val_id_to_dataset[idx] = ds
    elif isinstance(data, dict):
        for key, item in data.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if isinstance(item, dict):
                ds = item.get("source_dataset")
                if isinstance(ds, str):
                    val_id_to_dataset[idx] = ds
    if not val_id_to_dataset:
        config = _load_gepa_data_config(run_dir)
        if config:
            print("[inspect_evolution] Warning: val_meta.json empty; reconstructing from raw data.")
            return _reconstruct_val_meta_from_raw(config)
    return val_id_to_dataset


def _dataset_acc(
    scores_by_val_id: dict[int, float],
    val_id_to_dataset: dict[int, str],
) -> dict[str, tuple[float, int]]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for val_id, score in scores_by_val_id.items():
        ds = val_id_to_dataset.get(val_id)
        if ds is None and isinstance(val_id, str) and val_id.isdigit():
            ds = val_id_to_dataset.get(int(val_id))
        if not ds:
            continue
        totals[ds] = totals.get(ds, 0.0) + score
        counts[ds] = counts.get(ds, 0) + 1
    return {
        ds: (totals[ds] / counts[ds], counts[ds])
        for ds in totals.keys()
        if counts.get(ds, 0) > 0
    }


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir
    state_path = os.path.join(run_dir, "gepa_state.bin")
    if not os.path.isfile(state_path):
        print(f"[inspect_evolution] No state found at {state_path}")
        return

    state: GEPAState = GEPAState.load(run_dir)
    candidates = state.program_candidates
    parents = state.parent_program_for_candidate
    scores = state.program_full_scores_val_set
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    discovery_calls = getattr(state, "num_metric_calls_by_discovery", None)
    predictor_ids = getattr(state, "named_predictor_id_to_update_next_for_program_candidate", None)
    list_of_predictors = getattr(state, "list_of_named_predictors", list(candidates[0].keys()) if candidates else [])
    val_id_to_dataset = _load_val_meta(run_dir, getattr(args, "val_meta", None))

    print("[inspect_evolution] GEPA state loaded.")
    print(f"  run_dir: {run_dir}")
    print(f"  frontier_type: {getattr(state, 'frontier_type', 'instance')}")
    print(f"  total candidates: {len(candidates)}")
    print(f"  best candidate index: {best_idx}")
    print(f"  best validation score: {scores[best_idx]:.4f}")
    print(f"  total_metric_calls: {getattr(state, 'total_num_evals', 'N/A')}")
    print(f"  num_full_ds_evals: {getattr(state, 'num_full_ds_evals', 'N/A')}")
    if val_id_to_dataset:
        best_subscores = state.prog_candidate_val_subscores[best_idx]
        per_ds = _dataset_acc(best_subscores, val_id_to_dataset)
        if per_ds:
            print("\n--- Best candidate per-dataset accuracy ---")
            for ds in sorted(per_ds.keys()):
                avg, n = per_ds[ds]
                print(f"  {ds}: acc={avg:.4f} (n={n})")

    # Summary table (with metric calls at discovery)
    print("\n--- Candidates summary ---")
    header = f"{'idx':<4} {'parent(s)':<12} {'val_score':<10}"
    if discovery_calls is not None:
        header += f" {'metric_calls_at_disc':<18}"
    if val_id_to_dataset:
        header += " per_dataset"
    header += " components"
    print(header)
    for i in range(len(candidates)):
        p = parents[i]
        pstr = str(p) if p else "[]"
        comps = list(candidates[i].keys())
        comp_str = ", ".join(f"{k}({len(candidates[i][k])}ch)" for k in comps)
        row = f"{i:<4} {pstr:<12} {scores[i]:<10.4f}"
        if discovery_calls is not None and i < len(discovery_calls):
            row += f" {discovery_calls[i]:<18}"
        if val_id_to_dataset:
            per_ds = _dataset_acc(state.prog_candidate_val_subscores[i], val_id_to_dataset)
            if per_ds:
                ds_str = ",".join(
                    f"{ds}={per_ds[ds][0]:.4f}" for ds in sorted(per_ds.keys())
                )
            else:
                ds_str = "-"
            row += f" {ds_str}"
        row += f" {comp_str}"
        print(row)
    if predictor_ids is not None and list_of_predictors and len(predictor_ids) == len(candidates):
        print("  (named_predictor_id_to_update_next: next component index to mutate for this candidate)")

    # Objective scores (if multi-objective)
    obj_scores = getattr(state, "prog_candidate_objective_scores", None)
    if obj_scores and any(obj_scores):
        print("\n--- Per-candidate objective aggregates ---")
        objectives = next((list(s.keys()) for s in obj_scores if s), [])
        if objectives:
            print(f"  objectives: {objectives}")
            for i in range(len(candidates)):
                s = obj_scores[i] if i < len(obj_scores) else {}
                print(f"  candidate {i}: {s}")

    # Dump candidate files
    if args.dump_dir:
        dump_path = Path(args.dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        for i, cand in enumerate(candidates):
            for name, text in cand.items():
                fname = dump_path / f"candidate_{i}_{name}.txt"
                fname.write_text(text, encoding="utf-8")
        print(f"\n[inspect_evolution] Dumped {len(candidates)} candidates to {dump_path}")

    # Run dir files
    if args.list_run_dir:
        print("\n--- Run dir contents ---")
        for name in sorted(os.listdir(run_dir)):
            p = os.path.join(run_dir, name)
            if os.path.isfile(p):
                print(f"  {name} (file)")
            else:
                sub = os.listdir(p) if os.path.isdir(p) else []
                print(f"  {name}/ ({len(sub)} items)")
                if name == "generated_best_outputs_valset" and sub:
                    for task in sorted(sub)[:10]:
                        task_path = os.path.join(p, task)
                        if os.path.isdir(task_path):
                            files = os.listdir(task_path)
                            print(f"    {task}/: {files[:5]}{'...' if len(files) > 5 else ''}")
                    if len(sub) > 10:
                        print(f"    ... and {len(sub) - 10} more task_*")

    # Pareto front (per val instance: best score and which candidates achieve it)
    if args.pareto:
        pareto_valset = getattr(state, "pareto_front_valset", None)
        program_at_pareto = getattr(state, "program_at_pareto_front_valset", None)
        if pareto_valset is not None and program_at_pareto is not None:
            print("\n--- Pareto front (valset instance) ---")
            for val_id in sorted(pareto_valset.keys(), key=lambda x: (str(type(x)), str(x))):
                score = pareto_valset[val_id]
                progs = program_at_pareto.get(val_id, set())
                print(f"  val_id={val_id} best_score={score:.4f} candidates={progs}")
        obj_front = getattr(state, "objective_pareto_front", None)
        if obj_front:
            print("\n--- Objective Pareto front ---")
            print(f"  {obj_front}")

    # Per-candidate per-val-instance scores
    if args.valset_scores:
        subscores = getattr(state, "prog_candidate_val_subscores", None)
        if subscores:
            print("\n--- Per-candidate valset subscores ---")
            for i in range(len(candidates)):
                s = subscores[i] if i < len(subscores) else {}
                n = len(s)
                avg = sum(s.values()) / n if n else 0
                line = f"  candidate {i}: n={n} avg={avg:.4f}"
                if val_id_to_dataset:
                    per_ds = _dataset_acc(s, val_id_to_dataset)
                    if per_ds:
                        ds_str = ", ".join(
                            f"{ds}={per_ds[ds][0]:.4f}(n={per_ds[ds][1]})" for ds in sorted(per_ds.keys())
                        )
                        line += f" | per-dataset: {ds_str}"
                line += f" | by val_id: {dict(sorted(s.items())[:15])}{'...' if len(s) > 15 else ''}"
                print(line)

    # Best outputs in state (if track_best_outputs was True)
    if args.best_outputs:
        best_out = getattr(state, "best_outputs_valset", None)
        if best_out:
            print("\n--- best_outputs_valset (in-memory) ---")
            print(f"  {len(best_out)} val tasks; each: list of (program_idx, output).")
            for val_id in sorted(best_out.keys(), key=str)[:3]:
                entries = best_out[val_id]
                prog_idx, out = entries[0] if entries else (None, None)
                keys = list(out.keys()) if isinstance(out, dict) else type(out).__name__
                print(f"  val_id={val_id}: best prog={prog_idx}, output keys={keys}")
        else:
            print("\n--- best_outputs_valset: not present (track_best_outputs was False) ---")

    # Iteration trace
    if args.trace and getattr(state, "full_program_trace", None):
        print("\n--- Iteration trace (full_program_trace) ---")
        print("  Each iter = one round: pick a candidate, sample a train minibatch, (maybe) propose a new prompt, evaluate it.")
        print("  subsample_scores = current candidate on this minibatch; new_subsample_scores = proposed candidate on same minibatch.")
        print("  New candidate is added only when proposed AND accepted → so #iterations >= #candidates (many iters add no new candidate).")
        for entry in state.full_program_trace:
            i = entry.get("i", "?")
            sel = entry.get("selected_program_candidate", "?")
            sub = entry.get("subsample_scores")
            new_sub = entry.get("new_subsample_scores")
            new_idx = entry.get("new_program_idx")
            eval_val = entry.get("evaluated_val_indices")
            line = f"  iter {i}: selected_candidate={sel}"
            if sub is not None:
                line += f", subsample_scores={sub}"
            if new_sub is not None:
                line += f", new_subsample_scores={new_sub}"
            if new_idx is not None:
                line += f"  → new candidate accepted (candidate idx {new_idx})"
                if eval_val is not None:
                    line += f", full valset evaled (n={len(eval_val)} val ids)"
            extra = {k: v for k, v in entry.items() if k not in (
                "i", "selected_program_candidate", "subsample_ids", "subsample_scores",
                "new_subsample_scores", "new_program_idx", "evaluated_val_indices",
            )}
            if extra:
                line += f" | {extra}"
            print(line)

    # Diff: seed vs best
    if args.diff_seed_best and best_idx != 0:
        seed_cand = candidates[0]
        best_cand = candidates[best_idx]
        for name in seed_cand:
            if name in best_cand and seed_cand[name] != best_cand[name]:
                _text_diff(f"component '{name}' (seed 0 vs best {best_idx})", seed_cand[name], best_cand[name])

    # Diff: each candidate vs its parent
    if args.diff_parent_child:
        for i in range(1, len(candidates)):
            p_list = parents[i]
            if not p_list or p_list[0] is None:
                continue
            p_idx = p_list[0]
            parent_cand = candidates[p_idx]
            child_cand = candidates[i]
            print(f"\n--- Candidate {i} (parent {p_idx}) ---")
            for name in child_cand:
                if name not in parent_cand or parent_cand[name] != child_cand[name]:
                    old = parent_cand.get(name, "")
                    new = child_cand[name]
                    _text_diff(f"component '{name}'", old, new)
    
    print('\n')


if __name__ == "__main__":
    main()
