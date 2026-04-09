"""MemEvolve-style tournament orchestration for prompt-only extraction evolution."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evolution_gepa.prepare_data import prepare_gepa_data
from src.evolution_clue.core.cluster_manager import ClusterManager
from src.evolution_clue.core.example_summarizer import ExampleSummarizer
from src.evolution_clue.core.prompt_analyzer import PromptAnalyzer
from src.evolution_clue.core.prompt_generator import PromptGenerator
from src.evolution_clue.eval import evaluate_prompt_on_batch
from src.evolution_clue.schemas import EvalSummary


class AutoPromptEvolver:
    """Tournament evolution over system_prompt candidates."""

    def __init__(
        self,
        split_file: str,
        work_dir: str,
        base_system_prompt: str,
        analysis_model: str,
        generation_model: str,
        extraction_model: str,
        inference_model: str,
        num_systems: int = 3,
        task_batch_x: int = 20,
        top_t: int = 2,
        extra_sample_y: int = 5,
        seed: int = 42,
        max_workers: int = 1,
        extraction_use_async: bool = False,
        extraction_num_workers: int = 1,
        extraction_async_max_concurrency: int = 8,
        use_pareto_selection: bool = False,
        resume: bool = True,
        refresh_split: bool = False,
        clustering_model: str | None = None,
        max_clusters: int = 7,
    ):
        self.split_file = split_file
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_model = analysis_model
        self.generation_model = generation_model
        self.extraction_model = extraction_model
        self.inference_model = inference_model
        self.clustering_model = clustering_model or analysis_model
        self.num_systems = int(num_systems)
        self.task_batch_x = int(task_batch_x)
        self.top_t = int(top_t)
        self.extra_sample_y = int(extra_sample_y)
        self.max_clusters = int(max_clusters)
        self.seed = int(seed)
        self.max_workers = max(1, int(max_workers))
        self.extraction_use_async = bool(extraction_use_async)
        self.extraction_num_workers = max(1, int(extraction_num_workers))
        self.extraction_async_max_concurrency = max(1, int(extraction_async_max_concurrency))
        self.use_pareto_selection = bool(use_pareto_selection)
        self.resume = bool(resume)
        self.refresh_split = bool(refresh_split)
        self.random = random.Random(self.seed)

        self.analyzer = PromptAnalyzer(analysis_model=self.analysis_model, max_steps=20)
        self.generator = PromptGenerator(generation_model=self.generation_model)
        self.summarizer = ExampleSummarizer(model=self.analysis_model)
        self.cluster_manager = ClusterManager(
            model=self.clustering_model,
            cluster_pool_path=str(self.work_dir / "cluster_pool.json"),
            max_clusters=self.max_clusters,
        )
        self.pool = self._prepare_pool()
        self.state_file = self.work_dir / "evolve_state.json"
        self.state = self._load_state(base_system_prompt)

    def _log(self, msg: str) -> None:
        print(f"[memevolve] {msg}", flush=True)

    def _prepare_pool(self) -> list[dict[str, Any]]:
        trainset, _ = prepare_gepa_data(
            datasets=[],
            source_file_paths=[],
            target_file_paths=[],
            seed=self.seed,
            split_file_path=self.split_file,
            refresh=self.refresh_split,
            val_size=0.0,
        )
        if not trainset:
            raise ValueError("No train examples produced by prepare_gepa_data.")
        return trainset

    def _load_state(self, base_system_prompt: str) -> dict[str, Any]:
        if self.state_file.exists() and self.resume:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "round": 0,
            "cursor": 0,
            "base_prompt_id": "base_seed",
            "base_system_prompt": base_system_prompt,
            "history": [],
        }

    def _save_state(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _checkpoint_path(self, round_id: int) -> Path:
        return self.work_dir / f"round_{round_id:02d}" / "checkpoint.json"

    def _load_checkpoint(self, round_id: int) -> dict[str, Any] | None:
        p = self._checkpoint_path(round_id)
        if self.resume and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_checkpoint(self, round_id: int, data: dict[str, Any]) -> None:
        p = self._checkpoint_path(round_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _select_indices(self, start: int, count: int) -> list[int]:
        n = len(self.pool)
        return [((start + i) % n) for i in range(count)]

    def _evaluate_many(
        self,
        prompts: dict[str, str],
        indices: list[int],
        out_dir: Path,
    ) -> dict[str, EvalSummary]:
        out_dir.mkdir(parents=True, exist_ok=True)
        batch = [self.pool[i] for i in indices]
        results: dict[str, EvalSummary] = {}
        total_prompts = len(prompts)
        examples_per_prompt = len(indices)
        total_examples = total_prompts * examples_per_prompt
        self._log(
            f"evaluate_many start: prompts={total_prompts} samples={examples_per_prompt} "
            f"total_example_evals={total_examples} out_dir={out_dir}"
        )
        self._log(
            "evaluate_many progress legend: "
            "'overall' = all prompt-example evaluations, "
            "'[Adapter][extraction/inference]' = per-prompt stage progress"
        )
        self._log(
            "evaluate_many mode: prompt_serial "
            f"extraction_use_async={self.extraction_use_async} "
            f"extraction_num_workers={self.extraction_num_workers} "
            f"extraction_async_max_concurrency={self.extraction_async_max_concurrency}"
        )

        def _print_progress(done: int, total_count: int) -> None:
            if total_count <= 0:
                return
            width = 30
            ratio = done / total_count
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\r[memevolve] overall: [{bar}] {done}/{total_count} ({ratio * 100:5.1f}%)",
                end="",
                flush=True,
            )
            if done >= total_count:
                print("", flush=True)

        def _run_one(prompt_id: str, system_prompt: str) -> EvalSummary:
            return evaluate_prompt_on_batch(
                batch=batch,
                prompt_id=prompt_id,
                system_prompt=system_prompt,
                extraction_model=self.extraction_model,
                inference_model=self.inference_model,
                logs_dir=str(out_dir / prompt_id),
                num_workers=self.extraction_num_workers,
                use_async=self.extraction_use_async,
                async_max_concurrency=self.extraction_async_max_concurrency,
                progress_callback=True,
            )

        _print_progress(0, total_examples)
        for done, (pid, prompt_text) in enumerate(prompts.items(), start=1):
            summary = _run_one(pid, prompt_text)
            results[pid] = summary
            done_examples = done * examples_per_prompt
            _print_progress(done_examples, total_examples)
            self._log(
                "evaluate_many done: "
                f"prompt_id={pid} accuracy={float(summary.get('accuracy', 0.0)):.6f}"
            )
        self._log(f"evaluate_many end: completed={len(results)} out_dir={out_dir}")
        return results

    def _rank_prompts(self, eval_results: dict[str, EvalSummary]) -> list[str]:
        rows = []
        for pid, res in eval_results.items():
            acc = float(res.get("accuracy", 0.0))
            tok = res.get("avg_total_tokens")
            t = res.get("avg_elapsed_time")
            rows.append(
                {
                    "prompt_id": pid,
                    "accuracy": acc,
                    "avg_total_tokens": float(tok) if tok is not None else float("inf"),
                    "avg_elapsed_time": float(t) if t is not None else float("inf"),
                }
            )
        rows.sort(
            key=lambda x: (
                -x["accuracy"],
                x["avg_total_tokens"],
                x["avg_elapsed_time"],
                x["prompt_id"],
            )
        )
        return [r["prompt_id"] for r in rows]

    def _choose_top(
        self, eval_results: dict[str, EvalSummary], k: int, use_pareto: bool
    ) -> list[str]:
        if not use_pareto:
            return self._rank_prompts(eval_results)[:k]
        # Lightweight pareto: rank by non-domination, then fallback scalarized score.
        rows = []
        for pid, res in eval_results.items():
            rows.append(
                {
                    "prompt_id": pid,
                    "accuracy": float(res.get("accuracy", 0.0)),
                    "tokens": float(res.get("avg_total_tokens")) if res.get("avg_total_tokens") is not None else float("inf"),
                    "time": float(res.get("avg_elapsed_time")) if res.get("avg_elapsed_time") is not None else float("inf"),
                }
            )

        def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
            not_worse = a["accuracy"] >= b["accuracy"] and a["tokens"] <= b["tokens"] and a["time"] <= b["time"]
            better = a["accuracy"] > b["accuracy"] or a["tokens"] < b["tokens"] or a["time"] < b["time"]
            return bool(not_worse and better)

        remaining = rows[:]
        rank_by_id: dict[str, int] = {}
        rank = 1
        while remaining:
            front = []
            for r in remaining:
                if not any(dominates(other, r) for other in remaining if other is not r):
                    front.append(r)
            for r in front:
                rank_by_id[r["prompt_id"]] = rank
                remaining.remove(r)
            rank += 1

        tokens = [r["tokens"] for r in rows if r["tokens"] != float("inf")]
        times = [r["time"] for r in rows if r["time"] != float("inf")]
        min_tok, max_tok = (min(tokens), max(tokens)) if tokens else (0.0, 1.0)
        min_t, max_t = (min(times), max(times)) if times else (0.0, 1.0)

        ranked = []
        for r in rows:
            tok_score = 1.0 if max_tok == min_tok else (max_tok - r["tokens"]) / (max_tok - min_tok)
            time_score = 1.0 if max_t == min_t else (max_t - r["time"]) / (max_t - min_t)
            scalar = 0.6 * r["accuracy"] + 0.25 * tok_score + 0.15 * time_score
            ranked.append((rank_by_id.get(r["prompt_id"], 999), -scalar, r["prompt_id"]))
        ranked.sort()
        return [x[2] for x in ranked[:k]]

    def run(self, num_rounds: int) -> dict[str, Any]:
        start_round = int(self.state.get("round", 0))
        history: list[dict[str, Any]] = []
        total_rounds = int(num_rounds)
        self._log(
            "run start: "
            f"num_rounds={total_rounds} start_round={start_round} "
            f"num_systems={self.num_systems} task_batch_x={self.task_batch_x} "
            f"top_t={self.top_t} extra_sample_y={self.extra_sample_y} max_workers={self.max_workers} "
            f"extraction_use_async={self.extraction_use_async} "
            f"extraction_num_workers={self.extraction_num_workers} "
            f"extraction_async_max_concurrency={self.extraction_async_max_concurrency} "
            f"pool_size={len(self.pool)} work_dir={self.work_dir}"
        )

        if start_round >= total_rounds:
            self._log(
                f"run skip: start_round={start_round} >= num_rounds={total_rounds}, "
                "all rounds already completed"
            )

        for round_id in range(start_round, total_rounds):
            round_started_at = time.perf_counter()
            round_dir = self.work_dir / f"round_{round_id:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = self._load_checkpoint(round_id) or {"step_completed": 0}

            cursor = int(self.state["cursor"])
            base_prompt_id = str(self.state["base_prompt_id"])
            base_system_prompt = str(self.state["base_system_prompt"])
            self._log(
                f"round {round_id} start: cursor={cursor} base_prompt_id={base_prompt_id} "
                f"resume_step={checkpoint.get('step_completed', 0)}"
            )

            # Step 1: base batch
            if checkpoint["step_completed"] < 1:
                batch_indices = self._select_indices(cursor, self.task_batch_x)
                self._log(
                    f"round {round_id} step1 start: batch_size={len(batch_indices)} "
                    f"indices_head={batch_indices[:3]} indices_tail={batch_indices[-3:]}"
                )
                base_eval = self._evaluate_many(
                    prompts={base_prompt_id: base_system_prompt},
                    indices=batch_indices,
                    out_dir=round_dir / "base_logs",
                )[base_prompt_id]
                checkpoint["batch_indices"] = batch_indices
                checkpoint["base_eval"] = base_eval
                checkpoint["step_completed"] = 1
                self._save_checkpoint(round_id, checkpoint)
                self._log(
                    f"round {round_id} step1 end: base_accuracy={float(base_eval.get('accuracy', 0.0)):.6f} "
                    f"logs_dir={round_dir / 'base_logs' / base_prompt_id}"
                )
            else:
                batch_indices = checkpoint["batch_indices"]
                base_eval = checkpoint["base_eval"]
                self._log(
                    f"round {round_id} step1 skip: loaded checkpoint with batch_size={len(batch_indices)}"
                )

            # Step 2: cluster-based analyze + generate
            # Sub-steps: 2a summarize, 2b cluster, 2c per-cluster analysis (once), 2d cross-cluster synthesis (num_systems times)
            if checkpoint["step_completed"] < 2:
                step2_logs_dir = str(round_dir / "base_logs" / base_prompt_id)

                # Step 2a: Summarize
                if not checkpoint.get("step2a_completed"):
                    self._log(f"round {round_id} step2a start: summarize batch")
                    summaries = self.summarizer.summarize_batch(step2_logs_dir)
                    checkpoint["summaries"] = summaries
                    checkpoint["step2a_completed"] = True
                    self._save_checkpoint(round_id, checkpoint)
                    self._log(
                        f"round {round_id} step2a end: {len(summaries)} summaries"
                    )
                else:
                    summaries = checkpoint["summaries"]
                    self._log(
                        f"round {round_id} step2a skip: loaded {len(summaries)} summaries"
                    )

                # Step 2b: Cluster
                if not checkpoint.get("step2b_completed"):
                    self._log(f"round {round_id} step2b start: cluster examples")
                    cluster_pool, cluster_assignments = self.cluster_manager.assign_to_clusters(
                        summaries=summaries,
                        round_id=round_id,
                        output_json_path=str(round_dir / "clustering_result.json"),
                    )
                    checkpoint["cluster_pool"] = cluster_pool
                    checkpoint["cluster_assignments"] = cluster_assignments
                    checkpoint["step2b_completed"] = True
                    self._save_checkpoint(round_id, checkpoint)
                    self._log(
                        f"round {round_id} step2b end: {len(cluster_pool['clusters'])} clusters"
                    )
                else:
                    cluster_pool = checkpoint["cluster_pool"]
                    cluster_assignments = checkpoint["cluster_assignments"]
                    self._log(
                        f"round {round_id} step2b skip: loaded {len(cluster_pool['clusters'])} clusters"
                    )

                # Build cluster lookup for quick access
                cluster_lookup = {c["cluster_id"]: c for c in cluster_pool["clusters"]}

                # Step 2c: Per-cluster analysis (run once, reused for all candidate generations)
                if not checkpoint.get("step2c_completed"):
                    self._log(
                        f"round {round_id} step2c start: "
                        f"analyzing {len(cluster_assignments)} clusters"
                    )
                    cluster_analyses = []
                    for cluster_id, task_ids in cluster_assignments.items():
                        cluster_info = cluster_lookup.get(cluster_id, {})
                        analysis = self.analyzer.analyze_cluster(
                            logs_dir=step2_logs_dir,
                            base_prompt=base_system_prompt,
                            round_id=round_id,
                            cluster_id=cluster_id,
                            cluster_label=cluster_info.get("label", cluster_id),
                            cluster_description=cluster_info.get("description", ""),
                            task_ids=task_ids,
                            output_json_path=str(
                                round_dir / f"cluster_analysis_{cluster_id}.json"
                            ),
                        )
                        cluster_analyses.append(analysis)
                    checkpoint["cluster_analyses"] = cluster_analyses
                    checkpoint["step2c_completed"] = True
                    self._save_checkpoint(round_id, checkpoint)
                    self._log(
                        f"round {round_id} step2c end: "
                        f"{len(cluster_analyses)} cluster analyses"
                    )
                else:
                    cluster_analyses = checkpoint["cluster_analyses"]
                    self._log(
                        f"round {round_id} step2c skip: loaded {len(cluster_analyses)} cluster analyses"
                    )

                # Step 2d: Cross-cluster synthesis (run num_systems times, reusing cluster_analyses)
                candidates_so_far: list = checkpoint.get("candidates_so_far", [])

                for i in range(len(candidates_so_far), self.num_systems):
                    self._log(
                        f"round {round_id} step2d iter {i + 1}/{self.num_systems} start: "
                        f"cross-cluster synthesis"
                    )
                    try:
                        one_candidates = self.generator.generate_from_clusters(
                            cluster_analyses=cluster_analyses,
                            base_system_prompt=base_system_prompt,
                            parent_prompt_id=base_prompt_id,
                            output_json_path=str(
                                round_dir / "candidates" / f"candidate_{i:02d}.json"
                            ),
                        )
                    except Exception as e:
                        self._log(
                            f"round {round_id} step2d iter {i + 1}/{self.num_systems} FAILED: "
                            f"{e}, skipping this candidate"
                        )
                        continue
                    for c in one_candidates:
                        c["candidate_id"] = f"cand_r{round_id:02d}_{i + 1:02d}"
                    candidates_so_far.extend(one_candidates)
                    checkpoint["candidates_so_far"] = candidates_so_far
                    self._save_checkpoint(round_id, checkpoint)
                    self._log(
                        f"round {round_id} step2d iter {i + 1}/{self.num_systems} done: "
                        f"got {len(one_candidates)} candidate(s) total_so_far={len(candidates_so_far)}"
                    )

                if not candidates_so_far:
                    raise RuntimeError(
                        f"round {round_id} step2d: all {self.num_systems} candidate generations failed"
                    )

                candidates = candidates_so_far
                checkpoint["candidates"] = candidates
                checkpoint["step_completed"] = 2
                self._save_checkpoint(round_id, checkpoint)
                self._log(
                    f"round {round_id} step2 end: num_valid_candidates={len(candidates)}"
                )
            else:
                candidates = checkpoint.get("candidates", [])
                self._log(
                    f"round {round_id} step2 skip: loaded checkpoint with num_candidates={len(candidates)}"
                )

            # Step 3: tournament
            if checkpoint["step_completed"] < 3:
                prompt_map = {base_prompt_id: base_system_prompt}
                for c in candidates:
                    candidate_id = str(c["candidate_id"])
                    system_prompt = str(c["system_prompt"])
                    prompt_map[candidate_id] = system_prompt
                self._log(
                    f"round {round_id} step3 start: num_prompts={len(prompt_map)} batch_size={len(batch_indices)}"
                )
                tournament = self._evaluate_many(
                    prompts=prompt_map,
                    indices=batch_indices,
                    out_dir=round_dir / "eval_tournament",
                )
                finalists = self._choose_top(
                    eval_results=tournament,
                    k=self.top_t,
                    use_pareto=self.use_pareto_selection,
                )
                checkpoint["tournament"] = tournament
                checkpoint["finalists"] = finalists
                checkpoint["step_completed"] = 3
                self._save_checkpoint(round_id, checkpoint)
                self._log(
                    f"round {round_id} step3 end: finalists={finalists}"
                )
            else:
                finalists = checkpoint["finalists"]
                self._log(
                    f"round {round_id} step3 skip: finalists={finalists}"
                )

            # Step 4: finals
            if checkpoint["step_completed"] < 4:
                sampled = self.random.sample(
                    batch_indices, min(self.extra_sample_y, len(batch_indices))
                )
                new_indices = self._select_indices(cursor + self.task_batch_x, self.task_batch_x)
                finalist_indices = sampled + new_indices
                self._log(
                    f"round {round_id} step4 start: sampled_count={len(sampled)} "
                    f"new_count={len(new_indices)} finalist_indices_count={len(finalist_indices)}"
                )

                prompt_lookup = {base_prompt_id: base_system_prompt}
                for c in candidates:
                    prompt_lookup[str(c["candidate_id"])] = str(c["system_prompt"])
                finalist_prompts = {pid: prompt_lookup[pid] for pid in finalists}
                finals = self._evaluate_many(
                    prompts=finalist_prompts,
                    indices=finalist_indices,
                    out_dir=round_dir / "eval_finals",
                )
                winner = self._choose_top(
                    eval_results=finals,
                    k=1,
                    use_pareto=self.use_pareto_selection,
                )[0]
                checkpoint["finalist_indices"] = finalist_indices
                checkpoint["finals"] = finals
                checkpoint["winner"] = winner
                checkpoint["step_completed"] = 4
                self._save_checkpoint(round_id, checkpoint)
                winner_acc = float(finals.get(winner, {}).get("accuracy", 0.0))
                self._log(
                    f"round {round_id} step4 end: winner={winner} winner_accuracy={winner_acc:.6f}"
                )
            else:
                winner = checkpoint["winner"]
                finalist_indices = checkpoint["finalist_indices"]
                finals = checkpoint["finals"]
                winner_acc = float(finals.get(winner, {}).get("accuracy", 0.0))
                self._log(
                    f"round {round_id} step4 skip: winner={winner} winner_accuracy={winner_acc:.6f}"
                )

            # Step 5: state update
            self._log(f"round {round_id} step5 start: persist state and artifacts")
            prompt_lookup = {base_prompt_id: base_system_prompt}
            for c in candidates:
                prompt_lookup[str(c["candidate_id"])] = str(c["system_prompt"])
            winner_prompt = prompt_lookup[winner]
            next_cursor = cursor + (2 * self.task_batch_x)
            round_summary = {
                "round": round_id,
                "timestamp": datetime.now().isoformat(),
                "base_prompt_id": base_prompt_id,
                "batch_indices": batch_indices,
                "finalist_indices": finalist_indices,
                "winner": winner,
                "base_eval": base_eval,
                "tournament": checkpoint.get("tournament", {}),
                "finals": finals,
                "num_candidates": len(candidates),
            }
            with open(round_dir / "round_summary.json", "w", encoding="utf-8") as f:
                json.dump(round_summary, f, indent=2, ensure_ascii=False)
            history.append(round_summary)

            self.state["round"] = round_id + 1
            self.state["cursor"] = next_cursor
            self.state["base_prompt_id"] = winner
            self.state["base_system_prompt"] = winner_prompt
            self.state.setdefault("history", []).append(
                {"round": round_id, "winner": winner, "cursor": next_cursor}
            )
            self._save_state()
            with open(self.work_dir / "best_prompt.txt", "w", encoding="utf-8") as f:
                f.write(winner_prompt)
            self._log(
                f"round {round_id} step5 end: next_cursor={next_cursor} best_prompt_path={self.work_dir / 'best_prompt.txt'}"
            )
            round_elapsed = time.perf_counter() - round_started_at
            self._log(
                f"round {round_id} completed: winner={winner} elapsed_sec={round_elapsed:.2f}"
            )

        with open(self.work_dir / "auto_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        self._log(
            f"run completed: rounds={len(history)} history_path={self.work_dir / 'auto_history.json'} "
            f"best_prompt_path={self.work_dir / 'best_prompt.txt'}"
        )
        return {
            "rounds": len(history),
            "history_path": str(self.work_dir / "auto_history.json"),
            "best_prompt_path": str(self.work_dir / "best_prompt.txt"),
        }
