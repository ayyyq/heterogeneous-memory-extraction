"""LLM-based clustering by extraction strategy pattern."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import json_repair
import litellm
import yaml
from openai import OpenAI
import os

from src.evolution_clue.schemas import Cluster, ClusterPool, ExampleSummary
from src.evolution_clue.utils import split_thinking_visible, suggest_max_tokens, retry_with_backoff


def _log(msg: str) -> None:
    print(f"[memevolve] {msg}", flush=True)


class ClusterManager:
    """Maintain and update a cluster pool using LLM-based strategy clustering."""

    def __init__(
        self,
        model: str,
        cluster_pool_path: str,
        max_clusters: int = 7,
    ):
        self.model = model
        self.cluster_pool_path = Path(cluster_pool_path)
        self.max_clusters = max_clusters
        if not self._is_gemini_model(model):
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE")),
            )
        else:
            self.client = None

    @staticmethod
    def _is_gemini_model(model_name: str) -> bool:
        return "gemini" in model_name.lower()

    @staticmethod
    def _decoding_kwargs(model_name: str) -> dict[str, Any]:
        if "qwen3" in model_name.lower():
            return {
                "temperature": 0.6,
                "top_p": 0.95,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            }
        elif "gemini" in model_name.lower():
            return {
                "temperature": 1.0,
                "top_p": 0.95,
            }
        return {}

    def _load_pool(self) -> ClusterPool | None:
        if self.cluster_pool_path.exists():
            with open(self.cluster_pool_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_pool(self, pool: ClusterPool) -> None:
        self.cluster_pool_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cluster_pool_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, ensure_ascii=False)

    def _load_template(self) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / "clustering_prompt.yaml"
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("prompt_template", "").strip()

    def _format_batch_summaries(self, summaries: list[ExampleSummary]) -> str:
        lines = []
        for s in summaries:
            reward_label = "SUCCESS" if s["target_reward"] > 0 else "FAILURE"
            lines.append(f"[{s['task_id']}] ({reward_label}) {s['summary']}")
        return "\n".join(lines)

    def _format_existing_pool(self, pool: ClusterPool | None) -> str:
        if pool is None or not pool.get("clusters"):
            return (
                "No existing cluster pool. You are creating clusters from scratch for the first time."
            )
        lines = ["Existing cluster pool (from previous rounds):"]
        for c in pool["clusters"]:
            lines.append(
                f"  - {c['cluster_id']}: \"{c['label']}\" — {c['description']}"
            )
        lines.append(
            "\nYou may reuse, update, merge, or split these clusters based on the new batch."
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_json_block(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def assign_to_clusters(
        self,
        summaries: list[ExampleSummary],
        round_id: int,
        output_json_path: str | None = None,
    ) -> tuple[ClusterPool, dict[str, list[str]]]:
        """Assign examples to clusters. Returns (updated_pool, {cluster_id: [task_ids]})."""
        existing_pool = self._load_pool()
        template = self._load_template()

        prompt = template.format(
            max_clusters=self.max_clusters,
            existing_pool_section=self._format_existing_pool(existing_pool),
            batch_summaries=self._format_batch_summaries(summaries),
        )

        decoding_kwargs = self._decoding_kwargs(self.model)
        max_tokens = 8192
        _log(
            f"cluster_manager start: round={round_id} num_examples={len(summaries)} "
            f"existing_clusters={len(existing_pool['clusters']) if existing_pool else 0}"
        )

        max_attempts = 3
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            # LLM call with context-overflow retry
            retry_used = False
            while True:
                try:
                    def _call_llm():
                        if self._is_gemini_model(self.model):
                            gemini_params = {
                                "model": self.model,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": max_tokens,
                                **decoding_kwargs,
                            }
                            # Only Gemini 3.0+ supports thinkingLevel
                            if "gemini-3" in self.model.lower():
                                gemini_params["thinkingConfig"] = {"thinkingLevel": "MEDIUM", "includeThoughts": True}
                            elif "gemini-2.5" in self.model.lower():
                                gemini_params["thinkingConfig"] = {"thinkingBudget": 6144, "includeThoughts": True}
                            return litellm.completion(**gemini_params)
                        else:
                            return self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_tokens,
                                **decoding_kwargs,
                            )
                    response = retry_with_backoff(_call_llm, logger=_log)
                    break
                except Exception as e:
                    if retry_used:
                        raise
                    suggested = suggest_max_tokens(str(e), max_tokens)
                    if suggested is None:
                        raise
                    retry_used = True
                    _log(
                        f"cluster_manager context overflow, retry: {max_tokens} -> {suggested}"
                    )
                    max_tokens = suggested

            if self._is_gemini_model(self.model):
                msg = response.choices[0]["message"] if isinstance(response.choices[0], dict) else response.choices[0].message
                raw = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
            else:
                raw = response.choices[0].message.content or ""
            _thinking, visible = split_thinking_visible(raw)
            payload = self._extract_json_block(visible)

            # JSON parsing with json_repair fallback
            try:
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    _log("cluster_manager: json.loads failed, falling back to json_repair")
                    parsed = json_repair.loads(payload)
                    if not isinstance(parsed, (list, dict)):
                        raise ValueError(
                            f"ClusterManager: failed to parse JSON. Preview: {visible[:500]}"
                        )

                clusters_data = parsed.get("clusters", parsed) if isinstance(parsed, dict) else parsed
                if not isinstance(clusters_data, list):
                    raise ValueError(
                        f"ClusterManager: expected list of clusters, got {type(clusters_data)}"
                    )
                break  # success
            except (ValueError, Exception) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    _log(
                        f"cluster_manager: attempt {attempt + 1}/{max_attempts} failed ({e}), retrying"
                    )
                else:
                    raise

        clusters: list[Cluster] = []
        assignments: dict[str, list[str]] = {}
        for item in clusters_data:
            cluster_id = str(item.get("cluster_id", f"cluster_{len(clusters)+1:02d}"))
            cluster: Cluster = {
                "cluster_id": cluster_id,
                "label": str(item.get("label", "")),
                "description": str(item.get("description", "")),
                "example_task_ids": [str(tid) for tid in item.get("example_task_ids", [])],
            }
            clusters.append(cluster)
            assignments[cluster_id] = cluster["example_task_ids"]

        pool: ClusterPool = {
            "clusters": clusters,
            "round_updated": round_id,
        }
        self._save_pool(pool)

        if output_json_path:
            out_path = Path(output_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round_id": round_id,
                        "num_examples": len(summaries),
                        "num_clusters": len(clusters),
                        "cluster_pool": pool,
                        "assignments": assignments,
                        "raw_response": raw,
                        "visible_response": visible,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        _log(
            f"cluster_manager end: round={round_id} num_clusters={len(clusters)} "
            f"assignments={{{', '.join(f'{k}: {len(v)}' for k, v in assignments.items())}}}"
        )
        return pool, assignments
