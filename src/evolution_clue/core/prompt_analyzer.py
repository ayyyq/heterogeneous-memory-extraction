"""Analysis phase for extraction prompt evolution."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from FlashOAgents import OpenAIServerModel, ToolCallingAgent
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    flash_searcher_root = repo_root / "MemEvolve" / "Flash-Searcher-main"
    if flash_searcher_root.is_dir():
        sys.path.insert(0, str(flash_searcher_root))
        from FlashOAgents import OpenAIServerModel, ToolCallingAgent
    else:
        raise

from src.evolution_clue.tools import (
    CaseDetailsTool,
    ExtractionViewerTool,
)
from src.evolution_clue.utils import split_thinking_visible, suggest_max_tokens, retry_with_backoff
from src.litellm_model import LiteLLMModel


def _log(msg: str) -> None:
    print(f"[memevolve] {msg}", flush=True)


class PromptAnalyzer:
    """Run AnalysisAgent-like tool-calling analysis on extraction logs."""

    def __init__(self, analysis_model: str, max_steps: int = 20):
        self.analysis_model = analysis_model
        self.max_steps = max_steps
        self.model_kwargs = self._decoding_kwargs(analysis_model)
        self.model = self._build_model()

    def _build_model(self, override_max_tokens: int | None = None):
        model_kwargs = dict(self.model_kwargs)
        if "gemini-2.5" in self.analysis_model.lower():
            model_kwargs["max_tokens"] = 20000
        if override_max_tokens is not None:
            model_kwargs["max_tokens"] = int(override_max_tokens)
        if "gemini" in self.analysis_model.lower():
            extra = {}
            if "gemini-3" in self.analysis_model.lower():
                extra["thinking_level"] = "MEDIUM"
            elif "gemini-2.5" in self.analysis_model.lower():
                extra["thinking_budget"] = -1
            return LiteLLMModel(
                model_id=self.analysis_model,
                **extra,
                **model_kwargs,
            )
        return OpenAIServerModel(
            model_id=self.analysis_model,
            api_base=os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE")),
            api_key=os.getenv("OPENAI_API_KEY"),
            **model_kwargs,
        )

    @staticmethod
    def _decoding_kwargs(model_name: str) -> dict[str, Any]:
        if "qwen3" in model_name.lower():
            return {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 8192,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            }
        elif "gemini" in model_name.lower():
            return {
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 8192,
            }
        return {}

    def _load_template(self) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / "analysis_prompt.yaml"
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("prompt_template", "").strip()

    def _build_prompt(
        self,
        logs_dir: str,
        base_prompt: str,
        round_id: int,
        num_samples: int,
    ) -> str:
        template = self._load_template()
        return template.format(
            logs_dir=logs_dir,
            base_prompt=base_prompt,
            round_id=round_id,
            num_samples=num_samples,
        )

    @staticmethod
    def _validate_logs_dir(logs_dir: str) -> int:
        p = Path(logs_dir)
        if not p.exists() or not p.is_dir():
            raise ValueError(f"PromptAnalyzer: logs_dir does not exist or is not a directory: {logs_dir}")

        pair_files = sorted([fp for fp in p.glob("*.json") if fp.name != "summary.json"])
        if not pair_files:
            raise ValueError(
                f"PromptAnalyzer: no pair json files found in {logs_dir}. "
                "Expected files like 0000.json (summary.json excluded)."
            )

        required_keys = {
            "source_conversation",
            "extracted_memory",
            "target_conversation",
            "target_reward",
        }
        for fp in pair_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception as e:
                raise ValueError(f"PromptAnalyzer: invalid JSON file {fp}: {e}") from e

            if not isinstance(rec, dict):
                raise ValueError(f"PromptAnalyzer: file {fp} must contain a JSON object.")

            missing = sorted(required_keys - set(rec.keys()))
            if missing:
                raise ValueError(
                    f"PromptAnalyzer: file {fp} missing required keys: {missing}. "
                    f"Required schema keys: {sorted(required_keys)}"
                )

            try:
                float(rec.get("target_reward"))
            except Exception as e:
                raise ValueError(
                    f"PromptAnalyzer: file {fp} has non-numeric target_reward={rec.get('target_reward')!r}"
                ) from e

        return len(pair_files)

    def analyze(
        self,
        logs_dir: str,
        base_prompt: str,
        round_id: int,
        num_samples: int,
        output_json_path: str,
    ) -> dict[str, Any]:
        _log(
            f"prompt_analyzer validate start: round={round_id} logs_dir={logs_dir}"
        )
        validated_count = self._validate_logs_dir(logs_dir)
        _log(
            f"prompt_analyzer validate end: round={round_id} pair_files={validated_count}"
        )
        tools = [
            ExtractionViewerTool(logs_dir, model=self.analysis_model),
            CaseDetailsTool(logs_dir),
        ]
        prompt = self._build_prompt(
            logs_dir=logs_dir,
            base_prompt=base_prompt,
            round_id=round_id,
            num_samples=num_samples,
        )
        configured_max_tokens = int(self.model_kwargs.get("max_tokens", 0) or 0)
        attempt_max_tokens = configured_max_tokens
        _log(
            "prompt_analyzer run start: "
            f"round={round_id} model={self.analysis_model} max_steps={self.max_steps} "
            f"max_tokens={attempt_max_tokens if attempt_max_tokens else 'default'}"
        )
        retry_used = False
        while True:
            self.model = self._build_model(
                override_max_tokens=attempt_max_tokens if attempt_max_tokens > 0 else None
            )
            agent = ToolCallingAgent(
                model=self.model,
                tools=tools,
                summary_interval=5,
                max_steps=self.max_steps,
                prompts_type="default",
                memory_provider=None,
            )
            try:
                result = retry_with_backoff(lambda: agent.run(prompt), logger=_log)
                break
            except Exception as e:
                if retry_used:
                    raise
                suggested = suggest_max_tokens(str(e), attempt_max_tokens)
                if suggested is None:
                    raise
                retry_used = True
                _log(
                    "prompt_analyzer context overflow detected, retry with reduced max_tokens: "
                    f"{attempt_max_tokens} -> {suggested}"
                )
                attempt_max_tokens = suggested
        _log(f"prompt_analyzer run end: round={round_id}")
        raw_text = result if isinstance(result, str) else str(result)
        thinking_text, visible_text = split_thinking_visible(raw_text)
        out = {
            "round_id": round_id,
            "logs_dir": logs_dir,
            "analysis_model": self.analysis_model,
            "validated_logs_dir": True,
            "validated_pair_files_count": validated_count,
            "analysis_result": visible_text,
            "analysis_raw_result": raw_text,
            "analysis_thinking_detected": bool(thinking_text),
        }
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        _log(
            f"prompt_analyzer wrote report: round={round_id} output={out_path}"
        )
        return out

    # ------------------------------------------------------------------
    # Cluster-scoped analysis
    # ------------------------------------------------------------------

    def _load_cluster_template(self) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / "cluster_analysis_prompt.yaml"
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("prompt_template", "").strip()

    def _build_cluster_prompt(
        self,
        logs_dir: str,
        base_prompt: str,
        round_id: int,
        cluster_label: str,
        cluster_description: str,
        task_ids: list[str],
    ) -> str:
        template = self._load_cluster_template()
        return template.format(
            logs_dir=logs_dir,
            base_prompt=base_prompt,
            round_id=round_id,
            cluster_label=cluster_label,
            cluster_description=cluster_description,
            task_ids=", ".join(task_ids),
            num_cluster_examples=len(task_ids),
        )

    def analyze_cluster(
        self,
        logs_dir: str,
        base_prompt: str,
        round_id: int,
        cluster_id: str,
        cluster_label: str,
        cluster_description: str,
        task_ids: list[str],
        output_json_path: str,
    ) -> dict[str, Any]:
        """Run tool-calling analysis scoped to a single cluster's task IDs."""
        _log(
            f"prompt_analyzer cluster analyze start: round={round_id} "
            f"cluster={cluster_id} ({cluster_label}) num_tasks={len(task_ids)}"
        )
        tools = [
            ExtractionViewerTool(logs_dir, model=self.analysis_model),
            CaseDetailsTool(logs_dir),
        ]
        prompt = self._build_cluster_prompt(
            logs_dir=logs_dir,
            base_prompt=base_prompt,
            round_id=round_id,
            cluster_label=cluster_label,
            cluster_description=cluster_description,
            task_ids=task_ids,
        )
        configured_max_tokens = int(self.model_kwargs.get("max_tokens", 0) or 0)
        attempt_max_tokens = configured_max_tokens
        retry_used = False
        while True:
            self.model = self._build_model(
                override_max_tokens=attempt_max_tokens if attempt_max_tokens > 0 else None
            )
            agent = ToolCallingAgent(
                model=self.model,
                tools=tools,
                summary_interval=5,
                max_steps=self.max_steps,
                prompts_type="default",
                memory_provider=None,
            )
            try:
                result = retry_with_backoff(lambda: agent.run(prompt), logger=_log)
                break
            except Exception as e:
                if retry_used:
                    raise
                suggested = suggest_max_tokens(str(e), attempt_max_tokens)
                if suggested is None:
                    raise
                retry_used = True
                _log(
                    f"prompt_analyzer cluster overflow, retry: {attempt_max_tokens} -> {suggested}"
                )
                attempt_max_tokens = suggested

        raw_text = result if isinstance(result, str) else str(result)
        thinking_text, visible_text = split_thinking_visible(raw_text)
        out = {
            "round_id": round_id,
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "cluster_description": cluster_description,
            "task_ids": task_ids,
            "logs_dir": logs_dir,
            "analysis_model": self.analysis_model,
            "analysis_result": visible_text,
            "analysis_raw_result": raw_text,
            "analysis_thinking_detected": bool(thinking_text),
        }
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        _log(
            f"prompt_analyzer cluster analyze end: round={round_id} "
            f"cluster={cluster_id} output={out_path}"
        )
        return out
