"""Prompt candidate generation for extraction evolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import json_repair
import litellm
import yaml
from openai import OpenAI

from src.evolution_clue.schemas import PromptCandidate
from src.evolution_clue.utils import split_thinking_visible, suggest_max_tokens, retry_with_backoff


def _log(msg: str) -> None:
    print(f"[memevolve] {msg}", flush=True)


class PromptGenerator:
    """Generate system_prompt-only candidates from analysis report."""

    def __init__(self, generation_model: str):
        self.generation_model = generation_model
        if not self._is_gemini_model(generation_model):
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

    def _load_template(self) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / "generation_prompt.yaml"
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("prompt_template", "").strip()

    def _build_prompt(
        self,
        analysis_text: str,
        base_system_prompt: str,
        num_systems: int,
        parent_prompt_id: str,
    ) -> str:
        template = self._load_template()
        return template.format(
            analysis_text=analysis_text,
            base_system_prompt=base_system_prompt,
            num_systems=num_systems,
            parent_prompt_id=parent_prompt_id,
        )

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

    def generate(
        self,
        analysis_report: dict[str, Any],
        base_system_prompt: str,
        num_systems: int,
        parent_prompt_id: str,
        output_json_path: str,
    ) -> list[PromptCandidate]:
        _log(
            f"prompt_generator start: model={self.generation_model} num_requested={num_systems} parent_prompt_id={parent_prompt_id}"
        )
        analysis_text = str(analysis_report.get("analysis_result", ""))
        prompt = self._build_prompt(
            analysis_text=analysis_text,
            base_system_prompt=base_system_prompt,
            num_systems=num_systems,
            parent_prompt_id=parent_prompt_id,
        )
        decoding_kwargs = self._decoding_kwargs(self.generation_model)
        max_tokens = 20000
        _log(
            f"prompt_generator model call start: model={self.generation_model} max_tokens={max_tokens}"
        )
        retry_used = False
        while True:
            try:
                def _call_llm():
                    if self._is_gemini_model(self.generation_model):
                        gemini_params = {
                            "model": self.generation_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            **decoding_kwargs,
                        }
                        # Only Gemini 3.0+ supports thinkingLevel
                        if "gemini-3" in self.generation_model.lower():
                            gemini_params["thinkingConfig"] = {"thinkingLevel": "MEDIUM", "includeThoughts": True}
                        elif "gemini-2.5" in self.generation_model.lower():
                            gemini_params["thinkingConfig"] = {"thinkingBudget": -1, "includeThoughts": True}
                        return litellm.completion(**gemini_params)
                    else:
                        return self.client.chat.completions.create(
                            model=self.generation_model,
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
                    "prompt_generator context overflow detected, retry with reduced max_tokens: "
                    f"{max_tokens} -> {suggested}"
                )
                max_tokens = suggested
        _log("prompt_generator model call end")
        if self._is_gemini_model(self.generation_model):
            msg = response.choices[0]["message"] if isinstance(response.choices[0], dict) else response.choices[0].message
            raw = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
        else:
            raw = response.choices[0].message.content or ""
        thinking_text, visible = split_thinking_visible(raw)
        payload = self._extract_json_block(visible)
        try:
            parsed: list[dict[str, Any]] = json.loads(payload)
        except json.JSONDecodeError:
            _log("prompt_generator: json.loads failed, falling back to json_repair")
            parsed = json_repair.loads(payload)
            if not isinstance(parsed, (list, dict)):
                preview = visible[:500].replace("\n", "\\n")
                raise ValueError(
                    "PromptGenerator: failed to parse JSON from visible output. "
                    f"visible_preview='{preview}'"
                )
        if isinstance(parsed, dict):
            parsed = [parsed]
        candidates: list[PromptCandidate] = []
        for i, item in enumerate(parsed):
            system_prompt = str(item.get("system_prompt", "")).strip()
            if not system_prompt:
                continue
            candidate_id = str(item.get("candidate_id", f"cand_{i + 1:02d}"))
            rationale = str(item.get("rationale", "")).strip()
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "system_prompt": system_prompt,
                    "rationale": rationale,
                    "parent_prompt_id": parent_prompt_id,
                }
            )

        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generation_model": self.generation_model,
                    "num_requested": num_systems,
                    "num_valid": len(candidates),
                    "candidates": candidates,
                    "raw_response": raw,
                    "visible_response": visible,
                    "thinking_detected": bool(thinking_text),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        _log(
            f"prompt_generator end: num_valid={len(candidates)} output={out_path}"
        )
        return candidates

    # ------------------------------------------------------------------
    # Cross-cluster synthesis
    # ------------------------------------------------------------------

    def _load_cross_cluster_template(self) -> str:
        prompt_file = (
            Path(__file__).parent.parent / "prompts" / "cross_cluster_generation_prompt.yaml"
        )
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("prompt_template", "").strip()

    @staticmethod
    def _format_cluster_analyses(cluster_analyses: list[dict[str, Any]]) -> str:
        sections = []
        for ca in cluster_analyses:
            sections.append(
                f"### Cluster: {ca.get('cluster_label', ca.get('cluster_id', '?'))}\n\n"
                f"{ca.get('analysis_result', '')}"
            )
        return "\n\n---\n\n".join(sections)

    def generate_from_clusters(
        self,
        cluster_analyses: list[dict[str, Any]],
        base_system_prompt: str,
        parent_prompt_id: str,
        output_json_path: str,
    ) -> list[PromptCandidate]:
        """Generate one candidate by synthesizing per-cluster analysis reports."""
        num_clusters = len(cluster_analyses)
        _log(
            f"prompt_generator cross_cluster start: model={self.generation_model} "
            f"num_clusters={num_clusters} parent_prompt_id={parent_prompt_id}"
        )
        template = self._load_cross_cluster_template()
        prompt = template.format(
            num_clusters=num_clusters,
            cluster_analyses_text=self._format_cluster_analyses(cluster_analyses),
            base_system_prompt=base_system_prompt,
            parent_prompt_id=parent_prompt_id,
        )

        decoding_kwargs = self._decoding_kwargs(self.generation_model)
        max_tokens = 20000
        retry_used = False
        while True:
            try:
                def _call_llm_cross():
                    if self._is_gemini_model(self.generation_model):
                        gemini_params = {
                            "model": self.generation_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            **decoding_kwargs,
                        }
                        # Only Gemini 3.0+ supports thinkingLevel
                        if "gemini-3" in self.generation_model.lower():
                            gemini_params["thinkingConfig"] = {"thinkingLevel": "MEDIUM", "includeThoughts": True}
                        elif "gemini-2.5" in self.generation_model.lower():
                            gemini_params["thinkingConfig"] = {"thinkingBudget": -1, "includeThoughts": True}
                        return litellm.completion(**gemini_params)
                    else:
                        return self.client.chat.completions.create(
                            model=self.generation_model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            **decoding_kwargs,
                        )
                response = retry_with_backoff(_call_llm_cross, logger=_log)
                break
            except Exception as e:
                if retry_used:
                    raise
                suggested = suggest_max_tokens(str(e), max_tokens)
                if suggested is None:
                    raise
                retry_used = True
                _log(
                    f"prompt_generator cross_cluster overflow, retry: {max_tokens} -> {suggested}"
                )
                max_tokens = suggested

        if self._is_gemini_model(self.generation_model):
            msg = response.choices[0]["message"] if isinstance(response.choices[0], dict) else response.choices[0].message
            raw = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
        else:
            raw = response.choices[0].message.content or ""
        thinking_text, visible = split_thinking_visible(raw)
        payload = self._extract_json_block(visible)
        try:
            parsed: list[dict[str, Any]] = json.loads(payload)
        except json.JSONDecodeError:
            _log("prompt_generator cross_cluster: json.loads failed, falling back to json_repair")
            parsed = json_repair.loads(payload)
            if not isinstance(parsed, (list, dict)):
                preview = visible[:500].replace("\n", "\\n")
                raise ValueError(
                    "PromptGenerator cross_cluster: failed to parse JSON. "
                    f"visible_preview='{preview}'"
                )

        if isinstance(parsed, dict):
            parsed = [parsed]
        candidates: list[PromptCandidate] = []
        for i, item in enumerate(parsed):
            system_prompt = str(item.get("system_prompt", "")).strip()
            if not system_prompt:
                continue
            candidate_id = str(item.get("candidate_id", f"cand_{i + 1:02d}"))
            rationale = str(item.get("rationale", "")).strip()
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "system_prompt": system_prompt,
                    "rationale": rationale,
                    "parent_prompt_id": parent_prompt_id,
                }
            )

        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generation_model": self.generation_model,
                    "generation_mode": "cross_cluster",
                    "num_clusters": num_clusters,
                    "num_valid": len(candidates),
                    "candidates": candidates,
                    "raw_response": raw,
                    "visible_response": visible,
                    "thinking_detected": bool(thinking_text),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        _log(
            f"prompt_generator cross_cluster end: num_valid={len(candidates)} output={out_path}"
        )
        return candidates
