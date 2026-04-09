"""Extraction-targeted summarization for each training example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.llm import LLMChatCompletion, Message
from src.evolution_clue.schemas import ExampleSummary


def _log(msg: str) -> None:
    print(f"[memevolve] {msg}", flush=True)


class ExampleSummarizer:
    """Generate extraction-targeted summaries for pair-level logs."""

    CACHE_KEY = "extraction_strategy_summary"

    def __init__(self, model: str):
        self.model = model
        # LLMChatCompletion uses litellm which requires "openai/" prefix for vllm
        litellm_model = model if "/" in model else f"openai/{model}"
        self.llm = LLMChatCompletion(model_name=litellm_model)

    def _load_template(self) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / "summarization_prompt.yaml"
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("prompt_template", "").strip()

    def _summarize_one(self, rec: dict[str, Any]) -> str:
        template = self._load_template()
        prompt = template.format(
            source_preview=(rec.get("source_conversation", "") or "")[:4096],
            extracted_memory=rec.get("extracted_memory", "") or "",
            target_preview=(rec.get("target_conversation", "") or "")[:4096],
            target_reward=rec.get("target_reward", 0.0),
        )
        if "qwen3" in self.model.lower():
            result = self.llm(
                messages=[Message(role="user", content=prompt)],
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                enable_thinking=True,
            )
        elif "gemini" in self.model.lower():
            result = self.llm(
                messages=[Message(role="user", content=prompt)],
                temperature=1.0,
                top_p=0.95,
                max_tokens=2048,
                enable_thinking=True,
                thinking_budget=1536 if "gemini-2.5" in self.model.lower() else None,
                thinking_level="MEDIUM" if "gemini-3" in self.model.lower() else None,
            )
        else:
            result = self.llm(
                messages=[Message(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=2048,
                enable_thinking=True,
            )
        return (result.visible_content or result.raw_response or "").strip()

    def summarize_batch(self, logs_dir: str) -> list[ExampleSummary]:
        """Summarize all pair logs in *logs_dir*, caching results in the JSON files."""
        p = Path(logs_dir)
        pair_files = sorted(
            fp for fp in p.glob("*.json") if fp.name != "summary.json"
        )
        if not pair_files:
            raise ValueError(f"ExampleSummarizer: no pair files in {logs_dir}")

        summaries: list[ExampleSummary] = []
        total = len(pair_files)
        _log(f"example_summarizer start: logs_dir={logs_dir} num_files={total}")

        for idx, fp in enumerate(pair_files):
            task_id = fp.stem
            with open(fp, "r", encoding="utf-8") as f:
                rec = json.load(f)

            # Use cached summary if available
            if self.CACHE_KEY in rec:
                summary_text = rec[self.CACHE_KEY]
            else:
                summary_text = self._summarize_one(rec)
                rec[self.CACHE_KEY] = summary_text
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2, ensure_ascii=False)

            summaries.append(
                ExampleSummary(
                    task_id=task_id,
                    summary=summary_text,
                    target_reward=float(rec.get("target_reward", 0.0)),
                )
            )
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                _log(f"example_summarizer progress: {idx + 1}/{total}")

        _log(f"example_summarizer end: {len(summaries)} summaries")
        return summaries
