"""ToolCallingAgent tools for extraction pair logs.

Mirrors the original MemEvolve trajectory_tools.py,
adapted for the memory-extraction domain:

  TrajectoryViewerTool -> ExtractionViewerTool  (view_extractions)
  StepViewerTool       -> CaseDetailsTool        (view_case_details)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from FlashOAgents import Tool

from src.llm import LLMChatCompletion, Message


# ---------------------------------------------------------------------------
# Internal repo helper
# ---------------------------------------------------------------------------

class _ExtractionLogRepo:
    def __init__(self, logs_dir: str):
        self.logs_dir = Path(logs_dir)

    def load_all(self) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        if not self.logs_dir.exists():
            return recs
        for path in sorted(self.logs_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    recs.append(json.load(f))
            except Exception:
                continue
        return recs

    def by_task_id(self, task_id: str) -> dict[str, Any] | None:
        path = self.logs_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_task(self, task_id: str, data: dict[str, Any]) -> None:
        path = self.logs_dir / f"{task_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def list_task_ids(self) -> list[str]:
        if not self.logs_dir.exists():
            return []
        return sorted(
            p.stem for p in self.logs_dir.glob("*.json") if p.name != "summary.json"
        )


# ---------------------------------------------------------------------------
# ExtractionFeedbackAggregator  (mirrors TrajectoryFeedbackAggregator)
# ---------------------------------------------------------------------------

class ExtractionFeedbackAggregator:
    """Compute multi-dimensional feedback metrics for an extraction record."""

    def compute(self, rec: dict[str, Any]) -> dict[str, Any]:
        reward = float(rec.get("target_reward", 0.0) or 0.0)
        mem = rec.get("extracted_memory", "") or ""
        src = rec.get("source_conversation", "") or ""
        tgt = rec.get("target_conversation", "") or ""
        return {
            "accuracy": 1 if reward > 0 else 0,
            "reward_value": reward,
            "memory_chars": len(mem),
            "source_chars": len(src),
            "target_chars": len(tgt),
        }


# ---------------------------------------------------------------------------
# LLM summary generation  (mirrors _generate_trajectory_summary)
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
You are analyzing a memory extraction result. Given the following data, write a concise \
2-4 sentence extraction_summary that:
- Describes what information was extracted from the source conversation
- States whether the extraction was CORRECT (reward=1) or INCORRECT (reward=0)
- Notes the key reason for success or failure based on extracted memory vs target needs

source_conversation (first 600 chars):
{source_preview}

extracted_memory:
{extracted_memory}

target_reward: {target_reward}

Write only the extraction_summary, no extra formatting."""


def _generate_extraction_summary(rec: dict[str, Any], llm: LLMChatCompletion) -> str:
    prompt = _SUMMARY_PROMPT.format(
        source_preview=(rec.get("source_conversation", "") or "")[:600],
        extracted_memory=(rec.get("extracted_memory", "") or "")[:800],
        target_reward=rec.get("target_reward", 0.0),
    )
    try:
        if "qwen3" in llm.model_name.lower():
            result = llm(
                messages=[Message(role="user", content=prompt)],
                temperature=0.7,
                top_p=0.8,
                max_tokens=256,
                enable_thinking=False,
            )
        elif "gemini" in llm.model_name.lower():
            result = llm(
                messages=[Message(role="user", content=prompt)],
                temperature=1.0,
                max_tokens=256,
                thinking_budget=0 if "gemini-2.5" in llm.model_name.lower() else None,
                thinking_level="MINIMAL" if "gemini-3" in llm.model_name.lower() else None,
            )
        else:
            result = llm(
                messages=[Message(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=256,
            )
        return (result.visible_content or result.raw_response or "").strip()
    except Exception as e:
        return f"(summary generation failed: {e})"


# ---------------------------------------------------------------------------
# Tool 1: ExtractionViewerTool  (mirrors TrajectoryViewerTool)
# ---------------------------------------------------------------------------

class ExtractionViewerTool(Tool):
    """View extraction results for up to 5 task IDs with LLM-generated summaries.

    For each task the tool returns:
      - is_correct: whether target_reward > 0
      - source_preview: first 800 chars of the source conversation
      - extracted_memory: first 600 chars of the extracted memory string
      - target_reward: raw reward value
      - feedback: multi-dimensional metrics (accuracy, memory_chars, etc.)
      - extraction_summary: LLM-generated summary (cached after first call)

    Summary caching: first call generates and saves the summary. Future calls
    use the cached version. Limited to 5 tasks at once to avoid overwhelming context.
    """

    name = "view_extractions"
    description = (
        "View detailed extraction results for specific task IDs with LLM-generated "
        "extraction summaries. The extraction_summary describes what was extracted and "
        "why the target succeeded or failed. "
        "Summary caching: first call generates summary and saves it; future calls use "
        "the cached version. Limited to 5 tasks at once to avoid overwhelming context."
    )
    inputs = {
        "task_ids": {
            "type": "array",
            "description": "List of task IDs to view (maximum 5 tasks at once, e.g. ['0000','0003']).",
        }
    }
    output_type = "string"

    def __init__(
        self,
        logs_dir: str,
        model: str = "openai/Qwen3-32B",
        max_tasks: int = 5,
    ):
        super().__init__()
        self.repo = _ExtractionLogRepo(logs_dir)
        self.max_tasks = max_tasks
        litellm_model = model if "/" in model else f"openai/{model}"
        self.llm = LLMChatCompletion(model_name=litellm_model)
        self.feedback_agg = ExtractionFeedbackAggregator()

    def forward(self, task_ids: List[str]) -> str:
        if len(task_ids) > self.max_tasks:
            return json.dumps(
                {"error": f"Too many task ids. max={self.max_tasks}, got={len(task_ids)}"},
                indent=2,
            )
        out: dict[str, Any] = {}
        for task_id in task_ids:
            rec = self.repo.by_task_id(task_id)
            if rec is None:
                out[task_id] = {
                    "error": "not_found",
                    "suggestion": f"Valid task IDs: {self.repo.list_task_ids()[:10]}",
                }
                continue

            # Generate or retrieve cached extraction_summary
            if "extraction_summary" not in rec:
                summary = _generate_extraction_summary(rec, self.llm)
                rec["extraction_summary"] = summary
                self.repo.save_task(task_id, rec)
            else:
                summary = rec["extraction_summary"]

            feedback = self.feedback_agg.compute(rec)
            out[task_id] = {
                "is_correct": feedback["accuracy"] == 1,
                "target_reward": rec.get("target_reward"),
                "source_preview": (rec.get("source_conversation", "") or "")[:800],
                "extracted_memory": (rec.get("extracted_memory", "") or "")[:600],
                "feedback": feedback,
                "extraction_summary": summary,
            }
        return json.dumps(out, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: CaseDetailsTool  (mirrors StepViewerTool)
# ---------------------------------------------------------------------------

_VALID_FIELDS = frozenset(
    {"source_conversation", "extracted_memory", "target_conversation", "target_reward"}
)


class CaseDetailsTool(Tool):
    """View complete details of specific fields in a task's extraction record.

    Use this after reviewing extraction summaries from view_extractions to examine
    specific fields in full detail. When no fields are specified, all fields are returned.
    Valid fields: source_conversation, extracted_memory, target_conversation, target_reward.
    """

    name = "view_case_details"
    description = (
        "View complete details of specific fields in a task's extraction record. "
        "Use this after view_extractions to examine fields in full detail. "
        "Valid fields: source_conversation, extracted_memory, target_conversation, target_reward. "
        "Omit 'fields' to return all."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "Task ID to examine, e.g. '0007'.",
        },
        "fields": {
            "type": "array",
            "description": (
                "Optional list of fields to return. "
                "Valid: source_conversation, extracted_memory, target_conversation, target_reward. "
                "Defaults to all fields."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, logs_dir: str):
        super().__init__()
        self.repo = _ExtractionLogRepo(logs_dir)

    def forward(self, task_id: str, fields: Optional[List[str]] = None) -> str:
        rec = self.repo.by_task_id(task_id)
        if rec is None:
            return json.dumps(
                {
                    "error": "not_found",
                    "task_id": task_id,
                    "suggestion": f"Valid task IDs: {self.repo.list_task_ids()[:10]}",
                },
                indent=2,
            )

        if not fields:
            requested = sorted(_VALID_FIELDS)
        else:
            unknown = [f for f in fields if f not in _VALID_FIELDS]
            if unknown:
                return json.dumps(
                    {
                        "error": f"Unknown fields: {unknown}",
                        "valid_fields": sorted(_VALID_FIELDS),
                    },
                    indent=2,
                )
            requested = fields

        result: dict[str, Any] = {"task_id": task_id, "requested_fields": requested}
        for field in requested:
            result[field] = rec.get(field)
        return json.dumps(result, indent=2, ensure_ascii=False)

