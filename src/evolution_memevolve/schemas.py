"""Typed schemas for tournament prompt evolution."""

from __future__ import annotations

from typing import TypedDict


class PromptCandidate(TypedDict):
    """A prompt candidate in outer-loop evolution."""

    candidate_id: str
    system_prompt: str
    rationale: str
    parent_prompt_id: str


class EvalSummary(TypedDict):
    """Evaluation summary for one candidate/base prompt."""

    prompt_id: str
    logs_dir: str
    num_samples: int
    accuracy: float
