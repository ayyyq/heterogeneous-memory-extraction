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


class ExampleSummary(TypedDict):
    """Extraction-targeted summary for one training example."""

    task_id: str
    summary: str
    target_reward: float


class Cluster(TypedDict):
    """A cluster of examples grouped by extraction strategy pattern."""

    cluster_id: str
    label: str
    description: str
    example_task_ids: list[str]


class ClusterPool(TypedDict):
    """Pool of clusters that persists and evolves across rounds."""

    clusters: list[Cluster]
    round_updated: int


class ClusterAnalysis(TypedDict):
    """Analysis result for a single cluster."""

    cluster_id: str
    cluster_label: str
    cluster_description: str
    analysis_result: str
