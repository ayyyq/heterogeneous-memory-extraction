"""
Evaluation for ToolBench: Pass Rate (SoPR) + Win Rate (SoWR).

Pass Rate (SoPR):
  - LLM-as-judge evaluates if the model's answer solves the instruction.
  - Scoring: Solved (1.0), Unsure (0.5), Unsolved (0.0).

Win Rate (SoWR):
  - Pairwise comparison between model answer and reference.
  - Reports win/tie/lose ratios.

Both metrics follow StableToolBench evaluation protocol.

Two evaluation modes:
  - Heuristic: fast, no LLM calls, used during collection.
  - LLM judge: accurate, uses GPT-4o by default, run as post-processing step.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.data_processing.schemas import InferenceOutputExample
from src.evaluation.toolbench_judge_prompts import (
    CHECK_ANSWER_STATUS_PROMPT,
    SELECT_BETTER_ANSWER_PROMPT,
)
from src.llm import LLMChatCompletion, Message


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MODEL_ZOO = {
    "gpt-4o": ("gpt-4o-2024-08-06", "openai"),
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18", "openai"),
}

# StableToolBench default judge parameters
JUDGE_TEMPERATURE = 0.2
JUDGE_MAX_TOKENS = 1000
JUDGE_TOP_P = 1.0
JUDGE_NUM_COMPS = 1
JUDGE_EVALUATE_TIMES = 1  # Default 1 round; use --evaluate_times 3 for majority vote

_FEEDBACK_MESSAGES = frozenset({
    "You successfully finished this task!",
    "You failed this task.",
})


def _load_outputs(path: str) -> List[InferenceOutputExample]:
    outputs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(InferenceOutputExample.from_dict(json.loads(line)))
    return outputs


def _save_outputs(path: str, outputs: List[InferenceOutputExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(out.to_dict(), ensure_ascii=False) + "\n")


def _format_answer_json(out: InferenceOutputExample) -> str:
    """Format an output example into the JSON answer format expected by judge prompts."""
    final_answer = out.metadata.get("final_answer", "")
    give_up = out.metadata.get("give_up", False)
    num_steps = out.metadata.get("num_steps", 0)
    trajectory = out.metadata.get("trajectory", [])

    # Build answer_details from trajectory
    answer_details = []
    for step in trajectory:
        answer_details.append({
            "step": step.get("step", 0),
            "thought": step.get("thought", ""),
            "action": step.get("action", ""),
            "action_input": step.get("action_input", ""),
        })

    answer = {
        "final_answer": final_answer,
        "give_up": give_up,
        "total_steps": num_steps,
        "answer_details": answer_details,
    }
    return json.dumps(answer, ensure_ascii=False)


def _parse_answer_status(raw: str) -> str:
    """Parse judge response for answer_status: Solved/Unsure/Unsolved."""
    raw_lower = raw.lower()
    # Try JSON parse first
    try:
        m = re.search(r"\{[^}]*\}", raw, re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            status = d.get("answer_status", "").strip()
            if status.lower() in ("solved", "unsure", "unsolved"):
                return status.capitalize()
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: look for keywords
    if "solved" in raw_lower and "unsolved" not in raw_lower:
        return "Solved"
    if "unsolved" in raw_lower:
        return "Unsolved"
    if "unsure" in raw_lower:
        return "Unsure"
    return "Unsolved"


def _status_to_score(status: str) -> float:
    return {"Solved": 1.0, "Unsure": 0.5, "Unsolved": 0.0}.get(status, 0.0)


def _parse_better_index(raw: str) -> Optional[int]:
    """Parse judge response for better_answer_index: 0 or 1."""
    # Try JSON parse first
    try:
        m = re.search(r"\{[^}]*\}", raw, re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            idx = d.get("better_answer_index")
            if idx in (0, 1):
                return idx
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: look for the last digit 0 or 1
    digits = re.findall(r"[01]", raw)
    if digits:
        return int(digits[-1])
    return None


# ---------------------------------------------------------------------------
# Heuristic evaluation (fast, no LLM calls)
# ---------------------------------------------------------------------------

def evaluate_toolbench_pass_rate_heuristic(
    output_file: str,
) -> Dict[str, float]:
    """
    Evaluate ToolBench pass rate using heuristics (no LLM judge).

    Scoring:
    - Solved (1.0): non-empty final answer, no give_up.
    - Unsure (0.5): very short answer (< 10 chars).
    - Unsolved (0.0): give_up, no final answer, or exhausted steps.
    """
    outputs = _load_outputs(output_file)
    if not outputs:
        print("[ToolBench] No outputs to evaluate.")
        return {}

    split_scores: Dict[str, List[float]] = {}
    all_scores: List[float] = []

    for out in outputs:
        final_answer = out.metadata.get("final_answer", "")
        give_up = out.metadata.get("give_up", False)
        split = out.metadata.get("split", "unknown")

        if give_up or not final_answer:
            score = 0.0
        elif len(str(final_answer).strip()) < 10:
            score = 0.5
        else:
            score = 1.0

        split_scores.setdefault(split, []).append(score)
        all_scores.append(score)
        out.label = score >= 1.0

    _save_outputs(output_file, outputs)

    results: Dict[str, float] = {}
    for split, scores in sorted(split_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        results[f"pass_rate/{split}"] = avg
        print(f"  [ToolBench] Pass Rate ({split}): {avg:.4f} ({len(scores)} samples)")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    results["pass_rate/overall"] = overall
    print(f"  [ToolBench] Pass Rate (overall): {overall:.4f} ({len(all_scores)} samples)")
    return results


# ---------------------------------------------------------------------------
# LLM judge: Pass Rate (SoPR)
# ---------------------------------------------------------------------------

async def evaluate_toolbench_pass_rate_llm_judge(
    output_file: str,
    metric_model: str = "gpt-4o",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    concurrency: int = 10,
    evaluate_times: int = JUDGE_EVALUATE_TIMES,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate ToolBench pass rate via LLM judge (SoPR).

    For each sample, the judge is called `evaluate_times` rounds.
    Final status is determined by majority vote across rounds.
    """
    outputs = _load_outputs(output_file)
    if not outputs:
        print("[ToolBench] No outputs to evaluate.")
        return {}

    if metric_model not in MODEL_ZOO:
        raise ValueError(f"metric_model must be one of {list(MODEL_ZOO.keys())}, got {metric_model!r}")

    model_name, model_source = MODEL_ZOO[metric_model]
    if model_source == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        base = api_base or os.getenv("OPENAI_API_BASE")
    else:
        key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        base = api_base or os.getenv("OPENAI_API_BASE") or "http://localhost:8001/v1"

    llm = LLMChatCompletion(model_name=model_name, api_base=base, api_key=key)
    sem = asyncio.Semaphore(concurrency)

    async def _judge_one(idx: int, out: InferenceOutputExample) -> Tuple[int, InferenceOutputExample]:
        query = out.task_main or ""
        final_answer = out.metadata.get("final_answer", "")
        give_up = out.metadata.get("give_up", False)

        # Short-circuit: no answer or gave up → Unsolved without LLM call
        if not final_answer or give_up:
            statuses: List[str] = ["Unsolved"]
        else:
            # Pass only final_answer string (aligned with StableToolBench check_answer_status)
            prompt = CHECK_ANSWER_STATUS_PROMPT.format(query=query, answer=final_answer)
            statuses = []
            for _round in range(evaluate_times):
                async with sem:
                    result = await llm.acall(
                        [Message(role="user", content=prompt)],
                        temperature=JUDGE_TEMPERATURE,
                        max_tokens=JUDGE_MAX_TOKENS,
                        top_p=JUDGE_TOP_P,
                        num_comps=JUDGE_NUM_COMPS,
                        enable_thinking=False,
                    )
                raw = (result.raw_response or "").strip()
                status = _parse_answer_status(raw)
                statuses.append(status)

        # Majority vote
        counter = Counter(statuses)
        final_status = counter.most_common(1)[0][0]
        score = _status_to_score(final_status)

        # Append success/fail feedback as final user message (like other agentic tasks)
        # Strip prior feedback first to make re-runs idempotent
        updated_messages = list(out.messages)
        while (updated_messages
               and updated_messages[-1].get("role") == "user"
               and updated_messages[-1].get("content", "") in _FEEDBACK_MESSAGES):
            updated_messages.pop()
        if final_status == "Solved":
            updated_messages.append({"role": "user", "content": "You successfully finished this task!"})
        else:
            updated_messages.append({"role": "user", "content": "You failed this task."})

        new_meta = dict(out.metadata)
        new_meta["autoeval_label"] = {
            "model": model_name,
            "answer_status": final_status,
            "score": score,
            "all_statuses": statuses,
        }
        new_out = InferenceOutputExample(
            source=out.source,
            id=out.id,
            task_main=out.task_main,
            messages=updated_messages,
            label=score >= 1.0,
            metadata=new_meta,
        )

        if verbose:
            print(
                f"  [ToolBench judge] id={out.id} status={final_status} "
                f"score={score} votes={statuses}"
            )
        if pbar is not None:
            pbar.update(1)
        return idx, new_out

    pbar = tqdm(total=len(outputs), desc="ToolBench pass rate", unit="ex") if not verbose else None
    tasks = [_judge_one(i, out) for i, out in enumerate(outputs)]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    if pbar is not None:
        pbar.close()

    ordered: List[Optional[InferenceOutputExample]] = [None] * len(outputs)
    first_error = None
    for r in results_list:
        if isinstance(r, Exception):
            if first_error is None:
                first_error = r
            continue
        idx, new_out = r
        ordered[idx] = new_out

    # If all failed, raise the first error so the user sees the full traceback
    if first_error is not None and all(o is None for o in ordered):
        raise first_error

    # Fill in any failed entries with originals
    for i, o in enumerate(ordered):
        if o is None:
            ordered[i] = outputs[i]

    _save_outputs(output_file, ordered)

    # Compute metrics
    split_scores: Dict[str, List[float]] = {}
    all_scores: List[float] = []
    for out in ordered:
        autoeval = out.metadata.get("autoeval_label", {})
        score = autoeval.get("score", 0.0)
        split = out.metadata.get("split", "unknown")
        split_scores.setdefault(split, []).append(score)
        all_scores.append(score)

    results: Dict[str, float] = {}
    for split, scores in sorted(split_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        results[f"pass_rate/{split}"] = avg
        print(f"  [ToolBench] SoPR ({split}): {avg:.4f} ({len(scores)} samples)")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    results["pass_rate/overall"] = overall
    print(f"  [ToolBench] SoPR (overall): {overall:.4f} ({len(all_scores)} samples)")
    return results


# ---------------------------------------------------------------------------
# LLM judge: Win Rate (SoWR)
# ---------------------------------------------------------------------------

async def evaluate_toolbench_win_rate_llm_judge(
    output_file: str,
    reference_file: str,
    metric_model: str = "gpt-4o",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    concurrency: int = 10,
    evaluate_times: int = JUDGE_EVALUATE_TIMES,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate ToolBench win rate via LLM judge (SoWR).

    Compares model outputs against reference outputs.
    Answer order is randomized across rounds to reduce evaluator bias.
    """
    if not os.path.isfile(reference_file):
        print(f"[ToolBench] Win rate skipped: reference file not found: {reference_file}")
        return {}

    outputs = _load_outputs(output_file)
    references = _load_outputs(reference_file)
    if not outputs or not references:
        print("[ToolBench] Win rate skipped: empty outputs or references.")
        return {}

    if metric_model not in MODEL_ZOO:
        raise ValueError(f"metric_model must be one of {list(MODEL_ZOO.keys())}, got {metric_model!r}")

    model_name, model_source = MODEL_ZOO[metric_model]
    if model_source == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required.")
        base = api_base or os.getenv("OPENAI_API_BASE")
    else:
        key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        base = api_base or os.getenv("OPENAI_API_BASE") or "http://localhost:8001/v1"

    llm = LLMChatCompletion(model_name=model_name, api_base=base, api_key=key)
    sem = asyncio.Semaphore(concurrency)
    ref_by_id = {r.id: r for r in references}

    # Build pairs
    pairs: List[Tuple[InferenceOutputExample, InferenceOutputExample]] = []
    for out in outputs:
        ref = ref_by_id.get(out.id)
        if ref is not None:
            pairs.append((out, ref))

    if not pairs:
        print("[ToolBench] Win rate skipped: no matching pairs.")
        return {}

    async def _judge_pair(
        out: InferenceOutputExample, ref: InferenceOutputExample,
    ) -> str:
        """Returns 'win', 'lose', or 'tie'."""
        query = out.task_main or ""
        out_answer = _format_answer_json(out)
        ref_answer = _format_answer_json(ref)

        votes: List[str] = []
        for round_idx in range(evaluate_times):
            # Swap order on odd rounds to reduce position bias
            if round_idx % 2 == 0:
                a0, a1 = out_answer, ref_answer
                model_is_0 = True
            else:
                a0, a1 = ref_answer, out_answer
                model_is_0 = False

            prompt = SELECT_BETTER_ANSWER_PROMPT.format(
                query=query, answer_0=a0, answer_1=a1,
            )
            async with sem:
                result = await llm.acall(
                    [Message(role="user", content=prompt)],
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=JUDGE_MAX_TOKENS,
                    top_p=JUDGE_TOP_P,
                    num_comps=JUDGE_NUM_COMPS,
                    enable_thinking=False,
                )
            raw = (result.raw_response or "").strip()
            better_idx = _parse_better_index(raw)

            if better_idx is None:
                votes.append("tie")
            elif (better_idx == 0 and model_is_0) or (better_idx == 1 and not model_is_0):
                votes.append("win")
            else:
                votes.append("lose")

        # Majority vote
        counter = Counter(votes)
        result = counter.most_common(1)[0][0]
        if pbar is not None:
            pbar.update(1)
        return result

    pbar = tqdm(total=len(pairs), desc="ToolBench win rate", unit="pair") if not verbose else None
    tasks = [_judge_pair(out, ref) for out, ref in pairs]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    if pbar is not None:
        pbar.close()

    wins = ties = losses = 0
    for r in results_list:
        if isinstance(r, Exception):
            print(f"  [ToolBench win rate] Error: {r}")
            ties += 1
            continue
        if r == "win":
            wins += 1
        elif r == "lose":
            losses += 1
        else:
            ties += 1

    total = wins + ties + losses
    results: Dict[str, float] = {}
    if total > 0:
        results["win_rate"] = wins / total
        results["tie_rate"] = ties / total
        results["lose_rate"] = losses / total
        print(
            f"  [ToolBench] SoWR: win={wins/total:.4f} tie={ties/total:.4f} "
            f"lose={losses/total:.4f} (W={wins}, T={ties}, L={losses}, total={total})"
        )

    if verbose:
        print(f"  [ToolBench] Win rate judge model: {model_name}")

    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_toolbench(
    dataset_name: str,
    output_file: str,
    reference_file: Optional[str] = None,
) -> None:
    """
    Run heuristic ToolBench evaluation (used during collection).
    For LLM judge evaluation, use run_llm_judge_evaluation.py.
    """
    print(f"\n=== ToolBench Evaluation ({dataset_name}) ===")
    print(f"  Output: {output_file}")
    evaluate_toolbench_pass_rate_heuristic(output_file)


def print_toolbench_metrics(output_file: str) -> None:
    """Print ToolBench metrics from an existing output file (checks for autoeval_label first)."""
    outputs = _load_outputs(output_file)
    if not outputs:
        print("[ToolBench] No outputs.")
        return

    # Check if LLM judge labels exist
    has_autoeval = any(out.metadata.get("autoeval_label") for out in outputs)

    if has_autoeval:
        print(f"\n=== ToolBench Evaluation (LLM Judge) ===")
        print(f"  File: {output_file}")
        split_scores: Dict[str, List[float]] = {}
        all_scores: List[float] = []
        for out in outputs:
            autoeval = out.metadata.get("autoeval_label", {})
            score = autoeval.get("score", 0.0)
            split = out.metadata.get("split", "unknown")
            split_scores.setdefault(split, []).append(score)
            all_scores.append(score)

        for split, scores in sorted(split_scores.items()):
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"  [ToolBench] SoPR ({split}): {avg:.4f} ({len(scores)} samples)")
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"  [ToolBench] SoPR (overall): {overall:.4f} ({len(all_scores)} samples)")
    else:
        # Fallback to heuristic
        evaluate_toolbench("toolbench", output_file)
