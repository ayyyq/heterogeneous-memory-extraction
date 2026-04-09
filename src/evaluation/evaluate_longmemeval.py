"""
LongMemEval evaluation: LLM-as-judge on InferenceOutputExample jsonl.

Reads/writes same file: loads InferenceOutputExample jsonl, runs judge per row,
updates label and metadata.autoeval_label, writes back. Also provides
print_longmemeval_metrics() to re-print metrics from an already-evaluated file.

Uses src.llm for judge calls; decoding params for judge are hardcoded.
"""

import json
from multiprocessing import Value
import os
from typing import List, Optional

import numpy as np

from src.data_processing.schemas import InferenceOutputExample
from src.llm import LLMChatCompletion, Message
from src.utils import print_example

MODEL_ZOO = {
    "llama-3.1-70b-instruct": ("openai/Llama-3.1-70B-Instruct", "local"),
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18", "openai"),
    "gpt-4o": ("gpt-4o-2024-08-06", "openai"),
}

# Hardcoded decoding params for judge (yes/no)
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 10
JUDGE_TOP_P = 1.0
JUDGE_NUM_COMPS = 1

TASK_TYPES = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    """Build judge prompt per LongMemEval evaluate_qa.py."""
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no. In addition, "
                "do not penalize off-by-one errors for the number of days. If the question asks for "
                "the number of days/weeks/months, etc., and the model makes off-by-one errors "
                "(e.g., predicting 19 days when the answer is 18), the model's response is still correct.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response "
                "from a model. Please answer yes if the response satisfies the desired response. "
                "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(f"Task type '{task}' not supported")
    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is given "
        "but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def _get_hypothesis(out: InferenceOutputExample) -> str:
    """Get hypothesis from metadata.hypothesis or last assistant message."""
    hyp = (out.metadata.get("hypothesis") or "").strip()
    if hyp:
        return hyp
    for m in reversed(out.messages or []):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _load_outputs(path: str) -> List[InferenceOutputExample]:
    with open(path, "r", encoding="utf-8") as f:
        return [InferenceOutputExample.from_dict(json.loads(line)) for line in f if line.strip()]


def _save_outputs(path: str, outputs: List[InferenceOutputExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(out.to_dict(), ensure_ascii=False) + "\n")


def _print_metrics_from_outputs(outputs: List[InferenceOutputExample]) -> None:
    """Aggregate and print per-task, task-averaged, overall, abstention accuracy."""
    type2acc = {t: [] for t in TASK_TYPES}
    abstention_acc: List[int] = []
    for out in outputs:
        autoeval = out.metadata.get("autoeval_label")
        if autoeval is None:
            continue
        qtype = out.metadata.get("question_type")
        if qtype in type2acc:
            type2acc[qtype].append(1 if autoeval.get("label") else 0)
        qid = str(out.metadata.get("question_id", ""))
        if "_abs" in qid:
            abstention_acc.append(1 if autoeval.get("label") else 0)
    all_acc: List[int] = []
    task_acc: List[float] = []
    print("\nEvaluation results by task:")
    for k, v in type2acc.items():
        if v:
            acc = float(np.mean(v))
            task_acc.append(acc)
            all_acc.extend(v)
            print(f"\t{k}: {round(acc, 4)} ({len(v)})")
    task_avg = float(np.mean(task_acc)) if task_acc else 0.0
    overall = float(np.mean(all_acc)) if all_acc else 0.0
    abs_acc = float(np.mean(abstention_acc)) if abstention_acc else 0.0
    print(f"\nTask-averaged Accuracy: {round(task_avg, 4)}")
    print(f"Overall Accuracy: {round(overall, 4)}")
    print(f"Abstention Accuracy: {round(abs_acc, 4)} ({len(abstention_acc)})")


def print_longmemeval_metrics(output_file: str) -> None:
    """
    Load InferenceOutputExample jsonl (already evaluated, with metadata.autoeval_label)
    and print per-task / task-averaged / overall / abstention accuracy. Does not write file.
    """
    outputs = _load_outputs(output_file)
    _print_metrics_from_outputs(outputs)


def evaluate_longmemeval(
    output_file: str,
    metric_model: str = "gpt-4o",
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Load InferenceOutputExample jsonl, run LLM-as-judge per row, update label and
    metadata.autoeval_label, write back to the same file, then print metrics.
    Uses src.llm with hardcoded judge decoding params.
    """
    if metric_model not in MODEL_ZOO:
        raise ValueError(f"metric_model must be one of {list(MODEL_ZOO.keys())}, got {metric_model!r}")

    model_name, model_source = MODEL_ZOO[metric_model]
    if model_source == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass openai_api_key.")
        api_base = openai_api_base or os.getenv("OPENAI_API_BASE")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        api_base = openai_api_base or os.getenv("OPENAI_API_BASE") or "http://localhost:8001/v1"

    llm = LLMChatCompletion(model_name=model_name, api_base=api_base, api_key=api_key)

    outputs = _load_outputs(output_file)
    print_example("First InferenceOutputExample", outputs[0].to_dict())

    updated = []
    for out in outputs:
        meta = out.metadata
        if 'metadata' in meta:
            meta['metadata']['source_id'] = meta['source_id']
            meta = meta['metadata']
        question_id = meta.get("question_id")
        question = meta.get("question") or out.task_main or ""
        answer = meta.get("answer") or ""
        qtype = meta.get("question_type", "")
        hypothesis = _get_hypothesis(out)

        if question_id is None or not qtype:
            if verbose:
                print(f"Warning: skipping id={out.id} (missing question_id or question_type)")
            updated.append(out)
            continue

        abstention = "_abs" in str(question_id)
        prompt = get_anscheck_prompt(qtype, question, answer, hypothesis, abstention=abstention)
        messages = [Message(role="user", content=prompt)]
        result = llm(
            messages,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            top_p=JUDGE_TOP_P,
            num_comps=JUDGE_NUM_COMPS,
            enable_thinking=False,
        )
        raw_response = (result.raw_response or "").strip()
        label = "yes" in raw_response.lower()

        autoeval_label = {"model": model_name, "label": label, "raw_response": raw_response}
        new_meta = dict(meta)
        new_meta["autoeval_label"] = autoeval_label
        new_out = InferenceOutputExample(
            source=out.source,
            id=out.id,
            task_main=out.task_main,
            messages=out.messages,
            label=label,
            metadata=new_meta,
        )
        updated.append(new_out)
        if verbose:
            print(
                json.dumps(
                    {"question": question[:80], "answer": answer[:80], "hypothesis": hypothesis[:80], "autoeval_label": label},
                    indent=2,
                ),
                flush=True,
            )

    _save_outputs(output_file, updated)
    print(f"Saved to {output_file}")
    _print_metrics_from_outputs(updated)
