"""
Utilities for evolution_ace.

Mirrors ACE utils: evaluate_test_set, extract_answer, count_tokens, log_bullet_usage.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def extract_answer(response: str) -> str:
    """Extract final_answer from generator response. Mirrors ACE utils.extract_answer."""
    try:
        parsed = json.loads(response)
        return str(parsed.get("final_answer", "No final answer found"))
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    # Fallback for non-JSON
    for key in ['"final_answer"', "'final_answer'"]:
        if key in response:
            # Simple extraction
            pass
    return "No final answer found"


def count_tokens(text: str) -> int:
    """Approximate token count. Use tiktoken if available."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def _extract_bullets_with_content(playbook: str, bullet_ids: List[str]) -> List[Dict[str, Any]]:
    """Extract bullet content from playbook for given bullet IDs."""
    from .playbook import parse_playbook_line
    bullets_with_content = []
    if playbook and bullet_ids:
        for line in playbook.split("\n"):
            parsed = parse_playbook_line(line)
            if parsed and parsed["id"] in bullet_ids:
                bullets_with_content.append({
                    "bullet_id": parsed["id"],
                    "content": parsed.get("content") or "Content not found",
                })
    for bid in bullet_ids:
        if not any(b["bullet_id"] == bid for b in bullets_with_content):
            bullets_with_content.append({"bullet_id": bid, "content": None})
    return bullets_with_content


def log_bullet_usage(
    usage_log_path: Optional[str],
    epoch: int,
    step: int,
    task_dict: Dict[str, Any],
    bullet_ids: List[str],
    playbook: str = "",
    reflection_content: str = "",
    is_correct: bool = False,
) -> None:
    """Log bullet usage for analysis. ACE format. Accepts task_dict for backward compatibility."""
    if not usage_log_path:
        return
    try:
        bullets_with_content = _extract_bullets_with_content(playbook, bullet_ids)
        sample_context = task_dict.get("context", "") if task_dict else ""
        sample_question = task_dict.get("question", "") if task_dict else ""
        from .logger import log_bullet_usage as _log
        _log(
            usage_log_path,
            epoch,
            step,
            bullet_ids,
            bullets_with_content,
            is_correct,
            sample_context=sample_context,
            sample_question=sample_question,
            reflection_summary=reflection_content or None,
        )
    except Exception as e:
        print(f"[evolution_ace] log_bullet_usage error: {e}")


def evaluate_single_test_sample(
    args_tuple: Tuple,
    data_processor,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Evaluate one sample. Passes task_dict to generator for memory extraction flow.
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        gen_response, bullet_ids, _ = generator.generate(
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
            call_id=f"test_eval_{i}",
            log_dir=log_dir,
            task_dict=task_dict,
        )

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return {
            "index": i,
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "success": True,
        }, None
    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(
    data_processor,
    generator,
    playbook: str,
    test_samples: List[Dict[str, Any]],
    max_tokens: int = 4096,
    log_dir: Optional[str] = None,
    max_workers: int = 20,
) -> Tuple[Dict, Dict]:
    """Parallel evaluation. Mirrors ACE utils.evaluate_test_set."""
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir)
        for i, sample in enumerate(test_samples)
    ]

    results = {"correct": 0, "total": 0, "no_answer": 0, "answers": [], "targets": [], "errors": []}

    def eval_wrapper(args):
        return evaluate_single_test_sample(args, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_wrapper, a): a for a in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            result, error = future.result()
            if error:
                print(error)
                continue
            if result and result.get("success"):
                results["correct"] += 1 if result["is_correct"] else 0
                results["total"] += 1
                results["answers"].append(result["final_answer"])
                results["targets"].append(result["target"])
                if not result["is_correct"]:
                    results["errors"].append({
                        "index": result["index"],
                        "prediction": result["final_answer"],
                        "ground_truth": result["target"],
                    })
                if result["final_answer"] == "No final answer found":
                    results["no_answer"] += 1
            if i % 50 == 0 and results["total"] > 0:
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {results['correct']/results['total']:.3f}")

    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(results["answers"], results["targets"])
        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"],
        }
        error_logs = {"accuracy": accuracy, "errors": results["errors"]}
        print(f"\nFinal Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")
    else:
        final_results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print("\nNo valid results!")

    return final_results, error_logs


async def aevaluate_test_set_safe(
    data_processor,
    generator,
    playbook: str,
    test_samples: List[Dict[str, Any]],
    max_tokens: int = 4096,
    log_dir: Optional[str] = None,
    async_max_concurrency: int = 8,
) -> Tuple[Dict, Dict]:
    """
    Async safe evaluation:
    - concurrent extraction (bounded by async_max_concurrency)
    - serial inference + reward (inside generator batch path)
    """
    print(f"\n{'='*40}")
    print(
        "ASYNC SAFE EVALUATING TEST SET - "
        f"{len(test_samples)} samples, extraction_concurrency={async_max_concurrency}"
    )
    print(f"{'='*40}")

    batch_results, batch_errors = await generator.abatch_generate_for_eval(
        task_dicts=test_samples,
        playbook=playbook,
        max_tokens=max_tokens,
        log_dir=log_dir,
        async_max_concurrency=async_max_concurrency,
    )

    results = {"correct": 0, "total": 0, "no_answer": 0, "answers": [], "targets": [], "errors": []}
    for i, result in enumerate(batch_results, 1):
        if not result or not result.get("success"):
            continue
        results["correct"] += 1 if result["is_correct"] else 0
        results["total"] += 1
        results["answers"].append(result["final_answer"])
        results["targets"].append(result["target"])
        if not result["is_correct"]:
            results["errors"].append(
                {
                    "index": result["index"],
                    "prediction": result["final_answer"],
                    "ground_truth": result["target"],
                }
            )
        if result["final_answer"] == "No final answer found":
            results["no_answer"] += 1
        if i % 50 == 0 and results["total"] > 0:
            print(f"Progress: {i}/{len(batch_results)}, Accuracy: {results['correct']/results['total']:.3f}")

    for error in batch_errors:
        if error:
            print(error)

    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(results["answers"], results["targets"])
        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"],
        }
        error_logs = {"accuracy": accuracy, "errors": results["errors"]}
        print(f"\nFinal Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")
    else:
        final_results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print("\nNo valid results!")

    return final_results, error_logs
