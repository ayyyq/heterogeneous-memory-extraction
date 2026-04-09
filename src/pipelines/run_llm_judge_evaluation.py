"""
LLM-judge evaluation pipeline CLI.

Supports datasets that require a separate LLM-as-judge evaluation step:
  - longmemeval: answer correctness judge
  - toolbench: pass rate (SoPR) + win rate (SoWR) judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import List, Optional

from tqdm import tqdm

from src.data_processing.schemas import InferenceOutputExample
from src.evaluation.evaluate_longmemeval import (
    JUDGE_MAX_TOKENS,
    JUDGE_NUM_COMPS,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    MODEL_ZOO,
    get_anscheck_prompt,
    _get_hypothesis,
    _load_outputs,
    _save_outputs,
    _print_metrics_from_outputs,
)
from src.llm import LLMChatCompletion, Message


def run_llm_judge_evaluation(
    dataset_name: str,
    work_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    metric_model: str = "gpt-4o",
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    verbose: bool = True,
    concurrency: int = 10,
    reference_file: Optional[str] = None,
    evaluate_times: int = 1,
) -> str:
    """
    Run LLM-judge evaluation on the given output file.

    If output_file is not provided, it defaults to:
        {work_dir}/source_target_output.jsonl
    Returns the resolved output file path.
    """
    if dataset_name not in ("longmemeval", "toolbench"):
        raise ValueError(
            "LLM-judge evaluation supports dataset_name='longmemeval' or 'toolbench'."
        )

    if not output_file:
        if not work_dir:
            raise ValueError("Either output_file or work_dir must be provided.")
        work_dir = os.path.normpath(work_dir)
        output_file = os.path.join(work_dir, "source_target_output.jsonl")

    if not os.path.isfile(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    if concurrency is None or concurrency < 1:
        concurrency = 1

    # ── ToolBench: delegate to dedicated async evaluation ──
    if dataset_name == "toolbench":
        from src.evaluation.evaluate_toolbench import (
            MODEL_ZOO as TB_MODEL_ZOO,
            evaluate_toolbench_pass_rate_llm_judge,
            evaluate_toolbench_win_rate_llm_judge,
        )
        if metric_model not in TB_MODEL_ZOO:
            raise ValueError(
                f"metric_model must be one of {list(TB_MODEL_ZOO.keys())}, got {metric_model!r}"
            )
        print(f"[ToolBench LLM Judge] Pass rate evaluation: {output_file}")
        asyncio.run(evaluate_toolbench_pass_rate_llm_judge(
            output_file=output_file,
            metric_model=metric_model,
            api_key=openai_api_key,
            api_base=openai_api_base,
            concurrency=concurrency,
            evaluate_times=evaluate_times,
            verbose=verbose,
        ))
        if reference_file and os.path.isfile(reference_file):
            print(f"[ToolBench LLM Judge] Win rate evaluation vs: {reference_file}")
            asyncio.run(evaluate_toolbench_win_rate_llm_judge(
                output_file=output_file,
                reference_file=reference_file,
                metric_model=metric_model,
                api_key=openai_api_key,
                api_base=openai_api_base,
                concurrency=concurrency,
                evaluate_times=evaluate_times,
                verbose=verbose,
            ))
        return output_file

    # ── LongMemEval ──
    if metric_model not in MODEL_ZOO:
        raise ValueError(f"metric_model must be one of {list(MODEL_ZOO.keys())}, got {metric_model!r}")

    model_name, model_source = MODEL_ZOO[metric_model]
    if model_source == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass openai_api_key.")
        api_base = openai_api_base or os.getenv("OPENAI_API_BASE")
    else:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        api_base = openai_api_base or os.getenv("OPENAI_API_BASE") or "http://localhost:8001/v1"

    outputs = _load_outputs(output_file)
    llm = LLMChatCompletion(model_name=model_name, api_base=api_base, api_key=api_key)

    async def _judge_one(
        idx: int, out: InferenceOutputExample, sem: asyncio.Semaphore, pbar: Optional[tqdm],
    ) -> tuple[int, InferenceOutputExample]:
        meta = out.metadata
        if 'metadata' in meta:
            meta = meta['metadata']
        question_id = meta.get("question_id")
        question = meta.get("question") or out.task_main or ""
        answer = meta.get("answer") or ""
        qtype = meta.get("question_type", "")
        hypothesis = _get_hypothesis(out)

        if question_id is None or not qtype:
            if verbose:
                print(f"Warning: skipping id={out.id} (missing question_id or question_type)")
            if pbar:
                pbar.update(1)
            return idx, out

        abstention = "_abs" in str(question_id)
        prompt = get_anscheck_prompt(qtype, question, answer, hypothesis, abstention=abstention)
        messages = [Message(role="user", content=prompt)]

        async with sem:
            result = await llm.acall(
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

        if verbose:
            print(
                json.dumps(
                    {
                        "question": question[:80],
                        "answer": answer[:80],
                        "hypothesis": hypothesis[:80],
                        "autoeval_label": label,
                    },
                    indent=2,
                ),
                flush=True,
            )
        if pbar:
            pbar.update(1)

        return idx, new_out

    async def _run_all() -> List[Optional[InferenceOutputExample]]:
        sem = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=len(outputs), desc="LLM Judge eval", unit="ex") if not verbose else None

        tasks = [_judge_one(i, out, sem, pbar) for i, out in enumerate(outputs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        if pbar:
            pbar.close()

        ordered: List[Optional[InferenceOutputExample]] = [None] * len(outputs)
        for r in results:
            if isinstance(r, Exception):
                print(f"Error: {r}")
                continue
            idx, new_out = r
            ordered[idx] = new_out
        return ordered

    ordered = asyncio.run(_run_all())
    valid_outputs = [o for o in ordered if o is not None]

    _save_outputs(output_file, valid_outputs)
    print(f"Saved to {output_file}")
    _print_metrics_from_outputs(valid_outputs)
    return output_file


def _build_arg_parser() -> argparse.ArgumentParser:
    # Merge model zoos from both datasets
    from src.evaluation.evaluate_toolbench import MODEL_ZOO as TB_MODEL_ZOO
    all_models = sorted(set(MODEL_ZOO.keys()) | set(TB_MODEL_ZOO.keys()))

    p = argparse.ArgumentParser(
        description="Run LLM-judge evaluation for datasets that require separate judge (LongMemEval, ToolBench).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset_name",
        required=True,
        choices=["longmemeval", "toolbench"],
        help="Dataset name requiring LLM-judge evaluation.",
    )
    p.add_argument(
        "--work_dir",
        default=None,
        help="Work dir containing source_target_output.jsonl (used if --output_file not set).",
    )
    p.add_argument(
        "--output_file",
        default=None,
        help="Path to source_target_output.jsonl to evaluate.",
    )
    p.add_argument(
        "--metric_model",
        default="gpt-4o",
        choices=all_models,
        help="Judge model alias.",
    )
    p.add_argument(
        "--openai_api_key",
        default=None,
        help="OpenAI API key (optional; otherwise uses OPENAI_API_KEY).",
    )
    p.add_argument(
        "--openai_api_base",
        default=None,
        help="OpenAI API base (optional; otherwise uses OPENAI_API_BASE or localhost).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-example verbose logging; show progress bar instead.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent async LLM-judge calls.",
    )
    p.add_argument(
        "--reference_file",
        default=None,
        help="Reference output file for win rate comparison (ToolBench only).",
    )
    p.add_argument(
        "--evaluate_times",
        type=int,
        default=1,
        help="Number of judge evaluation rounds per sample for majority vote (ToolBench only). StableToolBench uses 3.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    verbose = not args.quiet

    out_path = run_llm_judge_evaluation(
        dataset_name=args.dataset_name,
        work_dir=args.work_dir,
        output_file=args.output_file,
        metric_model=args.metric_model,
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        verbose=verbose,
        concurrency=args.concurrency,
        reference_file=args.reference_file,
        evaluate_times=args.evaluate_times,
    )

    print("[LLM Judge Evaluation] Done.")
    print(f"[LLM Judge Evaluation] output_file={out_path}")


if __name__ == "__main__":
    main()
