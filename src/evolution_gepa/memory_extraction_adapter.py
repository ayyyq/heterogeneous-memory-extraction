"""
GEPA Adapter for Memory Extraction Prompt Optimization.

This adapter optimizes the memory extraction prompt by evaluating how well
extracted memories help solve target tasks.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import TypedDict, Any

from gepa.core.adapter import GEPAAdapter, EvaluationBatch

from src.data_processing.generators.gmemory_generator import GMM_DATASETS
from src.data_processing.generators.dynamic_cheatsheet_generator import DC_TASKS
from src.data_processing.schemas import SourceExample, InferenceInputExample, InferenceOutputExample
from src.evaluation import evaluate_dynamic_cheatsheet as dc_eval
from src.evaluation.evaluate_bigcodebench import evaluate_bigcodebench_one
from src.extraction.llm_extractor import LLMExtractor
from src.inference import get_inference_engine
from .utils import build_extraction_prompt

SUPPORTED_DATASETS = {
    "membench-high",
    "membench-low",
    "personamem-v2-val",
    "prefeval-mcq",
    "alfworld",
    "fever",
    "pddl",
    "babyai",
    "scienceworld",
    "GameOf24",
    "AIME_2024",
    "AIME_2025",
    "AIME_2020_2024",
    "MMLU_Pro_Physics",
    "MMLU_Pro_Engineering",
    "MathEquationBalancer",
    "bigcodebench",
}


# Type definitions
class MemoryExtractionDataInst(TypedDict):
    """Input data for memory extraction evaluation."""
    source: SourceExample
    target: InferenceInputExample
    source_dataset: str  # For tracking dataset identity across mixed minibatches.


class MemoryEvaluationTrace(TypedDict):
    """Trace of a single memory extraction and evaluation."""
    source_id: int
    target_id: int
    source_dataset: str
    source_conversation: str
    extraction_prompt: dict[str, str]  # {system_prompt, user_prompt}
    extracted_memory: str
    target_inference_trajectory: str   # 使用extracted memory进行inference时的完整对话历史
    score: float                       # 0 or 1 (target inference是否成功)


class MemoryExtractionOutput(TypedDict):
    """Output from evaluating one source-target pair."""
    success: bool
    extracted_memory: str


class MemoryExtractionAdapter(GEPAAdapter[MemoryExtractionDataInst, MemoryEvaluationTrace, MemoryExtractionOutput]):
    """
    GEPA Adapter for optimizing memory extraction prompts.
    
    This adapter evaluates candidate extraction prompts by:
    1. Extracting memory from source conversations using the candidate prompt
    2. Applying extracted memory to target tasks
    3. Measuring target task success as the quality metric
    """
    
    def __init__(
        self,
        extraction_model: str = "openai/Qwen3-32B",
        inference_model: str = "openai/Qwen3-32B",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        enable_thinking: bool = True,
        verbose: bool = False,
        num_workers: int = 1,
        use_async: bool = False,
        async_max_concurrency: int = 8,
    ):
        """
        Initialize the adapter.
        
        Args:
            extraction_model: Model for memory extraction
            inference_model: Model for target inference
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Max tokens for extraction
            enable_thinking: Enable thinking mode for extraction
            verbose: Print detailed logs
        """
        self.extraction_model = extraction_model
        self.inference_model = inference_model
        self.temperature = temperature
        self.top_p = top_p
        # Auto-adjust defaults for Gemini models
        if "gemini" in extraction_model.lower():
            self.temperature = 1.0
            self.top_p = 0.95
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.verbose = verbose
        self.num_workers = max(1, int(num_workers))
        self.use_async = bool(use_async)
        self.async_max_concurrency = max(1, int(async_max_concurrency))
        if self.use_async and self.num_workers > 1:
            raise ValueError(
                "MemoryExtractionAdapter: use_async=True cannot be combined with num_workers>1."
            )

        # Extraction layer (default shared extractor for serial path).
        self._llm_extractor = self._build_extractor()

        # Cache inference engines per normalized dataset name (serial inference path).
        self._engines: dict[str, Any] = {}

    def _build_extractor(self) -> LLMExtractor:
        return LLMExtractor(
            prompt_type="general",
            model_name=self.extraction_model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )

    @staticmethod
    def _emit_progress(
        stage: str,
        done: int,
        total: int,
        score: float | None = None,
    ) -> None:
        if total <= 0:
            return
        if score is None:
            print(f"[Adapter][{stage}] {done}/{total}", flush=True)
        else:
            print(f"[Adapter][{stage}] {done}/{total} score={score:.1f}", flush=True)
        
    def _extract_memory_with_prompt(
        self,
        source: SourceExample,
        system_prompt: str,
        user_prompt: str,
        extractor: LLMExtractor | None = None,
    ) -> str:
        """
        Extract memory from source using given prompts.
        
        Args:
            source: Source example
            system_prompt: System prompt for extraction
            user_prompt: User prompt for extraction
            
        Returns:
            Extracted memory string
        """
        # Delegate to the shared LLMExtractor so that:
        # - prompts/response traces land in source.metadata["memory_extraction"]
        # - downstream debugging is consistent across pipelines.
        use_extractor = extractor or self._llm_extractor
        try:
            return use_extractor.extract_with_prompts(
                source=source,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            if self.verbose:
                print(f"[Adapter] Error extracting memory: {e}")
            return ""

    async def _aextract_memory_with_prompt(
        self,
        source: SourceExample,
        system_prompt: str,
        user_prompt: str,
        extractor: LLMExtractor | None = None,
    ) -> str:
        use_extractor = extractor or self._llm_extractor
        try:
            return await use_extractor.aextract_with_prompts(
                source=source,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            if self.verbose:
                print(f"[Adapter] Async error extracting memory: {e}")
            return ""

    def _normalize_dataset_for_engine(self, dataset_name: str) -> str:
        """
        Map dataset_name to the inference-engine dataset key.
        Must stay aligned with run_evaluation._normalize_dataset_for_engine.
        """
        if dataset_name in ("alfworld", "fever", "pddl"):
            return dataset_name
        if dataset_name in ("membench", "membench-high", "membench-low"):
            return "membench"
        if dataset_name in ("personamem-v2-val",):
            return "personamem-v2"
        if dataset_name == "prefeval-mcq":
            return "prefeval-mcq"
        if dataset_name in DC_TASKS:
            return "dynamic_cheatsheet"
        return dataset_name  # babyai, scienceworld pass through unchanged

    def _get_engine(self, dataset_name: str):
        norm_name = self._normalize_dataset_for_engine(dataset_name)
        engine = self._engines.get(norm_name)
        if engine is None:
            few_shots_num = 1 if norm_name in ("babyai", "scienceworld") else 0
            engine = get_inference_engine(
                dataset_name=norm_name,
                model_name=self.inference_model,
                few_shots_num=few_shots_num,
                enable_thinking=False,
                print_adapter_messages=False,
            )
            self._engines[norm_name] = engine
        return engine

    def _compute_reward(
        self,
        dataset_name: str,
        target: InferenceInputExample,
        output: InferenceOutputExample,
    ) -> float:
        """
        Compute per-example reward using the same task-specific logic
        as the standard evaluation pipeline.
        """
        LABEL_BASED_DATASETS = GMM_DATASETS | {
            "membench", "membench-high", "membench-low",
            "babyai", "scienceworld",
            "personamem-v2-val", "prefeval-mcq",
        }
        if dataset_name in LABEL_BASED_DATASETS:
            return 1.0 if bool(output.label) else 0.0

        if dataset_name == "bigcodebench":
            try:
                res = evaluate_bigcodebench_one(output)
                if self.verbose:
                    status = res.get("status")
                    err = res.get("error")
                    if err:
                        err = str(err)[:200]
                    print(f"[Adapter] BigCodeBench eval status={status}, error={err or ''}")
                return 1.0 if bool(res.get("label")) else 0.0
            except Exception as e:
                if self.verbose:
                    print(f"[Adapter] BigCodeBench reward computation failed: {e}")
                return 0.0

        # Dynamic-cheatsheet tasks reuse evaluate_dynamic_cheatsheet core utilities.
        if dataset_name in DC_TASKS:
            meta = output.metadata
            final_answer = (meta.get("final_answer") or "").strip()
            formatted_input = meta.get("input", "") or ""
            target_gt = meta.get("target", "") or ""
            raw_input = output.task_main or ""

            if dataset_name == "GameOf24":
                label = dc_eval.eval_for_GameOf24(raw_input, final_answer)
            elif dataset_name in ("AIME_2024", "AIME_2025", "AIME_2020_2024"):
                label = dc_eval.eval_for_exact_matching_with_no_punctuation(
                    final_answer.lower(),
                    target_gt.lower(),
                )
            elif dataset_name in ("GPQA_Diamond", "MMLU_Pro_Physics", "MMLU_Pro_Engineering"):
                label = dc_eval.eval_for_multiple_choice(
                    formatted_input,
                    final_answer,
                    target_gt,
                )
            elif dataset_name == "MathEquationBalancer":
                label = dc_eval.eval_equation_balancer(None, final_answer, target_gt)
            else:
                label = False
            return 1.0 if label else 0.0

        # Fallback: rely on output.label if present.
        return 1.0 if bool(output.label) else 0.0

    def _run_target_inference(
        self,
        source: SourceExample,
        target: InferenceInputExample,
        memory: str,
        need_trajectory: bool = True,
    ) -> tuple[InferenceOutputExample, str]:
        """
        Run inference on target with extracted memory using the unified
        BaseInferenceEngine + dataset adapters.
        
        Returns:
            Tuple of (InferenceOutputExample, trajectory_text)
        """
        engine = self._get_engine(target.source)
        output = engine.run_one(target, memory=memory, use_memory=True)
        trajectory = output.get_conversation_text() if need_trajectory else ""
        return output, trajectory

    async def _arun_target_inference(
        self,
        source: SourceExample,
        target: InferenceInputExample,
        memory: str,
        need_trajectory: bool = True,
    ) -> tuple[InferenceOutputExample, str]:
        engine = self._get_engine(target.source)
        output = await engine.run_one_async(target, memory=memory, use_memory=True)
        trajectory = output.get_conversation_text() if need_trajectory else ""
        return output, trajectory
    
    def _build_eval_records(
        self,
        batch: list[MemoryExtractionDataInst],
        candidate: dict[str, str],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for data_inst in batch:
            source = data_inst["source"]
            target = data_inst["target"]
            dataset_name = data_inst["source_dataset"]
            if dataset_name not in SUPPORTED_DATASETS:
                raise ValueError(
                    f"Unsupported dataset in MemoryExtractionAdapter.evaluate: {dataset_name}. "
                    f"Supported datasets: {sorted(SUPPORTED_DATASETS)}"
                )
            conversation = source.get_conversation_text()
            system_prompt, user_prompt = build_extraction_prompt(candidate, conversation)
            records.append(
                {
                    "source": source,
                    "target": target,
                    "dataset_name": dataset_name,
                    "conversation": conversation,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
        return records

    def _evaluate_sync(
        self,
        batch: list[MemoryExtractionDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
        progress_callback: bool = False,
    ) -> EvaluationBatch[MemoryEvaluationTrace, MemoryExtractionOutput]:
        """
        Evaluate a candidate extraction prompt on a batch of source-target pairs.
        
        For each (source, target) pair:
        1. Use candidate prompt to extract memory from source
        2. Apply memory to target inference
        3. Get target success (0/1) as score
        
        Args:
            batch: List of source-target pairs
            candidate: Dict with 'memory_taxonomy' and 'general_guidelines'
            capture_traces: Whether to capture detailed traces for reflection
            
        Returns:
            EvaluationBatch with outputs, scores, and optional traces
        """
        outputs: list[MemoryExtractionOutput] = []
        scores: list[float] = []
        trajectories: list[MemoryEvaluationTrace] | None = [] if capture_traces else None

        records = self._build_eval_records(batch, candidate)

        # Stage A: extraction (parallel only for extraction path)
        extracted_memories: list[str] = [""] * len(records)
        if self.num_workers <= 1 or len(records) <= 1:
            for i, rec in enumerate(records):
                extracted_memories[i] = self._extract_memory_with_prompt(
                    rec["source"], rec["system_prompt"], rec["user_prompt"]
                )
                if progress_callback:
                    self._emit_progress("extraction", i + 1, len(records))
        else:
            thread_local = threading.local()
            done_extract = 0

            def _get_thread_extractor() -> LLMExtractor:
                extractor = getattr(thread_local, "extractor", None)
                if extractor is None:
                    extractor = self._build_extractor()
                    thread_local.extractor = extractor
                return extractor

            def _run_extract(idx: int, rec: dict[str, Any]) -> tuple[int, str]:
                memory = self._extract_memory_with_prompt(
                    rec["source"],
                    rec["system_prompt"],
                    rec["user_prompt"],
                    extractor=_get_thread_extractor(),
                )
                return idx, memory

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_run_extract, i, rec): i for i, rec in enumerate(records)}
                for future in as_completed(futures):
                    idx, memory = future.result()
                    extracted_memories[idx] = memory
                    done_extract += 1
                    if progress_callback:
                        self._emit_progress("extraction", done_extract, len(records))

        # Stage B: inference + reward (serial to avoid env/state race across datasets)
        for i, rec in enumerate(records):
            source = rec["source"]
            target = rec["target"]
            dataset_name = rec["dataset_name"]
            conversation = rec["conversation"]
            extracted_memory = extracted_memories[i]

            # Keep source.memory in sync in main thread only.
            source.memory = extracted_memory

            output_ex, target_trajectory = self._run_target_inference(
                source,
                target,
                extracted_memory,
                need_trajectory=capture_traces,
            )
            score = self._compute_reward(dataset_name, target, output_ex)
            scores.append(score)

            outputs.append(
                {
                    "success": bool(score > 0.0),
                    "extracted_memory": extracted_memory,
                }
            )

            if capture_traces:
                trajectories.append(
                    {
                        "source_conversation": conversation,
                        "extraction_prompt": {
                            "system_prompt": rec["system_prompt"],
                            "user_prompt": rec["user_prompt"],
                        },
                        "extraction_output": extracted_memory,
                        "target_inference_trajectory": target_trajectory,
                        "score": score,
                    }
                )

            if self.verbose:
                print(f"[Adapter] Evaluated source {source.id} -> target {target.id}: score={score}")
            if progress_callback:
                self._emit_progress("inference", i + 1, len(records), score)
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )

    async def aevaluate(
        self,
        batch: list[MemoryExtractionDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
        progress_callback: bool = False,
    ) -> EvaluationBatch[MemoryEvaluationTrace, MemoryExtractionOutput]:
        """
        Async evaluation path:
        - extraction: async concurrent with semaphore limits
        - inference + reward: async serial to avoid env/state races
        """
        outputs: list[MemoryExtractionOutput] = []
        scores: list[float] = []
        trajectories: list[MemoryEvaluationTrace] | None = [] if capture_traces else None

        records = self._build_eval_records(batch, candidate)
        extracted_memories: list[str] = [""] * len(records)

        sem = asyncio.Semaphore(self.async_max_concurrency)

        async def _extract_one(idx: int, rec: dict[str, Any]) -> tuple[int, str]:
            async with sem:
                memory = await self._aextract_memory_with_prompt(
                    rec["source"],
                    rec["system_prompt"],
                    rec["user_prompt"],
                )
                return idx, memory

        extraction_tasks = [_extract_one(i, rec) for i, rec in enumerate(records)]
        extraction_results = await asyncio.gather(*extraction_tasks)
        done_extract = 0
        for idx, memory in extraction_results:
            extracted_memories[idx] = memory
            done_extract += 1
            if progress_callback:
                self._emit_progress("extraction", done_extract, len(records))

        # Stage B: inference + reward (serial to avoid env/state race across datasets)
        for i, rec in enumerate(records):
            source = rec["source"]
            target = rec["target"]
            dataset_name = rec["dataset_name"]
            conversation = rec["conversation"]
            extracted_memory = extracted_memories[i]

            source.memory = extracted_memory
            output_ex, target_trajectory = await self._arun_target_inference(
                source,
                target,
                extracted_memory,
                need_trajectory=capture_traces,
            )
            score = self._compute_reward(dataset_name, target, output_ex)
            scores.append(score)

            outputs.append(
                {
                    "success": bool(score > 0.0),
                    "extracted_memory": extracted_memory,
                }
            )

            if capture_traces:
                trajectories.append(
                    {
                        "source_conversation": conversation,
                        "extraction_prompt": {
                            "system_prompt": rec["system_prompt"],
                            "user_prompt": rec["user_prompt"],
                        },
                        "extraction_output": extracted_memory,
                        "target_inference_trajectory": target_trajectory,
                        "score": score,
                    }
                )

            if self.verbose:
                print(f"[Adapter] Evaluated source {source.id} -> target {target.id}: score={score}")
            if progress_callback:
                self._emit_progress("inference", i + 1, len(records), score)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def evaluate(
        self,
        batch: list[MemoryExtractionDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
        progress_callback: bool = False,
    ) -> EvaluationBatch[MemoryEvaluationTrace, MemoryExtractionOutput]:
        if not self.use_async:
            return self._evaluate_sync(
                batch,
                candidate,
                capture_traces=capture_traces,
                progress_callback=progress_callback,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aevaluate(
                    batch,
                    candidate,
                    capture_traces=capture_traces,
                    progress_callback=progress_callback,
                )
            )

        raise RuntimeError(
            "MemoryExtractionAdapter.evaluate() cannot call asyncio.run() inside an active event loop. "
            "Use `await MemoryExtractionAdapter.aevaluate(...)` instead."
        )
    
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[MemoryEvaluationTrace, MemoryExtractionOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset from evaluation traces.
        
        For each trace, create a record with:
        - Inputs: source conversation (truncated) and target description
        - Generated Outputs: extracted memory
        - Feedback: success/failure message
        
        Args:
            candidate: Current candidate
            eval_batch: Evaluation batch with traces
            components_to_update: Components to update (e.g., ['memory_taxonomy', 'general_guidelines'])
            
        Returns:
            Dict mapping component name -> list of reflective records
        """
        if eval_batch.trajectories is None:
            raise ValueError("Traces must be captured to build reflective dataset")
        
        records = []
        
        for trace in eval_batch.trajectories:
            # Get source and target info
            score = trace['score']
            source_conv = trace.get('source_conversation', '')
            # Truncate source conversation for inputs (first 500 chars) to keep reflection prompts compact.
            source_desc = source_conv[:500]
            if len(source_conv) > 500:
                source_desc += "\n...[truncated]..."
            
            # Get target description
            target_trajectory = trace['target_inference_trajectory']
            target_desc = target_trajectory[:500]
            if len(target_trajectory) > 500:
                target_desc += "\n...[truncated]..."
            
            # Build feedback
            if score == 1.0:
                # Success - simple feedback
                feedback = f"""The assistant's response is retrieved and provided as extracted memory to assist a target request.

<target_conversation>
{target_desc}
</target_conversation>

Outcome: The target request was completed successfully.

Your job is NOT to rewrite this specific memory. Instead, infer what kinds of memories tend to help in situations like this, and propose improvements to the memory extraction guidelines.

Reflect on:
- Does the extracted memory assist the target request, and how does it help?
- What characteristics make this memory useful (e.g., level of abstraction, specificity, type of information, or transferability)?
- What generalizable insights can be drawn for improving future memory extraction guidelines?"""
            else:
                feedback = f"""The assistant's response is retrieved and provided as extracted memory to assist a target request.

<target_conversation>
{target_desc}
</target_conversation>

Outcome: The target request was not completed successfully.

Do NOT focus on fixing the memory for this single example. Instead, diagnose what the memory extraction guidelines failed to capture, and propose improvements to the guidelines.

Reflect on:
- What kind of memory extracted from the input could have better supported completing the target request?
- What was missing or ineffective in the extracted memory (e.g., missing key information, too specific, too vague, or not transferable)?
- Based on this, how should the memory extraction guidelines be revised?"""
            
            # Create reflective record. We expose both the original source
            # conversation (truncated) and the downstream behaviour with
            # this memory, so the reflection LM can reason about *why*
            # a particular memory did or did not help.
            records.append(
                {
                    "Input": source_desc,
                    "Output": trace['extraction_output'],
                    "Feedback": feedback,
                }
            )
        
        # Return same records for all components to update
        # (they both benefit from seeing all the examples)
        return {comp: records for comp in components_to_update}
