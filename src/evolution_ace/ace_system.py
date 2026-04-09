"""
ACE system for evolution_ace.

Three-role architecture (Generator/Extractor, Reflector, Curator)
for offline training with optional validation-based checkpoint selection.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evolution_gepa.memory_extraction_adapter import MemoryExtractionAdapter

from .curator import Curator
from .data_processor import MemoryExtractionDataProcessor
from .extractor import Extractor
from .playbook import (
    build_prefix_counters,
    extract_playbook_bullets,
    get_playbook_stats,
    seed_playbook_from_factual_experiential,
    seed_playbook_from_seed,
    seed_playbook_from_simple,
    update_bullet_counts,
)
from .reflector import Reflector
from .utils import (
    aevaluate_test_set_safe,
    count_tokens,
    evaluate_test_set,
    extract_answer,
    log_bullet_usage,
)


class Generator:
    """Hybrid Generator: Extractor + Target Inference."""

    def __init__(self, extractor: Extractor, inference_adapter: MemoryExtractionAdapter):
        self.extractor = extractor
        self.adapter = inference_adapter

    @staticmethod
    def _build_response(bullet_ids: List[str], reward: float) -> str:
        result = "SUCCESS" if reward > 0 else "FAILED"
        return json.dumps({"final_answer": result, "bullet_ids": bullet_ids})

    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        call_id: str = "gen",
        log_dir: Optional[str] = None,
        task_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        if not task_dict or "source" not in task_dict or "target_example" not in task_dict:
            return json.dumps({"final_answer": "FAILED", "bullet_ids": []}), [], {}

        source = task_dict["source"]
        target_ex = task_dict["target_example"]
        dataset = task_dict["source_dataset"]
        memory, bullet_ids, call_info = self.extractor.extract(
            source=source,
            playbook=playbook,
            reflection=reflection,
            call_id=call_id,
            log_dir=log_dir,
        )
        source.memory = memory
        output, target_conversation = self.adapter._run_target_inference(source, target_ex, memory)
        reward = self.adapter._compute_reward(dataset, target_ex, output)
        response = self._build_response(bullet_ids, reward)
        merged_call_info = dict(call_info or {})
        merged_call_info["extracted_memory"] = memory
        merged_call_info["target_conversation"] = target_conversation
        merged_call_info["extractor_raw"] = (call_info or {}).get("raw", "")
        return response, bullet_ids, merged_call_info

    async def agenerate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        call_id: str = "gen",
        log_dir: Optional[str] = None,
        task_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        if not task_dict or "source" not in task_dict or "target_example" not in task_dict:
            return json.dumps({"final_answer": "FAILED", "bullet_ids": []}), [], {}

        source = task_dict["source"]
        target_ex = task_dict["target_example"]
        dataset = task_dict["source_dataset"]
        memory, bullet_ids, call_info = await self.extractor.aextract(
            source=source,
            playbook=playbook,
            reflection=reflection,
            call_id=call_id,
            log_dir=log_dir,
        )
        source.memory = memory
        output, target_conversation = await self.adapter._arun_target_inference(source, target_ex, memory)
        reward = self.adapter._compute_reward(dataset, target_ex, output)
        response = self._build_response(bullet_ids, reward)
        merged_call_info = dict(call_info or {})
        merged_call_info["extracted_memory"] = memory
        merged_call_info["target_conversation"] = target_conversation
        merged_call_info["extractor_raw"] = (call_info or {}).get("raw", "")
        return response, bullet_ids, merged_call_info

    async def abatch_generate_for_eval(
        self,
        task_dicts: List[Dict[str, Any]],
        playbook: str,
        max_tokens: int = 4096,
        log_dir: Optional[str] = None,
        async_max_concurrency: int = 8,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        import asyncio

        sem = asyncio.Semaphore(max(1, int(async_max_concurrency)))
        extracted: List[Tuple[str, List[str]]] = [("", []) for _ in task_dicts]

        async def _extract_one(i: int, task_dict: Dict[str, Any]) -> Optional[str]:
            if "source" not in task_dict or "target_example" not in task_dict:
                return f"Error evaluating sample {i}: missing source/target_example in task_dict"
            try:
                async with sem:
                    memory, bullet_ids, _ = await self.extractor.aextract(
                        source=task_dict["source"],
                        playbook=playbook,
                        reflection="(empty)",
                        call_id=f"test_eval_{i}",
                        log_dir=log_dir,
                    )
                extracted[i] = (memory, bullet_ids)
            except Exception as e:
                return f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"
            return None

        extraction_errors = await asyncio.gather(*[_extract_one(i, t) for i, t in enumerate(task_dicts)])
        errors = [e for e in extraction_errors if e]

        results: List[Dict[str, Any]] = []
        for i, task_dict in enumerate(task_dicts):
            if "source" not in task_dict or "target_example" not in task_dict:
                continue
            try:
                source = task_dict["source"]
                target_ex = task_dict["target_example"]
                dataset = task_dict["source_dataset"]
                target = task_dict.get("target", "SUCCESS")
                memory, bullet_ids = extracted[i]
                source.memory = memory
                output, _ = await self.adapter._arun_target_inference(source, target_ex, memory)
                reward = self.adapter._compute_reward(dataset, target_ex, output)
                final_answer = extract_answer(self._build_response(bullet_ids, reward))
                is_correct = str(final_answer).strip().upper() == str(target).strip().upper()
                results.append(
                    {
                        "index": i,
                        "final_answer": final_answer,
                        "target": target,
                        "is_correct": is_correct,
                        "success": True,
                    }
                )
            except Exception as e:
                errors.append(f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}")
        return results, errors


class ACE:
    """Main ACE orchestrator for memory extraction evolution."""

    def __init__(
        self,
        extraction_model: str = "openai/Qwen3-32B",
        inference_model: str = "openai/Qwen3-32B",
        reflector_model: str = "openai/Qwen3-32B",
        curator_model: str = "openai/Qwen3-32B",
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_async: bool = False,
        async_max_concurrency: int = 8,
        prompt_type: str = "factual-experiential",
    ):
        self.adapter = MemoryExtractionAdapter(
            extraction_model=extraction_model,
            inference_model=inference_model,
            verbose=False,
        )
        self.extractor = Extractor(model_name=extraction_model, max_tokens=max_tokens, enable_thinking=True, prompt_type=prompt_type)
        self.reflector = Reflector(model_name=reflector_model, max_tokens=max_tokens, enable_thinking=True)
        self.curator = Curator(model_name=curator_model, max_tokens=max_tokens, enable_thinking=True, prompt_type=prompt_type)
        self.generator = Generator(self.extractor, self.adapter)
        if initial_playbook:
            self.playbook = initial_playbook
        elif prompt_type == "survey":
            self.playbook = seed_playbook_from_seed()
        elif prompt_type == "simple":
            self.playbook = seed_playbook_from_simple()
        else:
            self.playbook = seed_playbook_from_factual_experiential()
        self.best_playbook = self.playbook
        self.prefix_counters = build_prefix_counters(self.playbook)
        self.max_tokens = max_tokens
        self.use_async = bool(use_async)
        self.async_max_concurrency = max(1, int(async_max_concurrency))

    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "num_epochs": config.get("num_epochs", 1),
            "max_num_rounds": config.get("max_num_rounds", 3),
            "curator_frequency": config.get("curator_frequency", 1),
            "eval_steps": config.get("eval_steps", 100),
            "save_steps": config.get("save_steps", 50),
            "token_budget": config.get("playbook_token_budget", 80000),
            "task_name": config.get("task_name", "memory_extraction"),
            "save_dir": config.get("save_dir", "./results"),
            "test_workers": config.get("test_workers", 1),
        }

    def _setup_paths(self, save_dir: str, task_name: str) -> Tuple[str, str, str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"ace_run_{timestamp}_{task_name}_offline"
        save_path = os.path.join(save_dir, run_folder)
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)
        return save_path, usage_log_path, playbook_dir, log_dir

    def _sanitize_reflection_for_extractor(self, reflection: str) -> str:
        if not reflection or reflection == "(empty)":
            return reflection
        try:
            parsed = json.loads(reflection)
        except (json.JSONDecodeError, TypeError):
            return reflection
        if not isinstance(parsed, dict) or "input_summary" not in parsed:
            return reflection
        parsed.pop("input_summary", None)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    def _build_train_tracking(self, final_answer: str, is_correct: bool) -> Dict[str, Any]:
        return {
            "final_answer": final_answer,
            "is_correct": is_correct,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook),
        }

    def _train_single_sample(
        self,
        task_dict: Dict[str, Any],
        data_processor: MemoryExtractionDataProcessor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: Optional[str],
        log_dir: Optional[str],
        config_params: Dict[str, Any],
        total_samples: int,
    ) -> Tuple[str, str, Dict[str, Any]]:
        max_num_rounds = config_params["max_num_rounds"]
        curator_frequency = config_params["curator_frequency"]
        token_budget = config_params["token_budget"]
        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "SUCCESS")

        gen_response, bullet_ids, gen_call_info = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir,
            task_dict=task_dict,
        )
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer
        log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids, playbook=self.playbook, is_correct=is_correct)
        tracking_dict = {"pre_train_result": self._build_train_tracking(final_answer, is_correct)}
        reflection_content = "(empty)"

        if not is_correct:
            for round_num in range(max_num_rounds):
                playbook_bullets = extract_playbook_bullets(self.playbook, bullet_ids)
                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    source_conversation=question,
                    reasoning_trace=gen_call_info.get("reasoning", ""),
                    extracted_memory=gen_call_info.get("extracted_memory", ""),
                    target_conversation=gen_call_info.get("target_conversation", ""),
                    environment_feedback="Target inference failed",
                    bullets_used=playbook_bullets,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir,
                )
                if bullet_tags:
                    self.playbook = update_bullet_counts(self.playbook, bullet_tags)
                reflection_for_extractor = self._sanitize_reflection_for_extractor(reflection_content)
                gen_response, bullet_ids, gen_call_info = self.generator.generate(
                    question=question,
                    playbook=self.playbook,
                    context=context,
                    reflection=reflection_for_extractor,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir,
                    task_dict=task_dict,
                )
                final_answer = extract_answer(gen_response)
                if data_processor.answer_is_correct(final_answer, target):
                    break
        else:
            playbook_bullets = extract_playbook_bullets(self.playbook, bullet_ids)
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                source_conversation=question,
                reasoning_trace=gen_call_info.get("reasoning", ""),
                extracted_memory=gen_call_info.get("extracted_memory", ""),
                target_conversation=gen_call_info.get("target_conversation", ""),
                environment_feedback="Target inference succeeded",
                bullets_used=playbook_bullets,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir,
            )
            if bullet_tags:
                self.playbook = update_bullet_counts(self.playbook, bullet_tags)

        if step % curator_frequency == 0:
            stats = get_playbook_stats(self.playbook)
            self.playbook, self.prefix_counters, _, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                call_id=step_id,
                log_dir=log_dir,
                prefix_counters=self.prefix_counters,
            )

        gen_response, _, _ = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir,
            task_dict=task_dict,
        )
        final_answer = extract_answer(gen_response)
        tracking_dict["post_train_result"] = self._build_train_tracking(
            final_answer, data_processor.answer_is_correct(final_answer, target)
        )
        return pre_train_answer, final_answer, tracking_dict

    async def _atrain_single_sample(
        self,
        task_dict: Dict[str, Any],
        data_processor: MemoryExtractionDataProcessor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: Optional[str],
        log_dir: Optional[str],
        config_params: Dict[str, Any],
        total_samples: int,
    ) -> Tuple[str, str, Dict[str, Any]]:
        max_num_rounds = config_params["max_num_rounds"]
        curator_frequency = config_params["curator_frequency"]
        token_budget = config_params["token_budget"]
        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "SUCCESS")

        gen_response, bullet_ids, gen_call_info = await self.generator.agenerate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir,
            task_dict=task_dict,
        )
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer
        log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids, playbook=self.playbook, is_correct=is_correct)
        tracking_dict = {"pre_train_result": self._build_train_tracking(final_answer, is_correct)}
        reflection_content = "(empty)"

        if not is_correct:
            for round_num in range(max_num_rounds):
                playbook_bullets = extract_playbook_bullets(self.playbook, bullet_ids)
                reflection_content, bullet_tags, _ = await self.reflector.areflect(
                    source_conversation=question,
                    reasoning_trace=gen_call_info.get("reasoning", ""),
                    extracted_memory=gen_call_info.get("extracted_memory", ""),
                    target_conversation=gen_call_info.get("target_conversation", ""),
                    environment_feedback="Target inference failed",
                    bullets_used=playbook_bullets,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir,
                )
                if bullet_tags:
                    self.playbook = update_bullet_counts(self.playbook, bullet_tags)
                reflection_for_extractor = self._sanitize_reflection_for_extractor(reflection_content)
                gen_response, bullet_ids, gen_call_info = await self.generator.agenerate(
                    question=question,
                    playbook=self.playbook,
                    context=context,
                    reflection=reflection_for_extractor,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir,
                    task_dict=task_dict,
                )
                final_answer = extract_answer(gen_response)
                if data_processor.answer_is_correct(final_answer, target):
                    break
        else:
            playbook_bullets = extract_playbook_bullets(self.playbook, bullet_ids)
            reflection_content, bullet_tags, _ = await self.reflector.areflect(
                source_conversation=question,
                reasoning_trace=gen_call_info.get("reasoning", ""),
                extracted_memory=gen_call_info.get("extracted_memory", ""),
                target_conversation=gen_call_info.get("target_conversation", ""),
                environment_feedback="Target inference succeeded",
                bullets_used=playbook_bullets,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir,
            )
            if bullet_tags:
                self.playbook = update_bullet_counts(self.playbook, bullet_tags)

        if step % curator_frequency == 0:
            stats = get_playbook_stats(self.playbook)
            self.playbook, self.prefix_counters, _, _ = await self.curator.acurate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                call_id=step_id,
                log_dir=log_dir,
                prefix_counters=self.prefix_counters,
            )

        gen_response, _, _ = await self.generator.agenerate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir,
            task_dict=task_dict,
        )
        final_answer = extract_answer(gen_response)
        tracking_dict["post_train_result"] = self._build_train_tracking(
            final_answer, data_processor.answer_is_correct(final_answer, target)
        )
        return pre_train_answer, final_answer, tracking_dict

    def _save_step_outputs(
        self,
        save_path: str,
        playbook_dir: Optional[str],
        epoch: int,
        step: int,
        save_steps: int,
        has_validation: bool,
        best_accuracy: float,
        results: List[Dict[str, Any]],
        val_logs: List[Dict[str, Any]],
    ) -> None:
        if step % save_steps == 0 and playbook_dir:
            p = os.path.join(playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.playbook)
        if has_validation:
            with open(os.path.join(save_path, "train_results.json"), "w", encoding="utf-8") as f:
                json.dump({"has_validation": True, "best_validation_accuracy": best_accuracy, "results": results}, f, indent=2)
            with open(os.path.join(save_path, "val_results.json"), "w", encoding="utf-8") as f:
                json.dump(val_logs, f, indent=2)

    def _finalize_training_outputs(
        self,
        save_path: str,
        has_validation: bool,
        best_accuracy: float,
        results: List[Dict[str, Any]],
        val_logs: List[Dict[str, Any]],
        pre_train_post_train_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        with open(os.path.join(save_path, "pre_train_post_train_results.json"), "w", encoding="utf-8") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        normalized_best = best_accuracy if has_validation and best_accuracy >= 0.0 else None
        with open(os.path.join(save_path, "train_results.json"), "w", encoding="utf-8") as f:
            json.dump({"has_validation": has_validation, "best_validation_accuracy": normalized_best, "results": results}, f, indent=2)
        if has_validation:
            with open(os.path.join(save_path, "val_results.json"), "w", encoding="utf-8") as f:
                json.dump(val_logs, f, indent=2)
            with open(os.path.join(save_path, "best_playbook.txt"), "w", encoding="utf-8") as f:
                f.write(self.best_playbook)
        with open(os.path.join(save_path, "final_playbook.txt"), "w", encoding="utf-8") as f:
            f.write(self.playbook)
        return {"has_validation": has_validation, "best_validation_accuracy": normalized_best}

    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: Optional[List[Dict[str, Any]]],
        data_processor: MemoryExtractionDataProcessor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: Optional[str],
        playbook_dir: Optional[str],
        log_dir: Optional[str],
    ) -> Dict[str, Any]:
        cfg = self._extract_config_params(config)
        num_epochs, eval_steps, save_steps = cfg["num_epochs"], cfg["eval_steps"], cfg["save_steps"]
        has_validation = bool(val_samples)
        best_accuracy = -1.0
        results: List[Dict[str, Any]] = []
        val_logs: List[Dict[str, Any]] = []
        pre_train_post_train_results: List[Dict[str, Any]] = []
        self.best_playbook = self.playbook

        for epoch in range(1, num_epochs + 1):
            epoch_answers_pre: List[str] = []
            epoch_targets_pre: List[str] = []
            epoch_answers_post: List[str] = []
            epoch_targets_post: List[str] = []
            for step, task_dict in enumerate(train_samples, start=1):
                tgt = task_dict.get("target", "SUCCESS")
                pre, post, tracking = self._train_single_sample(
                    task_dict, data_processor, f"train_e_{epoch}_s_{step}", epoch, step, usage_log_path, log_dir, cfg, len(train_samples)
                )
                epoch_answers_pre.append(pre)
                epoch_targets_pre.append(tgt)
                epoch_answers_post.append(post)
                epoch_targets_post.append(tgt)
                pre_train_post_train_results.append({"epoch": epoch, "step": step, "target": tgt, **tracking})
                if has_validation and step % eval_steps == 0:
                    pre_acc = data_processor.evaluate_accuracy(epoch_answers_pre, epoch_targets_pre)
                    post_acc = data_processor.evaluate_accuracy(epoch_answers_post, epoch_targets_post)
                    val_results, val_error_log = evaluate_test_set(
                        data_processor, self.generator, self.playbook, val_samples, self.max_tokens, log_dir, max_workers=cfg["test_workers"]
                    )
                    val_acc = val_results.get("accuracy", 0.0)
                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        self.best_playbook = self.playbook
                    results.append(
                        {
                            "epoch": epoch,
                            "step": step,
                            "train_result": {"pre_train_accuracy": pre_acc, "post_train_accuracy": post_acc},
                            "val_result": val_results,
                            "playbook_num_tokens": count_tokens(self.playbook),
                            "playbook_length": len(self.playbook),
                            "playbook_stats": get_playbook_stats(self.playbook),
                            "best_validation_accuracy_so_far": best_accuracy,
                        }
                    )
                    val_logs.append({"epoch": epoch, "step": step, "val_results": val_results, "val_error_log": val_error_log})
                self._save_step_outputs(save_path, playbook_dir, epoch, step, save_steps, has_validation, best_accuracy, results, val_logs)
        return self._finalize_training_outputs(save_path, has_validation, best_accuracy, results, val_logs, pre_train_post_train_results)

    async def _aoffline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: Optional[List[Dict[str, Any]]],
        data_processor: MemoryExtractionDataProcessor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: Optional[str],
        playbook_dir: Optional[str],
        log_dir: Optional[str],
    ) -> Dict[str, Any]:
        cfg = self._extract_config_params(config)
        num_epochs, eval_steps, save_steps = cfg["num_epochs"], cfg["eval_steps"], cfg["save_steps"]
        has_validation = bool(val_samples)
        best_accuracy = -1.0
        results: List[Dict[str, Any]] = []
        val_logs: List[Dict[str, Any]] = []
        pre_train_post_train_results: List[Dict[str, Any]] = []
        self.best_playbook = self.playbook

        for epoch in range(1, num_epochs + 1):
            epoch_answers_pre: List[str] = []
            epoch_targets_pre: List[str] = []
            epoch_answers_post: List[str] = []
            epoch_targets_post: List[str] = []
            for step, task_dict in enumerate(train_samples, start=1):
                tgt = task_dict.get("target", "SUCCESS")
                pre, post, tracking = await self._atrain_single_sample(
                    task_dict, data_processor, f"train_e_{epoch}_s_{step}", epoch, step, usage_log_path, log_dir, cfg, len(train_samples)
                )
                epoch_answers_pre.append(pre)
                epoch_targets_pre.append(tgt)
                epoch_answers_post.append(post)
                epoch_targets_post.append(tgt)
                pre_train_post_train_results.append({"epoch": epoch, "step": step, "target": tgt, **tracking})
                if has_validation and step % eval_steps == 0:
                    pre_acc = data_processor.evaluate_accuracy(epoch_answers_pre, epoch_targets_pre)
                    post_acc = data_processor.evaluate_accuracy(epoch_answers_post, epoch_targets_post)
                    val_results, val_error_log = await aevaluate_test_set_safe(
                        data_processor, self.generator, self.playbook, val_samples, self.max_tokens, log_dir,
                        async_max_concurrency=self.async_max_concurrency,
                    )
                    val_acc = val_results.get("accuracy", 0.0)
                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        self.best_playbook = self.playbook
                    results.append(
                        {
                            "epoch": epoch,
                            "step": step,
                            "train_result": {"pre_train_accuracy": pre_acc, "post_train_accuracy": post_acc},
                            "val_result": val_results,
                            "playbook_num_tokens": count_tokens(self.playbook),
                            "playbook_length": len(self.playbook),
                            "playbook_stats": get_playbook_stats(self.playbook),
                            "best_validation_accuracy_so_far": best_accuracy,
                        }
                    )
                    val_logs.append({"epoch": epoch, "step": step, "val_results": val_results, "val_error_log": val_error_log})
                self._save_step_outputs(save_path, playbook_dir, epoch, step, save_steps, has_validation, best_accuracy, results, val_logs)
        return self._finalize_training_outputs(save_path, has_validation, best_accuracy, results, val_logs, pre_train_post_train_results)

    def run(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor: Optional[MemoryExtractionDataProcessor] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if train_samples is None:
            raise ValueError("train_samples is required")
        config = config or {}
        task_name = self._extract_config_params(config)["task_name"]
        save_dir = self._extract_config_params(config)["save_dir"]
        save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(save_dir, task_name)
        with open(os.path.join(save_path, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({"task_name": task_name, "mode": "offline", "config": config}, f, indent=2)
        proc = data_processor or MemoryExtractionDataProcessor()
        training_results = self._offline_train(train_samples, val_samples, proc, config, save_path, usage_log_path, playbook_dir, log_dir)
        results = {"training_results": training_results, "has_validation": bool(val_samples), "best_validation_accuracy": training_results.get("best_validation_accuracy")}
        with open(os.path.join(save_path, "final_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results

    async def arun(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor: Optional[MemoryExtractionDataProcessor] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if train_samples is None:
            raise ValueError("train_samples is required")
        config = config or {}
        task_name = self._extract_config_params(config)["task_name"]
        save_dir = self._extract_config_params(config)["save_dir"]
        save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(save_dir, task_name)
        with open(os.path.join(save_path, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({"task_name": task_name, "mode": "offline_async", "config": config}, f, indent=2)
        proc = data_processor or MemoryExtractionDataProcessor()
        training_results = await self._aoffline_train(train_samples, val_samples, proc, config, save_path, usage_log_path, playbook_dir, log_dir)
        results = {"training_results": training_results, "has_validation": bool(val_samples), "best_validation_accuracy": training_results.get("best_validation_accuracy")}
        with open(os.path.join(save_path, "final_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results
