"""
Curator for evolution_ace.

Mirrors ACE Curator: adds new bullets to playbook based on reflection.
"""

import json
import sys
import os
from typing import Tuple, List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm import LLMChatCompletion, Message

from .playbook import (
    extract_json_from_text,
    apply_curator_operations,
    get_playbook_stats,
)
from .prompts.curator import CURATOR_PROMPT, CURATOR_PROMPT_SEED
from .logger import log_curator_operation_diff, log_curator_failure


class Curator:
    """
    Curator agent that updates playbook based on reflection.
    Analogous to ACE Curator.
    """

    def __init__(
        self,
        model_name: str = "openai/Qwen3-32B",
        max_tokens: int = 4096,
        enable_thinking: bool = True,
        prompt_type: str = "factual-experiential",
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.prompt_type = prompt_type
        self._curator_prompt = CURATOR_PROMPT_SEED if prompt_type in ("survey", "simple") else CURATOR_PROMPT
        self._forbidden_ops: set = (
            {"UPDATE_TYPE_DEFINITION", "UPDATE_TYPE_WHEN_TO_APPLY"}
            if prompt_type in ("survey", "simple")
            else set()
        )
        self._require_def_when = (prompt_type not in ("survey", "simple"))
        if 'qwen3' in model_name.lower():
            if self.enable_thinking:
                self.temperature = 0.6
                self.top_p = 0.95
            else:
                self.temperature = 0.7
                self.top_p = 0.8
        elif 'gemini' in model_name.lower():
            self.temperature = 1.0
            self.top_p = 0.95
        else:
            raise NotImplementedError(f"Curator for model {model_name} is not implemented")
        self.llm = LLMChatCompletion(model_name=model_name)

    def curate(
        self,
        current_playbook: str,
        recent_reflection: str,
        current_step: int,
        total_samples: int,
        token_budget: int,
        playbook_stats: Dict[str, Any],
        call_id: str = "curate",
        log_dir: Optional[str] = None,
        prefix_counters: Optional[Dict[str, int]] = None,
    ) -> Tuple[str, Dict[str, int], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Curate playbook based on reflection.

        Returns:
            Tuple of (updated_playbook, prefix_counters, operations, call_info)
        """
        prefix_counters = dict(prefix_counters or {})
        stats_str = json.dumps(playbook_stats, indent=2)
        prompt = self._curator_prompt.format(
            token_budget=token_budget,
            current_step=current_step,
            total_samples=total_samples,
            playbook_stats=stats_str,
            recent_reflection=recent_reflection,
            current_playbook=current_playbook,
        )
        print(f"[Curator][Input]\n{prompt}")

        messages = [Message(role="user", content=prompt)]
        try:
            result = self.llm(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
            raw = result.visible_content or result.raw_response or ""
            parsed = extract_json_from_text(raw)
            if parsed is not None:
                print(
                    "[Curator][Output][Parsed]\n"
                    + json.dumps(parsed, ensure_ascii=False, indent=2)
                )
            else:
                print(f"[Curator][Output][Raw]\n{raw}")
            if not parsed or "operations" not in parsed:
                if log_dir:
                    save_path = os.path.dirname(log_dir)
                    log_curator_failure(save_path, current_step, "no_operations", raw, 0, "Missing operations in response")
                return current_playbook, prefix_counters, [], {"raw": raw}
            ops = parsed.get("operations", [])
            if not isinstance(ops, list):
                if log_dir:
                    save_path = os.path.dirname(log_dir)
                    log_curator_failure(save_path, current_step, "invalid_operations", raw, 0, "Operations is not a list")
                return current_playbook, prefix_counters, [], {"raw": raw}
            # Validate structured operations
            valid_ops = []
            allowed_ops = {
                "ADD_MEMORY_TYPE",
                "DELETE_MEMORY_TYPE",
                "UPDATE_MEMORY_TYPE_NAME",
                "UPDATE_TYPE_DEFINITION",
                "UPDATE_TYPE_WHEN_TO_APPLY",
                "ADD_TYPE_GUIDELINE",
                "UPDATE_TYPE_GUIDELINE",
                "DELETE_TYPE_GUIDELINE",
                "ADD_GENERAL_GUIDELINE",
                "UPDATE_GENERAL_GUIDELINE",
                "DELETE_GENERAL_GUIDELINE",
            }
            for op in ops:
                if not isinstance(op, dict) or "type" not in op:
                    continue
                ot = op.get("type")
                if ot not in allowed_ops:
                    print(f"  Warning: Unknown operation type {ot}, skipping")
                    continue
                if ot == "ADD_MEMORY_TYPE":
                    if op.get("type_name"):
                        if self._require_def_when and not (op.get("definition") and op.get("when_to_apply")):
                            print("  Warning: ADD_MEMORY_TYPE missing definition/when_to_apply, skipping")
                        else:
                            valid_ops.append(op)
                    else:
                        print("  Warning: ADD_MEMORY_TYPE missing required fields, skipping")
                elif ot in {
                    "DELETE_MEMORY_TYPE",
                    "UPDATE_MEMORY_TYPE_NAME",
                    "UPDATE_TYPE_DEFINITION",
                    "UPDATE_TYPE_WHEN_TO_APPLY",
                    "ADD_TYPE_GUIDELINE",
                }:
                    if op.get("type_slug"):
                        if ot == "UPDATE_MEMORY_TYPE_NAME" and not op.get("new_type_name"):
                            print("  Warning: UPDATE_MEMORY_TYPE_NAME missing new_type_name, skipping")
                        elif ot in {"UPDATE_TYPE_DEFINITION", "UPDATE_TYPE_WHEN_TO_APPLY", "ADD_TYPE_GUIDELINE"} and op.get("content") is None:
                            print(f"  Warning: {ot} missing content, skipping")
                        else:
                            valid_ops.append(op)
                    else:
                        print(f"  Warning: {ot} missing type_slug, skipping")
                elif ot in {"UPDATE_TYPE_GUIDELINE", "DELETE_TYPE_GUIDELINE", "UPDATE_GENERAL_GUIDELINE", "DELETE_GENERAL_GUIDELINE"}:
                    if op.get("bullet_id"):
                        if ot in {"UPDATE_TYPE_GUIDELINE", "UPDATE_GENERAL_GUIDELINE"} and op.get("content") is None:
                            print(f"  Warning: {ot} missing content, skipping")
                        else:
                            valid_ops.append(op)
                    else:
                        print(f"  Warning: {ot} missing bullet_id, skipping")
                elif ot == "ADD_GENERAL_GUIDELINE":
                    if op.get("content") is not None:
                        valid_ops.append(op)
                    else:
                        print("  Warning: ADD_GENERAL_GUIDELINE missing content, skipping")
                else:
                    # Should not reach here due to allowed_ops check
                    continue
            ops = valid_ops
            if self._forbidden_ops:
                ops = [op for op in ops if op.get("type") not in self._forbidden_ops]
            # Log each operation diff before applying
            diff_log_dir = os.path.dirname(log_dir) if log_dir else None
            for op in ops:
                try:
                    log_curator_operation_diff(diff_log_dir, op, current_playbook, call_id)
                except Exception as e:
                    print(f"  Warning: Failed to log curator operation diff: {e}")
            updated, updated_prefix_counters = apply_curator_operations(
                current_playbook, ops, prefix_counters
            )
            return updated, updated_prefix_counters, ops, {"raw": raw}
        except Exception as e:
            print(f"[Curator] Error: {e}")
            if log_dir:
                save_path = os.path.dirname(log_dir)
                log_curator_failure(save_path, current_step, "exception", str(e), 0, str(e))
            return current_playbook, prefix_counters, [], {"error": str(e)}

    async def acurate(
        self,
        current_playbook: str,
        recent_reflection: str,
        current_step: int,
        total_samples: int,
        token_budget: int,
        playbook_stats: Dict[str, Any],
        call_id: str = "curate",
        log_dir: Optional[str] = None,
        prefix_counters: Optional[Dict[str, int]] = None,
    ) -> Tuple[str, Dict[str, int], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Async variant of curate(). Behavior and outputs match sync path.
        """
        prefix_counters = dict(prefix_counters or {})
        stats_str = json.dumps(playbook_stats, indent=2)
        prompt = self._curator_prompt.format(
            token_budget=token_budget,
            current_step=current_step,
            total_samples=total_samples,
            playbook_stats=stats_str,
            recent_reflection=recent_reflection,
            current_playbook=current_playbook,
        )
        print(f"[Curator][Input]\n{prompt}")

        messages = [Message(role="user", content=prompt)]
        try:
            result = await self.llm.acall(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
            raw = result.visible_content or result.raw_response or ""
            parsed = extract_json_from_text(raw)
            if parsed is not None:
                print(
                    "[Curator][Output][Parsed]\n"
                    + json.dumps(parsed, ensure_ascii=False, indent=2)
                )
            else:
                print(f"[Curator][Output][Raw]\n{raw}")
            if not parsed or "operations" not in parsed:
                if log_dir:
                    save_path = os.path.dirname(log_dir)
                    log_curator_failure(save_path, current_step, "no_operations", raw, 0, "Missing operations in response")
                return current_playbook, prefix_counters, [], {"raw": raw}
            ops = parsed.get("operations", [])
            if not isinstance(ops, list):
                if log_dir:
                    save_path = os.path.dirname(log_dir)
                    log_curator_failure(save_path, current_step, "invalid_operations", raw, 0, "Operations is not a list")
                return current_playbook, prefix_counters, [], {"raw": raw}
            valid_ops = []
            allowed_ops = {
                "ADD_MEMORY_TYPE",
                "DELETE_MEMORY_TYPE",
                "UPDATE_MEMORY_TYPE_NAME",
                "UPDATE_TYPE_DEFINITION",
                "UPDATE_TYPE_WHEN_TO_APPLY",
                "ADD_TYPE_GUIDELINE",
                "UPDATE_TYPE_GUIDELINE",
                "DELETE_TYPE_GUIDELINE",
                "ADD_GENERAL_GUIDELINE",
                "UPDATE_GENERAL_GUIDELINE",
                "DELETE_GENERAL_GUIDELINE",
            }
            for op in ops:
                if not isinstance(op, dict) or "type" not in op:
                    continue
                ot = op.get("type")
                if ot not in allowed_ops:
                    print(f"  Warning: Unknown operation type {ot}, skipping")
                    continue
                if ot == "ADD_MEMORY_TYPE":
                    if op.get("type_name"):
                        if self._require_def_when and not (op.get("definition") and op.get("when_to_apply")):
                            print("  Warning: ADD_MEMORY_TYPE missing definition/when_to_apply, skipping")
                        else:
                            valid_ops.append(op)
                    else:
                        print("  Warning: ADD_MEMORY_TYPE missing required fields, skipping")
                elif ot in {
                    "DELETE_MEMORY_TYPE",
                    "UPDATE_MEMORY_TYPE_NAME",
                    "UPDATE_TYPE_DEFINITION",
                    "UPDATE_TYPE_WHEN_TO_APPLY",
                    "ADD_TYPE_GUIDELINE",
                }:
                    if op.get("type_slug"):
                        if ot == "UPDATE_MEMORY_TYPE_NAME" and not op.get("new_type_name"):
                            print("  Warning: UPDATE_MEMORY_TYPE_NAME missing new_type_name, skipping")
                        elif ot in {"UPDATE_TYPE_DEFINITION", "UPDATE_TYPE_WHEN_TO_APPLY", "ADD_TYPE_GUIDELINE"} and op.get("content") is None:
                            print(f"  Warning: {ot} missing content, skipping")
                        else:
                            valid_ops.append(op)
                    else:
                        print(f"  Warning: {ot} missing type_slug, skipping")
                elif ot in {"UPDATE_TYPE_GUIDELINE", "DELETE_TYPE_GUIDELINE", "UPDATE_GENERAL_GUIDELINE", "DELETE_GENERAL_GUIDELINE"}:
                    if op.get("bullet_id"):
                        if ot in {"UPDATE_TYPE_GUIDELINE", "UPDATE_GENERAL_GUIDELINE"} and op.get("content") is None:
                            print(f"  Warning: {ot} missing content, skipping")
                        else:
                            valid_ops.append(op)
                    else:
                        print(f"  Warning: {ot} missing bullet_id, skipping")
                elif ot == "ADD_GENERAL_GUIDELINE":
                    if op.get("content") is not None:
                        valid_ops.append(op)
                    else:
                        print("  Warning: ADD_GENERAL_GUIDELINE missing content, skipping")
                else:
                    continue
            ops = valid_ops
            if self._forbidden_ops:
                ops = [op for op in ops if op.get("type") not in self._forbidden_ops]
            diff_log_dir = os.path.dirname(log_dir) if log_dir else None
            for op in ops:
                try:
                    log_curator_operation_diff(diff_log_dir, op, current_playbook, call_id)
                except Exception as e:
                    print(f"  Warning: Failed to log curator operation diff: {e}")
            updated, updated_prefix_counters = apply_curator_operations(
                current_playbook, ops, prefix_counters
            )
            return updated, updated_prefix_counters, ops, {"raw": raw}
        except Exception as e:
            print(f"[Curator][Async] Error: {e}")
            if log_dir:
                save_path = os.path.dirname(log_dir)
                log_curator_failure(save_path, current_step, "exception", str(e), 0, str(e))
            return current_playbook, prefix_counters, [], {"error": str(e)}
