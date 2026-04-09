"""
Extractor for evolution_ace.

Mirrors ACE Generator: produces extracted memory + bullet_ids from playbook + reflection + conversation.
Uses JSON output format for structured parsing.
"""

import json
import sys
import os
from typing import Tuple, List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_processing.schemas import SourceExample
from src.llm import LLMChatCompletion, Message
from src.extraction.extraction_prompts import GENERAL_USER_PROMPT

from .playbook import render_playbook_to_system_prompt, extract_json_from_text
from .prompts.extractor import EXTRACTOR_USER_PROMPT, EXTRACTOR_OUTPUT_JSON_FORMAT
from .logger import log_llm_call


class Extractor:
    """
    Extractor agent that produces memory from conversation using playbook and reflection.
    Analogous to ACE Generator.
    """

    def __init__(
        self,
        model_name: str = "openai/Qwen3-32B",
        max_tokens: int = 2048,
        enable_thinking: bool = True,
        prompt_type: str = "factual-experiential",
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.prompt_type = prompt_type
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
            raise NotImplementedError(f"Extractor for model {model_name} is not implemented")
        self.llm = LLMChatCompletion(model_name=model_name)

    def extract(
        self,
        source: SourceExample,
        playbook: str,
        reflection: str = "(empty)",
        call_id: str = "extract",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Extract memory from source using playbook and optional reflection.

        Args:
            source: Source example with conversation
            playbook: Bullet-form playbook
            reflection: Reflection from previous attempt (for retry)
            call_id: Call identifier for logging
            log_dir: Optional log directory

        Returns:
            Tuple of (extracted_memory, bullet_ids, call_info)
        """
        system_prompt = render_playbook_to_system_prompt(playbook, prompt_type=self.prompt_type)
        conversation = source.get_conversation_text()
        general_user_prompt = GENERAL_USER_PROMPT.format(conversation=conversation)

        user_prompt = EXTRACTOR_USER_PROMPT.format(
            reflection=reflection,
            general_user_prompt=general_user_prompt,
            output_json_format=EXTRACTOR_OUTPUT_JSON_FORMAT,
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            result = self.llm(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
            raw = result.visible_content or result.raw_response or ""
            if log_dir:
                call_info = {
                    "role": "extractor",
                    "call_id": call_id,
                    "model": self.model_name,
                    "raw": raw[:500],
                }
                log_llm_call(log_dir, call_info)

            # Persist to source metadata for debugging
            try:
                if isinstance(source.metadata, dict):
                    mm = source.metadata.get("memory_extraction", {})
                    mm["system_prompt"] = system_prompt
                    mm["user_prompt"] = user_prompt
                    mm["response"] = result.raw_response
                    source.metadata["memory_extraction"] = mm
            except Exception:
                pass

            parsed = extract_json_from_text(raw)
            if parsed:
                reasoning = parsed.get("reasoning", "")
                memory = parsed.get("extracted_memory", "")
                bullet_ids = parsed.get("bullet_ids", [])
                if isinstance(bullet_ids, list):
                    # Normalize IDs from model output (trim spaces, keep order, dedupe).
                    normalized_ids: List[str] = []
                    seen = set()
                    for b in bullet_ids:
                        bid = str(b).strip()
                        if bid and bid not in seen:
                            normalized_ids.append(bid)
                            seen.add(bid)
                    bullet_ids = normalized_ids
                else:
                    bullet_ids = []
                return memory, bullet_ids, {"raw": raw, "reasoning": str(reasoning)}
            # Fallback: treat entire response as memory
            return raw, [], {"raw": raw, "reasoning": ""}
        except Exception as e:
            print(f"[Extractor] Error: {e}")
            return "", [], {"error": str(e)}

    async def aextract(
        self,
        source: SourceExample,
        playbook: str,
        reflection: str = "(empty)",
        call_id: str = "extract",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Async variant of extract(). Behavior and outputs match sync path.
        """
        system_prompt = render_playbook_to_system_prompt(playbook, prompt_type=self.prompt_type)
        conversation = source.get_conversation_text()
        general_user_prompt = GENERAL_USER_PROMPT.format(conversation=conversation)

        user_prompt = EXTRACTOR_USER_PROMPT.format(
            reflection=reflection,
            general_user_prompt=general_user_prompt,
            output_json_format=EXTRACTOR_OUTPUT_JSON_FORMAT,
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            result = await self.llm.acall(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
            raw = result.visible_content or result.raw_response or ""
            if log_dir:
                call_info = {
                    "role": "extractor",
                    "call_id": call_id,
                    "model": self.model_name,
                    "raw": raw[:500],
                }
                log_llm_call(log_dir, call_info)

            try:
                if isinstance(source.metadata, dict):
                    mm = source.metadata.get("memory_extraction", {})
                    mm["system_prompt"] = system_prompt
                    mm["user_prompt"] = user_prompt
                    mm["response"] = result.raw_response
                    source.metadata["memory_extraction"] = mm
            except Exception:
                pass

            parsed = extract_json_from_text(raw)
            if parsed:
                reasoning = parsed.get("reasoning", "")
                memory = parsed.get("extracted_memory", "")
                bullet_ids = parsed.get("bullet_ids", [])
                if isinstance(bullet_ids, list):
                    normalized_ids: List[str] = []
                    seen = set()
                    for b in bullet_ids:
                        bid = str(b).strip()
                        if bid and bid not in seen:
                            normalized_ids.append(bid)
                            seen.add(bid)
                    bullet_ids = normalized_ids
                else:
                    bullet_ids = []
                return memory, bullet_ids, {"raw": raw, "reasoning": str(reasoning)}
            return raw, [], {"raw": raw, "reasoning": ""}
        except Exception as e:
            print(f"[Extractor][Async] Error: {e}")
            return "", [], {"error": str(e)}
