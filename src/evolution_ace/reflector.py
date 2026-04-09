"""
Reflector for evolution_ace.

Mirrors ACE Reflector: analyzes extraction outcome, tags bullets as helpful/harmful/neutral.
"""

import json
import sys
import os
from typing import Tuple, List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm import LLMChatCompletion, Message

from .playbook import extract_json_from_text
from .prompts.reflector import REFLECTOR_PROMPT
from .logger import log_llm_call


class Reflector:
    """
    Reflector agent that analyzes extraction outcome and tags bullets.
    Analogous to ACE Reflector.
    """

    def __init__(
        self,
        model_name: str = "openai/Qwen3-32B",
        max_tokens: int = 4096,
        enable_thinking: bool = True
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
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
            raise NotImplementedError(f"Reflector for model {model_name} is not implemented")
        self.llm = LLMChatCompletion(model_name=model_name)

    def reflect(
        self,
        source_conversation: str,
        reasoning_trace: str,
        extracted_memory: str,
        target_conversation: str,
        environment_feedback: str,
        bullets_used: str,
        call_id: str = "reflect",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Analyze extraction and tag bullets.

        Returns:
            Tuple of (reflection_content, bullet_tags, call_info)
        """
        prompt = REFLECTOR_PROMPT.format(
            source_conversation=source_conversation[:4096],
            reasoning_trace=reasoning_trace,
            extracted_memory=extracted_memory,
            target_conversation=target_conversation[:4096],
            environment_feedback=environment_feedback,
            bullets_used=bullets_used,
        )
        print(f"[Reflector][Input]\n{prompt}")

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
                    "[Reflector][Output][Parsed]\n"
                    + json.dumps(parsed, ensure_ascii=False, indent=2)
                )
            else:
                print(f"[Reflector][Output][Raw]\n{raw}")
            if log_dir:
                call_info = {
                    "role": "reflector",
                    "call_id": call_id,
                    "model": self.model_name,
                    "raw": raw[:500],
                }
                log_llm_call(log_dir, call_info)
            bullet_tags = []
            if parsed and "bullet_tags" in parsed:
                for t in parsed["bullet_tags"]:
                    if isinstance(t, dict) and "id" in t and "tag" in t:
                        bullet_tags.append({"id": str(t["id"]), "tag": str(t["tag"])})
            return raw, bullet_tags, {"raw": raw}
        except Exception as e:
            print(f"[Reflector] Error: {e}")
            return "", [], {"error": str(e)}

    async def areflect(
        self,
        source_conversation: str,
        reasoning_trace: str,
        extracted_memory: str,
        target_conversation: str,
        environment_feedback: str,
        bullets_used: str,
        call_id: str = "reflect",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Async variant of reflect(). Behavior and outputs match sync path.
        """
        prompt = REFLECTOR_PROMPT.format(
            source_conversation=source_conversation[:4096],
            reasoning_trace=reasoning_trace,
            extracted_memory=extracted_memory,
            target_conversation=target_conversation[:4096],
            environment_feedback=environment_feedback,
            bullets_used=bullets_used,
        )
        print(f"[Reflector][Input]\n{prompt}")

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
                    "[Reflector][Output][Parsed]\n"
                    + json.dumps(parsed, ensure_ascii=False, indent=2)
                )
            else:
                print(f"[Reflector][Output][Raw]\n{raw}")
            if log_dir:
                call_info = {
                    "role": "reflector",
                    "call_id": call_id,
                    "model": self.model_name,
                    "raw": raw[:500],
                }
                log_llm_call(log_dir, call_info)
            bullet_tags = []
            if parsed and "bullet_tags" in parsed:
                for t in parsed["bullet_tags"]:
                    if isinstance(t, dict) and "id" in t and "tag" in t:
                        bullet_tags.append({"id": str(t["id"]), "tag": str(t["tag"])})
            return raw, bullet_tags, {"raw": raw}
        except Exception as e:
            print(f"[Reflector][Async] Error: {e}")
            return "", [], {"error": str(e)}
