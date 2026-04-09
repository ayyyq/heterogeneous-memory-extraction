"""LiteLLM-backed Model adapter for FlashOAgents.

Allows ToolCallingAgent to use any model supported by litellm (e.g. Gemini)
instead of being limited to OpenAI-compatible APIs.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

try:
    from FlashOAgents.models import (
        ChatMessage,
        ChatMessageToolCall,
        ChatMessageToolCallDefinition,
        Model,
        get_tool_json_schema,
        parse_tool_args_if_needed,
    )
    from FlashOAgents.tools import Tool
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parents[1]
    _flash_path = _repo_root / "MemEvolve" / "Flash-Searcher-main"
    if _flash_path.is_dir():
        sys.path.insert(0, str(_flash_path))
        from FlashOAgents.models import (
            ChatMessage,
            ChatMessageToolCall,
            ChatMessageToolCallDefinition,
            Model,
            get_tool_json_schema,
            parse_tool_args_if_needed,
        )
        from FlashOAgents.tools import Tool
    else:
        raise

logger = logging.getLogger(__name__)

litellm.set_verbose = False
litellm.drop_params = True


class LiteLLMModel(Model):
    """FlashOAgents-compatible model that delegates to litellm.completion().

    Parameters:
        model_id: LiteLLM model identifier (e.g. "gemini/gemini-2.5-flash").
        thinking_level: Gemini thinking level (HIGH/MEDIUM/LOW/MINIMAL).
        custom_role_conversions: Optional role conversion mapping.
        **kwargs: Extra keyword arguments forwarded to litellm.completion().
    """

    def __init__(
        self,
        model_id: str,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.custom_role_conversions = custom_role_conversions

    def _is_gemini(self) -> bool:
        return "gemini" in self.model_id.lower()

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Gemini thinking config
        if self._is_gemini():
            if "gemini-3" in self.model_id.lower() and self.thinking_level:
                level = self.thinking_level.upper()
                completion_kwargs["thinkingConfig"] = {
                    "thinkingLevel": level,
                    "includeThoughts": True,
                }
            elif "gemini-2.5" in self.model_id.lower():
                # Default: disable thinking for Gemini 2.5 in ToolCallingAgent
                # because thinking + tool calling causes empty responses (litellm#24442).
                # Callers can override via thinking_budget parameter.
                budget = self.thinking_budget if self.thinking_budget is not None else 0
                include = budget != 0
                completion_kwargs["thinkingConfig"] = {
                    "thinkingBudget": budget,
                    "includeThoughts": include,
                }

        max_retries = 5
        base_delay = 1.0
        backoff_factor = 2.0
        max_delay = 16.0

        for attempt in range(max_retries):
            try:
                response = litellm.completion(**completion_kwargs)

                # Update token counts
                if hasattr(response, "usage") and response.usage:
                    self.last_input_token_count = getattr(response.usage, "prompt_tokens", 0)
                    self.last_output_token_count = getattr(response.usage, "completion_tokens", 0)
                    self.total_input_tokens += self.last_input_token_count or 0
                    self.total_output_tokens += self.last_output_token_count or 0
                    self.total_api_calls += 1

                choice = response.choices[0]
                msg = choice["message"] if isinstance(choice, dict) else choice.message

                raw_content = ""
                if isinstance(msg, dict):
                    raw_content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or ""
                    raw_tool_calls = msg.get("tool_calls")
                else:
                    raw_content = getattr(msg, "content", "") or ""
                    reasoning = getattr(msg, "reasoning_content", "") or ""
                    raw_tool_calls = getattr(msg, "tool_calls", None)

                if not raw_content and not raw_tool_calls and not reasoning:
                    # Log response details for debugging
                    finish_reason = getattr(choice, "finish_reason", None) or (choice.get("finish_reason") if isinstance(choice, dict) else None)
                    usage = getattr(response, "usage", None)
                    comp_tokens = getattr(usage, "completion_tokens", None) if usage else None
                    logger.warning(
                        "LiteLLMModel: empty response — finish_reason=%s, usage=%s, "
                        "raw_content=%r, reasoning=%r, tool_calls=%r",
                        finish_reason, usage, raw_content[:200] if raw_content else raw_content,
                        reasoning[:200] if reasoning else reasoning, raw_tool_calls,
                    )
                    # Gemini safety filter: finish_reason=stop but 0 completion tokens.
                    # Retryable since safety filter has randomness.
                    if comp_tokens == 0 and attempt < max_retries - 1:
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        logger.warning(
                            "LiteLLMModel: likely safety filter (0 completion tokens), "
                            "retrying in %.0fs (attempt %d/%d) ...",
                            delay, attempt + 1, max_retries,
                        )
                        time.sleep(delay)
                        continue
                    raise RuntimeError("LiteLLMModel: empty response from LLM")

                # Remove <think>...</think> tags from visible content
                visible_content = self._remove_think_tags(raw_content)

                # Normalize JSON content for FlashOAgents ToolCallingAgent compatibility.
                # FlashOAgents expects tool calls in content as:
                #   {"think": "...", "tools": [{"name": "...", "arguments": {...}}]}
                # Gemini may produce nested lists or other variants; fix them here.
                visible_content = self._normalize_tool_json(visible_content)

                # Build ChatMessage
                tool_calls = None
                if raw_tool_calls:
                    tool_calls = self._parse_tool_calls(raw_tool_calls)

                message = ChatMessage(
                    role="assistant",
                    content=visible_content,
                    reasoning_content=reasoning if reasoning else None,
                    tool_calls=tool_calls,
                    raw=response,
                )

                if tools_to_call_from is not None:
                    return parse_tool_args_if_needed(message)
                return message

            except Exception as e:
                error_message = str(e)
                retryable = any(
                    kw in error_message.lower()
                    for kw in ["429", "500", "502", "503", "529", "rate limit", "resource exhausted", "quota", "connection", "timeout", "timed out"]
                )
                if retryable and attempt < max_retries - 1:
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        "LiteLLMModel: retryable error (attempt %d/%d): %s. Retrying in %.0fs ...",
                        attempt + 1, max_retries, error_message, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

    @staticmethod
    def _normalize_tool_json(content: str) -> str:
        """Normalize JSON tool-call content for FlashOAgents compatibility.

        FlashOAgents ToolCallingAgent parses ``model.content`` as JSON and
        expects one of:
          - ``{"think": "...", "tools": [{"name": ..., "arguments": ...}]}``
          - ``[{"think": "...", "tools": [...]}]``

        Gemini may produce a top-level list with stray non-dict items mixed
        in, e.g.::

            [["Specific Example"], {"think": "...", "tools": [...]}]

        This method detects such cases and keeps only the dict(s) that
        carry a ``"tools"`` key, also flattening any nested tool lists.
        """
        if not content or not content.strip():
            return content

        import json
        try:
            import json_repair
        except ImportError:
            json_repair = None

        # Try to parse
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            if json_repair:
                try:
                    parsed = json_repair.loads(content)
                except Exception:
                    return content
            else:
                return content

        # --- helpers ---
        def _flatten_tool_list(tools):
            """Flatten nested lists into a flat list of dicts.

            Strings are treated as bare tool names and converted to
            ``{"name": <string>, "arguments": {}}`` so FlashOAgents
            can process them.
            """
            flat = []
            for item in tools:
                if isinstance(item, dict):
                    flat.append(item)
                elif isinstance(item, str):
                    flat.append({"name": item, "arguments": {}})
                elif isinstance(item, list):
                    flat.extend(_flatten_tool_list(item))
            return flat

        def _fix_tools_in_obj(obj):
            """Ensure obj['tools'] is a flat list of dicts. Returns True if changed."""
            if isinstance(obj, dict) and "tools" in obj:
                tools = obj["tools"]
                if isinstance(tools, list):
                    fixed = _flatten_tool_list(tools)
                    if len(fixed) != len(tools) or fixed != tools:
                        obj["tools"] = fixed
                        return True
                    obj["tools"] = fixed
            return False

        normalized = False

        if isinstance(parsed, list):
            # Gemini sometimes produces a list with stray non-dict items
            # mixed alongside the real tool-call dict.
            # Strategy: keep only dicts that contain "tools", fall back to
            # keeping all dicts if none has "tools".
            dicts_with_tools = [x for x in parsed if isinstance(x, dict) and "tools" in x]
            all_dicts = [x for x in parsed if isinstance(x, dict)]

            if dicts_with_tools and len(dicts_with_tools) < len(parsed):
                # There were stray items; keep only tool-bearing dicts
                parsed = dicts_with_tools if len(dicts_with_tools) > 1 else dicts_with_tools[0]
                normalized = True
            elif all_dicts and len(all_dicts) < len(parsed):
                # No "tools" key found but there are stray non-dict items
                parsed = all_dicts if len(all_dicts) > 1 else all_dicts[0]
                normalized = True

            # Flatten nested tool lists inside each dict
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        normalized |= _fix_tools_in_obj(item)
            elif isinstance(parsed, dict):
                normalized |= _fix_tools_in_obj(parsed)

        elif isinstance(parsed, dict):
            normalized |= _fix_tools_in_obj(parsed)

        if normalized:
            return json.dumps(parsed, ensure_ascii=False)
        return content

    @staticmethod
    def _remove_think_tags(content: str) -> str:
        if not content:
            return content
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        return content.strip()

    @staticmethod
    def _parse_tool_calls(raw_tool_calls) -> List[ChatMessageToolCall]:
        tool_calls = []
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tool_calls.append(
                    ChatMessageToolCall(
                        function=ChatMessageToolCallDefinition(
                            arguments=func.get("arguments", ""),
                            name=func.get("name", ""),
                        ),
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                    )
                )
            else:
                # Object-style (e.g. litellm response objects)
                func = getattr(tc, "function", None)
                tool_calls.append(
                    ChatMessageToolCall(
                        function=ChatMessageToolCallDefinition(
                            arguments=getattr(func, "arguments", "") if func else "",
                            name=getattr(func, "name", "") if func else "",
                        ),
                        id=getattr(tc, "id", ""),
                        type=getattr(tc, "type", "function"),
                    )
                )
        return tool_calls
