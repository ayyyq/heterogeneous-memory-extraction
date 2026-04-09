import os
import time
import asyncio
import logging

from typing import (
    Protocol,
    Literal,
    Optional,
    List,
)
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import litellm
except ImportError as exc:
    raise ImportError(
        "LiteLLM is required for src.llm. Install with: pip install litellm"
    ) from exc


# Reduce LiteLLM verbosity and drop unsupported params gracefully.
litellm.set_verbose = False
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


MAX_TOKEN = 512
TEMPERATURE = 0.1
TOP_P = 1.0
NUM_COMPS = 1


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 6:
        return "*" * len(key)
    return f"{key[:3]}***{key[-3:]}"


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMCompletionResult:
    """Result of an LLM completion call."""

    raw_response: str
    visible_content: str
    thinking_content: str


def _split_thinking(text: str) -> tuple[str, str]:
    """
    Split raw LLM output into (thinking_content, visible_content).
    Supports <think>...</think>VISIBLE_ANSWER format.
    """
    start_tag = "<think>"
    end_tag = "</think>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start != -1 and end != -1 and end > start:
        thinking_content = text[start + len(start_tag) : end].strip()
        visible_content = text[end + len(end_tag) :].strip()
        if not visible_content:
            visible_content = text.strip()
        return thinking_content, visible_content
    return "", text.strip()


class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass


class LLM(ABC):

    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass


class LLMChatCompletion(LLM):

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name=model_name)
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # if self.api_base:
        #     print("# api base: ", self.api_base)
        # if self.api_key:
        #     print("# api key: ", _mask_key(self.api_key))

    def _is_gemini(self) -> bool:
        return "gemini" in self.model_name.lower()

    def _is_openai_native(self) -> bool:
        """Return True for models served by OpenAI's official API (not vLLM)."""
        name = self.model_name.lower()
        return name.startswith("gpt-") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4")

    def _build_params(
        self,
        messages: List[Message],
        temperature: float,
        top_p: float,
        max_tokens: int,
        num_comps: int,
        stop_strs: Optional[List[str]],
        enable_thinking: bool,
        thinking_budget: Optional[int],
        thinking_level: Optional[str],
    ) -> dict:
        messages_payload = [{"role": msg.role, "content": msg.content} for msg in messages]
        params = {
            "model": self.model_name,
            "messages": messages_payload,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop_strs,
            "n": num_comps,
        }
        if self._is_gemini():
            if "gemini-3" in self.model_name.lower():
                # Gemini 3.0+ supports thinkingLevel
                level = (thinking_level or "MEDIUM").upper()
                params["thinkingConfig"] = {
                    "thinkingLevel": level,
                    "includeThoughts": True,
                }
            elif "gemini-2.5" in self.model_name.lower():
                # Gemini 2.5: use -1 (dynamic) by default; callers with small
                # max_tokens should pass an explicit budget to avoid truncation.
                budget = thinking_budget if thinking_budget is not None else -1
                include = budget != 0
                params["thinkingConfig"] = {
                    "thinkingBudget": budget,
                    "includeThoughts": include,
                }
        elif not self._is_openai_native():
            # vLLM / OpenAI-compatible: use extra_body.chat_template_kwargs
            chat_template_kwargs = {"enable_thinking": enable_thinking}
            if thinking_budget is not None:
                chat_template_kwargs["thinking_budget"] = thinking_budget
            params["extra_body"] = {
                "chat_template_kwargs": chat_template_kwargs
            }
        return params

    @staticmethod
    def _is_retryable(error_message: str) -> bool:
        return (
            "429" in error_message
            or "500" in error_message
            or "502" in error_message
            or "503" in error_message
            or "529" in error_message
            or "rate limit" in error_message.lower()
            or "resource exhausted" in error_message.lower()
            or "quota" in error_message.lower()
            or "connection" in error_message.lower()
            or "timeout" in error_message.lower()
            or "timed out" in error_message.lower()
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = -1,
        num_comps: int = 1,
        stop_strs: Optional[List[str]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
    ) -> LLMCompletionResult:
        max_retries = 5
        base_delay = 1.0
        backoff_factor = 2.0
        max_delay = 16.0
        last_error: Optional[str] = None

        assert max_tokens is not None

        params = self._build_params(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            num_comps=num_comps,
            stop_strs=stop_strs,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )

        for attempt in range(max_retries):
            try:
                response = litellm.completion(**params)

                msg = response.choices[0]["message"]
                raw_response = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or "").strip()

                if not raw_response and not reasoning:
                    last_error = "LLM returned empty content"
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    print(
                        f"[llm] Empty content (attempt {attempt + 1}/{max_retries}, "
                        f"model={self.model_name}). Retrying in {delay:.0f}s ...",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue

                if reasoning:
                    thinking_content = reasoning
                    visible_content = raw_response
                elif enable_thinking:
                    thinking_content, visible_content = _split_thinking(raw_response)
                else:
                    thinking_content = ""
                    visible_content = raw_response

                return LLMCompletionResult(
                    raw_response=raw_response,
                    visible_content=visible_content,
                    thinking_content=thinking_content,
                )

            except Exception as e:
                error_message = str(e)
                last_error = error_message
                if self._is_retryable(error_message):
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    print(
                        f"[llm] Transient error (attempt {attempt + 1}/{max_retries}): "
                        f"{error_message[:200]}. Retrying in {delay:.0f}s ...",
                        flush=True,
                    )
                    time.sleep(delay)
                else:
                    break

        raise RuntimeError(
            f"LLMChatCompletion failed after {max_retries} retries "
            f"(model={self.model_name}). Last error: {last_error or 'unknown error'}"
        )

    async def acall(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = -1,
        num_comps: int = 1,
        stop_strs: Optional[List[str]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
    ) -> LLMCompletionResult:
        max_retries = 5
        base_delay = 1.0
        backoff_factor = 2.0
        max_delay = 16.0
        last_error: Optional[str] = None

        assert max_tokens is not None

        params = self._build_params(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            num_comps=num_comps,
            stop_strs=stop_strs,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )

        for attempt in range(max_retries):
            try:
                response = await litellm.acompletion(**params)

                msg = response.choices[0]["message"]
                raw_response = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or "").strip()

                if not raw_response and not reasoning:
                    last_error = "LLM returned empty content"
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    print(
                        f"[llm] Empty content (attempt {attempt + 1}/{max_retries}, "
                        f"model={self.model_name}). Retrying in {delay:.0f}s ...",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue

                if reasoning:
                    thinking_content = reasoning
                    visible_content = raw_response
                elif enable_thinking:
                    thinking_content, visible_content = _split_thinking(raw_response)
                else:
                    thinking_content = ""
                    visible_content = raw_response

                return LLMCompletionResult(
                    raw_response=raw_response,
                    visible_content=visible_content,
                    thinking_content=thinking_content,
                )

            except Exception as e:
                error_message = str(e)
                last_error = error_message
                if self._is_retryable(error_message):
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    print(
                        f"[llm] Transient error (attempt {attempt + 1}/{max_retries}): "
                        f"{error_message[:200]}. Retrying in {delay:.0f}s ...",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        raise RuntimeError(
            f"LLMChatCompletion.acall failed after {max_retries} retries "
            f"(model={self.model_name}). Last error: {last_error or 'unknown error'}"
        )
