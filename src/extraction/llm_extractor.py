"""
LLM-based memory extractor.

Uses language models to extract insights/memories from source trajectories.
"""

import sys
import os
from typing import Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_processing.schemas import SourceExample
from src.extraction.extraction_prompts import get_extraction_prompt
from src.llm import LLMChatCompletion, Message


class LLMExtractor:
    """
    LLM-based memory extractor.
    
    Uses a language model to extract insights, patterns, or summaries
    from source trajectories.
    """
    
    def __init__(
        self,
        prompt_type: str = "general",
        model_name: str = "Qwen3-32B",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        enable_thinking: bool = True,
        thinking_level: Optional[str] = None,
    ):
        """
        Initialize the LLM extractor.

        Args:
            prompt_type: Type of extraction prompt
            model_name: Name of the LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            thinking_level: Gemini thinking level (HIGH/MEDIUM/LOW/MINIMAL)
        """
        # Prompt configuration
        self.prompt_type = prompt_type

        # Initialize LLM
        self.llm = LLMChatCompletion(model_name=model_name)

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_level = thinking_level

        # Auto-adjust defaults for Gemini models
        if "gemini" in model_name.lower():
            self.temperature = 1.0
            self.top_p = 0.95
            # Only Gemini 3.0+ supports thinkingLevel
            if "gemini-3" in model_name.lower():
                self.thinking_level = "MEDIUM"
    
    def _postprocess_memory(self, raw_memory: str) -> str:
        """
        Post-process extracted memory based on prompt_type.

        Handles known JSON formats:
        - mem0: {"facts": ["fact1", "fact2", ...]} → numbered list
        - ACE:  {"extracted_memory": "..."} → plain text
        Falls back to raw text if not JSON.
        """
        from src.evolution_ace.playbook import extract_json_from_text

        parsed = extract_json_from_text(raw_memory)
        if isinstance(parsed, dict):
            # mem0 format: {"facts": ["fact1", "fact2", ...]}
            if "facts" in parsed and isinstance(parsed["facts"], list):
                facts = [f for f in parsed["facts"] if isinstance(f, str) and f.strip()]
                if facts:
                    return "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(facts))
                return ""
            # ACE format: {"extracted_memory": "..."}
            if "extracted_memory" in parsed:
                return parsed["extracted_memory"]
        return raw_memory

    def _run_llm_with_prompts(
        self,
        source: SourceExample,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Low-level helper to call the LLM with explicit prompts and
        persist traces into ``source.metadata['memory_extraction']``.
        
        This is shared by both the standard ``extract()`` path and
        evolution/GEPA, so that all memory-extraction runs leave
        metadata in a consistent format.
        """
        # Prepare messages
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        result = self.llm(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
            thinking_level=self.thinking_level,
        )

        # Post-process memory based on prompt_type (e.g., extract JSON for mem0)
        memory = self._postprocess_memory(result.visible_content)

        # Persist prompts, raw response and thinking trace into metadata for later analysis.
        try:
            if isinstance(source.metadata, dict):
                mm_meta = source.metadata.get("memory_extraction", {})
                mm_meta["system_prompt"] = system_prompt
                mm_meta["user_prompt"] = user_prompt
                mm_meta["response"] = result.raw_response
                # mm_meta["visible_response"] = visible_response  # Response after removing thinking tags
                if self.llm._is_gemini() and result.thinking_content:
                    mm_meta["thinking_content"] = result.thinking_content
                source.metadata["memory_extraction"] = mm_meta
        except Exception:
            # Metadata persistence is best-effort; don't break extraction.
            pass

        return memory

    async def _arun_llm_with_prompts(
        self,
        source: SourceExample,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Async variant of `_run_llm_with_prompts`.
        """
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        result = await self.llm.acall(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
            thinking_level=self.thinking_level,
        )

        memory = self._postprocess_memory(result.visible_content)

        try:
            if isinstance(source.metadata, dict):
                mm_meta = source.metadata.get("memory_extraction", {})
                mm_meta["system_prompt"] = system_prompt
                mm_meta["user_prompt"] = user_prompt
                mm_meta["response"] = result.raw_response
                if self.llm._is_gemini() and result.thinking_content:
                    mm_meta["thinking_content"] = result.thinking_content
                source.metadata["memory_extraction"] = mm_meta
        except Exception:
            pass

        return memory

    def extract(self, source: SourceExample) -> str:
        """
        Extract memory from a source example using the default
        extraction prompt for the configured ``prompt_type``.
        """
        system_prompt, user_prompt = get_extraction_prompt(
            prompt_type=self.prompt_type,
            source=source,
        )
        return self._run_llm_with_prompts(source, system_prompt, user_prompt)

    def extract_with_prompts(
        self,
        source: SourceExample,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Extract memory using **custom prompts**.
        
        This is primarily used by the evolution/GEPA pipeline, which
        constructs candidate memory-extraction prompts and wants to
        reuse the same LLM invocation + metadata tracing logic as
        the standard extractor.
        """
        return self._run_llm_with_prompts(source, system_prompt, user_prompt)

    async def aextract(self, source: SourceExample) -> str:
        """
        Async extraction using the default prompt for `prompt_type`.
        """
        system_prompt, user_prompt = get_extraction_prompt(
            prompt_type=self.prompt_type,
            source=source,
        )
        return await self._arun_llm_with_prompts(source, system_prompt, user_prompt)

    async def aextract_with_prompts(
        self,
        source: SourceExample,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Async extraction using custom prompts.
        """
        return await self._arun_llm_with_prompts(source, system_prompt, user_prompt)
    
    def extract_batch(self, sources: list[SourceExample], verbose: bool = True) -> list[str]:
        """
        Extract memory from multiple sources.
        
        Args:
            sources: List of source examples
            verbose: Whether to print progress
            
        Returns:
            List of extracted memory strings
        """
        memories = []
        for i, source in enumerate(sources):
            if verbose:
                print(f"[LLMExtractor] Extracting memory {i+1}/{len(sources)} (source_id={source.id})")
            memory = self.extract(source)
            memories.append(memory)
        return memories
