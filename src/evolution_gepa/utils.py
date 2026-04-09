"""
Utility functions for GEPA memory extraction optimization.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.extraction.extraction_prompts import (
    GENERAL_USER_PROMPT,
    FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT,
    SURVEY_SYSTEM_PROMPT,
    SIMPLE_SYSTEM_PROMPT,
)


def extract_seed_candidate(prompt_type: str = "factual-experiential") -> dict[str, str]:
    """
    Extract the seed candidate for GEPA.

    GEPA optimizes the *entire* system prompt as a single component.
    The user prompt is fixed (see build_extraction_prompt).
    """
    if prompt_type == "factual-experiential":
        return {"system_prompt": FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT}
    elif prompt_type == "survey":
        return {"system_prompt": SURVEY_SYSTEM_PROMPT}
    elif prompt_type == "simple":
        return {"system_prompt": SIMPLE_SYSTEM_PROMPT}
    else:
        raise ValueError(
            f"Unsupported prompt_type: {prompt_type!r}. "
            "Supported: 'factual-experiential', 'survey', 'simple'."
        )


def build_extraction_prompt(candidate: dict[str, str], conversation: str) -> tuple[str, str]:
    """
    Build extraction prompts from a GEPA candidate.

    - system_prompt: optimized by GEPA (single component)
    - user_prompt: fixed and unchanged
    """
    system_prompt = candidate.get("system_prompt", FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT)
    user_prompt = GENERAL_USER_PROMPT.format(conversation=conversation)
    return system_prompt, user_prompt
