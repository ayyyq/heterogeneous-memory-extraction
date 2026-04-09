"""
PersonaMem-v2 inference prompts (MCQ, single-turn).

Helpers:
- assign_choice_letters(choices_list, correct_answer) → (letters_dict, correct_letter)
- build_system_prompt(memory, use_memory)             → str  (system turn)
- build_prompt(question, letters_dict)                → str  (user turn)
- extract_answer(raw_response, letters_dict)          → letter str (A/B/C/D/...)
"""

import re
from typing import Dict, List, Optional, Tuple

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYSTEM_PROMPT = "You are a helpful AI assistant."

_MEMORY_BLOCK = """\
# Retrieved Memory
The following are retrieved memories from past interactions.
<memory>
{memory}
</memory>
Instruction: Treat the above memories as potentially useful context. Consider them when \
responding to the current request, and use them when relevant; ignore them if they are \
unrelated to the current request."""

# User-turn template aligned with official PersonaMem query_llm structure:
#   {question}  +  {choices_text}  +  answer instruction
_PROMPT_TEMPLATE = """\
{question}

{choices_text}

Find the most appropriate model response and give your final answer (A), (B), (C), or (D)."""


def build_system_prompt(memory: str = "", use_memory: bool = False) -> str:
    """Return the system prompt, optionally with a retrieved-memory block appended."""
    if memory or use_memory:
        block = _MEMORY_BLOCK.format(memory=(memory or "").strip())
        return f"{SYSTEM_PROMPT}\n\n{block}"
    return SYSTEM_PROMPT


def assign_choice_letters(
    choices_list: List[str],
    correct_answer: str,
) -> Tuple[Dict[str, str], str]:
    """
    Assign letter labels (A, B, C, ...) to choices in list order.

    Returns:
        letters_dict: {letter: choice_text}
        correct_letter: letter corresponding to correct_answer (empty string if not found)
    """
    letters_dict: Dict[str, str] = {}
    correct_letter = ""
    for i, text in enumerate(choices_list):
        letter = _LETTERS[i] if i < len(_LETTERS) else str(i)
        letters_dict[letter] = text
        if text == correct_answer or text.strip() == correct_answer.strip():
            correct_letter = letter
    return letters_dict, correct_letter


def _format_choices(letters_dict: Dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in letters_dict.items())


def build_prompt(
    question: str,
    letters_dict: Dict[str, str],
) -> str:
    """
    Build the user-turn prompt for a PersonaMem-v2 MCQ.
    Memory is injected into the system prompt via build_system_prompt, not here.
    """
    choices_text = _format_choices(letters_dict)
    return _PROMPT_TEMPLATE.format(
        question=question.strip(),
        choices_text=choices_text,
    )


def extract_answer(raw_response: str, letters_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Extract the chosen letter from a model response.

    Priority:
    1. Single valid letter (exact response).
    2. Bracketed letter like (A) or [A].
    3. "Answer: A" or "answer is A" pattern.
    4. Leading "Letter." pattern like "A. some text".
    5. First standalone word that is a valid letter.
    6. If letters_dict provided, match choice text in response.
    """
    valid = set((letters_dict or {}).keys()) or set(_LETTERS[:4])
    action = (raw_response or "").strip()

    # Single letter
    if len(action) == 1 and action.upper() in valid:
        return action.upper()

    upper = action.upper()

    # Brackets: (A) or [A]
    m = re.search(r"[\(\[]([A-Z])[\)\]]", upper)
    if m and m.group(1) in valid:
        return m.group(1)

    # "Answer: A" or "answer is A"
    m = re.search(r"(?:answer(?:\s+is)?|option)[:\s]+([A-Z])\b", upper)
    if m and m.group(1) in valid:
        return m.group(1)

    # Leading choice label: "A." or "A:"
    m = re.match(r"^([A-Z])[.\s:]", upper)
    if m and m.group(1) in valid:
        return m.group(1)

    # Standalone letter
    m = re.search(r"\b([A-Z])\b", upper)
    if m and m.group(1) in valid:
        return m.group(1)

    # Match choice text in response
    if letters_dict:
        action_lower = action.lower()
        for letter, text in letters_dict.items():
            if text and text.lower() in action_lower:
                return letter

    return ""
