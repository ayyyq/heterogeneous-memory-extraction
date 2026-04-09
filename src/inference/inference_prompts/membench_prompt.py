"""
MemBench inference prompts (INSTRUCTION_FIRST only).

Single template: memory + question + time + choices; rule-based extract_answer.
No GMemory dependency. No third-person template.
"""

import json
import re
from typing import Dict, Optional


# INSTRUCTION_FIRST from MembenchAgent.py (first-person only)
INSTRUCTION_FIRST_TEMPLATE = (
    "Please answer the following question based on past memories of your conversation with the user.\n"
    "Past memory: {memory}\n"
    "Question: (current time is {time}) {question}\n"
    "Choices:\n"
    "A. {choice_A}\n"
    "B. {choice_B}\n"
    "C. {choice_C}\n"
    "D. {choice_D}\n"
    "Please output the correct option for the question, only one corresponding letter, without any other messages.\n"
    "Example: D"
)


def get_question_choices_instruction(
    question: str, time: str, choices: Dict[str, str], memory: str = "", use_memory: bool = False
) -> str:
    """
    Build the user message part: question + time + choices + instruction.
    choices: dict with keys "A", "B", "C", "D".
    use_memory: If True, use the with_memory template even when memory="".
    """
    choice_a = (choices.get("A") or "").strip()
    choice_b = (choices.get("B") or "").strip()
    choice_c = (choices.get("C") or "").strip()
    choice_d = (choices.get("D") or "").strip()
    if memory or use_memory:
        return (
            f"Please answer the following question based on past memories of your conversation with the user.\n"
            f"Past memory:\n{memory}\n\n"
            f"Question: (current time is {time}) {question}\n"
            f"Choices:\n"
            f"A. {choice_a}\n"
            f"B. {choice_b}\n"
            f"C. {choice_c}\n"
            f"D. {choice_d}\n"
            f"Please output the correct option for the question, only one corresponding letter, without any other messages.\n"
            f"Example: D"
        )
    return (
        f"Please answer the following question.\n"
        f"Question: (current time is {time}) {question}\n"
        f"Choices:\n"
        f"A. {choice_a}\n"
        f"B. {choice_b}\n"
        f"C. {choice_c}\n"
        f"D. {choice_d}\n"
        "Please output the correct option for the question, only one corresponding letter, without any other messages.\n"
        "Example: D"
    )


def extract_answer(raw_response: str, choices: Optional[Dict[str, str]] = None) -> str:
    """
    Rule-based extraction of option letter (A/B/C/D) from model output.
    Same logic as original MemBench evaluate_membench (extract_choice_letter).
    """
    action = (raw_response or "").strip()

    # JSON format: {"choice": "A"}
    if action.startswith("{"):
        try:
            data = json.loads(action)
            if "choice" in data:
                return (data["choice"] or "").strip().upper()
        except Exception:
            pass

    # Single letter
    if len(action) == 1 and action.upper() in ("A", "B", "C", "D"):
        return action.upper()

    # Brackets: (A) or [A]
    match = re.search(r"[\(\[]([ABCD])[\)\]]", action.upper())
    if match:
        return match.group(1)

    # Standalone letter
    match = re.search(r"\b([ABCD])\b", action.upper())
    if match:
        return match.group(1)

    # Match choice text in response
    if choices:
        action_lower = action.lower()
        for letter, text in (choices or {}).items():
            if text and (text.lower() in action_lower):
                return letter.upper() if isinstance(letter, str) else str(letter).upper()

    return ""
