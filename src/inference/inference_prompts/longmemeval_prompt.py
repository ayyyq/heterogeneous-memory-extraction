"""
LongMemEval inference prompts (no-retrieval, cot=True).

Aligned with LongMemEval run_generation.py no-retrieval + cot logic.
No GMemory dependency.
"""


def get_first_user_message_content(question: str, memory: str = "", use_memory: bool = False) -> str:
    """
    First user message for no-retrieval + CoT inference.
    Official: answer_prompt_template = '{}Answer step by step.', format(question_string).
    use_memory: If True, use the with_memory template even when memory="".
    """
    if memory or use_memory:
        memory_instruction = f"""Please answer the question based on past memories of your conversation with the user. Answer the question step by step.
Memory:
{memory}

Question: {question.strip()}
Answer:"""
        return memory_instruction
    return "Answer the question step by step.\n\nQuestion: " + question.strip() + "\nAnswer:"
