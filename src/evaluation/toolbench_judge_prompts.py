"""
ToolBench LLM judge prompts.

Ported from StableToolBench evaluation templates:
  - check_answer_status: for pass rate (SoPR) evaluation
  - select_better_answer: for win rate (SoWR) evaluation

Reference: https://github.com/THUNLP-MT/StableToolBench
  toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default/template.txt
"""

# ---------------------------------------------------------------------------
# Pass Rate (SoPR) — check_answer_status
# ---------------------------------------------------------------------------

CHECK_ANSWER_STATUS_PROMPT = """\
Giving the query and answer, you need give `answer_status` of the answer by following rules:
1. If the answer is a sorry message or not a positive/straight response for the given query, return "Unsolved".
2. If the answer is a positive/straight response for the given query, you have to further check.
2.1 If the answer is not sufficient to determine whether the solve the query or not, return "Unsure".
2.2 If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Solved" or "Unsolved".

Query: {query}
Answer: {answer}

Now give your reason in "content" and `answer_status` in the following JSON format:
{{"content": "your reasoning", "answer_status": "Solved" or "Unsure" or "Unsolved"}}
"""

# ---------------------------------------------------------------------------
# Win Rate (SoWR) — select_better_answer
# ---------------------------------------------------------------------------

SELECT_BETTER_ANSWER_PROMPT = """\
Query: {query}

Answer_0: {answer_0}
Answer_1: {answer_1}

Given above query and answers in JSON format, you must follow the rules to select the relatively better answer and give the index of the answer **(0 for Answer_0, 1 for Answer_1)**:
1. Compare the value of "final_answer" in following aspects:
- Informative: whether it contains all necessary information to reply to the query.
- Factuality: whether it accurately describes what has been done, and what failed in the end.
- Reasoning: If answer does not solve the query, whether gives a detailed and accurate reason for failure.
2. If you cannot determine yet, compare the value of "answer_details" in following aspects:
- Tool calling costs: calculating the percentage of failed and replicated tools calling.
- Running costs: calculating the total tokens T used in execution.
- Milestone: calculating the milestone(fixed subtasks) reached in execution.
- Exploration: whether tries potential useful tools in execution. Just count times of successful tool calling with different tools/arguments in execution.

If you have made your decision, give the index of the better answer. If you cannot determine, select a random answer.

Give your reason in "content" and `better_answer_index` in the following JSON format:
{{"content": "your reasoning", "better_answer_index": 0 or 1}}
"""
