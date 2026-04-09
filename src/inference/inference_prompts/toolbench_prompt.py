"""
ToolBench inference prompts.

ReACT-style multi-turn format aligned with ToolBench/ToolLLM:
  Thought: reasoning about what to do next
  Action: API name to call
  Action Input: JSON parameters for the API
  Observation: API response (provided by environment)

Final answer format:
  Thought: I have gathered enough information.
  Action: Finish
  Action Input: {"return_type": "give_answer", "final_answer": "..."}
"""

TOOLBENCH_SYSTEM_PROMPT = """\
Answer the following questions as best you can. You have access to the following tools and APIs:

{tool_descriptions}

Specifically, you have access to the following APIs:

{api_descriptions}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the above APIs
Action Input: the input to the action in JSON format

If you believe that you have obtained enough information to answer the task, please call:
Action: Finish
Action Input: {{"return_type": "give_answer", "final_answer": "your answer string"}}

If you believe the task cannot be solved with the available tools, please call:
Action: Finish
Action Input: {{"return_type": "give_up_and_restart"}}

Remember:
1. The state change is irreversible, you can't go back to a former state.
2. All the thought is short, at most in 5 sentences.
3. You can do more than one try, so if your plan is to continuously try some conditions, you can do one of the conditions per try.
"""

TOOLBENCH_SYSTEM_PROMPT_WITH_MEMORY = """\
Answer the following questions as best you can. You have access to the following tools and APIs:

{tool_descriptions}

Specifically, you have access to the following APIs:

{api_descriptions}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the above APIs
Action Input: the input to the action in JSON format

If you believe that you have obtained enough information to answer the task, please call:
Action: Finish
Action Input: {{"return_type": "give_answer", "final_answer": "your answer string"}}

If you believe the task cannot be solved with the available tools, please call:
Action: Finish
Action Input: {{"return_type": "give_up_and_restart"}}

You may receive a Retrieved Memory block with past experiences from similar tasks.
Use the memories as inspiration, but always analyze your current task independently.

Remember:
1. The state change is irreversible, you can't go back to a former state.
2. All the thought is short, at most in 5 sentences.
3. You can do more than one try, so if your plan is to continuously try some conditions, you can do one of the conditions per try.
"""

MEMORY_BLOCK_TEMPLATE = """
# Retrieved Memory
The following are retrieved memories from past interactions.
<memory>
{memory}
</memory>
Instruction: Treat the above memories as potentially useful context. Consider them when responding to the current request, and use them when relevant; ignore them if they are unrelated.
"""

# Finish action format
FINISH_ACTION = "Finish"
FINISH_INPUT_GIVE_ANSWER = {"return_type": "give_answer", "final_answer": ""}
FINISH_INPUT_GIVE_UP = {"return_type": "give_up_and_restart"}


def format_api_descriptions(api_list: list) -> str:
    """Format API list into a description string for the system prompt."""
    if not api_list:
        return "No APIs available."
    parts = []
    for i, api in enumerate(api_list):
        if isinstance(api, str):
            parts.append(f"{i+1}. {api}")
            continue
        name = api.get("name", api.get("api_name", f"api_{i}"))
        desc = api.get("description", "No description available.")
        tool_name = api.get("tool_name", "")
        params = api.get("required_parameters", [])
        optional = api.get("optional_parameters", [])

        param_strs = []
        for p in (params or []):
            if isinstance(p, dict):
                pname = p.get("name", "")
                ptype = p.get("type", "string")
                pdesc = p.get("description", "")
                param_strs.append(f"    - {pname} ({ptype}): {pdesc}")
            else:
                param_strs.append(f"    - {p}")
        for p in (optional or []):
            if isinstance(p, dict):
                pname = p.get("name", "")
                ptype = p.get("type", "string")
                pdesc = p.get("description", "")
                pdefault = p.get("default", "")
                param_strs.append(f"    - {pname} ({ptype}, optional, default={pdefault}): {pdesc}")
            else:
                param_strs.append(f"    - {p} (optional)")

        entry = f"{name}"
        if tool_name:
            entry = f"{tool_name}.{name}"
        entry += f": {desc}"
        if param_strs:
            entry += "\n  Parameters:\n" + "\n".join(param_strs)
        parts.append(entry)
    return "\n\n".join(parts)


def format_tool_descriptions(api_list: list) -> str:
    """Extract unique tool-level descriptions from API list."""
    seen = set()
    parts = []
    for api in api_list:
        if not isinstance(api, dict):
            continue
        tool_name = api.get("tool_name", "")
        if tool_name and tool_name not in seen:
            seen.add(tool_name)
            tool_desc = api.get("tool_description", "")
            parts.append(f"- {tool_name}: {tool_desc}" if tool_desc else f"- {tool_name}")
    return "\n".join(parts) if parts else "General purpose tools."


def build_system_prompt(api_list: list, memory: str = "", use_memory: bool = False) -> str:
    """Build the full system prompt with tool/API descriptions and optional memory."""
    api_desc = format_api_descriptions(api_list)
    tool_desc = format_tool_descriptions(api_list)

    if use_memory or memory:
        prompt = TOOLBENCH_SYSTEM_PROMPT_WITH_MEMORY.format(
            tool_descriptions=tool_desc,
            api_descriptions=api_desc,
        )
        if memory:
            prompt += "\n" + MEMORY_BLOCK_TEMPLATE.format(memory=memory)
        return prompt

    return TOOLBENCH_SYSTEM_PROMPT.format(
        tool_descriptions=tool_desc,
        api_descriptions=api_desc,
    )


def build_first_user_message(query: str) -> str:
    """Build the first user message with the task instruction."""
    return query
