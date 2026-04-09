"""
Extractor prompts for evolution_ace.

System prompt is general and reusable (role/input/instructions/playbook/output format).
User prompt is sample-specific (reflection + concrete conversation).
"""

EXTRACTOR_OUTPUT_JSON_FORMAT = """{
  "reasoning": "[Your chain of thought / reasoning process]",
  "bullet_ids": ["fac-00001", "gen-00001"],
  "extracted_memory": "1. Memory 1\\n2. Memory 2"
}"""

EXTRACTOR_USER_PROMPT = """# Reflection From Previous Attempt
Use this reflection as additional guidance. If it conflicts with the current conversation, trust the current conversation.

{reflection}

# Task Input
{general_user_prompt}

# Use this exact JSON format:
{output_json_format}
"""
