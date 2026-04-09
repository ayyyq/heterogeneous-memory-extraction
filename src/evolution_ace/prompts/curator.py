"""
Curator prompts for evolution_ace.

Curator evolves the playbook structure itself:
- memory taxonomy (types + definition + when to apply + guidelines)
- general guidelines
"""

CURATOR_PROMPT = """You are a master curator of memory extraction playbooks. Your job is to improve the playbook based on recent reflection.

**Context:**
- The playbook contains:
  1) Memory Taxonomy: memory types, each with definition, when-to-apply, extraction guidelines
  2) General Guidelines
- An extractor produced a memory from a source conversation to help with a target conversation. The reflection evaluates that extraction using downstream target behavior, which will not be available at extraction time.
- You must improve reusable extraction guidance, not overfit to one sample.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review recent_reflection + current playbook.
- `recent_reflection` includes:
  - `input_summary`: short source-conversation background
  - diagnosis fields: error_identification, root_cause_analysis, correct_approach, key_insight
  - `bullet_tags`
- Use `input_summary` for context, but make changes based on reusable patterns from the full reflection.
- You may add/delete/rename memory types.
- For each memory type, you may update:
  - definition
  - when to apply
  - extraction guidelines
- You may add/update/delete general guidelines.
- Prefer minimal, high-impact edits over large rewrites.
- Do not optimize for one episode.
- All operations must be concrete and executable.

**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought
- operations: list of structured operations

**Allowed Operation Types:**
- ADD_MEMORY_TYPE:
  {{"type":"ADD_MEMORY_TYPE","type_name":"Procedural Memory","type_slug":"pro","definition":"...","when_to_apply":"...","guidelines":["...","..."]}}
- DELETE_MEMORY_TYPE:
  {{"type":"DELETE_MEMORY_TYPE","type_slug":"exp"}}
- UPDATE_MEMORY_TYPE_NAME:
  {{"type":"UPDATE_MEMORY_TYPE_NAME","type_slug":"fac","new_type_name":"Factual Knowledge"}}
- UPDATE_TYPE_DEFINITION:
  {{"type":"UPDATE_TYPE_DEFINITION","type_slug":"fac","content":"..."}}
- UPDATE_TYPE_WHEN_TO_APPLY:
  {{"type":"UPDATE_TYPE_WHEN_TO_APPLY","type_slug":"fac","content":"..."}}
- ADD_TYPE_GUIDELINE:
  {{"type":"ADD_TYPE_GUIDELINE","type_slug":"fac","content":"..."}}
- UPDATE_TYPE_GUIDELINE:
  {{"type":"UPDATE_TYPE_GUIDELINE","bullet_id":"fac-00003","content":"..."}}
- DELETE_TYPE_GUIDELINE:
  {{"type":"DELETE_TYPE_GUIDELINE","bullet_id":"fac-00004"}}
- ADD_GENERAL_GUIDELINE:
  {{"type":"ADD_GENERAL_GUIDELINE","content":"..."}}
- UPDATE_GENERAL_GUIDELINE:
  {{"type":"UPDATE_GENERAL_GUIDELINE","bullet_id":"gen-00001","content":"..."}}
- DELETE_GENERAL_GUIDELINE:
  {{"type":"DELETE_GENERAL_GUIDELINE","bullet_id":"gen-00002"}}

**RESPONSE FORMAT:**
{{
  "reasoning": "[Your analysis]",
  "operations": [
    {{"type":"UPDATE_TYPE_DEFINITION","type_slug":"fac","content":"[Revised definition...]"}},
    {{"type":"ADD_TYPE_GUIDELINE","type_slug":"exp","content":"[New experiential guideline...]"}},
    {{"type":"ADD_GENERAL_GUIDELINE","content":"[New general guideline...]"}}
  ]
}}

---
"""

CURATOR_PROMPT_NO_GT = CURATOR_PROMPT

CURATOR_PROMPT_SEED = """You are a master curator of memory extraction playbooks. Your job is to improve the playbook based on recent reflection.

**Context:**
- The playbook contains:
  1) Memory Taxonomy: memory types, each with a description and extraction guidelines
  2) General Guidelines
- An extractor produced a memory from a source conversation to help with a target conversation. The reflection evaluates that extraction using downstream target behavior, which will not be available at extraction time.
- You must improve reusable extraction guidance, not overfit to one sample.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review recent_reflection + current playbook.
- `recent_reflection` includes:
  - `input_summary`: short source-conversation background
  - diagnosis fields: error_identification, root_cause_analysis, correct_approach, key_insight
  - `bullet_tags`
- Use `input_summary` for context, but make changes based on reusable patterns from the full reflection.
- You may add/delete/rename memory types.
- For each memory type, you may update extraction guidelines.
- You may add/update/delete general guidelines.
- Prefer minimal, high-impact edits over large rewrites.
- Do not optimize for one episode.
- All operations must be concrete and executable.

**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought
- operations: list of structured operations

**Allowed Operation Types:**
- ADD_MEMORY_TYPE:
  {{"type":"ADD_MEMORY_TYPE","type_name":"Procedural Memory","type_slug":"pro","guidelines":["...","..."]}}
- DELETE_MEMORY_TYPE:
  {{"type":"DELETE_MEMORY_TYPE","type_slug":"exp"}}
- UPDATE_MEMORY_TYPE_NAME:
  {{"type":"UPDATE_MEMORY_TYPE_NAME","type_slug":"fac","new_type_name":"Factual Knowledge"}}
- ADD_TYPE_GUIDELINE:
  {{"type":"ADD_TYPE_GUIDELINE","type_slug":"fac","content":"..."}}
- UPDATE_TYPE_GUIDELINE:
  {{"type":"UPDATE_TYPE_GUIDELINE","bullet_id":"fac-00003","content":"..."}}
- DELETE_TYPE_GUIDELINE:
  {{"type":"DELETE_TYPE_GUIDELINE","bullet_id":"fac-00004"}}
- ADD_GENERAL_GUIDELINE:
  {{"type":"ADD_GENERAL_GUIDELINE","content":"..."}}
- UPDATE_GENERAL_GUIDELINE:
  {{"type":"UPDATE_GENERAL_GUIDELINE","bullet_id":"gen-00001","content":"..."}}
- DELETE_GENERAL_GUIDELINE:
  {{"type":"DELETE_GENERAL_GUIDELINE","bullet_id":"gen-00002"}}

**RESPONSE FORMAT:**
{{
  "reasoning": "[Your analysis]",
  "operations": [
    {{"type":"ADD_TYPE_GUIDELINE","type_slug":"fac","content":"[New factual guideline...]"}},
    {{"type":"ADD_TYPE_GUIDELINE","type_slug":"exp","content":"[New experiential guideline...]"}},
    {{"type":"ADD_GENERAL_GUIDELINE","content":"[New general guideline...]"}}
  ]
}}

---
"""
