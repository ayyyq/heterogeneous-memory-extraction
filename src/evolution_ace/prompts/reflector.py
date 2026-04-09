"""
Reflector prompts for evolution_ace.

Mirrors ACE Reflector: diagnose extraction outcome, tag bullets as helpful/harmful/neutral.
Adapted for memory extraction context (source conversation, extracted memory, target conversation).
"""

REFLECTOR_PROMPT = """You are an expert analyst and educator. An extractor generated a memory from a source conversation to assist with a target conversation. Your job is to diagnose why the extracted memory helped or failed to help by analyzing the extraction strategy.

**Instructions:**
- **Analyze Strategy:** Evaluate the extractor's reasoning trace to understand the high-level selection and writing strategy. Focus on the root cause of success or failure, not surface-level memory content.
- **Evaluate Usefulness:** Use the target conversation and environment feedback to judge the memory based on relevance, durability, non-redundancy, actionability, compactness, and safety.
- **Tag Playbook Bullets:** Analyze the playbook bullet points used by the extractor. Assign a 'helpful', 'harmful', or 'neutral' tag to each based on its actual utility.
- **Strict Constraints:**
  - Do NOT propose or rewrite the specific memory content.
  - Do NOT quote or paraphrase specific details from the source or target conversations.
  - Do NOT write episode-specific rules. Generalize your insights to patterns and use placeholders (e.g., USER, PROJECT, TOOL).

**Source Conversation:**
<source_conversation>
{source_conversation}
</source_conversation>

**Extractor's Reasoning Trace:**
{reasoning_trace}

**Extracted Memory:**
{extracted_memory}

**Target Conversation:**
<target_conversation>
{target_conversation}
</target_conversation>

**Environment Feedback:**
{environment_feedback}

**Part of Playbook used by the extractor:**
{bullets_used}

**Answer in this exact JSON format. Return valid JSON only, without any markdown formatting, code fences, or additional text:**
{{
  "input_summary": "[Brief neutral summary of the source conversation only; no target outcome info]",
  "reasoning": "[Detailed analysis of the high-level extraction strategy, avoiding step-by-step internal deliberation]",
  "error_identification": "[What specifically went wrong or worked well in the strategy and process?]",
  "root_cause_analysis": "[Why did the extraction help or fail downstream in generalizable terms?]",
  "correct_approach": "[Reusable extraction guidelines, including specific selection criteria, taxonomy updates, or writing methods]",
  "key_insight": "[What maximally general principle for extracting useful memories should be remembered?]",
  "bullet_tags": [
    {{"id": "fac-00001", "tag": "helpful"}},
    {{"id": "gen-00001", "tag": "harmful"}}
  ]
}}
---
"""

REFLECTOR_PROMPT_NO_GT = REFLECTOR_PROMPT  # Same format; ground truth is implicit in outcome
