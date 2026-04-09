"""
Playbook utilities for evolution_ace.

Mirrors ACE's playbook_utils: parse, format, update counts, apply curator operations.
Adapted for memory extraction prompt structure (Factual Memory, Experiential Memory, General Guidelines).
"""

import json
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.extraction.extraction_prompts import FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT, SURVEY_SYSTEM_PROMPT

# Section slugs for memory extraction playbook (aligned with ACE format)
SECTION_SLUG_MAP = {
    "factual_memory": "fac",
    "experiential_memory": "exp",
    "general_guidelines": "gen",
    "others": "misc",
}


def get_section_slug(section_name: str) -> str:
    """Convert section name to slug format (3-5 chars). Mirrors ace/utils.get_section_slug."""
    clean = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    if clean in SECTION_SLUG_MAP:
        return SECTION_SLUG_MAP[clean]
    words = clean.split("_")
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[0] for w in words[:5])


def parse_playbook_line(line: str) -> dict | None:
    """Parse a single playbook line. Same format as ACE: [id] helpful=X harmful=Y :: content"""
    pattern = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
    match = re.match(pattern, line.strip())
    if match:
        return {
            "id": match.group(1),
            "helpful": int(match.group(2)),
            "harmful": int(match.group(3)),
            "content": match.group(4),
            "raw_line": line,
        }
    return None


def format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    """Format a bullet into playbook line format. Same as ACE."""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def build_prefix_counters(playbook_text: str) -> dict[str, int]:
    """Build max counters per prefix from existing playbook IDs."""
    counters: dict[str, int] = {}
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9_]*)-(\d+)$", parsed["id"])
        if not m:
            continue
        prefix = m.group(1)
        num = int(m.group(2))
        counters[prefix] = max(counters.get(prefix, 0), num)
    return counters


def next_id_for_prefix(prefix: str, counters: dict[str, int]) -> str:
    """Allocate next ID for a prefix and update counters in-place."""
    prefix = str(prefix).strip().lower()
    next_num = counters.get(prefix, 0) + 1
    counters[prefix] = next_num
    return f"{prefix}-{next_num:05d}"


def get_next_global_id(playbook_text: str) -> int:
    """Deprecated: kept for compatibility. Prefer build_prefix_counters()."""
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            m = re.search(r"-(\d+)$", parsed["id"])
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def update_bullet_counts(playbook_text: str, bullet_tags: list) -> str:
    """Update helpful/harmful counts based on tags. Same logic as ACE playbook_utils."""
    lines = playbook_text.strip().split("\n")
    updated_lines = []
    tag_map = {}
    if isinstance(bullet_tags, list) and bullet_tags:
        for tag in bullet_tags:
            if isinstance(tag, dict):
                bid = tag.get("id") or tag.get("bullet", "")
                tval = tag.get("tag", "neutral")
                if bid:
                    tag_map[bid] = tval

    if not tag_map:
        return playbook_text

    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            updated_lines.append(line)
            continue
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in tag_map:
            t = tag_map[parsed["id"]]
            if t == "helpful":
                parsed["helpful"] += 1
            elif t == "harmful":
                parsed["harmful"] += 1
            updated_lines.append(
                format_playbook_line(
                    parsed["id"], parsed["helpful"], parsed["harmful"], parsed["content"]
                )
            )
        else:
            updated_lines.append(line)
    return "\n".join(updated_lines)


def extract_playbook_bullets(playbook_text: str, bullet_ids: list) -> str:
    """Extract specific bullets for Reflector input. Same logic as ACE."""
    if not bullet_ids:
        return "(No bullets used by generator)"
    requested_ids = []
    seen = set()
    for b in bullet_ids:
        bid = str(b).strip()
        if bid and bid not in seen:
            requested_ids.append(bid)
            seen.add(bid)
    lines = playbook_text.strip().split("\n")
    found = []
    found_ids = set()
    for line in lines:
        if line.strip():
            parsed = parse_playbook_line(line)
            if parsed and parsed["id"] in requested_ids:
                found.append(
                    f"[{parsed['id']}] helpful={parsed['helpful']} harmful={parsed['harmful']} :: {parsed['content']}"
                )
                found_ids.add(parsed["id"])
    missing_ids = [bid for bid in requested_ids if bid not in found_ids]
    if not found:
        if missing_ids:
            return (
                "(Generator referenced bullet IDs but none were found in playbook: "
                + ", ".join(missing_ids)
                + ")"
            )
        return "(Generator referenced bullet IDs but none were found in playbook)"
    if missing_ids:
        found.append(
            "(Some referenced bullet IDs were not found in playbook: "
            + ", ".join(missing_ids)
            + ")"
        )
    return "\n".join(found)


def apply_curator_operations(
    playbook_text: str,
    operations: list,
    prefix_counters: dict[str, int] | None = None,
) -> tuple[str, dict[str, int]]:
    """
    Apply structured curator operations to playbook.

    Supports:
    - memory type ops
    - type-level definition/when/guideline ops
    - general guideline ops
    """
    def _normalize_header(title: str) -> str:
        return title.strip().lower().replace(" ", "_").replace("&", "and")

    def _bullet_slug(bullet_id: str) -> str:
        return str(bullet_id).split("-", 1)[0] if "-" in str(bullet_id) else str(bullet_id)

    def _new_bullet(bid: str, content: str) -> dict:
        return {"id": bid, "helpful": 0, "harmful": 0, "content": content}

    counters = dict(prefix_counters or build_prefix_counters(playbook_text))

    def _alloc(prefix: str) -> str:
        return next_id_for_prefix(prefix, counters)

    lines = playbook_text.strip().split("\n")
    sections = []
    current = None
    for line in lines:
        if line.strip().startswith("##"):
            title = line.strip()[2:].strip()
            current = {"title": title, "norm": _normalize_header(title), "bullets": []}
            sections.append(current)
            continue
        parsed = parse_playbook_line(line)
        if parsed and current is not None:
            current["bullets"].append(parsed)

    def find_section_by_norm(norm: str) -> dict | None:
        for s in sections:
            if s["norm"] == norm:
                return s
        return None

    def find_section_by_type_slug(type_slug: str) -> dict | None:
        type_slug = (type_slug or "").strip()
        for s in sections:
            if s["norm"] in {"general_guidelines", "others"}:
                continue
            for b in s["bullets"]:
                if _bullet_slug(b["id"]) == type_slug:
                    return s
        return None

    def find_bullet(bullet_id: str) -> tuple[dict | None, int]:
        for s in sections:
            for idx, b in enumerate(s["bullets"]):
                if b["id"] == bullet_id:
                    return s, idx
        return None, -1

    general = find_section_by_norm("general_guidelines")
    if general is None:
        general = {"title": "General Guidelines", "norm": "general_guidelines", "bullets": []}
        others = find_section_by_norm("others")
        if others is None:
            sections.append(general)
            sections.append({"title": "OTHERS", "norm": "others", "bullets": []})
        else:
            idx = sections.index(others)
            sections.insert(idx, general)

    for op in operations:
        op_type = op.get("type")

        if op_type == "ADD_MEMORY_TYPE":
            type_name = op.get("type_name", "").strip()
            if not type_name:
                continue
            type_slug = (op.get("type_slug") or get_section_slug(type_name)).strip().lower()
            if find_section_by_type_slug(type_slug):
                print(f"  Warning: memory type slug already exists: {type_slug}, skipping ADD_MEMORY_TYPE")
                continue
            definition = op.get("definition", "").strip()
            when_to_apply = op.get("when_to_apply", "").strip()
            guidelines = op.get("guidelines", [])
            if not isinstance(guidelines, list):
                guidelines = []

            new_section = {
                "title": type_name,
                "norm": _normalize_header(type_name),
                "bullets": [],
            }
            if definition:
                bid = _alloc(type_slug)
                new_section["bullets"].append(_new_bullet(bid, definition))
            if when_to_apply:
                bid = _alloc(type_slug)
                new_section["bullets"].append(_new_bullet(bid, when_to_apply))
            for g in guidelines:
                bid = _alloc(type_slug)
                new_section["bullets"].append(_new_bullet(bid, str(g)))

            insert_at = len(sections)
            for i, s in enumerate(sections):
                if s["norm"] == "general_guidelines":
                    insert_at = i
                    break
            sections.insert(insert_at, new_section)
            print(f"  Added memory type '{type_name}' ({type_slug})")
            continue

        if op_type == "DELETE_MEMORY_TYPE":
            type_slug = (op.get("type_slug") or "").strip().lower()
            sec = find_section_by_type_slug(type_slug)
            if sec is not None:
                sections.remove(sec)
                print(f"  Deleted memory type with slug {type_slug}")
            continue

        if op_type == "UPDATE_MEMORY_TYPE_NAME":
            type_slug = (op.get("type_slug") or "").strip().lower()
            new_name = op.get("new_type_name", "").strip()
            sec = find_section_by_type_slug(type_slug)
            if sec is not None and new_name:
                sec["title"] = new_name
                sec["norm"] = _normalize_header(new_name)
                print(f"  Renamed memory type {type_slug} -> {new_name}")
            continue

        if op_type == "UPDATE_TYPE_DEFINITION":
            type_slug = (op.get("type_slug") or "").strip().lower()
            content = op.get("content", "")
            sec = find_section_by_type_slug(type_slug)
            if sec is not None:
                if sec["bullets"]:
                    sec["bullets"][0]["content"] = content
                else:
                    bid = _alloc(type_slug)
                    sec["bullets"].append(_new_bullet(bid, content))
                print(f"  Updated definition for type {type_slug}")
            continue

        if op_type == "UPDATE_TYPE_WHEN_TO_APPLY":
            type_slug = (op.get("type_slug") or "").strip().lower()
            content = op.get("content", "")
            sec = find_section_by_type_slug(type_slug)
            if sec is not None:
                if len(sec["bullets"]) > 1:
                    sec["bullets"][1]["content"] = content
                else:
                    while len(sec["bullets"]) < 1:
                        bid = _alloc(type_slug)
                        sec["bullets"].append(_new_bullet(bid, ""))
                    bid = _alloc(type_slug)
                    sec["bullets"].append(_new_bullet(bid, content))
                print(f"  Updated when-to-apply for type {type_slug}")
            continue

        if op_type == "ADD_TYPE_GUIDELINE":
            type_slug = (op.get("type_slug") or "").strip().lower()
            content = op.get("content", "")
            sec = find_section_by_type_slug(type_slug)
            if sec is not None:
                bid = _alloc(type_slug)
                sec["bullets"].append(_new_bullet(bid, content))
                print(f"  Added guideline to type {type_slug}")
            continue

        if op_type == "UPDATE_TYPE_GUIDELINE":
            bullet_id = str(op.get("bullet_id", ""))
            content = op.get("content", "")
            sec, idx = find_bullet(bullet_id)
            if sec is not None and idx >= 2 and sec["norm"] not in {"general_guidelines", "others"}:
                sec["bullets"][idx]["content"] = content
                print(f"  Updated type guideline {bullet_id}")
            else:
                print(f"  Warning: {bullet_id} is not a type guideline bullet, skipping")
            continue

        if op_type == "DELETE_TYPE_GUIDELINE":
            bullet_id = str(op.get("bullet_id", ""))
            sec, idx = find_bullet(bullet_id)
            if sec is not None and idx >= 2 and sec["norm"] not in {"general_guidelines", "others"}:
                del sec["bullets"][idx]
                print(f"  Deleted type guideline {bullet_id}")
            else:
                print(f"  Warning: {bullet_id} is not a type guideline bullet, skipping")
            continue

        if op_type == "ADD_GENERAL_GUIDELINE":
            content = op.get("content", "")
            bid = _alloc("gen")
            general["bullets"].append(_new_bullet(bid, content))
            print(f"  Added general guideline {bid}")
            continue

        if op_type == "UPDATE_GENERAL_GUIDELINE":
            bullet_id = str(op.get("bullet_id", ""))
            content = op.get("content", "")
            sec, idx = find_bullet(bullet_id)
            if sec is not None and sec["norm"] == "general_guidelines" and idx >= 0:
                sec["bullets"][idx]["content"] = content
                print(f"  Updated general guideline {bullet_id}")
            continue

        if op_type == "DELETE_GENERAL_GUIDELINE":
            bullet_id = str(op.get("bullet_id", ""))
            sec, idx = find_bullet(bullet_id)
            if sec is not None and sec["norm"] == "general_guidelines" and idx >= 0:
                del sec["bullets"][idx]
                print(f"  Deleted general guideline {bullet_id}")
            continue

        print(f"  Warning: Unsupported operation type {op_type}, skipping")

    rendered = []
    for sec in sections:
        rendered.append(f"## {sec['title']}")
        for b in sec["bullets"]:
            rendered.append(format_playbook_line(b["id"], b["helpful"], b["harmful"], b["content"]))
        rendered.append("")

    return "\n".join(rendered).strip(), counters


def get_playbook_stats(playbook_text: str) -> dict:
    """Generate statistics about the playbook. Same structure as ACE."""
    lines = playbook_text.strip().split("\n")
    stats = {
        "total_bullets": 0,
        "high_performing": 0,
        "problematic": 0,
        "unused": 0,
        "by_section": {},
    }
    current_section = "general"
    for line in lines:
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip()
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            stats["total_bullets"] += 1
            if parsed["helpful"] > 5 and parsed["harmful"] < 2:
                stats["high_performing"] += 1
            elif parsed["harmful"] >= parsed["helpful"] and parsed["harmful"] > 0:
                stats["problematic"] += 1
            elif parsed["helpful"] + parsed["harmful"] == 0:
                stats["unused"] += 1
            if current_section not in stats["by_section"]:
                stats["by_section"][current_section] = {"count": 0, "helpful": 0, "harmful": 0}
            stats["by_section"][current_section]["count"] += 1
            stats["by_section"][current_section]["helpful"] += parsed["helpful"]
            stats["by_section"][current_section]["harmful"] += parsed["harmful"]
    return stats


def extract_json_from_text(text: str, json_key: str = None) -> dict | None:
    """Extract JSON from text. Same logic as ACE playbook_utils."""
    try:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
    except Exception:
        pass
    return None


def seed_playbook_from_factual_experiential() -> str:
    """
    Convert FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT into bullet-form playbook.
    Each modifiable unit (Definition, When to apply, Extraction Guidelines, General Guidelines) becomes bullets.
    """
    # Parse the prompt and create bullets
    # Factual Memory
    fac_def = "Stable facts or persistent information that are likely to remain valid across future interactions and can directly support understanding or decision-making. This includes user-specific information, task constraints, environment facts, definitions, or clarified requirements established during the conversation."
    fac_when = "The conversation establishes new factual information that may be needed later. The user states preferences, constraints, or persistent context. Important task-related facts are clarified or disambiguated. The information is unlikely to change frequently and is reusable beyond the current interaction."
    fac_guidelines = [
        "Extract information that reduces ambiguity or repeated clarification in future interactions.",
        "Preserve the factual content while abstracting away episode-specific details.",
        "Avoid transient states, temporary intermediate results, or one-time observations.",
        "Do not restate the full conversation or summarize events; extract only reusable facts.",
    ]
    # Experiential Memory
    exp_def = "Generalizable insights derived from experience, including strategies, reasoning patterns, decision principles, or lessons learned from success or failure. These memories help guide future behavior in similar situations."
    exp_when = "The conversation reveals why a particular approach succeeded or failed. A useful strategy, interpretation, or decision pattern emerges. The assistant or user identifies a recurring difficulty, inefficiency, or effective practice. The insight can improve performance in future tasks beyond this specific case."
    exp_guidelines = [
        "Focus on underlying reasoning or decision principles rather than specific actions.",
        "Abstract away from episode-specific details and concrete entities.",
        "Prefer insights that transfer across tasks or environments.",
        "Avoid step-by-step procedures, exhaustive action traces, or superficial summaries.",
        "Ensure the lesson is supported by evidence in the conversation rather than speculation.",
    ]
    # General Guidelines
    gen_guidelines = [
        "Extract at most 5 memory items in total.",
        "Memories of different types may be extracted from the same conversation; prioritize those most likely to benefit future interactions.",
        "Precision over recall: extract only high-confidence, high-value memories. Do not hallucinate user intent or external facts.",
        "Each memory must be self-contained and understandable without access to the original conversation.",
        "Prefer reusable and transferable information over episode-specific details.",
        "Prefer fewer, more general memories over multiple overlapping ones; if several memories reflect the same underlying insight, extract only the most general form.",
        "Do not extract memories that merely summarize what happened; memories should be useful for future reasoning or decision-making.",
    ]

    lines = []

    lines.append("## Factual Memory")
    lines.append(format_playbook_line("fac-00001", 0, 0, fac_def))
    lines.append(format_playbook_line("fac-00002", 0, 0, fac_when))
    for i, g in enumerate(fac_guidelines, start=3):
        lines.append(format_playbook_line(f"fac-{i:05d}", 0, 0, g))

    lines.append("")
    lines.append("## Experiential Memory")
    lines.append(format_playbook_line("exp-00001", 0, 0, exp_def))
    lines.append(format_playbook_line("exp-00002", 0, 0, exp_when))
    for i, g in enumerate(exp_guidelines, start=3):
        lines.append(format_playbook_line(f"exp-{i:05d}", 0, 0, g))

    lines.append("")
    lines.append("## General Guidelines")
    for i, g in enumerate(gen_guidelines, start=1):
        lines.append(format_playbook_line(f"gen-{i:05d}", 0, 0, g))

    lines.append("")
    lines.append("## OTHERS")

    return "\n".join(lines)


def seed_playbook_from_seed() -> str:
    """
    Convert SURVEY_SYSTEM_PROMPT into bullet-form playbook.

    Each memory type has a single description bullet (no separate definition /
    when-to-apply bullets). General guidelines become gen-xxxxx bullets.
    This structure intentionally excludes UPDATE_TYPE_DEFINITION and
    UPDATE_TYPE_WHEN_TO_APPLY curator operations.
    """
    fac_desc = "Factual memory refers to the assistant's declarative knowledge base, established to ensure consistency, coherence, and adaptability by recalling explicit facts, user preferences, and environmental states."
    exp_desc = "Experiential memory refers to the assistant's procedural and strategic knowledge, accumulated to enable continual learning and self-evolution by abstracting from past trajectories, failures, and successes."
    gen_guidelines = [
        "Extract at most 5 memories in total.",
    ]

    lines = []

    lines.append("## Factual Memory")
    lines.append(format_playbook_line("fac-00001", 0, 0, fac_desc))

    lines.append("")
    lines.append("## Experiential Memory")
    lines.append(format_playbook_line("exp-00001", 0, 0, exp_desc))

    lines.append("")
    lines.append("## General Guidelines")
    for i, g in enumerate(gen_guidelines, start=1):
        lines.append(format_playbook_line(f"gen-{i:05d}", 0, 0, g))

    lines.append("")
    lines.append("## OTHERS")

    return "\n".join(lines)


def seed_playbook_from_simple() -> str:
    """
    Create a minimal playbook from SIMPLE_SYSTEM_PROMPT.

    Unlike seed/factual-experiential, there is no memory taxonomy —
    only general guidelines and an OTHERS section.
    """
    gen_guidelines = [
        "Extract at most 5 memories in total.",
    ]

    lines = []

    lines.append("## General Guidelines")
    for i, g in enumerate(gen_guidelines, start=1):
        lines.append(format_playbook_line(f"gen-{i:05d}", 0, 0, g))

    lines.append("")
    lines.append("## OTHERS")

    return "\n".join(lines)


def render_playbook_to_system_prompt(playbook: str, prompt_type: str = "factual-experiential") -> str:
    """
    Convert bullet playbook into a system prompt seeded from
    FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT.

    The resulting system prompt is generic and reusable. It includes:
    - Input format
    - Instructions
    - Playbook (Memory Taxonomy + General Guidelines)
    - Output format
    """
    # Keep role + input format from the seed prompt.
    seed_parts = FACTUAL_EXPERIENTIAL_SYSTEM_PROMPT.split("# Memory Taxonomy", 1)
    seed_prefix = seed_parts[0].strip()

    sections = []
    current = None
    for raw_line in playbook.strip().split("\n"):
        line = raw_line.rstrip()
        if line.strip().startswith("##"):
            title = line.strip()[2:].strip()
            current = {"title": title, "bullets": []}
            sections.append(current)
            continue
        parsed = parse_playbook_line(line)
        if parsed and current is not None:
            current["bullets"].append(parsed)

    general_titles = {"general_guidelines", "general guidelines"}
    other_titles = {"others", "other"}

    taxonomy_sections = []
    general_guidelines = []
    for sec in sections:
        normalized = sec["title"].strip().lower().replace(" ", "_")
        if normalized in general_titles:
            general_guidelines = sec["bullets"]
        elif normalized in other_titles:
            continue
        else:
            taxonomy_sections.append(sec)

    out = [seed_prefix, "\n\n# Instructions\n"]
    out.append("- Read the playbook carefully and apply relevant memory-type criteria.\n")
    out.append("- Include only bullet IDs that were genuinely useful for extraction.\n")
    out.append("- Extract high-value reusable memories; avoid episode-only details.\n")
    out.append("- Return ONLY valid JSON in the required schema.\n\n")

    out.append("# Playbook\n")
    out.append("## Memory Taxonomy\n")
    if taxonomy_sections:
        for sec in taxonomy_sections:
            bullets = sec["bullets"]
            out.append(f"\n### {sec['title']}\n")
            if prompt_type in ("survey", "simple"):
                if bullets:
                    out.append("Guidelines:\n")
                    for b in bullets:
                        out.append(f"- {b['content']}\n")
            else:
                if bullets:
                    out.append("Definition:\n")
                    out.append(f"- {bullets[0]['content']}\n")
                if len(bullets) > 1:
                    out.append("When to apply:\n")
                    out.append(f"- {bullets[1]['content']}\n")
                if len(bullets) > 2:
                    out.append("Extraction Guidelines:\n")
                    for b in bullets[2:]:
                        out.append(f"- {b['content']}\n")
    else:
        out.append("\n- (No memory taxonomy defined in playbook)\n")

    out.append("\n## General Guidelines\n")
    if general_guidelines:
        for b in general_guidelines:
            out.append(f"- {b['content']}\n")
    else:
        out.append("- (No general guidelines defined in playbook)\n")

    out.append("\n# Output Format\n")
    out.append("Return ONLY a JSON object with exact fields:\n")
    out.append("- reasoning: your reasoning process\n")
    out.append("- bullet_ids: list of bullet IDs you used (e.g., [\"fac-00001\", \"gen-00001\"])\n")
    out.append("- extracted_memory: numbered list string, e.g. \"1. Memory 1\\n2. Memory 2\"\n")
    return "".join(out)
