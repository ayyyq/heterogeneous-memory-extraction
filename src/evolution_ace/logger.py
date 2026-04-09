"""
Logger for evolution_ace.

Mirrors ACE logger: log_llm_call, log_bullet_usage, log_curator_operation_diff, log_curator_failure.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .playbook import parse_playbook_line


def log_llm_call(log_dir: Optional[str], call_info: Dict[str, Any]) -> None:
    """Log detailed information about each LLM call."""
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{call_info.get('role', 'unknown')}_{call_info.get('call_id', 'call')}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    call_info = dict(call_info)
    call_info["timestamp"] = timestamp
    call_info["datetime"] = datetime.now().isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(call_info, f, indent=2, ensure_ascii=False)
    print(f"[LOG] {call_info.get('role', 'unknown')} call logged to {filename}")


def log_bullet_usage(
    usage_log_path: Optional[str],
    epoch: int,
    step: int,
    bullet_ids_used: List[str],
    bullets_with_content: List[Dict[str, Any]],
    is_correct: Optional[bool],
    sample_context: str = "",
    sample_question: str = "",
    reflection_summary: Optional[str] = None,
) -> None:
    """Log which bullets were used in each training sample for future curator reference."""
    if not usage_log_path:
        return
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "step": step,
        "sample_id": f"epoch_{epoch}_step_{step}",
        "bullet_ids_used": bullet_ids_used,
        "bullets_with_content": bullets_with_content,
        "is_correct": is_correct,
        "sample_context": sample_context[:500],
        "sample_question": sample_question[:200],
        "reflection_summary": (reflection_summary or "")[:300] if reflection_summary else None,
        "bullet_count": len(bullet_ids_used),
    }
    with open(usage_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def log_curator_operation_diff(
    log_dir: Optional[str],
    operation: Dict[str, Any],
    playbook_text: str,
    call_id: str,
) -> None:
    """Log detailed diff for curator operations (ADD, UPDATE, DELETE)."""
    if not log_dir:
        return
    try:
        curator_diff_log_path = os.path.join(log_dir, "curator_operations_diff.jsonl")
        op_type = operation.get("type", "UNKNOWN")
        operation_diff = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": op_type,
            "call_id": call_id,
        }
        lines = playbook_text.strip().split("\n")
        if op_type == "UPDATE":
            bullet_id = operation.get("bullet_id", "")
            new_content = operation.get("content", "")
            old_content = None
            for line in lines:
                parsed = parse_playbook_line(line)
                if parsed and parsed["id"] == bullet_id:
                    old_content = parsed["content"]
                    break
            operation_diff.update({
                "bullet_id": bullet_id,
                "old_content": old_content,
                "content": new_content,
                "content_comparison": {
                    "old_length": len(old_content) if old_content else 0,
                    "new_length": len(new_content),
                    "length_change": len(new_content) - (len(old_content) if old_content else 0),
                },
            })
        elif op_type == "ADD":
            operation_diff.update({
                "section": operation.get("section", "others"),
                "content": operation.get("content", ""),
                "content_length": len(operation.get("content", "")),
            })
        elif op_type == "DELETE":
            bullet_id = operation.get("bullet_id") or operation.get("bullet_ids", [])
            if isinstance(bullet_id, list):
                operation_diff["bullet_ids"] = bullet_id
            else:
                operation_diff["bullet_id"] = bullet_id
        with open(curator_diff_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(operation_diff, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log curator operation diff: {e}")


def log_curator_failure(
    save_path: str,
    step: int,
    failure_type: str,
    curator_response: str,
    epoch: int = 0,
    error_details: Optional[str] = None,
) -> None:
    """Log curator failures to a dedicated log file for analysis."""
    curator_failure_log_path = os.path.join(save_path, "curator_failures.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
{'='*60}
CURATOR FAILURE LOG
{'='*60}
Timestamp: {timestamp}
Epoch: {epoch}
Step: {step}
Failure Type: {failure_type}
Error Details: {error_details or 'N/A'}

Raw Curator Response (first 1000 chars):
{'-'*40}
{curator_response[:1000]}
{'-'*40}

Full Raw Curator Response:
{'-'*40}
{curator_response}
{'-'*40}

"""
    try:
        os.makedirs(os.path.dirname(curator_failure_log_path) or ".", exist_ok=True)
        with open(curator_failure_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"[evolution_ace] Curator failure logged to: {curator_failure_log_path}")
    except Exception as e:
        print(f"[evolution_ace] Failed to write curator failure log: {e}")
