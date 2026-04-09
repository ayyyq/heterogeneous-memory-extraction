"""Shared utility helpers for MemEvolve prompt evolution."""

from __future__ import annotations

import re
import time


def split_thinking_visible(text: str) -> tuple[str, str]:
    """Return (thinking_content, visible_content) from raw model output."""
    raw = (text or "").strip()
    start_tag = "<think>"
    end_tag = "</think>"
    start = raw.find(start_tag)
    end = raw.find(end_tag)
    if start != -1 and end != -1 and end > start:
        thinking = raw[start + len(start_tag) : end].strip()
        visible = raw[end + len(end_tag) :].strip()
        if not visible:
            visible = raw
        return thinking, visible
    return "", raw


_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length is\s+(\d+)\s+tokens\s+and your request has\s+(\d+)\s+input tokens",
    flags=re.IGNORECASE,
)


def extract_context_numbers(err_text: str) -> tuple[int, int] | None:
    """Return (context_limit, input_tokens) parsed from overflow errors."""
    m = _CONTEXT_OVERFLOW_RE.search(err_text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def is_context_overflow_error(err_text: str) -> bool:
    """Whether the error text looks like a context-length overflow."""
    return extract_context_numbers(err_text) is not None


def suggest_max_tokens(
    err_text: str,
    current_max_tokens: int,
    safety_margin: int = 128,
) -> int | None:
    """Suggest a lower max_tokens value, or None if fallback should not apply."""
    if int(current_max_tokens) <= 0:
        return None
    nums = extract_context_numbers(err_text)
    if nums is None:
        return None
    context_limit, input_tokens = nums
    available = context_limit - input_tokens - int(safety_margin)
    if available <= 0:
        return None
    suggested = min(int(current_max_tokens), int(available))
    if suggested >= int(current_max_tokens):
        return None
    return max(1, suggested)


_RATE_LIMIT_RE = re.compile(
    r"rate.?limit|429|quota|too many requests|resource.?exhausted",
    flags=re.IGNORECASE,
)


def is_retryable_error(err: Exception) -> bool:
    """Whether the error is transient and worth retrying (rate limit, 5xx, timeout)."""
    err_text = str(err)
    if _RATE_LIMIT_RE.search(err_text):
        return True
    if any(code in err_text for code in ("500", "502", "503", "529")):
        return True
    if "timeout" in err_text.lower() or "timed out" in err_text.lower():
        return True
    return False


def retry_with_backoff(
    fn,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    backoff_factor: float = 2.0,
    logger=None,
):
    """Call *fn()* with exponential backoff on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries or not is_retryable_error(e):
                raise
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            msg = f"Transient error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.0f}s ..."
            if logger:
                logger(msg)
            else:
                print(f"[memevolve] {msg}", flush=True)
            time.sleep(delay)


__all__ = [
    "split_thinking_visible",
    "extract_context_numbers",
    "is_context_overflow_error",
    "suggest_max_tokens",
    "is_retryable_error",
    "retry_with_backoff",
]
