"""JSON repair loop for LLM outputs.

When a parse attempt fails, re-prompt the LLM with the error context
and the original prompt to get a corrected response.

Usage::

    from .repair import parse_with_repair

    raw = llm.chat(prompt)
    data = parse_with_repair(raw, prompt, llm, max_retries=3)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict
from collections import Counter
from copy import deepcopy

from ..prompts import SYSTEM_MSG


_REPAIR_PROMPT = """Your previous response could not be parsed as valid JSON.
Parse error: {error}

Respond ONLY with a valid JSON object — no markdown fences, no explanation,
no prefix text, no trailing text.

Your previous (broken) response was:
{prev_response}

Original task prompt (for context):
{original_prompt}
"""


_MAX_ERROR_EXAMPLES = 20
_REPAIR_STATS: Dict[str, Any] = {
    "pre_repair_invalid_json": 0,
    "post_repair_valid_json": 0,
    "failed_after_max_retries": 0,
    "retry_count_histogram": Counter(),
    "common_invalid_patterns": Counter(),
    "invalid_examples": [],
}


def reset_repair_stats() -> None:
    """Reset global repair statistics for a new pipeline run."""
    _REPAIR_STATS["pre_repair_invalid_json"] = 0
    _REPAIR_STATS["post_repair_valid_json"] = 0
    _REPAIR_STATS["failed_after_max_retries"] = 0
    _REPAIR_STATS["retry_count_histogram"].clear()
    _REPAIR_STATS["common_invalid_patterns"].clear()
    _REPAIR_STATS["invalid_examples"] = []


def get_repair_stats() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of global repair statistics."""
    out = deepcopy(_REPAIR_STATS)
    out["retry_count_histogram"] = dict(sorted(out["retry_count_histogram"].items()))
    out["common_invalid_patterns"] = dict(sorted(out["common_invalid_patterns"].items()))
    return out


def _error_pattern(exc: Exception) -> str:
    msg = str(exc)
    if "Expecting value" in msg:
        return "expecting_value"
    if "Extra data" in msg:
        return "extra_data"
    if "Unterminated string" in msg:
        return "unterminated_string"
    if "delimiter" in msg:
        return "delimiter_error"
    if "property name" in msg:
        return "property_name_error"
    if "invalid" in msg.lower():
        return "invalid_json"
    return "other_parse_error"


def _record_invalid_example(exc: Exception, bad_output: str) -> None:
    if len(_REPAIR_STATS["invalid_examples"]) >= _MAX_ERROR_EXAMPLES:
        return
    _REPAIR_STATS["invalid_examples"].append({
        "pattern": _error_pattern(exc),
        "error": str(exc)[:180],
        "output_snippet": bad_output[:180],
    })


def _strip_and_parse(s: str) -> Dict[str, Any]:
    """Strip markdown fences, extract the first JSON object, and parse it."""
    s = s.strip()
    # Remove optional markdown code fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # Extract first {...} block
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


def parse_with_repair(
    raw_output: str,
    original_prompt: str,
    llm,
    max_retries: int = 3,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Try to parse raw_output as JSON; if it fails, re-prompt LLM to fix it.

    Args:
        raw_output:       Raw string from the LLM's first response.
        original_prompt:  The original user message used to generate raw_output.
        llm:              LLM object with a .chat(msg, ...) method.
        max_retries:      Maximum number of repair attempts before raising.
        max_new_tokens:   Token budget for repair responses.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If all repair attempts are exhausted.
    """
    last_output = raw_output
    retries_used = 0
    had_initial_parse_error = False
    for attempt in range(max_retries + 1):
        try:
            parsed = _strip_and_parse(last_output)
            _REPAIR_STATS["retry_count_histogram"][str(retries_used)] += 1
            if had_initial_parse_error:
                _REPAIR_STATS["post_repair_valid_json"] += 1
            return parsed
        except Exception as exc:
            if not had_initial_parse_error:
                _REPAIR_STATS["pre_repair_invalid_json"] += 1
                had_initial_parse_error = True
            _REPAIR_STATS["common_invalid_patterns"][_error_pattern(exc)] += 1
            _record_invalid_example(exc, last_output)
            if attempt == max_retries:
                _REPAIR_STATS["failed_after_max_retries"] += 1
                raise ValueError(
                    f"JSON repair failed after {max_retries} attempts. "
                    f"Last error: {exc}. "
                    f"Last output snippet: {last_output[:200]!r}"
                ) from exc

            retries_used += 1
            repair_msg = _REPAIR_PROMPT.format(
                error=str(exc),
                prev_response=last_output[:600],
                original_prompt=original_prompt[:1200],
            )
            last_output = llm.chat(
                repair_msg,
                system_msg=SYSTEM_MSG,
                max_new_tokens=max_new_tokens,
                temperature=0.1,   # deterministic for repair
            )

    raise ValueError("Unreachable")   # pragma: no cover


def parse_without_repair(raw_output: str) -> Dict[str, Any]:
    """Parse a JSON response once and record invalid-pattern stats on failure.

    This is intended for high-throughput batch paths where full retry-based repair
    would be prohibitively expensive.
    """
    try:
        parsed = _strip_and_parse(raw_output)
        _REPAIR_STATS["retry_count_histogram"]["0"] += 1
        return parsed
    except Exception as exc:
        _REPAIR_STATS["pre_repair_invalid_json"] += 1
        _REPAIR_STATS["common_invalid_patterns"][_error_pattern(exc)] += 1
        _record_invalid_example(exc, raw_output)
        raise
