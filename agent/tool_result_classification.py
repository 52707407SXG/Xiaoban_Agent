"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
import re
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})
_FAILED_STATUSES = frozenset(
    {
        "error",
        "failed",
        "failure",
        "cancelled",
        "canceled",
        "timeout",
    }
)
_PLAIN_FAILURE_PREFIX = re.compile(
    r"^\s*(?:"
    r"\[?tool execution cancell?ed\b|"
    r"error(?:\s+executing\s+tool\b|:)|"
    r"exception:|"
    r"traceback \(most recent call last\):"
    r")",
    re.IGNORECASE,
)


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False


def tool_result_failed(tool_name: str, result: Any) -> bool:
    """Classify one tool result from explicit, shared status metadata.

    Executor logging, loop guardrails and streaming all use this helper.  A
    top-level failure signal wins over contradictory success metadata; nested
    business data is never keyword-scanned.  Plain text is treated as failure
    only when it starts with an executor-owned error marker.
    """
    if result is None or file_mutation_result_landed(tool_name, result):
        return False

    payload = result
    decoded_json = not isinstance(result, str)
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
            decoded_json = False
        else:
            decoded_json = True

    if isinstance(payload, dict):
        status = payload.get("status")
        status_failed = False
        if isinstance(status, int) and not isinstance(status, bool):
            status_failed = 400 <= status <= 599
        elif isinstance(status, str):
            normalized = status.strip().lower()
            status_failed = bool(
                re.fullmatch(r"[45]\d{2}", normalized)
                or normalized in _FAILED_STATUSES
            )
        exit_code = payload.get("exit_code")
        failed_exit = bool(
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code != 0
        )
        return bool(
            payload.get("ok") is False
            or payload.get("success") is False
            or payload.get("failed") is True
            or payload.get("is_error") is True
            or payload.get("cancelled") is True
            or payload.get("canceled") is True
            or status_failed
            or failed_exit
            or payload.get("error")
        )

    if decoded_json or not isinstance(result, str):
        return False
    return bool(_PLAIN_FAILURE_PREFIX.match(result))
