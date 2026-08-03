"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})
CANONICAL_TOOL_RESULT_SCHEMA = "xiaoban.tool-result.v1"
CANONICAL_TOOL_RESULT_INTERNAL_KEY = "_xiaoban_tool_result"
TRUSTED_STEER_INTERNAL_KEY = "_xiaoban_trusted_steer"
CANONICAL_TOOL_RESULT_DISPATCH_STATES = frozenset({"not_dispatched", "dispatched"})
CANONICAL_TOOL_RESULT_OUTCOMES = frozenset(
    {"success", "empty", "not_found", "denied", "failed", "unknown", "cancelled"}
)
_STATUS_OUTCOMES = {
    "empty": "empty", "no_content": "empty",
    "not_found": "not_found", "not-found": "not_found", "404": "not_found",
    "denied": "denied", "forbidden": "denied", "unauthorized": "denied",
    "permission_denied": "denied", "access_denied": "denied",
    "401": "denied", "403": "denied",
    "failed": "failed", "failure": "failed", "error": "failed", "timeout": "failed",
    "unknown": "unknown", "ambiguous": "unknown", "indeterminate": "unknown",
    "cancelled": "cancelled", "canceled": "cancelled",
    "interrupted": "cancelled", "aborted": "cancelled",
}
_MODEL_VALUE_LIMIT = 100_000
_TRUSTED_AUX_VALUE_LIMIT = 8_000
_MODEL_METADATA_FIELDS = (
    "schema", "requestId", "turnId", "callId", "toolName", "dispatchState",
    "outcome", "retrySafe", "recordRefs", "coverage", "truncated", "continuation",
)
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


def _decoded_top_level(result: Any) -> Any:
    if not isinstance(result, str):
        return result
    try:
        return json.loads(result)
    except (TypeError, ValueError):
        return result


def _classify_canonical_outcome(*, tool_name, result, dispatch_state, outcome_hint) -> str:
    if outcome_hint in CANONICAL_TOOL_RESULT_OUTCOMES:
        return str(outcome_hint)

    payload = _decoded_top_level(result)
    mapped = ""
    if isinstance(payload, Mapping):
        for value in (payload.get("outcome"), payload.get("status"), payload.get("code")):
            if isinstance(value, int) and not isinstance(value, bool):
                value = str(value)
            if isinstance(value, str):
                mapped = _STATUS_OUTCOMES.get(value.strip().lower(), "")
                if mapped:
                    break
        if payload.get("cancelled") is True or payload.get("canceled") is True:
            mapped = "cancelled"
    if mapped == "cancelled":
        return mapped
    if dispatch_state == "not_dispatched":
        return "denied"
    if mapped in {"not_found", "denied", "unknown"}:
        return mapped
    if mapped == "failed" or tool_result_failed(tool_name, result):
        return "failed"
    if mapped == "empty":
        return "empty"
    return "success"


def _bounded_json_value(value: Any, limit: int) -> Any | None:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) <= limit:
            return json.loads(encoded)
    except (TypeError, ValueError):
        return None
    return None


def normalize_tool_result(
    *,
    request_id: str,
    turn_id: str,
    call_id: str,
    tool_name: str,
    dispatch_state: str,
    result: Any,
    outcome_hint: str | None = None,
    trusted_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize runtime-owned IDs/state plus explicitly trusted receipt fields."""
    if dispatch_state not in CANONICAL_TOOL_RESULT_DISPATCH_STATES:
        raise ValueError(f"invalid ToolResult dispatch state: {dispatch_state!r}")

    outcome = _classify_canonical_outcome(
        tool_name=str(tool_name or ""), result=result,
        dispatch_state=dispatch_state, outcome_hint=outcome_hint,
    )
    trusted = trusted_fields if isinstance(trusted_fields, Mapping) else {}
    normalized: dict[str, Any] = {
        "schema": CANONICAL_TOOL_RESULT_SCHEMA,
        "requestId": str(request_id or ""),
        "turnId": str(turn_id or ""),
        "callId": str(call_id or ""),
        "toolName": str(tool_name or ""),
        "dispatchState": dispatch_state,
        "outcome": outcome,
        "retrySafe": trusted.get("retrySafe") is True and outcome != "unknown",
    }
    if dispatch_state == "not_dispatched" and outcome in {"denied", "cancelled"}:
        return normalized

    refs = trusted.get("recordRefs")
    if isinstance(refs, (list, tuple)):
        refs = [str(item)[:256] for item in refs[:32] if isinstance(item, (str, int))]
        if refs:
            normalized["recordRefs"] = refs
    for key in ("coverage", "continuation"):
        value = _bounded_json_value(trusted.get(key), _TRUSTED_AUX_VALUE_LIMIT)
        if value is not None:
            normalized[key] = value
    if isinstance(trusted.get("truncated"), bool):
        normalized["truncated"] = trusted["truncated"]
    return normalized


def canonical_tool_result_for_persistence(
    value: Any,
    *,
    call_id: str | None = None,
    tool_name: str | None = None,
) -> dict[str, Any] | None:
    """Return the bounded canonical sidecar that may cross a DB boundary.

    Session rows are durable input to later model calls.  Treat their JSON as
    untrusted on replay: only the executor-owned v1 shape is restored, and its
    correlation fields must still match the surrounding tool message.
    """
    if not isinstance(value, Mapping):
        return None
    if value.get("schema") != CANONICAL_TOOL_RESULT_SCHEMA:
        return None
    dispatch_state = value.get("dispatchState")
    outcome = value.get("outcome")
    if dispatch_state not in CANONICAL_TOOL_RESULT_DISPATCH_STATES:
        return None
    if outcome not in CANONICAL_TOOL_RESULT_OUTCOMES:
        return None

    identifiers: dict[str, str] = {}
    for key, limit in (
        ("requestId", 512),
        ("turnId", 512),
        ("callId", 512),
        ("toolName", 256),
    ):
        item = value.get(key)
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item) > limit
        ):
            return None
        identifiers[key] = item
    if call_id is not None and identifiers["callId"] != str(call_id):
        return None
    if tool_name is not None and identifiers["toolName"] != str(tool_name):
        return None

    persisted: dict[str, Any] = {
        "schema": CANONICAL_TOOL_RESULT_SCHEMA,
        **identifiers,
        "dispatchState": dispatch_state,
        "outcome": outcome,
        "retrySafe": value.get("retrySafe") is True and outcome != "unknown",
    }
    refs = value.get("recordRefs")
    if isinstance(refs, (list, tuple)):
        bounded_refs = [
            str(item)[:256]
            for item in refs[:32]
            if isinstance(item, (str, int)) and not isinstance(item, bool)
        ]
        if bounded_refs:
            persisted["recordRefs"] = bounded_refs
    for key in ("coverage", "continuation"):
        bounded = _bounded_json_value(value.get(key), _TRUSTED_AUX_VALUE_LIMIT)
        if bounded is not None:
            persisted[key] = bounded
    if isinstance(value.get("truncated"), bool):
        persisted["truncated"] = value["truncated"]
    return persisted


def _bounded_model_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        if len(value) <= _MODEL_VALUE_LIMIT:
            decoded = _decoded_top_level(value)
            return decoded, False
        return value[:_MODEL_VALUE_LIMIT], True
    bounded = _bounded_json_value(value, _MODEL_VALUE_LIMIT)
    if bounded is not None:
        return bounded, False
    rendered = str(value)
    return rendered[:_MODEL_VALUE_LIMIT], len(rendered) > _MODEL_VALUE_LIMIT


def _fit_multimodal_text_part(
    part: Mapping[str, Any],
    max_encoded_size: int,
) -> tuple[dict[str, Any], int] | None:
    """Fit one JSON-safe text part without producing an invalid partial part."""
    text = part.get("text")
    if not isinstance(text, str) or max_encoded_size <= 0:
        return None
    base = dict(part)
    base["text"] = ""
    base_encoded = json.dumps(
        base,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    if len(base_encoded) > max_encoded_size:
        return None

    low = 0
    high = len(text)
    best = base
    best_size = len(base_encoded)
    while low <= high:
        midpoint = (low + high) // 2
        candidate = dict(part)
        candidate["text"] = text[:midpoint]
        encoded = json.dumps(
            candidate,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        if len(encoded) <= max_encoded_size:
            best = candidate
            best_size = len(encoded)
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best, best_size


def _bounded_multimodal_parts(
    content: list[Any],
    limit: int = _MODEL_VALUE_LIMIT,
) -> tuple[list[Any], bool]:
    """Bound nested multimodal payload bytes while retaining valid small parts."""
    normalized: list[tuple[int, Any, int, bool]] = []
    was_truncated = False
    for index, part in enumerate(content):
        try:
            encoded = json.dumps(
                part,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            cloned = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            was_truncated = True
            continue
        is_text = bool(
            isinstance(cloned, Mapping)
            and cloned.get("type") == "text"
            and isinstance(cloned.get("text"), str)
        )
        normalized.append((index, cloned, len(encoded), is_text))

    # Reserve valid non-text blocks first so an oversized text part cannot
    # crowd a small image out of the projection. Large image/data blocks are
    # omitted whole rather than truncated into an invalid URL or base64 value.
    has_text = any(item[3] for item in normalized)
    payload_capacity = max(0, limit - 2)  # JSON list brackets.
    minimum_text_reserve = min(4_096, payload_capacity) if has_text else 0
    non_text_capacity = payload_capacity - minimum_text_reserve
    reserved_non_text: set[int] = set()
    reserved_size = 0
    for index, _part, encoded_size, is_text in normalized:
        if is_text:
            continue
        item_size = encoded_size + 1  # Conservatively include a comma.
        if reserved_size + item_size <= non_text_capacity:
            reserved_non_text.add(index)
            reserved_size += item_size
        else:
            was_truncated = True

    bounded: list[Any] = []
    used_size = 2
    remaining_reserved = reserved_size
    for index, part, encoded_size, is_text in normalized:
        item_size = encoded_size + 1
        if not is_text:
            if index not in reserved_non_text:
                continue
            bounded.append(part)
            used_size += item_size
            remaining_reserved -= item_size
            continue

        available_item_size = limit - used_size - remaining_reserved
        max_encoded_size = available_item_size - 1
        if max_encoded_size <= 0:
            was_truncated = True
            continue
        if encoded_size <= max_encoded_size:
            bounded.append(part)
            used_size += item_size
            continue
        fitted = _fit_multimodal_text_part(part, max_encoded_size)
        if fitted is not None:
            fitted_part, fitted_size = fitted
            bounded.append(fitted_part)
            used_size += fitted_size + 1
        was_truncated = True

    return bounded, was_truncated


def _trusted_steer_markers(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _without_trusted_steer_suffix(content: Any, markers: list[str]) -> Any:
    """Remove only the exact runtime-authenticated suffix before projection."""
    if not markers:
        return content
    if isinstance(content, str):
        suffix = "".join(markers)
        if suffix and content.endswith(suffix):
            return content[:-len(suffix)]
        return content
    if isinstance(content, list):
        stripped = list(content)
        for marker in reversed(markers):
            if not stripped:
                return content
            tail = stripped[-1]
            if not (
                isinstance(tail, Mapping)
                and tail.get("type") == "text"
                and tail.get("text") == marker.lstrip()
            ):
                return content
            stripped.pop()
        return stripped
    return content


def append_trusted_steer_markers_for_model(
    content: str | list[Any],
    trusted_steer_markers: Any,
) -> str | list[Any]:
    """Append authenticated user steering outside the tool-result projection."""
    markers = _trusted_steer_markers(trusted_steer_markers)
    if not markers:
        return content
    if isinstance(content, list):
        return [
            *content,
            *({"type": "text", "text": marker.lstrip()} for marker in markers),
        ]
    return content + "".join(markers)


def project_tool_result_for_model(
    content: Any,
    metadata: Mapping[str, Any],
    trusted_steer_markers: Any = None,
) -> str | list[Any]:
    """Build the bounded model-facing projection for one canonical result."""
    markers = _trusted_steer_markers(trusted_steer_markers)
    content = _without_trusted_steer_suffix(content, markers)
    projected = {key: metadata[key] for key in _MODEL_METADATA_FIELDS if key in metadata}
    outcome = str(projected.get("outcome") or "unknown")

    if outcome in {"success", "empty"}:
        if isinstance(content, list):
            bounded_content, was_truncated = _bounded_multimodal_parts(content)
            projected["modelResult"] = {
                "contentType": "multimodal",
                "parts": len(bounded_content),
            }
            if was_truncated:
                projected["modelResult"]["originalParts"] = len(content)
                projected["truncated"] = True
            header = json.dumps(projected, ensure_ascii=False, default=str)
            return append_trusted_steer_markers_for_model(
                [{"type": "text", "text": header}, *bounded_content],
                markers,
            )
        model_value, was_truncated = _bounded_model_value(content)
        projected["modelResult"] = model_value
        if was_truncated:
            projected["truncated"] = True
    elif outcome == "failed":
        model_value, was_truncated = _bounded_model_value(content)
        candidate = model_value.get("code") if isinstance(model_value, Mapping) else None
        code = str(candidate)[:128] if isinstance(candidate, (str, int)) else "failed"
        projected["modelError"] = {"code": code, "details": model_value}
        if was_truncated:
            projected["truncated"] = True
    else:
        projected["modelError"] = {"code": outcome}

    return append_trusted_steer_markers_for_model(
        json.dumps(projected, ensure_ascii=False, default=str),
        markers,
    )
