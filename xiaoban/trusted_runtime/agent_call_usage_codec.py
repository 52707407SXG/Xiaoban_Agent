"""Strict plaintext-free codec and monotonic merge for provider call receipts."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


AGENT_CALL_USAGE_SCHEMA = "mystand.agent-call-usage.v1"
AGENT_CALL_LIMIT = 8
AGENT_CALL_USAGE_MAX_BYTES = 64 * 1024

_EXECUTION_ID = re.compile(r"^[a-f0-9]{32}$")
_SAFE_ROUTE = re.compile(r"^[A-Za-z0-9_.:/+-]{1,120}$")
_SAFE_CATEGORY = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_RECEIPT_ACTIVE = {"reserved", "running"}
_RECEIPT_TERMINALS = {
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "not_dispatched",
}
_LEDGER_TERMINALS = {"completed", "failed", "cancelled"}


def receipt_dict(receipt: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "callId": receipt.call_id,
        "ordinal": receipt.ordinal,
        "provider": receipt.provider,
        "model": receipt.model,
        "role": receipt.role,
        "startedAtMs": receipt.started_at_ms,
        "endedAtMs": receipt.ended_at_ms,
        "status": receipt.status,
        "inputTokens": receipt.input_tokens,
        "outputTokens": receipt.output_tokens,
        "totalTokens": receipt.total_tokens,
        "cachedInputTokens": receipt.cached_input_tokens,
        "usageStatus": receipt.usage_status,
    }
    if receipt.error_category:
        item["errorCategory"] = receipt.error_category
    if receipt.cost_usd is not None:
        item["costUsd"] = receipt.cost_usd
    if receipt.cost_status:
        item["costStatus"] = receipt.cost_status
    if receipt.cost_source:
        item["costSource"] = receipt.cost_source
    return item


def project_agent_call_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one ledger; unknown fields are rejected, never persisted."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "executionId",
        "status",
        "calls",
    }:
        raise ValueError("invalid agent call usage ledger")
    if value.get("schema") != AGENT_CALL_USAGE_SCHEMA:
        raise ValueError("invalid agent call usage schema")
    execution_id = str(value.get("executionId") or "")
    status = str(value.get("status") or "")
    calls = value.get("calls")
    if (
        not _EXECUTION_ID.fullmatch(execution_id)
        or status not in {"running", *_LEDGER_TERMINALS}
        or not isinstance(calls, list)
        or len(calls) > AGENT_CALL_LIMIT
    ):
        raise ValueError("invalid agent call usage ledger")
    projected_calls = [
        _project_receipt(item, execution_id=execution_id)
        for item in calls
    ]
    if [item["ordinal"] for item in projected_calls] != list(
        range(1, len(projected_calls) + 1)
    ):
        raise ValueError("invalid agent call ordinal sequence")
    if status == "completed" and any(
        item["status"] in {*_RECEIPT_ACTIVE, "not_dispatched"}
        for item in projected_calls
    ):
        raise ValueError("completed agent ledger has unresolved provider call")
    projected = {
        "schema": AGENT_CALL_USAGE_SCHEMA,
        "executionId": execution_id,
        "status": status,
        "calls": projected_calls,
    }
    encoded = json.dumps(
        projected,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > AGENT_CALL_USAGE_MAX_BYTES:
        raise ValueError("agent call usage ledger too large")
    return projected


def merge_agent_call_usage(
    current: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
    *,
    allow_stopped_late_accounting: bool = False,
    allow_restart_late_accounting: bool = False,
) -> dict[str, Any]:
    incoming_projected = project_agent_call_usage(incoming)
    if current is None:
        return incoming_projected
    current_projected = project_agent_call_usage(current)
    if (
        current_projected["schema"] != incoming_projected["schema"]
        or current_projected["executionId"]
        != incoming_projected["executionId"]
    ):
        raise ValueError("conflicting agent call ledger identity")
    current_calls = current_projected["calls"]
    incoming_calls = incoming_projected["calls"]
    enforce_terminal_call_set_immutable(
        current_status=current_projected["status"],
        current_calls=current_calls,
        incoming_calls=incoming_calls,
    )
    shorter = min(len(current_calls), len(incoming_calls))
    if any(
        current_calls[index]["callId"] != incoming_calls[index]["callId"]
        for index in range(shorter)
    ):
        raise ValueError("conflicting agent call sequence")
    merged_calls = [
        _merge_receipt(
            current_calls[index],
            incoming_calls[index],
            allow_late_accounting=allow_stopped_late_accounting,
            allow_restart_late_accounting=(
                allow_restart_late_accounting
            ),
        )
        for index in range(shorter)
    ]
    if len(current_calls) > shorter:
        merged_calls.extend(current_calls[shorter:])
    elif len(incoming_calls) > shorter:
        merged_calls.extend(incoming_calls[shorter:])
    merged = {
        "schema": AGENT_CALL_USAGE_SCHEMA,
        "executionId": current_projected["executionId"],
        "status": (
            "failed"
            if (
                allow_restart_late_accounting
                and current_projected["status"] == "failed"
                and incoming_projected["status"] in _LEDGER_TERMINALS
            )
            else _merge_ledger_status(
                current_projected["status"],
                incoming_projected["status"],
            )
        ),
        "calls": merged_calls,
    }
    return project_agent_call_usage(merged)


def enforce_terminal_call_set_immutable(
    *,
    current_status: str,
    current_calls: list[Mapping[str, Any]],
    incoming_calls: list[Mapping[str, Any]],
) -> None:
    """A terminal ledger may fill known receipts but never add dispatches."""

    if current_status not in _LEDGER_TERMINALS:
        return
    known_call_ids = {
        str(item.get("callId") or "")
        for item in current_calls
    }
    if any(
        str(item.get("callId") or "") not in known_call_ids
        for item in incoming_calls
    ):
        raise ValueError("terminal provider call set is immutable")


def fill_usage_once(
    receipt: Any,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    cached_input_tokens: int | None,
    usage_status: str,
) -> None:
    if getattr(receipt, "status", None) == "not_dispatched":
        if usage_status != "unavailable":
            raise RuntimeError("not dispatched agent call is immutable")
        return
    if usage_status == "unavailable":
        return
    for field, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("total_tokens", total_tokens),
        ("cached_input_tokens", cached_input_tokens),
    ):
        if value is not None and getattr(receipt, field) is None:
            setattr(receipt, field, value)
    if usage_status == "reported" and all(
        isinstance(getattr(receipt, field), int)
        and not isinstance(getattr(receipt, field), bool)
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
        )
    ):
        receipt.usage_status = "reported"
    elif receipt.usage_status != "reported":
        receipt.usage_status = "partial"


def normalize_usage(
    usage: Any,
) -> tuple[int | None, int | None, int | None, int | None, str]:
    if usage is None:
        return None, None, None, None, "unavailable"

    def member(source: Any, name: str) -> tuple[bool, Any]:
        if isinstance(source, Mapping):
            return (name in source, source.get(name))
        if hasattr(source, name):
            return (True, getattr(source, name, None))
        return (False, None)

    def nonnegative_int(raw: Any) -> int | None:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw if raw >= 0 else None
        if isinstance(raw, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", raw):
            return int(raw)
        return None

    def value(*names: str) -> int | None:
        for name in names:
            present, raw = member(usage, name)
            if not present or raw is None:
                continue
            parsed = nonnegative_int(raw)
            if parsed is not None:
                return parsed
        return None

    cached_values: list[int] = []
    cached_invalid = False
    for name in (
        "cached_input_tokens",
        "cachedInputTokens",
        "prompt_cache_hit_tokens",
        "cached_prompt_tokens",
        "cache_read_input_tokens",
        "cache_read_tokens",
    ):
        present, raw = member(usage, name)
        if not present or raw is None:
            continue
        parsed = nonnegative_int(raw)
        if parsed is None:
            cached_invalid = True
        else:
            cached_values.append(parsed)
    for details_name in ("prompt_tokens_details", "input_tokens_details"):
        present, details = member(usage, details_name)
        if not present or details is None:
            continue
        cached_present, raw = member(details, "cached_tokens")
        if not cached_present or raw is None:
            continue
        parsed = nonnegative_int(raw)
        if parsed is None:
            cached_invalid = True
        else:
            cached_values.append(parsed)
    cached_input_tokens = (
        cached_values[0]
        if (
            not cached_invalid
            and cached_values
            and len(set(cached_values)) == 1
        )
        else None
    )
    input_tokens = value("input_tokens", "prompt_tokens")
    output_tokens = value("output_tokens", "completion_tokens")
    total_tokens = value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if (
        input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
        and total_tokens != input_tokens + output_tokens
    ):
        total_tokens = None
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None, None, None, None, "unavailable"
    if (
        cached_input_tokens is not None
        and input_tokens is not None
        and cached_input_tokens > input_tokens
    ):
        cached_input_tokens = None
    usage_status = (
        "reported"
        if all(
            item is not None
            for item in (
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
            )
        )
        else "partial"
    )
    return (
        input_tokens,
        output_tokens,
        total_tokens,
        cached_input_tokens,
        usage_status,
    )


def safe_category(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(value or ""))[:80]
    return cleaned or "unknown"


def project_route(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_ROUTE.fullmatch(text):
        raise ValueError(f"invalid agent call {field}")
    return text


def _project_receipt(
    value: Any,
    *,
    execution_id: str,
) -> dict[str, Any]:
    required = {
        "callId",
        "ordinal",
        "provider",
        "model",
        "role",
        "startedAtMs",
        "endedAtMs",
        "status",
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cachedInputTokens",
        "usageStatus",
    }
    optional = {"errorCategory", "costUsd", "costStatus", "costSource"}
    if not isinstance(value, Mapping) or not required.issubset(value) or (
        set(value) - required - optional
    ):
        raise ValueError("invalid agent call receipt")
    ordinal = _nonnegative_int(value.get("ordinal"), required=True)
    started_at_ms = _nonnegative_int(value.get("startedAtMs"), required=True)
    ended_at_ms = _nonnegative_int(value.get("endedAtMs"))
    status = str(value.get("status") or "")
    usage_status = str(value.get("usageStatus") or "")
    projected: dict[str, Any] = {
        "callId": str(value.get("callId") or ""),
        "ordinal": ordinal,
        "provider": project_route(value.get("provider"), field="provider"),
        "model": project_route(value.get("model"), field="model"),
        "role": project_route(value.get("role"), field="role"),
        "startedAtMs": started_at_ms,
        "endedAtMs": ended_at_ms,
        "status": status,
        "inputTokens": _nonnegative_int(value.get("inputTokens")),
        "outputTokens": _nonnegative_int(value.get("outputTokens")),
        "totalTokens": _nonnegative_int(value.get("totalTokens")),
        "cachedInputTokens": _nonnegative_int(
            value.get("cachedInputTokens")
        ),
        "usageStatus": usage_status,
    }
    if (
        ordinal < 1
        or ordinal > AGENT_CALL_LIMIT
        or projected["callId"]
        != f"{execution_id}:call:{ordinal:06d}"
        or status not in {*_RECEIPT_ACTIVE, *_RECEIPT_TERMINALS}
        or usage_status not in {"unavailable", "partial", "reported"}
        or (ended_at_ms is not None and ended_at_ms < started_at_ms)
    ):
        raise ValueError("invalid agent call receipt")
    tokens = (
        projected["inputTokens"],
        projected["outputTokens"],
        projected["totalTokens"],
        projected["cachedInputTokens"],
    )
    if usage_status == "unavailable" and any(item is not None for item in tokens):
        raise ValueError("unavailable agent call usage has counters")
    if usage_status == "reported" and not all(item is not None for item in tokens):
        raise ValueError("reported agent call usage is incomplete")
    if usage_status == "partial" and (
        not any(item is not None for item in tokens)
        or all(item is not None for item in tokens)
    ):
        raise ValueError("invalid partial agent call usage")
    if (
        all(item is not None for item in tokens[:3])
        and projected["totalTokens"]
        != projected["inputTokens"] + projected["outputTokens"]
    ):
        raise ValueError("inconsistent agent call token total")
    if (
        projected["cachedInputTokens"] is not None
        and projected["inputTokens"] is not None
        and projected["cachedInputTokens"] > projected["inputTokens"]
    ):
        raise ValueError("invalid agent call cached input")
    for field in ("errorCategory", "costStatus", "costSource"):
        if value.get(field) is not None:
            text = str(value.get(field) or "")
            if not _SAFE_CATEGORY.fullmatch(text):
                raise ValueError("invalid agent call category")
            projected[field] = text
    if value.get("costUsd") is not None:
        amount = value.get("costUsd")
        if isinstance(amount, bool):
            raise ValueError("invalid agent call cost")
        try:
            parsed = float(amount)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid agent call cost") from exc
        if parsed < 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
            raise ValueError("invalid agent call cost")
        projected["costUsd"] = parsed
    if status in _RECEIPT_ACTIVE and ended_at_ms is not None:
        raise ValueError("active agent call receipt has end time")
    if status in _RECEIPT_TERMINALS and ended_at_ms is None:
        raise ValueError("terminal agent call receipt has no end time")
    if status == "not_dispatched" and (
        ended_at_ms is None
        or usage_status != "unavailable"
        or any(item is not None for item in tokens)
        or projected.get("errorCategory")
        != "provider_dispatch_fence_closed"
        or any(
            projected.get(field) is not None
            for field in ("costUsd", "costStatus", "costSource")
        )
    ):
        raise ValueError("invalid not-dispatched agent call receipt")
    return projected


def _merge_receipt(
    current: Mapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    allow_late_accounting: bool,
    allow_restart_late_accounting: bool,
) -> dict[str, Any]:
    for field in ("callId", "ordinal", "provider", "model", "role", "startedAtMs"):
        if current.get(field) != incoming.get(field):
            raise ValueError("conflicting agent call receipt identity")
    current_status = str(current.get("status") or "")
    incoming_status = str(incoming.get("status") or "")
    trusted_restart_fence = bool(
        allow_restart_late_accounting
        and current_status in {"failed", "timed_out"}
        and current.get("errorCategory")
        == "agent_restart_outcome_unknown"
    )
    trusted_late_accounting = bool(
        (
            allow_late_accounting
            and current_status in {"cancelled", "timed_out"}
        )
        or trusted_restart_fence
    )
    if current_status == incoming_status:
        status = current_status
    elif current_status == "reserved" and incoming_status in {
        "running",
        "not_dispatched",
    }:
        status = incoming_status
    elif current_status == "running" and incoming_status == "reserved":
        status = current_status
    elif current_status == "running" and incoming_status in (
        _RECEIPT_TERMINALS - {"not_dispatched"}
    ):
        status = incoming_status
    elif (
        current_status in _RECEIPT_TERMINALS
        and incoming_status in _RECEIPT_ACTIVE
        and not (
            current_status == "not_dispatched"
            and incoming_status == "running"
        )
    ):
        status = current_status
    elif (
        trusted_late_accounting
        and incoming_status in (
            _RECEIPT_TERMINALS - {"not_dispatched"}
        )
    ):
        status = current_status
    else:
        raise ValueError("conflicting terminal agent call state")
    merged = dict(current)
    merged["status"] = status
    for field in (
        "endedAtMs",
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cachedInputTokens",
        "costUsd",
        "errorCategory",
        "costStatus",
        "costSource",
    ):
        current_value = current.get(field)
        incoming_value = incoming.get(field)
        if (
            trusted_late_accounting
            and field in {"endedAtMs", "errorCategory"}
            and current_value is not None
        ):
            value = current_value
        elif current_value is None:
            value = incoming_value
        elif incoming_value is None or incoming_value == current_value:
            value = current_value
        else:
            raise ValueError(f"conflicting agent call {field}")
        if value is not None:
            merged[field] = value
    tokens = (
        merged.get("inputTokens"),
        merged.get("outputTokens"),
        merged.get("totalTokens"),
        merged.get("cachedInputTokens"),
    )
    merged["usageStatus"] = (
        "reported"
        if all(item is not None for item in tokens)
        else "partial"
        if any(item is not None for item in tokens)
        else "unavailable"
    )
    return merged


def _merge_ledger_status(current: str, incoming: str) -> str:
    if current == incoming:
        return current
    if current == "running":
        return incoming
    if incoming == "running":
        return current
    raise ValueError("conflicting terminal agent call ledger state")


def fill_cost_once(
    receipt: Any,
    *,
    cost_usd: float | None,
    cost_status: str | None,
    cost_source: str | None,
) -> None:
    if getattr(receipt, "status", None) == "not_dispatched":
        if any(
            value is not None
            for value in (cost_usd, cost_status, cost_source)
        ):
            raise RuntimeError("not dispatched agent call is immutable")
        return
    if cost_usd is not None and receipt.cost_usd is None:
        parsed = float(cost_usd)
        if parsed < 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
            raise ValueError("invalid agent call cost")
        receipt.cost_usd = parsed
    if cost_status and receipt.cost_status is None:
        receipt.cost_status = safe_category(cost_status)
    if cost_source and receipt.cost_source is None:
        receipt.cost_source = safe_category(cost_source)


def _nonnegative_int(value: Any, *, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise ValueError("missing agent call integer")
        return None
    if isinstance(value, bool):
        raise ValueError("invalid agent call integer")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        return int(value)
    raise ValueError("invalid agent call integer")


__all__ = [
    "AGENT_CALL_LIMIT",
    "AGENT_CALL_USAGE_MAX_BYTES",
    "AGENT_CALL_USAGE_SCHEMA",
    "fill_cost_once",
    "fill_usage_once",
    "enforce_terminal_call_set_immutable",
    "merge_agent_call_usage",
    "normalize_usage",
    "project_agent_call_usage",
    "project_route",
    "receipt_dict",
    "safe_category",
]
