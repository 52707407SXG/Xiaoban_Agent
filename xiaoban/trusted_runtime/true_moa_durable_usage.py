"""Monotonic projection and merge rules for durable true-MoA usage."""

from __future__ import annotations

import json
from typing import Any, Mapping

from xiaoban.trusted_runtime.true_moa_durable_shared import (
    TRUE_MOA_DURABLE_MAX_CALLS,
    TRUE_MOA_DURABLE_MAX_FINAL_CALLS,
    TRUE_MOA_DURABLE_USAGE_MAX_BYTES,
    _FIXED_ADVISOR_ORDER,
    _FIXED_SLOT_ORDER,
    _FIXED_SLOTS,
    _LEDGER_STATUS_RANK,
    _LEDGER_TERMINAL_STATES,
    _MODE_EPOCH,
    _RECEIPT_STATUS_RANK,
    _RECEIPT_TERMINAL_STATES,
    _WAVE_ID,
    _safe_nonnegative_float,
    _safe_nonnegative_int,
    _safe_text,
)

def _project_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    projected = {
        "slotId": _safe_text(value.get("slotId"), required=True),
        "callId": _safe_text(value.get("callId"), required=True),
        "provider": _safe_text(value.get("provider"), required=True),
        "model": _safe_text(value.get("model"), required=True),
        "role": _safe_text(value.get("role"), required=True),
        "startedAtMs": _safe_nonnegative_int(value.get("startedAtMs")),
        "endedAtMs": _safe_nonnegative_int(value.get("endedAtMs")),
        "status": _safe_text(value.get("status"), required=True),
        "inputTokens": _safe_nonnegative_int(value.get("inputTokens")),
        "outputTokens": _safe_nonnegative_int(value.get("outputTokens")),
        "totalTokens": _safe_nonnegative_int(value.get("totalTokens")),
        "cachedInputTokens": _safe_nonnegative_int(
            value.get("cachedInputTokens")
        ),
        "usageStatus": _safe_text(value.get("usageStatus"), required=True),
    }
    for name in ("errorCategory", "costStatus", "costSource"):
        if value.get(name) is not None:
            projected[name] = _safe_text(value.get(name), required=True)
    if value.get("costUsd") is not None:
        projected["costUsd"] = _safe_nonnegative_float(value.get("costUsd"))
    fixed_route = _FIXED_SLOTS.get(projected["slotId"])
    if fixed_route is None or (
        projected["provider"],
        projected["model"],
        projected["role"],
    ) != fixed_route:
        raise ValueError("invalid true MoA durable route")
    if (
        projected["status"] not in _RECEIPT_STATUS_RANK
        and projected["status"] not in _RECEIPT_TERMINAL_STATES
    ):
        raise ValueError("invalid true MoA durable receipt state")
    if projected["usageStatus"] not in {"unavailable", "partial", "reported"}:
        raise ValueError("invalid true MoA durable usage state")
    base_token_values = (
        projected["inputTokens"],
        projected["outputTokens"],
        projected["totalTokens"],
    )
    token_values = (
        *base_token_values,
        projected["cachedInputTokens"],
    )
    if projected["usageStatus"] == "unavailable" and any(
        item is not None for item in token_values
    ):
        raise ValueError("unavailable true MoA usage has token values")
    if projected["usageStatus"] == "reported" and not all(
        item is not None for item in token_values
    ):
        raise ValueError("reported true MoA usage is incomplete")
    if projected["usageStatus"] == "partial" and all(
        item is not None for item in token_values
    ):
        raise ValueError("partial true MoA usage is already complete")
    if (
        all(item is not None for item in base_token_values)
        and projected["totalTokens"]
        != projected["inputTokens"] + projected["outputTokens"]
    ):
        raise ValueError("inconsistent true MoA token total")
    if (
        projected["cachedInputTokens"] is not None
        and projected["inputTokens"] is not None
        and projected["cachedInputTokens"] > projected["inputTokens"]
    ):
        raise ValueError("invalid true MoA cached input tokens")
    if (
        projected["endedAtMs"] is not None
        and projected["startedAtMs"] is not None
        and projected["endedAtMs"] < projected["startedAtMs"]
    ):
        raise ValueError("invalid true MoA durable receipt timestamps")
    return projected


def project_true_moa_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid true MoA durable usage ledger")
    if value.get("schema") != "mystand.true-moa.usage.v1":
        raise ValueError("invalid true MoA durable usage schema")
    slots = value.get("slots")
    calls = value.get("calls")
    if not isinstance(slots, list) or len(slots) > 3:
        raise ValueError("invalid true MoA durable slots")
    if (
        not isinstance(calls, list)
        or len(calls) > TRUE_MOA_DURABLE_MAX_CALLS
    ):
        raise ValueError("invalid true MoA durable calls")
    projected = {
        "schema": "mystand.true-moa.usage.v1",
        "waveId": _safe_text(value.get("waveId"), required=True),
        "mode": _safe_text(value.get("mode"), required=True),
        "modeEpoch": _safe_text(value.get("modeEpoch"), required=True),
        "presetId": _safe_text(value.get("presetId"), required=True),
        "presetRevision": _safe_text(
            value.get("presetRevision"),
            required=True,
        ),
        "status": _safe_text(value.get("status"), required=True),
        "slots": [_project_receipt(item) for item in slots],
        "calls": [_project_receipt(item) for item in calls],
    }
    if projected["mode"] != "moa":
        raise ValueError("invalid true MoA durable mode")
    if not _MODE_EPOCH.fullmatch(projected["modeEpoch"]):
        raise ValueError("invalid true MoA durable mode epoch")
    if projected["presetId"] != "mystand-true-moa-v1":
        raise ValueError("invalid true MoA durable preset")
    if projected["presetRevision"] != "2026-07-27.1":
        raise ValueError("invalid true MoA durable preset revision")
    if not _WAVE_ID.fullmatch(projected["waveId"]):
        raise ValueError("invalid true MoA durable wave id")
    if (
        projected["status"] not in _LEDGER_STATUS_RANK
        and projected["status"] not in _LEDGER_TERMINAL_STATES
    ):
        raise ValueError("invalid true MoA durable ledger state")
    if tuple(item["slotId"] for item in projected["slots"]) != _FIXED_SLOT_ORDER:
        raise ValueError("invalid true MoA durable slot set")
    advisor_calls = [
        item for item in projected["calls"] if item["role"] == "advisor"
    ]
    final_calls = [
        item for item in projected["calls"] if item["role"] == "final_executor"
    ]
    if len(final_calls) > TRUE_MOA_DURABLE_MAX_FINAL_CALLS:
        raise ValueError("invalid true MoA durable final call count")
    if projected["calls"] != [*advisor_calls, *final_calls]:
        raise ValueError("invalid true MoA durable call order")
    advisor_slot_ids = tuple(item["slotId"] for item in advisor_calls)
    if (
        len(advisor_slot_ids) != len(set(advisor_slot_ids))
        or advisor_slot_ids
        != tuple(
            slot_id
            for slot_id in _FIXED_ADVISOR_ORDER
            if slot_id in advisor_slot_ids
        )
    ):
        raise ValueError("invalid true MoA durable advisor calls")
    if any(
        item["slotId"] != "final-deepseek-v4-pro"
        for item in final_calls
    ):
        raise ValueError("invalid true MoA durable final calls")
    if any(
        item["startedAtMs"] is None or item["status"] == "not_started"
        for item in projected["calls"]
    ):
        raise ValueError("undispatched true MoA durable call")
    call_ids = [item["callId"] for item in projected["calls"]]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("duplicate true MoA durable call id")
    wave_id = projected["waveId"]
    slot_by_id = {item["slotId"]: item for item in projected["slots"]}
    for item in projected["slots"]:
        if item["callId"] != f"{wave_id}:{item['slotId']}":
            raise ValueError("invalid true MoA durable slot call id")
    if any(
        item["callId"] != f"{wave_id}:{item['slotId']}"
        for item in advisor_calls
    ):
        raise ValueError("invalid true MoA durable advisor call id")
    advisor_call_slots = {item["slotId"] for item in advisor_calls}
    if any(
        slot_by_id[slot_id]["status"] == "completed"
        and slot_id not in advisor_call_slots
        for slot_id in _FIXED_ADVISOR_ORDER
    ) or any(
        slot_by_id[slot_id]["status"] == "not_started"
        for slot_id in advisor_call_slots
    ):
        raise ValueError("advisor call disagrees with true MoA slot lifecycle")
    final_prefix = f"{wave_id}:final-deepseek-v4-pro:"
    if any(
        not item["callId"].startswith(final_prefix)
        for item in final_calls
    ):
        raise ValueError("invalid true MoA durable final call id")
    encoded = json.dumps(
        projected,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > TRUE_MOA_DURABLE_USAGE_MAX_BYTES:
        raise ValueError("true MoA durable usage ledger too large")
    return projected


def _merge_status(
    current: str,
    incoming: str,
    *,
    ranks: Mapping[str, int],
    terminals: set[str],
    stopped_wins: bool = False,
    interrupted_is_provisional: bool = False,
) -> str:
    if current == incoming:
        return current
    if stopped_wins and current == "stopped":
        return current
    if interrupted_is_provisional:
        if current == "interrupted" and incoming in terminals:
            return incoming
        if incoming == "interrupted" and current in terminals:
            return current
    if current in terminals:
        if incoming in ranks:
            return current
        raise ValueError("conflicting terminal true MoA durable state")
    if incoming in terminals:
        return incoming
    if current not in ranks or incoming not in ranks:
        raise ValueError("invalid true MoA durable state transition")
    return current if ranks[current] >= ranks[incoming] else incoming


def _merge_fill_once(current: Any, incoming: Any, *, field: str) -> Any:
    if current is None:
        return incoming
    if incoming is None or incoming == current:
        return current
    raise ValueError(f"conflicting true MoA durable {field}")


def _merge_receipt(
    current: Mapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    allow_stopped_late_accounting: bool = False,
) -> dict[str, Any]:
    for field in ("slotId", "callId", "provider", "model", "role"):
        if current.get(field) != incoming.get(field):
            raise ValueError("conflicting true MoA durable receipt identity")
    merged = dict(current)
    current_status = str(current.get("status") or "")
    incoming_status = str(incoming.get("status") or "")
    merged["status"] = (
        current_status
        if (
            allow_stopped_late_accounting
            and current_status in {"cancelled", "timed_out"}
            and incoming_status in _RECEIPT_TERMINAL_STATES
        )
        else _merge_status(
            current_status,
            incoming_status,
            ranks=_RECEIPT_STATUS_RANK,
            terminals=_RECEIPT_TERMINAL_STATES,
        )
    )
    for field in (
        "startedAtMs",
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
        value = _merge_fill_once(
            current.get(field),
            incoming.get(field),
            field=field,
        )
        if value is not None:
            merged[field] = value
    merged["usageStatus"] = _merge_status(
        str(current.get("usageStatus") or ""),
        str(incoming.get("usageStatus") or ""),
        ranks={"unavailable": 0, "partial": 1, "reported": 2},
        terminals=set(),
    )
    merged_token_values = (
        merged.get("inputTokens"),
        merged.get("outputTokens"),
        merged.get("totalTokens"),
        merged.get("cachedInputTokens"),
    )
    if all(item is not None for item in merged_token_values):
        merged["usageStatus"] = "reported"
    elif any(item is not None for item in merged_token_values):
        merged["usageStatus"] = "partial"
    else:
        merged["usageStatus"] = "unavailable"
    return merged


def _merge_usage(
    current: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
    *,
    allow_stopped_late_accounting: bool = False,
) -> dict[str, Any]:
    if current is None:
        return project_true_moa_usage(incoming)
    current_projected = project_true_moa_usage(current)
    incoming_projected = project_true_moa_usage(incoming)
    for field in (
        "schema",
        "waveId",
        "mode",
        "modeEpoch",
        "presetId",
        "presetRevision",
    ):
        if current_projected[field] != incoming_projected[field]:
            raise ValueError("conflicting true MoA durable ledger identity")
    merged = dict(current_projected)
    merged["status"] = (
        current_projected["status"]
        if (
            allow_stopped_late_accounting
            and current_projected["status"] == "cancelled"
            and incoming_projected["status"] in _LEDGER_TERMINAL_STATES
        )
        else _merge_status(
            current_projected["status"],
            incoming_projected["status"],
            ranks=_LEDGER_STATUS_RANK,
            terminals=_LEDGER_TERMINAL_STATES,
        )
    )
    merged["slots"] = [
        _merge_receipt(
            current_item,
            incoming_item,
            allow_stopped_late_accounting=allow_stopped_late_accounting,
        )
        for current_item, incoming_item in zip(
            current_projected["slots"],
            incoming_projected["slots"],
            strict=True,
        )
    ]
    current_calls = {
        item["callId"]: item for item in current_projected["calls"]
    }
    incoming_calls = {
        item["callId"]: item for item in incoming_projected["calls"]
    }
    insertion_order = [
        item["callId"] for item in current_projected["calls"]
    ]
    insertion_order.extend(
        item["callId"]
        for item in incoming_projected["calls"]
        if item["callId"] not in current_calls
    )
    merged_calls = {
        call_id: (
            _merge_receipt(
                current_calls[call_id],
                incoming_calls[call_id],
                allow_stopped_late_accounting=(
                    allow_stopped_late_accounting
                ),
            )
            if call_id in current_calls and call_id in incoming_calls
            else dict(current_calls.get(call_id) or incoming_calls[call_id])
        )
        for call_id in insertion_order
    }
    advisor_call_ids = {
        item["slotId"]: item["callId"]
        for item in merged_calls.values()
        if item["role"] == "advisor"
    }
    final_call_ids = [
        call_id
        for call_id in insertion_order
        if merged_calls[call_id]["role"] == "final_executor"
    ]
    call_order = [
        advisor_call_ids[slot_id]
        for slot_id in _FIXED_ADVISOR_ORDER
        if slot_id in advisor_call_ids
    ]
    call_order.extend(final_call_ids)
    merged["calls"] = [
        merged_calls[call_id]
        for call_id in call_order
    ]
    return project_true_moa_usage(merged)
