"""Plaintext-free durable idempotency and usage receipts for true MoA."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping


TRUE_MOA_DURABLE_USAGE_MAX_BYTES = 64 * 1024
TRUE_MOA_DURABLE_MAX_ROWS = 100_000
TRUE_MOA_DURABLE_MAX_CALLS = 10
TRUE_MOA_DURABLE_MAX_FINAL_CALLS = 8
TRUE_MOA_COMPLETED_OUTCOME_SCHEMA = (
    "mystand.true-moa.completed-outcome.v1"
)
TRUE_MOA_OUTCOME_BINDING_SCHEMA = "mystand.true-moa.outcome-binding.v1"
TRUE_MOA_OUTCOME_MAX_TEXT_BYTES = 64 * 1024
TRUE_MOA_OUTCOME_MAX_VERIFICATION_BYTES = 16 * 1024
TRUE_MOA_OUTCOME_MAX_PLAINTEXT_BYTES = 96 * 1024
TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS = 24 * 60 * 60
TRUE_MOA_OUTCOME_MAX_TTL_SECONDS = 7 * 24 * 60 * 60
_VALID_KINDS = {"binding", "execution"}
_VALID_STATES = {
    "claimed",
    "running",
    "completed",
    "failed",
    "stopped",
    "interrupted",
}
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_.:+-]{0,180}$")
_WAVE_ID = re.compile(r"^[a-f0-9]{32}$")
_MODE_EPOCH = re.compile(r"^(?:0|[1-9][0-9]{0,18})$")
_LEDGER_STATUS_RANK = {
    "pending": 0,
    "running": 1,
    "advisors_completed": 2,
}
_LEDGER_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_RECEIPT_STATUS_RANK = {"not_started": 0, "running": 1}
_RECEIPT_TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
    "timed_out",
}
_DURABLE_STATE_RANK = {"claimed": 0, "running": 1}
_DURABLE_TERMINAL_STATES = {
    "completed",
    "failed",
    "stopped",
    "interrupted",
}
_FIXED_SLOTS = {
    "advisor-kimi-k3": ("kimi-coding", "k3", "advisor"),
    "advisor-deepseek-v4-pro": (
        "deepseek",
        "deepseek-v4-pro",
        "advisor",
    ),
    "final-deepseek-v4-pro": (
        "deepseek",
        "deepseek-v4-pro",
        "final_executor",
    ),
}
_FIXED_SLOT_ORDER = tuple(_FIXED_SLOTS)
_FIXED_ADVISOR_ORDER = _FIXED_SLOT_ORDER[:2]
_OUTCOME_KEY_ID = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
_OUTCOME_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_DATASCOPE_FINGERPRINT = re.compile(r"^[a-f0-9]{16}$")
_OUTCOME_STATES = {"none", "sealed", "acked", "expired"}
_OUTCOME_BINDING_FIELDS = {
    "schema",
    "siteId",
    "userId",
    "deliveryId",
    "messageId",
    "attempt",
    "requestFingerprint",
    "datascopeFingerprint",
    "modeEpoch",
    "presetId",
    "presetRevision",
}


class TrueMoAOutcomeError(RuntimeError):
    """Base class for fail-closed sealed-outcome failures."""


class TrueMoAOutcomeUnavailableError(TrueMoAOutcomeError):
    """The dedicated outcome key or sealed result is unavailable."""


class TrueMoAOutcomeBindingError(TrueMoAOutcomeError):
    """The caller identity, ciphertext, or terminal outcome did not verify."""


def default_true_moa_durable_path() -> str:
    explicit = os.environ.get("XIAOBAN_TRUE_MOA_LEDGER_DB", "").strip()
    if explicit:
        return explicit
    home = Path(
        os.environ.get("XIAOBAN_HOME", "").strip()
        or Path.home() / ".xiaoban"
    ).expanduser()
    return str(home / "state" / "true-moa-idempotency.sqlite")


def _storage_key(value: str) -> str:
    return hashlib.sha256(
        f"mystand-true-moa-durable-v1\0{str(value or '')}".encode("utf-8")
    ).hexdigest()


def _safe_text(value: Any, *, required: bool = False) -> str:
    text = str(value or "").strip()
    if (required and not text) or not _SAFE_TEXT.fullmatch(text):
        raise ValueError("invalid true MoA durable ledger text")
    return text


def _safe_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("invalid true MoA durable integer")
    if isinstance(value, int):
        if value >= 0:
            return value
        raise ValueError("invalid true MoA durable integer")
    if isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        return int(value)
    raise ValueError("invalid true MoA durable integer")


def _safe_nonnegative_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("invalid true MoA durable number")
    parsed = float(value)
    if parsed < 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError("invalid true MoA durable number")
    return parsed


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid true MoA sealed outcome JSON") from exc
    return encoded.encode("utf-8")


def _decode_outcome_key(value: str) -> bytes:
    encoded = str(value or "").strip()
    if not encoded or len(encoded) > 128:
        raise ValueError("invalid true MoA outcome key")
    padded = encoded + ("=" * (-len(encoded) % 4))
    try:
        decoded = base64.b64decode(
            padded.encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("invalid true MoA outcome key") from exc
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if len(decoded) != 32 or canonical != encoded.rstrip("="):
        raise ValueError("invalid true MoA outcome key")
    return decoded


def _validated_outcome_keyring(
    value: Mapping[str, bytes] | None,
) -> dict[str, bytes]:
    if value is None:
        raw = str(
            os.environ.get("XIAOBAN_TRUE_MOA_OUTCOME_KEYS", "") or ""
        ).strip()
        if not raw:
            return {}
        if len(raw) > 4096 or re.search(r"[\r\n\x00]", raw):
            raise ValueError("invalid true MoA outcome keyring")
        parsed: dict[str, bytes] = {}
        for item in raw.split(","):
            key_id, separator, encoded = item.partition(":")
            if (
                not separator
                or not _OUTCOME_KEY_ID.fullmatch(key_id)
                or key_id in parsed
            ):
                raise ValueError("invalid true MoA outcome keyring")
            parsed[key_id] = _decode_outcome_key(encoded)
        value = parsed
    if not isinstance(value, Mapping) or len(value) > 8:
        raise ValueError("invalid true MoA outcome keyring")
    projected: dict[str, bytes] = {}
    for raw_key_id, raw_key in value.items():
        key_id = str(raw_key_id or "")
        if (
            not _OUTCOME_KEY_ID.fullmatch(key_id)
            or key_id in projected
            or not isinstance(raw_key, bytes)
            or len(raw_key) != 32
        ):
            raise ValueError("invalid true MoA outcome keyring")
        projected[key_id] = bytes(raw_key)
    return projected


def _validated_outcome_ttl(value: int | None) -> int:
    raw: Any = value
    if raw is None:
        raw = os.environ.get(
            "XIAOBAN_TRUE_MOA_OUTCOME_TTL_SECONDS",
            str(TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS),
        )
    if isinstance(raw, bool):
        raise ValueError("invalid true MoA outcome TTL")
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid true MoA outcome TTL") from exc
    if parsed < 1 or parsed > TRUE_MOA_OUTCOME_MAX_TTL_SECONDS:
        raise ValueError("invalid true MoA outcome TTL")
    return parsed


def project_true_moa_outcome_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OUTCOME_BINDING_FIELDS:
        raise ValueError("invalid true MoA outcome binding")
    projected = {
        "schema": str(value.get("schema") or ""),
        "siteId": str(value.get("siteId") or ""),
        "userId": str(value.get("userId") or ""),
        "deliveryId": str(value.get("deliveryId") or ""),
        "messageId": str(value.get("messageId") or ""),
        "attempt": _safe_nonnegative_int(value.get("attempt")),
        "requestFingerprint": str(
            value.get("requestFingerprint") or ""
        ).lower(),
        "datascopeFingerprint": str(
            value.get("datascopeFingerprint") or ""
        ).lower(),
        "modeEpoch": str(value.get("modeEpoch") or ""),
        "presetId": str(value.get("presetId") or ""),
        "presetRevision": str(value.get("presetRevision") or ""),
    }
    if (
        projected["schema"] != TRUE_MOA_OUTCOME_BINDING_SCHEMA
        or not re.fullmatch(
            r"[A-Za-z0-9._:@-]{1,120}",
            projected["siteId"],
        )
        or not re.fullmatch(
            r"[A-Za-z0-9._:@-]{1,200}",
            projected["userId"],
        )
        or not projected["deliveryId"]
        or len(projected["deliveryId"]) > 512
        or re.search(r"[\r\n\x00]", projected["deliveryId"])
        or not projected["messageId"]
        or len(projected["messageId"]) > 200
        or re.search(r"[\r\n\x00]", projected["messageId"])
        or projected["attempt"] is None
        or not _OUTCOME_DIGEST.fullmatch(projected["requestFingerprint"])
        or not _DATASCOPE_FINGERPRINT.fullmatch(
            projected["datascopeFingerprint"]
        )
        or not _MODE_EPOCH.fullmatch(projected["modeEpoch"])
        or projected["presetId"] != "mystand-true-moa-v1"
        or projected["presetRevision"] != "2026-07-27.1"
    ):
        raise ValueError("invalid true MoA outcome binding")
    return projected


def _project_trusted_verification(
    value: Any,
    *,
    output_digest: str,
    binding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("invalid true MoA trusted verification")
    encoded = _canonical_json_bytes(value)
    if len(encoded) > TRUE_MOA_OUTCOME_MAX_VERIFICATION_BYTES:
        raise ValueError("true MoA trusted verification is too large")
    try:
        projected = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid true MoA trusted verification") from exc
    if (
        not isinstance(projected, dict)
        or projected.get("schema")
        != "mystand.xiaoban-fact-verification.v1"
        or projected.get("verified") is not True
        or projected.get("output_digest") != output_digest
    ):
        raise ValueError("invalid true MoA trusted verification")
    if binding is not None and (
        projected.get("delivery_id") != binding["deliveryId"]
        or projected.get("message_id") != binding["messageId"]
        or projected.get("attempt") != binding["attempt"]
        or projected.get("request_fingerprint")
        != binding["requestFingerprint"]
        or projected.get("datascope_fingerprint")
        != binding["datascopeFingerprint"]
    ):
        raise TrueMoAOutcomeBindingError(
            "true MoA trusted verification binding mismatch"
        )
    return projected


def project_true_moa_completed_outcome(
    value: Mapping[str, Any],
    *,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid true MoA completed outcome")
    expected = {
        "schema",
        "completed",
        "finalResponse",
        "outputDigest",
        "factGuardRequired",
    }
    if value.get("trustedVerification") is not None:
        expected.add("trustedVerification")
    if set(value) != expected:
        raise ValueError("invalid true MoA completed outcome")
    final_response = value.get("finalResponse")
    if not isinstance(final_response, str):
        raise ValueError("invalid true MoA completed outcome")
    final_bytes = final_response.encode("utf-8")
    output_digest = str(value.get("outputDigest") or "").lower()
    fact_guard_required = value.get("factGuardRequired")
    projected_binding = (
        project_true_moa_outcome_binding(binding)
        if binding is not None
        else None
    )
    if (
        value.get("schema") != TRUE_MOA_COMPLETED_OUTCOME_SCHEMA
        or value.get("completed") is not True
        or not final_response.strip()
        or len(final_bytes) > TRUE_MOA_OUTCOME_MAX_TEXT_BYTES
        or not _OUTCOME_DIGEST.fullmatch(output_digest)
        or output_digest != hashlib.sha256(final_bytes).hexdigest()
        or not isinstance(fact_guard_required, bool)
    ):
        raise ValueError("invalid true MoA completed outcome")
    verification = _project_trusted_verification(
        value.get("trustedVerification"),
        output_digest=output_digest,
        binding=projected_binding,
    )
    if fact_guard_required and verification is None:
        raise ValueError("true MoA fact outcome lacks trusted verification")
    projected: dict[str, Any] = {
        "schema": TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
        "completed": True,
        "finalResponse": final_response,
        "outputDigest": output_digest,
        "factGuardRequired": fact_guard_required,
    }
    if verification is not None:
        projected["trustedVerification"] = verification
    if len(_canonical_json_bytes(projected)) > TRUE_MOA_OUTCOME_MAX_PLAINTEXT_BYTES:
        raise ValueError("true MoA completed outcome is too large")
    return projected


def _true_moa_outcome_aad(
    *,
    storage_key: str,
    fingerprint: str,
    usage: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> bytes:
    projected_usage = project_true_moa_usage(usage)
    projected_binding = project_true_moa_outcome_binding(binding)
    if (
        projected_binding["modeEpoch"] != projected_usage["modeEpoch"]
        or projected_binding["presetId"] != projected_usage["presetId"]
        or projected_binding["presetRevision"]
        != projected_usage["presetRevision"]
    ):
        raise TrueMoAOutcomeBindingError(
            "true MoA outcome snapshot binding mismatch"
        )
    if not _OUTCOME_DIGEST.fullmatch(storage_key):
        raise ValueError("invalid true MoA outcome storage key")
    aad = {
        "schema": "mystand.true-moa.outcome-aad.v1",
        "storageKey": storage_key,
        "durableFingerprint": _safe_text(fingerprint, required=True),
        "waveId": projected_usage["waveId"],
        "binding": projected_binding,
    }
    return _canonical_json_bytes(aad)


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
) -> dict[str, Any]:
    for field in ("slotId", "callId", "provider", "model", "role"):
        if current.get(field) != incoming.get(field):
            raise ValueError("conflicting true MoA durable receipt identity")
    merged = dict(current)
    merged["status"] = _merge_status(
        str(current.get("status") or ""),
        str(incoming.get("status") or ""),
        ranks=_RECEIPT_STATUS_RANK,
        terminals=_RECEIPT_TERMINAL_STATES,
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
    merged["status"] = _merge_status(
        current_projected["status"],
        incoming_projected["status"],
        ranks=_LEDGER_STATUS_RANK,
        terminals=_LEDGER_TERMINAL_STATES,
    )
    merged["slots"] = [
        _merge_receipt(current_item, incoming_item)
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
            _merge_receipt(current_calls[call_id], incoming_calls[call_id])
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


class TrueMoADurableStore:
    """SQLite usage ledger with a separately-keyed sealed outcome envelope."""

    def __init__(
        self,
        path: str,
        *,
        outcome_keys: Mapping[str, bytes] | None = None,
        active_outcome_key_id: str | None = None,
        outcome_ttl_seconds: int | None = None,
    ):
        raw_path = str(path or "").strip()
        if not raw_path:
            raise ValueError("true MoA durable ledger path is required")
        self.path = Path(raw_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._lock_path = Path(f"{self.path}.lock")
        self._lock_handle = self._lock_path.open("a+b")
        self._lock_path.chmod(0o600)
        self._outcome_key_error: Exception | None = None
        try:
            self._outcome_keys = _validated_outcome_keyring(outcome_keys)
            self._outcome_ttl_seconds = _validated_outcome_ttl(
                outcome_ttl_seconds
            )
            active_key_id = str(active_outcome_key_id or "")
            if active_key_id:
                if active_key_id not in self._outcome_keys:
                    raise ValueError("invalid true MoA active outcome key")
                self._active_outcome_key_id = active_key_id
            else:
                self._active_outcome_key_id = next(
                    iter(self._outcome_keys),
                    "",
                )
        except ValueError as exc:
            # Usage receipts remain available, but true-MoA preflight checks
            # outcome_ready and fails before a paid dispatch.
            self._outcome_keys = {}
            self._active_outcome_key_id = ""
            self._outcome_ttl_seconds = TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS
            self._outcome_key_error = exc
        try:
            fcntl.flock(
                self._lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as exc:
            self._lock_handle.close()
            raise RuntimeError(
                "true MoA durable ledger is already owned by another process"
            ) from exc
        try:
            self._initialize()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        handle = getattr(self, "_lock_handle", None)
        if handle is None or handle.closed:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    @property
    def outcome_ready(self) -> bool:
        return bool(
            self._active_outcome_key_id
            and self._active_outcome_key_id in self._outcome_keys
            and self._outcome_key_error is None
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _outcome_key(self, key_id: str, storage_key: str) -> bytes:
        master_key = self._outcome_keys.get(str(key_id or ""))
        if master_key is None:
            raise TrueMoAOutcomeUnavailableError(
                "true MoA outcome key is unavailable"
            )
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=bytes.fromhex(storage_key),
            info=(
                b"mystand.true-moa.completed-outcome.v1\0"
                + str(key_id).encode("ascii")
            ),
        ).derive(master_key)

    def _encrypt_outcome(
        self,
        *,
        storage_key: str,
        fingerprint: str,
        usage: Mapping[str, Any],
        outcome: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> tuple[str, bytes, bytes, str]:
        if not self.outcome_ready:
            raise TrueMoAOutcomeUnavailableError(
                "true MoA outcome key is unavailable"
            )
        projected_binding = project_true_moa_outcome_binding(binding)
        projected = project_true_moa_completed_outcome(
            outcome,
            binding=projected_binding,
        )
        plaintext = _canonical_json_bytes(projected)
        aad = _true_moa_outcome_aad(
            storage_key=storage_key,
            fingerprint=fingerprint,
            usage=usage,
            binding=projected_binding,
        )
        key_id = self._active_outcome_key_id
        nonce = os.urandom(12)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        ciphertext = AESGCM(
            self._outcome_key(key_id, storage_key)
        ).encrypt(nonce, plaintext, aad)
        receipt = hashlib.sha256(
            key_id.encode("ascii") + nonce + ciphertext
        ).hexdigest()
        return key_id, nonce, ciphertext, receipt

    def _decrypt_outcome_row(
        self,
        row: sqlite3.Row,
        *,
        storage_key: str,
        usage: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if str(row["outcome_state"] or "") != "sealed":
            raise TrueMoAOutcomeUnavailableError(
                "true MoA completed outcome is unavailable"
            )
        key_id = str(row["outcome_key_id"] or "")
        nonce = bytes(row["outcome_nonce"] or b"")
        ciphertext = bytes(row["outcome_ciphertext"] or b"")
        receipt = str(row["outcome_receipt"] or "")
        if (
            not _OUTCOME_KEY_ID.fullmatch(key_id)
            or len(nonce) != 12
            or len(ciphertext) < 16
            or not _OUTCOME_DIGEST.fullmatch(receipt)
            or receipt
            != hashlib.sha256(
                key_id.encode("ascii") + nonce + ciphertext
            ).hexdigest()
        ):
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA sealed outcome envelope"
            )
        aad = _true_moa_outcome_aad(
            storage_key=storage_key,
            fingerprint=str(row["fingerprint"] or ""),
            usage=usage,
            binding=binding,
        )
        binding_digest = str(row["outcome_binding_digest"] or "")
        if (
            not _OUTCOME_DIGEST.fullmatch(binding_digest)
            or binding_digest != hashlib.sha256(aad).hexdigest()
        ):
            raise TrueMoAOutcomeBindingError(
                "true MoA sealed outcome binding mismatch"
            )
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        try:
            plaintext = AESGCM(
                self._outcome_key(key_id, storage_key)
            ).decrypt(nonce, ciphertext, aad)
        except InvalidTag as exc:
            raise TrueMoAOutcomeBindingError(
                "true MoA sealed outcome authentication failed"
            ) from exc
        if len(plaintext) > TRUE_MOA_OUTCOME_MAX_PLAINTEXT_BYTES:
            raise TrueMoAOutcomeBindingError(
                "true MoA sealed outcome is too large"
            )
        try:
            decoded = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA sealed outcome payload"
            ) from exc
        try:
            projected = project_true_moa_completed_outcome(
                decoded,
                binding=binding,
            )
        except ValueError as exc:
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA sealed outcome payload"
            ) from exc
        return projected, receipt

    def _harden_files(self) -> None:
        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            try:
                if path.exists():
                    path.chmod(0o600)
            except OSError:
                pass

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS true_moa_idempotency (
                    storage_key TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    usage_json TEXT NOT NULL DEFAULT '',
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_true_moa_idem_updated
                    ON true_moa_idempotency(updated_at_ms);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(true_moa_idempotency)"
                )
            }
            migrations = {
                "outcome_state": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_state TEXT NOT NULL DEFAULT 'none'"
                ),
                "outcome_key_id": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_key_id TEXT NOT NULL DEFAULT ''"
                ),
                "outcome_nonce": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_nonce BLOB NOT NULL DEFAULT X''"
                ),
                "outcome_ciphertext": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_ciphertext BLOB NOT NULL DEFAULT X''"
                ),
                "outcome_receipt": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_receipt TEXT NOT NULL DEFAULT ''"
                ),
                "outcome_binding_digest": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_binding_digest TEXT NOT NULL DEFAULT ''"
                ),
                "outcome_expires_at_ms": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_expires_at_ms INTEGER NOT NULL DEFAULT 0"
                ),
                "outcome_acked_at_ms": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_acked_at_ms INTEGER NOT NULL DEFAULT 0"
                ),
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = 'interrupted', updated_at_ms = ?
                WHERE kind = 'execution'
                  AND state IN ('claimed', 'running')
                """,
                (int(time.time() * 1000),),
            )
        self._harden_files()

    def get(self, key: str) -> dict[str, Any] | None:
        storage_key = _storage_key(key)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM true_moa_idempotency WHERE storage_key = ?",
                (storage_key,),
            ).fetchone()
        if row is None:
            return None
        usage = None
        if row["usage_json"]:
            usage = project_true_moa_usage(json.loads(row["usage_json"]))
        return {
            "fingerprint": row["fingerprint"],
            "kind": row["kind"],
            "state": row["state"],
            "usage": usage,
            "outcomeState": str(row["outcome_state"] or "none"),
            "outcomeExpiresAtMs": int(row["outcome_expires_at_ms"] or 0),
        }

    def claim(self, key: str, fingerprint: str, *, kind: str) -> str:
        clean_fingerprint = _safe_text(fingerprint, required=True)
        if kind not in _VALID_KINDS:
            raise ValueError("invalid true MoA durable claim kind")
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT fingerprint, kind FROM true_moa_idempotency WHERE storage_key = ?",
                (storage_key,),
            ).fetchone()
            if row is not None:
                existing_fingerprint = str(row["fingerprint"] or "")
                if existing_fingerprint and existing_fingerprint != clean_fingerprint:
                    connection.rollback()
                    return "conflict"
                if row["kind"] != kind:
                    connection.rollback()
                    return "conflict"
                if not existing_fingerprint:
                    connection.execute(
                        """
                        UPDATE true_moa_idempotency
                        SET fingerprint = ?, updated_at_ms = ?
                        WHERE storage_key = ?
                        """,
                        (clean_fingerprint, timestamp, storage_key),
                    )
                connection.commit()
                self._harden_files()
                return "reusable"
            row_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM true_moa_idempotency"
                ).fetchone()[0]
            )
            if row_count >= TRUE_MOA_DURABLE_MAX_ROWS:
                connection.rollback()
                raise RuntimeError("true MoA durable ledger capacity exhausted")
            connection.execute(
                """
                INSERT INTO true_moa_idempotency (
                    storage_key, fingerprint, kind, state,
                    usage_json, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, 'claimed', '', ?, ?)
                """,
                (
                    storage_key,
                    clean_fingerprint,
                    kind,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        self._harden_files()
        return "missing"

    def save_usage(
        self,
        key: str,
        fingerprint: str,
        usage: Mapping[str, Any],
        *,
        state: str,
    ) -> None:
        if state not in _VALID_STATES:
            raise ValueError("invalid true MoA durable state")
        projected = project_true_moa_usage(usage)
        storage_key = _storage_key(key)
        clean_fingerprint = _safe_text(fingerprint, required=True)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT fingerprint, kind, state, usage_json
                FROM true_moa_idempotency
                WHERE storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RuntimeError("true MoA durable execution was not claimed")
            if (
                row["kind"] != "execution"
                or row["fingerprint"] != clean_fingerprint
            ):
                connection.rollback()
                raise RuntimeError("true MoA durable execution binding conflict")
            existing_usage = (
                project_true_moa_usage(json.loads(row["usage_json"]))
                if row["usage_json"]
                else None
            )
            projected = _merge_usage(existing_usage, projected)
            encoded = json.dumps(
                projected,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            merged_state = _merge_status(
                str(row["state"] or ""),
                state,
                ranks=_DURABLE_STATE_RANK,
                terminals=_DURABLE_TERMINAL_STATES,
                stopped_wins=True,
                interrupted_is_provisional=True,
            )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = ?, usage_json = ?, updated_at_ms = ?
                WHERE storage_key = ?
                """,
                (merged_state, encoded, timestamp, storage_key),
            )
            connection.commit()
        self._harden_files()

    def save_completed_outcome(
        self,
        key: str,
        fingerprint: str,
        usage: Mapping[str, Any],
        outcome: Mapping[str, Any],
        *,
        binding: Mapping[str, Any],
    ) -> str:
        """Atomically commit terminal usage and one encrypted visible result."""

        projected_usage = project_true_moa_usage(usage)
        if projected_usage["status"] != "completed":
            raise ValueError("true MoA outcome requires completed usage")
        projected_binding = project_true_moa_outcome_binding(binding)
        projected_outcome = project_true_moa_completed_outcome(
            outcome,
            binding=projected_binding,
        )
        storage_key = _storage_key(key)
        clean_fingerprint = _safe_text(fingerprint, required=True)
        timestamp = int(time.time() * 1000)
        expires_at_ms = timestamp + (self._outcome_ttl_seconds * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM true_moa_idempotency
                WHERE storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RuntimeError("true MoA durable execution was not claimed")
            if (
                row["kind"] != "execution"
                or row["fingerprint"] != clean_fingerprint
            ):
                connection.rollback()
                raise RuntimeError(
                    "true MoA durable execution binding conflict"
                )
            existing_usage = (
                project_true_moa_usage(json.loads(row["usage_json"]))
                if row["usage_json"]
                else None
            )
            merged_usage = _merge_usage(existing_usage, projected_usage)
            merged_state = _merge_status(
                str(row["state"] or ""),
                "completed",
                ranks=_DURABLE_STATE_RANK,
                terminals=_DURABLE_TERMINAL_STATES,
                stopped_wins=True,
                interrupted_is_provisional=True,
            )
            if merged_state != "completed":
                connection.rollback()
                raise TrueMoAOutcomeBindingError(
                    "true MoA terminal fence rejected completed outcome"
                )
            existing_outcome_state = str(row["outcome_state"] or "none")
            if existing_outcome_state == "sealed":
                existing_outcome, existing_receipt = self._decrypt_outcome_row(
                    row,
                    storage_key=storage_key,
                    usage=existing_usage,
                    binding=projected_binding,
                )
                if existing_outcome != projected_outcome:
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "conflicting true MoA completed outcome"
                    )
                if merged_usage != existing_usage:
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "sealed true MoA outcome usage cannot change"
                    )
                encoded = json.dumps(
                    existing_usage,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    """
                    UPDATE true_moa_idempotency
                    SET state = 'completed', usage_json = ?, updated_at_ms = ?
                    WHERE storage_key = ?
                    """,
                    (encoded, timestamp, storage_key),
                )
                connection.commit()
                self._harden_files()
                return existing_receipt
            if existing_outcome_state != "none":
                connection.rollback()
                raise TrueMoAOutcomeBindingError(
                    "true MoA completed outcome is no longer writable"
                )
            key_id, nonce, ciphertext, receipt = self._encrypt_outcome(
                storage_key=storage_key,
                fingerprint=clean_fingerprint,
                usage=merged_usage,
                outcome=projected_outcome,
                binding=projected_binding,
            )
            binding_digest = hashlib.sha256(
                _true_moa_outcome_aad(
                    storage_key=storage_key,
                    fingerprint=clean_fingerprint,
                    usage=merged_usage,
                    binding=projected_binding,
                )
            ).hexdigest()
            encoded = json.dumps(
                merged_usage,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = 'completed',
                    usage_json = ?,
                    outcome_state = 'sealed',
                    outcome_key_id = ?,
                    outcome_nonce = ?,
                    outcome_ciphertext = ?,
                    outcome_receipt = ?,
                    outcome_binding_digest = ?,
                    outcome_expires_at_ms = ?,
                    outcome_acked_at_ms = 0,
                    updated_at_ms = ?
                WHERE storage_key = ?
                """,
                (
                    encoded,
                    key_id,
                    sqlite3.Binary(nonce),
                    sqlite3.Binary(ciphertext),
                    receipt,
                    binding_digest,
                    expires_at_ms,
                    timestamp,
                    storage_key,
                ),
            )
            connection.commit()
        self._harden_files()
        return receipt

    def recover_completed_outcome(
        self,
        key: str,
        *,
        binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Decrypt one owner/snapshot-bound outcome without exposing keys."""

        projected_binding = project_true_moa_outcome_binding(binding)
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            if (
                str(row["state"] or "") != "completed"
                or not row["usage_json"]
            ):
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            usage = project_true_moa_usage(json.loads(row["usage_json"]))
            outcome, receipt = self._decrypt_outcome_row(
                row,
                storage_key=storage_key,
                usage=usage,
                binding=projected_binding,
            )
            connection.commit()
        return {
            **outcome,
            "outcomeId": receipt,
            # This is an operational acknowledgment deadline, never an
            # automatic deletion fence.  An unacknowledged paid result stays
            # recoverable until the owner-bound ACK or a reviewed admin action.
            "retentionOverdue": bool(
                int(row["outcome_expires_at_ms"] or 0) <= timestamp
            ),
        }

    def acknowledge_completed_outcome(
        self,
        key: str,
        *,
        binding: Mapping[str, Any],
        outcome_id: str,
    ) -> str:
        """Clear ciphertext only after an authenticated owner acknowledgment."""

        projected_binding = project_true_moa_outcome_binding(binding)
        clean_outcome_id = str(outcome_id or "").lower()
        if not _OUTCOME_DIGEST.fullmatch(clean_outcome_id):
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA outcome acknowledgment"
            )
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            state = str(row["outcome_state"] or "none")
            receipt = str(row["outcome_receipt"] or "")
            if state == "acked":
                if (
                    str(row["state"] or "") != "completed"
                    or not row["usage_json"]
                    or receipt != clean_outcome_id
                ):
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "conflicting true MoA outcome acknowledgment"
                    )
                usage = project_true_moa_usage(
                    json.loads(row["usage_json"])
                )
                aad = _true_moa_outcome_aad(
                    storage_key=storage_key,
                    fingerprint=str(row["fingerprint"] or ""),
                    usage=usage,
                    binding=projected_binding,
                )
                if (
                    str(row["outcome_binding_digest"] or "")
                    != hashlib.sha256(aad).hexdigest()
                ):
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "conflicting true MoA outcome acknowledgment"
                    )
                connection.commit()
                return "already_acknowledged"
            if (
                state != "sealed"
                or str(row["state"] or "") != "completed"
                or not row["usage_json"]
            ):
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            usage = project_true_moa_usage(json.loads(row["usage_json"]))
            _outcome, verified_receipt = self._decrypt_outcome_row(
                row,
                storage_key=storage_key,
                usage=usage,
                binding=projected_binding,
            )
            if verified_receipt != clean_outcome_id:
                connection.rollback()
                raise TrueMoAOutcomeBindingError(
                    "conflicting true MoA outcome acknowledgment"
                )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET outcome_state = 'acked',
                    outcome_key_id = '',
                    outcome_nonce = X'',
                    outcome_ciphertext = X'',
                    outcome_expires_at_ms = 0,
                    outcome_acked_at_ms = ?,
                    updated_at_ms = ?
                WHERE storage_key = ? AND outcome_state = 'sealed'
                """,
                (timestamp, timestamp, storage_key),
            )
            connection.commit()
        self._harden_files()
        return "acknowledged"

    def set_state(self, key: str, *, state: str) -> None:
        if state not in _VALID_STATES:
            raise ValueError("invalid true MoA durable state")
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return
            merged_state = _merge_status(
                str(row["state"] or ""),
                state,
                ranks=_DURABLE_STATE_RANK,
                terminals=_DURABLE_TERMINAL_STATES,
                stopped_wins=True,
                interrupted_is_provisional=True,
            )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = ?, updated_at_ms = ?
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (merged_state, timestamp, storage_key),
            )
            connection.commit()
        self._harden_files()

    def mark_stopped(self, key: str) -> bool:
        """Atomically install a stop fence unless a non-stoppable state won."""

        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT kind, state
                FROM true_moa_idempotency
                WHERE storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                row_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM true_moa_idempotency"
                    ).fetchone()[0]
                )
                if row_count >= TRUE_MOA_DURABLE_MAX_ROWS:
                    connection.rollback()
                    raise RuntimeError(
                        "true MoA durable ledger capacity exhausted"
                    )
                connection.execute(
                    """
                    INSERT INTO true_moa_idempotency (
                        storage_key, fingerprint, kind, state,
                        usage_json, created_at_ms, updated_at_ms
                    ) VALUES (?, '', 'execution', 'stopped', '', ?, ?)
                    """,
                    (storage_key, timestamp, timestamp),
                )
                accepted = True
            elif (
                row["kind"] == "execution"
                and row["state"]
                in {"claimed", "running", "interrupted", "stopped"}
            ):
                connection.execute(
                    """
                    UPDATE true_moa_idempotency
                    SET state = 'stopped', updated_at_ms = ?
                    WHERE storage_key = ?
                    """,
                    (timestamp, storage_key),
                )
                accepted = True
            else:
                accepted = False
            connection.commit()
        self._harden_files()
        return accepted


__all__ = [
    "TRUE_MOA_COMPLETED_OUTCOME_SCHEMA",
    "TRUE_MOA_DURABLE_MAX_CALLS",
    "TRUE_MOA_DURABLE_MAX_FINAL_CALLS",
    "TRUE_MOA_DURABLE_MAX_ROWS",
    "TRUE_MOA_DURABLE_USAGE_MAX_BYTES",
    "TRUE_MOA_OUTCOME_BINDING_SCHEMA",
    "TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS",
    "TrueMoAOutcomeBindingError",
    "TrueMoAOutcomeError",
    "TrueMoAOutcomeUnavailableError",
    "TrueMoADurableStore",
    "default_true_moa_durable_path",
    "project_true_moa_completed_outcome",
    "project_true_moa_outcome_binding",
    "project_true_moa_usage",
]
