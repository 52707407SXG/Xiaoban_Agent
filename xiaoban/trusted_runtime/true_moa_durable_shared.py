"""Validation and cryptographic projection contracts for true-MoA durability."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from xiaoban.trusted_runtime.protocol_contract import (
    MYSTAND_COMPLETED_OUTCOME_SCHEMA,
    MYSTAND_OUTCOME_AAD_SCHEMA,
    MYSTAND_OUTCOME_BINDING_SCHEMA,
    MYSTAND_TRUE_MOA_PRESET_ID,
    MYSTAND_TRUE_MOA_PRESET_REVISION,
    MYSTAND_TRUE_MOA_SLOTS,
)

TRUE_MOA_DURABLE_USAGE_MAX_BYTES = 64 * 1024
TRUE_MOA_DURABLE_MAX_ROWS = 100_000
TRUE_MOA_DURABLE_MAX_CALLS = 10
TRUE_MOA_DURABLE_MAX_FINAL_CALLS = 8
TRUE_MOA_COMPLETED_OUTCOME_SCHEMA = MYSTAND_COMPLETED_OUTCOME_SCHEMA
TRUE_MOA_OUTCOME_BINDING_SCHEMA = MYSTAND_OUTCOME_BINDING_SCHEMA
TRUE_MOA_OUTCOME_MAX_TEXT_BYTES = 64 * 1024
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
_RECEIPT_STATUS_RANK = {
    "not_started": 0,
    "reserved": 1,
    "running": 2,
}
_RECEIPT_TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "not_dispatched",
}
_DURABLE_STATE_RANK = {"claimed": 0, "running": 1}
_DURABLE_TERMINAL_STATES = {
    "completed",
    "failed",
    "stopped",
    "interrupted",
}
_FIXED_SLOTS = {
    str(slot["slotId"]): (
        str(slot["provider"]),
        str(slot["model"]),
        str(slot["role"]),
    )
    for slot in MYSTAND_TRUE_MOA_SLOTS
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


def _durable_max_rows() -> int:
    """Read the compatibility-facade limit so test/runtime overrides still apply."""

    from xiaoban.trusted_runtime import true_moa_durable

    return int(
        getattr(
            true_moa_durable,
            "TRUE_MOA_DURABLE_MAX_ROWS",
            TRUE_MOA_DURABLE_MAX_ROWS,
        )
    )


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
    field_names = frozenset(value) if isinstance(value, Mapping) else frozenset()
    if field_names != frozenset(_OUTCOME_BINDING_FIELDS):
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
        or projected["presetId"] != MYSTAND_TRUE_MOA_PRESET_ID
        or projected["presetRevision"]
        != MYSTAND_TRUE_MOA_PRESET_REVISION
    ):
        raise ValueError("invalid true MoA outcome binding")
    return projected


def project_true_moa_completed_outcome(
    value: Mapping[str, Any],
    *,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid true MoA completed outcome")
    if set(value) != {
        "schema",
        "completed",
        "finalResponse",
        "outputDigest",
    }:
        raise ValueError("invalid true MoA completed outcome")
    final_response = value.get("finalResponse")
    if not isinstance(final_response, str):
        raise ValueError("invalid true MoA completed outcome")
    final_bytes = final_response.encode("utf-8")
    output_digest = str(value.get("outputDigest") or "").lower()
    if binding is not None:
        project_true_moa_outcome_binding(binding)
    if (
        value.get("schema") != TRUE_MOA_COMPLETED_OUTCOME_SCHEMA
        or value.get("completed") is not True
        or not final_response.strip()
        or len(final_bytes) > TRUE_MOA_OUTCOME_MAX_TEXT_BYTES
        or not _OUTCOME_DIGEST.fullmatch(output_digest)
        or output_digest != hashlib.sha256(final_bytes).hexdigest()
    ):
        raise ValueError("invalid true MoA completed outcome")
    projected: dict[str, Any] = {
        "schema": TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
        "completed": True,
        "finalResponse": final_response,
        "outputDigest": output_digest,
    }
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
    from xiaoban.trusted_runtime.true_moa_durable_usage import (
        project_true_moa_usage,
    )

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
        "schema": MYSTAND_OUTCOME_AAD_SCHEMA,
        "storageKey": storage_key,
        "durableFingerprint": _safe_text(fingerprint, required=True),
        "waveId": projected_usage["waveId"],
        "binding": projected_binding,
    }
    return _canonical_json_bytes(aad)
