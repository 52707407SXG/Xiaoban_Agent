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
    MYSTAND_COMPLETION_PROTOCOL,
    MYSTAND_COMPLETION_VERIFICATION_SCHEMA,
    MYSTAND_FACT_VERIFICATION_SCHEMA,
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
_OUTCOME_DYNAMIC_BINDING_FIELDS = _OUTCOME_BINDING_FIELDS | {
    "completionProtocol",
    "invocationFingerprint",
}
_DYNAMIC_COMPLETION_PROTOCOL = MYSTAND_COMPLETION_PROTOCOL
_DYNAMIC_VERIFICATION_SCHEMA = MYSTAND_COMPLETION_VERIFICATION_SCHEMA


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
    if field_names not in {
        frozenset(_OUTCOME_BINDING_FIELDS),
        frozenset(_OUTCOME_DYNAMIC_BINDING_FIELDS),
    }:
        raise ValueError("invalid true MoA outcome binding")
    dynamic_binding = field_names == frozenset(_OUTCOME_DYNAMIC_BINDING_FIELDS)
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
    if dynamic_binding:
        projected["completionProtocol"] = str(
            value.get("completionProtocol") or ""
        )
        projected["invocationFingerprint"] = str(
            value.get("invocationFingerprint") or ""
        ).lower()
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
        or (
            dynamic_binding
            and (
                projected["completionProtocol"]
                != _DYNAMIC_COMPLETION_PROTOCOL
                or projected["attempt"] < 1
                or not _OUTCOME_DIGEST.fullmatch(
                    projected["invocationFingerprint"]
                )
            )
        )
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
        isinstance(projected, dict)
        and projected.get("schema") == _DYNAMIC_VERIFICATION_SCHEMA
    ):
        expected_fields = {
            "schema",
            "completion_kind",
            "binding_verified",
            "semantic_verified",
            "delivery_id",
            "request_id",
            "attempt",
            "message_id",
            "request_fingerprint",
            "invocation_fingerprint",
            "datascope_fingerprint",
            "action_count",
            "evidence_count",
            "output_digest",
            "decision",
            "verified_at",
            "index_count",
            "index_has_more",
            "index_receipt_digest",
            "index_resource_refs_digest",
            "record_refs",
            "record_refs_digest",
            "evidence_digest",
        }
        record_refs = projected.get("record_refs")
        if (
            set(projected) != expected_fields
            or projected.get("completion_kind") != "evidence-bound"
            or projected.get("binding_verified") is not True
            or projected.get("semantic_verified") is not False
            or isinstance(projected.get("action_count"), bool)
            or projected.get("action_count") != 2
            or isinstance(projected.get("evidence_count"), bool)
            or projected.get("evidence_count") != 1
            or isinstance(projected.get("attempt"), bool)
            or not isinstance(projected.get("attempt"), int)
            or projected.get("decision") != "projected_evidence"
            or projected.get("index_has_more") is not False
            or isinstance(projected.get("index_count"), bool)
            or not isinstance(projected.get("index_count"), int)
            or projected["index_count"] < 1
            or not isinstance(record_refs, list)
            or not record_refs
            or len(record_refs) > projected["index_count"]
            or record_refs != sorted(set(record_refs))
            or any(
                not isinstance(ref, str)
                or not ref
                or len(ref) > 240
                or re.search(r"[\r\n\x00]", ref)
                for ref in record_refs
            )
            or any(
                not _OUTCOME_DIGEST.fullmatch(
                    str(projected.get(name) or "")
                )
                for name in (
                    "request_fingerprint",
                    "invocation_fingerprint",
                    "output_digest",
                    "index_receipt_digest",
                    "index_resource_refs_digest",
                    "record_refs_digest",
                    "evidence_digest",
                )
            )
            or projected.get("output_digest") != output_digest
            or projected.get("record_refs_digest")
            != hashlib.sha256(
                _canonical_json_bytes(record_refs)
            ).hexdigest()
            or not isinstance(projected.get("verified_at"), str)
            or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
                r"(?:\.\d+)?Z",
                projected["verified_at"],
            )
            or not isinstance(binding, Mapping)
            or binding.get("completionProtocol")
            != _DYNAMIC_COMPLETION_PROTOCOL
            or projected.get("delivery_id") != binding.get("deliveryId")
            or projected.get("request_id") != binding.get("deliveryId")
            or projected.get("message_id") != binding.get("messageId")
            or projected.get("attempt") != binding.get("attempt")
            or projected.get("request_fingerprint")
            != binding.get("requestFingerprint")
            or projected.get("invocation_fingerprint")
            != binding.get("invocationFingerprint")
            or projected.get("datascope_fingerprint")
            != binding.get("datascopeFingerprint")
        ):
            raise TrueMoAOutcomeBindingError(
                "true MoA dynamic verification binding mismatch"
            )
        return projected
    if (
        not isinstance(projected, dict)
        or projected.get("schema")
        != MYSTAND_FACT_VERIFICATION_SCHEMA
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
    if "completionProtocol" in value:
        expected.add("completionProtocol")
    if set(value) != expected:
        raise ValueError("invalid true MoA completed outcome")
    final_response = value.get("finalResponse")
    if not isinstance(final_response, str):
        raise ValueError("invalid true MoA completed outcome")
    final_bytes = final_response.encode("utf-8")
    output_digest = str(value.get("outputDigest") or "").lower()
    fact_guard_required = value.get("factGuardRequired")
    completion_protocol = str(value.get("completionProtocol") or "")
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
        or (
            "completionProtocol" in value
            and completion_protocol != _DYNAMIC_COMPLETION_PROTOCOL
        )
    ):
        raise ValueError("invalid true MoA completed outcome")
    verification = _project_trusted_verification(
        value.get("trustedVerification"),
        output_digest=output_digest,
        binding=projected_binding,
    )
    if fact_guard_required and verification is None:
        raise ValueError("true MoA fact outcome lacks trusted verification")
    if completion_protocol:
        if (
            fact_guard_required
            or verification is None
            or verification.get("schema") != _DYNAMIC_VERIFICATION_SCHEMA
            or projected_binding is None
            or projected_binding.get("completionProtocol")
            != _DYNAMIC_COMPLETION_PROTOCOL
        ):
            raise ValueError("invalid true MoA dynamic completion outcome")
    elif (
        verification is not None
        and verification.get("schema") == _DYNAMIC_VERIFICATION_SCHEMA
    ):
        raise ValueError("dynamic verification lacks completion protocol")
    projected: dict[str, Any] = {
        "schema": TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
        "completed": True,
        "finalResponse": final_response,
        "outputDigest": output_digest,
        "factGuardRequired": fact_guard_required,
    }
    if completion_protocol:
        projected["completionProtocol"] = completion_protocol
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
