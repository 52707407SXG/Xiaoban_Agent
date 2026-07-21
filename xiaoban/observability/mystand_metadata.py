"""Strict metadata-only events for My Stand Xiaoban requests.

The payload schema is intentionally closed.  Prompt text, response text,
reasoning, tool arguments/results, attachment details, raw account IDs and
exception strings have no accepted field and therefore cannot be logged by
this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("xiaoban.mystand.metadata")

_EVENTS = frozenset({
    "request_started",
    "request_completed",
    "request_failed",
    "tool_started",
    "tool_completed",
})
_STATUSES = frozenset({"accepted", "completed", "failed", "running"})
_DELIVERY_STATUSES = frozenset({
    "accepted", "generating", "delivering", "delivered", "failed",
    "stopped", "interrupted", "settlement_blocked",
})
_SAFE_LABEL_RE = re.compile(r"[A-Za-z0-9._:/@+-]{1,160}\Z")
_HEX_64_RE = re.compile(r"[a-f0-9]{64}\Z")
_HEX_32_RE = re.compile(r"[a-f0-9]{32}\Z")
_INT_FIELDS = frozenset({
    "attempt", "retry_count", "duration_ms", "tool_count", "tool_duration_ms",
    "memory_hit_count", "input_tokens", "output_tokens", "total_tokens",
})
_BOOL_FIELDS = frozenset({"memory_enabled", "success"})
_LABEL_FIELDS = frozenset({"provider", "model", "tool_name", "error_code"})
_ALLOWED_FIELDS = frozenset({
    "event", "timestamp_ms", "trace_id", "account_scope", "status",
    "delivery_status", *_INT_FIELDS, *_BOOL_FIELDS, *_LABEL_FIELDS,
})


class MetadataValidationError(ValueError):
    """Raised when a caller attempts to emit data outside the safe schema."""


def _account_scope(*, secret: str, site_id: str, user_id: str) -> str:
    secret_bytes = str(secret or "").encode("utf-8")
    if not secret_bytes:
        raise MetadataValidationError("metadata scope secret is unavailable")
    payload = f"mystand-trace-v1\0{site_id}\0{user_id}".encode("utf-8")
    return hmac.new(secret_bytes, payload, hashlib.sha256).hexdigest()


def _safe_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise MetadataValidationError(f"{field_name} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise MetadataValidationError(f"{field_name} must be an integer") from exc
    if normalized < 0:
        raise MetadataValidationError(f"{field_name} must be non-negative")
    return normalized


@dataclass
class MystandMetadataTrace:
    """One request's closed-schema metadata trace."""

    secret: str
    site_id: str
    user_id: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.account_scope = _account_scope(
            secret=self.secret,
            site_id=self.site_id,
            user_id=self.user_id,
        )
        if not _HEX_32_RE.fullmatch(self.trace_id):
            raise MetadataValidationError("trace_id must be a random hex identifier")

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Validate, log and return a metadata event.

        Returning the event keeps tests deterministic without requiring a log
        capture fixture.  Unknown fields fail closed.
        """
        unknown = set(fields) - (_ALLOWED_FIELDS - {"event", "timestamp_ms", "trace_id", "account_scope"})
        if unknown:
            raise MetadataValidationError(f"unsupported metadata fields: {','.join(sorted(unknown))}")
        normalized_event = str(event or "").strip()
        if normalized_event not in _EVENTS:
            raise MetadataValidationError("unsupported metadata event")
        payload: dict[str, Any] = {
            "event": normalized_event,
            "timestamp_ms": int(time.time() * 1000),
            "trace_id": self.trace_id,
            "account_scope": self.account_scope,
        }
        for name, value in fields.items():
            if name in _INT_FIELDS:
                payload[name] = _safe_non_negative_int(value, name)
            elif name in _BOOL_FIELDS:
                if not isinstance(value, bool):
                    raise MetadataValidationError(f"{name} must be boolean")
                payload[name] = value
            elif name == "status":
                if value not in _STATUSES:
                    raise MetadataValidationError("unsupported metadata status")
                payload[name] = value
            elif name == "delivery_status":
                if value not in _DELIVERY_STATUSES:
                    raise MetadataValidationError("unsupported delivery status")
                payload[name] = value
            elif name in _LABEL_FIELDS:
                label = str(value or "").strip()
                if not _SAFE_LABEL_RE.fullmatch(label):
                    raise MetadataValidationError(f"unsafe {name}")
                payload[name] = label
        if not _HEX_64_RE.fullmatch(payload["account_scope"]):
            raise MetadataValidationError("invalid account scope")
        logger.info("mystand_metadata %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return payload

    def elapsed_ms(self) -> int:
        return max(0, int((time.monotonic() - self.started_at) * 1000))

    def safe_emit(self, event: str, **fields: Any) -> None:
        """Emit without allowing optional telemetry to break a request."""
        try:
            self.emit(event, **fields)
        except Exception:
            logger.warning("mystand metadata event rejected", exc_info=False)
