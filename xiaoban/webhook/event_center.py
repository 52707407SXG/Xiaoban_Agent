"""My Stand Event Center webhook contract."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any

from xiaoban.connectors import NormalizedInboundEvent
from xiaoban.identity import ChannelIdentity


@dataclass(frozen=True)
class EventCenterEvent:
    site_id: str
    event_id: str
    event_type: str
    user_id: str
    occurred_at: str
    title: str
    body: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_id(self) -> str:
        return f"event-center:{self.site_id}:{self.event_id}"


def verify_event_center_signature(
    body: bytes,
    *,
    timestamp: str,
    signature: str,
    secret: str,
) -> bool:
    if not secret or not timestamp or not signature:
        return False
    signed = timestamp.encode("utf-8") + b"." + body
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def normalize_event_center_payload(payload: dict[str, Any]) -> EventCenterEvent:
    return EventCenterEvent(
        site_id=str(payload.get("site_id") or payload.get("siteId") or ""),
        event_id=str(payload.get("event_id") or payload.get("eventId") or ""),
        event_type=str(payload.get("event_type") or payload.get("eventType") or ""),
        user_id=str(payload.get("user_id") or payload.get("userId") or ""),
        occurred_at=str(payload.get("occurred_at") or payload.get("occurredAt") or ""),
        title=str(payload.get("title") or ""),
        body=str(payload.get("body") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def build_event_delivery_draft(event: EventCenterEvent) -> dict[str, Any]:
    return {
        "message_id": event.message_id,
        "site_id": event.site_id,
        "user_id": event.user_id,
        "event_type": event.event_type,
        "text": _format_event_text(event),
        "metadata": event.metadata,
    }


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def event_center_to_inbound_event(event: EventCenterEvent) -> NormalizedInboundEvent:
    channel_identity = ChannelIdentity(
        channel="mystand-event-center",
        channel_account_id=event.site_id,
        external_chat_id=event.user_id,
        external_user_id=event.user_id,
    )
    return NormalizedInboundEvent(
        connector="mystand-event-center",
        channel_identity=channel_identity,
        message_id=event.message_id,
        conversation_id=f"mystand:{event.site_id}:user:{event.user_id}:events",
        client_ts=event.occurred_at,
        kind="event",
        text=_format_event_text(event),
        metadata={"eventType": event.event_type, **event.metadata},
    )


def _format_event_text(event: EventCenterEvent) -> str:
    if event.body:
        return f"{event.title}\n{event.body}"
    return event.title
