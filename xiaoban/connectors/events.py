"""Normalized inbound event contracts for Xiaoban connectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from xiaoban.identity import ChannelIdentity

MessageKind = Literal["text", "image", "file", "event", "unknown"]


@dataclass(frozen=True)
class DeliveryKey:
    channel: str
    channel_account_id: str
    external_chat_id: str
    external_message_id: str

    @property
    def stable_key(self) -> str:
        return ":".join(
            [
                self.channel,
                self.channel_account_id or "default",
                self.external_chat_id,
                self.external_message_id,
            ]
        )


@dataclass(frozen=True)
class NormalizedInboundEvent:
    connector: str
    channel_identity: ChannelIdentity
    message_id: str
    conversation_id: str
    client_ts: str
    kind: MessageKind
    text: str = ""
    attachments: tuple[dict[str, Any], ...] = ()
    raw_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delivery_key(self) -> DeliveryKey:
        return DeliveryKey(
            channel=self.channel_identity.channel,
            channel_account_id=self.channel_identity.channel_account_id or "default",
            external_chat_id=self.channel_identity.external_chat_id,
            external_message_id=self.message_id,
        )


@dataclass(frozen=True)
class DurableReceiveResult:
    accepted: bool
    delivery_key: DeliveryKey
    status: Literal["accepted", "duplicate"]

