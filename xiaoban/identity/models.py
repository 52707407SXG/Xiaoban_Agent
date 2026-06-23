"""My Stand identity models.

Connectors should normalize external platform identities into
``ChannelIdentity``. The Agent runtime then resolves them to trusted
``MyStandUserIdentity`` records before memory, context, or tools are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MyStandUserIdentity:
    site_id: str
    user_id: str
    display_name: str
    role: str
    company_id: str | None = None
    team_id: str | None = None
    is_owner: bool = False
    capabilities: frozenset[str] = field(default_factory=frozenset)
    scopes: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ChannelIdentity:
    channel: str
    external_user_id: str
    external_chat_id: str
    channel_account_id: str | None = None
    is_group: bool = False

    @property
    def stable_key(self) -> str:
        account = self.channel_account_id or "default"
        chat_kind = "group" if self.is_group else "direct"
        return f"{self.channel}:{account}:{chat_kind}:{self.external_chat_id}:{self.external_user_id}"


@dataclass(frozen=True)
class MemoryScope:
    site_id: str
    user_id: str | None = None
    channel: str | None = None

    @property
    def namespace(self) -> str:
        if self.user_id:
            return f"mystand:{self.site_id}:user:{self.user_id}:memory"
        return f"mystand:{self.site_id}:site:memory"

