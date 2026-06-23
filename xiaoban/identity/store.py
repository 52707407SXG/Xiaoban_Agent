"""Identity directory interfaces and in-memory implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ChannelIdentity, MemoryScope, MyStandUserIdentity


@dataclass
class InMemoryIdentityDirectory:
    users: dict[tuple[str, str], MyStandUserIdentity] = field(default_factory=dict)
    channel_map: dict[str, tuple[str, str]] = field(default_factory=dict)

    def add_user(self, user: MyStandUserIdentity) -> None:
        self.users[(user.site_id, user.user_id)] = user

    def bind_channel(self, site_id: str, user_id: str, channel_identity: ChannelIdentity) -> None:
        if (site_id, user_id) not in self.users:
            raise KeyError(f"unknown My Stand user: {site_id}/{user_id}")
        self.channel_map[channel_identity.stable_key] = (site_id, user_id)

    def resolve_channel(self, channel_identity: ChannelIdentity) -> MyStandUserIdentity | None:
        mapped = self.channel_map.get(channel_identity.stable_key)
        if mapped is None:
            return None
        return self.users.get(mapped)

    def memory_scope_for(self, user: MyStandUserIdentity) -> MemoryScope:
        return MemoryScope(site_id=user.site_id, user_id=user.user_id)

    def site_memory_scope(self, site_id: str) -> MemoryScope:
        return MemoryScope(site_id=site_id)

