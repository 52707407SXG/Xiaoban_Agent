"""Channel identity policy for Xiaoban-Agent."""

from __future__ import annotations

from .models import ChannelIdentity, MyStandUserIdentity

WEBSITE_CHANNELS = {"web", "website", "cli"}


def can_use_channel(user: MyStandUserIdentity, channel_identity: ChannelIdentity) -> bool:
    """Return whether *user* may use Xiaoban through *channel_identity*.

    Website/CLI access is governed by My Stand session and role checks.
    External channels default to owner-only until My Stand exposes a richer
    binding policy.
    """

    if channel_identity.channel in WEBSITE_CHANNELS:
        return True
    return user.is_owner

