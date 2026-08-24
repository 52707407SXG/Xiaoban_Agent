"""Shared, fail-closed My Stand owner identity configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any


MYSTAND_OWNER_USER_ID_ENV = "MYSTAND_XIAOBAN_OWNER_USER_ID"
_OWNER_USER_ID_RE = re.compile(r"^[A-Za-z0-9._:@-]{1,200}$")


def configured_mystand_owner_user_id(
    env: Mapping[str, Any] | None = None,
) -> str:
    """Return the validated configured owner id, or empty to deny access."""
    source = os.environ if env is None else env
    value = str(source.get(MYSTAND_OWNER_USER_ID_ENV, "") or "").strip()
    return value if _OWNER_USER_ID_RE.fullmatch(value) else ""


def is_configured_mystand_owner(
    user_id: Any,
    env: Mapping[str, Any] | None = None,
) -> bool:
    """Match one authenticated id against the configured owner exactly."""
    owner_user_id = configured_mystand_owner_user_id(env)
    return bool(owner_user_id and str(user_id or "").strip() == owner_user_id)


__all__ = [
    "MYSTAND_OWNER_USER_ID_ENV",
    "configured_mystand_owner_user_id",
    "is_configured_mystand_owner",
]
