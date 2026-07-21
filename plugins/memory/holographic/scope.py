"""Fail-closed My Stand account scoping for holographic memory.

My Stand memory is deliberately stored outside the historical global
``memory_store.db``.  The on-disk filename is an opaque HMAC so neither the
website username nor site identifier is exposed in paths or SQLite metadata.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path


_SITE_ID_RE = re.compile(r"[A-Za-z0-9._:@-]{1,120}\Z")
_USER_ID_RE = re.compile(r"[A-Za-z0-9._:@-]{1,200}\Z")
MEMORY_MODES = frozenset({"disabled", "user"})


class MemoryScopeError(ValueError):
    """Raised when trusted My Stand memory scope headers are invalid."""


def validate_memory_scope(site_id: str, user_id: str, mode: str) -> tuple[str, str, str]:
    """Validate and normalize the server-supplied My Stand memory scope."""
    normalized_site = str(site_id or "").strip()
    normalized_user = str(user_id or "").strip()
    normalized_mode = str(mode or "").strip().lower()
    if not _SITE_ID_RE.fullmatch(normalized_site):
        raise MemoryScopeError("invalid My Stand site identity")
    if not _USER_ID_RE.fullmatch(normalized_user):
        raise MemoryScopeError("invalid My Stand user identity")
    if normalized_mode not in MEMORY_MODES:
        raise MemoryScopeError("invalid My Stand memory mode")
    return normalized_site, normalized_user, normalized_mode


def memory_scope_id(*, secret: str, site_id: str, user_id: str) -> str:
    """Return the opaque account identifier used as the SQLite basename."""
    normalized_site, normalized_user, _ = validate_memory_scope(site_id, user_id, "user")
    secret_bytes = str(secret or "").encode("utf-8")
    if not secret_bytes:
        raise MemoryScopeError("memory scope secret is unavailable")
    payload = f"mystand-memory-v1\0{normalized_site}\0{normalized_user}".encode("utf-8")
    return hmac.new(secret_bytes, payload, hashlib.sha256).hexdigest()


def scoped_memory_path(
    *,
    secret: str,
    site_id: str,
    user_id: str,
    xiaoban_home: "str | Path | None" = None,
) -> Path:
    """Resolve a per-account database path without reading the legacy store."""
    if xiaoban_home is None:
        from xiaoban_constants import get_xiaoban_home

        root = get_xiaoban_home()
    else:
        root = Path(xiaoban_home)
    opaque_id = memory_scope_id(secret=secret, site_id=site_id, user_id=user_id)
    return Path(root) / "memory" / "users" / f"{opaque_id}.db"


def open_scoped_memory_store(
    *,
    secret: str,
    site_id: str,
    user_id: str,
    xiaoban_home: "str | Path | None" = None,
):
    """Open the account's private store with restrictive filesystem modes."""
    from .store import MemoryStore

    return MemoryStore(
        db_path=scoped_memory_path(
            secret=secret,
            site_id=site_id,
            user_id=user_id,
            xiaoban_home=xiaoban_home,
        ),
        secure_permissions=True,
    )
