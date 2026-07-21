from __future__ import annotations

import stat

import pytest

from plugins.memory.holographic.scope import (
    MemoryScopeError,
    open_scoped_memory_store,
    scoped_memory_path,
    validate_memory_scope,
)
from plugins.memory.holographic.store import MemoryStore


def _open(tmp_path, site: str, user: str):
    return open_scoped_memory_store(
        secret="scope-secret",
        site_id=site,
        user_id=user,
        xiaoban_home=tmp_path,
    )


def test_sites_and_users_get_opaque_independent_stores(tmp_path):
    paths = {
        scoped_memory_path(
            secret="scope-secret",
            site_id=site,
            user_id=user,
            xiaoban_home=tmp_path,
        )
        for site, user in (
            ("site-a", "alice"),
            ("site-a", "bob"),
            ("site-b", "alice"),
        )
    }

    assert len(paths) == 3
    for path in paths:
        assert path.parent == tmp_path / "memory" / "users"
        assert "alice" not in path.name
        assert "bob" not in path.name
        assert "site-a" not in path.name
        assert len(path.stem) == 64


def test_same_fact_is_allowed_per_account_and_cross_account_id_is_missing(tmp_path):
    first = _open(tmp_path, "site-a", "alice")
    second = _open(tmp_path, "site-a", "bob")
    try:
        first.add_fact("喜欢简洁的中文回复")
        private_id = first.add_fact("alice only")
        second.add_fact("喜欢简洁的中文回复")

        assert [row["content"] for row in first.list_facts()] == [
            "喜欢简洁的中文回复",
            "alice only",
        ]
        assert [row["content"] for row in second.list_facts()] == [
            "喜欢简洁的中文回复",
        ]
        assert second.update_fact(private_id, content="forged") is False
        assert second.remove_fact(private_id) is False
    finally:
        first.close()
        second.close()


def test_scoped_store_never_reads_legacy_global_database(tmp_path):
    legacy = MemoryStore(db_path=tmp_path / "memory_store.db")
    try:
        legacy.add_fact("legacy-global-secret")
    finally:
        legacy.close()

    scoped = _open(tmp_path, "site-a", "alice")
    try:
        assert scoped.list_facts() == []
    finally:
        scoped.close()


def test_scoped_database_and_directory_are_owner_only(tmp_path):
    store = _open(tmp_path, "site-a", "alice")
    path = store.db_path
    store.close()

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("site", "user", "mode"),
    [
        ("", "alice", "user"),
        ("site-a", "", "user"),
        ("site a", "alice", "user"),
        ("site-a", "alice", "global"),
    ],
)
def test_invalid_or_global_scope_fails_closed(site, user, mode):
    with pytest.raises(MemoryScopeError):
        validate_memory_scope(site, user, mode)
