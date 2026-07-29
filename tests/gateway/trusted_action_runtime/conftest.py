"""Shared no-network durability substrate for trusted gateway HTTP tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def trusted_gateway_durable_cache(monkeypatch, tmp_path):
    """Signed requests now fail closed unless their durable ledger is ready."""

    from gateway.platforms import api_server

    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "trusted-gateway.sqlite"),
        outcome_keys={"test-v1": b"\x41" * 32},
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    yield cache
    cache._durable.close()
