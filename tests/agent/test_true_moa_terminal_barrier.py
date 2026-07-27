"""No-network tests for the fixed true-MoA final-call exit barrier."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent import chat_completion_helpers as helpers


def _strict_agent(*, stale_timeout: float = 5.0):
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._strict_no_automatic_paid_retry = True
    agent._compute_non_stream_stale_timeout.return_value = stale_timeout
    agent.verbose_logging = False
    return agent


def _blocking_client(agent, *, trigger_interrupt: bool):
    released = threading.Event()
    worker_exited = threading.Event()
    fake_client = MagicMock()

    def create(**_kwargs):
        if trigger_interrupt:
            agent._interrupt_requested = True
        assert released.wait(2)
        worker_exited.set()
        raise httpx.RemoteProtocolError("socket closed by terminal fence")

    def abort(_client, *, reason):
        assert reason in {"interrupt_abort", "stale_call_kill"}
        threading.Timer(0.15, released.set).start()

    fake_client.chat.completions.create.side_effect = create
    agent._create_request_openai_client.return_value = fake_client
    agent._abort_request_openai_client.side_effect = abort
    agent._close_request_openai_client = MagicMock()
    return worker_exited


def test_strict_interrupt_returns_only_after_provider_worker_exits():
    agent = _strict_agent()
    worker_exited = _blocking_client(agent, trigger_interrupt=True)

    started = time.monotonic()
    with pytest.raises(InterruptedError):
        helpers.interruptible_api_call(
            agent,
            {"model": "deepseek-v4-pro", "messages": []},
        )

    assert worker_exited.is_set()
    assert time.monotonic() - started >= 0.14


def test_strict_timeout_returns_only_after_provider_worker_exits():
    agent = _strict_agent(stale_timeout=0.05)
    worker_exited = _blocking_client(agent, trigger_interrupt=False)

    started = time.monotonic()
    with pytest.raises((TimeoutError, httpx.RemoteProtocolError)):
        helpers.interruptible_api_call(
            agent,
            {"model": "deepseek-v4-pro", "messages": []},
        )

    assert worker_exited.is_set()
    assert time.monotonic() - started >= 0.14


def test_strict_response_cannot_escape_if_interrupt_wins_at_worker_exit(
    monkeypatch,
):
    """Cover the zero-iteration race where the worker exits before polling."""

    agent = _strict_agent()
    fake_client = MagicMock()
    sentinel = SimpleNamespace(choices=[SimpleNamespace(message="PRIVATE_LATE")])

    def create(**_kwargs):
        agent._interrupt_requested = True
        return sentinel

    fake_client.chat.completions.create.side_effect = create
    agent._create_request_openai_client.return_value = fake_client

    class _SynchronousThread:
        def __init__(self, *, target, daemon):
            assert daemon is True
            self._target = target
            self._alive = False

        def start(self):
            self._alive = True
            try:
                self._target()
            finally:
                self._alive = False

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            assert timeout is None or timeout >= 0

    monkeypatch.setattr(helpers.threading, "Thread", _SynchronousThread)

    with pytest.raises(InterruptedError):
        helpers.interruptible_api_call(
            agent,
            {"model": "deepseek-v4-pro", "messages": []},
        )

    fake_client.chat.completions.create.assert_called_once()
