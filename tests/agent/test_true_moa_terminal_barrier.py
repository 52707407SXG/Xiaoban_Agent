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
    sentinel = SimpleNamespace(
        choices=[SimpleNamespace(message="PRIVATE_LATE")]
    )

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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("-1", helpers._STRICT_PAID_SHUTDOWN_GRACE_MIN_SECONDS),
        ("999", helpers._STRICT_PAID_SHUTDOWN_GRACE_MAX_SECONDS),
        ("nan", helpers._STRICT_PAID_SHUTDOWN_GRACE_DEFAULT_SECONDS),
    ],
)
def test_strict_paid_shutdown_grace_has_hard_bounds(
    monkeypatch,
    configured,
    expected,
):
    monkeypatch.setenv(
        helpers._STRICT_PAID_SHUTDOWN_GRACE_ENV,
        configured,
    )

    assert helpers._strict_paid_shutdown_grace_seconds() == expected


def test_strict_paid_shutdown_grace_cancels_and_discards_late_response(
    monkeypatch,
    caplog,
):
    agent = _strict_agent(stale_timeout=0.01)
    release_worker = threading.Event()
    worker_finished = threading.Event()
    fake_client = MagicMock()
    sentinel = SimpleNamespace(
        choices=[SimpleNamespace(message="PRIVATE_LATE")],
        usage=SimpleNamespace(
            input_tokens=7,
            output_tokens=3,
            total_tokens=10,
            cached_input_tokens=0,
        ),
    )
    late_usage = MagicMock()
    agent._strict_late_provider_usage_callback = late_usage

    def create(**_kwargs):
        release_worker.wait()
        return sentinel

    def close(_client, *, reason):
        assert reason == "request_complete"
        worker_finished.set()

    fake_client.chat.completions.create.side_effect = create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client.side_effect = close
    monkeypatch.setattr(
        helpers,
        "_strict_paid_shutdown_grace_seconds",
        lambda: 0.05,
    )
    caplog.set_level("DEBUG", logger=helpers.__name__)

    started = time.monotonic()
    try:
        with pytest.raises(helpers.StrictPaidWorkerShutdownTimeout) as exc_info:
            helpers.interruptible_api_call(
                agent,
                {"model": "deepseek-v4-pro", "messages": []},
            )
    finally:
        release_worker.set()

    assert exc_info.value.reason == "stale_call_kill"
    assert exc_info.value.grace_seconds == 0.05
    assert time.monotonic() - started < 1.5
    assert worker_finished.wait(1)
    late_usage.assert_called_once_with(sentinel)
    assert (
        "Discarding provider response produced after request cancellation."
        in caplog.messages
    )
    assert not any(
        "waiting for strict paid worker shutdown" in str(call.args[0])
        for call in agent._touch_activity.call_args_list
    )


def test_strict_paid_shutdown_grace_recovers_usage_published_before_close_hang(
    monkeypatch,
    caplog,
):
    agent = _strict_agent(stale_timeout=0.01)
    release_close = threading.Event()
    worker_finished = threading.Event()
    fake_client = MagicMock()
    sentinel = SimpleNamespace(
        choices=[SimpleNamespace(message="PRIVATE_ALREADY_PUBLISHED")],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=5,
            total_tokens=16,
            cached_input_tokens=0,
        ),
    )
    late_usage = MagicMock()
    agent._strict_late_provider_usage_callback = late_usage

    def close(_client, *, reason):
        assert reason == "request_complete"
        assert release_close.wait(2)
        worker_finished.set()

    fake_client.chat.completions.create.return_value = sentinel
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client.side_effect = close
    monkeypatch.setattr(
        helpers,
        "_strict_paid_shutdown_grace_seconds",
        lambda: 0.05,
    )
    caplog.set_level("DEBUG", logger=helpers.__name__)

    try:
        with pytest.raises(helpers.StrictPaidWorkerShutdownTimeout) as exc_info:
            helpers.interruptible_api_call(
                agent,
                {"model": "deepseek-v4-pro", "messages": []},
            )
    finally:
        release_close.set()

    assert worker_finished.wait(1)
    assert not hasattr(exc_info.value, "usage")
    assert exc_info.value.late_accounting_pending is False
    late_usage.assert_called_once_with(sentinel)
    assert (
        "Discarding provider response published before request cancellation."
        in caplog.messages
    )


def test_strict_paid_cancel_keeps_published_usage_out_of_timeout_callback(
    monkeypatch,
):
    agent = _strict_agent()
    release_close = threading.Event()
    worker_finished = threading.Event()
    fake_client = MagicMock()
    sentinel = SimpleNamespace(
        choices=[SimpleNamespace(message="PRIVATE_CANCELLED_RESPONSE")],
        usage=SimpleNamespace(
            input_tokens=13,
            output_tokens=8,
            total_tokens=21,
            cached_input_tokens=0,
        ),
    )
    late_usage = MagicMock()
    agent._strict_late_provider_usage_callback = late_usage

    def create(**_kwargs):
        agent._interrupt_requested = True
        return sentinel

    def close(_client, *, reason):
        assert reason == "request_complete"
        assert release_close.wait(2)
        worker_finished.set()

    fake_client.chat.completions.create.side_effect = create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client.side_effect = close
    monkeypatch.setattr(
        helpers,
        "_strict_paid_shutdown_grace_seconds",
        lambda: 0.05,
    )

    try:
        with pytest.raises(helpers.StrictPaidWorkerShutdownTimeout) as exc_info:
            helpers.interruptible_api_call(
                agent,
                {"model": "deepseek-v4-pro", "messages": []},
            )
    finally:
        release_close.set()

    assert worker_finished.wait(1)
    assert exc_info.value.reason == "interrupt_abort"
    assert exc_info.value.usage is sentinel.usage
    assert exc_info.value.late_accounting_pending is False
    late_usage.assert_not_called()
