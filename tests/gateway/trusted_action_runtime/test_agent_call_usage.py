"""No-network gates for signed My Stand per-provider-call accounting."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import httpx
from openai import OpenAI

from agent import chat_completion_helpers as helpers
from agent import true_moa_conversation_policy as conversation_policy
from agent.true_moa_conversation_policy import execute_llm_request
from agent.chat_completion_helpers import StrictPaidWorkerShutdownTimeout
from gateway.platforms.agent_call_accounting import bind_paid_call_ledger
from gateway.platforms.api_server import _IdempotencyCache
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.agent_call_accounting import (
    initialize_normal_call_ledger,
)
from gateway.platforms.true_moa_runner_workflow import TrueMoARunWorkflow
from xiaoban.trusted_runtime.agent_call_usage import (
    AGENT_CALL_LIMIT,
    AgentCallUsageLedger,
    merge_agent_call_usage,
    project_agent_call_usage,
)
from xiaoban.trusted_runtime.paid_call_policy import (
    LEGACY_SIGNED_MYSTAND_AGENT_POLICY,
    LEGACY_SIGNED_MYSTAND_AGENT_POLICY_REVISION,
    PaidCallPolicyError,
    SIGNED_MYSTAND_AGENT_POLICY,
    SIGNED_MYSTAND_AGENT_POLICY_REGISTRY,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
    enforce_openai_chat_paid_call_dispatch_budget,
    enforce_openai_responses_paid_call_dispatch_budget,
    resolve_signed_mystand_agent_policy,
    serialize_openai_chat_request_body,
    serialize_openai_responses_request_body,
)
from xiaoban.trusted_runtime.true_moa_durable import TrueMoADurableStore
from xiaoban.trusted_runtime.true_moa import (
    FINAL_EXECUTOR_SLOT,
    TRUE_MOA_MODE,
    TRUE_MOA_PRESET_ID,
    TRUE_MOA_PRESET_REVISION,
    TrueMoACancelController,
    TrueMoASnapshot,
    TrueMoAUsageLedger,
)


def _usage(input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=0,
    )


def test_signed_chat_byte_preflight_matches_openai_sdk_wire_body():
    captured: list[bytes] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-byte-proof",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "中文字节核对"}],
        "max_tokens": 4096,
        "timeout": 17,
        "extra_headers": {"x-local-proof": "header-not-body"},
        "extra_query": {"local": "query-not-body"},
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    client = OpenAI(
        api_key="local-test-key",
        base_url="https://local.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(_capture),
        ),
    )
    client.chat.completions.create(**payload)

    encoded = serialize_openai_chat_request_body(payload)
    assert captured == [encoded]
    assert b'"timeout"' not in encoded
    assert b'"extra_body"' not in encoded
    assert b'"thinking":{"type":"enabled"}' in encoded


def test_signed_responses_preflight_accepts_codex_logical_output_cap():
    payload = {
        "model": SIGNED_MYSTAND_AGENT_POLICY.model,
        "instructions": "system",
        "input": [{"role": "user", "content": "中文字节核对"}],
        "store": False,
        "reasoning": {"effort": "max", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "extra_headers": {"session_id": "header-not-body"},
    }
    encoded = serialize_openai_responses_request_body(payload)
    assert b'"stream":true' in encoded
    assert b'header-not-body' not in encoded
    assert enforce_openai_responses_paid_call_dispatch_budget(
        SIGNED_MYSTAND_AGENT_POLICY,
        payload=payload,
        configured_output_max_tokens=(
            SIGNED_MYSTAND_AGENT_POLICY.output_max_tokens
        ),
        error_prefix="signed_test",
    ) == len(encoded)


def test_signed_responses_preflight_accepts_provider_native_output_window():
    payload = {
        "model": SIGNED_MYSTAND_AGENT_POLICY.model,
        "instructions": "system",
        "input": [{"role": "user", "content": "长任务不使用应用层 Token 硬上限"}],
        "store": False,
    }
    encoded = serialize_openai_responses_request_body(payload)
    assert enforce_openai_responses_paid_call_dispatch_budget(
        SIGNED_MYSTAND_AGENT_POLICY,
        payload=payload,
        configured_output_max_tokens=None,
        error_prefix="signed_test",
    ) == len(encoded)


def test_signed_responses_preflight_rejects_explicit_output_override():
    with pytest.raises(
        PaidCallPolicyError,
        match="signed_test_output_token_cap_exceeded",
    ):
        enforce_openai_responses_paid_call_dispatch_budget(
            SIGNED_MYSTAND_AGENT_POLICY,
            payload={
                "model": SIGNED_MYSTAND_AGENT_POLICY.model,
                "instructions": "system",
                "input": [{"role": "user", "content": "hello"}],
                "store": False,
            },
            configured_output_max_tokens=(
                SIGNED_MYSTAND_AGENT_POLICY.output_max_tokens + 1
            ),
            error_prefix="signed_test",
        )


@pytest.mark.parametrize(
    "extra_body",
    [
        {"model": "unexpected-model"},
        {"max_tokens": 4097},
        {"messages": [{"role": "user", "content": "replacement"}]},
        {"tools": [{"type": "function", "function": {"name": "late"}}]},
    ],
)
def test_signed_chat_rejects_extra_body_controlled_field_override(extra_body):
    with pytest.raises(
        PaidCallPolicyError,
        match="signed_test_protected_field_override",
    ):
        enforce_openai_chat_paid_call_dispatch_budget(
            SIGNED_MYSTAND_AGENT_POLICY,
            payload={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "original"}],
                "max_tokens": 4096,
                "extra_body": extra_body,
            },
            error_prefix="signed_test",
        )


def test_signed_normal_has_no_artificial_wire_byte_boundary():
    payload = {
        "model": SIGNED_MYSTAND_AGENT_POLICY.model,
        "instructions": "system",
        "input": [{"role": "user", "content": ""}],
        "store": False,
    }
    base_size = enforce_openai_responses_paid_call_dispatch_budget(
        SIGNED_MYSTAND_AGENT_POLICY,
        payload=payload,
        configured_output_max_tokens=(
            SIGNED_MYSTAND_AGENT_POLICY.output_max_tokens
        ),
        error_prefix="signed_test",
    )
    assert SIGNED_MYSTAND_AGENT_POLICY.input_max_bytes is None
    payload["input"][0]["content"] = "x" * 140_519
    encoded_size = enforce_openai_responses_paid_call_dispatch_budget(
        SIGNED_MYSTAND_AGENT_POLICY,
        payload=payload,
        configured_output_max_tokens=(
            SIGNED_MYSTAND_AGENT_POLICY.output_max_tokens
        ),
        error_prefix="signed_test",
    )
    assert encoded_size > base_size
    assert encoded_size > 131_072


def _agent(ledger: AgentCallUsageLedger):
    return SimpleNamespace(
        _paid_call_usage_ledger=ledger,
        _true_moa_usage_ledger=None,
        _true_moa_cancel_controller=None,
        _paid_call_policy_revision=(
            LEGACY_SIGNED_MYSTAND_AGENT_POLICY_REVISION
        ),
        _interrupt_requested=False,
        provider="deepseek",
        model="deepseek-v4-pro",
        api_mode="chat_completions",
        base_url="",
        api_key="",
        session_id="test-session",
        platform="test",
    )


def _header_value(headers, name: str) -> str:
    for key, value in dict(headers or {}).items():
        if str(key).lower() == name.lower():
            return str(value)
    return ""


def _normal_workflow(
    *,
    callback=None,
    revision: str = SIGNED_MYSTAND_AGENT_POLICY_REVISION,
):
    headers = {
        SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER: revision,
    }
    return SimpleNamespace(
        true_moa_ledger=None,
        agent_call_ledger=None,
        agent_call_policy_revision="",
        agent_call_terminal_settlement_confirmed=None,
        agent=None,
        request=SimpleNamespace(
            adapter=SimpleNamespace(_header_value=_header_value),
            request_headers=headers,
            mystand_request=True,
            durable_paid_call=True,
            true_moa_snapshot=None,
            paid_call_usage_callback=callback,
            agent_ref=[None, False, None],
            request_delivery_id="xbd_" + ("7" * 40),
        ),
    )


def _execute(
    agent,
    provider_call,
    *,
    request_id: str,
    count: int,
    payload=None,
):
    request = payload or {
        "model": "deepseek-v4-pro",
        "messages": [],
        "max_tokens": 4096,
    }
    return execute_llm_request(
        agent,
        request,
        provider_call,
        strict=True,
        original_request=dict(request),
        middleware_trace=[],
        task_id="task",
        turn_id="turn",
        api_request_id=request_id,
        api_call_count=count,
    )


def test_two_tool_loop_calls_get_two_durable_ordinals_not_request_ids():
    snapshots: list[dict] = []
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="a" * 32,
        on_change=lambda value: snapshots.append(value),
    )
    agent = _agent(ledger)

    _execute(
        agent,
        lambda _kwargs: SimpleNamespace(usage=_usage(10, 2)),
        request_id="reused-request-id",
        count=1,
    )
    _execute(
        agent,
        lambda _kwargs: SimpleNamespace(usage=_usage(12, 3)),
        request_id="reused-request-id",
        count=2,
    )

    calls = ledger.to_dict()["calls"]
    assert [call["ordinal"] for call in calls] == [1, 2]
    assert [call["callId"] for call in calls] == [
        f"{'a' * 32}:call:000001",
        f"{'a' * 32}:call:000002",
    ]
    assert len({call["callId"] for call in calls}) == 2
    assert all("reused-request-id" not in call["callId"] for call in calls)
    assert snapshots[-1]["calls"][-1]["usageStatus"] == "reported"


def test_durable_callback_failure_prevents_provider_dispatch():
    provider_calls = 0

    def fail_callback(_value):
        raise OSError("fake durable writer unavailable")

    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="b" * 32,
        on_change=fail_callback,
    )
    agent = _agent(ledger)

    def provider_call(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    with pytest.raises(RuntimeError, match="durable"):
        _execute(agent, provider_call, request_id="request", count=1)

    assert provider_calls == 0
    receipt = ledger.to_dict()["calls"][0]
    assert receipt["status"] == "not_dispatched"
    assert receipt["usageStatus"] == "unavailable"
    assert receipt["errorCategory"] == "provider_dispatch_fence_closed"


def test_dispatch_marker_confirmation_failure_never_calls_provider():
    snapshots: list[dict] = []
    provider_calls = 0

    def persist(snapshot):
        snapshots.append(snapshot)
        if snapshot["calls"][0]["status"] == "running":
            raise OSError("fake running marker write failed")

    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="4" * 32,
        on_change=persist,
    )
    agent = _agent(ledger)

    def provider_call(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    with pytest.raises(RuntimeError, match="dispatch marker"):
        _execute(agent, provider_call, request_id="request", count=1)

    assert provider_calls == 0
    assert [snapshot["calls"][0]["status"] for snapshot in snapshots] == [
        "reserved",
        "running",
    ]
    receipt = ledger.to_dict()["calls"][0]
    assert receipt["status"] == "failed"
    assert receipt["usageStatus"] == "unavailable"
    assert receipt["endedAtMs"] is not None


@pytest.mark.parametrize(
    ("running_marker_reached_store", "expected_recovered_status"),
    [
        (False, "not_dispatched"),
        (True, "timed_out"),
    ],
)
def test_finished_request_releases_stale_dispatch_marker_drain_owner(
    tmp_path: Path,
    monkeypatch,
    running_marker_reached_store: bool,
    expected_recovered_status: str,
):
    async def scenario():
        cache = _IdempotencyCache(
            durable_path=str(tmp_path / "stale-owner.sqlite"),
        )
        key = "stale-dispatch-marker-owner"
        fingerprint = "5" * 64
        provider_calls = 0
        agent_ref = [None, False, None]
        original_save_usage = cache._durable.save_usage

        def fail_running_marker(
            save_key,
            save_fingerprint,
            usage,
            *,
            state,
            **lease_fence,
        ):
            call_status = (
                usage["calls"][0]["status"]
                if usage.get("calls")
                else None
            )
            if call_status == "running":
                if running_marker_reached_store:
                    original_save_usage(
                        save_key,
                        save_fingerprint,
                        usage,
                        state=state,
                        **lease_fence,
                    )
                raise OSError("fake dispatch marker confirmation failure")
            return original_save_usage(
                save_key,
                save_fingerprint,
                usage,
                state=state,
                **lease_fence,
            )

        monkeypatch.setattr(
            cache._durable,
            "save_usage",
            fail_running_marker,
        )

        async def compute():
            nonlocal provider_calls
            ledger = AgentCallUsageLedger(
                provider="deepseek",
                model="deepseek-v4-pro",
                execution_id="5" * 32,
                on_change=lambda snapshot: cache.persist_usage(
                    key,
                    fingerprint,
                    snapshot,
                ),
            )
            agent = _agent(ledger)
            agent_ref[0] = agent

            def provider_call(_kwargs):
                nonlocal provider_calls
                provider_calls += 1
                return SimpleNamespace(usage=_usage(1, 1))

            return _execute(
                agent,
                provider_call,
                request_id="marker-confirmation-failed",
                count=1,
            )

        with pytest.raises(RuntimeError, match="dispatch marker"):
            await cache.get_or_set(
                key,
                fingerprint,
                compute,
                agent_ref=agent_ref,
                durable=True,
            )
        await asyncio.sleep(0)

        assert provider_calls == 0
        assert cache.has_active_usage_drain(key) is False
        before_stop = cache.durable_record(key)
        assert before_stop["usage"]["calls"][0]["status"] == (
            "running"
            if running_marker_reached_store
            else "reserved"
        )
        assert cache.stop(key, durable=True) is True
        recovered = cache.terminalize_orphaned_stopped_usage(key)
        call = recovered["usage"]["calls"][0]
        assert call["status"] == expected_recovered_status
        assert call["usageStatus"] == "unavailable"
        assert call["endedAtMs"] is not None
        if expected_recovered_status == "not_dispatched":
            assert (
                call["errorCategory"]
                == "provider_dispatch_fence_closed"
            )
        else:
            assert (
                call["errorCategory"]
                == "agent_restart_outcome_unknown"
            )
        cache._durable.close()

    asyncio.run(scenario())


def test_stop_fence_keeps_late_usage_without_reopening_status():
    ledger = AgentCallUsageLedger(
        provider="fake-provider",
        model="fake-model",
        execution_id="c" * 32,
    )
    call_id = ledger.start_call()
    ledger.mark_dispatched(call_id)
    ledger.terminalize_running(
        status="timed_out",
        error_category="completion_stopped",
    )
    ledger.finish_call(
        call_id,
        status="completed",
        usage=_usage(7, 4),
    )

    receipt = ledger.to_dict()["calls"][0]
    assert receipt["status"] == "timed_out"
    assert receipt["errorCategory"] == "completion_stopped"
    assert receipt["usageStatus"] == "reported"
    assert receipt["totalTokens"] == 11


def test_worker_shutdown_timeout_blocks_zero_settlement_until_late_usage():
    snapshots: list[dict] = []
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="e" * 32,
        on_change=lambda value: snapshots.append(value),
    )
    agent = _agent(ledger)
    late_callback = None

    def provider_call(_kwargs):
        nonlocal late_callback
        late_callback = agent._strict_late_provider_usage_callback
        raise StrictPaidWorkerShutdownTimeout(
            reason="stale_call_kill",
            grace_seconds=10,
        )

    with pytest.raises(StrictPaidWorkerShutdownTimeout):
        _execute(agent, provider_call, request_id="late-usage", count=1)

    before = ledger.to_dict()["calls"][0]
    assert before["status"] == "timed_out"
    assert before["usageStatus"] == "unavailable"
    assert before["errorCategory"] == "provider_worker_shutdown_timeout"
    assert late_callback is not None

    late_callback(SimpleNamespace(usage=_usage(9, 4)))

    after = ledger.to_dict()["calls"][0]
    assert after["status"] == "timed_out"
    assert after["usageStatus"] == "reported"
    assert after["totalTokens"] == 13
    assert snapshots[-1]["calls"][0]["totalTokens"] == 13


def test_worker_shutdown_usage_keeps_cancelled_terminal_status():
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="f" * 32,
    )
    agent = _agent(ledger)

    def provider_call(_kwargs):
        agent._interrupt_requested = True
        error = StrictPaidWorkerShutdownTimeout(
            reason="interrupt_abort",
            grace_seconds=10,
        )
        error.usage = _usage(12, 6)
        error.late_accounting_pending = False
        raise error

    with pytest.raises(StrictPaidWorkerShutdownTimeout):
        _execute(agent, provider_call, request_id="cancelled-usage", count=1)

    receipt = ledger.to_dict()["calls"][0]
    assert receipt["status"] == "cancelled"
    assert receipt["usageStatus"] == "reported"
    assert receipt["totalTokens"] == 18
    assert receipt["errorCategory"] == "completion_stopped"


def test_interrupt_reports_usage_after_cancelled_main_thread_returns(
    monkeypatch,
):
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="1" * 32,
    )
    agent = _agent(ledger)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    worker_finished = threading.Event()
    fake_client = MagicMock()
    sentinel = SimpleNamespace(
        choices=[SimpleNamespace(message="PRIVATE_LATE_CANCELLED")],
        usage=_usage(14, 9),
    )

    def create(**_kwargs):
        provider_entered.set()
        agent._interrupt_requested = True
        assert release_provider.wait(2)
        return sentinel

    def close(_client, *, reason):
        assert reason == "request_complete"
        worker_finished.set()

    fake_client.chat.completions.create.side_effect = create
    agent._strict_no_automatic_paid_retry = True
    agent._compute_non_stream_stale_timeout = lambda _kwargs: 5.0
    agent._touch_activity = MagicMock()
    agent._create_request_openai_client = MagicMock(
        return_value=fake_client
    )
    agent._abort_request_openai_client = MagicMock()
    agent._close_request_openai_client = MagicMock(side_effect=close)
    monkeypatch.setattr(
        helpers,
        "_strict_paid_shutdown_grace_seconds",
        lambda: 0.05,
    )

    def provider_call(kwargs):
        return helpers.interruptible_api_call(agent, kwargs)

    with pytest.raises(StrictPaidWorkerShutdownTimeout) as exc_info:
        _execute(
            agent,
            provider_call,
            request_id="late-after-cancel",
            count=1,
        )

    assert provider_entered.is_set()
    assert "PRIVATE_LATE_CANCELLED" not in str(exc_info.value)
    before = ledger.to_dict()["calls"][0]
    assert before["status"] == "cancelled"
    assert before["usageStatus"] == "unavailable"
    assert before["errorCategory"] == "completion_stopped"

    release_provider.set()
    assert worker_finished.wait(1)

    after = ledger.to_dict()["calls"][0]
    assert after["status"] == "cancelled"
    assert after["usageStatus"] == "reported"
    assert after["totalTokens"] == 23
    assert after["errorCategory"] == "completion_stopped"


def test_interrupt_response_during_shutdown_grace_does_not_deadlock(
    monkeypatch,
):
    usage_reported = threading.Event()

    def capture_usage(snapshot):
        calls = snapshot.get("calls") or []
        if calls and calls[0].get("usageStatus") == "reported":
            usage_reported.set()

    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="3" * 32,
        on_change=capture_usage,
    )
    agent = _agent(ledger)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    abort_seen = threading.Event()
    worker_finished = threading.Event()
    fake_client = MagicMock()
    sentinel = SimpleNamespace(
        choices=[SimpleNamespace(message="PRIVATE_DURING_GRACE")],
        usage=_usage(16, 11),
    )

    def create(**_kwargs):
        provider_entered.set()
        agent._interrupt_requested = True
        assert release_provider.wait(3)
        return sentinel

    def abort(_client, *, reason):
        assert reason == "interrupt_abort"
        abort_seen.set()

    def close(_client, *, reason):
        assert reason == "request_complete"
        worker_finished.set()

    fake_client.chat.completions.create.side_effect = create
    agent._strict_no_automatic_paid_retry = True
    agent._compute_non_stream_stale_timeout = lambda _kwargs: 5.0
    agent._touch_activity = MagicMock()
    agent._create_request_openai_client = MagicMock(
        return_value=fake_client
    )
    agent._abort_request_openai_client = MagicMock(side_effect=abort)
    agent._close_request_openai_client = MagicMock(side_effect=close)
    monkeypatch.setattr(
        helpers,
        "_strict_paid_shutdown_grace_seconds",
        lambda: 2.0,
    )

    caught: list[BaseException] = []

    def provider_call(kwargs):
        return helpers.interruptible_api_call(agent, kwargs)

    def run_request():
        try:
            _execute(
                agent,
                provider_call,
                request_id="response-during-grace",
                count=1,
            )
        except BaseException as exc:
            caught.append(exc)

    request_thread = threading.Thread(target=run_request)
    request_thread.start()
    try:
        assert provider_entered.wait(1)
        assert abort_seen.wait(1)
        time.sleep(0.05)
        release_provider.set()
        assert worker_finished.wait(1)
        request_thread.join(timeout=1)
    finally:
        release_provider.set()
        request_thread.join(timeout=3)

    assert not request_thread.is_alive()
    assert len(caught) == 1
    assert type(caught[0]) is InterruptedError
    assert "PRIVATE_DURING_GRACE" not in str(caught[0])
    assert usage_reported.wait(1)

    after = ledger.to_dict()["calls"][0]
    assert after["status"] == "cancelled"
    assert after["usageStatus"] == "reported"
    assert after["totalTokens"] == 27
    assert after["errorCategory"] == "completion_stopped"


def test_interrupt_usage_waits_for_slow_cancel_terminalization(
    monkeypatch,
):
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="2" * 32,
    )
    agent = _agent(ledger)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    worker_finished = threading.Event()
    outer_finish_entered = threading.Event()
    release_outer_finish = threading.Event()
    usage_reported = threading.Event()
    fake_client = MagicMock()
    sentinel = SimpleNamespace(
        choices=[SimpleNamespace(message="PRIVATE_SLOW_CANCELLED")],
        usage=_usage(15, 10),
    )

    def create(**_kwargs):
        provider_entered.set()
        agent._interrupt_requested = True
        assert release_provider.wait(3)
        return sentinel

    def close(_client, *, reason):
        assert reason == "request_complete"
        worker_finished.set()

    fake_client.chat.completions.create.side_effect = create
    agent._strict_no_automatic_paid_retry = True
    agent._compute_non_stream_stale_timeout = lambda _kwargs: 5.0
    agent._touch_activity = MagicMock()
    agent._create_request_openai_client = MagicMock(
        return_value=fake_client
    )
    agent._abort_request_openai_client = MagicMock()
    agent._close_request_openai_client = MagicMock(side_effect=close)
    monkeypatch.setattr(
        helpers,
        "_strict_paid_shutdown_grace_seconds",
        lambda: 0.05,
    )

    real_finish_paid_provider_call = (
        conversation_policy.finish_paid_provider_call
    )

    def slow_finish_paid_provider_call(*args, **kwargs):
        if kwargs.get("status") == "cancelled":
            outer_finish_entered.set()
            assert release_outer_finish.wait(3)
        result = real_finish_paid_provider_call(*args, **kwargs)
        receipt = ledger.to_dict()["calls"][0]
        if receipt["usageStatus"] == "reported":
            usage_reported.set()
        return result

    monkeypatch.setattr(
        conversation_policy,
        "finish_paid_provider_call",
        slow_finish_paid_provider_call,
    )

    caught: list[BaseException] = []

    def provider_call(kwargs):
        return helpers.interruptible_api_call(agent, kwargs)

    def run_request():
        try:
            _execute(
                agent,
                provider_call,
                request_id="slow-cancel-terminal",
                count=1,
            )
        except BaseException as exc:
            caught.append(exc)

    request_thread = threading.Thread(target=run_request)
    request_thread.start()
    try:
        assert provider_entered.wait(1)
        assert outer_finish_entered.wait(2)
        release_provider.set()
        assert worker_finished.wait(1)

        # The old one-second polling fence wrote timed_out here while the
        # intentionally slow cancelled terminalization was still blocked.
        time.sleep(1.1)
        during_block = ledger.to_dict()["calls"][0]
        assert during_block["status"] == "running"
        assert during_block["usageStatus"] == "unavailable"
    finally:
        release_provider.set()
        release_outer_finish.set()
        request_thread.join(timeout=2)

    assert not request_thread.is_alive()
    assert len(caught) == 1
    assert isinstance(caught[0], StrictPaidWorkerShutdownTimeout)
    assert "PRIVATE_SLOW_CANCELLED" not in str(caught[0])
    assert usage_reported.wait(1)

    after = ledger.to_dict()["calls"][0]
    assert after["status"] == "cancelled"
    assert after["usageStatus"] == "reported"
    assert after["totalTokens"] == 25
    assert after["errorCategory"] == "completion_stopped"


def test_restart_terminalizes_running_as_unknown_not_zero(tmp_path: Path):
    path = tmp_path / "agent-calls.sqlite"
    key = "scoped-normal-delivery"
    fingerprint = "f" * 64
    store = TrueMoADurableStore(str(path))
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    ledger = AgentCallUsageLedger(
        provider="fake-provider",
        model="fake-model",
        execution_id="d" * 32,
    )
    reserved_call_id = ledger.start_call()
    running_call_id = ledger.start_call()
    ledger.mark_dispatched(running_call_id)
    store.save_usage(key, fingerprint, ledger.to_dict(), state="running")
    assert store.mark_stopped(key) is True
    store.close()

    restarted = TrueMoADurableStore(str(path))
    assert restarted.terminalize_stopped_running_calls(key) is True
    record = restarted.get(key)
    reserved_call, running_call = record["usage"]["calls"]
    assert record["state"] == "stopped"
    assert reserved_call["callId"] == reserved_call_id
    assert reserved_call["status"] == "not_dispatched"
    assert reserved_call["usageStatus"] == "unavailable"
    assert reserved_call["endedAtMs"] is not None
    assert (
        reserved_call["errorCategory"]
        == "provider_dispatch_fence_closed"
    )
    assert running_call["callId"] == running_call_id
    assert running_call["status"] == "timed_out"
    assert running_call["usageStatus"] == "unavailable"
    assert running_call["inputTokens"] is None
    assert running_call["outputTokens"] is None
    assert running_call["totalTokens"] is None
    restarted.close()


def test_call_receipt_requires_reserved_then_running_before_completion():
    snapshots: list[dict] = []
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="2" * 32,
        on_change=lambda value: snapshots.append(value),
    )

    call_id = ledger.start_call()
    assert ledger.to_dict()["calls"][0]["status"] == "reserved"
    assert snapshots[-1]["calls"][0]["status"] == "reserved"

    ledger.mark_dispatched(call_id)
    assert ledger.to_dict()["calls"][0]["status"] == "running"
    assert snapshots[-1]["calls"][0]["status"] == "running"

    ledger.finish_call(
        call_id,
        status="completed",
        usage=_usage(3, 1),
    )
    assert ledger.to_dict()["calls"][0]["status"] == "completed"


def test_not_dispatched_receipt_is_exact_and_immutable():
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="3" * 32,
    )
    call_id = ledger.start_call()
    reserved = ledger.to_dict()

    ledger.finish_not_dispatched(call_id)
    fenced = ledger.to_dict()
    receipt = fenced["calls"][0]
    assert receipt["status"] == "not_dispatched"
    assert receipt["usageStatus"] == "unavailable"
    assert receipt["endedAtMs"] is not None
    assert receipt["errorCategory"] == "provider_dispatch_fence_closed"
    assert receipt["inputTokens"] is None
    assert receipt["outputTokens"] is None
    assert receipt["totalTokens"] is None
    assert receipt["cachedInputTokens"] is None
    assert receipt.get("costUsd") is None
    assert receipt.get("costStatus") is None
    assert receipt.get("costSource") is None
    assert merge_agent_call_usage(reserved, fenced) == fenced
    assert merge_agent_call_usage(fenced, reserved) == fenced

    rewritten = {
        **fenced,
        "calls": [
            {
                **receipt,
                "status": "completed",
                "inputTokens": 1,
                "outputTokens": 1,
                "totalTokens": 2,
                "cachedInputTokens": 0,
                "usageStatus": "reported",
            }
        ],
    }
    with pytest.raises(ValueError, match="terminal agent call state"):
        merge_agent_call_usage(fenced, rewritten)

    with pytest.raises(RuntimeError, match="not dispatched"):
        ledger.mark_dispatched(call_id)
    with pytest.raises(RuntimeError, match="not dispatched"):
        ledger.finish_call(
            call_id,
            status="completed",
            usage=_usage(1, 1),
        )


def test_completed_ledger_rejects_not_dispatched_call():
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="b" * 32,
    )
    call_id = ledger.start_call()
    ledger.finish_not_dispatched(call_id)
    fenced = ledger.to_dict()

    with pytest.raises(
        ValueError,
        match="completed agent ledger has unresolved provider call",
    ):
        ledger.set_status("completed")
    assert ledger.to_dict()["status"] == "running"

    with pytest.raises(
        ValueError,
        match="completed agent ledger has unresolved provider call",
    ):
        project_agent_call_usage({
            **fenced,
            "status": "completed",
        })


def test_durable_same_key_replay_never_dispatches_twice(tmp_path: Path):
    path = tmp_path / "replay.sqlite"
    key = "scoped-replay"
    fingerprint = "e" * 64
    calls = 0

    async def scenario():
        nonlocal calls
        first = _IdempotencyCache(durable_path=str(path))

        async def compute():
            nonlocal calls
            calls += 1
            ledger = AgentCallUsageLedger(
                provider="fake-provider",
                model="fake-model",
                execution_id="e" * 32,
            )
            call_id = ledger.start_call()
            ledger.mark_dispatched(call_id)
            ledger.finish_call(
                call_id,
                status="completed",
                usage=_usage(3, 2),
            )
            ledger.set_status("completed")
            snapshot = ledger.to_dict()
            return (
                {"final_response": "first", "completed": True},
                {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "agent_calls": snapshot,
                },
            )

        first_result = await first.get_or_set(
            key,
            fingerprint,
            compute,
            durable=True,
        )
        assert first_result[0]["final_response"] == "first"
        first._durable.close()

        second = _IdempotencyCache(durable_path=str(path))

        async def forbidden_compute():
            nonlocal calls
            calls += 1
            raise AssertionError("replay dispatched")

        replay = await second.get_or_set(
            key,
            fingerprint,
            forbidden_compute,
            durable=True,
        )
        assert replay[0]["failed"] is True
        assert replay[1]["agent_calls"]["status"] == "completed"
        second._durable.close()

    asyncio.run(scenario())
    assert calls == 1


def test_generic_projection_is_strict_monotonic_and_plaintext_free():
    ledger = AgentCallUsageLedger(
        provider="fake-provider",
        model="fake-model",
        execution_id="f" * 32,
    )
    call_id = ledger.start_call()
    ledger.mark_dispatched(call_id)
    running = ledger.to_dict()
    ledger.finish_call(
        call_id,
        status="completed",
        usage=_usage(5, 1),
    )
    completed = ledger.to_dict()

    merged = merge_agent_call_usage(running, completed)
    stale_merged = merge_agent_call_usage(merged, running)
    assert stale_merged == merged
    assert "messages" not in str(stale_merged).lower()
    assert "content" not in str(stale_merged).lower()

    invalid = dict(completed)
    invalid["prompt"] = "private body"
    with pytest.raises(ValueError):
        project_agent_call_usage(invalid)

    conflict = {
        **completed,
        "calls": [
            {
                **completed["calls"][0],
                "totalTokens": 999,
            }
        ],
    }
    with pytest.raises(ValueError):
        merge_agent_call_usage(completed, conflict)


def test_terminal_generic_ledger_rejects_appended_provider_call():
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="0" * 32,
    )
    first = ledger.start_call()
    ledger.mark_dispatched(first)
    ledger.finish_call(first, status="completed", usage=_usage(1, 1))
    current = ledger.to_dict()
    second = ledger.start_call()
    ledger.mark_dispatched(second)
    ledger.finish_call(second, status="completed", usage=_usage(1, 1))
    ledger.set_status("completed")
    incoming = ledger.to_dict()
    current["status"] = "completed"

    with pytest.raises(ValueError, match="call set is immutable"):
        merge_agent_call_usage(current, incoming)


def test_ninth_physical_call_is_rejected_before_provider():
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="1" * 32,
    )
    agent = _agent(ledger)
    provider_calls = 0

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    for count in range(1, AGENT_CALL_LIMIT + 1):
        _execute(
            agent,
            provider,
            request_id=f"request-{count}",
            count=count,
        )
    with pytest.raises(RuntimeError, match="call limit"):
        _execute(
            agent,
            provider,
            request_id="request-9",
            count=AGENT_CALL_LIMIT + 1,
        )

    assert provider_calls == AGENT_CALL_LIMIT


def test_signed_normal_route_drift_fails_before_provider_dispatch():
    snapshots: list[dict] = []
    workflow = _normal_workflow(callback=snapshots.append)
    assert initialize_normal_call_ledger(workflow) is None
    assert snapshots == [
        {
            "schema": "mystand.agent-call-usage.v1",
            "executionId": workflow.agent_call_ledger.execution_id,
            "status": "running",
            "calls": [],
        }
    ]
    agent = SimpleNamespace(
        provider="unexpected-provider",
        model="unexpected-model",
        max_iterations=90,
        max_tokens=99_999,
    )

    with pytest.raises(RuntimeError, match="fixed_route_mismatch"):
        bind_paid_call_ledger(workflow, agent)

    assert workflow.agent_call_ledger is agent._paid_call_usage_ledger
    assert snapshots[-1]["status"] == "failed"
    assert snapshots[-1]["calls"] == []


def test_signed_normal_route_is_rechecked_at_physical_dispatch():
    provider_calls = 0
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="2" * 32,
    )
    agent = _agent(ledger)
    agent.provider = "unexpected-provider"

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    with pytest.raises(RuntimeError, match="fixed_route_mismatch"):
        _execute(agent, provider, request_id="route-drift", count=1)

    assert provider_calls == 0
    assert ledger.to_dict()["calls"] == []


def test_signed_normal_route_binds_call_limit_without_token_cap():
    workflow = _normal_workflow()
    assert initialize_normal_call_ledger(workflow) is None
    agent = SimpleNamespace(
        provider=SIGNED_MYSTAND_AGENT_POLICY.provider,
        model=SIGNED_MYSTAND_AGENT_POLICY.model,
        max_iterations=90,
        max_tokens=99_999,
    )

    bind_paid_call_ledger(workflow, agent)

    assert workflow.agent_call_ledger is agent._paid_call_usage_ledger
    assert agent.max_iterations == 90
    assert agent.max_tokens is None


@pytest.mark.parametrize("revision", ["", "stale-policy"])
def test_signed_normal_policy_revision_is_durable_and_fail_closed(
    revision,
):
    snapshots: list[dict] = []
    workflow = _normal_workflow(
        callback=snapshots.append,
        revision=revision,
    )

    terminal = initialize_normal_call_ledger(workflow)

    assert terminal is not None
    result, usage = terminal
    assert result["failed"] is True
    assert result["error"] == "billing policy revision mismatch"
    assert usage["agent_calls"]["status"] == "failed"
    assert usage["agent_calls"]["calls"] == []
    assert workflow.agent is None
    assert [item["status"] for item in snapshots] == [
        "running",
        "failed",
    ]


def test_pre_agent_stop_settles_zero_call_ledger_without_dispatch():
    snapshots: list[dict] = []
    workflow = _normal_workflow(callback=snapshots.append)
    workflow.request.agent_ref[1] = True

    terminal = initialize_normal_call_ledger(workflow)

    assert terminal is not None
    result, usage = terminal
    assert result["interrupted"] is True
    assert usage["agent_calls"]["status"] == "cancelled"
    assert usage["agent_calls"]["calls"] == []
    assert workflow.agent is None
    assert [item["status"] for item in snapshots] == [
        "running",
        "cancelled",
    ]


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda payload: {**payload, "max_tokens": 4_097},
            "signed_mystand_paid_call",
        ),
        (
            lambda payload: {**payload, "model": "unexpected-model"},
            "signed_mystand_fixed_route_mismatch",
        ),
    ],
)
def test_execution_middleware_cannot_bypass_physical_budget(
    monkeypatch,
    mutate,
    error,
):
    provider_calls = 0
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="3" * 32,
    )
    agent = _agent(ledger)

    def middleware(payload, next_call, **_kwargs):
        return next_call(mutate(payload))

    monkeypatch.setattr(
        "xiaoban_cli.middleware.run_llm_execution_middleware",
        middleware,
    )

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    with pytest.raises(RuntimeError, match=error):
        execute_llm_request(
            agent,
            {
                "model": "deepseek-v4-pro",
                "messages": [],
                "max_tokens": 4_096,
            },
            provider,
            strict=False,
            original_request={
                "model": "deepseek-v4-pro",
                "messages": [],
                "max_tokens": 4_096,
            },
            middleware_trace=[],
            task_id="task",
            turn_id="turn",
            api_request_id="middleware-drift",
            api_call_count=1,
        )

    assert provider_calls == 0
    assert ledger.to_dict()["calls"] == []


def test_signed_normal_input_is_not_rejected_by_the_retired_byte_gate():
    provider_calls = 0
    ledger = AgentCallUsageLedger(
        provider=SIGNED_MYSTAND_AGENT_POLICY.provider,
        model=SIGNED_MYSTAND_AGENT_POLICY.model,
        execution_id="4" * 32,
    )
    agent = _agent(ledger)
    agent._paid_call_policy_revision = SIGNED_MYSTAND_AGENT_POLICY_REVISION
    agent.provider = SIGNED_MYSTAND_AGENT_POLICY.provider
    agent.model = SIGNED_MYSTAND_AGENT_POLICY.model
    agent.api_mode = "codex_responses"
    agent.max_tokens = None

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    _execute(
        agent,
        provider,
        request_id="large-normal-input",
        count=1,
        payload={
            "model": SIGNED_MYSTAND_AGENT_POLICY.model,
            "instructions": "system",
            "input": [
                {"role": "user", "content": "x" * 140_519}
            ],
            "store": False,
        },
    )

    assert provider_calls == 1
    assert ledger.to_dict()["calls"][0]["status"] == "completed"


def test_true_moa_route_guard_is_independent_at_physical_dispatch():
    provider_calls = 0
    true_moa_ledger = object()
    agent = SimpleNamespace(
        _paid_call_usage_ledger=true_moa_ledger,
        _true_moa_usage_ledger=true_moa_ledger,
        _true_moa_cancel_controller=None,
        provider="unexpected-provider",
        model="unexpected-model",
    )

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    with pytest.raises(RuntimeError, match="fixed true MoA final route"):
        execute_llm_request(
            agent,
            {
                "model": "deepseek-v4-pro",
                "messages": [],
                "max_tokens": 4_096,
            },
            provider,
            strict=True,
            original_request={},
            middleware_trace=[],
            task_id="task",
            turn_id="turn",
            api_request_id="true-moa-route-drift",
            api_call_count=1,
        )

    assert provider_calls == 0


def test_true_moa_failed_call_reservation_closes_not_dispatched_before_provider():
    provider_calls = 0

    def fail_callback(_value):
        raise OSError("fake durable writer unavailable")

    ledger = TrueMoAUsageLedger(
        TrueMoASnapshot(
            mode=TRUE_MOA_MODE,
            mode_epoch="1",
            preset_id=TRUE_MOA_PRESET_ID,
            preset_revision=TRUE_MOA_PRESET_REVISION,
        ),
        on_change=fail_callback,
    )
    ledger.start_slot(FINAL_EXECUTOR_SLOT, notify=False)
    controller = TrueMoACancelController()
    agent = SimpleNamespace(
        _paid_call_usage_ledger=ledger,
        _true_moa_usage_ledger=ledger,
        _true_moa_cancel_controller=controller,
        _interrupt_requested=False,
        provider=FINAL_EXECUTOR_SLOT.provider,
        model=FINAL_EXECUTOR_SLOT.model,
        api_mode="codex_responses",
        max_tokens=4_096,
    )

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage(1, 1))

    with pytest.raises(RuntimeError, match="durable reservation failed"):
        execute_llm_request(
            agent,
            {
                "model": FINAL_EXECUTOR_SLOT.model,
                "instructions": "system",
                "input": [{"role": "user", "content": "hello"}],
                "store": False,
            },
            provider,
            strict=True,
            original_request={},
            middleware_trace=[],
            task_id="task",
            turn_id="turn",
            api_request_id="true-moa-reservation-failure",
            api_call_count=1,
        )

    final_calls = [
        call for call in ledger.to_dict()["calls"]
        if call["role"] == "final_executor"
    ]
    assert provider_calls == 0
    assert controller.state == "failed"
    assert len(final_calls) == 1
    assert final_calls[0]["status"] == "not_dispatched"
    assert final_calls[0]["usageStatus"] == "unavailable"
    assert final_calls[0]["errorCategory"] == "provider_dispatch_fence_closed"


def test_billing_policy_revision_is_in_idempotency_fingerprint():
    body = {"model": "test", "messages": []}
    headers = {
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Request-Fingerprint": "a" * 64,
        SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER: (
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
        ),
    }

    baseline = APIServerAdapter._chat_idempotency_fingerprint(
        body,
        headers,
    )
    changed = APIServerAdapter._chat_idempotency_fingerprint(
        body,
        {
            **headers,
            SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER: "stale-policy",
        },
    )

    assert changed != baseline


def test_billing_policy_registry_keeps_revision_specific_policy():
    assert (
        resolve_signed_mystand_agent_policy(
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
        )
        is SIGNED_MYSTAND_AGENT_POLICY
    )
    assert SIGNED_MYSTAND_AGENT_POLICY_REGISTRY == {
        LEGACY_SIGNED_MYSTAND_AGENT_POLICY_REVISION: (
            LEGACY_SIGNED_MYSTAND_AGENT_POLICY
        ),
        SIGNED_MYSTAND_AGENT_POLICY_REVISION: (
            SIGNED_MYSTAND_AGENT_POLICY
        ),
    }
    with pytest.raises(TypeError):
        SIGNED_MYSTAND_AGENT_POLICY_REGISTRY["future"] = (
            SIGNED_MYSTAND_AGENT_POLICY
        )
    assert (
        resolve_signed_mystand_agent_policy(
            LEGACY_SIGNED_MYSTAND_AGENT_POLICY_REVISION
        )
        is LEGACY_SIGNED_MYSTAND_AGENT_POLICY
    )
    assert LEGACY_SIGNED_MYSTAND_AGENT_POLICY.input_max_bytes == 131072


def test_agent_constructor_failure_settles_prebound_zero_call_ledger():
    snapshots: list[dict] = []
    create_calls = 0

    class FailingAdapter:
        _header_value = staticmethod(_header_value)

        @staticmethod
        def _bind_api_server_session(**_kwargs):
            return None

        @staticmethod
        def _create_agent(**_kwargs):
            nonlocal create_calls
            create_calls += 1
            assert snapshots[0]["status"] == "running"
            assert snapshots[0]["calls"] == []
            raise RuntimeError("synthetic configuration failure")

    request = SimpleNamespace(
        adapter=FailingAdapter(),
        user_message="synthetic request",
        conversation_history=[],
        effective_system_prompt=None,
        session_id="synthetic-session",
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        traced_tool_start=lambda *_args: None,
        traced_tool_complete=lambda *_args: None,
        agent_ref=[None, False, None],
        gateway_session_key=None,
        request_headers={
            SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER: (
                SIGNED_MYSTAND_AGENT_POLICY_REVISION
            ),
        },
        async_delivery=False,
        true_moa_snapshot=None,
        paid_call_usage_callback=snapshots.append,
        request_user_id="",
        request_message_id="",
        request_delivery_id="xbd_" + ("6" * 40),
        enabled_toolsets_override=[],
        mystand_request=True,
        durable_paid_call=True,
        memory_identity=None,
        metadata_trace=None,
        trace_state=SimpleNamespace(tool_count=0),
    )

    result, usage = TrueMoARunWorkflow(request).run()

    assert create_calls == 1
    assert result["failed"] is True
    assert result["error"] == "agent run failed"
    assert usage["agent_calls"]["status"] == "failed"
    assert usage["agent_calls"]["calls"] == []
    assert [item["status"] for item in snapshots] == [
        "running",
        "failed",
    ]
