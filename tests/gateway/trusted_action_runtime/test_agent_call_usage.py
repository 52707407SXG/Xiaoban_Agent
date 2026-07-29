"""No-network gates for signed My Stand per-provider-call accounting."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.true_moa_conversation_policy import execute_llm_request
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
    SIGNED_MYSTAND_AGENT_POLICY,
    SIGNED_MYSTAND_AGENT_POLICY_REGISTRY,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
    resolve_signed_mystand_agent_policy,
)
from xiaoban.trusted_runtime.true_moa_durable import TrueMoADurableStore


def _usage(input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=0,
    )


def _agent(ledger: AgentCallUsageLedger):
    return SimpleNamespace(
        _paid_call_usage_ledger=ledger,
        _true_moa_usage_ledger=None,
        _true_moa_cancel_controller=None,
        _paid_call_policy_revision=(
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
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


def _execute(agent, provider_call, *, request_id: str, count: int):
    return execute_llm_request(
        agent,
        {
            "model": "deepseek-v4-pro",
            "messages": [],
            "max_tokens": 4096,
        },
        provider_call,
        strict=True,
        original_request={
            "model": "deepseek-v4-pro",
            "messages": [],
            "max_tokens": 4096,
        },
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
    assert receipt["status"] == "reserved"
    assert receipt["usageStatus"] == "unavailable"


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


def test_signed_normal_route_binds_shared_caps_before_dispatch():
    workflow = _normal_workflow()
    assert initialize_normal_call_ledger(workflow) is None
    agent = SimpleNamespace(
        provider="deepseek",
        model="deepseek-v4-pro",
        max_iterations=90,
        max_tokens=99_999,
    )

    bind_paid_call_ledger(workflow, agent)

    assert workflow.agent_call_ledger is agent._paid_call_usage_ledger
    assert agent.max_iterations == 8
    assert agent.max_tokens == 4096


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
            lambda payload: {
                **payload,
                "messages": [
                    {"role": "user", "content": "x" * 131_072},
                ],
            },
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
        SIGNED_MYSTAND_AGENT_POLICY_REVISION: (
            SIGNED_MYSTAND_AGENT_POLICY
        ),
    }
    with pytest.raises(TypeError):
        SIGNED_MYSTAND_AGENT_POLICY_REGISTRY["future"] = (
            SIGNED_MYSTAND_AGENT_POLICY
        )


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
        fact_requirement=None,
        completion_protocol="",
        completion_binding={},
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
        resolve_mystand_initial_tool_choice=(
            lambda *_args, **_kwargs: ""
        ),
        run_mystand_preexecuted_evidence=lambda **_kwargs: None,
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
