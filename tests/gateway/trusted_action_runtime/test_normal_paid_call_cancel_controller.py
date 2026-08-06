"""No-network gates for signed normal paid-call cancellation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.true_moa_conversation_policy import (
    claim_public_result,
    claim_response_consumption,
    execute_llm_request,
)
from agent.true_moa_tool_fence import (
    claim_strict_tool_dispatch,
    claim_strict_tool_execute,
    claim_strict_tool_handler,
    claim_strict_tool_result,
)
from gateway.platforms.agent_call_accounting import (
    bind_paid_call_ledger,
    finalize_normal_call_usage,
    initialize_normal_call_ledger,
)
from xiaoban.trusted_runtime.paid_call_policy import (
    SIGNED_MYSTAND_AGENT_POLICY_REVISION,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
)


def _header_value(headers, name: str) -> str:
    for key, value in dict(headers or {}).items():
        if str(key).lower() == name.lower():
            return str(value)
    return ""


def _workflow(*, callback=None):
    return SimpleNamespace(
        true_moa_ledger=None,
        agent_call_ledger=None,
        agent_call_policy_revision="",
        agent_call_policy=None,
        agent_call_terminal_settlement_confirmed=None,
        agent=None,
        request=SimpleNamespace(
            adapter=SimpleNamespace(_header_value=_header_value),
            request_headers={
                SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER: (
                    SIGNED_MYSTAND_AGENT_POLICY_REVISION
                ),
            },
            mystand_request=True,
            durable_paid_call=True,
            true_moa_snapshot=None,
            paid_call_usage_callback=callback,
            agent_ref=[None, False, None],
            request_delivery_id="xbd_" + ("9" * 40),
        ),
    )


def _agent():
    return SimpleNamespace(
        _true_moa_usage_ledger=None,
        _true_moa_cancel_controller=None,
        _strict_no_automatic_paid_retry=True,
        _interrupt_requested=False,
        provider="deepseek",
        model="deepseek-v4-pro",
        api_mode="chat_completions",
        base_url="",
        api_key="",
        max_iterations=90,
        max_tokens=99_999,
        session_id="test-session",
        platform="test",
    )


def _usage(input_tokens: int = 1, output_tokens: int = 1):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=0,
    )


def _execute(agent, provider_call, *, request_id: str):
    payload = {
        "model": "deepseek-v4-pro",
        "messages": [],
        "max_tokens": 4096,
    }
    return execute_llm_request(
        agent,
        payload,
        provider_call,
        strict=True,
        original_request=dict(payload),
        middleware_trace=[],
        task_id="task",
        turn_id="turn",
        api_request_id=request_id,
        api_call_count=1,
    )


def _success_result():
    return {
        "final_response": "safe answer",
        "messages": [{"role": "assistant", "content": "safe answer"}],
        "completed": True,
        "failed": False,
    }


def _empty_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_controller_is_installed_before_agent_and_bound_to_dispatch():
    workflow = _workflow()

    assert initialize_normal_call_ledger(workflow) is None

    controller = workflow.agent_call_controller
    assert controller.state == "running"
    assert workflow.request.agent_ref[2] is controller

    agent = _agent()
    bind_paid_call_ledger(workflow, agent)

    assert agent._paid_call_cancel_controller is controller


def test_stop_before_reservation_dispatches_zero_provider_calls():
    workflow = _workflow()
    assert initialize_normal_call_ledger(workflow) is None
    agent = _agent()
    bind_paid_call_ledger(workflow, agent)
    assert workflow.agent_call_controller.cancel() is True
    provider_calls = 0

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage())

    with pytest.raises(InterruptedError, match="cancelled before"):
        _execute(agent, provider, request_id="stop-before-reservation")

    assert provider_calls == 0
    assert workflow.agent_call_ledger.to_dict()["calls"] == []


def test_normal_stop_fences_tool_result_response_and_public_commit():
    workflow = _workflow()
    assert initialize_normal_call_ledger(workflow) is None
    agent = _agent()
    bind_paid_call_ledger(workflow, agent)
    assert workflow.agent_call_controller.cancel() is True

    assert claim_strict_tool_dispatch(agent, "tool-1") is False
    assert claim_strict_tool_handler(agent, "tool-1") is False
    assert claim_strict_tool_execute(agent, "tool-1") is False
    assert claim_strict_tool_result(agent, "tool-1") is False
    assert claim_response_consumption(agent, "request-1") is False
    assert claim_public_result(agent, "request-1", kind="final") is False


def test_stop_after_durable_reservation_dispatches_zero_provider_calls():
    snapshots: list[dict] = []
    workflow = None

    def persist(snapshot):
        snapshots.append(snapshot)
        if snapshot["status"] == "running" and snapshot["calls"]:
            controller = getattr(workflow, "agent_call_controller", None)
            if controller is not None:
                controller.cancel()

    workflow = _workflow(callback=persist)
    assert initialize_normal_call_ledger(workflow) is None
    agent = _agent()
    bind_paid_call_ledger(workflow, agent)
    provider_calls = 0

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage())

    with pytest.raises(InterruptedError, match="cancelled before"):
        _execute(agent, provider, request_id="stop-after-reservation")

    assert provider_calls == 0
    receipt = workflow.agent_call_ledger.to_dict()["calls"][0]
    assert receipt["status"] == "not_dispatched"
    assert receipt["usageStatus"] == "unavailable"
    assert receipt["errorCategory"] == "provider_dispatch_fence_closed"
    assert any(
        item["calls"] and item["calls"][0]["status"] == "reserved"
        for item in snapshots
    )


def test_unique_dispatch_keys_allow_multiple_tool_loop_provider_calls():
    workflow = _workflow()
    assert initialize_normal_call_ledger(workflow) is None
    agent = _agent()
    bind_paid_call_ledger(workflow, agent)
    provider_calls = 0

    def provider(_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=_usage())

    _execute(agent, provider, request_id="tool-loop-round-1")
    _execute(agent, provider, request_id="tool-loop-round-2")

    assert provider_calls == 2
    assert [
        (receipt["ordinal"], receipt["status"])
        for receipt in workflow.agent_call_ledger.to_dict()["calls"]
    ] == [(1, "completed"), (2, "completed")]


def test_success_completes_controller_after_durable_terminal_confirmation():
    observed: list[tuple[str, str | None]] = []
    workflow = None

    def persist(snapshot):
        controller = getattr(workflow, "agent_call_controller", None)
        observed.append(
            (
                snapshot["status"],
                controller.state if controller is not None else None,
            )
        )

    workflow = _workflow(callback=persist)
    assert initialize_normal_call_ledger(workflow) is None

    result, usage = finalize_normal_call_usage(
        workflow,
        _success_result(),
        _empty_usage(),
    )

    assert observed[-1] == ("completed", "running")
    assert workflow.agent_call_controller.state == "completed"
    assert result["final_response"] == "safe answer"
    assert usage["agent_calls"]["status"] == "completed"


def test_stop_winning_terminal_confirmation_blocks_public_output():
    workflow = None

    def persist(snapshot):
        if snapshot["status"] == "completed":
            workflow.agent_call_controller.cancel()

    workflow = _workflow(callback=persist)
    assert initialize_normal_call_ledger(workflow) is None

    result, usage = finalize_normal_call_usage(
        workflow,
        _success_result(),
        _empty_usage(),
    )

    assert workflow.agent_call_controller.state == "cancelled"
    assert result["final_response"] == ""
    assert result["messages"] == []
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["interrupted"] is True
    assert result["error"] == "completion stopped"
    assert usage["agent_calls"]["status"] == "completed"


def test_pre_failed_controller_rejects_apparent_success_output():
    workflow = _workflow()
    assert initialize_normal_call_ledger(workflow) is None
    assert workflow.agent_call_controller.fail() is True

    result, usage = finalize_normal_call_usage(
        workflow,
        _success_result(),
        _empty_usage(),
    )

    assert workflow.agent_call_controller.state == "failed"
    assert result["final_response"] == ""
    assert result["messages"] == []
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["interrupted"] is False
    assert result["error"] == "provider call terminalization failed"
    assert usage["agent_calls"]["status"] == "failed"


def test_failed_terminal_result_fails_controller():
    workflow = _workflow()
    assert initialize_normal_call_ledger(workflow) is None

    result, usage = finalize_normal_call_usage(
        workflow,
        {
            "final_response": "",
            "messages": [],
            "completed": False,
            "failed": True,
            "error": "agent failed",
        },
        _empty_usage(),
    )

    assert workflow.agent_call_controller.state == "failed"
    assert result["failed"] is True
    assert usage["agent_calls"]["status"] == "failed"
