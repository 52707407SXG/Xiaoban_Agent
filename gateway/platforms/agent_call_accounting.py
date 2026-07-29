"""Normal-mode binding and terminal projection for provider call receipts."""

from __future__ import annotations

from typing import Any

from xiaoban.trusted_runtime.agent_call_usage import (
    AGENT_CALL_DURABLE_CONFIRM_SECONDS,
    AgentCallUsageLedger,
)
from xiaoban.trusted_runtime.paid_call_policy import (
    SIGNED_MYSTAND_AGENT_POLICY,
    SIGNED_MYSTAND_AGENT_POLICY_REGISTRY,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
    enforce_fixed_paid_call_route,
    enforce_signed_mystand_policy_revision,
    resolve_signed_mystand_agent_policy,
)
from xiaoban.trusted_runtime.true_moa_cancel import (
    TrueMoACancelController as PaidCallCancelController,
)


def _is_signed_normal_request(workflow: Any) -> bool:
    request = workflow.request
    return bool(
        request.mystand_request
        and request.durable_paid_call
        and getattr(request, "true_moa_snapshot", None) is None
    )


def initialize_normal_call_ledger(workflow: Any) -> tuple | None:
    """Persist the zero-call normal ledger before Agent construction."""

    if not _is_signed_normal_request(workflow):
        return None
    if workflow.agent_call_ledger is not None:
        raise RuntimeError("signed My Stand call ledger initialized twice")
    controller = PaidCallCancelController()
    workflow.agent_call_controller = controller
    agent_ref = workflow.request.agent_ref
    if agent_ref is not None:
        while len(agent_ref) < 3:
            agent_ref.append(None)
        agent_ref[2] = controller
        if agent_ref[1]:
            controller.cancel()
    observed_revision = workflow.request.adapter._header_value(
        workflow.request.request_headers,
        SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
    )
    policy = SIGNED_MYSTAND_AGENT_POLICY_REGISTRY.get(
        str(observed_revision or "").strip(),
        SIGNED_MYSTAND_AGENT_POLICY,
    )
    ledger = AgentCallUsageLedger(
        provider=policy.provider,
        model=policy.model,
        on_change=workflow.request.paid_call_usage_callback,
        max_calls=policy.call_limit,
    )
    workflow.agent_call_ledger = ledger
    workflow.agent_call_terminal_settlement_confirmed = None
    if not ledger.confirm_change():
        return failed_normal_result(
            workflow,
            interrupted=False,
            error="provider call durable initialization failed",
        )
    if not workflow.request.request_delivery_id:
        return failed_normal_result(
            workflow,
            interrupted=False,
            error="durable delivery identity missing",
        )
    try:
        revision = enforce_signed_mystand_policy_revision(
            observed_revision
        )
        policy = resolve_signed_mystand_agent_policy(revision)
    except BaseException:
        return failed_normal_result(
            workflow,
            interrupted=False,
            error="billing policy revision mismatch",
        )
    workflow.agent_call_policy_revision = revision
    workflow.agent_call_policy = policy
    if (
        controller.state == "cancelled"
        or (
            agent_ref is not None
            and len(agent_ref) > 1
            and agent_ref[1]
        )
    ):
        controller.cancel()
        return failed_normal_result(
            workflow,
            interrupted=True,
            error="completion stopped",
        )
    return None


def _settle_normal_call_ledger(
    workflow: Any,
    *,
    ledger_status: str,
    receipt_status: str,
    error_category: str,
) -> bool:
    ledger = workflow.agent_call_ledger
    snapshot = ledger.to_dict()
    if snapshot["status"] != "running":
        prior = getattr(
            workflow,
            "agent_call_terminal_settlement_confirmed",
            None,
        )
        return bool(prior)
    ledger.terminalize_running(
        status=receipt_status,
        error_category=error_category,
        notify=False,
    )
    ledger.set_status(ledger_status, notify=False)
    notification = ledger.notify_change_async()
    notification.wait(AGENT_CALL_DURABLE_CONFIRM_SECONDS)
    workflow.agent_call_terminal_settlement_confirmed = (
        notification.confirmed
    )
    return notification.confirmed


def bind_paid_call_ledger(workflow: Any, agent: Any) -> None:
    """Attach a pre-existing ledger and validate its physical route."""

    if workflow.true_moa_ledger is not None:
        agent._paid_call_usage_ledger = workflow.true_moa_ledger
        return
    if not _is_signed_normal_request(workflow):
        return
    ledger = workflow.agent_call_ledger
    if ledger is None:
        raise RuntimeError(
            "signed My Stand call ledger was not initialized"
        )
    controller = getattr(workflow, "agent_call_controller", None)
    if controller is None:
        raise RuntimeError(
            "signed My Stand call controller was not initialized"
        )
    agent._paid_call_usage_ledger = ledger
    agent._paid_call_cancel_controller = controller
    agent._paid_call_policy_revision = (
        workflow.agent_call_policy_revision
    )
    policy = workflow.agent_call_policy
    if policy is None:
        raise RuntimeError("signed My Stand paid-call policy is missing")
    agent.max_iterations = min(
        max(
            1,
            int(
                getattr(agent, "max_iterations", policy.call_limit)
                or policy.call_limit
            ),
        ),
        policy.call_limit,
    )
    agent.max_tokens = policy.output_max_tokens
    try:
        enforce_fixed_paid_call_route(
            policy,
            provider=getattr(agent, "provider", ""),
            model=getattr(agent, "model", ""),
            error_code="signed_mystand_fixed_route_mismatch",
        )
    except BaseException:
        controller.fail()
        _settle_normal_call_ledger(
            workflow,
            ledger_status="failed",
            receipt_status="failed",
            error_category="agent_route_mismatch",
        )
        raise


def finalize_normal_call_usage(
    workflow: Any,
    result: dict[str, Any],
    usage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Land the terminal generic ledger before returning public output."""

    ledger = workflow.agent_call_ledger
    if ledger is None:
        return result, usage
    controller = getattr(workflow, "agent_call_controller", None)
    interrupted = bool(result.get("interrupted"))
    failed = bool(
        interrupted
        or result.get("partial")
        or result.get("failed")
        or not result.get("completed", True)
    )
    controller_forced_failure = False
    if controller is not None:
        controller_state = controller.state
        if controller_state == "cancelled":
            interrupted = True
            failed = True
        elif controller_state == "failed":
            controller_forced_failure = not failed
            failed = True
        if interrupted:
            controller.cancel()
        elif failed:
            controller.fail()
    if interrupted:
        terminal_status = "cancelled"
        fence_status = "timed_out"
        error_category = "completion_stopped"
    elif failed:
        terminal_status = "failed"
        fence_status = "failed"
        error_category = "agent_run_failed"
    else:
        terminal_status = "completed"
        fence_status = "failed"
        error_category = "provider_call_unsettled"
    settlement_confirmed = _settle_normal_call_ledger(
        workflow,
        ledger_status=terminal_status,
        receipt_status=fence_status,
        error_category=error_category,
    )
    snapshot = ledger.to_dict()
    result["_agent_call_usage"] = snapshot
    usage["agent_calls"] = snapshot
    controller_blocked = False
    if settlement_confirmed and not failed and controller is not None:
        controller_blocked = not controller.complete()
    elif (
        not settlement_confirmed
        and controller is not None
        and controller.state == "running"
    ):
        controller.fail()
    stopped = bool(
        controller is not None
        and controller.state == "cancelled"
    )
    if stopped or controller_blocked or controller_forced_failure:
        result.update(
            {
                "final_response": "",
                "messages": [],
                "completed": False,
                "failed": True,
                "interrupted": stopped,
                "error": (
                    "completion stopped"
                    if stopped
                    else "provider call terminalization failed"
                ),
            }
        )
    elif not settlement_confirmed:
        result.update(
            {
                "final_response": "",
                "messages": [],
                "completed": False,
                "failed": True,
                "interrupted": interrupted,
                "error": "provider call durable settlement failed",
            }
        )
    return result, usage


def failed_normal_result(
    workflow: Any,
    *,
    interrupted: bool,
    error: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = {
        "final_response": "",
        "messages": [],
        "completed": False,
        "failed": True,
        "interrupted": interrupted,
        "error": error,
        "_mystand_request": True,
    }
    usage = {
        "input_tokens": (
            getattr(workflow.agent, "session_prompt_tokens", 0) or 0
        ),
        "output_tokens": (
            getattr(workflow.agent, "session_completion_tokens", 0) or 0
        ),
        "total_tokens": (
            getattr(workflow.agent, "session_total_tokens", 0) or 0
        ),
    }
    return finalize_normal_call_usage(workflow, result, usage)


__all__ = [
    "bind_paid_call_ledger",
    "failed_normal_result",
    "finalize_normal_call_usage",
    "initialize_normal_call_ledger",
]
