"""Strict true-MoA policy adapters for the mature conversation loop."""

from __future__ import annotations

from typing import Any, Callable

from agent.paid_call_accounting import (
    finish_paid_provider_call,
    record_strict_terminal_usage,
)


def strict_mode(agent: Any) -> bool:
    return bool(getattr(agent, "_strict_no_automatic_paid_retry", False))


def initialize_session_side_effects(agent: Any, logger: Any) -> None:
    """Run legacy session hooks only when the fixed paid-call ledger is absent."""

    if strict_mode(agent):
        return
    try:
        from xiaoban_cli.plugins import invoke_hook

        invoke_hook(
            "on_session_start",
            session_id=agent.session_id,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_start hook failed: %s", exc)
    try:
        from agent.credits_tracker import seed_credits_at_session_start

        seed_credits_at_session_start(agent)
    except Exception:
        logger.debug("cold-start credits seed failed (fail-open)", exc_info=True)


def prepare_llm_request(
    agent: Any,
    api_kwargs: dict[str, Any],
    *,
    task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bool]:
    """Apply extension middleware only to the legacy, non-strict route."""

    if strict_mode(agent):
        return api_kwargs, dict(api_kwargs), [], True
    try:
        from xiaoban_cli.middleware import apply_llm_request_middleware

        result = apply_llm_request_middleware(
            api_kwargs,
            task_id=task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=agent.session_id or "",
            platform=agent.platform or "",
            model=agent.model,
            provider=agent.provider,
            base_url=agent.base_url,
            api_mode=agent.api_mode,
            api_call_count=api_call_count,
        )
        return result.payload, result.original_payload, result.trace, False
    except Exception:
        return api_kwargs, dict(api_kwargs), [], False


def execute_llm_request(
    agent: Any,
    api_kwargs: dict[str, Any],
    perform_api_call: Callable[[dict[str, Any]], Any],
    *,
    strict: bool,
    original_request: dict[str, Any],
    middleware_trace: list[dict[str, Any]],
    task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
) -> Any:
    """Run one provider request through either the strict fence or middleware."""

    true_moa_ledger = getattr(agent, "_true_moa_usage_ledger", None)
    ledger = (
        getattr(agent, "_paid_call_usage_ledger", None)
        or true_moa_ledger
    )
    normal_policy = None
    if ledger is not None and true_moa_ledger is None:
        from xiaoban.trusted_runtime.paid_call_policy import (
            resolve_signed_mystand_agent_policy,
        )

        normal_policy = resolve_signed_mystand_agent_policy(
            getattr(agent, "_paid_call_policy_revision", "")
        )
    controller = getattr(agent, "_true_moa_cancel_controller", None)
    if controller is None:
        controller = getattr(agent, "_paid_call_cancel_controller", None)

    def accounted_provider_call(next_api_kwargs: dict[str, Any]) -> Any:
        if true_moa_ledger is not None:
            from xiaoban.trusted_runtime.true_moa import (
                enforce_true_moa_dispatch_budget,
                enforce_true_moa_final_route,
            )

            enforce_true_moa_final_route(
                provider=getattr(agent, "provider", ""),
                model=next_api_kwargs.get("model"),
            )
            enforce_true_moa_dispatch_budget(
                role="final_executor",
                payload=next_api_kwargs,
            )
        elif normal_policy is not None:
            from xiaoban.trusted_runtime.paid_call_policy import (
                enforce_fixed_paid_call_route,
                enforce_paid_call_dispatch_budget,
                enforce_signed_mystand_policy_revision,
            )

            enforce_signed_mystand_policy_revision(
                getattr(agent, "_paid_call_policy_revision", "")
            )
            enforce_fixed_paid_call_route(
                normal_policy,
                provider=getattr(agent, "provider", ""),
                model=next_api_kwargs.get("model"),
                error_code="signed_mystand_fixed_route_mismatch",
            )
            enforce_paid_call_dispatch_budget(
                normal_policy,
                payload=next_api_kwargs,
                error_prefix="signed_mystand_paid_call",
            )
        if controller is not None and not controller.try_begin_dispatch(
            f"final-llm-reservation:{api_request_id}"
        ):
            raise InterruptedError(
                "True MoA cancelled before final provider dispatch"
            )
        if ledger is None:
            call_id = None
        elif true_moa_ledger is not None:
            call_id = ledger.start_final_call(
                api_request_id,
                notify=False,
            )
        else:
            call_id = ledger.start_call(
                provider=str(agent.provider or ""),
                model=str(agent.model or ""),
                role=normal_policy.role,
                notify=False,
            )
        if ledger is not None and not ledger.confirm_change():
            if controller is not None:
                controller.fail()
            # The durable owner may still have only the reserved snapshot.
            # Never invent a physical dispatch or a zero-call terminal here.
            raise RuntimeError("provider call durable reservation failed")
        if controller is not None and not controller.try_begin_dispatch(
            f"final-llm:{api_request_id}"
        ):
            if ledger is not None:
                ledger.finish_not_dispatched(call_id, notify=False)
                if not ledger.confirm_change():
                    raise RuntimeError(
                        "provider call not-dispatched confirmation failed"
                    )
            raise InterruptedError(
                "True MoA cancelled before final provider dispatch"
            )
        if ledger is not None:
            ledger.mark_dispatched(call_id, notify=False)
            if not ledger.confirm_change():
                if controller is not None:
                    controller.fail()
                finish_paid_provider_call(
                    agent,
                    ledger,
                    call_id,
                    status="failed",
                    error_category=(
                        "durable_dispatch_confirmation_failed"
                    ),
                )
                raise RuntimeError(
                    "provider call durable dispatch marker failed"
                )
        try:
            response = perform_api_call(next_api_kwargs)
        except BaseException as exc:
            cancelled = bool(
                agent._interrupt_requested
                or (
                    controller is not None
                    and controller.state == "cancelled"
                )
            )
            finish_paid_provider_call(
                agent,
                ledger,
                call_id,
                status="cancelled" if cancelled else "failed",
                response=exc,
                error_category=(
                    "completion_stopped"
                    if cancelled
                    else "provider_call_failed"
                ),
            )
            raise
        finish_paid_provider_call(
            agent,
            ledger,
            call_id,
            status="completed",
            response=response,
        )
        if strict and (
            agent._interrupt_requested
            or (
                controller is not None
                and controller.state != "running"
            )
        ):
            record_strict_terminal_usage(agent, response)
            raise InterruptedError(
                "True MoA cancelled before response consumption"
            )
        return response

    if strict:
        return accounted_provider_call(api_kwargs)

    from xiaoban_cli.middleware import run_llm_execution_middleware

    return run_llm_execution_middleware(
        api_kwargs,
        accounted_provider_call,
        original_request=original_request,
        task_id=task_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        session_id=agent.session_id or "",
        platform=agent.platform or "",
        model=agent.model,
        provider=agent.provider,
        base_url=agent.base_url,
        api_mode=agent.api_mode,
        api_call_count=api_call_count,
        middleware_trace=list(middleware_trace),
    )


def strict_failure_result(
    agent: Any,
    messages: list[Any],
    conversation_history: list[Any],
    *,
    api_call_count: int,
    error: str,
    partial: bool = False,
    cleanup_task_id: str | None = None,
    drop_scaffolding: bool = False,
    include_final_response: bool = True,
) -> dict[str, Any]:
    if cleanup_task_id is not None:
        agent._cleanup_task_resources(cleanup_task_id)
    if drop_scaffolding:
        agent._drop_trailing_empty_response_scaffolding(messages)
    agent._persist_session(messages, conversation_history)
    result: dict[str, Any] = {
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": error,
    }
    if include_final_response:
        result["final_response"] = None
    if partial:
        result["partial"] = True
    return result


def claim_response_consumption(agent: Any, api_request_id: str) -> bool:
    if not strict_mode(agent):
        return True
    controller = getattr(agent, "_true_moa_cancel_controller", None)
    allowed = not agent._interrupt_requested
    if allowed and controller is not None:
        allowed = controller.try_begin_dispatch(
            f"response-consume:{api_request_id}"
        )
    if not allowed and controller is not None:
        controller.cancel()
    return allowed


def claim_public_result(
    agent: Any,
    api_request_id: str,
    *,
    kind: str,
) -> bool:
    if not strict_mode(agent):
        return True
    controller = getattr(agent, "_true_moa_cancel_controller", None)
    allowed = not agent._interrupt_requested
    if allowed and controller is not None:
        deferred = getattr(agent, "_defer_true_moa_final_commit", False)
        key = f"{kind}-response"
        allowed = (
            controller.try_begin_dispatch(f"{key}-stage:{api_request_id}")
            if deferred
            else controller.try_commit_final(f"{key}:{api_request_id}")
        )
    if not allowed and controller is not None:
        controller.cancel()
    return allowed


def emit_post_api_request(
    agent: Any,
    assistant_message: Any,
    response: Any,
    *,
    finish_reason: str,
    api_start_time: float,
    api_duration: float,
    api_messages: list[Any],
    task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
) -> None:
    if strict_mode(agent):
        return
    try:
        from xiaoban_cli.plugins import has_hook, invoke_hook

        if not has_hook("post_api_request"):
            return
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        text = assistant_message.content or ""
        invoke_hook(
            "post_api_request",
            task_id=task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=agent.session_id or "",
            platform=agent.platform or "",
            model=agent.model,
            provider=agent.provider,
            base_url=agent.base_url,
            api_mode=agent.api_mode,
            api_call_count=api_call_count,
            api_duration=api_duration,
            started_at=api_start_time,
            ended_at=api_start_time + api_duration,
            finish_reason=finish_reason,
            message_count=len(api_messages),
            response_model=getattr(response, "model", None),
            response=agent._api_response_payload_for_hook(
                response,
                assistant_message,
                finish_reason=finish_reason,
            ),
            usage=agent._usage_summary_for_api_request_hook(response),
            assistant_message=assistant_message,
            assistant_content_chars=len(text),
            assistant_tool_call_count=len(tool_calls),
        )
    except Exception:
        pass
