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
        enforce_strict_paid_request(agent, next_api_kwargs)
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
            # The Provider callable has not run. Close only this local
            # reservation as not-dispatched so billing/recovery never sees a
            # permanently active paid call after the durable writer failed.
            ledger.finish_not_dispatched(call_id, notify=False)
            if controller is not None:
                controller.fail()
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
        previous_late_usage_callback = getattr(
            agent,
            "_strict_late_provider_usage_callback",
            None,
        )

        def persist_late_usage(late_response: Any) -> None:
            finish_paid_provider_call(
                agent,
                ledger,
                call_id,
                status="timed_out",
                response=late_response,
                error_category="provider_worker_shutdown_timeout",
            )

        agent._strict_late_provider_usage_callback = persist_late_usage
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
                status=(
                    "cancelled"
                    if cancelled
                    else (
                        "timed_out"
                        if getattr(exc, "usage_indeterminate", False)
                        else "failed"
                    )
                ),
                response=exc,
                error_category=(
                    "completion_stopped"
                    if cancelled
                    else (
                        "provider_worker_shutdown_timeout"
                        if getattr(exc, "usage_indeterminate", False)
                        else "provider_call_failed"
                    )
                ),
            )
            terminal_accounting_event = getattr(
                exc,
                "_strict_paid_terminal_accounting_event",
                None,
            )
            if terminal_accounting_event is not None:
                terminal_accounting_event.set()
            raise
        finally:
            if (
                getattr(
                    agent,
                    "_strict_late_provider_usage_callback",
                    None,
                )
                is persist_late_usage
            ):
                agent._strict_late_provider_usage_callback = (
                    previous_late_usage_callback
                )
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
    failure_code: str = "agent_incomplete",
    failure_phase: str = "agent_loop",
    failure_retryable: bool = False,
    partial: bool = False,
    cleanup_task_id: str | None = None,
    drop_scaffolding: bool = False,
) -> dict[str, Any]:
    if cleanup_task_id is not None:
        agent._cleanup_task_resources(cleanup_task_id)
    if drop_scaffolding:
        agent._drop_trailing_empty_response_scaffolding(messages)
    if not getattr(agent, "_defer_true_moa_final_commit", False):
        agent._persist_session(messages, conversation_history)
    result: dict[str, Any] = {
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": error,
        "turn_exit_reason": (
            f"fatal({failure_phase}:{failure_code})"
        ),
        "failure": build_agent_failure(
            code=failure_code,
            phase=failure_phase,
            reason=error,
            retryable=failure_retryable,
        ),
    }
    result["final_response"] = None
    if partial:
        result["partial"] = True
    return result


def build_agent_failure(
    *,
    code: str,
    phase: str,
    reason: str,
    retryable: bool = False,
) -> dict[str, Any]:
    """Build one plaintext-safe fatal item without assistant-authored text."""

    return {
        "schema": "xiaoban.agent-failure.v1",
        "kind": "fatal",
        "code": str(code or "agent_incomplete"),
        "phase": str(phase or "agent_loop"),
        "reason": str(reason or "Agent execution failed"),
        "retryable": bool(retryable),
    }


def enforce_strict_paid_request(
    agent: Any,
    payload: dict[str, Any],
) -> int | None:
    """Apply the exact physical paid-call contract before dispatch.

    The conversation loop calls this once as a recoverable preflight and the
    provider boundary calls it again as the final fence.  Both sites therefore
    measure the same canonical JSON bytes and use the same fixed route policy.
    """

    true_moa_ledger = getattr(agent, "_true_moa_usage_ledger", None)
    if true_moa_ledger is not None:
        from xiaoban.trusted_runtime.true_moa import (
            TRUE_MOA_FINAL_PAID_CALL_POLICY,
            enforce_true_moa_final_route,
        )
        from xiaoban.trusted_runtime.paid_call_policy import (
            enforce_openai_chat_paid_call_dispatch_budget,
        )

        enforce_true_moa_final_route(
            provider=getattr(agent, "provider", ""),
            model=payload.get("model"),
        )
        return enforce_openai_chat_paid_call_dispatch_budget(
            TRUE_MOA_FINAL_PAID_CALL_POLICY,
            payload=payload,
            error_prefix="true_moa",
        )

    ledger = getattr(agent, "_paid_call_usage_ledger", None)
    if ledger is None:
        return None

    from xiaoban.trusted_runtime.paid_call_policy import (
        enforce_fixed_paid_call_route,
        enforce_openai_chat_paid_call_dispatch_budget,
        enforce_signed_mystand_policy_revision,
        resolve_signed_mystand_agent_policy,
    )

    policy = resolve_signed_mystand_agent_policy(
        getattr(agent, "_paid_call_policy_revision", "")
    )
    enforce_signed_mystand_policy_revision(
        getattr(agent, "_paid_call_policy_revision", "")
    )
    enforce_fixed_paid_call_route(
        policy,
        provider=getattr(agent, "provider", ""),
        model=payload.get("model"),
        error_code="signed_mystand_fixed_route_mismatch",
    )
    return enforce_openai_chat_paid_call_dispatch_budget(
        policy,
        payload=payload,
        error_prefix="signed_mystand_paid_call",
    )


def compact_strict_paid_history(
    agent: Any,
    messages: list[Any],
    current_turn_user_idx: int,
) -> tuple[list[Any], int, bool]:
    """Deterministically compact only history before the active user turn.

    This reuses the built-in compressor's local, redacting fallback summary and
    never calls an auxiliary model.  The current user request and its complete
    assistant/tool-result tail (including trusted ToolResult/steer sidecars)
    remain byte-for-byte intact.
    """

    if (
        not isinstance(current_turn_user_idx, int)
        or current_turn_user_idx <= 0
        or current_turn_user_idx >= len(messages)
    ):
        return messages, current_turn_user_idx, False
    current_user = messages[current_turn_user_idx]
    if not isinstance(current_user, dict) or current_user.get("role") != "user":
        return messages, current_turn_user_idx, False

    historical = list(messages[:current_turn_user_idx])
    active_tail = list(messages[current_turn_user_idx:])
    if not historical:
        return messages, current_turn_user_idx, False

    # System/developer instructions are not conversational history and must
    # survive compaction verbatim.  Only earlier user/assistant/tool turns are
    # eligible for the local redacting summary.
    prefix_end = 0
    while prefix_end < len(historical):
        item = historical[prefix_end]
        if not isinstance(item, dict) or item.get("role") not in {
            "system",
            "developer",
        }:
            break
        prefix_end += 1
    preserved_prefix = historical[:prefix_end]
    compressible_history = historical[prefix_end:]
    if not compressible_history:
        return messages, current_turn_user_idx, False

    compressor = getattr(agent, "context_compressor", None)
    summary_builder = getattr(
        compressor,
        "_build_static_fallback_summary",
        None,
    )
    if not callable(summary_builder):
        return messages, current_turn_user_idx, False
    summary = summary_builder(
        compressible_history,
        reason="strict paid request exceeded its exact input byte cap",
    )
    if not isinstance(summary, str) or not summary.strip():
        return messages, current_turn_user_idx, False

    compacted = [
        *preserved_prefix,
        {
            "role": "assistant",
            "content": summary.strip(),
        },
        *active_tail,
    ]
    return compacted, len(preserved_prefix) + 1, compacted != messages


def strict_exception_failure_result(
    agent: Any,
    messages: list[Any],
    conversation_history: list[Any],
    *,
    api_call_count: int,
    error: BaseException,
    failure_source: str,
) -> dict[str, Any]:
    """Project a strict exception to a stable, plaintext-safe fatal item."""

    from xiaoban.trusted_runtime.paid_call_policy import PaidCallPolicyError
    from xiaoban.trusted_runtime.true_moa import TrueMoACostCapError

    policy_error = isinstance(
        error,
        (PaidCallPolicyError, TrueMoACostCapError),
    )
    raw_code = (
        str(getattr(error, "code", "") or "").strip()
        if policy_error or failure_source == "request_preflight"
        else ""
    )
    if raw_code.endswith("_input_byte_cap_exceeded"):
        code = "input_payload_too_large"
        phase = "request_preflight"
        reason = "The exact provider request exceeded the 131072-byte input limit"
    elif raw_code.endswith("_output_token_cap_exceeded"):
        code = "output_token_limit_exceeded"
        phase = "request_preflight"
        reason = "The requested model output exceeded the fixed 4096-token limit"
    elif raw_code.endswith("_output_token_cap_invalid"):
        code = "output_token_limit_invalid"
        phase = "request_preflight"
        reason = "The provider request carried an invalid output-token limit"
    elif "fixed_route_mismatch" in raw_code:
        code = "provider_route_mismatch"
        phase = "request_preflight"
        reason = "The configured provider route did not match the signed request policy"
    elif raw_code.endswith("_input_payload_invalid"):
        code = "input_payload_invalid"
        phase = "request_preflight"
        reason = "The provider request could not be serialized safely"
    elif raw_code:
        code = "paid_call_policy_rejected"
        phase = "request_preflight"
        reason = "The provider request failed its signed pre-dispatch policy check"
    elif failure_source == "response_processing":
        code = "provider_response_processing_failed"
        phase = "response_processing"
        reason = "The model service responded, but Xiaoban could not safely process the response"
    else:
        code = "provider_call_failed"
        phase = "provider_call"
        reason = "The model service call failed before Xiaoban received a usable response"
    return strict_failure_result(
        agent,
        messages,
        conversation_history,
        api_call_count=api_call_count,
        error=reason,
        failure_code=code,
        failure_phase=phase,
        failure_retryable=code in {
            "provider_call_failed",
            "provider_response_processing_failed",
        },
    )


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
