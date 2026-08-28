"""Strict true-MoA policy adapters for the mature conversation loop."""

from __future__ import annotations

import copy
import json
import uuid
from types import SimpleNamespace
from typing import Any, Callable

from agent.paid_call_accounting import (
    finish_paid_provider_call,
    record_strict_terminal_usage,
)


def strict_mode(agent: Any) -> bool:
    return bool(getattr(agent, "_strict_no_automatic_paid_retry", False))


def signed_normal_mode(agent: Any) -> bool:
    """Return whether this is the normal, fully-accounted My Stand loop."""

    return bool(
        strict_mode(agent)
        and getattr(agent, "_paid_call_usage_ledger", None) is not None
        and getattr(agent, "_true_moa_usage_ledger", None) is None
    )


def strict_cancel_controller(agent: Any) -> Any:
    """Return the request-local stop fence for either strict execution mode."""

    return (
        getattr(agent, "_true_moa_cancel_controller", None)
        or getattr(agent, "_paid_call_cancel_controller", None)
    )


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
    controller = strict_cancel_controller(agent)

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
            interrupted = InterruptedError(
                "True MoA cancelled before response consumption"
            )
            interrupted._strict_session_usage_recorded = True
            raise interrupted
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
    api_call_count += int(
        getattr(agent, "_strict_compaction_call_count", 0) or 0
    )
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
            enforce_openai_responses_paid_call_dispatch_budget,
        )

        enforce_true_moa_final_route(
            provider=getattr(agent, "provider", ""),
            model=payload.get("model"),
        )
        if getattr(agent, "api_mode", "") == "codex_responses":
            return enforce_openai_responses_paid_call_dispatch_budget(
                TRUE_MOA_FINAL_PAID_CALL_POLICY,
                payload=payload,
                configured_output_max_tokens=getattr(
                    agent,
                    "max_tokens",
                    None,
                ),
                error_prefix="true_moa",
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
        enforce_openai_responses_paid_call_dispatch_budget,
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
    if getattr(agent, "api_mode", "") == "codex_responses":
        return enforce_openai_responses_paid_call_dispatch_budget(
            policy,
            payload=payload,
            configured_output_max_tokens=getattr(
                agent,
                "max_tokens",
                None,
            ),
            error_prefix="signed_mystand_paid_call",
        )
    return enforce_openai_chat_paid_call_dispatch_budget(
        policy,
        payload=payload,
        error_prefix="signed_mystand_paid_call",
    )


def summarize_signed_normal_context(
    agent: Any,
    turns_to_summarize: list[dict[str, Any]],
    focus_topic: str | None = None,
) -> str | None:
    """Create one paid, same-model checkpoint from the model-visible transcript."""

    if not (
        signed_normal_mode(agent)
        or getattr(agent, "_true_moa_usage_ledger", None) is not None
    ) or not turns_to_summarize:
        return None
    compressor = getattr(agent, "context_compressor", None)
    serialize = getattr(compressor, "_serialize_for_summary", None)
    if not callable(serialize):
        compressor._last_summary_error = (
            "same-model compaction serializer unavailable"
        )
        return None

    # Project canonical ToolResults exactly as a normal sampling request does;
    # private tool content must never enter the compaction prompt.
    projected_turns = agent._sanitize_api_messages(
        copy.deepcopy(turns_to_summarize)
    )
    source = serialize(projected_turns)
    previous_summary = str(
        getattr(compressor, "_previous_summary", "") or ""
    ).strip()
    focus = str(focus_topic or "").strip()
    prompt = (
        "Create a compact same-turn context checkpoint for the agent that will "
        "continue the task immediately after compaction. Do not answer the user "
        "and do not call tools. Preserve concrete completed actions, actual tool "
        "results, IDs, counts, blockers, decisions, and remaining work. Clearly "
        "separate completed work from work still required. Never include API "
        "keys, tokens, passwords, credentials, or connection strings. The latest "
        "user request remains outside this checkpoint and remains authoritative."
        "\n\n"
        + (
            f"PREVIOUS CHECKPOINT:\n{previous_summary}\n\n"
            if previous_summary
            else ""
        )
        + (f"FOCUS:\n{focus}\n\n" if focus else "")
        + f"CONTEXT TO COMPACT:\n{source}"
    )
    compact_messages = [
        {
            "role": "system",
            "content": (
                "You compact agent context. Return only the checkpoint text in "
                "the user's language; do not solve the task."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    api_kwargs = agent._build_api_kwargs(compact_messages)
    api_kwargs.pop("tools", None)
    api_kwargs.pop("tool_choice", None)
    api_kwargs.pop("parallel_tool_calls", None)
    turn_id = str(getattr(agent, "_current_turn_id", "") or "compact")
    api_request_id = f"{turn_id}:compact:{uuid.uuid4().hex}"
    physical_calls_used = int(
        getattr(agent, "_api_call_count", 0) or 0
    ) + int(getattr(agent, "_strict_compaction_call_count", 0) or 0)
    if physical_calls_used + 2 > int(getattr(agent, "max_iterations", 0) or 0):
        compressor._last_summary_error = (
            "same-model context compaction has no iteration continuation slot"
        )
        return None
    ledger = (
        getattr(agent, "_paid_call_usage_ledger", None)
        or getattr(agent, "_true_moa_usage_ledger", None)
    )
    ledger_before = ledger.to_dict() if ledger is not None else {"calls": []}
    calls_before = list(ledger_before.get("calls") or [])
    max_calls = int(getattr(ledger, "max_calls", 0) or 0)
    if max_calls and len(calls_before) + 2 > max_calls:
        compressor._last_summary_error = (
            "same-model context compaction has no paid continuation slot"
        )
        return None
    call_ids_before = {
        str(item.get("callId") or "")
        for item in calls_before
        if isinstance(item, dict)
    }
    terminal_usage_recorded = False

    try:
        response = execute_llm_request(
            agent,
            api_kwargs,
            agent._interruptible_api_call,
            strict=True,
            original_request=dict(api_kwargs),
            middleware_trace=[],
            task_id="context-compaction",
            turn_id=turn_id,
            api_request_id=api_request_id,
            api_call_count=int(getattr(agent, "_api_call_count", 0) or 0) + 1,
        )
        if not claim_response_consumption(agent, api_request_id):
            record_strict_terminal_usage(agent, response)
            terminal_usage_recorded = getattr(response, "usage", None) is not None
            raise InterruptedError(
                "same-model context compaction cancelled before consumption"
            )
        record_strict_terminal_usage(agent, response)
        terminal_usage_recorded = getattr(response, "usage", None) is not None
        transport = agent._get_transport()
        normalize_kwargs = {}
        if getattr(agent, "api_mode", "") == "anthropic_messages":
            normalize_kwargs["strip_tool_prefix"] = bool(
                getattr(agent, "_is_anthropic_oauth", False)
            )
        normalized = transport.normalize_response(response, **normalize_kwargs)
        raw_finish_reason: Any = ...
        choices = getattr(response, "choices", None)
        if isinstance(choices, (list, tuple)) and choices:
            raw_finish_reason = getattr(choices[0], "finish_reason", None)
        elif hasattr(response, "stop_reason"):
            raw_finish_reason = getattr(response, "stop_reason", None)
        finish_reason = str(
            (
                getattr(normalized, "finish_reason", "")
                if raw_finish_reason is ...
                else raw_finish_reason
            )
            or ""
        ).lower()
        if (
            getattr(normalized, "tool_calls", None)
            or finish_reason not in {"stop", "end_turn"}
        ):
            raise RuntimeError("compaction response was incomplete")
        summary = agent._strip_think_blocks(
            str(getattr(normalized, "content", "") or "")
        ).strip()
        summary = "\n".join(
            line
            for line in summary.splitlines()
            if not line.lstrip().startswith(
                (
                    "XIAOBAN_RUNTIME_CHECKPOINT_JSON:",
                    "[RUNTIME-OWNED COMPACTION CHECKPOINT",
                )
            )
        ).strip()
        if not summary:
            raise RuntimeError("compaction response was empty")
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if getattr(exc, "_strict_session_usage_recorded", False):
            terminal_usage_recorded = True
        if not terminal_usage_recorded:
            usage_source = exc if getattr(exc, "usage", None) is not None else None
            if usage_source is None and ledger is not None:
                calls_after_error = ledger.to_dict().get("calls") or []
                usage_call = next(
                    (
                        item
                        for item in reversed(calls_after_error)
                        if isinstance(item, dict)
                        and str(item.get("callId") or "")
                        not in call_ids_before
                        and item.get("usageStatus") in {"partial", "reported"}
                    ),
                    None,
                )
                if usage_call is not None:
                    usage = {
                        "prompt_tokens": usage_call.get("inputTokens"),
                        "completion_tokens": usage_call.get("outputTokens"),
                        "total_tokens": usage_call.get("totalTokens"),
                    }
                    cached_tokens = usage_call.get("cachedInputTokens")
                    if isinstance(cached_tokens, int):
                        usage["prompt_tokens_details"] = {
                            "cached_tokens": cached_tokens,
                        }
                    usage_source = SimpleNamespace(usage=usage)
            if usage_source is not None:
                record_strict_terminal_usage(agent, usage_source)
        compressor._last_summary_error = (
            "same-model context compaction cancelled"
            if isinstance(exc, InterruptedError)
            else "same-model context compaction failed"
        )
        return None
    finally:
        if ledger is not None:
            calls_after = ledger.to_dict().get("calls") or []
            dispatched_compactions = sum(
                1
                for item in calls_after
                if (
                    isinstance(item, dict)
                    and str(item.get("callId") or "") not in call_ids_before
                    and item.get("status") not in {
                        "reserved",
                        "not_dispatched",
                    }
                )
            )
            agent._strict_compaction_call_count = int(
                getattr(agent, "_strict_compaction_call_count", 0) or 0
            ) + dispatched_compactions

    from agent.context_compressor import redact_sensitive_text

    summary = redact_sensitive_text(summary)
    with_prefix = getattr(compressor, "_with_summary_prefix", None)
    if callable(with_prefix):
        summary = with_prefix(summary)
    compressor._previous_summary = compressor._strip_summary_prefix(summary)
    compressor._last_summary_error = None
    compressor._last_summary_fallback_used = False
    compressor._last_summary_dropped_count = 0
    compressor._last_compress_aborted = False
    return summary


def signed_normal_runtime_checkpoint(
    agent: Any,
    messages: list[Any],
) -> str:
    """Preserve unresolved side effects and verified writes outside LLM prose."""

    if not (
        signed_normal_mode(agent)
        or getattr(agent, "_true_moa_usage_ledger", None) is not None
    ):
        return ""
    from agent.context_compressor import redact_sensitive_text
    from agent.tool_result_classification import (
        RUNTIME_CHECKPOINT_INTERNAL_KEY,
        _verified_write_receipt,
        _verified_write_receipt_digest,
        project_runtime_checkpoint_for_model,
        project_tool_result_for_model,
    )

    existing_facts: list[Any] = []
    trusted_steers: list[str] = []

    for item in messages:
        if not isinstance(item, dict):
            continue
        prior = item.get(RUNTIME_CHECKPOINT_INTERNAL_KEY)
        if (
            item.get("role") in {"assistant", "user"}
            and item.get("_compressed_summary") is True
            and isinstance(prior, dict)
            and prior.get("schema")
            == "xiaoban.runtime-compaction-checkpoint.v1"
        ):
            if isinstance(prior.get("facts"), list):
                existing_facts.extend(prior["facts"][:64])
            if isinstance(prior.get("trustedSteers"), list):
                trusted_steers.extend(
                    str(value)
                    for value in prior["trustedSteers"][:32]
                    if isinstance(value, str)
                )
        steer = item.get("_xiaoban_trusted_steer")
        if isinstance(steer, (list, tuple)):
            trusted_steers.extend(
                str(value)
                for value in steer
                if isinstance(value, str) and value
            )
        sidecar = item.get("_xiaoban_tool_result")
        if isinstance(sidecar, dict):
            sidecar_steers = sidecar.get("trustedSteers")
            if isinstance(sidecar_steers, list):
                trusted_steers.extend(
                    str(value)
                    for value in sidecar_steers[:32]
                    if isinstance(value, str) and value
                )

    fresh_protected: list[Any] = []
    for item in messages:
        if not isinstance(item, dict) or item.get("role") != "tool":
            continue
        sidecar = item.get("_xiaoban_tool_result")
        if not isinstance(sidecar, dict):
            continue
        call_id = str(item.get("tool_call_id") or "")
        verified_receipt = _verified_write_receipt(
            sidecar.get("verifiedWriteReceipt")
        )
        verified_write = bool(
            sidecar.get("toolName") == "mystand_authorization_write"
            and sidecar.get("dispatchState") == "dispatched"
            and sidecar.get("outcome") == "success"
            and isinstance(verified_receipt, dict)
            and sidecar.get("verifiedWriteReceiptDigest")
            == _verified_write_receipt_digest(verified_receipt)
        )
        unresolved = bool(
            sidecar.get("dispatchState") == "dispatched"
            and sidecar.get("outcome") == "unknown"
        )
        if not (verified_write or unresolved):
            continue
        if verified_write:
            model_projection = {
                key: sidecar[key]
                for key in (
                    "schema",
                    "requestId",
                    "turnId",
                    "callId",
                    "toolName",
                    "dispatchState",
                    "outcome",
                    "retrySafe",
                )
                if key in sidecar
            }
            model_projection["modelResult"] = verified_receipt
        else:
            model_projection = project_tool_result_for_model(
                item.get("content"),
                sidecar,
            )
            if isinstance(model_projection, str):
                try:
                    model_projection = json.loads(model_projection)
                except (TypeError, ValueError):
                    pass
        if model_projection is not None:
            fresh_protected.append(model_projection)
    protected = [*fresh_protected, *existing_facts]
    deduped_facts: list[Any] = []
    seen_facts: set[str] = set()
    for fact in protected:
        key = json.dumps(
            fact,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if key in seen_facts:
            continue
        seen_facts.add(key)
        deduped_facts.append(fact)
        if len(deduped_facts) >= 64:
            break
    deduped_steers: list[str] = []
    seen_steers: set[str] = set()
    for steer in trusted_steers:
        safe_steer = redact_sensitive_text(steer)[:8_192]
        if not safe_steer or safe_steer in seen_steers:
            continue
        seen_steers.add(safe_steer)
        deduped_steers.append(safe_steer)
        if len(deduped_steers) >= 32:
            break
    if not deduped_facts and not deduped_steers:
        return ""
    checkpoint = {
        "schema": "xiaoban.runtime-compaction-checkpoint.v1",
        "facts": deduped_facts,
        "trustedSteers": deduped_steers,
    }
    return project_runtime_checkpoint_for_model(checkpoint) or ""


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
        reason = "The exact provider request exceeded the fixed input limit"
    elif raw_code.endswith("_output_token_cap_exceeded"):
        code = "output_token_limit_exceeded"
        phase = "request_preflight"
        reason = "The requested model output exceeded the configured per-call limit"
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
    elif failure_source == "context_compaction":
        code = "context_compaction_failed"
        phase = "context_compaction"
        reason = (
            "The model context could not be compacted safely for continuation"
        )
    elif failure_source == "context_overflow_usage_unavailable":
        code = "provider_usage_unavailable"
        phase = "provider_call"
        reason = (
            "The model request failed with a context overflow and no trusted "
            "usage receipt was available for a paid continuation"
        )
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
    controller = strict_cancel_controller(agent)
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
    controller = strict_cancel_controller(agent)
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
