"""Thread-isolated normal and fixed true-MoA Agent run coordination."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from gateway.platforms.api_server import (
    CompletionStoppedError,
    _MYSTAND_STREAM_DELIVERY_ID_RE,
    _append_mystand_preexecuted_evidence,
    _build_guarded_fact_persistence_transcript,
    _build_mystand_preexecuted_prompt,
    _build_mystand_runtime_integrity_reminder,
    _content_to_visible_text,
    _finalize_mystand_egress_result,
    _install_signed_fact_persistence_guard,
    _merge_temporal_context,
    _mystand_index_has_candidates,
    _mystand_tool_result_failed,
    _required_mystand_evidence_groups,
    _resolve_mystand_initial_tool_choice,
    _resolved_mystand_egress_text,
    _run_mystand_preexecuted_evidence,
    _tool_result_looks_successful,
    _true_moa_usage_summary,
    logger,
)


class TrueMoARunnerMixin:
    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
        gateway_session_key: Optional[str] = None,
        request_headers: Any = None,
        async_delivery: bool = False,
        fact_requirement: Optional[Dict[str, Any]] = None,
        true_moa_snapshot: Any = None,
        true_moa_usage_callback=None,
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.
        """
        # Keep the historical api_server monkeypatch surface used by deterministic
        # gateway tests and downstream embedders after moving this coordinator.
        from gateway.platforms import api_server as _api_server

        _resolve_mystand_initial_tool_choice = (
            _api_server._resolve_mystand_initial_tool_choice
        )
        _run_mystand_preexecuted_evidence = (
            _api_server._run_mystand_preexecuted_evidence
        )
        loop = asyncio.get_running_loop()
        effective_system_prompt = _merge_temporal_context(
            ephemeral_system_prompt,
            headers=request_headers,
        )
        request_user_id = self._header_value(request_headers, "X-Xiaoban-User-Id")
        request_message_id = self._header_value(request_headers, "X-Xiaoban-Message-Id")
        request_delivery_id = self._header_value(request_headers, "X-Xiaoban-Delivery-Id")
        if not _MYSTAND_STREAM_DELIVERY_ID_RE.fullmatch(request_delivery_id):
            request_delivery_id = ""
        enabled_toolsets_override = self._toolsets_for_request_headers(request_headers)
        mystand_request = enabled_toolsets_override is not None
        memory_identity = self._mystand_memory_identity(request_headers) if mystand_request else None
        metadata_trace = None
        if memory_identity and memory_identity[0] and self._api_key:
            try:
                from xiaoban.observability.mystand_metadata import MystandMetadataTrace

                metadata_trace = MystandMetadataTrace(
                    secret=self._api_key,
                    site_id=memory_identity[0],
                    user_id=memory_identity[1],
                )
            except Exception:
                logger.warning("My Stand metadata trace unavailable", exc_info=False)

        tool_started_at: dict[str, float] = {}
        tool_count = 0
        evidence_followup = {
            "agent": None,
            "resource_index_required": False,
        }
        original_tool_start_callback = tool_start_callback
        original_tool_complete_callback = tool_complete_callback

        def _traced_tool_start(tool_call_id, function_name, function_args):
            nonlocal tool_count
            tool_count += 1
            if tool_call_id:
                tool_started_at[str(tool_call_id)] = time.monotonic()
            if metadata_trace is not None:
                metadata_trace.safe_emit(
                    "tool_started",
                    status="running",
                    tool_name=str(function_name or "unknown"),
                )
            if original_tool_start_callback is not None:
                original_tool_start_callback(tool_call_id, function_name, function_args)

        def _traced_tool_complete(tool_call_id, function_name, function_args, function_result):
            started = tool_started_at.pop(str(tool_call_id), None) if tool_call_id else None
            duration_ms = max(0, int((time.monotonic() - started) * 1000)) if started else 0
            tool_failed = _mystand_tool_result_failed(function_name, function_result)
            if metadata_trace is not None:
                metadata_trace.safe_emit(
                    "tool_completed",
                    status="failed" if tool_failed else "completed",
                    tool_name=str(function_name or "unknown"),
                    tool_duration_ms=duration_ms,
                    success=not tool_failed,
                )
            if original_tool_complete_callback is not None:
                original_tool_complete_callback(
                    tool_call_id,
                    function_name,
                    function_args,
                    function_result,
                )
            if (
                evidence_followup["resource_index_required"]
                and function_name == "mystand_resource_index"
                and _mystand_index_has_candidates(function_result)
            ):
                evidence_agent = evidence_followup["agent"]
                if (
                    evidence_agent is not None
                    and "mystand_authorization" in evidence_agent.valid_tool_names
                ):
                    evidence_agent._ephemeral_tool_choice = "mystand_authorization"

        def _run():
            from gateway.session_context import clear_session_vars

            run_system_prompt = effective_system_prompt
            true_moa_controller = None
            true_moa_ledger = None
            true_moa_final_commit_key = ""
            true_moa_terminal_notification = None
            true_moa_terminal_settlement_deadline: float | None = None
            agent = None
            memory_hit_count = 0
            if mystand_request and memory_identity and memory_identity[2] == "user":
                try:
                    memory_block, memory_hit_count = self._load_mystand_memory_context(
                        identity=memory_identity,
                        user_message=user_message,
                    )
                    if memory_block:
                        run_system_prompt = "\n\n".join(
                            part for part in (run_system_prompt, memory_block) if part
                        )
                except Exception:
                    logger.warning("My Stand scoped memory recall unavailable", exc_info=False)
            if mystand_request:
                integrity_reminder = _build_mystand_runtime_integrity_reminder(
                    user_message,
                    conversation_history,
                )
                if integrity_reminder:
                    run_system_prompt = "\n\n".join(
                        part
                        for part in (run_system_prompt, integrity_reminder)
                        if part
                    )
            if true_moa_snapshot is not None:
                from xiaoban.trusted_runtime.true_moa import (
                    FINAL_EXECUTOR_SLOT,
                    TRUE_MOA_ADVISOR_USAGE_DRAIN_TIMEOUT_SECONDS,
                    TRUE_MOA_FINAL_CALL_LIMIT,
                    TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS,
                    TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS,
                    TRUE_MOA_FINAL_TIMEOUT_SECONDS,
                    TrueMoACancelController,
                    TrueMoAExecutionError,
                    TrueMoAUsageLedger,
                    run_true_moa_advisors,
                )
                from xiaoban.trusted_runtime.true_moa_providers import (
                    strict_advisor_call,
                )

                true_moa_controller = TrueMoACancelController()
                true_moa_ledger = TrueMoAUsageLedger(
                    true_moa_snapshot,
                    on_change=true_moa_usage_callback,
                )

                def _begin_true_moa_terminal_settlement(
                    *,
                    slot_status: str,
                    wave_status: str,
                    error_category: str | None,
                    timeout_final: bool = False,
                ) -> None:
                    """Land the local terminal receipt before durable I/O."""

                    nonlocal true_moa_terminal_notification
                    nonlocal true_moa_terminal_settlement_deadline
                    if timeout_final:
                        true_moa_ledger.timeout_final_execution(
                            error_category=(
                                error_category or "final_executor_timeout"
                            ),
                            notify=False,
                        )
                    true_moa_ledger.finish_slot(
                        FINAL_EXECUTOR_SLOT,
                        status=slot_status,
                        usage=true_moa_ledger.final_call_usage(),
                        error_category=error_category,
                        cost_usd=(
                            getattr(
                                agent,
                                "session_estimated_cost_usd",
                                None,
                            )
                            if agent is not None
                            and getattr(
                                agent,
                                "session_cost_status",
                                "",
                            )
                            != "unavailable"
                            else None
                        ),
                        cost_status=(
                            getattr(agent, "session_cost_status", None)
                            if agent is not None
                            else None
                        ),
                        cost_source=(
                            getattr(agent, "session_cost_source", None)
                            if agent is not None
                            else None
                        ),
                        notify=False,
                    )
                    true_moa_ledger.set_wave_status(
                        wave_status,
                        notify=False,
                    )
                    if true_moa_terminal_notification is None:
                        true_moa_terminal_settlement_deadline = (
                            time.monotonic()
                            + TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS
                        )
                        true_moa_terminal_notification = (
                            true_moa_ledger.notify_change_async()
                        )

                def _true_moa_terminal_settlement_confirmed() -> bool:
                    receipt = true_moa_terminal_notification
                    if receipt is None:
                        return False
                    deadline = true_moa_terminal_settlement_deadline
                    remaining = (
                        max(0.0, deadline - time.monotonic())
                        if deadline is not None
                        else 0.0
                    )
                    receipt.wait(remaining)
                    return receipt.confirmed

                def _interrupt_true_moa_agent_async(reason: str) -> None:
                    target = agent
                    if target is None:
                        return

                    def _interrupt() -> None:
                        try:
                            target.interrupt(reason)
                        except BaseException:
                            logger.warning(
                                "True MoA final interrupt failed",
                                exc_info=False,
                            )

                    threading.Thread(
                        target=_interrupt,
                        name="xiaoban-true-moa-final-interrupt",
                        daemon=True,
                    ).start()

                if agent_ref is not None:
                    while len(agent_ref) < 3:
                        agent_ref.append(None)
                    agent_ref[2] = true_moa_controller
                    if agent_ref[1]:
                        true_moa_controller.cancel()

                # Resolve and validate the fixed acting route before either
                # advisor can spend tokens. Agent construction initializes the
                # local runtime only; it performs no provider request.
                try:
                    agent = self._create_agent(
                        ephemeral_system_prompt=run_system_prompt,
                        session_id=session_id,
                        stream_delta_callback=stream_delta_callback,
                        tool_progress_callback=tool_progress_callback,
                        tool_start_callback=(
                            _traced_tool_start
                            if mystand_request
                            else tool_start_callback
                        ),
                        tool_complete_callback=(
                            _traced_tool_complete
                            if mystand_request
                            else tool_complete_callback
                        ),
                        gateway_session_key=gateway_session_key,
                        enabled_toolsets_override=enabled_toolsets_override,
                        request_user_id=request_user_id or None,
                        skip_memory=mystand_request,
                        strict_no_automatic_paid_retry=True,
                    )
                    from xiaoban_cli.model_normalize import (
                        normalize_model_for_provider,
                    )

                    final_provider = str(
                        getattr(agent, "provider", "") or ""
                    ).lower()
                    final_model = normalize_model_for_provider(
                        str(getattr(agent, "model", "") or ""),
                        "deepseek",
                    )
                    if (
                        final_provider != FINAL_EXECUTOR_SLOT.provider
                        or final_model != FINAL_EXECUTOR_SLOT.model
                    ):
                        raise RuntimeError("fixed true MoA final route mismatch")
                except Exception:
                    true_moa_controller.fail()
                    _begin_true_moa_terminal_settlement(
                        slot_status="failed",
                        wave_status="failed",
                        error_category="final_executor_preflight_failed",
                    )
                    settlement_confirmed = (
                        _true_moa_terminal_settlement_confirmed()
                    )
                    usage = _true_moa_usage_summary(true_moa_ledger)
                    return (
                        {
                            "final_response": "",
                            "messages": [],
                            "completed": False,
                            "failed": True,
                            "error": (
                                "true MoA final executor unavailable"
                                if settlement_confirmed
                                else "true MoA final settlement failed"
                            ),
                            "_mystand_request": True,
                            "_true_moa_usage": usage["true_moa"],
                        },
                        usage,
                    )

                agent._api_max_retries = 1
                agent._fallback_chain = []
                agent._fallback_index = 0
                agent._disable_streaming = True
                agent._strict_no_automatic_paid_retry = True
                agent._true_moa_cancel_controller = true_moa_controller
                agent._defer_true_moa_final_commit = True
                agent.compression_enabled = False
                agent.max_tokens = TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS
                if agent_ref is not None:
                    agent_ref[0] = agent
                    if agent_ref[1]:
                        true_moa_controller.cancel()
                        _interrupt_true_moa_agent_async(
                            "Stop requested via My Stand delivery",
                        )
                try:
                    advisor_bundle = run_true_moa_advisors(
                        true_moa_snapshot,
                        current_question=user_message,
                        conversation_history=conversation_history,
                        strict_caller=strict_advisor_call,
                        cancel_controller=true_moa_controller,
                        usage_ledger=true_moa_ledger,
                        usage_drain_timeout_seconds=(
                            TRUE_MOA_ADVISOR_USAGE_DRAIN_TIMEOUT_SECONDS
                        ),
                    )
                except TrueMoAExecutionError as exc:
                    true_moa_ledger = exc.ledger
                    usage = _true_moa_usage_summary(true_moa_ledger)
                    interrupted = true_moa_controller.state == "cancelled"
                    settlement_failed = (
                        exc.category == "durable_settlement_failed"
                    )
                    return (
                        {
                            "final_response": "",
                            "messages": [],
                            "completed": False,
                            "failed": True,
                            "interrupted": interrupted,
                            "error": (
                                "true MoA final settlement failed"
                                if settlement_failed
                                else "true MoA request stopped"
                                if interrupted
                                else "true MoA advisor wave failed"
                            ),
                            "_mystand_request": True,
                            "_true_moa_usage": usage["true_moa"],
                        },
                        usage,
                    )
                true_moa_ledger = advisor_bundle.ledger
                agent._true_moa_usage_ledger = true_moa_ledger
                agent.max_iterations = min(
                    max(
                        1,
                        int(
                            getattr(
                                agent,
                                "max_iterations",
                                TRUE_MOA_FINAL_CALL_LIMIT,
                            )
                            or TRUE_MOA_FINAL_CALL_LIMIT
                        ),
                    ),
                    TRUE_MOA_FINAL_CALL_LIMIT,
                )
                if (
                    true_moa_controller.is_set
                    or (agent_ref is not None and agent_ref[1])
                ):
                    true_moa_controller.cancel()
                    _begin_true_moa_terminal_settlement(
                        slot_status="cancelled",
                        wave_status="cancelled",
                        error_category="completion_stopped",
                    )
                    settlement_confirmed = (
                        _true_moa_terminal_settlement_confirmed()
                    )
                    usage = _true_moa_usage_summary(true_moa_ledger)
                    return (
                        {
                            "final_response": "",
                            "messages": [],
                            "completed": False,
                            "failed": True,
                            "interrupted": True,
                            "error": (
                                "true MoA request stopped"
                                if settlement_confirmed
                                else "true MoA final settlement failed"
                            ),
                            "_mystand_request": True,
                            "_true_moa_usage": usage["true_moa"],
                        },
                        usage,
                    )
                run_system_prompt = "\n\n".join(
                    part
                    for part in (run_system_prompt, advisor_bundle.guidance)
                    if isinstance(part, str) and part.strip()
                )
            tokens = None
            trusted_turn = None
            trusted_turn_token = None
            deactivate_turn = None
            try:
                trusted_initial_tool_choice = (
                    _resolve_mystand_initial_tool_choice(
                        user_message,
                        run_system_prompt,
                        fact_requirement=fact_requirement,
                    )
                    if mystand_request
                    else ""
                )
                if metadata_trace is not None:
                    attempt_value = self._header_value(
                        request_headers,
                        "X-Xiaoban-Attempt",
                    )
                    try:
                        attempt = max(0, int(attempt_value or "0"))
                    except ValueError:
                        attempt = 0
                    metadata_trace.safe_emit(
                        "request_started",
                        status="accepted",
                        attempt=attempt,
                        memory_enabled=bool(
                            memory_identity and memory_identity[2] == "user"
                        ),
                        memory_hit_count=memory_hit_count,
                    )
                tokens = self._bind_api_server_session(
                    source="mystand" if mystand_request else "",
                    chat_id=session_id or "",
                    session_key=gateway_session_key or session_id or "",
                    session_id=session_id or "",
                    user_id=request_user_id,
                    message_id=request_message_id,
                    user_message=_content_to_visible_text(user_message),
                    conversation_history=conversation_history,
                    async_delivery=async_delivery,
                )
                if mystand_request:
                    from xiaoban.trusted_runtime.turns import (
                        activate_turn,
                        begin_turn,
                        deactivate_turn,
                    )
                    from xiaoban.trusted_runtime.types import TrustedIdentity

                    trusted_turn = begin_turn(
                        channel="web",
                        user_message=user_message,
                        conversation_history=conversation_history,
                        identity=(
                            TrustedIdentity(
                                account_id=str(request_user_id or ""),
                                data_scope="mystand",
                                source="server_session",
                            )
                            if request_user_id
                            else None
                        ),
                        # Trusted deliveries bind the turn to the server-verified
                        # delivery id; ad-hoc My Stand calls keep a random id.
                        request_id=(
                            request_delivery_id
                            if request_delivery_id
                            else f"mystand-req-{uuid.uuid4().hex}"
                        ),
                        message_id=str(request_message_id or ""),
                        evidence_required=bool(
                            trusted_initial_tool_choice or fact_requirement
                        ),
                        fact_requirement=fact_requirement,
                    )
                    trusted_turn_token = activate_turn(trusted_turn)
            except Exception:
                if trusted_turn_token is not None and callable(deactivate_turn):
                    try:
                        deactivate_turn(trusted_turn_token)
                    except Exception:
                        pass
                if tokens is not None:
                    try:
                        clear_session_vars(tokens)
                    except Exception:
                        pass
                if true_moa_ledger is not None:
                    interrupted = bool(
                        true_moa_controller.state == "cancelled"
                        or (
                            agent_ref is not None
                            and len(agent_ref) > 1
                            and agent_ref[1]
                        )
                    )
                    if interrupted:
                        true_moa_controller.cancel()
                        slot_status = "cancelled"
                        wave_status = "cancelled"
                        error_category = "final_setup_stopped"
                    else:
                        true_moa_controller.fail()
                        slot_status = "failed"
                        wave_status = "failed"
                        error_category = "final_setup_error"
                    _begin_true_moa_terminal_settlement(
                        slot_status=slot_status,
                        wave_status=wave_status,
                        error_category=error_category,
                    )
                    settlement_confirmed = (
                        _true_moa_terminal_settlement_confirmed()
                    )
                    usage = _true_moa_usage_summary(true_moa_ledger)
                    return (
                        {
                            "final_response": "",
                            "messages": [],
                            "completed": False,
                            "failed": True,
                            "interrupted": interrupted,
                            "error": (
                                "true MoA final settlement failed"
                                if not settlement_confirmed
                                else "true MoA request stopped"
                                if interrupted
                                else "true MoA final executor setup failed"
                            ),
                            "_mystand_request": True,
                            "_true_moa_usage": usage["true_moa"],
                        },
                        usage,
                    )
                raise
            try:
                if agent is None:
                    agent = self._create_agent(
                        ephemeral_system_prompt=run_system_prompt,
                        session_id=session_id,
                        stream_delta_callback=stream_delta_callback,
                        tool_progress_callback=tool_progress_callback,
                        tool_start_callback=(
                            _traced_tool_start
                            if mystand_request
                            else tool_start_callback
                        ),
                        tool_complete_callback=(
                            _traced_tool_complete
                            if mystand_request
                            else tool_complete_callback
                        ),
                        gateway_session_key=gateway_session_key,
                        enabled_toolsets_override=enabled_toolsets_override,
                        request_user_id=request_user_id or None,
                        skip_memory=mystand_request,
                    )
                elif true_moa_ledger is not None:
                    # The agent was preflighted before advisor fan-out. Append
                    # the bounded untrusted guidance to that exact instance so
                    # route validation and execution cannot observe two configs.
                    agent.ephemeral_system_prompt = "\n\n".join(
                        part
                        for part in (
                            getattr(agent, "ephemeral_system_prompt", None),
                            advisor_bundle.guidance,
                        )
                        if isinstance(part, str) and part.strip()
                    )
                if true_moa_ledger is not None:
                    from xiaoban_cli.model_normalize import (
                        normalize_model_for_provider,
                    )

                    final_provider = str(getattr(agent, "provider", "") or "").lower()
                    final_model = normalize_model_for_provider(
                        str(getattr(agent, "model", "") or ""),
                        "deepseek",
                    )
                    if (
                        final_provider != FINAL_EXECUTOR_SLOT.provider
                        or final_model != FINAL_EXECUTOR_SLOT.model
                    ):
                        true_moa_controller.fail()
                        _begin_true_moa_terminal_settlement(
                            slot_status="failed",
                            wave_status="failed",
                            error_category="final_executor_route_mismatch",
                        )
                        settlement_confirmed = (
                            _true_moa_terminal_settlement_confirmed()
                        )
                        usage = _true_moa_usage_summary(true_moa_ledger)
                        return (
                            {
                                "final_response": "",
                                "messages": [],
                                "completed": False,
                                "failed": True,
                                "error": (
                                    "true MoA final executor unavailable"
                                    if settlement_confirmed
                                    else "true MoA final settlement failed"
                                ),
                                "_mystand_request": True,
                                "_true_moa_usage": usage["true_moa"],
                            },
                            usage,
                        )
                    # The fixed preset permits purposeful tool iterations but
                    # never an automatic provider retry, continuation,
                    # compression call, or fallback route.
                    agent._api_max_retries = 1
                    agent._fallback_chain = []
                    agent._fallback_index = 0
                    agent._disable_streaming = True
                    agent._strict_no_automatic_paid_retry = True
                    agent._defer_true_moa_final_commit = True
                    agent.compression_enabled = False
                if agent_ref is not None:
                    agent_ref[0] = agent
                    if len(agent_ref) > 1 and agent_ref[1]:
                        try:
                            if true_moa_controller is not None:
                                true_moa_controller.cancel()
                                _interrupt_true_moa_agent_async(
                                    "Stop requested via My Stand delivery",
                                )
                            else:
                                agent.interrupt(
                                    "Stop requested via My Stand delivery",
                                )
                        finally:
                            raise CompletionStoppedError("request stopped before execution")
                effective_task_id = session_id or str(uuid.uuid4())
                if true_moa_controller is not None:
                    true_moa_final_commit_key = (
                        f"gateway-final-handoff:{trusted_turn.request_id}"
                    )

                def _execute_final_stage() -> dict[str, Any]:
                    if (
                        true_moa_controller is not None
                        and true_moa_controller.state != "running"
                    ):
                        raise CompletionStoppedError(
                            "true MoA final stage stopped before tools",
                        )
                    initial_tool_choice = trusted_initial_tool_choice
                    if (
                        not initial_tool_choice
                        or initial_tool_choice not in agent.valid_tool_names
                    ):
                        initial_tool_choice = ""
                    evidence_followup["agent"] = agent
                    evidence_followup["resource_index_required"] = (
                        initial_tool_choice == "mystand_resource_index"
                    )
                    preexecuted_evidence: List[Dict[str, Any]] = []
                    if initial_tool_choice:
                        preexecuted_evidence = _run_mystand_preexecuted_evidence(
                            initial_tool_choice,
                            user_message=user_message,
                            system_prompt=run_system_prompt,
                            tool_start_callback=_traced_tool_start,
                            tool_complete_callback=_traced_tool_complete,
                            trusted_turn=trusted_turn,
                            fact_requirement=fact_requirement,
                            terminal_controller=true_moa_controller,
                        )
                        evidence_prompt = _build_mystand_preexecuted_prompt(
                            preexecuted_evidence
                        )
                        if evidence_prompt:
                            agent.ephemeral_system_prompt = "\n\n".join(
                                part
                                for part in (
                                    agent.ephemeral_system_prompt,
                                    evidence_prompt,
                                )
                                if isinstance(part, str) and part.strip()
                            )
                        # The harness already executed the authoritative read.
                        # Never ask the provider to repeat it through a hint.
                        agent._ephemeral_tool_choice = ""
                    if (
                        true_moa_controller is not None
                        and true_moa_controller.state != "running"
                    ):
                        raise CompletionStoppedError(
                            "true MoA final stage stopped after evidence",
                        )
                    if fact_requirement is not None:
                        # The signed plan has already executed deterministically.
                        # The provider only writes prose; it cannot repeat or
                        # branch into another business tool call.
                        agent.tools = []
                        agent.valid_tool_names = set()
                        _install_signed_fact_persistence_guard(
                            agent,
                            trusted_turn,
                        )
                    execution_history = (
                        []
                        if initial_tool_choice
                        and any(
                            _tool_result_looks_successful(item.get("content"))
                            for item in preexecuted_evidence
                        )
                        else conversation_history
                    )
                    if (
                        true_moa_controller is not None
                        and true_moa_controller.state != "running"
                    ):
                        raise CompletionStoppedError(
                            "true MoA final stage stopped before executor",
                        )
                    stage_result = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=execution_history,
                        task_id=effective_task_id,
                    )
                    if true_moa_ledger is None:
                        return {
                            "initial_tool_choice": initial_tool_choice,
                            "preexecuted_evidence": preexecuted_evidence,
                            "result": stage_result,
                        }
                    result = (
                        dict(stage_result)
                        if isinstance(stage_result, dict)
                        else {}
                    )
                    _append_mystand_preexecuted_evidence(
                        result,
                        preexecuted_evidence,
                    )
                    guarded_turn = copy.deepcopy(trusted_turn)
                    result["_mystand_request"] = mystand_request
                    if mystand_request:
                        result["_mystand_user_id"] = str(
                            request_user_id or "",
                        )
                        result["_mystand_request_id"] = str(
                            trusted_turn.request_id,
                        )
                        result["_mystand_message_id"] = str(
                            request_message_id or "",
                        )
                        # Completion/egress guards mutate terminal WorkTurn
                        # fields.  Project on an isolated copy so a guard that
                        # returns after timeout cannot mutate the live turn.
                        result["_trusted_turn"] = guarded_turn
                        result["_mystand_evidence_required"] = bool(
                            trusted_initial_tool_choice or fact_requirement
                        )
                        if fact_requirement is not None:
                            result["_mystand_fact_requirement"] = (
                                fact_requirement
                            )
                    if initial_tool_choice:
                        result["_mystand_required_evidence_groups"] = [
                            sorted(group)
                            for group in _required_mystand_evidence_groups(
                                initial_tool_choice
                            )
                        ]
                    # Full My Stand egress projection, including
                    # check_mystand_final_answer and its digest, is part of the
                    # same watchdog worker as the provider and trusted tools.
                    # The parent may commit only this sealed projection.
                    visible_text = _finalize_mystand_egress_result(
                        result,
                        user_message=user_message,
                        conversation_history=conversation_history,
                    )
                    (
                        safe_persistence_history,
                        safe_persistence_messages,
                    ) = _build_guarded_fact_persistence_transcript(
                        conversation_history,
                        user_message=user_message,
                        guarded_final_response=visible_text,
                    )
                    # Raw provider/tool bytes remain worker-local diagnostics.
                    result["messages"] = []
                    return {
                        "initial_tool_choice": initial_tool_choice,
                        "preexecuted_evidence": preexecuted_evidence,
                        "result": result,
                        "safe_persistence_history": (
                            safe_persistence_history
                        ),
                        "safe_persistence_messages": (
                            safe_persistence_messages
                        ),
                        "guarded_turn_projection": {
                            "state": guarded_turn.state,
                            "states": list(guarded_turn.states),
                            "terminal_reason": guarded_turn.terminal_reason,
                        },
                    }

                final_deadline_timed_out = False
                final_deadline_at: float | None = None
                final_stage_error: BaseException | None = None

                def _fence_expired_final_deadline() -> None:
                    nonlocal final_deadline_timed_out
                    if (
                        final_deadline_timed_out
                        or true_moa_ledger is None
                        or true_moa_controller is None
                        or final_deadline_at is None
                        or time.monotonic() < final_deadline_at
                    ):
                        return
                    if true_moa_controller.fail():
                        final_deadline_timed_out = True
                        _begin_true_moa_terminal_settlement(
                            slot_status="timed_out",
                            wave_status="failed",
                            error_category="final_executor_timeout",
                            timeout_final=True,
                        )

                if true_moa_ledger is not None:
                    # The final slot begins before deterministic evidence or
                    # any model-selected trusted tool. Its copied Context keeps
                    # request identity, DataScope and the active WorkTurn while
                    # the caller thread enforces a hard bounded return.
                    final_deadline_at = (
                        time.monotonic() + TRUE_MOA_FINAL_TIMEOUT_SECONDS
                    )
                    if not true_moa_controller.reserve_final_commit(
                        true_moa_final_commit_key,
                        deadline_monotonic=final_deadline_at,
                    ):
                        raise CompletionStoppedError(
                            "true MoA stopped before final handoff reservation",
                        )
                    # This is a local in-memory mutation only.  Any durable
                    # running notification happens in the worker; the terminal
                    # snapshot is always dispatched asynchronously.
                    true_moa_ledger.start_slot(
                        FINAL_EXECUTOR_SLOT,
                        notify=False,
                    )
                    if not true_moa_ledger.confirm_change(
                        TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS,
                    ):
                        true_moa_controller.fail()
                        true_moa_ledger.timeout_final_execution(
                            error_category="final_executor_timeout",
                            notify=False,
                        )
                        raise RuntimeError(
                            "true MoA durable final-slot reservation failed"
                        )
                    final_stage_done = threading.Event()
                    final_stage_box: dict[str, Any] = {}
                    final_stage_context = contextvars.copy_context()

                    def _run_final_stage_worker() -> None:
                        try:
                            final_stage_box["payload"] = _execute_final_stage()
                        except BaseException as exc:
                            final_stage_box["error"] = exc
                        finally:
                            final_stage_done.set()

                    final_stage_thread = threading.Thread(
                        target=lambda: final_stage_context.run(
                            _run_final_stage_worker,
                        ),
                        name="xiaoban-true-moa-final-stage",
                        daemon=True,
                    )
                    final_stage_thread.start()
                    while not final_stage_done.is_set():
                        controller_state = true_moa_controller.state
                        if controller_state in {"cancelled", "failed"}:
                            break
                        remaining = final_deadline_at - time.monotonic()
                        if remaining <= 0:
                            _fence_expired_final_deadline()
                            break
                        final_stage_done.wait(timeout=min(0.05, remaining))

                    if (
                        not final_stage_done.is_set()
                        and true_moa_controller.state in {"cancelled", "failed"}
                    ):
                        if (
                            true_moa_controller.state == "cancelled"
                            and true_moa_terminal_notification is None
                        ):
                            _begin_true_moa_terminal_settlement(
                                slot_status="cancelled",
                                wave_status="cancelled",
                                error_category="completion_stopped",
                            )
                        _interrupt_true_moa_agent_async(
                            (
                                "True MoA final executor deadline exceeded"
                                if final_deadline_timed_out
                                else "True MoA final stage stopped"
                            ),
                        )
                        final_stage_done.wait(
                            timeout=TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS,
                        )

                    if final_stage_done.is_set():
                        final_stage_error = final_stage_box.get("error")
                        final_stage_payload = final_stage_box.get("payload")
                        if not isinstance(final_stage_payload, dict):
                            final_stage_payload = {}
                    else:
                        # The daemon worker retains only its request-local
                        # copied Context. The failed/cancelled controller fences
                        # every later provider, tool-result and final commit.
                        final_stage_payload = {}
                else:
                    final_stage_payload = _execute_final_stage()

                # The worker may finish on the deadline edge before the parent
                # polling loop observes expiry. Re-check on the hand-off side
                # so a just-late payload cannot commit as completed.
                _fence_expired_final_deadline()
                if true_moa_ledger is None:
                    initial_tool_choice = str(
                        final_stage_payload.get("initial_tool_choice") or "",
                    )
                    preexecuted_evidence = final_stage_payload.get(
                        "preexecuted_evidence",
                    )
                    if not isinstance(preexecuted_evidence, list):
                        preexecuted_evidence = []
                    result = final_stage_payload.get("result")
                    result = (
                        dict(result)
                        if isinstance(result, dict)
                        else {}
                    )
                    _append_mystand_preexecuted_evidence(
                        result,
                        preexecuted_evidence,
                    )
                    if fact_requirement is not None:
                        from xiaoban.trusted_runtime.completion_guard import (
                            check_completion,
                        )

                        guarded_fact = check_completion(
                            result.get("final_response", ""),
                            trusted_turn,
                        )
                        result["final_response"] = guarded_fact.text
                        result["messages"] = []
                    result["_mystand_request"] = mystand_request
                    if mystand_request:
                        result["_mystand_user_id"] = str(
                            request_user_id or "",
                        )
                        result["_mystand_request_id"] = str(
                            trusted_turn.request_id,
                        )
                        result["_mystand_message_id"] = str(
                            request_message_id or "",
                        )
                        result["_trusted_turn"] = trusted_turn
                        result["_mystand_evidence_required"] = bool(
                            trusted_initial_tool_choice or fact_requirement
                        )
                        if fact_requirement is not None:
                            result["_mystand_fact_requirement"] = (
                                fact_requirement
                            )
                    if initial_tool_choice:
                        result["_mystand_required_evidence_groups"] = [
                            sorted(group)
                            for group in _required_mystand_evidence_groups(
                                initial_tool_choice
                            )
                        ]
                    usage = {
                        "input_tokens": (
                            getattr(agent, "session_prompt_tokens", 0) or 0
                        ),
                        "output_tokens": (
                            getattr(agent, "session_completion_tokens", 0)
                            or 0
                        ),
                        "total_tokens": (
                            getattr(agent, "session_total_tokens", 0) or 0
                        ),
                    }
                    effective_session_id = getattr(
                        agent,
                        "session_id",
                        session_id,
                    )
                    if (
                        isinstance(effective_session_id, str)
                        and effective_session_id
                    ):
                        result["session_id"] = effective_session_id
                    if metadata_trace is not None:
                        failed_result = bool(
                            result.get("interrupted")
                            or result.get("partial")
                            or result.get("failed")
                            or not result.get("completed", True)
                        )
                        metadata_trace.safe_emit(
                            (
                                "request_failed"
                                if failed_result
                                else "request_completed"
                            ),
                            status=(
                                "failed"
                                if failed_result
                                else "completed"
                            ),
                            duration_ms=metadata_trace.elapsed_ms(),
                            tool_count=tool_count,
                            **(
                                {
                                    "error_code": (
                                        "completion_stopped"
                                        if result.get("interrupted")
                                        else "output_truncated"
                                        if result.get("partial")
                                        else "agent_error"
                                    )
                                }
                                if failed_result
                                else {}
                            ),
                        )
                    return result, usage
                result = final_stage_payload.get("result")
                result = dict(result) if isinstance(result, dict) else {}
                deferred_persistence_messages = (
                    final_stage_payload.get("safe_persistence_messages")
                    if isinstance(
                        final_stage_payload.get(
                            "safe_persistence_messages",
                        ),
                        list,
                    )
                    else []
                )
                deferred_persistence_history = (
                    final_stage_payload.get("safe_persistence_history")
                    if isinstance(
                        final_stage_payload.get(
                            "safe_persistence_history",
                        ),
                        list,
                    )
                    else []
                )
                guarded_turn_projection = final_stage_payload.get(
                    "guarded_turn_projection",
                )
                request_stopped = bool(
                    not final_deadline_timed_out
                    and (
                        (
                            agent_ref is not None
                            and len(agent_ref) > 1
                            and agent_ref[1]
                        )
                        or (
                            true_moa_controller is not None
                            and true_moa_controller.state == "cancelled"
                        )
                    )
                )
                if final_deadline_timed_out:
                    result.update({
                        "final_response": "",
                        "completed": False,
                        "failed": True,
                        "interrupted": False,
                        "error": "true MoA final executor timed out",
                    })
                    result["messages"] = []
                elif request_stopped or isinstance(
                    final_stage_error,
                    (CompletionStoppedError, KeyboardInterrupt),
                ) or (
                    isinstance(final_stage_error, asyncio.CancelledError)
                ):
                    true_moa_controller.cancel()
                    result.update({
                        "final_response": "",
                        "completed": False,
                        "failed": True,
                        "interrupted": True,
                        "error": "completion stopped",
                    })
                    result["messages"] = []
                    request_stopped = True
                elif isinstance(final_stage_error, BaseException):
                    true_moa_controller.fail()
                    result.update({
                        "final_response": "",
                        "completed": False,
                        "failed": True,
                        "interrupted": False,
                        "error": "true MoA final executor failed",
                    })
                    result["messages"] = []
                elif (
                    result.get("_mystand_egress_finalized") is not True
                    or not isinstance(
                        result.get("_mystand_egress_output_digest"),
                        str,
                    )
                    or not isinstance(guarded_turn_projection, dict)
                    or not isinstance(
                        guarded_turn_projection.get("state"),
                        str,
                    )
                    or not isinstance(
                        guarded_turn_projection.get("states"),
                        list,
                    )
                    or not isinstance(
                        guarded_turn_projection.get("terminal_reason"),
                        str,
                    )
                ):
                    true_moa_controller.fail()
                    result.update({
                        "final_response": "",
                        "completed": False,
                        "failed": True,
                        "interrupted": False,
                        "error": "true MoA final executor failed",
                    })
                    result["messages"] = []
                else:
                    # Hash-only verification; every potentially blocking guard
                    # already ran inside the watchdog worker.
                    _resolved_mystand_egress_text(
                        result,
                        user_message=user_message,
                        conversation_history=conversation_history,
                    )
                _fence_expired_final_deadline()
                if final_deadline_timed_out:
                    result.update({
                        "final_response": "",
                        "completed": False,
                        "failed": True,
                        "interrupted": False,
                        "error": "true MoA final executor timed out",
                    })
                    result["messages"] = []
                result["_mystand_request"] = mystand_request
                usage = {
                    "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                    "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                    "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                }
                deferred_persistence_ready = False
                if true_moa_ledger is not None:
                    result_interrupted = bool(result.get("interrupted"))
                    result_partial = bool(result.get("partial"))
                    result_failed = bool(result.get("failed"))
                    result_completed = bool(result.get("completed", False))
                    if final_deadline_timed_out:
                        final_status = "timed_out"
                        final_error = "final_executor_timeout"
                        wave_status = "failed"
                    elif (
                        true_moa_controller.state == "cancelled"
                        or request_stopped
                        or result_interrupted
                    ):
                        true_moa_controller.cancel()
                        final_status = "cancelled"
                        final_error = "completion_stopped"
                        wave_status = "cancelled"
                    elif result_partial or result_failed or not result_completed:
                        true_moa_controller.fail()
                        final_status = "failed"
                        final_error = (
                            "output_truncated"
                            if result_partial
                            else "final_executor_failed"
                        )
                        wave_status = "failed"
                    else:
                        _fence_expired_final_deadline()
                        if (
                            not final_deadline_timed_out
                            and true_moa_controller.try_commit_final(
                                true_moa_final_commit_key,
                            )
                        ):
                            final_status = "completed"
                            final_error = None
                            wave_status = "completed"
                            deferred_persistence_ready = True
                        else:
                            # The controller checks its reserved monotonic
                            # deadline under the same lock as completion. If
                            # the edge crossed between the parent-side check
                            # and commit, settle it as timeout now.
                            _fence_expired_final_deadline()
                            if final_deadline_timed_out:
                                result.update({
                                    "final_response": "",
                                    "completed": False,
                                    "failed": True,
                                    "interrupted": False,
                                    "error": (
                                        "true MoA final executor timed out"
                                    ),
                                })
                                result["messages"] = []
                                final_status = "timed_out"
                                final_error = "final_executor_timeout"
                                wave_status = "failed"
                            elif true_moa_controller.state == "cancelled":
                                # A cancellation that wins the terminal-state
                                # lock suppresses even a provider result that
                                # arrived at the same time.
                                result.update({
                                    "final_response": "",
                                    "completed": False,
                                    "failed": True,
                                    "interrupted": True,
                                    "error": "completion stopped",
                                })
                                final_status = "cancelled"
                                final_error = "terminal_fence"
                                wave_status = "cancelled"
                            else:
                                result.update({
                                    "final_response": "",
                                    "completed": False,
                                    "failed": True,
                                    "interrupted": False,
                                    "error": (
                                        "true MoA final executor failed"
                                    ),
                                })
                                result["messages"] = []
                                final_status = "failed"
                                final_error = "terminal_fence"
                                wave_status = "failed"
                    _begin_true_moa_terminal_settlement(
                        slot_status=final_status,
                        wave_status=wave_status,
                        error_category=final_error,
                        timeout_final=final_deadline_timed_out,
                    )
                    settlement_confirmed = (
                        _true_moa_terminal_settlement_confirmed()
                    )
                    if not settlement_confirmed:
                        deferred_persistence_ready = False
                        if true_moa_controller.state == "completed":
                            true_moa_ledger.set_wave_status(
                                "failed",
                                notify=False,
                            )
                            true_moa_ledger.notify_change_async()
                        result.update({
                            "final_response": "",
                            "messages": [],
                            "completed": False,
                            "failed": True,
                            "interrupted": (
                                true_moa_controller.state == "cancelled"
                            ),
                            "error": "true MoA final settlement failed",
                        })
                    if deferred_persistence_ready and settlement_confirmed:
                        # The isolated guard projection becomes authoritative
                        # only after completion won and the terminal durable
                        # ledger write was confirmed.
                        trusted_turn.state = guarded_turn_projection["state"]
                        trusted_turn.states = list(
                            guarded_turn_projection["states"],
                        )
                        trusted_turn.terminal_reason = (
                            guarded_turn_projection["terminal_reason"]
                        )
                        result["_trusted_turn"] = trusted_turn
                        # Durable final-slot and wave callbacks must succeed
                        # before any transcript or trajectory is written.
                        # The completed outcome is sealed by the caller after
                        # egress finalization; this ordering prevents a failed
                        # ledger settlement from leaving private session state
                        # behind without a deliverable result.
                        from agent.turn_finalizer import (
                            persist_deferred_true_moa_turn,
                        )

                        deferred_persistence_errors = (
                            persist_deferred_true_moa_turn(
                                agent,
                                messages=deferred_persistence_messages,
                                conversation_history=(
                                    deferred_persistence_history
                                ),
                                user_message=user_message,
                                completed=True,
                            )
                        )
                        if deferred_persistence_errors:
                            result.setdefault(
                                "cleanup_errors",
                                [],
                            ).extend(deferred_persistence_errors)
                    elif not deferred_persistence_ready:
                        result.pop("_trusted_turn", None)
                    usage = _true_moa_usage_summary(true_moa_ledger)
                    result["_true_moa_usage"] = usage["true_moa"]
                # Include the effective session ID in the result so callers
                # (e.g. X-Xiaoban-Session-Id header) can track compression-
                # triggered session rotations. (#16938)
                _eff_sid = getattr(agent, "session_id", session_id)
                if isinstance(_eff_sid, str) and _eff_sid:
                    result["session_id"] = _eff_sid
                if metadata_trace is not None:
                    result_interrupted = isinstance(result, dict) and bool(result.get("interrupted"))
                    result_partial = isinstance(result, dict) and bool(result.get("partial"))
                    result_failed = isinstance(result, dict) and bool(result.get("failed"))
                    result_completed = isinstance(result, dict) and bool(result.get("completed", True))
                    if result_interrupted or result_partial or result_failed or not result_completed:
                        metadata_trace.safe_emit(
                            "request_failed",
                            status="failed",
                            duration_ms=metadata_trace.elapsed_ms(),
                            tool_count=tool_count,
                            error_code=(
                                "completion_stopped"
                                if result_interrupted
                                else "output_truncated"
                                if result_partial
                                else "agent_error"
                            ),
                        )
                    else:
                        completed_fields: dict[str, Any] = {
                            "status": "completed",
                            "duration_ms": metadata_trace.elapsed_ms(),
                            "tool_count": tool_count,
                            "memory_enabled": bool(memory_identity and memory_identity[2] == "user"),
                            "memory_hit_count": memory_hit_count,
                            "input_tokens": usage["input_tokens"],
                            "output_tokens": usage["output_tokens"],
                            "total_tokens": usage["total_tokens"],
                        }
                        if getattr(agent, "provider", ""):
                            completed_fields["provider"] = agent.provider
                        if getattr(agent, "model", ""):
                            completed_fields["model"] = agent.model
                        metadata_trace.safe_emit("request_completed", **completed_fields)
                return result, usage
            except CompletionStoppedError:
                if metadata_trace is not None:
                    metadata_trace.safe_emit(
                        "request_failed",
                        status="failed",
                        duration_ms=metadata_trace.elapsed_ms(),
                        tool_count=tool_count,
                        error_code="completion_stopped",
                    )
                if true_moa_ledger is not None:
                    if true_moa_controller.state == "running":
                        true_moa_controller.cancel()
                    controller_state = true_moa_controller.state
                    if controller_state == "cancelled":
                        _begin_true_moa_terminal_settlement(
                            slot_status="cancelled",
                            wave_status="cancelled",
                            error_category="completion_stopped",
                        )
                    else:
                        _begin_true_moa_terminal_settlement(
                            slot_status="failed",
                            wave_status="failed",
                            error_category="terminal_fence",
                        )
                    settlement_confirmed = (
                        _true_moa_terminal_settlement_confirmed()
                    )
                    usage = _true_moa_usage_summary(true_moa_ledger)
                    return (
                        {
                            "final_response": "",
                            "messages": [],
                            "completed": False,
                            "failed": True,
                            "interrupted": controller_state == "cancelled",
                            "error": (
                                "completion stopped"
                                if settlement_confirmed
                                and controller_state == "cancelled"
                                else "true MoA final settlement failed"
                                if not settlement_confirmed
                                else "true MoA final executor failed"
                            ),
                            "_mystand_request": True,
                            "_true_moa_usage": usage["true_moa"],
                        },
                        usage,
                    )
                raise
            except BaseException as exc:
                if metadata_trace is not None:
                    metadata_trace.safe_emit(
                        "request_failed",
                        status="failed",
                        duration_ms=metadata_trace.elapsed_ms(),
                        tool_count=tool_count,
                        error_code="agent_run_failed",
                    )
                if true_moa_ledger is not None:
                    controller_state = true_moa_controller.state
                    if controller_state == "running":
                        if isinstance(
                            exc,
                            (KeyboardInterrupt, asyncio.CancelledError),
                        ):
                            true_moa_controller.cancel()
                        else:
                            true_moa_controller.fail()
                        controller_state = true_moa_controller.state
                    if controller_state == "completed":
                        true_moa_ledger.set_wave_status(
                            "failed",
                            notify=False,
                        )
                        if true_moa_terminal_notification is None:
                            _begin_true_moa_terminal_settlement(
                                slot_status="completed",
                                wave_status="failed",
                                error_category=None,
                            )
                    elif controller_state == "cancelled":
                        _begin_true_moa_terminal_settlement(
                            slot_status="cancelled",
                            wave_status="cancelled",
                            error_category="completion_stopped",
                        )
                    elif final_deadline_timed_out:
                        _begin_true_moa_terminal_settlement(
                            slot_status="timed_out",
                            wave_status="failed",
                            error_category="final_executor_timeout",
                            timeout_final=True,
                        )
                    else:
                        _begin_true_moa_terminal_settlement(
                            slot_status="failed",
                            wave_status="failed",
                            error_category="final_executor_error",
                        )
                    settlement_confirmed = (
                        _true_moa_terminal_settlement_confirmed()
                    )
                    usage = _true_moa_usage_summary(true_moa_ledger)
                    return (
                        {
                            "final_response": "",
                            "messages": [],
                            "completed": False,
                            "failed": True,
                            "interrupted": controller_state == "cancelled",
                            "error": (
                                "true MoA final settlement failed"
                                if not settlement_confirmed
                                or controller_state == "completed"
                                else "completion stopped"
                                if controller_state == "cancelled"
                                else "true MoA final executor timed out"
                                if final_deadline_timed_out
                                else "true MoA final executor failed"
                            ),
                            "_mystand_request": True,
                            "_true_moa_usage": usage["true_moa"],
                        },
                        usage,
                    )
                raise
            finally:
                if trusted_turn_token is not None and callable(deactivate_turn):
                    try:
                        deactivate_turn(trusted_turn_token)
                    except Exception:
                        pass
                if tokens is not None:
                    try:
                        clear_session_vars(tokens)
                    except Exception:
                        pass

        self._inflight_agent_runs += 1
        try:
            return await loop.run_in_executor(None, _run)
        finally:
            self._inflight_agent_runs -= 1
