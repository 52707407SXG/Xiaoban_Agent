"""Request state and synchronous orchestration for gateway Agent runs."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from gateway.platforms.api_server import (
    _content_to_visible_text,
    _true_moa_usage_summary,
)
from gateway.platforms.true_moa_runner_final import (
    TrueMoARunnerFinalMixin,
)
from gateway.platforms.true_moa_runner_preflight import (
    TrueMoARunnerPreflightMixin,
)
from gateway.platforms.true_moa_stop_projection import (
    CompletionStoppedError,
)
@dataclass
class TrueMoARunnerTraceState:
    tool_count: int = 0


@dataclass(frozen=True)
class TrueMoARunRequest:
    adapter: Any
    user_message: str
    conversation_history: List[Dict[str, str]]
    effective_system_prompt: Optional[str]
    session_id: Optional[str]
    stream_delta_callback: Any
    tool_progress_callback: Any
    tool_start_callback: Any
    tool_complete_callback: Any
    traced_tool_start: Callable[..., Any]
    traced_tool_complete: Callable[..., Any]
    agent_ref: Optional[list]
    gateway_session_key: Optional[str]
    request_headers: Any
    async_delivery: bool
    true_moa_snapshot: Any
    paid_call_usage_callback: Any
    request_user_id: str
    request_message_id: str
    request_delivery_id: str
    enabled_toolsets_override: Any
    mystand_request: bool
    durable_paid_call: bool
    memory_identity: Any
    metadata_trace: Any
    trace_state: TrueMoARunnerTraceState


class TrueMoARunWorkflow(
    TrueMoARunnerPreflightMixin,
    TrueMoARunnerFinalMixin,
):
    """One thread-confined Agent run with optional fixed true-MoA fencing."""

    def __init__(self, request: TrueMoARunRequest):
        self.request = request
        self.run_system_prompt = request.effective_system_prompt
        self.true_moa_controller = None
        self.true_moa_ledger = None
        self.agent_call_ledger = None
        self.agent_call_policy_revision = ""
        self.agent_call_policy = None
        self.agent_call_terminal_settlement_confirmed = None
        self.true_moa_final_commit_key = ""
        self.true_moa_terminal_notification = None
        self.true_moa_terminal_settlement_deadline: float | None = None
        self.agent = None
        self.advisor_bundle = None
        self.memory_hit_count = 0
        self.final_deadline_timed_out = False
        self.final_deadline_at: float | None = None
        self.final_executor_slot = None
        self.final_shutdown_grace_seconds = 0.0
        self.final_timeout_seconds = 0.0

    def run(self) -> tuple:
        from gateway.platforms.agent_call_accounting import (
            failed_normal_result,
            initialize_normal_call_ledger,
        )

        terminal_result = initialize_normal_call_ledger(self)
        if terminal_result is not None:
            return terminal_result
        try:
            terminal_result = self.prepare_run()
            if terminal_result is not None:
                return terminal_result
            return self.run_bound_agent()
        except BaseException as exc:
            if self.agent_call_ledger is not None:
                return failed_normal_result(
                    self,
                    interrupted=isinstance(
                        exc,
                        (KeyboardInterrupt, asyncio.CancelledError),
                    ),
                    error="agent preflight failed",
                )
            raise

    def run_bound_agent(self) -> tuple:
        from gateway.session_context import clear_session_vars

        request = self.request
        tokens = None
        trusted_turn = None
        trusted_turn_token = None
        deactivate_turn = None
        try:
            if request.metadata_trace is not None:
                attempt_value = request.adapter._header_value(
                    request.request_headers,
                    "X-Xiaoban-Attempt",
                )
                try:
                    attempt = max(0, int(attempt_value or "0"))
                except ValueError:
                    attempt = 0
                request.metadata_trace.safe_emit(
                    "request_started",
                    status="accepted",
                    attempt=attempt,
                    memory_enabled=bool(
                        request.memory_identity
                        and request.memory_identity[2] == "user"
                    ),
                    memory_hit_count=self.memory_hit_count,
                )
            tokens = request.adapter._bind_api_server_session(
                source="mystand" if request.mystand_request else "",
                chat_id=request.session_id or "",
                session_key=(
                    request.gateway_session_key
                    or request.session_id
                    or ""
                ),
                session_id=request.session_id or "",
                user_id=request.request_user_id,
                message_id=request.request_message_id,
                user_message=_content_to_visible_text(
                    request.user_message
                ),
                conversation_history=request.conversation_history,
                async_delivery=request.async_delivery,
            )
            if request.mystand_request:
                from xiaoban.trusted_runtime.turns import (
                    activate_turn,
                    begin_turn,
                    deactivate_turn,
                )
                from xiaoban.trusted_runtime.types import TrustedIdentity

                trusted_turn = begin_turn(
                    channel="web",
                    user_message=request.user_message,
                    identity=(
                        TrustedIdentity(
                            account_id=str(
                                request.request_user_id or ""
                            ),
                            data_scope="mystand",
                            source="server_session",
                        )
                        if request.request_user_id
                        else None
                    ),
                    request_id=(
                        request.request_delivery_id
                        if request.request_delivery_id
                        else f"mystand-req-{uuid.uuid4().hex}"
                    ),
                    message_id=str(
                        request.request_message_id or ""
                    ),
                )
                trusted_turn_token = activate_turn(trusted_turn)
        except Exception:
            if (
                trusted_turn_token is not None
                and callable(deactivate_turn)
            ):
                try:
                    deactivate_turn(trusted_turn_token)
                except Exception:
                    pass
            if tokens is not None:
                try:
                    clear_session_vars(tokens)
                except Exception:
                    pass
            if self.true_moa_ledger is not None:
                interrupted = bool(
                    self.true_moa_controller.state == "cancelled"
                    or (
                        request.agent_ref is not None
                        and len(request.agent_ref) > 1
                        and request.agent_ref[1]
                    )
                )
                if interrupted:
                    self.true_moa_controller.cancel()
                    slot_status = "cancelled"
                    wave_status = "cancelled"
                    error_category = "final_setup_stopped"
                else:
                    self.true_moa_controller.fail()
                    slot_status = "failed"
                    wave_status = "failed"
                    error_category = "final_setup_error"
                self.begin_true_moa_terminal_settlement(
                    slot_status=slot_status,
                    wave_status=wave_status,
                    error_category=error_category,
                )
                settlement_confirmed = (
                    self.true_moa_terminal_settlement_confirmed()
                )
                usage = _true_moa_usage_summary(
                    self.true_moa_ledger
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
            if self.agent_call_ledger is not None:
                from gateway.platforms.agent_call_accounting import (
                    failed_normal_result,
                )

                return failed_normal_result(
                    self,
                    interrupted=False,
                    error="agent setup failed",
                )
            raise

        try:
            if self.agent is None:
                self.agent = request.adapter._create_agent(
                    ephemeral_system_prompt=self.run_system_prompt,
                    session_id=request.session_id,
                    stream_delta_callback=(
                        request.stream_delta_callback
                    ),
                    tool_progress_callback=(
                        request.tool_progress_callback
                    ),
                    tool_start_callback=(
                        request.traced_tool_start
                        if request.mystand_request
                        else request.tool_start_callback
                    ),
                    tool_complete_callback=(
                        request.traced_tool_complete
                        if request.mystand_request
                        else request.tool_complete_callback
                    ),
                    gateway_session_key=request.gateway_session_key,
                    enabled_toolsets_override=(
                        request.enabled_toolsets_override
                    ),
                    request_user_id=(
                        request.request_user_id or None
                    ),
                    skip_memory=request.mystand_request,
                    # Durable signed deliveries use one physical dispatch per
                    # receipt and wait for the worker before terminal return.
                    # Legacy signed direct calls retain their prior contract.
                    strict_no_automatic_paid_retry=(
                        request.durable_paid_call
                    ),
                )
            elif self.true_moa_ledger is not None:
                self.agent.ephemeral_system_prompt = "\n\n".join(
                    part
                    for part in (
                        getattr(
                            self.agent,
                            "ephemeral_system_prompt",
                            None,
                        ),
                        self.advisor_bundle.guidance,
                    )
                    if isinstance(part, str) and part.strip()
                )
            if self.true_moa_ledger is not None:
                from xiaoban.trusted_runtime.true_moa import (
                    enforce_true_moa_final_route,
                )

                try:
                    enforce_true_moa_final_route(
                        provider=getattr(self.agent, "provider", ""),
                        model=getattr(self.agent, "model", ""),
                    )
                except RuntimeError:
                    self.true_moa_controller.fail()
                    self.begin_true_moa_terminal_settlement(
                        slot_status="failed",
                        wave_status="failed",
                        error_category=(
                            "final_executor_route_mismatch"
                        ),
                    )
                    settlement_confirmed = (
                        self.true_moa_terminal_settlement_confirmed()
                    )
                    usage = _true_moa_usage_summary(
                        self.true_moa_ledger
                    )
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
                self.agent._api_max_retries = 1
                self.agent._fallback_chain = []
                self.agent._fallback_index = 0
                self.agent._disable_streaming = True
                self.agent._strict_no_automatic_paid_retry = True
                self.agent._defer_true_moa_final_commit = True
                self.agent.compression_enabled = False
            from gateway.platforms.agent_call_accounting import (
                bind_paid_call_ledger,
            )

            bind_paid_call_ledger(self, self.agent)
            if request.agent_ref is not None:
                request.agent_ref[0] = self.agent
                if (
                    len(request.agent_ref) > 1
                    and request.agent_ref[1]
                ):
                    try:
                        if self.true_moa_controller is not None:
                            self.true_moa_controller.cancel()
                            self.interrupt_true_moa_agent_async(
                                "Stop requested via My Stand delivery",
                            )
                        else:
                            self.agent.interrupt(
                                "Stop requested via My Stand delivery",
                            )
                    finally:
                        raise CompletionStoppedError(
                            "request stopped before execution"
                        )
            effective_task_id = request.session_id or str(uuid.uuid4())
            if self.true_moa_controller is not None:
                self.true_moa_final_commit_key = (
                    f"gateway-final-handoff:{trusted_turn.request_id}"
                )
            return self.run_final_flow(
                trusted_turn=trusted_turn,
                effective_task_id=effective_task_id,
            )
        except CompletionStoppedError:
            if request.metadata_trace is not None:
                request.metadata_trace.safe_emit(
                    "request_failed",
                    status="failed",
                    duration_ms=request.metadata_trace.elapsed_ms(),
                    tool_count=request.trace_state.tool_count,
                    error_code="completion_stopped",
                )
            if self.true_moa_ledger is not None:
                if self.true_moa_controller.state == "running":
                    self.true_moa_controller.cancel()
                controller_state = self.true_moa_controller.state
                if controller_state == "cancelled":
                    self.begin_true_moa_terminal_settlement(
                        slot_status="cancelled",
                        wave_status="cancelled",
                        error_category="completion_stopped",
                    )
                else:
                    self.begin_true_moa_terminal_settlement(
                        slot_status="failed",
                        wave_status="failed",
                        error_category="terminal_fence",
                    )
                settlement_confirmed = (
                    self.true_moa_terminal_settlement_confirmed()
                )
                usage = _true_moa_usage_summary(
                    self.true_moa_ledger
                )
                return (
                    {
                        "final_response": "",
                        "messages": [],
                        "completed": False,
                        "failed": True,
                        "interrupted": (
                            controller_state == "cancelled"
                        ),
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
            if self.agent_call_ledger is not None:
                from gateway.platforms.agent_call_accounting import (
                    failed_normal_result,
                )

                return failed_normal_result(
                    self,
                    interrupted=True,
                    error="completion stopped",
                )
            raise
        except BaseException as exc:
            if request.metadata_trace is not None:
                request.metadata_trace.safe_emit(
                    "request_failed",
                    status="failed",
                    duration_ms=request.metadata_trace.elapsed_ms(),
                    tool_count=request.trace_state.tool_count,
                    error_code="agent_run_failed",
                )
            if self.true_moa_ledger is not None:
                controller_state = self.true_moa_controller.state
                if controller_state == "running":
                    if isinstance(
                        exc,
                        (KeyboardInterrupt, asyncio.CancelledError),
                    ):
                        self.true_moa_controller.cancel()
                    else:
                        self.true_moa_controller.fail()
                    controller_state = self.true_moa_controller.state
                if controller_state == "completed":
                    self.true_moa_ledger.set_wave_status(
                        "failed",
                        notify=False,
                    )
                    if self.true_moa_terminal_notification is None:
                        self.begin_true_moa_terminal_settlement(
                            slot_status="completed",
                            wave_status="failed",
                            error_category=None,
                        )
                elif controller_state == "cancelled":
                    self.begin_true_moa_terminal_settlement(
                        slot_status="cancelled",
                        wave_status="cancelled",
                        error_category="completion_stopped",
                    )
                elif self.final_deadline_timed_out:
                    self.begin_true_moa_terminal_settlement(
                        slot_status="timed_out",
                        wave_status="failed",
                        error_category="final_executor_timeout",
                        timeout_final=True,
                    )
                else:
                    self.begin_true_moa_terminal_settlement(
                        slot_status="failed",
                        wave_status="failed",
                        error_category="final_executor_error",
                    )
                settlement_confirmed = (
                    self.true_moa_terminal_settlement_confirmed()
                )
                usage = _true_moa_usage_summary(
                    self.true_moa_ledger
                )
                return (
                    {
                        "final_response": "",
                        "messages": [],
                        "completed": False,
                        "failed": True,
                        "interrupted": (
                            controller_state == "cancelled"
                        ),
                        "error": (
                            "true MoA final settlement failed"
                            if not settlement_confirmed
                            or controller_state == "completed"
                            else "completion stopped"
                            if controller_state == "cancelled"
                            else "true MoA final executor timed out"
                            if self.final_deadline_timed_out
                            else "true MoA final executor failed"
                        ),
                        "_mystand_request": True,
                        "_true_moa_usage": usage["true_moa"],
                    },
                    usage,
                )
            if self.agent_call_ledger is not None:
                from gateway.platforms.agent_call_accounting import (
                    failed_normal_result,
                )

                return failed_normal_result(
                    self,
                    interrupted=isinstance(
                        exc,
                        (KeyboardInterrupt, asyncio.CancelledError),
                    ),
                    error="agent run failed",
                )
            raise
        finally:
            if (
                trusted_turn_token is not None
                and callable(deactivate_turn)
            ):
                try:
                    deactivate_turn(trusted_turn_token)
                except Exception:
                    pass
            if tokens is not None:
                try:
                    clear_session_vars(tokens)
                except Exception:
                    pass


def run_agent_sync(request: TrueMoARunRequest) -> tuple:
    return TrueMoARunWorkflow(request).run()
