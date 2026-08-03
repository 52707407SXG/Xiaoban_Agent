"""Memory preparation, fixed-route preflight, and advisor-wave startup."""

from __future__ import annotations

import threading
import time
from typing import Any

from gateway.platforms.api_server import (
    _true_moa_usage_summary,
    logger,
)


class TrueMoARunnerPreflightMixin:
    """Preflight helpers shared by the synchronous runner workflow."""

    request: Any
    run_system_prompt: str | None
    agent: Any
    advisor_bundle: Any
    true_moa_controller: Any
    true_moa_ledger: Any
    true_moa_terminal_notification: Any
    true_moa_terminal_settlement_deadline: float | None
    memory_hit_count: int
    final_executor_slot: Any
    final_shutdown_grace_seconds: float

    def begin_true_moa_terminal_settlement(
        self,
        *,
        slot_status: str,
        wave_status: str,
        error_category: str | None,
        timeout_final: bool = False,
    ) -> None:
        """Land the local terminal receipt before durable I/O."""

        ledger = self.true_moa_ledger
        if timeout_final:
            ledger.timeout_final_execution(
                error_category=(
                    error_category or "final_executor_timeout"
                ),
                notify=False,
            )
        ledger.finish_slot(
            self.final_executor_slot,
            status=slot_status,
            usage=ledger.final_call_usage(),
            error_category=error_category,
            cost_usd=(
                getattr(
                    self.agent,
                    "session_estimated_cost_usd",
                    None,
                )
                if self.agent is not None
                and getattr(
                    self.agent,
                    "session_cost_status",
                    "",
                )
                != "unavailable"
                else None
            ),
            cost_status=(
                getattr(self.agent, "session_cost_status", None)
                if self.agent is not None
                else None
            ),
            cost_source=(
                getattr(self.agent, "session_cost_source", None)
                if self.agent is not None
                else None
            ),
            notify=False,
        )
        ledger.set_wave_status(
            wave_status,
            notify=False,
        )
        if self.true_moa_terminal_notification is None:
            self.true_moa_terminal_settlement_deadline = (
                time.monotonic()
                + self.final_shutdown_grace_seconds
            )
            self.true_moa_terminal_notification = (
                ledger.notify_change_async()
            )

    def true_moa_terminal_settlement_confirmed(self) -> bool:
        receipt = self.true_moa_terminal_notification
        if receipt is None:
            return False
        deadline = self.true_moa_terminal_settlement_deadline
        remaining = (
            max(0.0, deadline - time.monotonic())
            if deadline is not None
            else 0.0
        )
        receipt.wait(remaining)
        return receipt.confirmed

    def interrupt_true_moa_agent_async(self, reason: str) -> None:
        target = self.agent
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

    def prepare_run(self) -> tuple | None:
        request = self.request
        run_system_prompt = request.effective_system_prompt
        self.memory_hit_count = 0
        if (
            request.mystand_request
            and request.memory_identity
            and request.memory_identity[2] == "user"
        ):
            try:
                memory_block, self.memory_hit_count = (
                    request.adapter._load_mystand_memory_context(
                        identity=request.memory_identity,
                        user_message=request.user_message,
                    )
                )
                if memory_block:
                    run_system_prompt = "\n\n".join(
                        part
                        for part in (run_system_prompt, memory_block)
                        if part
                    )
            except Exception:
                logger.warning(
                    "My Stand scoped memory recall unavailable",
                    exc_info=False,
                )
        self.run_system_prompt = run_system_prompt
        if request.true_moa_snapshot is None:
            return None

        # Resolve the runtime on each paid run. Tests and downstream embedders
        # patch these module attributes before dispatch.
        from xiaoban.trusted_runtime import true_moa as true_moa_runtime
        from xiaoban.trusted_runtime import (
            true_moa_providers as true_moa_provider_runtime,
        )

        advisor_usage_drain_timeout_seconds = (
            true_moa_runtime.TRUE_MOA_ADVISOR_USAGE_DRAIN_TIMEOUT_SECONDS
        )
        final_call_limit = (
            true_moa_runtime.TRUE_MOA_FINAL_CALL_LIMIT
        )
        final_output_max_tokens = (
            true_moa_runtime.TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS
        )
        run_true_moa_advisors = (
            true_moa_runtime.run_true_moa_advisors
        )
        true_moa_execution_error = (
            true_moa_runtime.TrueMoAExecutionError
        )
        strict_advisor_call = (
            true_moa_provider_runtime.strict_advisor_call
        )
        self.final_executor_slot = (
            true_moa_runtime.FINAL_EXECUTOR_SLOT
        )
        self.final_shutdown_grace_seconds = (
            true_moa_runtime.TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS
        )
        self.final_timeout_seconds = (
            true_moa_runtime.TRUE_MOA_FINAL_TIMEOUT_SECONDS
        )
        self.true_moa_controller = (
            true_moa_runtime.TrueMoACancelController()
        )
        self.true_moa_ledger = true_moa_runtime.TrueMoAUsageLedger(
            request.true_moa_snapshot,
            on_change=request.paid_call_usage_callback,
        )
        controller = self.true_moa_controller

        if request.agent_ref is not None:
            while len(request.agent_ref) < 3:
                request.agent_ref.append(None)
            request.agent_ref[2] = controller
            if request.agent_ref[1]:
                controller.cancel()

        # Resolve and validate the fixed acting route before either advisor
        # can spend tokens. Agent construction itself performs no provider call.
        try:
            self.agent = request.adapter._create_agent(
                ephemeral_system_prompt=run_system_prompt,
                session_id=request.session_id,
                stream_delta_callback=request.stream_delta_callback,
                tool_progress_callback=request.tool_progress_callback,
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
                enabled_toolsets_override=request.enabled_toolsets_override,
                request_user_id=request.request_user_id or None,
                skip_memory=request.mystand_request,
                strict_no_automatic_paid_retry=True,
            )
            true_moa_runtime.enforce_true_moa_final_route(
                provider=getattr(self.agent, "provider", ""),
                model=getattr(self.agent, "model", ""),
            )
        except Exception:
            controller.fail()
            self.begin_true_moa_terminal_settlement(
                slot_status="failed",
                wave_status="failed",
                error_category="final_executor_preflight_failed",
            )
            settlement_confirmed = (
                self.true_moa_terminal_settlement_confirmed()
            )
            usage = _true_moa_usage_summary(self.true_moa_ledger)
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

        agent = self.agent
        agent._api_max_retries = 1
        agent._fallback_chain = []
        agent._fallback_index = 0
        agent._disable_streaming = True
        agent._strict_no_automatic_paid_retry = True
        agent._true_moa_cancel_controller = controller
        agent._true_moa_usage_ledger = self.true_moa_ledger
        agent._defer_true_moa_final_commit = True
        agent.compression_enabled = False
        agent.max_tokens = final_output_max_tokens
        if request.agent_ref is not None:
            request.agent_ref[0] = agent
            if request.agent_ref[1]:
                controller.cancel()
                self.interrupt_true_moa_agent_async(
                    "Stop requested via My Stand delivery",
                )
        try:
            self.advisor_bundle = run_true_moa_advisors(
                request.true_moa_snapshot,
                current_question=request.user_message,
                conversation_history=request.conversation_history,
                strict_caller=strict_advisor_call,
                cancel_controller=controller,
                usage_ledger=self.true_moa_ledger,
                usage_drain_timeout_seconds=(
                    advisor_usage_drain_timeout_seconds
                ),
            )
        except true_moa_execution_error as exc:
            self.true_moa_ledger = exc.ledger
            usage = _true_moa_usage_summary(self.true_moa_ledger)
            interrupted = controller.state == "cancelled"
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
        self.true_moa_ledger = self.advisor_bundle.ledger
        agent.max_iterations = min(
            max(
                1,
                int(
                    getattr(
                        agent,
                        "max_iterations",
                        final_call_limit,
                    )
                    or final_call_limit
                ),
            ),
            final_call_limit,
        )
        if (
            controller.is_set
            or (
                request.agent_ref is not None
                and request.agent_ref[1]
            )
        ):
            controller.cancel()
            self.begin_true_moa_terminal_settlement(
                slot_status="cancelled",
                wave_status="cancelled",
                error_category="completion_stopped",
            )
            settlement_confirmed = (
                self.true_moa_terminal_settlement_confirmed()
            )
            usage = _true_moa_usage_summary(self.true_moa_ledger)
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
        self.run_system_prompt = "\n\n".join(
            part
            for part in (
                self.run_system_prompt,
                self.advisor_bundle.guidance,
            )
            if isinstance(part, str) and part.strip()
        )
        return None
