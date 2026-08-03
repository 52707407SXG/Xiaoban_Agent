"""Final-stage execution, hard deadline, egress projection, and settlement."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from typing import Any

from gateway.platforms.api_server import (
    _finalize_mystand_egress_result,
    _resolved_mystand_egress_text,
    _true_moa_usage_summary,
)
from gateway.platforms.mystand_egress_seal import (
    is_mystand_egress_sealed,
)
from gateway.platforms.true_moa_stop_projection import (
    CompletionStoppedError,
)


class TrueMoARunnerFinalMixin:
    """Run and settle one normal or fixed true-MoA final stage."""

    request: Any
    run_system_prompt: str | None
    agent: Any
    true_moa_controller: Any
    true_moa_ledger: Any
    true_moa_final_commit_key: str
    true_moa_terminal_notification: Any
    final_deadline_timed_out: bool
    final_deadline_at: float | None
    final_executor_slot: Any
    final_shutdown_grace_seconds: float
    final_timeout_seconds: float

    def execute_final_stage(
        self,
        *,
        trusted_turn: Any,
        effective_task_id: str,
    ) -> dict[str, Any]:
        request = self.request
        controller = self.true_moa_controller
        if controller is not None and controller.state != "running":
            raise CompletionStoppedError(
                "true MoA final stage stopped before tools",
            )
        if controller is not None and controller.state != "running":
            raise CompletionStoppedError(
                "true MoA final stage stopped before executor",
            )
        stage_result = self.agent.run_conversation(
            user_message=request.user_message,
            conversation_history=request.conversation_history,
            task_id=effective_task_id,
        )
        if self.true_moa_ledger is None:
            return {"result": stage_result}
        result = (
            dict(stage_result)
            if isinstance(stage_result, dict)
            else {}
        )
        deferred_persistence_messages = list(result.get("messages") or [])
        deferred_persistence_history = list(
            request.conversation_history or []
        )
        result["_mystand_request"] = request.mystand_request
        if request.mystand_request:
            result["_mystand_user_id"] = str(
                request.request_user_id or "",
            )
            result["_mystand_request_id"] = str(
                trusted_turn.request_id,
            )
            result["_mystand_message_id"] = str(
                request.request_message_id or "",
            )
        _finalize_mystand_egress_result(
            result,
            user_message=request.user_message,
            conversation_history=request.conversation_history,
        )
        # K5 keeps the real transcript private until durable settlement wins.
        result["messages"] = []
        return {
            "result": result,
            "deferred_persistence_history": deferred_persistence_history,
            "deferred_persistence_messages": deferred_persistence_messages,
            "turn_projection": {
                "state": trusted_turn.state,
                "states": list(trusted_turn.states),
                "terminal_reason": trusted_turn.terminal_reason,
            },
        }

    def fence_expired_final_deadline(self) -> None:
        if (
            self.final_deadline_timed_out
            or self.true_moa_ledger is None
            or self.true_moa_controller is None
            or self.final_deadline_at is None
            or time.monotonic() < self.final_deadline_at
        ):
            return
        if self.true_moa_controller.fail():
            self.final_deadline_timed_out = True
            self.begin_true_moa_terminal_settlement(
                slot_status="timed_out",
                wave_status="failed",
                error_category="final_executor_timeout",
                timeout_final=True,
            )

    def run_final_flow(
        self,
        *,
        trusted_turn: Any,
        effective_task_id: str,
    ) -> tuple:
        request = self.request
        controller = self.true_moa_controller
        ledger = self.true_moa_ledger
        self.final_deadline_timed_out = False
        self.final_deadline_at = None
        final_stage_error: BaseException | None = None

        def _execute_final_stage() -> dict[str, Any]:
            return self.execute_final_stage(
                trusted_turn=trusted_turn,
                effective_task_id=effective_task_id,
            )

        if ledger is not None:
            # The final slot and commit reservation begin before deterministic
            # evidence or any model-selected trusted tool.
            self.final_deadline_at = (
                time.monotonic() + self.final_timeout_seconds
            )
            if not controller.reserve_final_commit(
                self.true_moa_final_commit_key,
                deadline_monotonic=self.final_deadline_at,
            ):
                raise CompletionStoppedError(
                    "true MoA stopped before final handoff reservation",
                )
            ledger.start_slot(self.final_executor_slot, notify=False)
            if not ledger.confirm_change(
                self.final_shutdown_grace_seconds,
            ):
                controller.fail()
                # K5: this is a durable handoff failure before the Provider
                # worker exists, not an execution timeout. Keep the final call
                # set empty and give the slot its own terminal category.
                ledger.finish_slot(
                    self.final_executor_slot,
                    status="failed",
                    error_category="durable_final_slot_confirmation_failed",
                    notify=False,
                )
                ledger.set_wave_status("failed", notify=False)
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
                controller_state = controller.state
                if controller_state in {"cancelled", "failed"}:
                    break
                remaining = self.final_deadline_at - time.monotonic()
                if remaining <= 0:
                    self.fence_expired_final_deadline()
                    break
                final_stage_done.wait(timeout=min(0.05, remaining))

            if (
                not final_stage_done.is_set()
                and controller.state in {"cancelled", "failed"}
            ):
                if (
                    controller.state == "cancelled"
                    and self.true_moa_terminal_notification is None
                ):
                    self.begin_true_moa_terminal_settlement(
                        slot_status="cancelled",
                        wave_status="cancelled",
                        error_category="completion_stopped",
                    )
                self.interrupt_true_moa_agent_async(
                    (
                        "True MoA final executor deadline exceeded"
                        if self.final_deadline_timed_out
                        else "True MoA final stage stopped"
                    ),
                )
                final_stage_done.wait(
                    timeout=self.final_shutdown_grace_seconds,
                )

            if final_stage_done.is_set():
                final_stage_error = final_stage_box.get("error")
                final_stage_payload = final_stage_box.get("payload")
                if not isinstance(final_stage_payload, dict):
                    final_stage_payload = {}
            else:
                # The daemon retains only its request-local copied Context.
                final_stage_payload = {}
        else:
            final_stage_payload = _execute_final_stage()

        # Re-check on hand-off so a deadline-edge payload cannot commit.
        self.fence_expired_final_deadline()
        if ledger is None:
            return self.project_normal_result(
                final_stage_payload=final_stage_payload,
                trusted_turn=trusted_turn,
            )
        return self.settle_true_moa_result(
            final_stage_payload=final_stage_payload,
            final_stage_error=final_stage_error,
            trusted_turn=trusted_turn,
        )

    def project_normal_result(
        self,
        *,
        final_stage_payload: dict[str, Any],
        trusted_turn: Any,
    ) -> tuple:
        request = self.request
        result = final_stage_payload.get("result")
        result = dict(result) if isinstance(result, dict) else {}
        result["_mystand_request"] = request.mystand_request
        if request.mystand_request:
            result["_mystand_user_id"] = str(
                request.request_user_id or "",
            )
            result["_mystand_request_id"] = str(
                trusted_turn.request_id,
            )
            result["_mystand_message_id"] = str(
                request.request_message_id or "",
            )
            result["_trusted_turn"] = trusted_turn
        if request.mystand_request:
            _finalize_mystand_egress_result(
                result,
                user_message=request.user_message,
                conversation_history=request.conversation_history,
            )
        usage = {
            "input_tokens": (
                getattr(self.agent, "session_prompt_tokens", 0) or 0
            ),
            "output_tokens": (
                getattr(self.agent, "session_completion_tokens", 0) or 0
            ),
            "total_tokens": (
                getattr(self.agent, "session_total_tokens", 0) or 0
            ),
        }
        effective_session_id = getattr(
            self.agent,
            "session_id",
            request.session_id,
        )
        if (
            isinstance(effective_session_id, str)
            and effective_session_id
        ):
            result["session_id"] = effective_session_id
        if request.metadata_trace is not None:
            failed_result = bool(
                result.get("interrupted")
                or result.get("partial")
                or result.get("failed")
                or not result.get("completed", True)
            )
            request.metadata_trace.safe_emit(
                (
                    "request_failed"
                    if failed_result
                    else "request_completed"
                ),
                status="failed" if failed_result else "completed",
                duration_ms=request.metadata_trace.elapsed_ms(),
                tool_count=request.trace_state.tool_count,
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
        from gateway.platforms.agent_call_accounting import (
            finalize_normal_call_usage,
        )

        return finalize_normal_call_usage(self, result, usage)

    def settle_true_moa_result(
        self,
        *,
        final_stage_payload: dict[str, Any],
        final_stage_error: BaseException | None,
        trusted_turn: Any,
    ) -> tuple:
        request = self.request
        controller = self.true_moa_controller
        ledger = self.true_moa_ledger
        result = final_stage_payload.get("result")
        result = dict(result) if isinstance(result, dict) else {}
        deferred_persistence_messages = (
            final_stage_payload.get("deferred_persistence_messages")
            if isinstance(
                final_stage_payload.get("deferred_persistence_messages"),
                list,
            )
            else []
        )
        deferred_persistence_history = (
            final_stage_payload.get("deferred_persistence_history")
            if isinstance(
                final_stage_payload.get("deferred_persistence_history"),
                list,
            )
            else []
        )
        turn_projection = final_stage_payload.get(
            "turn_projection",
        )
        request_stopped = bool(
            not self.final_deadline_timed_out
            and (
                (
                    request.agent_ref is not None
                    and len(request.agent_ref) > 1
                    and request.agent_ref[1]
                )
                or controller.state == "cancelled"
            )
        )
        if self.final_deadline_timed_out:
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
        ) or isinstance(final_stage_error, asyncio.CancelledError):
            controller.cancel()
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
            controller.fail()
            result.update({
                "final_response": "",
                "completed": False,
                "failed": True,
                "interrupted": False,
                "error": "true MoA final executor failed",
            })
            result["messages"] = []
        elif (
            not is_mystand_egress_sealed(result)
            or not isinstance(
                result.get("_mystand_egress_output_digest"),
                str,
            )
            or not isinstance(turn_projection, dict)
            or not isinstance(
                turn_projection.get("state"),
                str,
            )
            or not isinstance(
                turn_projection.get("states"),
                list,
            )
            or not isinstance(
                turn_projection.get("terminal_reason"),
                str,
            )
        ):
            controller.fail()
            result.update({
                "final_response": "",
                "completed": False,
                "failed": True,
                "interrupted": False,
                "error": "true MoA final executor failed",
            })
            result["messages"] = []
        else:
            _resolved_mystand_egress_text(
                result,
                user_message=request.user_message,
                conversation_history=request.conversation_history,
            )
        self.fence_expired_final_deadline()
        if self.final_deadline_timed_out:
            result.update({
                "final_response": "",
                "completed": False,
                "failed": True,
                "interrupted": False,
                "error": "true MoA final executor timed out",
            })
            result["messages"] = []
        result["_mystand_request"] = request.mystand_request
        usage = {
            "input_tokens": (
                getattr(self.agent, "session_prompt_tokens", 0) or 0
            ),
            "output_tokens": (
                getattr(self.agent, "session_completion_tokens", 0) or 0
            ),
            "total_tokens": (
                getattr(self.agent, "session_total_tokens", 0) or 0
            ),
        }
        deferred_persistence_ready = False
        result_interrupted = bool(result.get("interrupted"))
        result_partial = bool(result.get("partial"))
        result_failed = bool(result.get("failed"))
        result_completed = bool(result.get("completed", False))
        if self.final_deadline_timed_out:
            final_status = "timed_out"
            final_error = "final_executor_timeout"
            wave_status = "failed"
        elif (
            controller.state == "cancelled"
            or request_stopped
            or result_interrupted
        ):
            controller.cancel()
            final_status = "cancelled"
            final_error = "completion_stopped"
            wave_status = "cancelled"
        elif result_partial or result_failed or not result_completed:
            controller.fail()
            final_status = "failed"
            final_error = (
                "output_truncated"
                if result_partial
                else "final_executor_failed"
            )
            wave_status = "failed"
        else:
            self.fence_expired_final_deadline()
            if (
                not self.final_deadline_timed_out
                and controller.try_commit_final(
                    self.true_moa_final_commit_key,
                )
            ):
                final_status = "completed"
                final_error = None
                wave_status = "completed"
                deferred_persistence_ready = True
            else:
                self.fence_expired_final_deadline()
                if self.final_deadline_timed_out:
                    result.update({
                        "final_response": "",
                        "completed": False,
                        "failed": True,
                        "interrupted": False,
                        "error": "true MoA final executor timed out",
                    })
                    result["messages"] = []
                    final_status = "timed_out"
                    final_error = "final_executor_timeout"
                    wave_status = "failed"
                elif controller.state == "cancelled":
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
                        "error": "true MoA final executor failed",
                    })
                    result["messages"] = []
                    final_status = "failed"
                    final_error = "terminal_fence"
                    wave_status = "failed"
        self.begin_true_moa_terminal_settlement(
            slot_status=final_status,
            wave_status=wave_status,
            error_category=final_error,
            timeout_final=self.final_deadline_timed_out,
        )
        settlement_confirmed = (
            self.true_moa_terminal_settlement_confirmed()
        )
        if not settlement_confirmed:
            deferred_persistence_ready = False
            if controller.state == "completed":
                ledger.set_wave_status("failed", notify=False)
                ledger.notify_change_async()
            result.update({
                "final_response": "",
                "messages": [],
                "completed": False,
                "failed": True,
                "interrupted": controller.state == "cancelled",
                "error": "true MoA final settlement failed",
            })
        if deferred_persistence_ready and settlement_confirmed:
            trusted_turn.state = turn_projection["state"]
            trusted_turn.states = list(
                turn_projection["states"],
            )
            trusted_turn.terminal_reason = (
                turn_projection["terminal_reason"]
            )
            result["_trusted_turn"] = trusted_turn
            from agent.turn_finalizer import (
                persist_deferred_true_moa_turn,
            )

            deferred_persistence_errors = (
                persist_deferred_true_moa_turn(
                    self.agent,
                    messages=deferred_persistence_messages,
                    conversation_history=deferred_persistence_history,
                    user_message=request.user_message,
                    completed=True,
                )
            )
            if deferred_persistence_errors:
                result.setdefault("cleanup_errors", []).extend(
                    deferred_persistence_errors
                )
        elif not deferred_persistence_ready:
            result.pop("_trusted_turn", None)
        usage = _true_moa_usage_summary(ledger)
        result["_true_moa_usage"] = usage["true_moa"]
        self.emit_final_metadata(result, usage)
        return result, usage

    def emit_final_metadata(
        self,
        result: dict[str, Any],
        usage: dict[str, Any],
    ) -> None:
        request = self.request
        effective_session_id = getattr(
            self.agent,
            "session_id",
            request.session_id,
        )
        if (
            isinstance(effective_session_id, str)
            and effective_session_id
        ):
            result["session_id"] = effective_session_id
        if request.metadata_trace is None:
            return
        result_interrupted = bool(result.get("interrupted"))
        result_partial = bool(result.get("partial"))
        result_failed = bool(result.get("failed"))
        result_completed = bool(result.get("completed", True))
        if (
            result_interrupted
            or result_partial
            or result_failed
            or not result_completed
        ):
            request.metadata_trace.safe_emit(
                "request_failed",
                status="failed",
                duration_ms=request.metadata_trace.elapsed_ms(),
                tool_count=request.trace_state.tool_count,
                error_code=(
                    "completion_stopped"
                    if result_interrupted
                    else "output_truncated"
                    if result_partial
                    else "agent_error"
                ),
            )
            return
        completed_fields: dict[str, Any] = {
            "status": "completed",
            "duration_ms": request.metadata_trace.elapsed_ms(),
            "tool_count": request.trace_state.tool_count,
            "memory_enabled": bool(
                request.memory_identity
                and request.memory_identity[2] == "user"
            ),
            "memory_hit_count": self.memory_hit_count,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        }
        if getattr(self.agent, "provider", ""):
            completed_fields["provider"] = self.agent.provider
        if getattr(self.agent, "model", ""):
            completed_fields["model"] = self.agent.model
        request.metadata_trace.safe_emit(
            "request_completed",
            **completed_fields,
        )
