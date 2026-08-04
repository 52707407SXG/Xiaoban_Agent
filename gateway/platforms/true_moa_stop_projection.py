"""Stop linearization and plaintext-free true-MoA response projection."""

from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class IdempotencyConflictError(ValueError):
    """Raised when one idempotency key is reused for a different request."""


class CompletionStoppedError(RuntimeError):
    """Raised when a trusted delivery is stopped before agent execution."""


def _interrupt_agent_async(agent: Any, reason: str) -> None:
    """Best-effort SDK interruption that cannot hold a true-MoA stop request."""

    def _interrupt() -> None:
        try:
            agent.interrupt(reason)
        except BaseException:
            logger.warning("Failed to interrupt idempotent API run", exc_info=False)

    threading.Thread(
        target=_interrupt,
        name="xiaoban-true-moa-agent-interrupt",
        daemon=True,
    ).start()


def _cancel_chat_agent_ref(agent_ref: Optional[list], reason: str) -> bool:
    """Cancel both true-MoA advisor work and the acting agent, if registered."""

    if agent_ref is None:
        return False
    while len(agent_ref) < 4:
        agent_ref.append(None)
    # The durable stop fence is already committed before this helper is
    # called.  Close the live approval/steer bridge first so no concurrent
    # control can release a waiting tool in the interval before Agent
    # interruption becomes visible.
    control_bridge = agent_ref[3]
    if control_bridge is not None:
        try:
            control_bridge.close()
        except BaseException:
            logger.warning("Failed to close stopped chat control bridge", exc_info=False)
    controller = agent_ref[2]
    if controller is not None:
        try:
            if not controller.cancel():
                return False
        except BaseException:
            logger.warning("Failed to cancel true MoA advisor wave", exc_info=False)
            return False
    agent_ref[1] = True
    agent = agent_ref[0]
    if agent is not None:
        if controller is not None:
            _interrupt_agent_async(agent, reason)
        else:
            try:
                agent.interrupt(reason)
            except Exception:
                logger.warning("Failed to interrupt idempotent API run", exc_info=False)
    return True


def _true_moa_usage_summary(ledger: Any) -> Dict[str, Any]:
    """Project one plaintext-free MoA ledger into aggregate OpenAI usage."""

    payload = ledger.to_dict()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    call_receipts = payload.get("calls")
    if not isinstance(call_receipts, list):
        call_receipts = payload.get("slots") or ()
    for call in call_receipts:
        if not isinstance(call, dict):
            continue
        call_input = call.get("inputTokens")
        call_output = call.get("outputTokens")
        call_total = call.get("totalTokens")
        if isinstance(call_input, int) and not isinstance(call_input, bool):
            input_tokens += max(0, call_input)
        if isinstance(call_output, int) and not isinstance(call_output, bool):
            output_tokens += max(0, call_output)
        if isinstance(call_total, int) and not isinstance(call_total, bool):
            total_tokens += max(0, call_total)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "true_moa": payload,
    }


def _stopped_chat_completion_response(response: Any) -> Any:
    """Return a plaintext-free stop result while preserving actual accounting.

    The idempotency tombstone is authoritative even if the worker completes in
    the same event-loop turn.  No assistant/provider text survives this
    projection.  Reported token/cost fields remain available for settlement,
    and a true-MoA final slot is marked cancelled without rewriting its
    already-reported usage.
    """

    is_pair = isinstance(response, tuple) and len(response) == 2
    raw_result, raw_usage = response if is_pair else (response, {})
    result = raw_result if isinstance(raw_result, dict) else {}
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    fence_timestamp = int(time.time() * 1000)

    stopped_usage: Dict[str, Any] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            stopped_usage[name] = max(0, value)
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        stopped_usage.setdefault(name, 0)

    ledger_source = usage.get("true_moa")
    if not isinstance(ledger_source, dict):
        ledger_source = result.get("_true_moa_usage")
    ledger = copy.deepcopy(ledger_source) if isinstance(ledger_source, dict) else None
    if ledger is not None:
        ledger["status"] = "cancelled"
        slots = ledger.get("slots")
        if isinstance(slots, list):
            for slot in slots:
                if not isinstance(slot, dict):
                    continue
                if (
                    slot.get("role") == "final_executor"
                    or slot.get("slotId") == "final-deepseek-v4-pro"
                ):
                    slot["status"] = "cancelled"
                    slot["errorCategory"] = "terminal_fence_after_stop"
                    slot["endedAtMs"] = max(
                        int(slot.get("startedAtMs") or 0),
                        int(slot.get("endedAtMs") or fence_timestamp),
                    )
        calls = ledger.get("calls")
        if isinstance(calls, list):
            for call in calls:
                if (
                    not isinstance(call, dict)
                    or (
                        call.get("role") != "final_executor"
                        and call.get("slotId") != "final-deepseek-v4-pro"
                    )
                ):
                    continue
                if call.get("status") == "reserved":
                    call["status"] = "not_dispatched"
                    call["errorCategory"] = (
                        "provider_dispatch_fence_closed"
                    )
                    call["endedAtMs"] = max(
                        int(call.get("startedAtMs") or 0),
                        fence_timestamp,
                    )
                    continue
                if call.get("status") != "running":
                    continue
                # Advisor calls keep running only for their exact usage drain.
                # The final transport still uses Agent interruption, so its
                # stop projection must become terminal instead of leaving My
                # Stand in stop_requested forever.  A late exact usage receipt
                # may still fill the empty accounting fields.
                call["status"] = "cancelled"
                call["errorCategory"] = "terminal_fence_after_stop"
                call["endedAtMs"] = max(
                    int(call.get("startedAtMs") or 0),
                    fence_timestamp,
                )
        stopped_usage["true_moa"] = ledger

    agent_call_source = usage.get("agent_calls")
    if not isinstance(agent_call_source, dict):
        agent_call_source = result.get("_agent_call_usage")
    agent_calls = (
        copy.deepcopy(agent_call_source)
        if isinstance(agent_call_source, dict)
        else None
    )
    if agent_calls is not None:
        agent_calls["status"] = "cancelled"
        calls = agent_calls.get("calls")
        if isinstance(calls, list):
            for call in calls:
                if (
                    not isinstance(call, dict)
                ):
                    continue
                if call.get("status") == "reserved":
                    call["status"] = "not_dispatched"
                    call["errorCategory"] = (
                        "provider_dispatch_fence_closed"
                    )
                elif call.get("status") == "running":
                    call["status"] = "timed_out"
                    call["errorCategory"] = "completion_stopped"
                else:
                    continue
                call["endedAtMs"] = max(
                    int(call.get("startedAtMs") or 0),
                    fence_timestamp,
                )
        stopped_usage["agent_calls"] = agent_calls

    stopped_result: Dict[str, Any] = {
        "final_response": "",
        "messages": [],
        "completed": False,
        "failed": True,
        "interrupted": True,
        "error": "completion stopped",
    }
    session_id = result.get("session_id")
    if isinstance(session_id, str) and session_id:
        stopped_result["session_id"] = session_id
    if ledger is not None:
        stopped_result["_true_moa_usage"] = ledger
    if agent_calls is not None:
        stopped_result["_agent_call_usage"] = agent_calls
    return (stopped_result, stopped_usage) if is_pair else stopped_result
