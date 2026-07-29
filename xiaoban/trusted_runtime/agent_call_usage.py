"""Runtime lifecycle for signed My Stand provider-call receipts."""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from xiaoban.trusted_runtime.agent_call_usage_codec import (
    AGENT_CALL_LIMIT,
    AGENT_CALL_USAGE_SCHEMA,
    fill_cost_once,
    fill_usage_once,
    merge_agent_call_usage,
    normalize_usage,
    project_agent_call_usage,
    project_route,
    receipt_dict,
    safe_category,
)


AGENT_CALL_DURABLE_CONFIRM_SECONDS = 5.0
_EXECUTION_ID = re.compile(r"^[a-f0-9]{32}$")
_RECEIPT_TERMINALS = {"completed", "failed", "cancelled", "timed_out"}
_LEDGER_TERMINALS = {"completed", "failed", "cancelled"}


@dataclass
class CallReceipt:
    """One provider dispatch receipt; contains counters, never request text."""

    call_id: str
    ordinal: int
    provider: str
    model: str
    role: str
    status: str
    started_at_ms: int | None = None
    ended_at_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    usage_status: str = "unavailable"
    cost_usd: float | None = None
    cost_status: str | None = None
    cost_source: str | None = None
    error_category: str | None = None


class DurableNotification:
    """Bounded-wait confirmation for an out-of-lock durable callback."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._error: BaseException | None = None

    @property
    def confirmed(self) -> bool:
        return self._done.is_set() and self._error is None

    @property
    def error(self) -> BaseException | None:
        return self._error if self._done.is_set() else None

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def finish(self, error: BaseException | None = None) -> None:
        self._error = error
        self._done.set()


class AgentCallUsageLedger:
    """Thread-safe call ledger for one signed normal-mode delivery."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        execution_id: str | None = None,
        on_change: Callable[[dict[str, Any]], None] | None = None,
        max_calls: int = AGENT_CALL_LIMIT,
    ):
        self.execution_id = execution_id or uuid.uuid4().hex
        if not _EXECUTION_ID.fullmatch(self.execution_id):
            raise ValueError("invalid agent call execution id")
        self.provider = project_route(provider, field="provider")
        self.model = project_route(model, field="model")
        if (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or max_calls < 1
            or max_calls > AGENT_CALL_LIMIT
        ):
            raise ValueError("invalid agent call limit")
        self.max_calls = max_calls
        self._lock = threading.Lock()
        self._callback_lock = threading.Lock()
        self._on_change = on_change
        self._callback_configured = on_change is not None
        self._callback_failure: BaseException | None = None
        self._status = "running"
        self._calls: list[CallReceipt] = []

    def start_call(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        role: str = "agent",
        started_at_ms: int | None = None,
        notify: bool = True,
    ) -> str:
        clean_provider = project_route(
            provider if provider is not None else self.provider,
            field="provider",
        )
        clean_model = project_route(
            model if model is not None else self.model,
            field="model",
        )
        clean_role = project_route(role, field="role")
        with self._lock:
            if self._status != "running":
                raise RuntimeError("agent call ledger is terminal")
            ordinal = len(self._calls) + 1
            if ordinal > self.max_calls:
                raise RuntimeError("signed My Stand provider call limit exceeded")
            call_id = f"{self.execution_id}:call:{ordinal:06d}"
            self._calls.append(
                CallReceipt(
                    call_id=call_id,
                    ordinal=ordinal,
                    provider=clean_provider,
                    model=clean_model,
                    role=clean_role,
                    status="reserved",
                    started_at_ms=started_at_ms or _now_ms(),
                )
            )
        if notify:
            self._notify_change()
        return call_id

    def mark_dispatched(
        self,
        call_id: str,
        *,
        notify: bool = True,
    ) -> None:
        """Commit one reservation at the physical provider boundary."""

        with self._lock:
            receipt = next(
                (item for item in self._calls if item.call_id == call_id),
                None,
            )
            if receipt is None:
                raise RuntimeError(f"unknown agent provider call: {call_id}")
            if receipt.status == "not_dispatched":
                raise RuntimeError("agent provider call was not dispatched")
            if receipt.status != "reserved":
                raise RuntimeError("agent provider call is not reserved")
            receipt.status = "running"
        if notify:
            self._notify_change()

    def finish_not_dispatched(
        self,
        call_id: str,
        *,
        ended_at_ms: int | None = None,
        notify: bool = True,
    ) -> None:
        """Prove a reservation lost its final provider dispatch fence."""

        with self._lock:
            receipt = next(
                (item for item in self._calls if item.call_id == call_id),
                None,
            )
            if receipt is None:
                raise RuntimeError(f"unknown agent provider call: {call_id}")
            if receipt.status == "not_dispatched":
                return
            if receipt.status != "reserved":
                raise RuntimeError(
                    "agent provider call is not a reserved dispatch"
                )
            receipt.status = "not_dispatched"
            receipt.ended_at_ms = ended_at_ms or _now_ms()
            receipt.error_category = "provider_dispatch_fence_closed"
        if notify:
            self._notify_change()

    def finish_call(
        self,
        call_id: str,
        *,
        status: str,
        usage: Any = None,
        ended_at_ms: int | None = None,
        error_category: str | None = None,
        cost_usd: float | None = None,
        cost_status: str | None = None,
        cost_source: str | None = None,
        notify: bool = True,
    ) -> None:
        if status not in _RECEIPT_TERMINALS:
            raise ValueError("invalid agent call terminal status")
        normalized = normalize_usage(usage)
        with self._lock:
            receipt = next(
                (item for item in self._calls if item.call_id == call_id),
                None,
            )
            if receipt is None:
                raise RuntimeError(f"unknown agent provider call: {call_id}")
            if receipt.status == "not_dispatched":
                raise RuntimeError("agent provider call was not dispatched")
            if receipt.status == "reserved":
                raise RuntimeError("agent provider call is only reserved")
            if receipt.status == "running":
                receipt.status = status
                receipt.ended_at_ms = ended_at_ms or _now_ms()
                receipt.error_category = (
                    safe_category(error_category)
                    if error_category
                    else None
                )
            elif receipt.ended_at_ms is None:
                receipt.ended_at_ms = ended_at_ms or _now_ms()
            fill_usage_once(receipt, *normalized)
            fill_cost_once(
                receipt,
                cost_usd=cost_usd,
                cost_status=cost_status,
                cost_source=cost_source,
            )
        if notify:
            self._notify_change()

    def terminalize_running(
        self,
        *,
        status: str,
        error_category: str,
        notify: bool = True,
    ) -> None:
        if status not in {"cancelled", "timed_out", "failed"}:
            raise ValueError("invalid agent call fence status")
        ended_at_ms = _now_ms()
        clean_error = safe_category(error_category)
        with self._lock:
            for receipt in self._calls:
                if receipt.status == "reserved":
                    receipt.status = "not_dispatched"
                    receipt.ended_at_ms = ended_at_ms
                    receipt.error_category = (
                        "provider_dispatch_fence_closed"
                    )
                elif receipt.status == "running":
                    receipt.status = status
                    receipt.ended_at_ms = ended_at_ms
                    receipt.error_category = clean_error
        if notify:
            self._notify_change()

    def set_status(self, status: str, *, notify: bool = True) -> None:
        if status not in {"running", *_LEDGER_TERMINALS}:
            raise ValueError("invalid agent call ledger status")
        with self._lock:
            if self._status in _LEDGER_TERMINALS and self._status != status:
                raise RuntimeError("agent call ledger is already terminal")
            if status == "completed" and any(
                receipt.status in {
                    "reserved",
                    "running",
                    "not_dispatched",
                }
                for receipt in self._calls
            ):
                raise ValueError(
                    "completed agent ledger has unresolved provider call"
                )
            self._status = status
        if notify:
            self._notify_change()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": AGENT_CALL_USAGE_SCHEMA,
                "executionId": self.execution_id,
                "status": self._status,
                "calls": [receipt_dict(receipt) for receipt in self._calls],
            }

    def notify_change_async(self) -> DurableNotification:
        notification = DurableNotification()
        callback, callback_failure = self._callback_state()
        if callback_failure is not None:
            notification.finish(callback_failure)
            return notification
        if callback is None:
            notification.finish(
                None
                if not self._callback_configured
                else RuntimeError("agent call durable callback detached")
            )
            return notification
        payload = self.to_dict()

        def run_callback() -> None:
            try:
                self._invoke_callback(callback, payload)
            except BaseException as exc:
                notification.finish(exc)
            else:
                notification.finish()

        threading.Thread(
            target=run_callback,
            name="xiaoban-agent-call-durable-notify",
            daemon=True,
        ).start()
        return notification

    def confirm_change(
        self,
        timeout: float = AGENT_CALL_DURABLE_CONFIRM_SECONDS,
    ) -> bool:
        notification = self.notify_change_async()
        notification.wait(timeout)
        return notification.confirmed

    def _notify_change(self) -> None:
        callback, callback_failure = self._callback_state()
        if callback_failure is not None:
            raise RuntimeError(
                "agent call durable usage ledger unavailable"
            ) from callback_failure
        if callback is None:
            return
        try:
            self._invoke_callback(callback, self.to_dict())
        except BaseException as exc:
            raise RuntimeError(
                "agent call durable usage ledger unavailable"
            ) from exc

    def _callback_state(
        self,
    ) -> tuple[
        Callable[[dict[str, Any]], None] | None,
        BaseException | None,
    ]:
        with self._lock:
            return self._on_change, self._callback_failure

    def _invoke_callback(
        self,
        callback: Callable[[dict[str, Any]], None],
        payload: dict[str, Any],
    ) -> None:
        with self._callback_lock:
            with self._lock:
                if self._callback_failure is not None:
                    raise self._callback_failure
                if self._on_change is not callback:
                    raise RuntimeError("agent call durable callback detached")
            try:
                callback(payload)
            except BaseException as exc:
                with self._lock:
                    if self._on_change is callback:
                        self._on_change = None
                    if self._callback_failure is None:
                        self._callback_failure = exc
                raise


def _now_ms() -> int:
    return int(time.time() * 1000)


__all__ = [
    "AGENT_CALL_DURABLE_CONFIRM_SECONDS",
    "AGENT_CALL_LIMIT",
    "AGENT_CALL_USAGE_SCHEMA",
    "AgentCallUsageLedger",
    "CallReceipt",
    "DurableNotification",
    "merge_agent_call_usage",
    "normalize_usage",
    "project_agent_call_usage",
]
