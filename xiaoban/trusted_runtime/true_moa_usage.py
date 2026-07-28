"""Thread-safe, plaintext-free true-MoA usage receipts."""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from xiaoban.trusted_runtime.true_moa_contracts import (
    FINAL_EXECUTOR_SLOT,
    TRUE_MOA_ADVISOR_SHUTDOWN_GRACE_SECONDS,
    TRUE_MOA_ADVISOR_SLOTS,
    TRUE_MOA_ALL_SLOTS,
    TRUE_MOA_FINAL_CALL_LIMIT,
    TRUE_MOA_USAGE_SCHEMA,
    TrueMoASlot,
    TrueMoASnapshot,
)

@dataclass
class _SlotReceipt:
    slot: TrueMoASlot
    call_id: str
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


class TrueMoADurableNotification:
    """One bounded-wait receipt for an out-of-lock durable ledger callback."""

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

    def _finish(self, error: BaseException | None = None) -> None:
        self._error = error
        self._done.set()


class TrueMoAUsageLedger:
    """Thread-safe, plaintext-free per-slot usage receipt."""

    def __init__(
        self,
        snapshot: TrueMoASnapshot,
        *,
        wave_id: str | None = None,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.snapshot = snapshot
        self.wave_id = wave_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._on_change = on_change
        self._callback_lock = threading.Lock()
        self._callback_configured = on_change is not None
        self._callback_failure: BaseException | None = None
        self._wave_status = "pending"
        self._receipts = {
            slot.slot_id: _SlotReceipt(
                slot=slot,
                call_id=f"{self.wave_id}:{slot.slot_id}",
                status="not_started",
            )
            for slot in TRUE_MOA_ALL_SLOTS
        }
        self._advisor_calls: dict[str, _SlotReceipt] = {}
        self._final_calls: dict[str, _SlotReceipt] = {}

    def set_wave_status(self, status: str, *, notify: bool = True) -> None:
        with self._lock:
            self._wave_status = status
        if notify:
            self._notify_change()

    def start_slot(
        self,
        slot: TrueMoASlot,
        *,
        started_at_ms: int | None = None,
        notify: bool = True,
    ) -> None:
        with self._lock:
            receipt = self._receipts[slot.slot_id]
            if receipt.status != "not_started":
                raise RuntimeError(f"slot already started: {slot.slot_id}")
            receipt.status = "running"
            receipt.started_at_ms = started_at_ms or _now_ms()
        if notify:
            self._notify_change()

    def finish_slot(
        self,
        slot: TrueMoASlot,
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
        (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            usage_status,
        ) = _normalize_usage(usage)
        with self._lock:
            receipt = self._receipts[slot.slot_id]
            # A timeout/cancel terminal fence wins over a late provider result.
            # Actual late usage may still fill empty accounting fields.
            if receipt.status in {"not_started", "running"}:
                receipt.status = status
                receipt.error_category = error_category
                receipt.ended_at_ms = ended_at_ms or _now_ms()
            elif receipt.ended_at_ms is None:
                receipt.ended_at_ms = ended_at_ms or _now_ms()
            _fill_usage_once(
                receipt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_input_tokens,
                usage_status=usage_status,
            )
            if cost_usd is not None and receipt.cost_usd is None:
                receipt.cost_usd = float(cost_usd)
            if cost_status and not receipt.cost_status:
                receipt.cost_status = _safe_category(cost_status)
            if cost_source and not receipt.cost_source:
                receipt.cost_source = _safe_category(cost_source)
        if notify:
            self._notify_change()

    def terminate_unfinished(
        self,
        *,
        status: str,
        error_category: str,
        preserve_running_calls: bool = False,
        notify: bool = True,
    ) -> None:
        ended_at_ms = _now_ms()
        with self._lock:
            for receipt in self._receipts.values():
                if receipt.slot.role != "advisor":
                    continue
                if receipt.status in {"not_started", "running"}:
                    receipt.status = status
                    receipt.error_category = error_category
                    receipt.ended_at_ms = ended_at_ms
                elif (
                    receipt.status == "cancelled"
                    and error_category.startswith("cascade_after_")
                ):
                    # A peer may observe the controller callback and mark
                    # itself cancelled before the coordinator writes the
                    # canonical wave-failure cause.  The coordinator cause
                    # wins so the durable ledger is independent of thread
                    # scheduling.
                    receipt.error_category = error_category
            if not preserve_running_calls:
                for receipt in self._advisor_calls.values():
                    if receipt.status == "running":
                        receipt.status = status
                        receipt.error_category = error_category
                        receipt.ended_at_ms = ended_at_ms
        if notify:
            self._notify_change()

    def start_advisor_call(
        self,
        slot: TrueMoASlot,
        *,
        started_at_ms: int | None = None,
        notify: bool = True,
    ) -> str:
        """Record one advisor call only at the provider dispatch boundary."""

        if slot not in TRUE_MOA_ADVISOR_SLOTS:
            raise RuntimeError(f"invalid true MoA advisor slot: {slot.slot_id}")
        call_id = f"{self.wave_id}:{slot.slot_id}"
        with self._lock:
            if slot.slot_id in self._advisor_calls:
                raise RuntimeError(f"advisor provider call already started: {call_id}")
            self._advisor_calls[slot.slot_id] = _SlotReceipt(
                slot=slot,
                call_id=call_id,
                status="running",
                started_at_ms=started_at_ms or _now_ms(),
            )
        if notify:
            self._notify_change()
        return call_id

    def finish_advisor_call(
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
    ) -> bool:
        """Fill one dispatched advisor call without rewriting its terminal fence."""

        (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            usage_status,
        ) = _normalize_usage(usage)
        terminal_transitioned = False
        with self._lock:
            receipt = next(
                (
                    item
                    for item in self._advisor_calls.values()
                    if item.call_id == str(call_id or "")
                ),
                None,
            )
            if receipt is None:
                raise RuntimeError(f"unknown advisor provider call: {call_id}")
            if receipt.status == "running":
                receipt.status = status
                receipt.error_category = error_category
                receipt.ended_at_ms = ended_at_ms or _now_ms()
                terminal_transitioned = True
            elif receipt.ended_at_ms is None:
                receipt.ended_at_ms = ended_at_ms or _now_ms()
            _fill_usage_once(
                receipt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_input_tokens,
                usage_status=usage_status,
            )
            if cost_usd is not None and receipt.cost_usd is None:
                receipt.cost_usd = float(cost_usd)
            if cost_status and not receipt.cost_status:
                receipt.cost_status = _safe_category(cost_status)
            if cost_source and not receipt.cost_source:
                receipt.cost_source = _safe_category(cost_source)
        if notify:
            self._notify_change()
        return terminal_transitioned

    def start_final_call(
        self,
        request_id: str,
        *,
        started_at_ms: int | None = None,
        notify: bool = True,
    ) -> str:
        """Open one independently billable final-executor provider call."""

        safe_request_id = _safe_category(request_id)
        call_id = (
            f"{self.wave_id}:{FINAL_EXECUTOR_SLOT.slot_id}:{safe_request_id}"
        )
        with self._lock:
            final_slot = self._receipts[FINAL_EXECUTOR_SLOT.slot_id]
            if final_slot.status == "timed_out":
                raise RuntimeError(
                    "true MoA final execution already timed out"
                )
            if call_id in self._final_calls:
                raise RuntimeError(f"final provider call already started: {call_id}")
            if len(self._final_calls) >= TRUE_MOA_FINAL_CALL_LIMIT:
                raise RuntimeError("true MoA final provider call limit exceeded")
            self._final_calls[call_id] = _SlotReceipt(
                slot=FINAL_EXECUTOR_SLOT,
                call_id=call_id,
                status="running",
                started_at_ms=started_at_ms or _now_ms(),
            )
        if notify:
            self._notify_change()
        return call_id

    def finish_final_call(
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
        """Fill one final-executor call exactly once without rewriting a fence."""

        (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            usage_status,
        ) = _normalize_usage(usage)
        with self._lock:
            receipt = self._final_calls.get(str(call_id or ""))
            if receipt is None:
                raise RuntimeError(f"unknown final provider call: {call_id}")
            if receipt.status == "running":
                receipt.status = status
                receipt.error_category = error_category
                receipt.ended_at_ms = ended_at_ms or _now_ms()
            elif receipt.ended_at_ms is None:
                receipt.ended_at_ms = ended_at_ms or _now_ms()
            _fill_usage_once(
                receipt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_input_tokens,
                usage_status=usage_status,
            )
            if cost_usd is not None and receipt.cost_usd is None:
                receipt.cost_usd = float(cost_usd)
            if cost_status and not receipt.cost_status:
                receipt.cost_status = _safe_category(cost_status)
            if cost_source and not receipt.cost_source:
                receipt.cost_source = _safe_category(cost_source)
            self._fill_terminal_final_slot_usage_locked()
        if notify:
            self._notify_change()

    def _fill_terminal_final_slot_usage_locked(self) -> None:
        """Backfill aggregate slot counters when a fenced late call settles."""

        final_slot = self._receipts[FINAL_EXECUTOR_SLOT.slot_id]
        if final_slot.status in {"not_started", "running"}:
            return
        receipts = list(self._final_calls.values())
        if not receipts:
            return

        aggregates: dict[str, int | None] = {}
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
        ):
            values = [getattr(receipt, field) for receipt in receipts]
            aggregates[field] = (
                sum(values)
                if all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    for value in values
                )
                else None
            )
        usage_status = (
            "reported"
            if all(
                receipt.usage_status == "reported"
                for receipt in receipts
            )
            else "partial"
        )
        _fill_usage_once(
            final_slot,
            input_tokens=aggregates["input_tokens"],
            output_tokens=aggregates["output_tokens"],
            total_tokens=aggregates["total_tokens"],
            cached_input_tokens=aggregates["cached_input_tokens"],
            usage_status=usage_status,
        )

    def final_call_usage(self) -> dict[str, int] | None:
        """Aggregate only counters actually reported by every final call."""

        with self._lock:
            receipts = list(self._final_calls.values())
            if not receipts:
                return None
            usage: dict[str, int] = {}
            for output_name, field in (
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
                ("cached_input_tokens", "cached_input_tokens"),
            ):
                values = [getattr(receipt, field) for receipt in receipts]
                if all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in values
                ):
                    usage[output_name] = sum(values)
            return usage or None

    def timeout_final_execution(
        self,
        *,
        error_category: str = "final_executor_timeout",
        notify: bool = True,
    ) -> None:
        """Fence the final slot and every in-flight paid call as timed out.

        The controller's terminal lock is the linearization point and must be
        won before calling this method. A provider result that arrives after
        this fence may fill previously empty accounting fields through the
        normal ``finish_*`` methods, but can never rewrite ``timed_out``.
        """

        ended_at_ms = _now_ms()
        safe_error = _safe_category(error_category)
        with self._lock:
            final_receipt = self._receipts[FINAL_EXECUTOR_SLOT.slot_id]
            if final_receipt.status in {"not_started", "running"}:
                final_receipt.status = "timed_out"
                final_receipt.error_category = safe_error
                final_receipt.ended_at_ms = ended_at_ms
            for receipt in self._final_calls.values():
                if receipt.status == "running":
                    receipt.status = "timed_out"
                    receipt.error_category = safe_error
                    receipt.ended_at_ms = ended_at_ms
            self._wave_status = "failed"
        if notify:
            self._notify_change()

    def _notify_change(self) -> None:
        callback, callback_failure = self._callback_state()
        if callback_failure is not None:
            raise RuntimeError(
                "true MoA durable usage ledger unavailable"
            ) from callback_failure
        if callback is None:
            return
        try:
            self._invoke_callback(callback, self.to_dict())
        except BaseException as exc:
            raise RuntimeError(
                "true MoA durable usage ledger unavailable"
            ) from exc

    def notify_change_async(self) -> TrueMoADurableNotification:
        """Dispatch one immutable snapshot without blocking its caller.

        Terminal owners use this only after the controller and ledger have
        already reached their in-memory terminal states.  The returned receipt
        can be waited on for a fixed grace budget; a hung SQLite write therefore
        cannot occupy the hard-deadline thread.
        """

        receipt = TrueMoADurableNotification()
        callback, callback_failure = self._callback_state()
        if callback_failure is not None:
            receipt._finish(callback_failure)
            return receipt
        if callback is None:
            receipt._finish(
                None
                if not self._callback_configured
                else RuntimeError("true MoA durable callback detached")
            )
            return receipt
        payload = self.to_dict()

        def _run_callback() -> None:
            try:
                self._invoke_callback(callback, payload)
            except BaseException as exc:
                receipt._finish(exc)
            else:
                receipt._finish()

        threading.Thread(
            target=_run_callback,
            name="xiaoban-true-moa-durable-notify",
            daemon=True,
        ).start()
        return receipt

    def confirm_change(
        self,
        timeout: float = TRUE_MOA_ADVISOR_SHUTDOWN_GRACE_SECONDS,
    ) -> bool:
        receipt = self.notify_change_async()
        receipt.wait(timeout)
        return receipt.confirmed

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
                    raise RuntimeError("true MoA durable callback detached")
            try:
                callback(payload)
            except BaseException as exc:
                # A failed callback is detached exactly once.  Terminal error
                # handlers may continue mutating the plaintext-free local
                # ledger without recursively invoking the same failed writer.
                with self._lock:
                    if self._on_change is callback:
                        self._on_change = None
                    if self._callback_failure is None:
                        self._callback_failure = exc
                raise

    @staticmethod
    def _receipt_dict(receipt: _SlotReceipt) -> dict[str, Any]:
        item: dict[str, Any] = {
            "slotId": receipt.slot.slot_id,
            "callId": receipt.call_id,
            "provider": receipt.slot.provider,
            "model": receipt.slot.model,
            "role": receipt.slot.role,
            "startedAtMs": receipt.started_at_ms,
            "endedAtMs": receipt.ended_at_ms,
            "status": receipt.status,
            "inputTokens": receipt.input_tokens,
            "outputTokens": receipt.output_tokens,
            "totalTokens": receipt.total_tokens,
            "cachedInputTokens": receipt.cached_input_tokens,
            "usageStatus": receipt.usage_status,
        }
        if receipt.error_category:
            item["errorCategory"] = receipt.error_category
        if receipt.cost_usd is not None:
            item["costUsd"] = receipt.cost_usd
        if receipt.cost_status:
            item["costStatus"] = receipt.cost_status
        if receipt.cost_source:
            item["costSource"] = receipt.cost_source
        return item

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            slots = [
                self._receipt_dict(self._receipts[slot.slot_id])
                for slot in TRUE_MOA_ALL_SLOTS
            ]
            calls = [
                self._receipt_dict(self._advisor_calls[slot.slot_id])
                for slot in TRUE_MOA_ADVISOR_SLOTS
                if slot.slot_id in self._advisor_calls
            ]
            calls.extend(
                self._receipt_dict(receipt)
                for receipt in self._final_calls.values()
            )
            return {
                "schema": TRUE_MOA_USAGE_SCHEMA,
                "waveId": self.wave_id,
                "mode": self.snapshot.mode,
                "modeEpoch": self.snapshot.mode_epoch,
                "presetId": self.snapshot.preset_id,
                "presetRevision": self.snapshot.preset_revision,
                "status": self._wave_status,
                "slots": slots,
                "calls": calls,
            }


def _fill_usage_once(
    receipt: _SlotReceipt,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    cached_input_tokens: int | None,
    usage_status: str,
) -> None:
    if usage_status == "unavailable":
        return
    for field, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("total_tokens", total_tokens),
        ("cached_input_tokens", cached_input_tokens),
    ):
        if value is not None and getattr(receipt, field) is None:
            setattr(receipt, field, value)
    if usage_status == "reported" and all(
        isinstance(getattr(receipt, field), int)
        and not isinstance(getattr(receipt, field), bool)
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
        )
    ):
        receipt.usage_status = "reported"
    elif receipt.usage_status != "reported":
        receipt.usage_status = "partial"


def _normalize_usage(
    usage: Any,
) -> tuple[int | None, int | None, int | None, int | None, str]:
    if usage is None:
        return None, None, None, None, "unavailable"

    def _member(source: Any, name: str) -> tuple[bool, Any]:
        if isinstance(source, Mapping):
            return (name in source, source.get(name))
        if hasattr(source, name):
            return (True, getattr(source, name, None))
        return (False, None)

    def _nonnegative_int(raw: Any) -> int | None:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw if raw >= 0 else None
        if isinstance(raw, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", raw):
            return int(raw)
        return None

    def _value(*names: str) -> int | None:
        for name in names:
            present, raw = _member(usage, name)
            if not present or raw is None:
                continue
            value = _nonnegative_int(raw)
            if value is not None:
                return value
        return None

    def _cached_value() -> int | None:
        values: list[int] = []
        invalid = False
        for name in (
            "cached_input_tokens",
            "cachedInputTokens",
            "prompt_cache_hit_tokens",
            "cached_prompt_tokens",
            "cache_read_input_tokens",
            "cache_read_tokens",
        ):
            present, raw = _member(usage, name)
            if not present or raw is None:
                continue
            value = _nonnegative_int(raw)
            if value is None:
                invalid = True
                continue
            values.append(value)
        for details_name in (
            "prompt_tokens_details",
            "input_tokens_details",
        ):
            present, details = _member(usage, details_name)
            if not present or details is None:
                continue
            cached_present, raw = _member(details, "cached_tokens")
            if not cached_present or raw is None:
                continue
            value = _nonnegative_int(raw)
            if value is None:
                invalid = True
                continue
            values.append(value)
        if invalid or not values or len(set(values)) != 1:
            return None
        return values[0]

    input_tokens = _value("input_tokens", "prompt_tokens")
    output_tokens = _value("output_tokens", "completion_tokens")
    total_tokens = _value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if (
        input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
        and total_tokens != input_tokens + output_tokens
    ):
        total_tokens = None
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None, None, None, None, "unavailable"
    cached_input_tokens = _cached_value()
    if (
        cached_input_tokens is not None
        and input_tokens is not None
        and cached_input_tokens > input_tokens
    ):
        cached_input_tokens = None
    status = (
        "reported"
        if all(
            value is not None
            for value in (
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
            )
        )
        else "partial"
    )
    return (
        input_tokens,
        output_tokens,
        total_tokens,
        cached_input_tokens,
        status,
    )


def _safe_category(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(value or ""))[:80]
    return cleaned or "unknown"


def _now_ms() -> int:
    return int(time.time() * 1000)
