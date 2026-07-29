"""Process-local coordination for durable provider-usage drains."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

from xiaoban.trusted_runtime.protocol_contract import (
    MYSTAND_TRUE_MOA_USAGE_SCHEMA,
)


class _TrueMoAUsageDrainMixin:
    """Own heartbeat, takeover, orphan fencing, and late usage persistence."""

    def _initialize_usage_drain_state(
        self,
        usage_drain_lease_seconds: float,
    ) -> None:
        self._usage_drains: set[str] = set()
        self._closed_usage_drain_owners: Dict[str, float] = {}
        self._usage_drain_owner_id = uuid.uuid4().hex
        self._usage_drain_lease_ms = max(
            100,
            int(float(usage_drain_lease_seconds) * 1000),
        )
        self._usage_drain_generations: Dict[str, int] = {}
        self._usage_drain_heartbeat_timers: Dict[
            str,
            threading.Timer,
        ] = {}
        self._usage_drains_lock = threading.Lock()

    @staticmethod
    def _has_running_usage_receipt(record: Any) -> bool:
        usage = record.get("usage") if isinstance(record, dict) else None
        return bool(
            isinstance(usage, dict)
            and any(
                isinstance(call, dict)
                and call.get("status") in {"reserved", "running"}
                for call in (usage.get("calls") or ())
            )
        )

    def has_active_usage_drain(self, key: str) -> bool:
        with self._usage_drains_lock:
            return key in self._usage_drains

    def _schedule_usage_drain_heartbeat(self, key: str) -> None:
        if self._durable is None:
            return
        with self._usage_drains_lock:
            current = self._usage_drain_heartbeat_timers.get(key)
            if current is not None and current.is_alive():
                return
            if (
                key not in self._usage_drains
                or key not in self._usage_drain_generations
            ):
                return
            interval = max(
                0.05,
                self._usage_drain_lease_ms / 3000,
            )
            timer = threading.Timer(
                interval,
                self._heartbeat_usage_drain,
                args=(key,),
            )
            timer.daemon = True
            self._usage_drain_heartbeat_timers[key] = timer
            timer.start()

    def _heartbeat_usage_drain(self, key: str) -> bool:
        if self._durable is None:
            return False
        with self._usage_drains_lock:
            self._usage_drain_heartbeat_timers.pop(key, None)
            generation = self._usage_drain_generations.get(key)
            active = key in self._usage_drains
        if not active or generation is None:
            return False
        try:
            renewed = self._durable.renew_usage_drain_lease(
                key,
                owner_id=self._usage_drain_owner_id,
                generation=generation,
                lease_ms=self._usage_drain_lease_ms,
            )
        except Exception:
            # A timer thread must never keep advertising local ownership
            # after it can no longer prove the durable fencing generation.
            renewed = False
        if not renewed:
            with self._usage_drains_lock:
                self._usage_drains.discard(key)
                self._usage_drain_generations.pop(key, None)
            return False
        self._schedule_usage_drain_heartbeat(key)
        return True

    def _claim_usage_drain_lease(self, key: str) -> int:
        if self._durable is None:
            raise RuntimeError("true MoA durable ledger is unavailable")
        with self._usage_drains_lock:
            generation = self._usage_drain_generations.get(key)
        if generation is not None and self._durable.renew_usage_drain_lease(
            key,
            owner_id=self._usage_drain_owner_id,
            generation=generation,
            lease_ms=self._usage_drain_lease_ms,
        ):
            with self._usage_drains_lock:
                self._usage_drains.add(key)
            self._schedule_usage_drain_heartbeat(key)
            return generation
        generation = self._durable.claim_usage_drain_lease(
            key,
            owner_id=self._usage_drain_owner_id,
            lease_ms=self._usage_drain_lease_ms,
        )
        if generation is None:
            raise RuntimeError("true MoA usage drain is owned elsewhere")
        with self._usage_drains_lock:
            self._usage_drain_generations[key] = generation
            self._usage_drains.add(key)
        self._schedule_usage_drain_heartbeat(key)
        return generation

    def _release_usage_drain_lease(self, key: str) -> None:
        if self._durable is None:
            return
        with self._usage_drains_lock:
            generation = self._usage_drain_generations.pop(key, None)
            self._usage_drains.discard(key)
            timer = self._usage_drain_heartbeat_timers.pop(key, None)
        if timer is not None:
            timer.cancel()
        if generation is not None:
            self._durable.release_usage_drain_lease(
                key,
                owner_id=self._usage_drain_owner_id,
                generation=generation,
            )

    def has_active_execution(self, key: str) -> bool:
        return any(
            existing_key == key and not task.done()
            for (existing_key, _), task in self._inflight.items()
        )

    @staticmethod
    def _local_usage_owner_finished(agent_ref: Any) -> bool:
        """Return true only when an attached local ledger has no live call."""

        agent = (
            agent_ref[0]
            if isinstance(agent_ref, (list, tuple)) and agent_ref
            else None
        )
        if agent is None:
            return False
        ledgers = []
        seen_ids: set[int] = set()
        for attribute in (
            "_paid_call_usage_ledger",
            "_true_moa_usage_ledger",
        ):
            ledger = getattr(agent, attribute, None)
            if ledger is None or id(ledger) in seen_ids:
                continue
            seen_ids.add(id(ledger))
            ledgers.append(ledger)
        if not ledgers:
            return False
        for ledger in ledgers:
            to_dict = getattr(ledger, "to_dict", None)
            if not callable(to_dict):
                return False
            try:
                snapshot = to_dict()
            except BaseException:
                return False
            calls = (
                snapshot.get("calls")
                if isinstance(snapshot, dict)
                else None
            )
            if not isinstance(calls, list):
                return False
            if any(
                isinstance(call, dict)
                and call.get("status") in {"reserved", "running"}
                for call in calls
            ):
                return False
        return True

    def _sync_usage_drain_owner(self, key: str) -> None:
        if self._durable is None:
            return
        has_running_receipt = self._has_running_usage_receipt(
            self._durable.get(key),
        )
        with self._usage_drains_lock:
            owner_is_closed = key in self._closed_usage_drain_owners
            if has_running_receipt and not owner_is_closed:
                self._usage_drains.add(key)
        if has_running_receipt and not owner_is_closed:
            self._schedule_usage_drain_heartbeat(key)
        else:
            self._release_usage_drain_lease(key)

    def terminalize_orphaned_stopped_usage(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Compatibility wrapper for the general orphan recovery fence."""

        return self.terminalize_orphaned_usage(key)

    def terminalize_orphaned_usage(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Close durable active receipts only when this process has no owner."""

        if self._durable is None:
            return None
        with self._usage_drains_lock:
            if key in self._usage_drains:
                return self._durable.get(key)
        if self.has_active_execution(key):
            return self._durable.get(key)
        try:
            generation = self._claim_usage_drain_lease(key)
        except RuntimeError:
            return self._durable.get(key)
        try:
            self._durable.terminalize_orphaned_running_calls(
                key,
                usage_drain_owner_id=self._usage_drain_owner_id,
                usage_drain_generation=generation,
            )
        finally:
            self._release_usage_drain_lease(key)
        return self._durable.get(key)

    def persist_usage(
        self,
        key: str,
        fingerprint: str,
        ledger: Dict[str, Any],
    ) -> None:
        if self._durable is None:
            raise RuntimeError("true MoA durable ledger is unavailable")
        state = str(ledger.get("status") or "running")
        is_true_moa = (
            ledger.get("schema") == MYSTAND_TRUE_MOA_USAGE_SCHEMA
        )
        durable_state = (
            "stopped"
            if state == "cancelled"
            else "completed"
            if state == "completed" and not is_true_moa
            else "failed"
            if state == "failed"
            else "running"
        )
        with self._usage_drains_lock:
            was_active = key in self._usage_drains
            owner_is_closed = key in self._closed_usage_drain_owners
        if owner_is_closed:
            raise RuntimeError("true MoA usage drain owner is closed")
        # Claim before SQLite exposes any provider callback.  The generation
        # is checked again in the same transaction that persists the ledger.
        generation = self._claim_usage_drain_lease(key)
        try:
            self._durable.save_usage(
                key,
                fingerprint,
                ledger,
                state=durable_state,
                usage_drain_owner_id=self._usage_drain_owner_id,
                usage_drain_generation=generation,
            )
        except BaseException:
            if not was_active:
                self._release_usage_drain_lease(key)
            raise
        # Provider workers outlive the outer asyncio request after a stop.
        # The durable merge is authoritative here: a stale callback cannot
        # re-register ownership after a newer terminal receipt has landed.
        self._sync_usage_drain_owner(key)
