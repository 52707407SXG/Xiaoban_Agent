"""Durable usage settlement and stop-state transitions."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from xiaoban.trusted_runtime.true_moa_durable_shared import (
    TRUE_MOA_DURABLE_MAX_ROWS,
    _DURABLE_STATE_RANK,
    _DURABLE_TERMINAL_STATES,
    _VALID_STATES,
    _durable_max_rows,
    _safe_text,
    _storage_key,
)
from xiaoban.trusted_runtime.true_moa_durable_usage import (
    _merge_status,
    _merge_usage,
    project_durable_usage,
)


class _TrueMoAAccountingMixin:
    def save_usage(
        self,
        key: str,
        fingerprint: str,
        usage: Mapping[str, Any],
        *,
        state: str,
        usage_drain_owner_id: str = "",
        usage_drain_generation: int = 0,
    ) -> None:
        if state not in _VALID_STATES:
            raise ValueError("invalid true MoA durable state")
        projected = project_durable_usage(usage)
        storage_key = _storage_key(key)
        clean_fingerprint = _safe_text(fingerprint, required=True)
        clean_lease_owner = (
            _safe_text(usage_drain_owner_id, required=True)
            if usage_drain_owner_id
            else ""
        )
        clean_lease_generation = int(usage_drain_generation)
        if bool(clean_lease_owner) != (clean_lease_generation > 0):
            raise ValueError("invalid usage drain lease fence")
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if clean_lease_owner:
                self._assert_usage_drain_lease_row(
                    connection,
                    storage_key=storage_key,
                    owner_id=clean_lease_owner,
                    generation=clean_lease_generation,
                    now_ms=timestamp,
                )
            row = connection.execute(
                """
                SELECT fingerprint, kind, state, usage_json
                FROM true_moa_idempotency
                WHERE storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RuntimeError("true MoA durable execution was not claimed")
            if (
                row["kind"] != "execution"
                or row["fingerprint"] != clean_fingerprint
            ):
                connection.rollback()
                raise RuntimeError("true MoA durable execution binding conflict")
            existing_usage = (
                project_durable_usage(json.loads(row["usage_json"]))
                if row["usage_json"]
                else None
            )
            allow_restart_late_accounting = bool(
                str(row["state"] or "") == "failed"
                and isinstance(existing_usage, Mapping)
                and any(
                    isinstance(call, Mapping)
                    and call.get("status") == "timed_out"
                    and call.get("errorCategory")
                    == "agent_restart_outcome_unknown"
                    for call in (existing_usage.get("calls") or ())
                )
            )
            projected = _merge_usage(
                existing_usage,
                projected,
                allow_stopped_late_accounting=(
                    str(row["state"] or "") == "stopped"
                ),
                allow_restart_late_accounting=(
                    allow_restart_late_accounting
                ),
            )
            if str(row["state"] or "") == "stopped":
                # A late provider callback may fill exact token/cost fields,
                # but it cannot rewrite the user-visible stop decision into a
                # completed execution.  Individual calls retain their physical
                # outcome while the delivery remains cancelled.
                projected["status"] = "cancelled"
                projected = project_durable_usage(projected)
            encoded = json.dumps(
                projected,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            merged_state = _merge_status(
                str(row["state"] or ""),
                state,
                ranks=_DURABLE_STATE_RANK,
                terminals=_DURABLE_TERMINAL_STATES,
                stopped_wins=True,
                interrupted_is_provisional=True,
            )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = ?, usage_json = ?, updated_at_ms = ?
                WHERE storage_key = ?
                """,
                (merged_state, encoded, timestamp, storage_key),
            )
            connection.commit()
        self._harden_files()

    def set_state(self, key: str, *, state: str) -> None:
        if state not in _VALID_STATES:
            raise ValueError("invalid true MoA durable state")
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return
            merged_state = _merge_status(
                str(row["state"] or ""),
                state,
                ranks=_DURABLE_STATE_RANK,
                terminals=_DURABLE_TERMINAL_STATES,
                stopped_wins=True,
                interrupted_is_provisional=True,
            )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = ?, updated_at_ms = ?
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (merged_state, timestamp, storage_key),
            )
            connection.commit()
        self._harden_files()

    def mark_stopped(self, key: str) -> bool:
        """Atomically install a stop fence unless a non-stoppable state won."""

        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT kind, state
                FROM true_moa_idempotency
                WHERE storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                row_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM true_moa_idempotency"
                    ).fetchone()[0]
                )
                if row_count >= _durable_max_rows():
                    connection.rollback()
                    raise RuntimeError(
                        "true MoA durable ledger capacity exhausted"
                    )
                connection.execute(
                    """
                    INSERT INTO true_moa_idempotency (
                        storage_key, fingerprint, kind, state,
                        usage_json, created_at_ms, updated_at_ms
                    ) VALUES (?, '', 'execution', 'stopped', '', ?, ?)
                    """,
                    (storage_key, timestamp, timestamp),
                )
                accepted = True
            elif (
                row["kind"] == "execution"
                and row["state"]
                in {"claimed", "running", "interrupted", "stopped"}
            ):
                connection.execute(
                    """
                    UPDATE true_moa_idempotency
                    SET state = 'stopped', updated_at_ms = ?
                    WHERE storage_key = ?
                    """,
                    (timestamp, storage_key),
                )
                accepted = True
            else:
                accepted = False
            connection.commit()
        self._harden_files()
        return accepted

    def terminalize_stopped_running_calls(
        self,
        key: str,
        *,
        usage_drain_owner_id: str = "",
        usage_drain_generation: int = 0,
    ) -> bool:
        """Fence orphaned receipts after a durable stop."""

        return self._terminalize_orphaned_running_calls(
            key,
            allowed_states={"stopped"},
            usage_drain_owner_id=usage_drain_owner_id,
            usage_drain_generation=usage_drain_generation,
        )

    def terminalize_orphaned_running_calls(
        self,
        key: str,
        *,
        usage_drain_owner_id: str = "",
        usage_drain_generation: int = 0,
    ) -> bool:
        """Fence orphaned receipts after either stop or process restart."""

        return self._terminalize_orphaned_running_calls(
            key,
            allowed_states={"running", "stopped", "interrupted", "failed"},
            usage_drain_owner_id=usage_drain_owner_id,
            usage_drain_generation=usage_drain_generation,
        )

    def _terminalize_orphaned_running_calls(
        self,
        key: str,
        *,
        allowed_states: set[str],
        usage_drain_owner_id: str = "",
        usage_drain_generation: int = 0,
    ) -> bool:
        """Atomically close active receipts whose process owner is gone.

        A live process owns its in-memory usage drain.  After restart no such
        worker exists, so leaving a durable call as ``running`` would strand
        My Stand forever.  This transition deliberately leaves token and cost
        fields empty.  The caller must prove there is no live process owner.
        """

        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        clean_lease_owner = (
            _safe_text(usage_drain_owner_id, required=True)
            if usage_drain_owner_id
            else ""
        )
        clean_lease_generation = int(usage_drain_generation)
        if bool(clean_lease_owner) != (clean_lease_generation > 0):
            raise ValueError("invalid usage drain lease fence")
        changed = False
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if clean_lease_owner:
                self._assert_usage_drain_lease_row(
                    connection,
                    storage_key=storage_key,
                    owner_id=clean_lease_owner,
                    generation=clean_lease_generation,
                    now_ms=timestamp,
                )
            row = connection.execute(
                """
                SELECT state, usage_json
                FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            current_state = (
                str(row["state"] or "")
                if row is not None
                else ""
            )
            if (
                row is None
                or current_state not in allowed_states
                or not row["usage_json"]
            ):
                connection.commit()
                return False
            stopped = current_state == "stopped"
            terminal_state = "stopped" if stopped else "failed"
            terminal_usage_status = "cancelled" if stopped else "failed"
            terminal_slot_status = "cancelled" if stopped else "failed"
            usage = project_durable_usage(
                json.loads(row["usage_json"])
            )
            if (
                not stopped
                and usage.get("status") == "completed"
            ):
                connection.commit()
                return False
            for slot in usage.get("slots", []):
                if slot["status"] == "running":
                    slot["status"] = terminal_slot_status
                    slot["endedAtMs"] = max(
                        int(slot.get("startedAtMs") or 0),
                        timestamp,
                    )
                    slot["errorCategory"] = (
                        "agent_restart_outcome_unknown"
                    )
                    changed = True
            for call in usage["calls"]:
                if call["status"] == "reserved":
                    call["status"] = "not_dispatched"
                    call["endedAtMs"] = max(
                        int(call.get("startedAtMs") or 0),
                        timestamp,
                    )
                    call["errorCategory"] = (
                        "provider_dispatch_fence_closed"
                    )
                    changed = True
                elif call["status"] == "running":
                    call["status"] = "timed_out"
                    call["endedAtMs"] = max(
                        int(call.get("startedAtMs") or 0),
                        timestamp,
                    )
                    call["errorCategory"] = (
                        "agent_restart_outcome_unknown"
                    )
                    changed = True
            if (
                not stopped
                and usage.get("status")
                not in {"completed", "failed", "cancelled"}
            ):
                changed = True
            if changed:
                usage["status"] = terminal_usage_status
                encoded = json.dumps(
                    project_durable_usage(usage),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    """
                    UPDATE true_moa_idempotency
                    SET state = ?, usage_json = ?, updated_at_ms = ?
                    WHERE storage_key = ? AND state = ?
                    """,
                    (
                        terminal_state,
                        encoded,
                        timestamp,
                        storage_key,
                        current_state,
                    ),
                )
            connection.commit()
        if changed:
            self._harden_files()
        return changed
