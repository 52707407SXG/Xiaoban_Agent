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
    project_true_moa_usage,
)


class _TrueMoAAccountingMixin:
    def save_usage(
        self,
        key: str,
        fingerprint: str,
        usage: Mapping[str, Any],
        *,
        state: str,
    ) -> None:
        if state not in _VALID_STATES:
            raise ValueError("invalid true MoA durable state")
        projected = project_true_moa_usage(usage)
        storage_key = _storage_key(key)
        clean_fingerprint = _safe_text(fingerprint, required=True)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                project_true_moa_usage(json.loads(row["usage_json"]))
                if row["usage_json"]
                else None
            )
            projected = _merge_usage(
                existing_usage,
                projected,
                allow_stopped_late_accounting=(
                    str(row["state"] or "") == "stopped"
                ),
            )
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

    def terminalize_stopped_running_calls(self, key: str) -> bool:
        """Fence orphaned running receipts after a stopped-process restart.

        A live process owns its in-memory usage drain.  After restart no such
        worker exists, so leaving a durable call as ``running`` would strand
        My Stand in ``stop_requested`` forever.  This transition deliberately
        leaves token, cost, end-time, and error fields empty: a later exact
        provider receipt may still fill them monotonically, while missing
        usage remains settlement-blocked instead of guessed or released.
        """

        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        changed = False
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state, usage_json
                FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            if (
                row is None
                or str(row["state"] or "") != "stopped"
                or not row["usage_json"]
            ):
                connection.commit()
                return False
            usage = project_true_moa_usage(
                json.loads(row["usage_json"])
            )
            for slot in usage["slots"]:
                if slot["status"] == "running":
                    slot["status"] = "cancelled"
                    changed = True
            for call in usage["calls"]:
                if call["status"] == "running":
                    call["status"] = "timed_out"
                    changed = True
            if changed:
                usage["status"] = "cancelled"
                encoded = json.dumps(
                    project_true_moa_usage(usage),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    """
                    UPDATE true_moa_idempotency
                    SET usage_json = ?, updated_at_ms = ?
                    WHERE storage_key = ? AND state = 'stopped'
                    """,
                    (encoded, timestamp, storage_key),
                )
            connection.commit()
        if changed:
            self._harden_files()
        return changed
