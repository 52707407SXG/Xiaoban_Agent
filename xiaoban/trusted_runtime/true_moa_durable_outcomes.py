"""Sealed completed-outcome persistence, recovery, and owner ACK."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Mapping

from xiaoban.trusted_runtime.true_moa_durable_shared import (
    TrueMoAOutcomeBindingError,
    TrueMoAOutcomeUnavailableError,
    _DURABLE_STATE_RANK,
    _DURABLE_TERMINAL_STATES,
    _OUTCOME_DIGEST,
    _safe_text,
    _storage_key,
    _true_moa_outcome_aad,
    project_true_moa_completed_outcome,
    project_true_moa_outcome_binding,
)
from xiaoban.trusted_runtime.true_moa_durable_usage import (
    _merge_status,
    _merge_usage,
    project_true_moa_usage,
)


class _TrueMoAOutcomeMixin:
    def save_completed_outcome(
        self,
        key: str,
        fingerprint: str,
        usage: Mapping[str, Any],
        outcome: Mapping[str, Any],
        *,
        binding: Mapping[str, Any],
    ) -> str:
        """Atomically commit terminal usage and one encrypted visible result."""

        projected_usage = project_true_moa_usage(usage)
        if projected_usage["status"] != "completed":
            raise ValueError("true MoA outcome requires completed usage")
        projected_binding = project_true_moa_outcome_binding(binding)
        projected_outcome = project_true_moa_completed_outcome(
            outcome,
            binding=projected_binding,
        )
        storage_key = _storage_key(key)
        clean_fingerprint = _safe_text(fingerprint, required=True)
        timestamp = int(time.time() * 1000)
        expires_at_ms = timestamp + (self._outcome_ttl_seconds * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
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
                raise RuntimeError(
                    "true MoA durable execution binding conflict"
                )
            merged_state = _merge_status(
                str(row["state"] or ""),
                "completed",
                ranks=_DURABLE_STATE_RANK,
                terminals=_DURABLE_TERMINAL_STATES,
                stopped_wins=True,
                interrupted_is_provisional=True,
            )
            if merged_state != "completed":
                connection.rollback()
                raise TrueMoAOutcomeBindingError(
                    "true MoA terminal fence rejected completed outcome"
                )
            existing_usage = (
                project_true_moa_usage(json.loads(row["usage_json"]))
                if row["usage_json"]
                else None
            )
            merged_usage = _merge_usage(existing_usage, projected_usage)
            existing_outcome_state = str(row["outcome_state"] or "none")
            if existing_outcome_state == "sealed":
                existing_outcome, existing_receipt = self._decrypt_outcome_row(
                    row,
                    storage_key=storage_key,
                    usage=existing_usage,
                    binding=projected_binding,
                )
                if existing_outcome != projected_outcome:
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "conflicting true MoA completed outcome"
                    )
                if merged_usage != existing_usage:
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "sealed true MoA outcome usage cannot change"
                    )
                encoded = json.dumps(
                    existing_usage,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    """
                    UPDATE true_moa_idempotency
                    SET state = 'completed', usage_json = ?, updated_at_ms = ?
                    WHERE storage_key = ?
                    """,
                    (encoded, timestamp, storage_key),
                )
                connection.commit()
                self._harden_files()
                return existing_receipt
            if existing_outcome_state != "none":
                connection.rollback()
                raise TrueMoAOutcomeBindingError(
                    "true MoA completed outcome is no longer writable"
                )
            key_id, nonce, ciphertext, receipt = self._encrypt_outcome(
                storage_key=storage_key,
                fingerprint=clean_fingerprint,
                usage=merged_usage,
                outcome=projected_outcome,
                binding=projected_binding,
            )
            binding_digest = hashlib.sha256(
                _true_moa_outcome_aad(
                    storage_key=storage_key,
                    fingerprint=clean_fingerprint,
                    usage=merged_usage,
                    binding=projected_binding,
                )
            ).hexdigest()
            encoded = json.dumps(
                merged_usage,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = 'completed',
                    usage_json = ?,
                    outcome_state = 'sealed',
                    outcome_key_id = ?,
                    outcome_nonce = ?,
                    outcome_ciphertext = ?,
                    outcome_receipt = ?,
                    outcome_binding_digest = ?,
                    outcome_expires_at_ms = ?,
                    outcome_acked_at_ms = 0,
                    updated_at_ms = ?
                WHERE storage_key = ?
                """,
                (
                    encoded,
                    key_id,
                    sqlite3.Binary(nonce),
                    sqlite3.Binary(ciphertext),
                    receipt,
                    binding_digest,
                    expires_at_ms,
                    timestamp,
                    storage_key,
                ),
            )
            connection.commit()
        self._harden_files()
        return receipt

    def recover_completed_outcome(
        self,
        key: str,
        *,
        binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Decrypt one owner/snapshot-bound outcome without exposing keys."""

        projected_binding = project_true_moa_outcome_binding(binding)
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            if (
                str(row["state"] or "") != "completed"
                or not row["usage_json"]
            ):
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            usage = project_true_moa_usage(json.loads(row["usage_json"]))
            outcome, receipt = self._decrypt_outcome_row(
                row,
                storage_key=storage_key,
                usage=usage,
                binding=projected_binding,
            )
            connection.commit()
        return {
            **outcome,
            "outcomeId": receipt,
            # This is an operational acknowledgment deadline, never an
            # automatic deletion fence.  An unacknowledged paid result stays
            # recoverable until the owner-bound ACK or a reviewed admin action.
            "retentionOverdue": bool(
                int(row["outcome_expires_at_ms"] or 0) <= timestamp
            ),
        }

    def acknowledge_completed_outcome(
        self,
        key: str,
        *,
        binding: Mapping[str, Any],
        outcome_id: str,
    ) -> str:
        """Clear ciphertext only after an authenticated owner acknowledgment."""

        projected_binding = project_true_moa_outcome_binding(binding)
        clean_outcome_id = str(outcome_id or "").lower()
        if not _OUTCOME_DIGEST.fullmatch(clean_outcome_id):
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA outcome acknowledgment"
            )
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM true_moa_idempotency
                WHERE storage_key = ? AND kind = 'execution'
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            state = str(row["outcome_state"] or "none")
            receipt = str(row["outcome_receipt"] or "")
            if state == "acked":
                if (
                    str(row["state"] or "") != "completed"
                    or not row["usage_json"]
                    or receipt != clean_outcome_id
                ):
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "conflicting true MoA outcome acknowledgment"
                    )
                usage = project_true_moa_usage(
                    json.loads(row["usage_json"])
                )
                aad = _true_moa_outcome_aad(
                    storage_key=storage_key,
                    fingerprint=str(row["fingerprint"] or ""),
                    usage=usage,
                    binding=projected_binding,
                )
                if (
                    str(row["outcome_binding_digest"] or "")
                    != hashlib.sha256(aad).hexdigest()
                ):
                    connection.rollback()
                    raise TrueMoAOutcomeBindingError(
                        "conflicting true MoA outcome acknowledgment"
                    )
                connection.commit()
                return "already_acknowledged"
            if (
                state != "sealed"
                or str(row["state"] or "") != "completed"
                or not row["usage_json"]
            ):
                connection.rollback()
                raise TrueMoAOutcomeUnavailableError(
                    "true MoA completed outcome is unavailable"
                )
            usage = project_true_moa_usage(json.loads(row["usage_json"]))
            _outcome, verified_receipt = self._decrypt_outcome_row(
                row,
                storage_key=storage_key,
                usage=usage,
                binding=projected_binding,
            )
            if verified_receipt != clean_outcome_id:
                connection.rollback()
                raise TrueMoAOutcomeBindingError(
                    "conflicting true MoA outcome acknowledgment"
                )
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET outcome_state = 'acked',
                    outcome_key_id = '',
                    outcome_nonce = X'',
                    outcome_ciphertext = X'',
                    outcome_expires_at_ms = 0,
                    outcome_acked_at_ms = ?,
                    updated_at_ms = ?
                WHERE storage_key = ? AND outcome_state = 'sealed'
                """,
                (timestamp, timestamp, storage_key),
            )
            connection.commit()
        self._harden_files()
        return "acknowledged"
