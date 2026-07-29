"""Owned SQLite connection, sealing primitives, and idempotency claims."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from xiaoban.trusted_runtime.true_moa_durable_shared import (
    TRUE_MOA_DURABLE_MAX_ROWS,
    TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS,
    TRUE_MOA_OUTCOME_MAX_PLAINTEXT_BYTES,
    TrueMoAOutcomeBindingError,
    TrueMoAOutcomeUnavailableError,
    _OUTCOME_DIGEST,
    _OUTCOME_KEY_ID,
    _VALID_KINDS,
    _canonical_json_bytes,
    _durable_max_rows,
    _safe_text,
    _storage_key,
    _true_moa_outcome_aad,
    _validated_outcome_keyring,
    _validated_outcome_ttl,
    project_true_moa_completed_outcome,
    project_true_moa_outcome_binding,
)
from xiaoban.trusted_runtime.true_moa_durable_usage import (
    project_durable_usage,
)

class _TrueMoADurableBase:
    """SQLite usage ledger with a separately-keyed sealed outcome envelope."""

    def __init__(
        self,
        path: str,
        *,
        outcome_keys: Mapping[str, bytes] | None = None,
        active_outcome_key_id: str | None = None,
        outcome_ttl_seconds: int | None = None,
    ):
        raw_path = str(path or "").strip()
        if not raw_path:
            raise ValueError("true MoA durable ledger path is required")
        self.path = Path(raw_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._lock_path = Path(f"{self.path}.lock")
        self._lock_handle = self._lock_path.open("a+b")
        self._lock_path.chmod(0o600)
        self._outcome_key_error: Exception | None = None
        try:
            self._outcome_keys = _validated_outcome_keyring(outcome_keys)
            self._outcome_ttl_seconds = _validated_outcome_ttl(
                outcome_ttl_seconds
            )
            active_key_id = str(active_outcome_key_id or "")
            if active_key_id:
                if active_key_id not in self._outcome_keys:
                    raise ValueError("invalid true MoA active outcome key")
                self._active_outcome_key_id = active_key_id
            else:
                self._active_outcome_key_id = next(
                    iter(self._outcome_keys),
                    "",
                )
        except ValueError as exc:
            # Usage receipts remain available, but true-MoA preflight checks
            # outcome_ready and fails before a paid dispatch.
            self._outcome_keys = {}
            self._active_outcome_key_id = ""
            self._outcome_ttl_seconds = TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS
            self._outcome_key_error = exc
        try:
            fcntl.flock(
                self._lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as exc:
            self._lock_handle.close()
            raise RuntimeError(
                "true MoA durable ledger is already owned by another process"
            ) from exc
        try:
            self._initialize()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        handle = getattr(self, "_lock_handle", None)
        if handle is None or handle.closed:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    @property
    def outcome_ready(self) -> bool:
        return bool(
            self._active_outcome_key_id
            and self._active_outcome_key_id in self._outcome_keys
            and self._outcome_key_error is None
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _outcome_key(self, key_id: str, storage_key: str) -> bytes:
        master_key = self._outcome_keys.get(str(key_id or ""))
        if master_key is None:
            raise TrueMoAOutcomeUnavailableError(
                "true MoA outcome key is unavailable"
            )
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=bytes.fromhex(storage_key),
            info=(
                b"mystand.true-moa.completed-outcome.v1\0"
                + str(key_id).encode("ascii")
            ),
        ).derive(master_key)

    def _encrypt_outcome(
        self,
        *,
        storage_key: str,
        fingerprint: str,
        usage: Mapping[str, Any],
        outcome: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> tuple[str, bytes, bytes, str]:
        if not self.outcome_ready:
            raise TrueMoAOutcomeUnavailableError(
                "true MoA outcome key is unavailable"
            )
        projected_binding = project_true_moa_outcome_binding(binding)
        projected = project_true_moa_completed_outcome(
            outcome,
            binding=projected_binding,
        )
        plaintext = _canonical_json_bytes(projected)
        aad = _true_moa_outcome_aad(
            storage_key=storage_key,
            fingerprint=fingerprint,
            usage=usage,
            binding=projected_binding,
        )
        key_id = self._active_outcome_key_id
        nonce = os.urandom(12)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        ciphertext = AESGCM(
            self._outcome_key(key_id, storage_key)
        ).encrypt(nonce, plaintext, aad)
        receipt = hashlib.sha256(
            key_id.encode("ascii") + nonce + ciphertext
        ).hexdigest()
        return key_id, nonce, ciphertext, receipt

    def _decrypt_outcome_row(
        self,
        row: sqlite3.Row,
        *,
        storage_key: str,
        usage: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if str(row["outcome_state"] or "") != "sealed":
            raise TrueMoAOutcomeUnavailableError(
                "true MoA completed outcome is unavailable"
            )
        key_id = str(row["outcome_key_id"] or "")
        nonce = bytes(row["outcome_nonce"] or b"")
        ciphertext = bytes(row["outcome_ciphertext"] or b"")
        receipt = str(row["outcome_receipt"] or "")
        if (
            not _OUTCOME_KEY_ID.fullmatch(key_id)
            or len(nonce) != 12
            or len(ciphertext) < 16
            or not _OUTCOME_DIGEST.fullmatch(receipt)
            or receipt
            != hashlib.sha256(
                key_id.encode("ascii") + nonce + ciphertext
            ).hexdigest()
        ):
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA sealed outcome envelope"
            )
        aad = _true_moa_outcome_aad(
            storage_key=storage_key,
            fingerprint=str(row["fingerprint"] or ""),
            usage=usage,
            binding=binding,
        )
        binding_digest = str(row["outcome_binding_digest"] or "")
        if (
            not _OUTCOME_DIGEST.fullmatch(binding_digest)
            or binding_digest != hashlib.sha256(aad).hexdigest()
        ):
            raise TrueMoAOutcomeBindingError(
                "true MoA sealed outcome binding mismatch"
            )
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        try:
            plaintext = AESGCM(
                self._outcome_key(key_id, storage_key)
            ).decrypt(nonce, ciphertext, aad)
        except InvalidTag as exc:
            raise TrueMoAOutcomeBindingError(
                "true MoA sealed outcome authentication failed"
            ) from exc
        if len(plaintext) > TRUE_MOA_OUTCOME_MAX_PLAINTEXT_BYTES:
            raise TrueMoAOutcomeBindingError(
                "true MoA sealed outcome is too large"
            )
        try:
            decoded = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA sealed outcome payload"
            ) from exc
        try:
            projected = project_true_moa_completed_outcome(
                decoded,
                binding=binding,
            )
        except ValueError as exc:
            raise TrueMoAOutcomeBindingError(
                "invalid true MoA sealed outcome payload"
            ) from exc
        return projected, receipt

    def _harden_files(self) -> None:
        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            try:
                if path.exists():
                    path.chmod(0o600)
            except OSError:
                pass

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS true_moa_idempotency (
                    storage_key TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    usage_json TEXT NOT NULL DEFAULT '',
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_true_moa_idem_updated
                    ON true_moa_idempotency(updated_at_ms);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(true_moa_idempotency)"
                )
            }
            migrations = {
                "outcome_state": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_state TEXT NOT NULL DEFAULT 'none'"
                ),
                "outcome_key_id": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_key_id TEXT NOT NULL DEFAULT ''"
                ),
                "outcome_nonce": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_nonce BLOB NOT NULL DEFAULT X''"
                ),
                "outcome_ciphertext": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_ciphertext BLOB NOT NULL DEFAULT X''"
                ),
                "outcome_receipt": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_receipt TEXT NOT NULL DEFAULT ''"
                ),
                "outcome_binding_digest": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_binding_digest TEXT NOT NULL DEFAULT ''"
                ),
                "outcome_expires_at_ms": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_expires_at_ms INTEGER NOT NULL DEFAULT 0"
                ),
                "outcome_acked_at_ms": (
                    "ALTER TABLE true_moa_idempotency "
                    "ADD COLUMN outcome_acked_at_ms INTEGER NOT NULL DEFAULT 0"
                ),
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET state = 'interrupted', updated_at_ms = ?
                WHERE kind = 'execution'
                  AND state IN ('claimed', 'running')
                """,
                (int(time.time() * 1000),),
            )
        self._harden_files()

    def get(self, key: str) -> dict[str, Any] | None:
        storage_key = _storage_key(key)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM true_moa_idempotency WHERE storage_key = ?",
                (storage_key,),
            ).fetchone()
        if row is None:
            return None
        usage = None
        if row["usage_json"]:
            usage = project_durable_usage(json.loads(row["usage_json"]))
        return {
            "fingerprint": row["fingerprint"],
            "kind": row["kind"],
            "state": row["state"],
            "usage": usage,
            "outcomeState": str(row["outcome_state"] or "none"),
            "outcomeExpiresAtMs": int(row["outcome_expires_at_ms"] or 0),
        }

    def claim(self, key: str, fingerprint: str, *, kind: str) -> str:
        clean_fingerprint = _safe_text(fingerprint, required=True)
        if kind not in _VALID_KINDS:
            raise ValueError("invalid true MoA durable claim kind")
        storage_key = _storage_key(key)
        timestamp = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT fingerprint, kind FROM true_moa_idempotency WHERE storage_key = ?",
                (storage_key,),
            ).fetchone()
            if row is not None:
                existing_fingerprint = str(row["fingerprint"] or "")
                if existing_fingerprint and existing_fingerprint != clean_fingerprint:
                    connection.rollback()
                    return "conflict"
                if row["kind"] != kind:
                    connection.rollback()
                    return "conflict"
                if not existing_fingerprint:
                    connection.execute(
                        """
                        UPDATE true_moa_idempotency
                        SET fingerprint = ?, updated_at_ms = ?
                        WHERE storage_key = ?
                        """,
                        (clean_fingerprint, timestamp, storage_key),
                    )
                connection.commit()
                self._harden_files()
                return "reusable"
            row_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM true_moa_idempotency"
                ).fetchone()[0]
            )
            if row_count >= _durable_max_rows():
                connection.rollback()
                raise RuntimeError("true MoA durable ledger capacity exhausted")
            connection.execute(
                """
                INSERT INTO true_moa_idempotency (
                    storage_key, fingerprint, kind, state,
                    usage_json, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, 'claimed', '', ?, ?)
                """,
                (
                    storage_key,
                    clean_fingerprint,
                    kind,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        self._harden_files()
        return "missing"
