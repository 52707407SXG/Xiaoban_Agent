"""Durable owner lease and fencing token for provider-usage drains."""

from __future__ import annotations

import time

from xiaoban.trusted_runtime.true_moa_durable_shared import (
    _safe_text,
    _storage_key,
)


class _TrueMoAUsageDrainLeaseMixin:
    def claim_usage_drain_lease(
        self,
        key: str,
        *,
        owner_id: str,
        lease_ms: int,
        now_ms: int | None = None,
    ) -> int | None:
        """Claim or renew one lease and return its fencing generation."""

        owner = _safe_text(owner_id, required=True)
        duration = int(lease_ms)
        if duration < 100:
            raise ValueError("usage drain lease is too short")
        timestamp = (
            int(now_ms)
            if now_ms is not None
            else int(time.time() * 1000)
        )
        storage_key = _storage_key(key)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT owner_id, generation, lease_until_ms
                FROM true_moa_usage_drain_leases
                WHERE storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
            if row is None:
                generation = 1
                connection.execute(
                    """
                    INSERT INTO true_moa_usage_drain_leases (
                        storage_key, owner_id, generation,
                        lease_until_ms, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        storage_key,
                        owner,
                        generation,
                        timestamp + duration,
                        timestamp,
                    ),
                )
            elif int(row["lease_until_ms"] or 0) > timestamp:
                if str(row["owner_id"] or "") != owner:
                    connection.rollback()
                    return None
                generation = int(row["generation"] or 0)
                connection.execute(
                    """
                    UPDATE true_moa_usage_drain_leases
                    SET lease_until_ms = ?, updated_at_ms = ?
                    WHERE storage_key = ?
                      AND owner_id = ? AND generation = ?
                    """,
                    (
                        timestamp + duration,
                        timestamp,
                        storage_key,
                        owner,
                        generation,
                    ),
                )
            else:
                generation = int(row["generation"] or 0) + 1
                connection.execute(
                    """
                    UPDATE true_moa_usage_drain_leases
                    SET owner_id = ?, generation = ?,
                        lease_until_ms = ?, updated_at_ms = ?
                    WHERE storage_key = ?
                    """,
                    (
                        owner,
                        generation,
                        timestamp + duration,
                        timestamp,
                        storage_key,
                    ),
                )
            connection.commit()
        self._harden_files()
        return generation

    def renew_usage_drain_lease(
        self,
        key: str,
        *,
        owner_id: str,
        generation: int,
        lease_ms: int,
        now_ms: int | None = None,
    ) -> bool:
        """Heartbeat only the still-current, unexpired generation."""

        owner = _safe_text(owner_id, required=True)
        clean_generation = int(generation)
        duration = int(lease_ms)
        if clean_generation < 1 or duration < 100:
            raise ValueError("invalid usage drain lease heartbeat")
        timestamp = (
            int(now_ms)
            if now_ms is not None
            else int(time.time() * 1000)
        )
        storage_key = _storage_key(key)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE true_moa_usage_drain_leases
                SET lease_until_ms = ?, updated_at_ms = ?
                WHERE storage_key = ?
                  AND owner_id = ?
                  AND generation = ?
                  AND lease_until_ms > ?
                """,
                (
                    timestamp + duration,
                    timestamp,
                    storage_key,
                    owner,
                    clean_generation,
                    timestamp,
                ),
            )
            renewed = cursor.rowcount == 1
            connection.commit()
        if renewed:
            self._harden_files()
        return renewed

    def release_usage_drain_lease(
        self,
        key: str,
        *,
        owner_id: str,
        generation: int,
        now_ms: int | None = None,
    ) -> bool:
        """Expire one generation without allowing generation reuse."""

        owner = _safe_text(owner_id, required=True)
        clean_generation = int(generation)
        if clean_generation < 1:
            raise ValueError("invalid usage drain lease release")
        timestamp = (
            int(now_ms)
            if now_ms is not None
            else int(time.time() * 1000)
        )
        storage_key = _storage_key(key)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE true_moa_usage_drain_leases
                SET lease_until_ms = 0, updated_at_ms = ?
                WHERE storage_key = ?
                  AND owner_id = ? AND generation = ?
                """,
                (timestamp, storage_key, owner, clean_generation),
            )
            released = cursor.rowcount == 1
            connection.commit()
        if released:
            self._harden_files()
        return released

    def usage_drain_lease(self, key: str) -> dict[str, int | str] | None:
        storage_key = _storage_key(key)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT owner_id, generation, lease_until_ms, updated_at_ms
                FROM true_moa_usage_drain_leases
                WHERE storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "ownerId": str(row["owner_id"] or ""),
            "generation": int(row["generation"] or 0),
            "leaseUntilMs": int(row["lease_until_ms"] or 0),
            "updatedAtMs": int(row["updated_at_ms"] or 0),
        }

    @staticmethod
    def _assert_usage_drain_lease_row(
        connection,
        *,
        storage_key: str,
        owner_id: str,
        generation: int,
        now_ms: int,
    ) -> None:
        row = connection.execute(
            """
            SELECT owner_id, generation, lease_until_ms
            FROM true_moa_usage_drain_leases
            WHERE storage_key = ?
            """,
            (storage_key,),
        ).fetchone()
        if (
            row is None
            or str(row["owner_id"] or "") != owner_id
            or int(row["generation"] or 0) != generation
            or int(row["lease_until_ms"] or 0) <= now_ms
        ):
            raise RuntimeError("usage drain lease fence rejected stale owner")
