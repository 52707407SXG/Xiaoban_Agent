"""Fail-closed identity checks for signed My Stand durable completions."""

from __future__ import annotations

import re


_DELIVERY_ATTEMPT_RE = re.compile(r"[1-9][0-9]{0,8}")


def normal_durable_identity_error(
    *,
    idempotency_key: str,
    delivery_id: str,
    attempt: str,
    delivery_attempt: str,
) -> tuple[str, str] | None:
    """Return a stable public error when one normal delivery has two identities."""

    normalized_key = str(idempotency_key or "").strip()
    normalized_delivery = str(delivery_id or "").strip()
    if normalized_key and normalized_key != normalized_delivery:
        return (
            "mystand_idempotency_identity_conflict",
            "Durable My Stand idempotency identity must match its delivery identity",
        )

    normalized_attempt = str(attempt or "").strip()
    normalized_delivery_attempt = str(delivery_attempt or "").strip()
    if (
        not _DELIVERY_ATTEMPT_RE.fullmatch(normalized_attempt)
        or normalized_attempt != normalized_delivery_attempt
    ):
        return (
            "mystand_delivery_attempt_conflict",
            "Durable My Stand attempt identities must match",
        )
    return None


__all__ = ["normal_durable_identity_error"]
