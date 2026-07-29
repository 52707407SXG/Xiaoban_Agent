"""Strict final-executor usage and durable terminal-call accounting."""

from __future__ import annotations

from typing import Any

from agent.paid_call_accounting import (
    finish_paid_provider_call as _finish_paid_provider_call,
)

def finish_true_moa_final_call(
    agent: Any,
    call_id: str | None,
    *,
    status: str,
    response: Any = None,
    error_category: str | None = None,
) -> None:
    """Commit one paid final-executor request to the true-MoA call ledger."""

    if not call_id:
        return
    ledger = getattr(agent, "_true_moa_usage_ledger", None)
    if ledger is None:
        return
    _finish_paid_provider_call(
        agent,
        ledger,
        call_id,
        status=status,
        response=response,
        error_category=error_category,
    )
