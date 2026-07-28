"""Strict final-executor usage and durable terminal-call accounting."""

from __future__ import annotations

from typing import Any

from agent.usage_pricing import estimate_usage_cost, normalize_usage


def record_strict_terminal_usage(agent: Any, response: Any) -> None:
    """Keep usage for a strict no-retry response that exits before normal accounting."""

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        return
    canonical = normalize_usage(
        raw_usage,
        provider=agent.provider,
        api_mode=agent.api_mode,
    )
    agent.session_prompt_tokens += canonical.prompt_tokens
    agent.session_completion_tokens += canonical.output_tokens
    agent.session_total_tokens += canonical.total_tokens
    agent.session_input_tokens += canonical.input_tokens
    agent.session_output_tokens += canonical.output_tokens
    agent.session_cache_read_tokens += canonical.cache_read_tokens
    agent.session_cache_write_tokens += canonical.cache_write_tokens
    agent.session_reasoning_tokens += canonical.reasoning_tokens
    agent.session_api_calls += 1
    cost_result = estimate_usage_cost(
        agent.model,
        canonical,
        provider=agent.provider,
        base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    if cost_result.amount_usd is not None:
        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source


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
    raw_usage = getattr(response, "usage", None)
    usage = None
    cost_usd = None
    cost_status = None
    cost_source = None
    if raw_usage is not None:
        canonical = normalize_usage(
            raw_usage,
            provider=agent.provider,
            api_mode=agent.api_mode,
        )
        cost_result = estimate_usage_cost(
            agent.model,
            canonical,
            provider=agent.provider,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
        )
        # Preserve the raw provider usage object for the true-MoA ledger so its
        # cache split is derived only from trusted provider counters.  The
        # ledger projects numeric fields only; no raw usage object is persisted.
        usage = raw_usage
        cost_usd = cost_result.amount_usd
        cost_status = cost_result.status
        cost_source = cost_result.source
    ledger.finish_final_call(
        call_id,
        status=status,
        usage=usage,
        error_category=error_category,
        cost_usd=cost_usd,
        cost_status=cost_status,
        cost_source=cost_source,
        notify=False,
    )
    ledger.notify_change_async()
