"""Topology-neutral provider response accounting for fixed paid calls."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from agent.usage_pricing import estimate_usage_cost, normalize_usage


def _usage_with_attributes(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return SimpleNamespace(
        **{
            str(key): _usage_with_attributes(item)
            for key, item in value.items()
        }
    )


def record_strict_terminal_usage(agent: Any, response: Any) -> None:
    """Keep usage when strict execution stops before normal loop accounting."""

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        return
    canonical = normalize_usage(
        _usage_with_attributes(raw_usage),
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


def finish_paid_provider_call(
    agent: Any,
    ledger: Any,
    call_id: str | None,
    *,
    status: str,
    response: Any = None,
    error_category: str | None = None,
) -> None:
    """Monotonically settle one generic or topology-specific receipt."""

    if not call_id or ledger is None:
        return
    raw_usage = getattr(response, "usage", None)
    usage = None
    cost_usd = None
    cost_status = None
    cost_source = None
    if raw_usage is not None:
        canonical = normalize_usage(
            _usage_with_attributes(raw_usage),
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
        usage = raw_usage
        cost_usd = cost_result.amount_usd
        cost_status = cost_result.status
        cost_source = cost_result.source
    finish_call = getattr(ledger, "finish_call", None)
    if callable(finish_call):
        finish_call(
            call_id,
            status=status,
            usage=usage,
            error_category=error_category,
            cost_usd=cost_usd,
            cost_status=cost_status,
            cost_source=cost_source,
            notify=False,
        )
    else:
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


__all__ = [
    "finish_paid_provider_call",
    "record_strict_terminal_usage",
]
