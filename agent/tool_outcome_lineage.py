"""Structured lineage for physical tool attempts within one user turn.

The model chooses actions, but it never gets authority to declare that one
action settles another action's failure.  The runtime recognises an identical
canonical retry itself.  A different action can settle a failure only through
an internal, target-bound, one-use server grant.

Only hashes of action arguments are retained in outcome state.  Raw arguments
remain in the normal tool transcript and are never duplicated here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


RUNTIME_LINKAGE_ARGUMENT = "_xiaoban_runtime"


def split_runtime_linkage(arguments: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Strip legacy/model-authored control metadata from handler arguments.

    Older candidates exposed this field in tool schemas.  Continue removing it
    defensively so it never reaches a handler, but deliberately return no
    authority metadata: model-authored lineage is untrusted input.
    """

    if not isinstance(arguments, dict):
        return {}, {}
    handler_arguments = dict(arguments)
    handler_arguments.pop(RUNTIME_LINKAGE_ARGUMENT, None)
    return handler_arguments, {}


def sanitize_runtime_linkage_arguments(
    tool_name: str,
    arguments: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Strip reserved metadata from direct and tool-search bridge arguments."""

    handler_arguments, outer_linkage = split_runtime_linkage(arguments)
    try:
        from tools import tool_search

        is_tool_call_bridge = (
            str(tool_name or "") == tool_search.TOOL_CALL_NAME
        )
    except Exception:
        is_tool_call_bridge = str(tool_name or "") == "tool_call"
    if not is_tool_call_bridge:
        return handler_arguments, outer_linkage

    raw_nested = handler_arguments.get("arguments")
    if isinstance(raw_nested, str):
        try:
            raw_nested = json.loads(raw_nested)
        except (TypeError, ValueError):
            raw_nested = None
    if not isinstance(raw_nested, dict):
        return handler_arguments, outer_linkage
    clean_nested, _ = split_runtime_linkage(raw_nested)
    handler_arguments = {
        **handler_arguments,
        "arguments": clean_nested,
    }
    return handler_arguments, outer_linkage


def action_fingerprint(tool_name: str, arguments: Any) -> str:
    """Hash a tool name plus canonical handler arguments."""

    handler_arguments, _ = split_runtime_linkage(arguments)
    canonical = json.dumps(
        {
            "tool_name": str(tool_name or ""),
            "arguments": handler_arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_lineage_action(
    tool_name: str,
    arguments: Any,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Resolve tool-search wrappers and return the real action identity."""

    handler_arguments, outer_linkage = sanitize_runtime_linkage_arguments(
        tool_name,
        arguments,
    )
    try:
        from tools import tool_search

        if str(tool_name or "") != tool_search.TOOL_CALL_NAME:
            return str(tool_name or ""), handler_arguments, outer_linkage
        underlying_name, underlying_arguments, error = (
            tool_search.resolve_underlying_call(handler_arguments)
        )
    except Exception:
        return str(tool_name or ""), handler_arguments, outer_linkage
    if error or not underlying_name or not isinstance(
        underlying_arguments,
        dict,
    ):
        return str(tool_name or ""), handler_arguments, outer_linkage

    clean_underlying, _ = split_runtime_linkage(underlying_arguments)
    return str(underlying_name), clean_underlying, outer_linkage


def unresolved_failure_ids(
    outcomes: Iterable[dict[str, Any]],
    *,
    server_grants: Mapping[str, Mapping[str, Any]] | None = None,
    expected_turn_id: str = "",
) -> list[str]:
    """Resolve only runtime-bound retries and authenticated server recovery."""

    material = [
        item
        for item in outcomes
        if isinstance(item, dict) and item.get("material") is True
    ]
    unresolved: list[str] = []
    known_failures: set[str] = set()
    failure_batches: dict[str, int] = {}
    consumed_recovery_targets: set[str] = set()
    trusted_grants = (
        server_grants
        if isinstance(server_grants, Mapping)
        else {}
    )
    material_by_id = {
        str(item.get("tool_call_id") or ""): item
        for item in material
        if str(item.get("tool_call_id") or "")
    }

    for item in material:
        call_id = str(item.get("tool_call_id") or "")
        recovery_of = str(item.get("recovery_of") or "")
        recovery_authority = str(
            item.get("recovery_authority") or ""
        )
        item_batch = max(0, int(item.get("batch_id") or 0))
        target = material_by_id.get(recovery_of)
        target_fingerprint = (
            str(target.get("action_fingerprint") or "")
            if isinstance(target, dict)
            else ""
        )
        current_fingerprint = str(
            item.get("action_fingerprint") or ""
        )
        exact_action_authorized = (
            recovery_authority == "exact-action"
            and bool(target_fingerprint)
            and current_fingerprint == target_fingerprint
        )
        recovery_grant_id = str(item.get("recovery_grant_id") or "")
        recovery_grant = trusted_grants.get(recovery_grant_id)
        server_grant_authorized = (
            recovery_authority == "server-grant"
            and bool(recovery_grant_id)
            and isinstance(recovery_grant, Mapping)
            and recovery_grant.get("schema")
            == "xiaoban.recovery-grant.v1"
            and str(recovery_grant.get("grant_id") or "")
            == recovery_grant_id
            and recovery_grant.get("max_uses") == 1
            and recovery_grant.get("used") is True
            and str(recovery_grant.get("target_event_id") or "")
            == recovery_of
            and str(
                recovery_grant.get(
                    "allowed_action_fingerprint"
                )
                or ""
            )
            == current_fingerprint
            and (
                not expected_turn_id
                or str(recovery_grant.get("turn_id") or "")
                == expected_turn_id
            )
        )
        if (
            recovery_of
            and (
                exact_action_authorized
                or server_grant_authorized
            )
            and recovery_of in known_failures
            and recovery_of in unresolved
            and recovery_of not in consumed_recovery_targets
            and failure_batches.get(recovery_of, item_batch) < item_batch
        ):
            consumed_recovery_targets.add(recovery_of)
            unresolved = [
                event_id
                for event_id in unresolved
                if event_id != recovery_of
            ]
        if item.get("failed") is True and call_id:
            known_failures.add(call_id)
            failure_batches[call_id] = item_batch
            unresolved.append(call_id)

    return unresolved


def used_recovery_targets(
    outcomes: Iterable[dict[str, Any]],
) -> set[str]:
    """Return event ids already consumed by a dispatched recovery attempt."""

    return {
        str(item.get("recovery_of") or "")
        for item in outcomes
        if isinstance(item, dict) and str(item.get("recovery_of") or "")
    }
