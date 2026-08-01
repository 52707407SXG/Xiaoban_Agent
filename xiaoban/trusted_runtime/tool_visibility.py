"""Request-local tool visibility for dynamic-evidence-v2 work turns."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional

from xiaoban.trusted_runtime.types import MYSTAND_COMPLETION_PROTOCOL_V2, WorkTurn


_RESOURCE_INDEX_TOOL = "mystand_resource_index"
_DYNAMIC_READ_TOOLS = frozenset(
    {"mystand_query", "mystand_authorization"}
)
_SEMANTIC_QUERY_FIELDS = (
    "operation",
    "resource",
    "entities",
    "fact_needs",
    "mode",
)
_SEMANTIC_QUERY_DESCRIPTION = (
    "Read the current user's authorized My Stand material with exactly one "
    "semantic query. Understand the latest request as a whole. Use a "
    "human-readable resource anchor from the safe index for the requested "
    "material; subject entities may further narrow that same target. Request "
    "only the fact categories needed for the answer. Use "
    "only the fields exposed in this schema and call this tool at most once in "
    "the current model turn. Do not supply identity, ownership, internal IDs, "
    "backend modules, typed plans, lookup steps, or query text."
)


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


def _choice_names(choice: Any) -> set[str]:
    if isinstance(choice, str):
        return {choice}
    if isinstance(choice, Mapping):
        names: set[str] = set()
        for key, value in choice.items():
            if key == "name" and isinstance(value, str):
                names.add(value)
            else:
                names.update(_choice_names(value))
        return names
    if isinstance(choice, (list, tuple)):
        names: set[str] = set()
        for item in choice:
            names.update(_choice_names(item))
        return names
    return set()


def _semantic_query_tool(tool: Any) -> Optional[Any]:
    """Return a request-local semantic-only query definition.

    The canonical registry has to support both server-signed typed reads and
    unsigned semantic reads.  Strict providers strip the canonical top-level
    union, so exposing that combined shape makes mutually exclusive fields
    look mixable.  Dynamic v2 reads therefore receive one concrete schema,
    matching the handler branch they are allowed to execute.
    """
    if not isinstance(tool, Mapping) or _tool_name(tool) != "mystand_query":
        return None
    projected = copy.deepcopy(dict(tool))
    function = projected.get("function")
    if not isinstance(function, dict):
        return None
    parameters = function.get("parameters")
    properties = (
        parameters.get("properties")
        if isinstance(parameters, Mapping)
        else None
    )
    if not isinstance(properties, Mapping):
        return None
    semantic_properties = {
        field: copy.deepcopy(properties[field])
        for field in _SEMANTIC_QUERY_FIELDS
        if field in properties
    }
    if set(semantic_properties) != set(_SEMANTIC_QUERY_FIELDS):
        return None
    function["description"] = _SEMANTIC_QUERY_DESCRIPTION
    function["parameters"] = {
        "type": "object",
        "properties": semantic_properties,
        # Strict backends reject a top-level anyOf. Dynamic reads happen only
        # after the server index has supplied safe resource labels, so one
        # concrete resource locator is mandatory here and in the handler.
        "required": ["operation", "resource", "fact_needs"],
        "additionalProperties": False,
    }
    return projected


def dynamic_evidence_allowed_tool_names(
    turn: Optional[WorkTurn],
) -> Optional[frozenset[str]]:
    """Return the request-local stage allow-list, or None outside dynamic v2."""
    if (
        turn is None
        or turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
        or getattr(turn, "fact_requirement", None) is not None
    ):
        return None
    if bool(getattr(turn, "business_tools_disabled", False)):
        return frozenset()
    if str(getattr(turn, "interaction_kind", "") or "") != "WORK":
        # Product help and other trusted CHAT turns do not need a business-data
        # index. Their request toolset remains authoritative (the owner policy
        # includes read-only My Stand source inspection; brokers do not).
        return None
    if str(getattr(turn, "completion_finalization", "") or ""):
        return frozenset()
    receipt = getattr(turn, "index_receipt", None)
    if receipt is not None and receipt.status == "found":
        return _DYNAMIC_READ_TOOLS
    return frozenset({_RESOURCE_INDEX_TOOL})


def filter_dynamic_evidence_tools(
    tools: list[Any],
    *,
    turn: Optional[WorkTurn] = None,
) -> list[Any]:
    """Filter canonical tool definitions before provider wire conversion."""
    if turn is None:
        try:
            from xiaoban.trusted_runtime.turns import current_turn

            turn = current_turn()
        except Exception:
            turn = None
    allowed_names = dynamic_evidence_allowed_tool_names(turn)
    if allowed_names is None or not isinstance(tools, list):
        return tools
    post_index_read = (
        "mystand_query" in allowed_names
        and _RESOURCE_INDEX_TOOL not in allowed_names
    )
    filtered: list[Any] = []
    for tool in tools:
        name = _tool_name(tool)
        if name not in allowed_names:
            continue
        if post_index_read and name == "mystand_query":
            projected = _semantic_query_tool(tool)
            if projected is None:
                # Never fall back to the canonical mixed typed/semantic union
                # when a strict stage projection cannot be built.
                continue
            filtered.append(projected)
        else:
            filtered.append(tool)
    return filtered


def filter_dynamic_evidence_api_kwargs(
    api_kwargs: dict[str, Any],
    *,
    turn: Optional[WorkTurn] = None,
) -> dict[str, Any]:
    """Return a request-local provider payload with the correct tool stage.

    The agent's shared ``tools`` catalog is never mutated.  An unsigned,
    dynamic-evidence work turn can only discover resources until a current-turn
    ``IndexReceipt(found)`` exists.  Once it exists, the index tool is hidden
    and the remaining read tools become visible.
    """
    if turn is None:
        try:
            from xiaoban.trusted_runtime.turns import current_turn

            turn = current_turn()
        except Exception:
            turn = None
    if not isinstance(api_kwargs, dict) or not isinstance(
        api_kwargs.get("tools"),
        list,
    ):
        return api_kwargs

    filtered_tools = filter_dynamic_evidence_tools(
        api_kwargs["tools"],
        turn=turn,
    )
    if filtered_tools is api_kwargs["tools"]:
        return api_kwargs

    filtered = {**api_kwargs, "tools": filtered_tools}
    visible_names = {
        name for name in (_tool_name(tool) for tool in filtered_tools) if name
    }
    selected_names = _choice_names(filtered.get("tool_choice"))
    named_choices = selected_names.difference({"", "auto", "none", "required"})
    if (
        not filtered_tools
        or named_choices
        and not named_choices.issubset(visible_names)
    ):
        filtered.pop("tool_choice", None)
    if not filtered_tools:
        filtered.pop("parallel_tool_calls", None)
    else:
        # The trusted runtime executes and records one read at a time. This
        # gives the model the result (including a safe error) before it chooses
        # a correction, and prevents a single response from dispatching several
        # speculative reads.
        filtered["parallel_tool_calls"] = False
    return filtered


__all__ = [
    "dynamic_evidence_allowed_tool_names",
    "filter_dynamic_evidence_api_kwargs",
    "filter_dynamic_evidence_tools",
]
