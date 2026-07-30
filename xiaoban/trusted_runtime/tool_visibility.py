"""Request-local tool visibility for dynamic-evidence-v2 work turns."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from xiaoban.trusted_runtime.types import MYSTAND_COMPLETION_PROTOCOL_V2, WorkTurn


_RESOURCE_INDEX_TOOL = "mystand_resource_index"
_DYNAMIC_READ_TOOLS = frozenset(
    {"mystand_query", "mystand_authorization"}
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
    return [tool for tool in tools if _tool_name(tool) in allowed_names]


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
    return filtered


__all__ = [
    "dynamic_evidence_allowed_tool_names",
    "filter_dynamic_evidence_api_kwargs",
    "filter_dynamic_evidence_tools",
]
