"""Strip obsolete runtime control metadata before physical tool dispatch."""

from __future__ import annotations

import json
from typing import Any


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
