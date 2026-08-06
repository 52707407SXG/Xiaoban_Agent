"""One-shot tool dispatch and result fences for strict true-MoA turns."""

from __future__ import annotations

import logging
from typing import Any, Callable


def strict_tool_mode(agent: Any) -> bool:
    return bool(getattr(agent, "_strict_no_automatic_paid_retry", False))


def _strict_controller(agent: Any) -> Any:
    from agent.true_moa_conversation_policy import strict_cancel_controller

    return strict_cancel_controller(agent)


def _tool_fence_key(agent: Any, phase: str, tool_call_id: str) -> str:
    request_id = str(
        getattr(agent, "_current_api_request_id", "") or ""
    ).strip()
    if request_id:
        return f"{phase}:{request_id}:{tool_call_id}"
    return f"{phase}:{tool_call_id}"


def claim_strict_tool_dispatch(agent: Any, tool_call_id: str) -> bool:
    """Atomically fence a true-MoA tool handler against terminal stop."""

    if not strict_tool_mode(agent):
        return True
    if getattr(agent, "_interrupt_requested", False):
        return False
    controller = _strict_controller(agent)
    if controller is None:
        return True
    return controller.try_begin_dispatch(
        _tool_fence_key(agent, "final-tool", tool_call_id)
    )


def claim_strict_tool_handler(agent: Any, tool_call_id: str) -> bool:
    """Linearize the actual handler start after preflight work."""

    if not strict_tool_mode(agent):
        return True
    if getattr(agent, "_interrupt_requested", False):
        return False
    controller = _strict_controller(agent)
    if controller is None:
        return True
    return controller.try_begin_dispatch(
        _tool_fence_key(agent, "final-tool-handler", tool_call_id)
    )


def claim_strict_tool_execute(agent: Any, tool_call_id: str) -> bool:
    """Install the final one-shot fence immediately before the real handler."""

    if not strict_tool_mode(agent):
        return True
    if getattr(agent, "_interrupt_requested", False):
        return False
    controller = _strict_controller(agent)
    if controller is None:
        return True
    return controller.try_begin_dispatch(
        _tool_fence_key(agent, "final-tool-execute", tool_call_id)
    )


def claim_strict_tool_result(agent: Any, tool_call_id: str) -> bool:
    """Atomically commit one real tool result against terminal stop."""

    if not strict_tool_mode(agent):
        return True
    controller = _strict_controller(agent)
    if controller is None:
        return not getattr(agent, "_interrupt_requested", False)
    return controller.try_begin_dispatch(
        _tool_fence_key(agent, "final-tool-result", tool_call_id)
    )


def strict_tool_terminal_stopped(agent: Any) -> bool:
    if not strict_tool_mode(agent):
        return False
    if getattr(agent, "_interrupt_requested", False):
        return True
    controller = _strict_controller(agent)
    return bool(controller is not None and controller.is_set)


def begin_strict_tool_handler(
    agent: Any,
    tool_call_id: str,
    function_name: str,
    function_args: dict[str, Any],
    *,
    build_preview: Callable[[str, dict[str, Any]], Any],
) -> bool:
    """Claim strict handler start, then expose callbacks beside dispatch."""

    if not strict_tool_mode(agent):
        return True
    if not claim_strict_tool_handler(agent, tool_call_id):
        return False
    agent._current_tool = function_name
    agent._touch_activity(f"executing tool: {function_name}")
    try:
        from tools.environments.base import set_activity_callback

        set_activity_callback(agent._touch_activity)
    except Exception:
        pass
    if agent.tool_progress_callback:
        try:
            preview = build_preview(function_name, function_args)
            agent.tool_progress_callback(
                "tool.started",
                function_name,
                preview,
                function_args,
            )
        except Exception:
            logging.debug(
                "Tool progress callback failed "
                "(category=callback_failure; strict details redacted)"
            )
    if agent.tool_start_callback:
        try:
            agent.tool_start_callback(
                tool_call_id,
                function_name,
                function_args,
            )
        except Exception:
            logging.debug(
                "Tool start callback failed "
                "(category=callback_failure; strict details redacted)"
            )
    return True


def strict_extension_hook_kwargs(agent: Any) -> dict[str, bool]:
    """Keep extension hooks outside the fixed true-MoA execution path."""

    return {"skip_extension_hooks": True} if strict_tool_mode(agent) else {}
