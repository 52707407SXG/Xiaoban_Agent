"""Tool-call execution — sequential and concurrent dispatch.

Both AIAgent methods (``_execute_tool_calls_sequential`` and
``_execute_tool_calls_concurrent``) live here as module-level
functions that take the parent ``AIAgent`` as their first argument.

``run_agent`` keeps thin wrappers so existing call sites work; tests
that patch ``run_agent._set_interrupt`` are honored because the
extracted functions reach back through the ``run_agent`` module via
``_ra()`` for that symbol.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import logging
import os
import random
import threading
import time
from typing import Any, Optional

from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    get_tool_emoji as _get_tool_emoji,
    _detect_tool_failure,
)
from agent.tool_guardrails import ToolGuardrailDecision
from agent.tool_outcome_lineage import (
    sanitize_runtime_linkage_arguments,
    split_runtime_linkage,
)
from agent.tool_result_classification import (
    CANONICAL_TOOL_RESULT_INTERNAL_KEY,
    normalize_tool_result,
)
from agent.true_moa_tool_fence import (
    begin_strict_tool_handler as _begin_strict_tool_handler,
    claim_strict_tool_dispatch as _claim_strict_tool_dispatch,
    claim_strict_tool_execute as _claim_strict_tool_execute,
    claim_strict_tool_result as _claim_strict_tool_result,
    strict_extension_hook_kwargs as _strict_extension_hook_kwargs,
    strict_tool_mode as _strict_tool_mode,
    strict_tool_terminal_stopped as _strict_tool_terminal_stopped,
)
from agent.tool_dispatch_helpers import (
    _is_destructive_command,
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,
    make_tool_result_message,
)
from tools.terminal_tool import (
    get_active_env,
)
from tools.thread_context import propagate_context_to_thread
from tools.tool_result_storage import (
    maybe_persist_tool_result,
    enforce_turn_budget,
)
from tools.budget_config import BudgetConfig, DEFAULT_BUDGET, budget_for_context_window

logger = logging.getLogger(__name__)
_CANONICAL_RESULT_UNSET = object()
_CANONICAL_TERMINAL_METADATA_FIELDS = (
    "schema",
    "requestId",
    "turnId",
    "callId",
    "toolName",
    "dispatchState",
    "outcome",
    "retrySafe",
)


def _budget_for_agent(agent) -> BudgetConfig:
    """Resolve a tool-result BudgetConfig scaled to the agent's context window.

    Large-context models keep the historical 100K/200K char defaults; small
    models (e.g. a 65K-token local model switched into mid-session) get a budget
    proportional to their window so a single large tool result can't push the
    request past the model's limit (#23767). Falls back to the default budget
    when the context length isn't resolvable.
    """
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        return budget_for_context_window(int(ctx)) if ctx else DEFAULT_BUDGET
    except Exception:
        return DEFAULT_BUDGET

# Maximum number of concurrent worker threads for parallel tool execution.
# Mirrors the constant in ``run_agent`` for tests/imports that look here.
_MAX_TOOL_WORKERS = 8


def _mystand_plan_mode_block(function_name: str, function_args: dict) -> str | None:
    """Keep Plan read-only at the My Stand tool boundary."""
    try:
        from gateway.session_context import get_session_env

        if get_session_env("XIAOBAN_SESSION_SOURCE", "") != "mystand":
            return None
        if get_session_env("XIAOBAN_SESSION_WORK_MODE", "") != "plan":
            return None
    except Exception:
        return None
    normalized_name = str(function_name or "").strip().lower()
    if normalized_name == "mystand_authorization_write":
        return "Plan 模式只做分析和规划，不能执行站内增删改。请切换到 Build 后再继续。"
    if normalized_name == "mystand_skill_manage":
        operation = str((function_args or {}).get("operation", "")).strip().lower()
        if operation not in {"list", "read", "view", "inspect"}:
            return "Plan 模式可以查看 Skill，但不能创建、更新、注册或删除。请切换到 Build 后再继续。"
    return None


def _ra():
    """Lazy reference to ``run_agent`` so patches like ``run_agent._set_interrupt`` work."""
    import run_agent
    return run_agent


def _log_tool_handler_exception(
    agent,
    owner: str,
    function_name: str,
    error: BaseException,
) -> None:
    """Keep strict tool handler failures useful without logging their body."""
    if _strict_tool_mode(agent):
        logger.error(
            "%s raised for %s (category=handler_exception; "
            "strict details redacted)",
            owner,
            function_name,
        )
        return
    logger.error(
        "%s raised for %s: %s",
        owner,
        function_name,
        error,
        exc_info=(type(error), error, error.__traceback__),
    )


def _log_tool_callback_exception(
    agent,
    callback_name: str,
    error: BaseException,
) -> None:
    """Redact callback exception bodies on the strict My Stand path."""
    if _strict_tool_mode(agent):
        logger.debug(
            "%s failed (category=callback_failure; strict details redacted)",
            callback_name,
        )
        return
    logger.debug("%s error: %s", callback_name, error)


def _trusted_guardrail_fields(
    decision: ToolGuardrailDecision | None,
) -> dict[str, Any] | None:
    """Expose post-call recovery guidance through runtime-owned metadata."""
    if decision is None or decision.action not in {"warn", "halt"}:
        return None
    return {
        "continuation": {
            "kind": "tool_guardrail",
            "guardrail": decision.to_metadata(),
        }
    }


def _emit_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    result: Any,
    effective_task_id: str,
    tool_call_id: str,
    duration_ms: int = 0,
    status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    middleware_trace: Optional[list[dict[str, Any]]] = None,
) -> None:
    if _strict_tool_mode(agent):
        return
    try:
        from model_tools import _emit_post_tool_call_hook
        _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "",
            tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception:
        pass


def _cancelled_tool_result(reason: str = "user interrupt") -> str:
    return json.dumps(
        {
            "error": f"Tool execution cancelled by {reason}",
            "status": "cancelled",
        },
        ensure_ascii=False,
    )


def _canonical_terminal_callback_metadata(
    canonical: dict[str, Any],
) -> dict[str, Any]:
    """Return the fixed public-safe correlation subset for observers."""
    return {
        key: canonical[key]
        for key in _CANONICAL_TERMINAL_METADATA_FIELDS
        if key in canonical
    }


def _invoke_tool_complete_callback(
    callback: Any,
    tool_call_id: str,
    function_name: str,
    function_args: dict[str, Any],
    function_result: Any,
    canonical_metadata: dict[str, Any],
) -> None:
    """Invoke a terminal observer with optional canonical metadata.

    Existing integrations accept four positional arguments. New integrations
    may opt into a fifth metadata argument. Signature binding happens before
    invocation so a TypeError raised *inside* a callback is never mistaken for
    an old callback and invoked twice.
    """
    use_metadata = False
    try:
        inspect.signature(callback).bind(
            tool_call_id,
            function_name,
            function_args,
            function_result,
            canonical_metadata,
        )
        use_metadata = True
    except (TypeError, ValueError):
        pass

    if use_metadata:
        callback(
            tool_call_id,
            function_name,
            function_args,
            function_result,
            canonical_metadata,
        )
        return
    callback(
        tool_call_id,
        function_name,
        function_args,
        function_result,
    )


def _append_canonical_tool_result(
    agent: Any,
    messages: list,
    name: str,
    content: Any,
    tool_call_id: str,
    dispatch_state: str,
    outcome_hint: str | None = None,
    classification_result: Any = _CANONICAL_RESULT_UNSET,
    trusted_fields: dict[str, Any] | None = None,
    *,
    function_args: dict[str, Any] | None = None,
    callback_result: Any = _CANONICAL_RESULT_UNSET,
    emit_terminal_callback: bool = False,
) -> dict[str, Any]:
    """Commit one canonical result, then notify its terminal observers."""
    message = make_tool_result_message(name, content, tool_call_id)
    effective_trusted_fields = dict(trusted_fields or {})
    if (
        name == "mystand_authorization_write"
        and dispatch_state == "dispatched"
        and isinstance(function_args, dict)
        and function_args.get("operation") == "commit_write"
    ):
        receipt_source = (
            callback_result
            if callback_result is not _CANONICAL_RESULT_UNSET
            else (
                classification_result
                if classification_result is not _CANONICAL_RESULT_UNSET
                else content
            )
        )
        try:
            parsed_receipt = (
                json.loads(receipt_source)
                if isinstance(receipt_source, str)
                else receipt_source
            )
        except (TypeError, ValueError):
            parsed_receipt = None
        from tools.mystand_authorization_tool import _current_session
        from tools.mystand_authorization_write_tool import (
            _verified_commit_receipt,
        )

        if _verified_commit_receipt(
            parsed_receipt,
            commit_args=function_args,
            session=_current_session(),
        ):
            effective_trusted_fields["verifiedWriteReceipt"] = (
                parsed_receipt
            )
    canonical = normalize_tool_result(
        request_id=str(getattr(agent, "_current_request_id", "") or ""),
        turn_id=str(getattr(agent, "_current_turn_id", "") or ""),
        call_id=tool_call_id,
        tool_name=name,
        dispatch_state=dispatch_state,
        result=content if classification_result is _CANONICAL_RESULT_UNSET else classification_result,
        outcome_hint=outcome_hint,
        trusted_fields=effective_trusted_fields,
    )
    message[CANONICAL_TOOL_RESULT_INTERNAL_KEY] = canonical
    messages.append(message)
    callback = getattr(agent, "tool_complete_callback", None)
    if emit_terminal_callback and callback:
        try:
            _invoke_tool_complete_callback(
                callback,
                tool_call_id,
                name,
                function_args if isinstance(function_args, dict) else {},
                content
                if callback_result is _CANONICAL_RESULT_UNSET
                else callback_result,
                _canonical_terminal_callback_metadata(canonical),
            )
        except Exception as cb_err:
            _log_tool_callback_exception(
                agent,
                "Tool complete callback",
                cb_err,
            )
    return message


def _emit_guardrail_preflight_lifecycle(
    agent,
    *,
    tool_call_id: str,
    function_name: str,
    function_args: dict,
    function_result: Any,
) -> None:
    """Project one blocked proposal as a correlated failed tool lifecycle.

    The handler did not run, so progress carries ``phase=preflight``. This
    helper establishes the correlated start; the later canonical append is
    the only source of the structured terminal callback.
    """
    if agent.tool_progress_callback:
        try:
            preview = _build_tool_preview(function_name, function_args)
            agent.tool_progress_callback(
                "tool.started",
                function_name,
                preview,
                function_args,
            )
        except Exception as cb_err:
            _log_tool_callback_exception(
                agent,
                "Tool progress callback",
                cb_err,
            )
    if agent.tool_start_callback:
        try:
            agent.tool_start_callback(
                tool_call_id,
                function_name,
                function_args,
            )
        except Exception as cb_err:
            _log_tool_callback_exception(
                agent,
                "Tool start callback",
                cb_err,
            )
    if agent.tool_progress_callback:
        try:
            agent.tool_progress_callback(
                "tool.completed",
                function_name,
                None,
                None,
                duration=0.0,
                is_error=True,
                result=function_result,
                phase="preflight",
            )
        except Exception as cb_err:
            _log_tool_callback_exception(
                agent,
                "Tool progress callback",
                cb_err,
            )


def _emit_cancelled_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    start_time: float,
    reason: str = "user interrupt",
    error_type: str = "keyboard_interrupt",
    middleware_trace: Optional[list[dict[str, Any]]] = None,
) -> str:
    result = _cancelled_tool_result(reason)
    _emit_terminal_post_tool_call(
        agent,
        function_name=function_name,
        function_args=function_args,
        result=result,
        effective_task_id=effective_task_id,
        tool_call_id=tool_call_id,
        duration_ms=int((time.time() - start_time) * 1000),
        status="cancelled",
        error_type=error_type,
        error_message=f"Tool execution cancelled by {reason}",
        middleware_trace=list(middleware_trace or []),
    )
    return result


def _tool_search_scoped_names(agent) -> frozenset:
    """Return the deferrable tool names the session may invoke via tool_call.

    The Tool Search unwrap dispatches the underlying tool directly, bypassing
    the bridge branch (and its scope check) in
    ``model_tools.handle_function_call``. To keep a restricted-toolset session
    (subagent, kanban worker, curated gateway session) from reaching tools it
    was never granted, the unwrap validates the underlying name against this
    set: the deferrable subset of the session's own enabled/disabled toolset
    scope.

    Result is cached on the agent and refreshed when the tool registry's
    generation changes (e.g. an MCP server reconnects), so the common case is
    a dict lookup, not a full tool-defs rebuild on every tool call.
    """
    try:
        import model_tools
        from tools import tool_search as _ts
        from tools.registry import registry as _registry
    except Exception:
        return frozenset()

    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    cache_key = (
        getattr(_registry, "_generation", 0),
        frozenset(enabled) if enabled is not None else None,
        frozenset(disabled) if disabled is not None else None,
    )
    cached = getattr(agent, "_tool_search_scope_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    try:
        scoped_defs = model_tools.get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
        names = _ts.scoped_deferrable_names(scoped_defs)
    except Exception:
        names = frozenset()
    try:
        agent._tool_search_scope_cache = (cache_key, names)
    except Exception:
        pass
    return names


def _apply_tool_request_middleware_for_agent(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
) -> tuple[dict, list[dict[str, Any]]]:
    if _strict_tool_mode(agent):
        return function_args, []
    try:
        from xiaoban_cli.middleware import apply_tool_request_middleware

        result = apply_tool_request_middleware(
            function_name,
            function_args,
            task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "",
            tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
        )
        payload = result.payload if isinstance(result.payload, dict) else function_args
        return payload, list(result.trace)
    except Exception as exc:
        logger.debug("tool_request middleware error: %s", exc)
        return function_args, []


def _run_agent_tool_execution_middleware(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
) -> tuple[Any, dict]:
    if _strict_tool_mode(agent):
        return execute(function_args), function_args
    observed_args = function_args

    def _execute(next_args: dict) -> Any:
        nonlocal observed_args
        observed_args = next_args if isinstance(next_args, dict) else function_args
        return execute(observed_args)

    from xiaoban_cli.middleware import run_tool_execution_middleware

    result = run_tool_execution_middleware(
        function_name,
        function_args,
        _execute,
        original_args=function_args,
        task_id=effective_task_id or "",
        session_id=getattr(agent, "session_id", "") or "",
        tool_call_id=tool_call_id or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
    )
    return result, observed_args


def _run_agent_tool_execution_middleware_captured(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
) -> tuple[Any, dict, bool]:
    """Run an inline Agent tool and convert handler exceptions to K2 input."""
    try:
        result, observed_args = _run_agent_tool_execution_middleware(
            agent,
            function_name=function_name,
            function_args=function_args,
            effective_task_id=effective_task_id,
            tool_call_id=tool_call_id,
            execute=execute,
        )
        return result, observed_args, False
    except Exception as exc:
        _log_tool_handler_exception(
            agent,
            "inline Agent tool",
            function_name,
            exc,
        )
        return (
            f"Error executing tool '{function_name}': {exc}",
            function_args,
            True,
        )


def execute_tool_calls_concurrent(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
    """Execute multiple tool calls concurrently using a thread pool.

    Results are collected in the original tool-call order and appended to
    messages so the API sees them in the expected sequence.
    """
    tool_calls = assistant_message.tool_calls
    num_tools = len(tool_calls)

    # Resolve the context-scaled tool-output budget once per turn (cheap, but
    # avoids rebuilding it per result inside the loop below).
    _tool_budget = _budget_for_agent(agent)

    # ── Pre-flight: interrupt check ──────────────────────────────────
    if agent._interrupt_requested:
        print(f"{agent.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
        for tc in tool_calls:
            _append_canonical_tool_result(
                agent,
                messages,
                tc.function.name,
                f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                tc.id,
                "not_dispatched",
                "cancelled",
            )
        return

    # ── Parse args + pre-execution bookkeeping ───────────────────────
    parsed_calls = []  # list of (tool_call, function_name, function_args, middleware_trace, block_result, blocked_by_guardrail)
    terminal_fenced_call_ids: set[str] = set()
    dispatch_chain_entered_call_ids: set[str] = set()
    handler_exception_call_ids: set[str] = set()
    lifecycle_started_call_ids: set[str] = set()
    terminal_fenced_lock = threading.Lock()
    for tool_call in tool_calls:
        function_name = tool_call.function.name

        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            function_args = {}
        if not isinstance(function_args, dict):
            function_args = {}
        function_args, _ = sanitize_runtime_linkage_arguments(
            function_name,
            function_args,
        )

        # ── Tool Search unwrap ────────────────────────────────────────
        # When the model invokes the tool_call bridge, peel it open so
        # every downstream check (checkpointing, guardrails, plugin
        # pre-tool-call hooks, the display/activity feed, the post-call
        # callback) sees the underlying tool — not the bridge. This is
        # the OpenClaw lesson: hooks must observe the real tool name.
        #
        # The original tool_call entry on ``tool_call.function`` is left
        # untouched so the conversation transcript and the matching
        # tool_call_id are preserved exactly as the model emitted them.
        #
        # Scope gate: the unwrap dispatches the underlying tool directly
        # (bypassing the bridge branch in handle_function_call and its
        # scope check), so we enforce session toolset scope HERE. A tool
        # the session was not granted is rejected before any checkpoint,
        # hook, or dispatch fires.
        _ts_scope_block = None
        try:
            from tools import tool_search as _ts
            if (
                _ts_scope_block is None
                and function_name == _ts.TOOL_CALL_NAME
            ):
                _underlying, _underlying_args, _err = _ts.resolve_underlying_call(function_args)
                if not _err and _underlying:
                    if _underlying in _tool_search_scoped_names(agent):
                        function_name = _underlying
                        function_args = _underlying_args
                        function_args, _ = split_runtime_linkage(
                            function_args
                        )
                    else:
                        _ts_scope_block = json.dumps({
                            "error": (
                                f"'{_underlying}' is not available in this session. "
                                "Use tool_search to find tools you can call."
                            ),
                        }, ensure_ascii=False)
        except Exception:
            pass

        if _ts_scope_block is None:
            function_args, middleware_trace = (
                _apply_tool_request_middleware_for_agent(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                )
            )
        else:
            middleware_trace = []

        tool_call_id = getattr(tool_call, "id", "") or ""
        terminal_dispatch_denied = not _claim_strict_tool_dispatch(
            agent,
            tool_call_id,
        )
        if terminal_dispatch_denied:
            terminal_fenced_call_ids.add(tool_call_id)

        # ── Block evaluation (BEFORE checkpoint preflight) ───────────
        # We must know whether the tool will execute before touching
        # checkpoint state (dedup slot, real snapshots).
        block_result = (
            _cancelled_tool_result("terminal fence")
            if terminal_dispatch_denied
            else None
        )
        blocked_by_guardrail = False
        if terminal_dispatch_denied:
            # Stop won the atomic dispatch gate.  No guardrail, trusted
            # PreAction, checkpoint, callback, display or handler may observe
            # the late tool proposal.
            pass
        elif _ts_scope_block is not None:
            # Out-of-scope tool_call: reject before hooks/guardrails/dispatch.
            block_result = _ts_scope_block
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=block_result,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tool_call, "id", "") or "",
                status="blocked",
                error_type="tool_scope_block",
                error_message=_ts_scope_block,
                middleware_trace=list(middleware_trace),
            )
        elif (plan_mode_block := _mystand_plan_mode_block(function_name, function_args)) is not None:
            block_result = json.dumps({"error": plan_mode_block}, ensure_ascii=False)
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=block_result,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tool_call, "id", "") or "",
                status="blocked",
                error_type="plan_mode_read_only",
                error_message=plan_mode_block,
                middleware_trace=list(middleware_trace),
            )
        else:
            block_message = None
            try:
                if not _strict_tool_mode(agent):
                    from xiaoban_cli.plugins import get_pre_tool_call_block_message
                    block_message = get_pre_tool_call_block_message(
                        function_name,
                        function_args,
                        task_id=effective_task_id or "",
                        session_id=getattr(agent, "session_id", "") or "",
                        tool_call_id=getattr(tool_call, "id", "") or "",
                        turn_id=getattr(agent, "_current_turn_id", "") or "",
                        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                        middleware_trace=list(middleware_trace),
                    )
            except Exception:
                block_message = None

            if block_message is not None:
                block_result = json.dumps({"error": block_message}, ensure_ascii=False)
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    result=block_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    status="blocked",
                    error_type="plugin_block",
                    error_message=block_message,
                    middleware_trace=list(middleware_trace),
                )
            else:
                guardrail_decision = agent._tool_guardrails.before_call(function_name, function_args)
                if not guardrail_decision.allows_execution:
                    block_result = agent._guardrail_block_result(guardrail_decision)
                    blocked_by_guardrail = True
                    _emit_terminal_post_tool_call(
                        agent,
                        function_name=function_name,
                        function_args=function_args,
                        result=block_result,
                        effective_task_id=effective_task_id,
                        tool_call_id=getattr(tool_call, "id", "") or "",
                        status="blocked",
                        error_type="guardrail_block",
                        error_message=getattr(guardrail_decision, "message", None) or "Tool blocked by guardrail policy",
                        middleware_trace=list(middleware_trace),
                    )

        if block_result is None:
            # Reset nudge counters only after the terminal and local policy
            # gates accepted a real execution. My Stand K3 is enforced at the
            # physical ToolRegistry dispatch boundary below this layer.
            if function_name == "memory":
                agent._turns_since_memory = 0
            elif function_name == "skill_manage":
                agent._iters_since_skill = 0

        # ── Checkpoint preflight (only for tools that will execute) ──
        if block_result is None:
            # Checkpoint for file-mutating tools
            if function_name in {"write_file", "patch"} and agent._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = agent._checkpoint_mgr.get_working_dir_for_path(file_path)
                        agent._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and agent._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        agent._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

        parsed_calls.append((tool_call, function_name, function_args, middleware_trace, block_result, blocked_by_guardrail))

    # ── Logging / callbacks ──────────────────────────────────────────
    visible_calls = [
        item
        for item in parsed_calls
        if (getattr(item[0], "id", "") or "") not in terminal_fenced_call_ids
    ]
    tool_names_str = ", ".join(name for _, name, _, _, _, _ in visible_calls)
    if (
        visible_calls
        and not _strict_tool_mode(agent)
        and not agent.quiet_mode
        and getattr(agent, "tool_progress_mode", "all") != "off"
    ):
        print(f"  ⚡ Concurrent: {len(visible_calls)} tool calls — {tool_names_str}")
        for i, (tc, name, args, middleware_trace, block_result, blocked_by_guardrail) in enumerate(visible_calls, 1):
            args_str = json.dumps(args, ensure_ascii=False)
            if agent.verbose_logging:
                print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                print(agent._wrap_verbose("Args: ", json.dumps(args, indent=2, ensure_ascii=False)))
            else:
                args_preview = args_str[:agent.log_prefix_chars] + "..." if len(args_str) > agent.log_prefix_chars else args_str
                print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

    for tc, name, args, middleware_trace, block_result, blocked_by_guardrail in parsed_calls:
        if (
            block_result is not None
            and blocked_by_guardrail
            and (getattr(tc, "id", "") or "")
            not in terminal_fenced_call_ids
        ):
            with terminal_fenced_lock:
                lifecycle_started_call_ids.add(
                    getattr(tc, "id", "") or ""
                )
            _emit_guardrail_preflight_lifecycle(
                agent,
                tool_call_id=getattr(tc, "id", "") or "",
                function_name=name,
                function_args=args,
                function_result=block_result,
            )

    for tc, name, args, middleware_trace, block_result, blocked_by_guardrail in parsed_calls:
        if block_result is not None:
            continue
        if _strict_tool_mode(agent):
            # Strict turns emit start callbacks only after the worker wins the
            # handler-start gate immediately beside the real dispatch.
            continue
        if agent.tool_progress_callback:
            try:
                preview = _build_tool_preview(name, args)
                agent.tool_progress_callback("tool.started", name, preview, args)
            except Exception as cb_err:
                _log_tool_callback_exception(
                    agent,
                    "Tool progress callback",
                    cb_err,
                )

    for tc, name, args, middleware_trace, block_result, blocked_by_guardrail in parsed_calls:
        if block_result is not None:
            continue
        if _strict_tool_mode(agent):
            continue
        if agent.tool_start_callback:
            try:
                agent.tool_start_callback(tc.id, name, args)
            except Exception as cb_err:
                _log_tool_callback_exception(
                    agent,
                    "Tool start callback",
                    cb_err,
                )
        with terminal_fenced_lock:
            lifecycle_started_call_ids.add(getattr(tc, "id", "") or "")

    # ── Concurrent execution ─────────────────────────────────────────
    # Each slot holds (function_name, function_args, function_result, duration, error_flag, blocked_flag, middleware_trace)
    results = [None] * num_tools
    for i, (tc, name, args, middleware_trace, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
        if block_result is not None:
            results[i] = (name, args, block_result, 0.0, True, True, middleware_trace)

    runnable_calls = [
        (i, tc, name, args)
        for i, (
            tc,
            name,
            args,
            middleware_trace,
            block_result,
            blocked_by_guardrail,
        ) in enumerate(parsed_calls)
        if block_result is None
    ]
    # Touch activity only when a handler really passed every gate.
    if (
        runnable_calls
        and not _strict_tool_mode(agent)
    ):
        agent._current_tool = tool_names_str
        agent._touch_activity(
            f"executing {len(runnable_calls)} tools concurrently: {tool_names_str}"
        )

    def _run_tool(index, tool_call, function_name, function_args, middleware_trace):
        """Worker function executed in a thread."""
        # Register this worker tid so the agent can fan out an interrupt
        # to it — see AIAgent.interrupt().  Must happen first thing, and
        # must be paired with discard + clear in the finally block.
        _worker_tid = threading.current_thread().ident
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(_worker_tid)
        # Race: if the agent was interrupted between fan-out (which
        # snapshotted an empty/earlier set) and our registration, apply
        # the interrupt to our own tid now so is_interrupted() inside
        # the tool returns True on the next poll.
        if agent._interrupt_requested:
            try:
                _ra()._set_interrupt(True, _worker_tid)
            except Exception:
                pass
        # Set the activity callback on THIS worker thread so
        # _wait_for_process (terminal commands) can fire heartbeats.
        # The callback is thread-local; the main thread's callback
        # is invisible to worker threads.
        try:
            from tools.environments.base import set_activity_callback
            set_activity_callback(agent._touch_activity)
        except Exception:
            pass
        # Approval/sudo callbacks (thread-local) and the agent turn's
        # ContextVars are propagated by propagate_context_to_thread() at the
        # submit site below (GHSA-qg5c-hvr5-hjgr, #13617).
        start = time.time()
        try:
            if not _begin_strict_tool_handler(
                agent,
                getattr(tool_call, "id", "") or "",
                function_name,
                function_args,
                build_preview=_build_tool_preview,
            ):
                result = _cancelled_tool_result("terminal fence")
                with terminal_fenced_lock:
                    terminal_fenced_call_ids.add(
                        getattr(tool_call, "id", "") or ""
                    )
                results[index] = (
                    function_name,
                    function_args,
                    result,
                    time.time() - start,
                    True,
                    True,
                    middleware_trace,
                )
                return
            with terminal_fenced_lock:
                lifecycle_started_call_ids.add(
                    getattr(tool_call, "id", "") or ""
                )
            if not _claim_strict_tool_execute(
                agent,
                getattr(tool_call, "id", "") or "",
            ):
                result = _cancelled_tool_result("terminal fence")
                with terminal_fenced_lock:
                    terminal_fenced_call_ids.add(
                        getattr(tool_call, "id", "") or ""
                    )
                results[index] = (
                    function_name,
                    function_args,
                    result,
                    time.time() - start,
                    True,
                    True,
                    middleware_trace,
                )
                return
            tool_error = None
            keyboard_interrupted = False
            try:
                # Entering this execution chain is K2 ``dispatched``; a
                # middleware-owned result must not be retried as pre-dispatch.
                with terminal_fenced_lock:
                    dispatch_chain_entered_call_ids.add(
                        getattr(tool_call, "id", "") or ""
                    )
                result = agent._invoke_tool(
                    function_name,
                    function_args,
                    effective_task_id,
                    tool_call.id,
                    messages=messages,
                    pre_tool_block_checked=True,
                    skip_tool_request_middleware=True,
                    tool_request_middleware_trace=list(middleware_trace),
                )
            except KeyboardInterrupt:
                keyboard_interrupted = True
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
                result = _cancelled_tool_result("keyboard interrupt")
            except Exception as exc:
                tool_error = exc
                with terminal_fenced_lock:
                    handler_exception_call_ids.add(getattr(tool_call, "id", "") or "")
                result = f"Error executing tool '{function_name}': {exc}"
            duration = time.time() - start
            if not _claim_strict_tool_result(
                agent,
                getattr(tool_call, "id", "") or "",
            ):
                result = _cancelled_tool_result("terminal fence")
                with terminal_fenced_lock:
                    terminal_fenced_call_ids.add(
                        getattr(tool_call, "id", "") or ""
                    )
                results[index] = (
                    function_name,
                    function_args,
                    result,
                    duration,
                    True,
                    True,
                    middleware_trace,
                )
                return
            if keyboard_interrupted:
                result = _emit_cancelled_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    start_time=start,
                    middleware_trace=list(middleware_trace),
                )
            if tool_error is not None:
                _log_tool_handler_exception(
                    agent,
                    "_invoke_tool",
                    function_name,
                    tool_error,
                )
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                if _strict_tool_mode(agent):
                    logger.info(
                        "tool %s failed (%.2fs, %d chars, category=tool_failure; "
                        "strict result redacted)",
                        function_name,
                        duration,
                        len(result),
                    )
                else:
                    logger.info(
                        "tool %s failed (%.2fs): %s",
                        function_name,
                        duration,
                        result[:200],
                    )
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error, False, middleware_trace)
        finally:
            # Tear down worker-tid tracking.  Clear any interrupt bit we may
            # have set so the next task scheduled onto this recycled tid
            # starts with a clean slate.  This MUST be in a finally block
            # because BaseException subclasses (CancelledError, KeyboardInterrupt)
            # bypass ``except Exception`` and would otherwise leak the tid
            # into _interrupted_threads, poisoning the recycled thread.
            with agent._tool_worker_threads_lock:
                agent._tool_worker_threads.discard(_worker_tid)
            try:
                _ra()._set_interrupt(False, _worker_tid)
            except Exception:
                pass

    # Start spinner for CLI mode (skip when TUI handles tool progress)
    spinner = None
    if (
        runnable_calls
        and not _strict_tool_mode(agent)
        and agent._should_emit_quiet_tool_messages()
        and agent._should_start_quiet_spinner()
    ):
        face = random.choice(KawaiiSpinner.get_waiting_faces())
        spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=agent._print_fn)
        spinner.start()

    try:
        futures = []
        if runnable_calls:
            max_workers = min(len(runnable_calls), _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, tc, name, args in runnable_calls:
                    # Propagate the agent turn's ContextVars (e.g.
                    # _approval_session_key) AND thread-local approval/sudo
                    # callbacks into the worker thread; clears callbacks on exit.
                    f = executor.submit(
                        propagate_context_to_thread(_run_tool), i, tc, name, args, parsed_calls[i][3]
                    )
                    futures.append(f)

                # Wait for all to complete with periodic heartbeats so the
                # gateway's inactivity monitor doesn't kill us during long
                # concurrent tool batches. Also check for user interrupts
                # so we don't block indefinitely when the user sends /stop
                # or a new message during concurrent tool execution.
                _conc_start = time.time()
                _interrupt_logged = False
                while True:
                    done, not_done = concurrent.futures.wait(
                        futures, timeout=5.0,
                    )
                    if not not_done:
                        break

                    # Check for interrupt — the per-thread interrupt signal
                    # already causes individual tools (terminal, execute_code)
                    # to abort, but tools without interrupt checks (web_search,
                    # read_file) will run to completion. Cancel any futures
                    # that haven't started yet so we don't block on them.
                    if agent._interrupt_requested:
                        if not _interrupt_logged:
                            _interrupt_logged = True
                            agent._vprint(
                                f"{agent.log_prefix}⚡ Interrupt: cancelling "
                                f"{len(not_done)} pending concurrent tool(s)",
                                force=True,
                            )
                        for f in not_done:
                            f.cancel()
                        # Give already-running tools a moment to notice the
                        # per-thread interrupt signal and exit gracefully.
                        concurrent.futures.wait(not_done, timeout=3.0)
                        break

                    _conc_elapsed = int(time.time() - _conc_start)
                    # Heartbeat every ~30s (6 × 5s poll intervals)
                    if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                        _still_running = [
                            parsed_calls[futures.index(f)][1]
                            for f in not_done
                            if f in futures
                        ]
                        agent._touch_activity(
                            f"concurrent tools running ({_conc_elapsed}s, "
                            f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                        )
    finally:
        if spinner:
            # Build a summary message for the spinner stop
            completed = sum(1 for r in results if r is not None)
            total_dur = sum(r[3] for r in results if r is not None)
            spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

    # ── Post-execution: display per-tool results ─────────────────────
    for i, (tc, name, args, middleware_trace, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
        r = results[i]
        tool_call_id = getattr(tc, "id", "") or ""
        guardrail_decision: ToolGuardrailDecision | None = None
        with terminal_fenced_lock:
            terminal_fenced = tool_call_id in terminal_fenced_call_ids
            dispatch_chain_entered = (
                tool_call_id in dispatch_chain_entered_call_ids
            )
            lifecycle_started = tool_call_id in lifecycle_started_call_ids
        if r is None and _strict_tool_terminal_stopped(agent):
            terminal_fenced = True
        if terminal_fenced:
            classification_result = r[2] if isinstance(r, tuple) else _cancelled_tool_result("terminal fence")
            _append_canonical_tool_result(
                agent,
                messages,
                name,
                _cancelled_tool_result("terminal fence"),
                tc.id,
                "dispatched" if dispatch_chain_entered else "not_dispatched",
                "unknown" if dispatch_chain_entered else "cancelled",
                classification_result,
                function_args=args,
                callback_result=_cancelled_tool_result("terminal fence"),
                emit_terminal_callback=lifecycle_started,
            )
            continue
        blocked = False
        if r is None:
            # Tool was cancelled (interrupt) or thread didn't return
            if agent._interrupt_requested:
                function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=name,
                    function_args=args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tc, "id", "") or "",
                    status="cancelled",
                    error_type="keyboard_interrupt",
                    error_message="Tool execution cancelled by user interrupt",
                    middleware_trace=list(middleware_trace),
                )
            else:
                function_result = f"Error executing tool '{name}': thread did not return a result"
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=name,
                    function_args=args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tc, "id", "") or "",
                    status="error",
                    error_type="thread_missing_result",
                    error_message=function_result,
                    middleware_trace=list(middleware_trace),
                )
            tool_duration = 0.0
            classification_result = function_result
            function_result, guardrail_decision = agent._append_guardrail_observation(
                name,
                args,
                function_result,
                failed=True,
                batch_id=api_call_count,
                tool_call_id=tool_call_id,
                return_decision=True,
            )
        else:
            function_name, function_args, function_result, tool_duration, is_error, blocked, middleware_trace = r
            classification_result = function_result

            if not blocked:
                function_result, guardrail_decision = agent._append_guardrail_observation(
                    function_name,
                    function_args,
                    function_result,
                    failed=is_error,
                    batch_id=api_call_count,
                    tool_call_id=getattr(tc, "id", "") or "",
                    return_decision=True,
                )
            if is_error:
                _err_text = _multimodal_text_summary(function_result)
                if _strict_tool_mode(agent):
                    logger.warning(
                        "Tool %s returned error (%.2fs, %d chars, "
                        "category=tool_failure; strict result redacted)",
                        function_name,
                        tool_duration,
                        len(_err_text),
                    )
                else:
                    result_preview = (
                        _err_text[:200]
                        if len(_err_text) > 200
                        else _err_text
                    )
                    logger.warning(
                        "Tool %s returned error (%.2fs): %s",
                        function_name,
                        tool_duration,
                        result_preview,
                    )

            # Track file-mutation outcome for the turn-end verifier.
            # `blocked` calls never actually ran — don't let a guardrail
            # block count as either a failure or a success.
            if not blocked:
                try:
                    agent._record_file_mutation_result(
                        function_name, function_args, function_result, is_error,
                    )
                except Exception as _ver_err:
                    logging.debug("file-mutation verifier record failed: %s", _ver_err)

            if not blocked and agent.tool_progress_callback:
                try:
                    agent.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=is_error,
                        result=function_result,
                    )
                except Exception as cb_err:
                    _log_tool_callback_exception(
                        agent,
                        "Tool progress callback",
                        cb_err,
                    )

            if agent.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                if _strict_tool_mode(agent):
                    logging.debug(
                        "Tool result redacted in strict mode (%d chars)",
                        len(_multimodal_text_summary(function_result)),
                    )
                else:
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

        # Print cute message per tool
        if agent._should_emit_quiet_tool_messages():
            cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
            agent._safe_print(f"  {cute_msg}")
        elif not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
            _preview_str = _multimodal_text_summary(function_result)
            if agent.verbose_logging:
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                if _strict_tool_mode(agent):
                    print(f"  Result redacted ({len(_preview_str)} chars)")
                else:
                    print(agent._wrap_verbose("Result: ", _preview_str))
            else:
                if _strict_tool_mode(agent):
                    print(
                        f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s "
                        f"- result redacted ({len(_preview_str)} chars)"
                    )
                else:
                    response_preview = _preview_str[:agent.log_prefix_chars] + "..." if len(_preview_str) > agent.log_prefix_chars else _preview_str
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

        agent._current_tool = None
        if not blocked:
            agent._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

        model_result = classification_result
        model_result = maybe_persist_tool_result(
            content=model_result,
            tool_name=name,
            tool_use_id=tc.id,
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        ) if not _is_multimodal_tool_result(model_result) else model_result

        subdir_hints = agent._subdirectory_hints.check_tool_call(name, args)
        if subdir_hints:
            if _is_multimodal_tool_result(model_result):
                # Append the hint to the text summary part so the model
                # still sees it; don't touch the image blocks.
                _append_subdir_hint_to_multimodal(model_result, subdir_hints)
            else:
                model_result += subdir_hints

        # Unwrap _multimodal dicts to an OpenAI-style content list so any
        # vision-capable provider receives [{type:text},{type:image_url}]
        # rather than a raw Python dict.  The Anthropic adapter already
        # accepts content lists; vision-capable OpenAI-compatible servers
        # (mlx-vlm, GPT-4o, …) accept image_url in tool messages natively.
        # Text-only servers get a string-safe fallback here so a rejected
        # image tool result never poisons canonical session history.
        # String results pass through unchanged.
        _tool_content = agent._tool_result_content_for_active_model(name, model_result)
        with terminal_fenced_lock:
            dispatch_chain_entered = (
                tool_call_id in dispatch_chain_entered_call_ids
            )
            lifecycle_started = tool_call_id in lifecycle_started_call_ids
        if tool_call_id in handler_exception_call_ids or (r is None and dispatch_chain_entered):
            outcome_hint = "unknown"
        elif r is None:
            outcome_hint = "cancelled" if agent._interrupt_requested else "failed"
        else:
            outcome_hint = None
        _append_canonical_tool_result(
            agent,
            messages,
            name,
            _tool_content,
            tc.id,
            "dispatched" if dispatch_chain_entered else "not_dispatched",
            outcome_hint,
            classification_result,
            trusted_fields=_trusted_guardrail_fields(guardrail_decision),
            function_args=args,
            callback_result=function_result,
            emit_terminal_callback=lifecycle_started,
        )

    # ── Per-turn aggregate budget enforcement ─────────────────────────
    num_tools = len(parsed_calls)
    if num_tools > 0:
        turn_tool_msgs = messages[-num_tools:]
        enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id), config=_tool_budget)

    # ── /steer injection ──────────────────────────────────────────────
    # Append any pending user steer text to the last tool result so the
    # agent sees it on its next iteration. Runs AFTER budget enforcement
    # so the steer marker is never truncated. See steer() for details.
    if num_tools > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools)



def execute_tool_calls_sequential(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
    """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools."""
    # Resolve the context-scaled tool-output budget once per turn.
    _tool_budget = _budget_for_agent(agent)
    deferred_keyboard_interrupt: KeyboardInterrupt | None = None
    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
        # SAFETY: check interrupt BEFORE starting each tool.
        # If the user sent "stop" during a previous tool's execution,
        # do NOT start any more tools -- skip them all immediately.
        if agent._interrupt_requested:
            remaining_calls = assistant_message.tool_calls[i-1:]
            if remaining_calls:
                agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
            for skipped_tc in remaining_calls:
                skipped_name = skipped_tc.function.name
                _append_canonical_tool_result(
                    agent,
                    messages,
                    skipped_name,
                    f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                    skipped_tc.id,
                    "not_dispatched",
                    "cancelled",
                )
            break

        function_name = tool_call.function.name

        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"Unexpected JSON error after validation: {e}")
            function_args = {}
        if not isinstance(function_args, dict):
            function_args = {}
        function_args, _ = sanitize_runtime_linkage_arguments(
            function_name,
            function_args,
        )

        # Tool Search unwrap — see execute_tool_calls_concurrent for full
        # rationale, including the scope gate (the unwrap dispatches the
        # underlying tool directly, so session toolset scope is enforced here).
        _ts_scope_block: Optional[str] = None
        try:
            from tools import tool_search as _ts
            if (
                _ts_scope_block is None
                and function_name == _ts.TOOL_CALL_NAME
            ):
                _underlying, _underlying_args, _err = _ts.resolve_underlying_call(function_args)
                if not _err and _underlying:
                    if _underlying in _tool_search_scoped_names(agent):
                        function_name = _underlying
                        function_args = _underlying_args
                        function_args, _ = split_runtime_linkage(
                            function_args
                        )
                    else:
                        _ts_scope_block = (
                            f"'{_underlying}' is not available in this session. "
                            "Use tool_search to find tools you can call."
                        )
        except Exception:
            pass

        if _ts_scope_block is None:
            function_args, middleware_trace = (
                _apply_tool_request_middleware_for_agent(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                )
            )
        else:
            middleware_trace = []

        _terminal_dispatch_denied = not _claim_strict_tool_dispatch(
            agent,
            getattr(tool_call, "id", "") or "",
        )

        # Check plugin hooks for a block directive before executing.
        _block_msg: Optional[str] = None
        _block_error_type = "plugin_block"
        if _terminal_dispatch_denied:
            # The terminal fence precedes guardrails, trusted PreAction,
            # checkpoints, callbacks and the actual handler.
            pass
        elif _ts_scope_block is not None:
            _block_msg = _ts_scope_block
            _block_error_type = "tool_scope_block"
        elif (plan_mode_block := _mystand_plan_mode_block(function_name, function_args)) is not None:
            _block_msg = plan_mode_block
            _block_error_type = "plan_mode_read_only"
        elif not _strict_tool_mode(agent):
            try:
                from xiaoban_cli.plugins import get_pre_tool_call_block_message
                _block_msg = get_pre_tool_call_block_message(
                    function_name,
                    function_args,
                    task_id=effective_task_id or "",
                    session_id=getattr(agent, "session_id", "") or "",
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    turn_id=getattr(agent, "_current_turn_id", "") or "",
                    api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                    middleware_trace=list(middleware_trace),
                )
            except Exception:
                pass

        _guardrail_block_decision: ToolGuardrailDecision | None = None
        if not _terminal_dispatch_denied and _block_msg is None:
            guardrail_decision = agent._tool_guardrails.before_call(function_name, function_args)
            if not guardrail_decision.allows_execution:
                _guardrail_block_decision = guardrail_decision

        _execution_blocked = (
            _terminal_dispatch_denied
            or _block_msg is not None
            or _guardrail_block_decision is not None
        )
        _strict_tool_turn = _strict_tool_mode(agent)

        if _execution_blocked:
            # Tool blocked by plugin or guardrail policy — skip counters,
            # callbacks, checkpointing, activity mutation, and real execution.
            pass
        # Reset nudge counters when the relevant tool is actually used
        elif function_name == "memory":
            agent._turns_since_memory = 0
        elif function_name == "skill_manage":
            agent._iters_since_skill = 0

        if (
            not _terminal_dispatch_denied
            and not _strict_tool_turn
            and not agent.quiet_mode
            and getattr(agent, "tool_progress_mode", "all") != "off"
        ):
            args_str = json.dumps(function_args, ensure_ascii=False)
            if agent.verbose_logging:
                print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                print(agent._wrap_verbose("Args: ", json.dumps(function_args, indent=2, ensure_ascii=False)))
            else:
                args_preview = args_str[:agent.log_prefix_chars] + "..." if len(args_str) > agent.log_prefix_chars else args_str
                print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

        if not _execution_blocked and not _strict_tool_turn:
            agent._current_tool = function_name
            agent._touch_activity(f"executing tool: {function_name}")

        # Set activity callback for long-running tool execution (terminal
        # commands, etc.) so the gateway's inactivity monitor doesn't kill
        # the agent while a command is running.
        if not _execution_blocked and not _strict_tool_turn:
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(agent._touch_activity)
            except Exception:
                pass

        if (
            not _execution_blocked
            and not _strict_tool_turn
            and agent.tool_progress_callback
        ):
            try:
                preview = _build_tool_preview(function_name, function_args)
                agent.tool_progress_callback("tool.started", function_name, preview, function_args)
            except Exception as cb_err:
                _log_tool_callback_exception(
                    agent,
                    "Tool progress callback",
                    cb_err,
                )

        if (
            not _execution_blocked
            and not _strict_tool_turn
            and agent.tool_start_callback
        ):
            try:
                agent.tool_start_callback(tool_call.id, function_name, function_args)
            except Exception as cb_err:
                _log_tool_callback_exception(
                    agent,
                    "Tool start callback",
                    cb_err,
                )

        # Checkpoint: snapshot working dir before file-mutating tools
        if not _execution_blocked and function_name in {"write_file", "patch"} and agent._checkpoint_mgr.enabled:
            try:
                file_path = function_args.get("path", "")
                if file_path:
                    work_dir = agent._checkpoint_mgr.get_working_dir_for_path(file_path)
                    agent._checkpoint_mgr.ensure_checkpoint(
                        work_dir, f"before {function_name}"
                    )
            except Exception:
                pass  # never block tool execution

        # Checkpoint before destructive terminal commands
        if not _execution_blocked and function_name == "terminal" and agent._checkpoint_mgr.enabled:
            try:
                cmd = function_args.get("command", "")
                if _is_destructive_command(cmd):
                    cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                    agent._checkpoint_mgr.ensure_checkpoint(
                        cwd, f"before terminal: {cmd[:60]}"
                    )
            except Exception:
                pass  # never block tool execution

        _handler_dispatched = not _execution_blocked
        _execute_dispatched = False
        if _handler_dispatched:
            _handler_dispatched = _begin_strict_tool_handler(
                agent,
                getattr(tool_call, "id", "") or "",
                function_name,
                function_args,
                build_preview=_build_tool_preview,
            )
            if not _handler_dispatched:
                _terminal_dispatch_denied = True
                _execution_blocked = True

        if _handler_dispatched:
            _execute_dispatched = _claim_strict_tool_execute(
                agent,
                getattr(tool_call, "id", "") or "",
            )
        if _handler_dispatched and not _execute_dispatched:
            _terminal_dispatch_denied = True
            _execution_blocked = True

        tool_start_time = time.time()
        _handler_exception = False
        guardrail_decision: ToolGuardrailDecision | None = None

        if _terminal_dispatch_denied:
            function_result = _cancelled_tool_result("terminal fence")
            tool_duration = 0.0
        elif _block_msg is not None:
            # Tool blocked by plugin policy — return error without executing.
            function_result = json.dumps({"error": _block_msg}, ensure_ascii=False)
            tool_duration = 0.0
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=function_result,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tool_call, "id", "") or "",
                status="blocked",
                error_type=_block_error_type,
                error_message=_block_msg,
                middleware_trace=list(middleware_trace),
            )
        elif _guardrail_block_decision is not None:
            # Tool blocked by tool-loop guardrail — synthesize exactly one
            # tool result for the original tool_call_id without executing.
            function_result = agent._guardrail_block_result(_guardrail_block_decision)
            tool_duration = 0.0
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=function_result,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tool_call, "id", "") or "",
                status="blocked",
                error_type="guardrail_block",
                error_message=getattr(_guardrail_block_decision, "message", None) or "Tool blocked by guardrail policy",
                middleware_trace=list(middleware_trace),
            )
            _emit_guardrail_preflight_lifecycle(
                agent,
                tool_call_id=getattr(tool_call, "id", "") or "",
                function_name=function_name,
                function_args=function_args,
                function_result=function_result,
            )
        elif function_name == "todo":
            def _execute(next_args: dict) -> Any:
                from tools.todo_tool import todo_tool as _todo_tool
                return _todo_tool(
                    todos=next_args.get("todos"),
                    merge=next_args.get("merge", False),
                    store=agent._todo_store,
                )
            function_result, function_args, _handler_exception = (
                _run_agent_tool_execution_middleware_captured(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    execute=_execute,
                )
            )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
        elif function_name == "session_search":
            def _execute(next_args: dict) -> Any:
                session_db = agent._get_session_db_for_recall()
                if not session_db:
                    from xiaoban_state import format_session_db_unavailable
                    return json.dumps({"success": False, "error": format_session_db_unavailable()})
                from tools.session_search_tool import session_search as _session_search
                return _session_search(
                    query=next_args.get("query", ""),
                    role_filter=next_args.get("role_filter"),
                    limit=next_args.get("limit", 3),
                    session_id=next_args.get("session_id"),
                    around_message_id=next_args.get("around_message_id"),
                    window=next_args.get("window", 5),
                    sort=next_args.get("sort"),
                    db=session_db,
                    current_session_id=agent.session_id,
                )
            function_result, function_args, _handler_exception = (
                _run_agent_tool_execution_middleware_captured(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    execute=_execute,
                )
            )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
        elif function_name == "memory":
            def _execute(next_args: dict) -> Any:
                target = next_args.get("target", "memory")
                operations = next_args.get("operations")
                from tools.memory_tool import memory_tool as _memory_tool
                result = _memory_tool(
                    action=next_args.get("action"),
                    target=target,
                    content=next_args.get("content"),
                    old_text=next_args.get("old_text"),
                    operations=operations,
                    store=agent._memory_store,
                )
                # Mirror successful built-in memory writes to external
                # providers. All gating/op-expansion lives behind the manager
                # interface (MemoryManager.notify_memory_tool_write).
                if agent._memory_manager:
                    agent._memory_manager.notify_memory_tool_write(
                        result,
                        next_args,
                        build_metadata=lambda: agent._build_memory_write_metadata(
                            task_id=effective_task_id,
                            tool_call_id=getattr(tool_call, "id", None),
                        ),
                    )
                return result
            function_result, function_args, _handler_exception = (
                _run_agent_tool_execution_middleware_captured(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    execute=_execute,
                )
            )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
        elif function_name == "clarify":
            def _execute(next_args: dict) -> Any:
                from tools.clarify_tool import clarify_tool as _clarify_tool
                return _clarify_tool(
                    question=next_args.get("question", ""),
                    choices=next_args.get("choices"),
                    callback=agent.clarify_callback,
                )
            function_result, function_args, _handler_exception = (
                _run_agent_tool_execution_middleware_captured(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    execute=_execute,
                )
            )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
        elif function_name == "read_terminal":
            def _execute(next_args: dict) -> Any:
                from tools.read_terminal_tool import read_terminal_tool as _read_terminal_tool
                return _read_terminal_tool(
                    start_line=next_args.get("start_line"),
                    count=next_args.get("count"),
                    callback=getattr(agent, "read_terminal_callback", None),
                )
            function_result, function_args, _handler_exception = (
                _run_agent_tool_execution_middleware_captured(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    execute=_execute,
                )
            )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('read_terminal', function_args, tool_duration, result=function_result)}")
        elif function_name == "delegate_task":
            tasks_arg = function_args.get("tasks")
            if tasks_arg and isinstance(tasks_arg, list):
                spinner_label = f"🔀 delegating {len(tasks_arg)} tasks · (/agents to monitor)"
            else:
                goal_preview = (function_args.get("goal") or "")[:30]
                spinner_label = (
                    f"🔀 {goal_preview} · (/agents to monitor)"
                    if goal_preview
                    else "🔀 delegating · (/agents to monitor)"
                )
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            agent._delegate_spinner = spinner
            _delegate_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent._dispatch_delegate_task(next_args)
                function_result, function_args, _handler_exception = (
                    _run_agent_tool_execution_middleware_captured(
                        agent,
                        function_name=function_name,
                        function_args=function_args,
                        effective_task_id=effective_task_id,
                        tool_call_id=getattr(tool_call, "id", "") or "",
                        execute=_execute,
                    )
                )
                _delegate_result = function_result
            finally:
                agent._delegate_spinner = None
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent._context_engine_tool_names and function_name in agent._context_engine_tool_names:
            # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
            spinner = None
            if agent._should_emit_quiet_tool_messages():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                preview = _build_tool_preview(function_name, function_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _ce_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent.context_compressor.handle_tool_call(function_name, next_args, messages=messages)
                function_result, function_args = _run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    execute=_execute,
                )
                _ce_result = function_result
            except Exception as tool_error:
                _handler_exception = True
                function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                _log_tool_handler_exception(
                    agent,
                    "context_engine.handle_tool_call",
                    function_name,
                    tool_error,
                )
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
            # Memory provider tools (hindsight_retain, honcho_search, etc.)
            # These are not in the tool registry — route through MemoryManager.
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                preview = _build_tool_preview(function_name, function_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _mem_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent._memory_manager.handle_tool_call(function_name, next_args)
                function_result, function_args = _run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tool_call, "id", "") or "",
                    execute=_execute,
                )
                _mem_result = function_result
            except Exception as tool_error:
                _handler_exception = True
                function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                _log_tool_handler_exception(
                    agent,
                    "memory_manager.handle_tool_call",
                    function_name,
                    tool_error,
                )
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent.quiet_mode:
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                preview = _build_tool_preview(function_name, function_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _spinner_result = None
            try:
                function_result = _ra().handle_function_call(
                    function_name, function_args, effective_task_id,
                    tool_call_id=tool_call.id,
                    session_id=agent.session_id or "",
                    turn_id=getattr(agent, "_current_turn_id", "") or "",
                    api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                    enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                    disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                    tool_request_middleware_trace=list(middleware_trace),
                    **_strict_extension_hook_kwargs(agent),
                )
                _spinner_result = function_result
            except KeyboardInterrupt as exc:
                deferred_keyboard_interrupt = exc
                function_result = _cancelled_tool_result("keyboard interrupt")
                _spinner_result = function_result
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
            except Exception as tool_error:
                _handler_exception = True
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                _log_tool_handler_exception(
                    agent,
                    "handle_function_call",
                    function_name,
                    tool_error,
                )
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        else:
            try:
                function_result = _ra().handle_function_call(
                    function_name, function_args, effective_task_id,
                    tool_call_id=tool_call.id,
                    session_id=agent.session_id or "",
                    turn_id=getattr(agent, "_current_turn_id", "") or "",
                    api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                    enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                    disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                    tool_request_middleware_trace=list(middleware_trace),
                    **_strict_extension_hook_kwargs(agent),
                )
            except KeyboardInterrupt as exc:
                deferred_keyboard_interrupt = exc
                function_result = _cancelled_tool_result("keyboard interrupt")
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
            except Exception as tool_error:
                _handler_exception = True
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                _log_tool_handler_exception(
                    agent,
                    "handle_function_call",
                    function_name,
                    tool_error,
                )
            tool_duration = time.time() - tool_start_time

        _result_commit_denied = (
            _execute_dispatched
            and not _claim_strict_tool_result(
                agent,
                getattr(tool_call, "id", "") or "",
            )
        )
        if _terminal_dispatch_denied or _result_commit_denied:
            classification_result = function_result
            function_result = _cancelled_tool_result("terminal fence")
            _append_canonical_tool_result(
                agent,
                messages,
                function_name,
                function_result,
                tool_call.id,
                "dispatched" if _execute_dispatched else "not_dispatched",
                "unknown" if _execute_dispatched else "cancelled",
                classification_result,
                function_args=function_args,
                callback_result=function_result,
                emit_terminal_callback=_handler_dispatched,
            )
            remaining_calls = assistant_message.tool_calls[i:]
            for skipped_tc in remaining_calls:
                _append_canonical_tool_result(
                    agent,
                    messages,
                    skipped_tc.function.name,
                    _cancelled_tool_result("terminal fence"),
                    skipped_tc.id,
                    "not_dispatched",
                    "cancelled",
                )
            break

        if deferred_keyboard_interrupt is not None:
            function_result = _emit_cancelled_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tool_call, "id", "") or "",
                start_time=tool_start_time,
                middleware_trace=list(middleware_trace),
            )

        classification_result = function_result
        if isinstance(function_result, str):
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
            _result_len = len(function_result)
        else:
            # Multimodal dict result (_multimodal=True) — not sliceable as string
            result_preview = function_result
            _result_len = len(str(function_result))

        # Log tool errors to the persistent error log so [error] tags
        # in the UI always have a corresponding detailed entry on disk.
        _is_error_result, _ = _detect_tool_failure(function_name, function_result)
        # The agent-runtime tools above (todo, session_search, memory,
        # context-engine, memory-manager, clarify, delegate_task) are
        # dispatched inline — they never reach handle_function_call, so the
        # executor is the one that has to fire post_tool_call. For
        # registry-dispatched tools the else-branch above invoked
        # handle_function_call, which already fires the hook.
        from agent.agent_runtime_helpers import agent_runtime_owns_post_tool_hook
        _executor_must_emit_post_hook = (
            not _execution_blocked
            and agent_runtime_owns_post_tool_hook(agent, function_name)
        )
        if _executor_must_emit_post_hook:
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=function_result,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tool_call, "id", "") or "",
                duration_ms=int(tool_duration * 1000),
                middleware_trace=list(middleware_trace),
            )
        if not _execution_blocked:
            function_result, guardrail_decision = agent._append_guardrail_observation(
                function_name,
                function_args,
                function_result,
                failed=_is_error_result,
                batch_id=api_call_count,
                tool_call_id=getattr(tool_call, "id", "") or "",
                return_decision=True,
            )
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
        if _is_error_result:
            if _strict_tool_mode(agent):
                logger.warning(
                    "Tool %s returned error (%.2fs, %d chars, category=%s; "
                    "strict result redacted)",
                    function_name,
                    tool_duration,
                    _result_len,
                    "handler_exception" if _handler_exception else "tool_failure",
                )
            else:
                logger.warning(
                    "Tool %s returned error (%.2fs): %s",
                    function_name,
                    tool_duration,
                    result_preview,
                )
        else:
            logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, _result_len)

        # Track file-mutation outcome for the turn-end verifier.  See
        # the concurrent path for the rationale; both paths must feed
        # the same state so the footer reflects every tool call in the
        # turn, not just the parallel ones.
        if not _execution_blocked:
            try:
                agent._record_file_mutation_result(
                    function_name, function_args, function_result, _is_error_result,
                )
            except Exception as _ver_err:
                logging.debug("file-mutation verifier record failed: %s", _ver_err)

        if not _execution_blocked and agent.tool_progress_callback:
            try:
                agent.tool_progress_callback(
                    "tool.completed", function_name, None, None,
                    duration=tool_duration, is_error=_is_error_result,
                    result=function_result,
                )
            except Exception as cb_err:
                _log_tool_callback_exception(
                    agent,
                    "Tool progress callback",
                    cb_err,
                )

        agent._current_tool = None
        agent._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

        if agent.verbose_logging:
            logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
            _log_result = _multimodal_text_summary(function_result)
            if _strict_tool_mode(agent):
                logging.debug(
                    "Tool result redacted in strict mode (%d chars)",
                    len(_log_result),
                )
            else:
                logging.debug(f"Tool result ({len(_log_result)} chars): {_log_result}")

        model_result = classification_result
        model_result = maybe_persist_tool_result(
            content=model_result,
            tool_name=function_name,
            tool_use_id=tool_call.id,
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        ) if not _is_multimodal_tool_result(model_result) else model_result

        # Discover subdirectory context files from tool arguments
        subdir_hints = agent._subdirectory_hints.check_tool_call(function_name, function_args)
        if subdir_hints:
            if _is_multimodal_tool_result(model_result):
                _append_subdir_hint_to_multimodal(model_result, subdir_hints)
            else:
                model_result += subdir_hints

        # Unwrap _multimodal dicts to an OpenAI-style content list
        # (see parallel path for rationale). String results pass through.
        _tool_content = agent._tool_result_content_for_active_model(function_name, model_result)
        _append_canonical_tool_result(
            agent,
            messages,
            function_name,
            _tool_content,
            tool_call.id,
            "dispatched" if _execute_dispatched else "not_dispatched",
            "unknown" if _handler_exception else None,
            classification_result,
            trusted_fields=_trusted_guardrail_fields(guardrail_decision),
            function_args=function_args,
            callback_result=function_result,
            emit_terminal_callback=(
                _handler_dispatched
                or _guardrail_block_decision is not None
            ),
        )

        if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
            if agent.verbose_logging:
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                if _strict_tool_mode(agent):
                    print(f"  Result redacted ({_result_len} chars)")
                else:
                    print(agent._wrap_verbose("Result: ", function_result))
            else:
                if _strict_tool_mode(agent):
                    print(
                        f"  ✅ Tool {i} completed in {tool_duration:.2f}s "
                        f"- result redacted ({_result_len} chars)"
                    )
                else:
                    _fr_str = function_result if isinstance(function_result, str) else str(function_result)
                    response_preview = _fr_str[:agent.log_prefix_chars] + "..." if len(_fr_str) > agent.log_prefix_chars else _fr_str
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

        if (
            (
                _terminal_dispatch_denied
                or agent._interrupt_requested
                or deferred_keyboard_interrupt is not None
            )
            and i < len(assistant_message.tool_calls)
        ):
            remaining = len(assistant_message.tool_calls) - i
            agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
            for skipped_tc in assistant_message.tool_calls[i:]:
                skipped_name = skipped_tc.function.name
                _append_canonical_tool_result(
                    agent,
                    messages,
                    skipped_name,
                    f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                    skipped_tc.id,
                    "not_dispatched",
                    "cancelled",
                )
            break

        if agent.tool_delay > 0 and i < len(assistant_message.tool_calls):
            time.sleep(agent.tool_delay)

    # ── Per-turn aggregate budget enforcement ─────────────────────────
    num_tools_seq = len(assistant_message.tool_calls)
    if num_tools_seq > 0:
        enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id), config=_tool_budget)

    # ── /steer injection ──────────────────────────────────────────────
    # See _execute_tool_calls_parallel for the rationale. Same hook,
    # applied to sequential execution as well.
    if num_tools_seq > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools_seq)

    if deferred_keyboard_interrupt is not None:
        raise deferred_keyboard_interrupt




__all__ = [
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
]
