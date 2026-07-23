"""
Session-scoped context variables for the Xiaoban gateway.

Replaces the previous ``os.environ``-based session state
(``XIAOBAN_SESSION_PLATFORM``, ``XIAOBAN_SESSION_CHAT_ID``, etc.) with
Python's ``contextvars.ContextVar``.

**Why this matters**

The gateway processes messages concurrently via ``asyncio``.  When two
messages arrive at the same time the old code did:

    os.environ["XIAOBAN_SESSION_THREAD_ID"] = str(context.source.thread_id)

Because ``os.environ`` is *process-global*, Message A's value was
silently overwritten by Message B before Message A's agent finished
running.  Background-task notifications and tool calls therefore routed
to the wrong thread.

``contextvars.ContextVar`` values are *task-local*: each ``asyncio``
task (and any ``run_in_executor`` thread it spawns) gets its own copy,
so concurrent messages never interfere.

**Backward compatibility**

The public helper ``get_session_env(name, default="")`` mirrors the old
``os.getenv("XIAOBAN_SESSION_*", ...)`` calls.  Existing tool code only
needs to replace the import + call site:

    # before
    import os
    platform = os.getenv("XIAOBAN_SESSION_PLATFORM", "")

    # after
    from gateway.session_context import get_session_env
    platform = get_session_env("XIAOBAN_SESSION_PLATFORM", "")
"""

import hashlib
import json
import os
import secrets
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Sentinel to distinguish "never set in this context" from "explicitly set to empty".
# When a contextvar holds _UNSET, we fall back to os.environ (CLI/cron compat).
# When it holds "" (after clear_session_vars resets it), we return "" — no fallback.
_UNSET: Any = object()

# ---------------------------------------------------------------------------
# Per-task session variables
# ---------------------------------------------------------------------------

_SESSION_PLATFORM: ContextVar = ContextVar("XIAOBAN_SESSION_PLATFORM", default=_UNSET)
_SESSION_SOURCE: ContextVar = ContextVar("XIAOBAN_SESSION_SOURCE", default=_UNSET)
_SESSION_CHAT_ID: ContextVar = ContextVar("XIAOBAN_SESSION_CHAT_ID", default=_UNSET)
_SESSION_CHAT_NAME: ContextVar = ContextVar("XIAOBAN_SESSION_CHAT_NAME", default=_UNSET)
_SESSION_THREAD_ID: ContextVar = ContextVar("XIAOBAN_SESSION_THREAD_ID", default=_UNSET)
_SESSION_USER_ID: ContextVar = ContextVar("XIAOBAN_SESSION_USER_ID", default=_UNSET)
_SESSION_USER_NAME: ContextVar = ContextVar("XIAOBAN_SESSION_USER_NAME", default=_UNSET)
_SESSION_KEY: ContextVar = ContextVar("XIAOBAN_SESSION_KEY", default=_UNSET)
_SESSION_ID: ContextVar = ContextVar("XIAOBAN_SESSION_ID", default=_UNSET)
# ID of the message that triggered the current turn. Used as a reply anchor
# so background-process notifications stay inside the originating Telegram
# private-chat topic (those lanes route only with thread id + reply anchor).
_SESSION_MESSAGE_ID: ContextVar = ContextVar("XIAOBAN_SESSION_MESSAGE_ID", default=_UNSET)
# Latest human message for the active turn.  Deliberately not exposed through
# _VAR_MAP or subprocess environments; only trusted in-process gates (such as
# the My Stand write-confirmation tool) may inspect it.
_SESSION_USER_MESSAGE: ContextVar = ContextVar("XIAOBAN_SESSION_USER_MESSAGE", default=_UNSET)
# Set for the remainder of the active user turn once a high-level My Stand
# private-data query is planned or invoked. Web tools consult this task-local
# flag as a hard egress boundary. It is deliberately not exported through
# _VAR_MAP or process environments.
_SESSION_MYSTAND_PRIVATE_QUERY: ContextVar = ContextVar(
    "XIAOBAN_SESSION_MYSTAND_PRIVATE_QUERY",
    default=False,
)
_MYSTAND_PRIVATE_TAINT_SCHEMA = "xiaoban.mystand-private-session-taints.v1"
_MYSTAND_PRIVATE_TAINT_FILE_ENV = "XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE"
_MYSTAND_PRIVATE_SESSION_TAINTS: set[str] = set()
_MYSTAND_PRIVATE_SESSION_TAINT_LOCK = threading.Lock()
_MYSTAND_PRIVATE_SESSION_TAINT_LOADED_PATH: Path | None = None
_MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED = False
_MYSTAND_PRIVATE_HISTORY_TOOL_NAMES = frozenset(
    {
        "mystand_query",
        "mystand_authorization_write",
        # Legacy model-visible tools are no longer exposed, but their trusted
        # structured history can still be replayed after a gateway restart.
        "mystand_authorization",
        "mystand_resource_index",
    }
)

# Whether the current session's delivery channel can route an ASYNC completion
# back to the agent AFTER the current turn ends (i.e. wake a fresh turn).
#
# True  — CLI (in-process completion_queue drain) and the real gateway
#         platforms (Telegram/Discord/Slack/...), which hold a persistent
#         outbound channel and run the watcher/drain loops.
# False — stateless request/response adapters (the API server: every route,
#         spec and proprietary, tears down its channel when the turn ends, so
#         a background completion that finishes later has nowhere to go).
#
# Tools that promise async delivery (terminal notify_on_complete /
# watch_patterns, delegate_task background=True) read this via
# ``async_delivery_supported()`` and refuse to hand out a promise the channel
# can't keep — turning a silent no-op into an explicit contract.
#
# Default _UNSET => treated as supported, so CLI (which never sets a platform)
# and any contextvar-unaware path keep working. Stateless adapters opt OUT by
# setting ``supports_async_delivery = False`` on the adapter class; the gateway
# propagates that into this contextvar at session-bind time.
_SESSION_ASYNC_DELIVERY: ContextVar = ContextVar("XIAOBAN_SESSION_ASYNC_DELIVERY", default=_UNSET)

# Cron auto-delivery vars — set per-job in run_job() so concurrent jobs
# don't clobber each other's delivery targets.
_CRON_AUTO_DELIVER_PLATFORM: ContextVar = ContextVar("XIAOBAN_CRON_AUTO_DELIVER_PLATFORM", default=_UNSET)
_CRON_AUTO_DELIVER_CHAT_ID: ContextVar = ContextVar("XIAOBAN_CRON_AUTO_DELIVER_CHAT_ID", default=_UNSET)
_CRON_AUTO_DELIVER_THREAD_ID: ContextVar = ContextVar("XIAOBAN_CRON_AUTO_DELIVER_THREAD_ID", default=_UNSET)

_VAR_MAP = {
    "XIAOBAN_SESSION_PLATFORM": _SESSION_PLATFORM,
    "XIAOBAN_SESSION_SOURCE": _SESSION_SOURCE,
    "XIAOBAN_SESSION_CHAT_ID": _SESSION_CHAT_ID,
    "XIAOBAN_SESSION_CHAT_NAME": _SESSION_CHAT_NAME,
    "XIAOBAN_SESSION_THREAD_ID": _SESSION_THREAD_ID,
    "XIAOBAN_SESSION_USER_ID": _SESSION_USER_ID,
    "XIAOBAN_SESSION_USER_NAME": _SESSION_USER_NAME,
    "XIAOBAN_SESSION_KEY": _SESSION_KEY,
    "XIAOBAN_SESSION_ID": _SESSION_ID,
    "XIAOBAN_SESSION_MESSAGE_ID": _SESSION_MESSAGE_ID,
    "XIAOBAN_CRON_AUTO_DELIVER_PLATFORM": _CRON_AUTO_DELIVER_PLATFORM,
    "XIAOBAN_CRON_AUTO_DELIVER_CHAT_ID": _CRON_AUTO_DELIVER_CHAT_ID,
    "XIAOBAN_CRON_AUTO_DELIVER_THREAD_ID": _CRON_AUTO_DELIVER_THREAD_ID,
}


def set_current_session_id(session_id: str) -> None:
    """Synchronize ``XIAOBAN_SESSION_ID`` across ContextVar and ``os.environ``.

    Long-lived single-process entrypoints like the CLI can rotate sessions via
    ``/new``, ``/resume``, ``/branch``, or compression splits without
    reconstructing the entire agent. Tools still consult
    ``get_session_env("XIAOBAN_SESSION_ID")`` with an ``os.environ`` fallback,
    so both storage paths must move together when the active session changes.
    """
    import os

    os.environ["XIAOBAN_SESSION_ID"] = session_id
    _SESSION_ID.set(session_id)


def set_session_vars(
    platform: str = "",
    source: str = "",
    chat_id: str = "",
    chat_name: str = "",
    thread_id: str = "",
    user_id: str = "",
    user_name: str = "",
    session_key: str = "",
    session_id: str = "",
    message_id: str = "",
    user_message: str = "",
    cwd: str = "",
    async_delivery: bool = True,
) -> list:
    """Set all session context variables and return reset tokens.

    Call ``clear_session_vars(tokens)`` in a ``finally`` block when the handler
    exits. Note ``clear_session_vars`` resets every var to ``""`` (to suppress
    the ``os.environ`` fallback) rather than restoring prior values — these
    helpers are not nestable/stack-safe, and the returned tokens are accepted
    only for API compatibility.

    ``cwd`` pins the logical working directory for this context.

    ``async_delivery`` declares whether this session's channel can route a
    background completion back to the agent after the turn ends (see
    ``_SESSION_ASYNC_DELIVERY`` / ``async_delivery_supported``). Stateless
    request/response adapters (the API server) pass ``False``.
    """
    tokens = [
        _SESSION_PLATFORM.set(platform),
        _SESSION_SOURCE.set(source),
        _SESSION_CHAT_ID.set(chat_id),
        _SESSION_CHAT_NAME.set(chat_name),
        _SESSION_THREAD_ID.set(thread_id),
        _SESSION_USER_ID.set(user_id),
        _SESSION_USER_NAME.set(user_name),
        _SESSION_KEY.set(session_key),
        _SESSION_ID.set(session_id),
        _SESSION_MESSAGE_ID.set(message_id),
        _SESSION_USER_MESSAGE.set(user_message),
        _SESSION_MYSTAND_PRIVATE_QUERY.set(False),
        _SESSION_ASYNC_DELIVERY.set(bool(async_delivery)),
    ]
    try:
        from agent.runtime_cwd import set_session_cwd

        set_session_cwd(cwd)
    except Exception:
        pass
    return tokens


def clear_session_vars(tokens: list) -> None:
    """Mark session context variables as explicitly cleared.

    Sets all variables to ``""`` so that ``get_session_env`` returns an empty
    string instead of falling back to (potentially stale) ``os.environ``
    values.  The *tokens* argument is accepted for API compatibility with
    callers that saved the return value of ``set_session_vars``, but the
    actual clearing uses ``var.set("")`` rather than ``var.reset(token)``
    to ensure the "explicitly cleared" state is distinguishable from
    "never set" (which holds the ``_UNSET`` sentinel).
    """
    for var in (
        _SESSION_PLATFORM,
        _SESSION_SOURCE,
        _SESSION_CHAT_ID,
        _SESSION_CHAT_NAME,
        _SESSION_THREAD_ID,
        _SESSION_USER_ID,
        _SESSION_USER_NAME,
        _SESSION_KEY,
        _SESSION_ID,
        _SESSION_MESSAGE_ID,
        _SESSION_USER_MESSAGE,
    ):
        var.set("")
    _SESSION_MYSTAND_PRIVATE_QUERY.set(False)
    # Reset async-delivery capability to the "never set" sentinel rather than a
    # falsy value: a cleared context should fall back to the default-supported
    # behavior (CLI / unaware paths), not be mistaken for an opted-out
    # stateless adapter.
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    try:
        from agent.runtime_cwd import clear_session_cwd

        clear_session_cwd()
    except Exception:
        pass


def get_session_user_message() -> str:
    """Return the latest human message without exporting it to child processes."""
    value = _SESSION_USER_MESSAGE.get()
    if value is _UNSET:
        return ""
    return str(value or "")


def mark_mystand_private_query_turn() -> None:
    """Block web egress for this turn and later turns in the same session."""
    _SESSION_MYSTAND_PRIVATE_QUERY.set(True)
    taint_keys = _mystand_private_session_taint_keys()
    if not taint_keys:
        return
    with _MYSTAND_PRIVATE_SESSION_TAINT_LOCK:
        path = _mystand_private_session_taint_path()
        _load_mystand_private_session_taints_locked(path)
        new_keys = set(taint_keys) - _MYSTAND_PRIVATE_SESSION_TAINTS
        if not new_keys:
            return
        _MYSTAND_PRIVATE_SESSION_TAINTS.update(new_keys)
        if not _MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED:
            _persist_mystand_private_session_taints_locked(path)


def mystand_private_query_turn_active() -> bool:
    """Return whether this turn or its stable session has touched private data."""
    if _SESSION_MYSTAND_PRIVATE_QUERY.get() is True:
        return True
    taint_keys = _mystand_private_session_taint_keys()
    if not taint_keys:
        return False
    with _MYSTAND_PRIVATE_SESSION_TAINT_LOCK:
        _load_mystand_private_session_taints_locked(
            _mystand_private_session_taint_path(),
        )
        return any(
            key in _MYSTAND_PRIVATE_SESSION_TAINTS
            for key in taint_keys
        )


def mystand_private_taint_persistence_failed() -> bool:
    """Return whether the durable private-session boundary is unavailable."""
    with _MYSTAND_PRIVATE_SESSION_TAINT_LOCK:
        _load_mystand_private_session_taints_locked(
            _mystand_private_session_taint_path(),
        )
        return _MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED


def mark_mystand_private_query_from_history(conversation_history: Any) -> bool:
    """Restore the private-session boundary from structured tool history.

    Only trusted tool metadata is inspected. User or assistant prose that
    merely contains a My Stand tool name never taints a session.
    """
    if not isinstance(conversation_history, (list, tuple)):
        return False
    for message in conversation_history:
        if _history_message_has_mystand_private_tool(message):
            mark_mystand_private_query_turn()
            return True
    return False


def _mystand_private_session_taint_keys() -> tuple[str, ...]:
    keys = []
    for namespace, var in (
        ("session_id", _SESSION_ID),
        ("session_key", _SESSION_KEY),
    ):
        value = var.get()
        if value is _UNSET:
            continue
        text = str(value or "").strip()
        if not text:
            continue
        digest = hashlib.sha256(
            f"{namespace}\0{text}".encode("utf-8"),
        ).hexdigest()
        keys.append(f"{namespace}:{digest}")
    return tuple(keys)


def _history_message_has_mystand_private_tool(message: Any) -> bool:
    if not isinstance(message, dict):
        return False

    message_type = str(message.get("type") or "").strip().lower()
    if (
        message_type == "function_call"
        and str(message.get("name") or "").strip()
        in _MYSTAND_PRIVATE_HISTORY_TOOL_NAMES
    ):
        return True

    role = str(message.get("role") or "").strip().lower()
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, (list, tuple)):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                else:
                    name = tool_call.get("name")
                if (
                    str(name or "").strip()
                    in _MYSTAND_PRIVATE_HISTORY_TOOL_NAMES
                ):
                    return True

    if role == "tool":
        return any(
            str(message.get(field) or "").strip()
            in _MYSTAND_PRIVATE_HISTORY_TOOL_NAMES
            for field in ("name", "tool_name")
        )
    return False


def _mystand_private_session_taint_path() -> Path:
    configured = os.environ.get(_MYSTAND_PRIVATE_TAINT_FILE_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    xiaoban_home = os.environ.get("XIAOBAN_HOME", "").strip()
    base = (
        Path(xiaoban_home).expanduser()
        if xiaoban_home
        else Path.home() / ".xiaoban"
    )
    return base / "mystand-private-session-taints.json"


def _load_mystand_private_session_taints_locked(path: Path) -> None:
    global _MYSTAND_PRIVATE_SESSION_TAINT_LOADED_PATH
    global _MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED

    if _MYSTAND_PRIVATE_SESSION_TAINT_LOADED_PATH == path:
        return
    _MYSTAND_PRIVATE_SESSION_TAINT_LOADED_PATH = path
    _MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED = False
    _MYSTAND_PRIVATE_SESSION_TAINTS.clear()

    try:
        try:
            path.stat()
        except FileNotFoundError:
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("taint sidecar must be an object")
        if payload.get("schema") != _MYSTAND_PRIVATE_TAINT_SCHEMA:
            raise ValueError("unsupported taint sidecar schema")
        raw_taints = payload.get("taints")
        if not isinstance(raw_taints, list):
            raise ValueError("taint sidecar entries must be an array")
        taints = set()
        for value in raw_taints:
            text = str(value or "")
            namespace, separator, digest = text.partition(":")
            if (
                separator != ":"
                or namespace not in {"session_id", "session_key"}
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("invalid taint sidecar entry")
            taints.add(text)
        path.chmod(0o600)
        _MYSTAND_PRIVATE_SESSION_TAINTS.update(taints)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        _MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED = True


def _persist_mystand_private_session_taints_locked(path: Path) -> None:
    global _MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED

    parent = path.parent
    temp_path: Path | None = None
    file_descriptor: int | None = None
    try:
        parent_existed = parent.exists()
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed:
            parent.chmod(0o700)

        for _attempt in range(10):
            candidate = parent / (
                f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
            )
            try:
                file_descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                temp_path = candidate
                break
            except FileExistsError:
                continue
        if file_descriptor is None or temp_path is None:
            raise OSError("unable to allocate private-session taint temp file")

        payload = {
            "schema": _MYSTAND_PRIVATE_TAINT_SCHEMA,
            "taints": sorted(_MYSTAND_PRIVATE_SESSION_TAINTS),
        }
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            file_descriptor = None
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        path.chmod(0o600)

        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        _MYSTAND_PRIVATE_SESSION_TAINT_PERSISTENCE_FAILED = True
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def get_session_env(name: str, default: str = "") -> str:
    """Read a session context variable by its legacy ``XIAOBAN_SESSION_*`` name.

    Drop-in replacement for ``os.getenv("XIAOBAN_SESSION_*", default)``.

    Resolution order:
    1. Context variable (set by the gateway for concurrency-safe access).
       If the variable was explicitly set (even to ``""``) via
       ``set_session_vars`` or ``clear_session_vars``, that value is
       returned — **no fallback to os.environ**.
    2. ``os.environ`` (only when the context variable was never set in
       this context — i.e. CLI, cron scheduler, and test processes that
       don't use ``set_session_vars`` at all).
    3. *default*
    """
    import os

    var = _VAR_MAP.get(name)
    if var is not None:
        value = var.get()
        if value is not _UNSET:
            return value
    # Fall back to os.environ for CLI, cron, and test compatibility
    return os.getenv(name, default)


def async_delivery_supported() -> bool:
    """Whether the current session can deliver a background completion later.

    Returns ``False`` only when the active session was explicitly bound by a
    stateless adapter (the API server) that cannot route a notification back to
    the agent after the turn ends. CLI, cron, and the real gateway platforms —
    and any path that never bound the contextvar — return ``True``.

    Tools that promise async delivery (``terminal`` notify_on_complete /
    watch_patterns, ``delegate_task`` background=True) consult this before
    registering a watcher / dispatching a detached child, so they can refuse a
    promise the channel can't keep instead of silently no-op'ing.
    """
    value = _SESSION_ASYNC_DELIVERY.get()
    if value is _UNSET:
        return True
    return bool(value)
