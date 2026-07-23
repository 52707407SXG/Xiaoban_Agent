"""AUTH-gated My Stand read/write bridge for Xiaoban API sessions.

This tool never reads My Stand storage directly.  It calls the loopback-only
My Stand internal API, which re-checks the current user, AUTH/OUT permissions,
domain ownership, preview confirmation, CAS, idempotency, verification, and
audit rules.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from gateway.session_context import (
    get_session_env,
    get_session_user_message,
)
from tools.registry import registry

_DEFAULT_API_URL = "http://127.0.0.1:18081"
_DEFAULT_ENV_FILE = "/opt/xiaoban-agent/.env"
_INTERNAL_TOKEN_KEYS = (
    "MYSTAND_XIAOBAN_MYSTAND_API_TOKEN",
    "MYSTAND_XIAOBAN_GATEWAY_INTERNAL_TOKEN",
)
_OPERATIONS = {
    "list": "/api/xiaoban/internal/authorization/list",
    "resolve": "/api/xiaoban/internal/authorization/resolve",
    "preview_write": "/api/xiaoban/internal/authorization/write/preview",
    "commit_write": "/api/xiaoban/internal/authorization/write/commit",
}
_WRITE_ACTIONS = {
    "note.append-content",
    "property-note.append-text-block",
    "profile-card.update-field",
    "knowledge-graph.add-node",
    "knowledge-graph.update-node",
    "knowledge-graph.add-edge",
}
_EXPLICIT_CONFIRMATION = "确认写入"
_EXPLICIT_CONFIRMATION_REPLY_RE = re.compile(
    r"(?:"
    r"(?:我\s*)?确认写入"
    r"|"
    r"预览(?:内容)?没问题[，,\s]*(?:我\s*)?确认写入"
    r")[。！!]?"
)


MYSTAND_AUTHORIZATION_SCHEMA = {
    "name": "mystand_authorization",
    "description": (
        "Use My Stand's server-enforced authorization wall. This is the only "
        "tool for Xiaoban to list or resolve AUTH/OUT records and to preview or "
        "commit supported My Stand writes. Never read a database or local file "
        "instead.\n\n"
        "READS: list returns only the current user's authorization index. "
        "After mystand_resource_index finds a resource, resolve it with the exact "
        "resource_uid; the server exchanges that opaque node for the current "
        "user's Xiaoban-bound default AUTH. Never pass resource_uid, source_id, "
        "KGREF, OUT, or module IDs as authorization_id. Direct AUTH/OUT resolve "
        "remains supported and is re-checked before content is returned. A feature "
        "explanation does not need this tool; real user data does.\n\n"
        "WRITES: OUT can never write. Only the fixed allowlisted actions are supported. "
        "First call preview_write with an internal AUTH whose canWrite is true, "
        "the target's current expected_version, and a fresh idempotency_key. "
        "Show the returned exact preview to the user and stop. Call commit_write "
        "only in a later user message whose complete reply is an unambiguous "
        "standalone confirmation such as '确认写入' or '预览没问题，确认写入'; "
        "reuse the preview_token and idempotency_key. Never infer confirmation "
        "from quoted text, questions, button labels, or analysis, never invent "
        "confirmation, never commit in the preview turn, and never describe a "
        "successful write unless the receipt says verified=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "resolve", "preview_write", "commit_write"],
                "description": "Authorization operation to perform.",
            },
            "authorization_id": {
                "type": "string",
                "description": "AUTH-... for internal read/write, or OUT-... for read-only resolve. OUT is rejected for every write.",
            },
            "resource_uid": {
                "type": "string",
                "description": "Exact opaque resourceUid returned by mystand_resource_index. Preferred for resolve after an index lookup; never reinterpret it as authorization_id.",
            },
            "query": {
                "type": "string",
                "description": "Optional list search text.",
            },
            "source_type": {
                "type": "string",
                "description": "Optional list filter such as note, profile-card, or knowledge-graph.",
            },
            "permission": {
                "type": "string",
                "enum": ["", "read", "write", "external-read"],
                "description": "Optional list permission filter.",
            },
            "id_type": {
                "type": "string",
                "enum": ["", "internal", "outbound"],
                "description": "Optional list ID type filter.",
            },
            "media_mode": {
                "type": "string",
                "enum": ["summary", "include"],
                "description": "Resolve media only when the current user explicitly asked to inspect it; otherwise use summary.",
            },
            "action": {
                "type": "string",
                "enum": sorted(_WRITE_ACTIONS),
                "description": "Fixed write action for preview_write.",
            },
            "payload": {
                "type": "object",
                "description": "Action-specific payload. Extra fields and generic patches are rejected by My Stand.",
            },
            "expected_version": {
                "type": "string",
                "description": "Current target version returned by an authorized read or domain view.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Fresh stable key for one logical write; reuse it for preview and commit retries.",
            },
            "preview_token": {
                "type": "string",
                "description": "Short-lived token returned by preview_write; required for commit_write.",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


def _read_env_file_value(path: str, key: str) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        return value.strip().strip("'\"")
    return ""


def _internal_token() -> str:
    for key in _INTERNAL_TOKEN_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            return value
    env_file = os.getenv("MYSTAND_XIAOBAN_GATEWAY_ENV_FILE", _DEFAULT_ENV_FILE)
    for key in _INTERNAL_TOKEN_KEYS:
        value = _read_env_file_value(env_file, key)
        if value:
            return value
    return ""


def _api_base_url() -> str:
    value = os.getenv("MYSTAND_XIAOBAN_MYSTAND_API_URL", _DEFAULT_API_URL).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return ""
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return ""
    return value


def check_mystand_authorization() -> bool:
    return bool(_internal_token() and _api_base_url())


def _json_result(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _error(message: str, *, code: str = "mystand_authorization_failed", status: int = 400) -> str:
    return _json_result({"ok": False, "status": status, "code": code, "error": message})


def _current_session() -> dict:
    return {
        "platform": get_session_env("XIAOBAN_SESSION_PLATFORM", "").strip().lower(),
        "user_id": get_session_env("XIAOBAN_SESSION_USER_ID", "").strip(),
        "message_id": get_session_env("XIAOBAN_SESSION_MESSAGE_ID", "").strip(),
        "session_id": get_session_env("XIAOBAN_SESSION_ID", "").strip(),
    }


def _safe_internal_header(value: str, limit: int = 200) -> str:
    text = str(value or "")
    if len(text) > limit or not re.fullmatch(r"[A-Za-z0-9._:@-]+", text):
        return ""
    return text


def _post_internal(
    path: str,
    payload: dict,
    *,
    session: dict | None = None,
    explicit_confirmation: bool = False,
) -> str:
    base_url = _api_base_url()
    token = _internal_token()
    if not base_url or not token:
        return _error(
            "My Stand 授权桥尚未配置，不能读取或写入业务资料。",
            code="mystand_authorization_unavailable",
            status=503,
        )
    trusted_session = session if isinstance(session, dict) else {}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }
    header_values = {
        "X-Xiaoban-User-Id": trusted_session.get("user_id"),
        "X-Xiaoban-Message-Id": trusted_session.get("message_id"),
        "X-Xiaoban-Session-Id": trusted_session.get("session_id"),
    }
    for name, value in header_values.items():
        safe_value = _safe_internal_header(value)
        if safe_value:
            headers[name] = safe_value
    if explicit_confirmation:
        headers["X-Xiaoban-Explicit-Confirmation"] = "1"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                return _error("My Stand 授权结果过大，已停止读取。", code="mystand_authorization_result_too_large", status=413)
            parsed = json.loads(raw.decode("utf-8")) if raw else {"ok": True}
            return _json_result(parsed)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(200_000)
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            parsed = {}
        return _json_result({
            "ok": False,
            "status": int(exc.code),
            "code": str(parsed.get("code") or "mystand_authorization_rejected"),
            "error": str(parsed.get("error") or parsed.get("message") or "My Stand 拒绝了这次授权操作")[:500],
            **({"details": parsed["details"]} if isinstance(parsed.get("details"), dict) else {}),
        })
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return _error(
            "My Stand 授权服务暂时没有接稳，请稍后重试。",
            code="mystand_authorization_transport_failed",
            status=502,
        )


def _require_text(args: dict, key: str) -> str:
    value = str(args.get(key) or "").strip()
    if not value:
        raise ValueError(f"缺少 {key}")
    return value


def mystand_authorization_tool_handler(args, **_kwargs):
    operation = str(args.get("operation") or "").strip()
    if operation not in _OPERATIONS:
        return _error("operation 不在允许范围内", code="invalid_authorization_operation")

    session = _current_session()
    if session["platform"] != "api_server" or not session["user_id"]:
        return _error(
            "该授权工具只允许 My Stand 已登录网页/API 会话使用。",
            code="mystand_session_required",
            status=403,
        )

    # Identity and confirmation context travel only in service-authenticated
    # headers assembled by this handler.  Never duplicate them in model-shaped
    # request JSON, where a future server change might accidentally trust them.
    body = {}
    try:
        if operation == "list":
            body.update({
                "q": str(args.get("query") or "").strip()[:240],
                "sourceType": str(args.get("source_type") or "").strip()[:80],
                "permission": str(args.get("permission") or "").strip()[:40],
                "idType": str(args.get("id_type") or "").strip()[:40],
            })
        elif operation == "resolve":
            resource_uid = str(args.get("resource_uid") or "").strip()
            authorization_id = str(args.get("authorization_id") or "").strip()
            if not resource_uid and not authorization_id:
                raise ValueError("缺少 resource_uid 或 authorization_id")
            body.update(
                {
                    **(
                        {"resourceUid": resource_uid}
                        if resource_uid
                        else {"authorizationId": authorization_id}
                    ),
                    "mediaMode": (
                        "include"
                        if args.get("media_mode") == "include"
                        else "summary"
                    ),
                }
            )
        elif operation == "preview_write":
            if not session["message_id"] or not session["session_id"]:
                return _error("当前请求缺少可信 messageId 或 sessionId，不能发起写入预览。", code="trusted_write_context_required", status=409)
            action = _require_text(args, "action")
            if action not in _WRITE_ACTIONS:
                return _error("写入动作不在安全白名单中。", code="authorization_write_action_not_allowed")
            payload = args.get("payload")
            if not isinstance(payload, dict):
                return _error("payload 必须是动作对应的对象。", code="invalid_write_payload")
            body.update({
                "authorizationId": _require_text(args, "authorization_id"),
                "action": action,
                "payload": payload,
                "expectedVersion": _require_text(args, "expected_version"),
                "idempotencyKey": _require_text(args, "idempotency_key"),
            })
        else:
            if not session["message_id"] or not session["session_id"]:
                return _error("当前请求缺少可信 messageId 或 sessionId，不能提交写入。", code="trusted_write_context_required", status=409)
            current_user_message = get_session_user_message().strip()
            if not _EXPLICIT_CONFIRMATION_REPLY_RE.fullmatch(current_user_message):
                return _error(
                    "只有用户在预览后的新消息里，用独立回复明确说“确认写入”，才能提交。",
                    code="explicit_user_confirmation_required",
                    status=409,
                )
            body.update({
                "previewToken": _require_text(args, "preview_token"),
                "idempotencyKey": _require_text(args, "idempotency_key"),
                "confirmationPhrase": _EXPLICIT_CONFIRMATION,
            })
    except ValueError as exc:
        return _error(str(exc), code="authorization_argument_missing")

    return _post_internal(
        _OPERATIONS[operation],
        body,
        session=session,
        explicit_confirmation=operation == "commit_write",
    )


registry.register(
    name="mystand_authorization",
    toolset="mystand_authorization",
    schema=MYSTAND_AUTHORIZATION_SCHEMA,
    handler=mystand_authorization_tool_handler,
    check_fn=check_mystand_authorization,
    requires_env=[],
    is_async=False,
    description="Server-enforced My Stand AUTH/OUT read and confirmed write bridge",
    emoji="🔐",
    max_result_size_chars=1_000_000,
)
