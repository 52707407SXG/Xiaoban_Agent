"""AUTH-gated My Stand read bridge for Xiaoban API sessions.

This tool never reads My Stand storage directly.  It calls the loopback-only
My Stand internal API, which re-checks the current user, AUTH/OUT permissions,
and domain ownership.  Confirmed writes use the separate model-visible
``mystand_authorization_write`` tool and a private delegate in this module.
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
from tools.mystand_authorization_write_payload import (
    AuthorizationWritePayloadError,
    normalize_authorization_write_payload,
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
    "resolve_many": "/api/xiaoban/internal/authorization/resolve",
    "preview_write": "/api/xiaoban/internal/authorization/write/preview",
    "commit_write": "/api/xiaoban/internal/authorization/write/commit",
}
_READ_OPERATIONS = frozenset({"list", "resolve", "resolve_many"})
_WRITE_OPERATIONS = frozenset({"preview_write", "commit_write"})
_WRITE_ACTIONS = {
    "note.append-content",
    "property-note.append-text-block",
    "profile-card.update-field",
    "knowledge-graph.add-node",
    "knowledge-graph.update-node",
    "knowledge-graph.add-edge",
    "finance-archive.update-row-fields",
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
        "Read through My Stand's server-enforced authorization wall. This tool "
        "only lists or resolves AUTH/OUT records; for every write preview or "
        "commit use mystand_authorization_write instead. Never read a database "
        "or local file instead.\n\n"
        "list returns only the current user's authorization index. If the user "
        "supplies an exact AUTH or OUT, resolve it directly with authorization_id; OUT "
        "remains read-only. If the user supplies an exact resourceUid, resolve it "
        "directly with resource_uid. Do not call the resource index first for either "
        "known locator. Use mystand_resource_index only when the current request gives "
        "a material name or business goal without an exact AUTH, OUT, or resourceUid, "
        "then resolve the returned resource_uid. If the question spans multiple exact "
        "resource_uids, use one resolve_many call; every item is independently re-checked "
        "by My Stand before combined content is returned. Never reinterpret resource_uid, "
        "source_id, KGREF, or module IDs as authorization_id. A feature "
        "explanation does not need this tool; real user data does. Resource discovery, "
        "index maps, AUTH IDs, internal queries, retries, truncation, and tool output "
        "are private backend details: never narrate them to the user; answer naturally "
        "from the final authorized result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "resolve", "resolve_many"],
                "description": "Read-only authorization operation to perform.",
            },
            "authorization_id": {
                "type": "string",
                "description": "Exact AUTH-... or OUT-... for direct read-only resolve.",
            },
            "resource_uid": {
                "type": "string",
                "description": "Exact opaque resourceUid supplied in the current request or returned by mystand_resource_index. Resolve it directly; never reinterpret it as authorization_id.",
            },
            "resource_uids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 100,
                "items": {"type": "string", "minLength": 1, "maxLength": 120},
                "description": "Exact opaque resourceUids supplied in the current request or returned by mystand_resource_index. Use resolve_many for a question spanning several resources.",
            },
            "query": {
                "type": "string",
                "description": (
                    "List: optional authorization-title search. Resolve: the user's exact "
                    "question or locator text so the authorized source can return only the "
                    "matching row, section, or node. If omitted on resolve, the current "
                    "trusted user message is used automatically."
                ),
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


def _resource_uid_list(value) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("resource_uids 必须包含 1 到 100 个站内资源 ID")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        uid = str(item or "").strip()
        if not uid or len(uid) > 120:
            raise ValueError("resource_uids 包含无效站内资源 ID")
        if uid not in seen:
            result.append(uid)
            seen.add(uid)
    return result


def _resolve_many(resource_uids: list[str], *, session: dict, query: str) -> str:
    resources = []
    for resource_uid in resource_uids:
        raw = _post_internal(
            _OPERATIONS["resolve"],
            {
                "resourceUid": resource_uid,
                "mediaMode": "summary",
                "query": query[:1200],
            },
            session=session,
        )
        try:
            resolved = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            resolved = {}
        if not isinstance(resolved, dict) or resolved.get("ok") is not True:
            try:
                status = int(resolved.get("status") or 409)
            except (TypeError, ValueError):
                status = 409
            return _error(
                "至少一份资料未通过当前账号的读取授权，本次没有返回部分结果。",
                code="mystand_authorization_batch_rejected",
                status=status,
            )
        resources.append(
            {
                "resourceUid": resource_uid,
                "content": str(resolved.get("content") or ""),
                "encrypted": resolved.get("encrypted") is True,
            }
        )
        if len(json.dumps(resources, ensure_ascii=False)) > 900_000:
            return _error(
                "本次授权资料合并结果过大，请缩小资料范围。",
                code="mystand_authorization_result_too_large",
                status=413,
            )
    return _json_result(
        {
            "ok": True,
            "content": json.dumps(
                {"resources": resources},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "recordRefs": sorted(resource_uids),
            "encrypted": any(item["encrypted"] for item in resources),
        }
    )


def _mystand_authorization_operation_handler(args, **_kwargs):
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
        elif operation in {"resolve", "resolve_many"}:
            trusted_user_message = get_session_user_message().strip()
            query = trusted_user_message or str(args.get("query") or "").strip()
            resource_uid = str(args.get("resource_uid") or "").strip()
            authorization_id = str(args.get("authorization_id") or "").strip()
            if operation == "resolve_many":
                if resource_uid or authorization_id:
                    return _error(
                        "resolve_many 只能提供 resource_uids",
                        code="invalid_authorization_arguments",
                    )
                return _resolve_many(
                    _resource_uid_list(args.get("resource_uids")),
                    session=session,
                    query=query,
                )
            if args.get("resource_uids") is not None:
                return _error(
                    "resolve 不能同时提供 resource_uids",
                    code="invalid_authorization_arguments",
                )
            if not resource_uid and not authorization_id:
                raise ValueError("缺少 resource_uid 或 authorization_id")
            if resource_uid and authorization_id:
                return _error(
                    "resource_uid 与 authorization_id 只能提供一个",
                    code="invalid_authorization_arguments",
                )
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
                    "query": query[:1200],
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
            payload = normalize_authorization_write_payload(action, payload)
            body.update({
                "authorizationId": _require_text(args, "authorization_id"),
                "action": action,
                "payload": payload,
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
    except AuthorizationWritePayloadError as exc:
        return _error(str(exc), code=exc.code, status=exc.status)
    except ValueError as exc:
        return _error(str(exc), code="authorization_argument_missing")

    return _post_internal(
        _OPERATIONS[operation],
        body,
        session=session,
        explicit_confirmation=operation == "commit_write",
    )


def _mystand_authorization_write_operation_handler(args, **kwargs):
    """Execute a write operation only for the dedicated write-only tool."""
    operation = str(args.get("operation") or "").strip()
    if operation not in _WRITE_OPERATIONS:
        return _error(
            "该私有委托只允许写入预览与确认提交。",
            code="invalid_authorization_write_operation",
        )
    return _mystand_authorization_operation_handler(args, **kwargs)


def mystand_authorization_tool_handler(args, **kwargs):
    """Expose only AUTH/OUT reads, even when hidden write args are supplied."""
    operation = str(args.get("operation") or "").strip()
    if operation in _WRITE_OPERATIONS:
        return _error(
            "写入必须使用 mystand_authorization_write。",
            code="authorization_write_tool_required",
            status=403,
        )
    if operation not in _READ_OPERATIONS:
        return _error(
            "operation 不在允许范围内",
            code="invalid_authorization_operation",
        )
    return _mystand_authorization_operation_handler(args, **kwargs)


registry.register(
    name="mystand_authorization",
    toolset="mystand_authorization",
    schema=MYSTAND_AUTHORIZATION_SCHEMA,
    handler=mystand_authorization_tool_handler,
    check_fn=check_mystand_authorization,
    requires_env=[],
    is_async=False,
    description="Server-enforced My Stand AUTH/OUT read-only bridge",
    emoji="🔐",
    max_result_size_chars=1_000_000,
)
