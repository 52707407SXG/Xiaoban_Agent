"""Write-only My Stand authorization surface for API-server model sessions."""

from __future__ import annotations

import json

from gateway.session_context import mark_mystand_private_query_turn
from tools.mystand_authorization_tool import (
    _WRITE_ACTIONS,
    check_mystand_authorization,
    _current_session,
    _post_internal,
    mystand_authorization_tool_handler,
)
from tools.registry import registry

MYSTAND_AUTHORIZATION_WRITE_SCHEMA = {
    "name": "mystand_authorization_write",
    "description": (
        "Preview or commit a supported My Stand write through the server-enforced "
        "authorization wall. This tool cannot list, resolve, discover, or read "
        "resources. Always preview first, show the exact preview, stop, and commit "
        "only after a later standalone user confirmation. preview_write REQUIRES "
        "idempotency_key: generate one fresh unique key (e.g. a random UUID) for "
        "each new write attempt, reuse the same key only when retrying that same "
        "preview, and never reuse keys across different writes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["preview_write", "commit_write"],
            },
            "resource": {
                "type": "object",
                "description": (
                    "Semantic target of preview_write. Give the human-readable "
                    "resource name; the server privately resolves its writable "
                    "index node and authorization."
                ),
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 240,
                    },
                    "type_hint": {
                        "type": "string",
                        "enum": [
                            "note",
                            "property-note",
                            "profile-card",
                            "knowledge-graph",
                            "finance-archive",
                        ],
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            "action": {
                "type": "string",
                "enum": sorted(_WRITE_ACTIONS),
            },
            "payload": {
                "type": "object",
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "REQUIRED for preview_write. Generate one fresh unique key "
                    "(e.g. a random UUID) for each new write attempt; reuse the "
                    "same key only when retrying the same preview so the server "
                    "can deduplicate safely. Never reuse keys across different "
                    "writes. Not used for commit_write (pass preview_token instead)."
                ),
            },
            "preview_token": {
                "type": "string",
                "description": "REQUIRED for commit_write: the previewToken returned by the matching preview_write.",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}

_WRITE_KEYS = {
    "operation",
    "resource",
    "action",
    "payload",
    "idempotency_key",
    "preview_token",
}

_ACTION_RESOURCE_TYPES = {
    "note.append-content": "note",
    "property-note.append-text-block": "property-note",
    "profile-card.update-field": "profile-card",
    "knowledge-graph.add-node": "knowledge-graph",
    "knowledge-graph.update-node": "knowledge-graph",
    "knowledge-graph.add-edge": "knowledge-graph",
    "finance-archive.update-row-fields": "finance-archive",
}
_FAILED_WRITE_INTEGRITY_NOTICE = (
    "本次写入没有成功；禁止向用户声称已经写入或已落库，"
    "必须如实说明失败。"
)


def _error(message: str, *, code: str, status: int = 400) -> str:
    return json.dumps(
        {
            "ok": False,
            "status": status,
            "code": code,
            "error": message,
            "integrity_notice": _FAILED_WRITE_INTEGRITY_NOTICE,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _with_integrity_notice(result, *, operation: str) -> str:
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if not isinstance(parsed, dict):
        parsed = {
            "ok": False,
            "status": 502,
            "code": "invalid_authorization_write_result",
            "error": "My Stand 返回了无效的写入结果。",
        }
    if operation == "preview_write" and parsed.get("ok") is True:
        notice = (
            "本次只是预览，没有写入；不得向用户声称已落库，"
            "必须等待后续独立确认。"
        )
    elif (
        operation == "commit_write"
        and parsed.get("ok") is True
        and parsed.get("verified") is True
        and parsed.get("receiptVersion") == "authorization-write-receipt-v2"
    ):
        notice = (
            "本次已取得 authorization-write-receipt-v2 且 verified=true "
            "回执，可以准确说明写入成功。"
        )
    else:
        notice = _FAILED_WRITE_INTEGRITY_NOTICE
    parsed["integrity_notice"] = notice
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def mystand_authorization_write_tool_handler(args, **kwargs):
    session = _current_session()
    if session["platform"] != "api_server" or not session["user_id"]:
        return _error(
            "该写入工具只允许 My Stand 已登录网页/API 会话使用。",
            code="mystand_session_required",
            status=403,
        )
    # After authenticating the My Stand lane, taint before argument validation,
    # preview, or commit delegation. A malformed or rejected write attempt still
    # belongs to a private turn, and same-batch web calls are pre-tainted too.
    mark_mystand_private_query_turn()
    if not isinstance(args, dict) or set(args) - _WRITE_KEYS:
        return _error(
            "写入参数包含不允许的字段。",
            code="invalid_authorization_write_arguments",
        )
    operation = str(args.get("operation") or "").strip()
    if operation not in {"preview_write", "commit_write"}:
        return _error(
            "该工具只允许写入预览与确认提交。",
            code="authorization_read_not_allowed",
            status=403,
        )
    if operation == "commit_write":
        return _with_integrity_notice(
            mystand_authorization_tool_handler(args, **kwargs),
            operation=operation,
        )

    if not session["message_id"] or not session["session_id"]:
        return _error(
            "当前请求缺少可信 messageId 或 sessionId，不能发起写入预览。",
            code="trusted_write_context_required",
            status=409,
        )
    action = str(args.get("action") or "").strip()
    if action not in _WRITE_ACTIONS:
        return _error(
            "写入动作不在安全白名单中。",
            code="authorization_write_action_not_allowed",
        )
    payload = args.get("payload")
    if not isinstance(payload, dict):
        return _error("payload 必须是动作对应的对象。", code="invalid_write_payload")
    resource = args.get("resource")
    if not isinstance(resource, dict) or set(resource) - {"name", "type_hint"}:
        return _error("写入目标资料无效。", code="invalid_write_resource")
    name = resource.get("name")
    if not isinstance(name, str) or not (2 <= len(name.strip()) <= 240):
        return _error("写入目标资料名称无效。", code="invalid_write_resource")
    expected_type = _ACTION_RESOURCE_TYPES[action]
    type_hint = str(resource.get("type_hint") or expected_type).strip()
    if type_hint != expected_type:
        return _error("写入动作与目标资料类型不匹配。", code="invalid_write_resource")
    idempotency_key = str(args.get("idempotency_key") or "").strip()
    if not idempotency_key:
        return _error("缺少 idempotency_key", code="authorization_argument_missing")
    return _with_integrity_notice(
        _post_internal(
            "/api/xiaoban/internal/authorization/write/preview",
            {
                "resource": {
                    "name": name.strip(),
                    "typeHint": type_hint,
                },
                "action": action,
                "payload": payload,
                "idempotencyKey": idempotency_key,
            },
            session=session,
        ),
        operation=operation,
    )


registry.register(
    name="mystand_authorization_write",
    toolset="mystand_authorization_write",
    schema=MYSTAND_AUTHORIZATION_WRITE_SCHEMA,
    handler=mystand_authorization_write_tool_handler,
    check_fn=check_mystand_authorization,
    requires_env=[],
    is_async=False,
    description="Write-only My Stand authorization bridge",
    emoji="✍️",
    max_result_size_chars=1_000_000,
)
