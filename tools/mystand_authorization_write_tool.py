"""Write-only My Stand authorization surface for API-server model sessions."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from gateway.session_context import mark_mystand_private_query_turn
from tools.approval import request_gateway_action_approval
from tools.mystand_authorization_tool import (
    _WRITE_ACTIONS,
    check_mystand_authorization,
    _current_session,
    _post_internal,
)
from tools.mystand_authorization_write_payload import (
    AuthorizationWritePayloadError,
    build_authorization_write_payload_schema,
    normalize_authorization_write_payload,
)
from tools.registry import registry

MYSTAND_AUTHORIZATION_WRITE_SCHEMA = {
    "name": "mystand_authorization_write",
    "description": (
        "Preview and apply one supported My Stand mutation through the server-enforced "
        "current-account authorization wall. Every call first shows the exact before/after "
        "change in the website and blocks until the user double-clicks approve or deny; no "
        "write happens before that decision. This tool cannot list, resolve, discover, clear, "
        "purge, or read resources. Use the resource index/read tool first, then call this tool "
        "once with the human-readable target, action and exact payload. The server privately "
        "resolves AUTH, rechecks account ownership and version, creates a seven-day recovery "
        "point, commits atomically and verifies the saved result. Never claim success unless "
        "the returned receipt is verified=true. Business deletion uses the recoverable "
        "business-archive.archive action; permanent server or cross-account deletion is not "
        "available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["preview_and_apply"],
                "description": (
                    "Always use preview_and_apply with resource, action, payload and a fresh "
                    "idempotency_key. The tool owns preview, user approval and commit."
                ),
            },
            "resource": {
                "type": "object",
                "description": (
                    "Semantic target of the requested change. Give the human-readable "
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
                            "business-archive",
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
            "payload": build_authorization_write_payload_schema(),
            "idempotency_key": {
                "type": "string",
                "description": (
                    "Required. Generate one "
                    "fresh unique key (e.g. a random UUID) for each logical write and "
                    "reuse it only when retrying that same write."
                ),
            },
        },
        "required": ["operation", "resource", "action", "payload", "idempotency_key"],
        "additionalProperties": False,
    },
}

_WRITE_KEYS = {
    "operation",
    "resource",
    "action",
    "payload",
    "idempotency_key",
}

_ACTION_RESOURCE_TYPES = {
    "note.append-content": "note",
    "property-note.append-text-block": "property-note",
    "property-note.edit-blocks": "property-note",
    "profile-card.update-field": "profile-card",
    "knowledge-graph.add-node": "knowledge-graph",
    "knowledge-graph.update-node": "knowledge-graph",
    "knowledge-graph.add-edge": "knowledge-graph",
    "knowledge-graph.delete": "knowledge-graph",
    "business-archive.archive": "business-archive",
    "finance-archive.update-row-fields": "finance-archive",
}
_FAILED_WRITE_INTEGRITY_NOTICE = (
    "本次写入没有成功；禁止向用户声称已经写入或已落库，"
    "必须如实说明失败。"
)
_VERIFIED_WRITE_INTEGRITY_NOTICE = (
    "本次已取得完整且与本次确认绑定的 "
    "authorization-write-receipt-v2 且 verified=true 回执，"
    "可以准确说明写入成功。"
)
_TRUSTED_CONTEXT_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:@-]{7,199}$"
)
_WRITE_ACTION_RE = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)+$")
_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")


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


def _preview_token_hash(preview_token) -> str:
    """Mirror Node's SHA256(JSON.stringify(String(previewToken)))."""
    serialized = json.dumps(
        str(preview_token or ""),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parseable_committed_at(value) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _verified_commit_receipt(parsed, *, commit_args) -> bool:
    """Validate the Node v2 receipt shape and bind it to this commit call."""
    if not isinstance(parsed, dict):
        return False
    args = commit_args if isinstance(commit_args, dict) else {}
    audit = parsed.get("audit")
    recovery = parsed.get("recovery")
    target = parsed.get("target")
    expected_version = str(parsed.get("expectedVersion") or "").strip()
    next_version = str(parsed.get("nextVersion") or "").strip()
    idempotency_key = str(parsed.get("idempotencyKey") or "")
    expected_idempotency_key = str(args.get("idempotency_key") or "").strip()
    expected_confirmation_id = str(args.get("gateway_approval_id") or "").strip()
    expected_preview_token = str(args.get("preview_token") or "").strip()
    expected_preview_hash = _preview_token_hash(expected_preview_token)
    return bool(
        parsed.get("ok") is True
        and parsed.get("status") == 200
        and parsed.get("receiptVersion")
        == "authorization-write-receipt-v2"
        and parsed.get("verified") is True
        and isinstance(audit, dict)
        and audit.get("recorded") is True
        and str(audit.get("auditId") or "").strip()
        and isinstance(recovery, dict)
        and str(recovery.get("recoveryId") or "").startswith("auth-recovery-")
        and recovery.get("retentionDays") == 7
        and _parseable_committed_at(recovery.get("expiresAt"))
        and expected_confirmation_id
        and parsed.get("confirmationId") == expected_confirmation_id
        and parsed.get("confirmationMode") == "separate-user-confirmation"
        and _TRUSTED_CONTEXT_ID_RE.fullmatch(
            str(parsed.get("confirmationId") or "")
        )
        and _WRITE_ACTION_RE.fullmatch(str(parsed.get("action") or ""))
        and parsed.get("action") in _WRITE_ACTIONS
        and isinstance(target, dict)
        and expected_version
        and next_version
        and expected_version != next_version
        and _TRUSTED_CONTEXT_ID_RE.fullmatch(idempotency_key)
        and (
            not expected_idempotency_key
            or idempotency_key == expected_idempotency_key
        )
        and _DIGEST_RE.fullmatch(
            str(parsed.get("requestFingerprint") or "")
        )
        and _DIGEST_RE.fullmatch(str(parsed.get("previewTokenHash") or ""))
        and (
            not expected_preview_token
            or parsed.get("previewTokenHash") == expected_preview_hash
        )
        and _DIGEST_RE.fullmatch(str(parsed.get("changeDigest") or ""))
        and _parseable_committed_at(parsed.get("committedAt"))
    )


def _with_integrity_notice(
    result,
    *,
    operation: str,
    commit_args=None,
) -> str:
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
    valid_commit_receipt = bool(
        operation == "preview_and_apply"
        and _verified_commit_receipt(
            parsed,
            commit_args=commit_args,
        )
    )
    if (
        operation == "preview_and_apply"
        and parsed.get("ok") is not False
        and not valid_commit_receipt
    ):
        parsed = {
            "ok": False,
            "status": 502,
            "code": "authorization_write_receipt_invalid",
            "error": (
                "My Stand 没有返回完整且与本次确认绑定的 verified=true "
                "authorization-write-receipt-v2，不能报告写入成功。"
            ),
        }
    if valid_commit_receipt:
        notice = _VERIFIED_WRITE_INTEGRITY_NOTICE
    else:
        notice = _FAILED_WRITE_INTEGRITY_NOTICE
    parsed["integrity_notice"] = notice
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _verified_tool_result_receipt(result, *, function_args):
    """Recover the receipt already verified inside preview/approve/commit.

    The tool handler owns the private preview token and approval id, so the
    executor cannot reconstruct those values from the model-visible call.
    Only the handler adds the exact verified integrity notice after binding
    both values. Recheck the public call identity here before promoting the
    receipt into the runtime-owned canonical sidecar.
    """
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    args = function_args if isinstance(function_args, dict) else {}
    if (
        not isinstance(parsed, dict)
        or args.get("operation") != "preview_and_apply"
        or parsed.get("integrity_notice")
        != _VERIFIED_WRITE_INTEGRITY_NOTICE
        or parsed.get("action") != args.get("action")
        or parsed.get("idempotencyKey") != args.get("idempotency_key")
    ):
        return None
    if not _verified_commit_receipt(
        parsed,
        commit_args={
            "gateway_approval_id": parsed.get("confirmationId"),
            "idempotency_key": args.get("idempotency_key"),
        },
    ):
        return None
    receipt = dict(parsed)
    receipt.pop("integrity_notice", None)
    return receipt


def _display_text(value, limit: int = 8_000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)[:limit]
    except (TypeError, ValueError):
        return str(value)[:limit]


def _write_preview_for_approval(parsed: dict, resource_name: str, action: str) -> dict:
    supplied = parsed.get("displayPreview")
    if isinstance(supplied, dict):
        preview = dict(supplied)
    else:
        raw = parsed.get("preview") if isinstance(parsed.get("preview"), dict) else {}
        before = raw.get("before", raw.get("beforeTail", ""))
        after = raw.get("after", raw.get("afterTail", raw.get("append", raw.get("change", ""))))
        preview = {
            "title": raw.get("title") or "核对资料改动",
            "action": action,
            "target": resource_name,
            "before": _display_text(before) or "（原内容不变或当前没有内容）",
            "after": _display_text(after) or "（按本次动作删除或归档）",
            "changeType": (
                "delete"
                if action.endswith(".delete") or action.endswith(".archive")
                else "add"
                if ".add-" in action or ".append-" in action or ".create-" in action
                else "update"
            ),
            "summary": raw.get("title") or "将按上方内容修改站内资料",
            "recoveryDays": 7,
        }
    normalized = {
        "title": _display_text(preview.get("title"), 160) or "核对资料改动",
        "action": _display_text(preview.get("action") or action, 120),
        "target": _display_text(preview.get("target") or resource_name, 240),
        "before": _display_text(preview.get("before"), 8_000) or "（当前没有内容）",
        "after": _display_text(preview.get("after"), 8_000) or "（删除或归档）",
        "changeType": str(preview.get("changeType") or "update")[:20],
        "summary": _display_text(preview.get("summary"), 300),
        "recoveryDays": 7,
    }
    if len(json.dumps(normalized, ensure_ascii=False)) > 20_000:
        normalized["before"] = normalized["before"][:6_000]
        normalized["after"] = normalized["after"][:6_000]
    return normalized


def mystand_authorization_write_tool_handler(args, **kwargs):
    session = _current_session()
    if session["platform"] != "api_server" or not session["user_id"]:
        return _error(
            "该写入工具只允许 My Stand 已登录网页/API 会话使用。",
            code="mystand_session_required",
            status=403,
        )
    mark_mystand_private_query_turn()
    if not isinstance(args, dict) or set(args) - _WRITE_KEYS:
        return _error(
            "写入参数包含不允许的字段。",
            code="invalid_authorization_write_arguments",
        )
    operation = str(args.get("operation") or "").strip()
    if operation != "preview_and_apply":
        return _error(
            "该工具只接受先预览、再由用户确认的写入。",
            code="authorization_read_not_allowed",
            status=403,
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
    try:
        payload = normalize_authorization_write_payload(action, payload)
    except AuthorizationWritePayloadError as exc:
        return _error(str(exc), code=exc.code, status=exc.status)
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
    preview_raw = _post_internal(
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
    )
    try:
        preview_result = json.loads(preview_raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        preview_result = {}
    if not isinstance(preview_result, dict) or preview_result.get("ok") is not True:
        return _with_integrity_notice(
            preview_result,
            operation=operation,
            commit_args=args,
        )
    preview_token = str(preview_result.get("previewToken") or "").strip()
    if not preview_token:
        return _error(
            "My Stand 没有返回可确认的写入预览。",
            code="authorization_write_preview_invalid",
            status=502,
        )
    public_preview = _write_preview_for_approval(preview_result, name.strip(), action)
    approval = request_gateway_action_approval(
        pattern_key=f"mystand-write:{hashlib.sha256(preview_token.encode('utf-8')).hexdigest()}",
        description=f"修改“{name.strip()}”前需要核对并确认本次改动。",
        command=json.dumps(public_preview, ensure_ascii=False, separators=(",", ":")),
        surface="mystand-authorization-write",
        choices=("once", "deny"),
        metadata={
            "approval_kind": "authorization-write",
            "write_preview": public_preview,
        },
        timeout_seconds=7 * 24 * 60 * 60,
    )
    approval_id = str(approval.get("approval_id") or "").strip()
    if approval.get("approved") is not True or not approval_id:
        _post_internal(
            "/api/xiaoban/internal/authorization/write/cancel",
            {
                "previewToken": preview_token,
                "idempotencyKey": idempotency_key,
            },
            session=session,
        )
        return _error(
            str(approval.get("message") or "用户没有同意这次改动，资料没有修改。"),
            code="authorization_write_user_denied",
            status=409,
        )
    commit_args = {
        "preview_token": preview_token,
        "idempotency_key": idempotency_key,
        "gateway_approval_id": approval_id,
    }
    commit_raw = _post_internal(
        "/api/xiaoban/internal/authorization/write/commit",
        {
            "previewToken": preview_token,
            "idempotencyKey": idempotency_key,
        },
        session=session,
        gateway_approval_id=approval_id,
    )
    return _with_integrity_notice(
        commit_raw,
        operation=operation,
        commit_args=commit_args,
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
