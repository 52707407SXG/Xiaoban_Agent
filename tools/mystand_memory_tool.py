"""Account-bound self-curation for My Stand website memory."""

from __future__ import annotations

import json
import os
import re

from gateway.session_context import (
    get_session_env,
    get_session_memory_resource_refs,
)
from plugins.memory.holographic.scope import open_scoped_memory_store
from tools.registry import registry
from xiaoban.mystand_account_memory import (
    manage_account_memory,
    normalize_memory_tier,
    resolve_memory_scope_secret,
    validate_authorized_memory_resource_refs,
)


_IDENTITY_RE = re.compile(r"[A-Za-z0-9._:@-]{1,200}\Z")
_SITE_RE = re.compile(r"[A-Za-z0-9._:@-]{1,120}\Z")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _error(message: str, code: str) -> str:
    return _json({"ok": False, "code": code, "error": message})


def mystand_memory_handler(args, **_kwargs) -> str:
    if not isinstance(args, dict):
        return _error("记忆参数必须是对象。", "invalid_memory_arguments")
    if (
        get_session_env("XIAOBAN_SESSION_PLATFORM", "").strip().lower() != "api_server"
        or get_session_env("XIAOBAN_SESSION_SOURCE", "").strip().lower() != "mystand"
    ):
        return _error("该记忆只允许当前 My Stand 网页会话使用。", "mystand_session_required")

    user_id = get_session_env("XIAOBAN_SESSION_USER_ID", "").strip()
    site_id = get_session_env("XIAOBAN_SESSION_MEMORY_SITE_ID", "").strip()
    raw_tier = get_session_env("XIAOBAN_SESSION_MEMORY_TIER", "").strip()
    if not _IDENTITY_RE.fullmatch(user_id) or not _SITE_RE.fullmatch(site_id):
        return _error("当前账号记忆作用域无效。", "invalid_memory_scope")
    try:
        tier = normalize_memory_tier(raw_tier)
    except ValueError:
        return _error("当前账号记忆等级无效。", "invalid_memory_scope")
    owner_user_id = str(os.getenv("MYSTAND_XIAOBAN_OWNER_USER_ID", "") or "").strip()
    if (tier == "owner") != bool(owner_user_id and user_id == owner_user_id):
        return _error("当前账号与记忆等级不匹配。", "invalid_memory_scope")
    secret = resolve_memory_scope_secret()
    if not secret:
        return _error("账号记忆服务尚未配置。", "memory_unavailable")

    allowed = {
        "action",
        "target",
        "summary",
        "oldText",
        "expiresAt",
        "resourceRefs",
    }
    if set(args) - allowed:
        return _error("包含不支持的记忆字段。", "invalid_memory_fields")
    authorized_refs = [
        {"referenceId": reference_id, "sourceType": source_type}
        for reference_id, source_type in get_session_memory_resource_refs()
    ]
    try:
        resource_refs = validate_authorized_memory_resource_refs(
            args.get("resourceRefs"),
            authorized_refs=authorized_refs,
        )
    except ValueError as exc:
        return _error(str(exc), "invalid_memory_operation")
    store = open_scoped_memory_store(secret=secret, site_id=site_id, user_id=user_id)
    try:
        result = manage_account_memory(
            store,
            tier=tier,
            action=args.get("action"),
            target=args.get("target", ""),
            content=args.get("summary", ""),
            old_text=args.get("oldText", ""),
            expires_at=args.get("expiresAt", ""),
            resource_refs=resource_refs,
            account_label=(get_session_env("XIAOBAN_SESSION_USER_NAME", "").strip() or user_id),
        )
    except ValueError as exc:
        return _error(str(exc), "invalid_memory_operation")
    except Exception:
        return _error("记忆没有更新，请继续完成当前回答。", "memory_write_failed")
    finally:
        store.close()
    return _json(result)


MYSTAND_MEMORY_SCHEMA = {
    "name": "mystand_memory",
    "description": (
        "输出并执行当前 My Stand 登录账号的结构化长期记忆决定。先由 Agent/Harness 判断长期价值："
        "skip 不写；upsert 新增或用 oldText 合并旧事实；correct 更正；forget 删除。summary 只写最小结论，"
        "站内详情只放经过验证的 resourceRefs，不复制原始对话、客户私密正文、凭证或联系方式。"
        "普通账号只能维护自己的有界服务小本；主账号可维护 profile 或 journal。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["skip", "upsert", "correct", "forget"],
            },
            "target": {
                "type": "string",
                "enum": ["profile", "journal", "notebook"],
                "description": "主账号用 profile/journal；普通账号只能用 notebook。",
            },
            "summary": {
                "type": "string",
                "maxLength": 1200,
                "description": "upsert/correct 的最小长期结论，不复制对话原文。",
            },
            "oldText": {
                "type": "string",
                "description": "upsert 合并或 correct/forget 定位旧条目的短原文。",
            },
            "resourceRefs": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "referenceId": {"type": "string"},
                        "sourceType": {"type": "string"},
                    },
                    "required": ["referenceId"],
                    "additionalProperties": False,
                },
                "description": "可选的站内资源引用；详细正文不进入长期记忆。",
            },
            "expiresAt": {
                "type": "string",
                "description": "可选 YYYY-MM-DD；到期后不再召回，并在后续维护时清理。",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


registry.register(
    name="mystand_memory",
    toolset="mystand_memory",
    schema=MYSTAND_MEMORY_SCHEMA,
    handler=mystand_memory_handler,
    requires_env=[],
    is_async=False,
    description=MYSTAND_MEMORY_SCHEMA["description"],
    emoji="🧠",
)
