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
import unicodedata
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
    build_authorization_write_payload_schema,
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
        "Use My Stand's server-enforced authorization wall. This is the only "
        "tool for Xiaoban to list or resolve AUTH/OUT records and to preview or "
        "commit supported My Stand writes. Never read a database or local file "
        "instead.\n\n"
        "READS: list returns only the current user's authorization index. "
        "For a content question that names a resource, prefer one resolve call "
        "with resource_query plus query; resource_query must reuse the resource-name "
        "wording present in the current trusted user message. This bridge silently searches the current "
        "user's resource index, requires one unambiguous match, and then resolves "
        "its Xiaoban-bound default AUTH. If mystand_resource_index already found a "
        "resource, resolve it with the exact resource_uid. Never pass resource_uid, source_id, "
        "KGREF, OUT, or module IDs as authorization_id. Direct AUTH/OUT resolve "
        "remains supported and is re-checked before content is returned. A feature "
        "explanation does not need this tool; real user data does. Resource discovery, "
        "index maps, AUTH IDs, internal queries, retries, truncation, and tool output "
        "are private backend details: never narrate them to the user; answer naturally "
        "from the final authorized result.\n\n"
        "WRITES: OUT can never write. Only the fixed allowlisted actions are supported. "
        "First call preview_write with an internal AUTH whose canWrite is true, "
        "the action payload, and a fresh idempotency_key; My Stand reads the "
        "current target version authoritatively. "
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
            "resource_query": {
                "type": "string",
                "description": "Safe resource-title search text for one-call resolve when the user names a resource but no resource_uid is known.",
            },
            "module_id": {
                "type": "string",
                "description": "Optional module filter used only with resource_query, such as property-dev.",
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
            "action": {
                "type": "string",
                "enum": sorted(_WRITE_ACTIONS),
                "description": "Fixed write action for preview_write.",
            },
            "payload": build_authorization_write_payload_schema(),
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


def _strip_resource_title_suffix(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    folded = text.casefold()
    if folded.endswith("楼盘md"):
        return text[: -len("楼盘md")].strip()
    for suffix in ("markdown", "md"):
        if not folded.endswith(suffix):
            continue
        prefix = text[: -len(suffix)]
        if prefix and (prefix[-1].isspace() or "\u4e00" <= prefix[-1] <= "\u9fff"):
            return prefix.strip()
    return text


def _resource_title_key(value: str) -> str:
    text = _strip_resource_title_suffix(value).casefold()
    text = re.sub(r"[\s\-_—–/／·,，.。()（）【】\[\]]+", "", text)
    return text


def _resource_search_text(value: str) -> str:
    return _strip_resource_title_suffix(value)


def _resource_query_is_specific(value: str) -> bool:
    key = _resource_title_key(value)
    if any("\u4e00" <= char <= "\u9fff" for char in key):
        return len(key) >= 2
    return len(key) >= 3


def _resource_query_match_occurrences(
    resource_query: str,
    user_message: str,
) -> list[tuple[int, bool]]:
    needle = unicodedata.normalize("NFKC", str(resource_query or "")).casefold().strip()
    haystack = unicodedata.normalize("NFKC", str(user_message or "")).casefold()
    if not needle or not haystack:
        return []
    location_or_fact = (
        r"(?:"
        r"[零〇○一二两三四五六七八九十百\d]{1,4}\s*"
        r"(?:栋|幢|座|号楼|单元)"
        r"|查|查看|查询|找|读取|看看|看一下"
        r"|业主|联系人|姓名|名字|电话|手机|联系方式"
        r"|车位|停车|面积|建面|价格|报价|总价|单价|租金"
        r"|内容|资料|情况|信息|有没有|有无|是否|这套|房源|房子"
        r")"
    )
    connected_fact = rf"(?:的|里面|里|中)(?:的)?{location_or_fact}"
    occurrences = []
    offset = 0
    while True:
        index = haystack.find(needle, offset)
        if index < 0:
            return occurrences
        prefix = haystack[:index].rstrip()
        suffix = haystack[index + len(needle):].lstrip()
        previous = prefix[-1:] if prefix else ""
        before_ok = (
            not prefix
            or not (previous.isalnum() or "\u4e00" <= previous <= "\u9fff")
            or re.search(
                r"(?:请|麻烦|帮我|帮忙)?"
                r"(?:查一下|查|查看|查询|搜索|找|打开|读取|看看|看一下)"
                r"(?:一下)?$",
                prefix,
            )
            is not None
            or prefix.endswith("楼盘md")
        )
        after_ok = (
            not suffix
            or not (suffix[0].isalnum() or "\u4e00" <= suffix[0] <= "\u9fff")
            or re.match(rf"^{location_or_fact}", suffix) is not None
            or re.match(rf"^{connected_fact}", suffix) is not None
            or re.match(
                r"^(?:楼盘)?\s*(?:md|markdown)"
                r"(?=$|[\s,，。；;])",
                suffix,
            )
            is not None
            or re.match(
                rf"^(?:楼盘)?\s*(?:md|markdown)\s*{connected_fact}",
                suffix,
            )
            is not None
            or re.match(
                r"^(?:楼盘)?\s*(?:md|markdown)\s*"
                r"[零〇○一二两三四五六七八九十百\d]{1,4}\s*"
                r"(?:栋|幢|座|号楼|单元)",
                suffix,
            )
            is not None
            or re.match(
                r"^(?:[零〇○一二两三四五六七八九十百\d]{1,4})\s*(?:栋|幢|座|号楼|单元)",
                suffix,
            )
            is not None
        )
        if before_ok and after_ok:
            clause_start = max(
                (
                    haystack.rfind(separator, 0, index)
                    for separator in ("，", ",", "。", "；", ";", "\n")
                ),
                default=-1,
            )
            local_prefix = haystack[clause_start + 1:index]
            negated_prefix = re.search(
                r"(?:不要|别|不查|不看|不找|不打开|不读取|排除|忽略|不是|并非)"
                r"(?:\s|帮我|帮忙|再|先|去|给我|查一下|查|查看|查询|搜索|找|"
                r"打开|读取|看看|看一下)*$",
                local_prefix,
            )
            negated_suffix = re.match(r"^\s*(?:以外|之外|除外)", suffix)
            occurrences.append(
                (index, negated_prefix is not None or negated_suffix is not None)
            )
        offset = index + 1


def _resource_query_latest_affirmative_index(
    resource_query: str,
    user_message: str,
) -> int | None:
    occurrences = _resource_query_match_occurrences(resource_query, user_message)
    if not occurrences or occurrences[-1][1]:
        return None
    latest_index = occurrences[-1][0]
    normalized_title = unicodedata.normalize(
        "NFKC",
        str(resource_query or ""),
    ).casefold().strip()
    normalized_message = unicodedata.normalize(
        "NFKC",
        str(user_message or ""),
    ).casefold()
    trailing_text = normalized_message[latest_index + len(normalized_title):]
    deictic = r"(?:这|那|该|它|刚才|之前|前面|后面|前者|后者|最后|最终|前|后)"
    cancellation = r"(?:不|别|甭|无需|无须|排除|忽略|取消|去掉)"
    deictic_cancellation = (
        re.search(
            rf"{deictic}[^，,。；;\n]{{0,20}}{cancellation}",
            trailing_text,
        )
        or re.search(
            rf"{cancellation}[^，,。；;\n]{{0,20}}{deictic}",
            trailing_text,
        )
    )
    if deictic_cancellation is not None:
        return None
    return latest_index


def _resource_query_matches_user_phrase(resource_query: str, user_message: str) -> bool:
    return _resource_query_latest_affirmative_index(
        resource_query,
        user_message,
    ) is not None


def _resolve_resource_query(
    resource_query: str,
    module_id: str,
    session: dict,
    trusted_user_message: str,
) -> tuple[str, str]:
    query_key = _resource_title_key(resource_query)
    search_text = _resource_search_text(resource_query)
    if not query_key or not search_text:
        return "", _error(
            "资料名称无效。",
            code="resource_query_not_found",
            status=404,
        )
    items = []
    cursor = ""
    has_more = False
    for _page in range(5):
        raw = _post_internal(
            "/api/xiaoban/internal/resource-index",
            {
                "operation": "list_resources",
                "moduleId": module_id[:80],
                "query": search_text[:240],
                "status": "all",
                "cursor": cursor,
                "limit": 100,
            },
            session=session,
        )
        try:
            result = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return "", _error(
                "My Stand 资料定位结果无效。",
                code="resource_query_failed",
                status=502,
            )
        if not result.get("ok"):
            return "", raw
        items.extend(
            item for item in result.get("items", []) if isinstance(item, dict)
        )
        has_more = result.get("hasMore") is True
        if not has_more:
            break
        next_cursor = str(result.get("nextCursor") or "").strip()
        if not next_cursor or next_cursor == cursor:
            return "", _error(
                "My Stand 资料索引分页结果无效。",
                code="resource_query_failed",
                status=502,
            )
        cursor = next_cursor
    raw_partial = [
        item
        for item in items
        if query_key in _resource_title_key(item.get("safeLabel", ""))
    ]
    raw_candidates = raw_partial
    verified_candidates = []
    for item in raw_candidates:
        match_index = _resource_query_latest_affirmative_index(
            _strip_resource_title_suffix(item.get("safeLabel", "")),
            trusted_user_message,
        )
        if match_index is not None:
            verified_candidates.append((item, match_index))
    candidates = []
    if verified_candidates:
        latest_mention = max(match_index for _item, match_index in verified_candidates)
        latest_candidates = [
            item
            for item, match_index in verified_candidates
            if match_index == latest_mention
        ]
        most_specific = max(
            len(_resource_title_key(item.get("safeLabel", "")))
            for item in latest_candidates
        )
        candidates = [
            item
            for item in latest_candidates
            if len(_resource_title_key(item.get("safeLabel", ""))) == most_specific
        ]
    if not candidates:
        if raw_candidates:
            labels = [
                str(item.get("safeLabel") or "")[:120]
                for item in raw_candidates[:8]
            ]
            return "", _json_result(
                {
                    "ok": False,
                    "status": 409,
                    "code": "resource_query_ambiguous",
                    "error": "资料名称还不够完整，请按列表补充完整名称。",
                    "candidates": labels,
                }
            )
        return "", _error(
            "没有找到名称匹配的可用资料。",
            code="resource_query_not_found",
            status=404,
        )
    if len(candidates) != 1 or has_more:
        labels = [str(item.get("safeLabel") or "")[:120] for item in candidates[:8]]
        return "", _json_result(
            {
                "ok": False,
                "status": 409,
                "code": "resource_query_ambiguous",
                "error": "资料名称对应多项结果，请补充更完整的资料名称。",
                "candidates": labels,
            }
        )
    selected = candidates[0]
    if selected.get("canRead") is not True:
        return "", _error(
            "这份资料当前未授权给小伴读取。",
            code="resource_query_not_readable",
            status=403,
        )
    resource_uid = str(selected.get("resourceUid") or "").strip()
    if not resource_uid:
        return "", _error(
            "资料索引缺少可解析节点。",
            code="resource_query_failed",
            status=502,
        )
    return resource_uid, ""


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
            resource_query = str(args.get("resource_query") or "").strip()
            trusted_user_message = get_session_user_message().strip()
            if not resource_uid and not authorization_id and not resource_query:
                raise ValueError("缺少 resource_uid、resource_query 或 authorization_id")
            if not resource_uid and not authorization_id:
                if not trusted_user_message:
                    return _error(
                        "当前用户消息没有可验证的资料名称。",
                        code="trusted_resource_query_required",
                        status=409,
                    )
                if not _resource_query_is_specific(resource_query):
                    return _error(
                        "资料名称过短，不能安全定位。",
                        code="resource_query_too_short",
                        status=409,
                    )
                if not _resource_query_matches_user_phrase(
                    resource_query,
                    trusted_user_message,
                ):
                    return _error(
                        "资料名称必须直接来自用户当前消息。",
                        code="resource_query_not_in_user_message",
                        status=409,
                    )
                resource_uid, lookup_error = _resolve_resource_query(
                    resource_query,
                    str(args.get("module_id") or "").strip(),
                    session,
                    trusted_user_message,
                )
                if lookup_error:
                    return lookup_error
            query = trusted_user_message
            if not query:
                query = str(args.get("query") or "").strip()
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
