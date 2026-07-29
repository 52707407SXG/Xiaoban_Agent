"""Read-only My Stand resource-index bridge for authenticated API sessions."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from gateway.session_context import get_session_env
from tools.registry import registry

_DEFAULT_API_URL = "http://127.0.0.1:18081"
_DEFAULT_ENV_FILE = "/opt/xiaoban-agent/.env"
_INTERNAL_TOKEN_KEYS = (
    "MYSTAND_XIAOBAN_MYSTAND_API_TOKEN",
    "MYSTAND_XIAOBAN_GATEWAY_INTERNAL_TOKEN",
)
_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "contracts" / "mystand-resource-index-tool.v1.json"
RESOURCE_INDEX_CONTRACT = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
MYSTAND_RESOURCE_INDEX_SCHEMA = RESOURCE_INDEX_CONTRACT["tool"]


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
        if name.strip() == key:
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


def check_mystand_resource_index() -> bool:
    return bool(_internal_token() and _api_base_url())


def _json_result(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _error(message: str, *, code: str = "mystand_resource_index_failed", status: int = 400) -> str:
    return _json_result({"ok": False, "status": status, "code": code, "error": message})


def _safe_header(value: str, limit: int = 200) -> str:
    text = str(value or "")
    if len(text) > limit or not re.fullmatch(r"[A-Za-z0-9._:@-]+", text):
        return ""
    return text


def _post_internal(payload: dict, user_id: str) -> str:
    base_url = _api_base_url()
    token = _internal_token()
    if not base_url or not token:
        return _error("My Stand 资源索引桥尚未配置。", code="mystand_resource_index_unavailable", status=503)
    safe_user = _safe_header(user_id)
    if not safe_user:
        return _error("当前 My Stand 登录身份无效。", code="mystand_session_required", status=403)
    request = urllib.request.Request(
        f"{base_url}/api/xiaoban/internal/resource-index",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "X-Xiaoban-User-Id": safe_user,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(300_001)
            if len(raw) > 300_000:
                return _error("My Stand 资源索引页过大，已停止读取。", code="mystand_resource_index_result_too_large", status=413)
            return _json_result(json.loads(raw.decode("utf-8")) if raw else {})
    except urllib.error.HTTPError as exc:
        try:
            parsed = json.loads(exc.read(100_000).decode("utf-8"))
        except Exception:
            parsed = {}
        return _error(
            str(parsed.get("error") or "My Stand 拒绝了这次索引读取")[:300],
            code=str(parsed.get("code") or parsed.get("error") or "mystand_resource_index_rejected")[:120],
            status=int(exc.code),
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return _error("My Stand 资源索引暂时没有接稳，请稍后重试。", code="mystand_resource_index_transport_failed", status=502)


def mystand_resource_index_tool_handler(args, **_kwargs):
    platform = get_session_env("XIAOBAN_SESSION_PLATFORM", "").strip().lower()
    user_id = get_session_env("XIAOBAN_SESSION_USER_ID", "").strip()
    if platform != "api_server" or not user_id:
        return _error("该索引工具只允许 My Stand 已登录网页/API 会话使用。", code="mystand_session_required", status=403)
    raw_limit = args.get("limit", 50)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return _error("limit 必须是 1 到 100 的整数", code="invalid_resource_index_limit")
    if limit < 1 or limit > 100:
        return _error("limit 必须是 1 到 100 的整数", code="invalid_resource_index_limit")
    status = str(args.get("status") or "all").strip()
    if status not in {"all", "active", "archived", "locked"}:
        return _error("status 不在允许范围内", code="invalid_resource_index_status")
    return _post_internal({
        "operation": "list_resources",
        "moduleId": str(args.get("module_id") or "").strip()[:80],
        "query": str(args.get("query") or "").strip()[:240],
        "status": status,
        "cursor": str(args.get("cursor") or "").strip()[:800],
        "limit": limit,
    }, user_id)


registry.register(
    name="mystand_resource_index",
    toolset="mystand_resource_index",
    schema=MYSTAND_RESOURCE_INDEX_SCHEMA,
    handler=mystand_resource_index_tool_handler,
    check_fn=check_mystand_resource_index,
    requires_env=[],
    is_async=False,
    description="Read-only server-filtered My Stand resource index",
    emoji="🗺️",
    max_result_size_chars=300_000,
)
