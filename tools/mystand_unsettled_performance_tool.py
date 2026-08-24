"""Exact owner-only bridge for the My Stand unpaid-performance card view."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from gateway.session_context import get_session_env, mark_mystand_private_query_turn
from tools.registry import registry


_DEFAULT_API_URL = "http://127.0.0.1:18081"
_DEFAULT_ENV_FILE = "/opt/xiaoban-agent/.env"
_INTERNAL_PATH = "/api/xiaoban/internal/finance/unsettled-ready"
_MAX_RESPONSE_BYTES = 65_536
_TOKEN_KEYS = (
    "MYSTAND_XIAOBAN_MYSTAND_API_TOKEN",
    "MYSTAND_XIAOBAN_GATEWAY_INTERNAL_TOKEN",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _error(message: str, code: str, status: int) -> str:
    return _json({"ok": False, "status": status, "code": code, "error": message})


def _env_file_value(path: str, key: str) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    return ""


def _internal_token() -> str:
    for key in _TOKEN_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            return value
    env_file = os.getenv("MYSTAND_XIAOBAN_GATEWAY_ENV_FILE", _DEFAULT_ENV_FILE)
    for key in _TOKEN_KEYS:
        value = _env_file_value(env_file, key)
        if value:
            return value
    return ""


def _api_base_url() -> str:
    value = os.getenv(
        "MYSTAND_XIAOBAN_MYSTAND_API_URL",
        _DEFAULT_API_URL,
    ).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        return ""
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return ""
    return value


def _safe_header(value: object, limit: int = 200) -> str:
    text = str(value or "")
    if len(text) > limit or not re.fullmatch(r"[A-Za-z0-9._:@-]+", text):
        return ""
    return text


def check_mystand_unsettled_performance() -> bool:
    return bool(_internal_token() and _api_base_url())


def mystand_unsettled_performance_handler(args, **_kwargs) -> str:
    if args not in ({}, None):
        return _error(
            "查未结算不接收筛选参数，只读取当前业绩未结算页面。",
            "invalid_unsettled_performance_arguments",
            400,
        )
    platform = get_session_env("XIAOBAN_SESSION_PLATFORM", "").strip().lower()
    user_id = _safe_header(get_session_env("XIAOBAN_SESSION_USER_ID", ""))
    if platform != "api_server" or not user_id:
        return _error(
            "该查询只允许 My Stand 已登录管理员会话使用。",
            "mystand_session_required",
            403,
        )
    token = _internal_token()
    base_url = _api_base_url()
    if not token or not base_url:
        return _error(
            "业绩未结算查询暂时不可用，请稍后重试。",
            "mystand_unsettled_performance_unavailable",
            503,
        )
    mark_mystand_private_query_turn()
    request = urllib.request.Request(
        f"{base_url}{_INTERNAL_PATH}",
        data=_json({"userId": user_id}).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "X-Xiaoban-User-Id": user_id,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            return _error(
                "业绩未结算结果过大，已停止读取。",
                "mystand_unsettled_performance_result_too_large",
                413,
            )
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            return _error(
                "业绩未结算页面返回了无效结果。",
                "mystand_unsettled_performance_invalid_result",
                502,
            )
        return _json({
            "ok": True,
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        })
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        message = (
            "当前账号无权读取业绩未结算公司聚合。"
            if status == 403
            else "业绩未结算页面暂时不可用。"
        )
        return _error(
            message,
            "mystand_unsettled_performance_rejected",
            status,
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return _error(
            "业绩未结算查询通道暂时没有接稳，请稍后重试。",
            "mystand_unsettled_performance_transport_failed",
            502,
        )


MYSTAND_UNSETTLED_PERFORMANCE_SCHEMA = {
    "name": "mystand_unsettled_performance",
    "description": (
        "只读查询 My Stand『财务模块 → 业绩总览 → 业绩未结算』当前卡片，"
        "并只返回仍未发提成且卡片双『是』（主确认项=是、店长确认=是）的可发工资记录。"
        "查未结算、没结算的单子、哪些单子没发钱等意图只能调用本工具一次；"
        "不要查询个人业务档案、月份或其他财务路径。"
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


registry.register(
    name="mystand_unsettled_performance",
    toolset="mystand_unsettled_performance",
    schema=MYSTAND_UNSETTLED_PERFORMANCE_SCHEMA,
    handler=mystand_unsettled_performance_handler,
    check_fn=check_mystand_unsettled_performance,
    requires_env=[],
    is_async=False,
    description=MYSTAND_UNSETTLED_PERFORMANCE_SCHEMA["description"],
    emoji="💰",
    max_result_size_chars=_MAX_RESPONSE_BYTES,
)
