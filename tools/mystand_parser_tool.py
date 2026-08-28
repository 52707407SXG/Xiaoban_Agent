"""Generated Xiaoban tool bridge for MyStand Parser Tools."""

from __future__ import annotations

import json
import os
import re
import sys
from urllib.parse import urlsplit

_PARSER_SRC = os.environ.get('MYSTAND_PARSER_PYTHONPATH', '/opt/mystand-parser-tools/src')
if _PARSER_SRC and _PARSER_SRC not in sys.path:
    sys.path.insert(0, _PARSER_SRC)

from mystand_parser_tools.xiaoban import (
    MYSTAND_PARSE_SCHEMA,
    check_mystand_parser,
    mystand_parse_tool_handler as _upstream_mystand_parse_tool_handler,
)
from gateway.session_context import get_session_env, get_session_user_message
from tools.registry import registry
from tools.web_egress_safety import web_egress_block_result

_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING_PROSE = ".,;:!?，。；：！？、)]}）】》"


def _tool_error(message: str, *, code: str) -> str:
    return json.dumps(
        {
            "success": False,
            "code": code,
            "error": message,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _trusted_user_http_urls() -> set[str]:
    message = get_session_user_message()
    return {
        match.group(0).rstrip(_URL_TRAILING_PROSE)
        for match in _HTTP_URL_RE.finditer(message)
    }


def _is_api_server_session() -> bool:
    return (
        get_session_env("XIAOBAN_SESSION_PLATFORM", "").strip().lower()
        == "api_server"
    )


def mystand_parse_tool_handler(args, **kwargs):
    """Keep local parsing local and hard-gate every remote parser input."""
    input_uri = (
        str(args.get("input") or "").strip()
        if isinstance(args, dict)
        else ""
    )
    parsed = urlsplit(input_uri)
    scheme = parsed.scheme.lower()

    if not scheme and _is_api_server_session():
        return _tool_error(
            "Blocked: local parser paths are not available in My Stand web/API chat.",
            code="local_parser_path_not_allowed",
        )
    if scheme in {"http", "https"}:
        blocked = web_egress_block_result([input_uri])
        if blocked is not None:
            return blocked
        if input_uri not in _trusted_user_http_urls():
            return _tool_error(
                "Blocked: remote parser URL must exactly match a URL in the "
                "current trusted user message.",
                code="untrusted_remote_parser_url",
            )
    elif scheme:
        return _tool_error(
            "Blocked: parser inputs only allow local paths or trusted http(s) URLs.",
            code="unsupported_remote_parser_scheme",
        )

    return _upstream_mystand_parse_tool_handler(args, **kwargs)

registry.register(
    name='mystand_parse',
    toolset='mystand_parser',
    schema=MYSTAND_PARSE_SCHEMA,
    handler=mystand_parse_tool_handler,
    check_fn=check_mystand_parser,
    requires_env=[],
    is_async=False,
    description='Parse files and URLs with MyStand Parser Tools',
    emoji='\U0001f4c4',
)
