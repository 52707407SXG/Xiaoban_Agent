"""Privacy tests for the My Stand parser gateway wrapper."""

import json

import pytest

from gateway.session_context import (
    clear_session_vars,
    mark_mystand_private_query_turn,
    set_session_vars,
)
from tools import mystand_parser_tool as parser_bridge
from tools import web_tools


@pytest.fixture
def api_session(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE",
        str(tmp_path / "private-taints.json"),
    )
    tokens = set_session_vars(
        platform="api_server",
        user_id="ZYJ005",
        user_message="请解析 https://example.com/public/report.pdf",
        message_id="msg-parser-1",
        session_id="session-parser-1",
        session_key="key-parser-1",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def _forbid_remote_work(monkeypatch):
    monkeypatch.setattr(
        parser_bridge,
        "_upstream_mystand_parse_tool_handler",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parser subprocess must not start")
        ),
    )
    monkeypatch.setattr(
        web_tools,
        "_ensure_web_plugins_loaded",
        lambda: (_ for _ in ()).throw(
            AssertionError("web provider discovery must not start")
        ),
    )


def test_api_session_local_file_stops_before_parser_subprocess(
    monkeypatch,
    api_session,
):
    _forbid_remote_work(monkeypatch)

    result = json.loads(
        parser_bridge.mystand_parse_tool_handler(
            {"input": "/tmp/uploaded-report.pdf"},
            task_id="task-parser",
        )
    )

    assert result["success"] is False
    assert result["code"] == "local_parser_path_not_allowed"


def test_api_session_without_user_id_still_blocks_local_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE",
        str(tmp_path / "private-taints.json"),
    )
    _forbid_remote_work(monkeypatch)
    tokens = set_session_vars(
        platform="api_server",
        user_id="",
        user_message="解析文件",
        message_id="anonymous-api-message",
        session_id="anonymous-api-session",
        session_key="anonymous-api-key",
    )
    try:
        result = json.loads(
            parser_bridge.mystand_parse_tool_handler(
                {"input": "/etc/passwd"}
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["success"] is False
    assert result["code"] == "local_parser_path_not_allowed"


def test_non_api_connector_local_file_preserves_upstream_behavior(
    tmp_path,
    monkeypatch,
):
    calls = []
    monkeypatch.setenv(
        "XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE",
        str(tmp_path / "private-taints.json"),
    )
    monkeypatch.setattr(
        parser_bridge,
        "_upstream_mystand_parse_tool_handler",
        lambda args, **kwargs: calls.append((args, kwargs)) or '{"ok":true}',
    )
    tokens = set_session_vars(
        platform="telegram",
        user_id="telegram-user",
        user_message="解析刚上传的文件",
        message_id="telegram-message",
        session_id="telegram-session",
        session_key="telegram-key",
    )
    try:
        result = json.loads(
            parser_bridge.mystand_parse_tool_handler(
                {"input": "/tmp/uploaded-report.pdf"},
                task_id="task-parser",
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["ok"] is True
    assert calls == [
        (
            {"input": "/tmp/uploaded-report.pdf"},
            {"task_id": "task-parser"},
        )
    ]


def test_exact_public_url_from_trusted_user_message_reaches_upstream(
    monkeypatch,
    api_session,
):
    calls = []
    monkeypatch.setattr(
        parser_bridge,
        "_upstream_mystand_parse_tool_handler",
        lambda args, **_kwargs: calls.append(args) or '{"ok":true}',
    )

    result = json.loads(
        parser_bridge.mystand_parse_tool_handler(
            {"input": "https://example.com/public/report.pdf"}
        )
    )

    assert result["ok"] is True
    assert calls == [{"input": "https://example.com/public/report.pdf"}]


@pytest.mark.parametrize(
    "remote_input",
    [
        "https://example.com/public/report.pdf?model_added=1",
        "https://example.com/public",
        "ftp://example.com/public/report.pdf",
        "file:///etc/passwd",
    ],
)
def test_untrusted_or_unsupported_remote_input_stops_before_provider_and_subprocess(
    remote_input,
    monkeypatch,
    api_session,
):
    _forbid_remote_work(monkeypatch)

    result = json.loads(
        parser_bridge.mystand_parse_tool_handler({"input": remote_input})
    )

    assert result["success"] is False
    assert result["code"] in {
        "untrusted_remote_parser_url",
        "unsupported_remote_parser_scheme",
    }


def test_prior_private_turn_does_not_block_exact_public_url(
    monkeypatch,
    api_session,
):
    calls = []
    monkeypatch.setattr(
        parser_bridge,
        "_upstream_mystand_parse_tool_handler",
        lambda args, **_kwargs: calls.append(args) or '{"ok":true}',
    )
    mark_mystand_private_query_turn()

    result = json.loads(
        parser_bridge.mystand_parse_tool_handler(
            {"input": "https://example.com/public/report.pdf"}
        )
    )

    assert result["ok"] is True
    assert calls == [{"input": "https://example.com/public/report.pdf"}]
