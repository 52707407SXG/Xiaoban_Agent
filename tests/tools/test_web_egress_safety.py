"""Tests for intent-led web research with a field-level privacy boundary."""

import json

import pytest

from gateway.session_context import (
    clear_session_vars,
    mark_mystand_private_query_turn,
    mystand_private_query_turn_active,
    set_session_vars,
)
from tools import web_tools
from tools.web_egress_safety import (
    contains_private_egress_data,
    web_egress_block_result,
)


@pytest.fixture(autouse=True)
def private_taint_file(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE",
        str(tmp_path / "mystand-private-session-taints.json"),
    )


def _session(*, user_message="查公开资料", session_id="web-egress-test"):
    return set_session_vars(
        platform="api_server",
        user_id="ZYJ005",
        user_message=user_message,
        message_id="msg-1",
        session_id=session_id,
        session_key=f"key-{session_id}",
    )


@pytest.mark.parametrize(
    "value",
    [
        "查手机号 13800138000",
        "客户姓名：张三",
        "身份证 510000199001011234",
        "邮箱 test@example.com",
        "家庭住址：成都市某街道12号",
        "https://example.com/search?q=13800138000",
    ],
)
def test_field_detector_blocks_identifying_values(value):
    assert contains_private_egress_data(value)


@pytest.mark.parametrize(
    "value",
    [
        "成都今天的天气",
        "复地金融岛149㎡户型现在什么价格",
        "中海城南一号 2-1-1001 公开挂牌价格",
        "史旭刚公开新闻",
        "如何保护客户资料隐私",
    ],
)
def test_field_detector_allows_public_research_terms(value):
    assert not contains_private_egress_data(value)
    assert web_egress_block_result([value]) is None


def _install_provider(monkeypatch, calls):
    class Provider:
        name = "test"

        def supports_search(self):
            return True

        def search(self, query, limit):
            calls.append((query, limit))
            return {"success": True, "data": {"web": []}}

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "")
    monkeypatch.setattr(
        "agent.web_search_registry.get_active_search_provider",
        lambda: Provider(),
    )


def test_prior_mystand_read_does_not_disable_public_search(monkeypatch):
    calls = []
    _install_provider(monkeypatch, calls)
    tokens = _session(session_id="formerly-tainted-session")
    try:
        mark_mystand_private_query_turn()
        assert mystand_private_query_turn_active()

        result = json.loads(
            web_tools.web_search_tool(
                "复地金融岛149㎡户型现在什么价格",
                limit=3,
            )
        )

        assert result["success"] is True
        assert calls == [("复地金融岛149㎡户型现在什么价格", 3)]
    finally:
        clear_session_vars(tokens)


def test_private_user_message_can_emit_sanitized_public_query(monkeypatch):
    calls = []
    _install_provider(monkeypatch, calls)
    tokens = _session(
        user_message="查复地金融岛某业主的电话，并了解项目公开行情",
        session_id="mixed-intent-session",
    )
    try:
        result = json.loads(web_tools.web_search_tool("复地金融岛公开行情"))

        assert result["success"] is True
        assert calls == [("复地金融岛公开行情", 5)]
    finally:
        clear_session_vars(tokens)


def test_sensitive_outbound_query_is_blocked_before_provider(monkeypatch):
    tokens = _session()
    try:
        monkeypatch.setattr(
            web_tools,
            "_ensure_web_plugins_loaded",
            lambda: (_ for _ in ()).throw(
                AssertionError("provider discovery must not run")
            ),
        )

        result = json.loads(web_tools.web_search_tool("客户姓名：张三 13800138000"))

        assert result["success"] is False
        assert result["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_web_extract_blocks_sensitive_url_before_network(monkeypatch):
    tokens = _session()
    try:
        monkeypatch.setattr(
            web_tools,
            "normalize_url_for_request",
            lambda _url: (_ for _ in ()).throw(
                AssertionError("URL normalization must not run")
            ),
        )

        result = json.loads(
            await web_tools.web_extract_tool(
                ["https://example.com/?phone=13800138000"],
                use_llm_processing=False,
            )
        )

        assert result["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(tokens)


def test_api_history_no_longer_restores_session_wide_web_block():
    from gateway.platforms.api_server import APIServerAdapter

    tokens = APIServerAdapter._bind_api_server_session(
        session_id="history-session",
        session_key="history-key",
        user_id="ZYJ005",
        conversation_history=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-private",
                        "function": {"name": "mystand_query", "arguments": "{}"},
                    }
                ],
            }
        ],
    )
    try:
        assert not mystand_private_query_turn_active()
        assert web_egress_block_result(["Python release notes"]) is None
    finally:
        clear_session_vars(tokens)
