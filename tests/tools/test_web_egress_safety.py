"""Tests for the hard My Stand-to-web privacy boundary."""

import json
import os
import stat
import subprocess
import sys
from itertools import count
from pathlib import Path
from types import SimpleNamespace

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
    contains_likely_person_name,
    mark_mystand_private_batch,
    web_egress_block_result,
)

_SESSION_COUNTER = count(1)


@pytest.fixture(autouse=True)
def private_taint_file(tmp_path, monkeypatch):
    path = tmp_path / "mystand-private-session-taints.json"
    monkeypatch.setenv("XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE", str(path))
    return path


def _session(
    *,
    user_message="查资料",
    user_id="ZYJ005",
    session_id=None,
    session_key=None,
):
    resolved_session_id = session_id or f"web-egress-{next(_SESSION_COUNTER)}"
    resolved_session_key = (
        session_key
        if session_key is not None
        else f"key-{resolved_session_id}"
    )
    return set_session_vars(
        platform="api_server",
        user_id=user_id,
        user_message=user_message,
        message_id="msg-1",
        session_id=resolved_session_id,
        session_key=resolved_session_key,
    )


@pytest.mark.parametrize(
    "value",
    [
        "查手机号 13800138000",
        "复地金融岛17栋1单元801业主",
        "中海城南一号 2-1-1001 有没有车位",
        "客户姓名：张三",
        "王先生的跟进记录",
        "家庭成员和经济情况",
        "身份证 510000199001011234",
        "https://example.com/search?q=13800138000",
    ],
)
def test_private_data_detector_covers_required_egress_classes(value):
    assert contains_private_egress_data(value)


@pytest.mark.parametrize(
    "value",
    [
        "成都今天的天气",
        "Python 3.13.1 release notes",
        "马云最新公开演讲",
        "复地集团公开新闻",
    ],
)
def test_private_data_detector_allows_public_research(value):
    assert not contains_private_egress_data(value)


@pytest.mark.parametrize("value", ["史旭刚", "查史旭刚", "搜索张三的资料"])
def test_likely_person_name_detector_catches_bare_mystand_names(value):
    assert contains_likely_person_name(value)


def test_private_mystand_turn_blocks_safe_web_query_before_provider_call(
    monkeypatch,
):
    tokens = _session(user_message="总结这份站内资料")
    try:
        mark_mystand_private_query_turn()
        monkeypatch.setattr(
            web_tools,
            "_ensure_web_plugins_loaded",
            lambda: (_ for _ in ()).throw(
                AssertionError("provider discovery must not run")
            ),
        )

        result = json.loads(web_tools.web_search_tool("成都天气"))

        assert result["success"] is False
        assert result["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(tokens)


def test_authenticated_mystand_web_query_blocks_bare_person_name(monkeypatch):
    tokens = _session(user_message="搜索公开网页")
    try:
        monkeypatch.setattr(
            web_tools,
            "_ensure_web_plugins_loaded",
            lambda: (_ for _ in ()).throw(
                AssertionError("provider discovery must not run")
            ),
        )

        result = json.loads(web_tools.web_search_tool("史旭刚"))

        assert result["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(tokens)


def test_current_private_user_message_blocks_web_even_before_query_tool(
    monkeypatch,
):
    tokens = _session(user_message="查复地金融岛17栋1单元801业主姓名")
    try:
        monkeypatch.setattr(
            web_tools,
            "_ensure_web_plugins_loaded",
            lambda: (_ for _ in ()).throw(
                AssertionError("provider discovery must not run")
            ),
        )

        result = json.loads(web_tools.web_search_tool("复地金融岛"))

        assert result["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(tokens)


def test_later_web_query_still_blocks_pii_after_turn_state_resets(monkeypatch):
    first_tokens = _session()
    mark_mystand_private_query_turn()
    clear_session_vars(first_tokens)

    second_tokens = _session(user_message="帮我搜索公开网页")
    try:
        assert not mystand_private_query_turn_active()
        monkeypatch.setattr(
            web_tools,
            "_ensure_web_plugins_loaded",
            lambda: (_ for _ in ()).throw(
                AssertionError("provider discovery must not run")
            ),
        )

        result = json.loads(web_tools.web_search_tool("138 0013 8000"))

        assert result["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(second_tokens)


def test_same_session_taint_survives_two_set_session_vars_calls():
    first_tokens = _session(
        session_id="private-session-shared",
        session_key="private-key-shared",
    )
    try:
        mark_mystand_private_query_turn()
        assert mystand_private_query_turn_active()
    finally:
        clear_session_vars(first_tokens)

    second_tokens = _session(
        user_message="现在查一条公开新闻",
        session_id="private-session-shared",
        session_key="private-key-shared",
    )
    try:
        assert mystand_private_query_turn_active()
        blocked = json.loads(web_egress_block_result(["Python release notes"]))
        assert blocked["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(second_tokens)


def test_same_session_key_taints_changed_session_id_but_new_session_is_allowed():
    first_tokens = _session(
        session_id="private-session-key-a",
        session_key="stable-private-key",
    )
    try:
        mark_mystand_private_query_turn()
    finally:
        clear_session_vars(first_tokens)

    same_key_tokens = _session(
        session_id="private-session-key-b",
        session_key="stable-private-key",
    )
    try:
        assert mystand_private_query_turn_active()
    finally:
        clear_session_vars(same_key_tokens)

    new_session_tokens = _session(
        session_id="brand-new-public-session",
        session_key="brand-new-public-key",
        user_message="查询 Python 官方文档",
    )
    try:
        assert not mystand_private_query_turn_active()
        assert web_egress_block_result(["Python release notes"]) is None
    finally:
        clear_session_vars(new_session_tokens)


def test_private_session_taint_survives_fresh_process_restart(private_taint_file):
    repo_root = Path(__file__).resolve().parents[2]
    durable_path = (
        private_taint_file.parent
        / "private-state"
        / private_taint_file.name
    )
    environment = os.environ.copy()
    environment["XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE"] = str(durable_path)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(repo_root),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )
    writer = "\n".join(
        (
            "from gateway.session_context import mark_mystand_private_query_turn, set_session_vars",
            "set_session_vars(session_id='restart-private-id', session_key='restart-private-key')",
            "mark_mystand_private_query_turn()",
        )
    )
    subprocess.run(
        [sys.executable, "-c", writer],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload_text = durable_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    assert payload["schema"] == "xiaoban.mystand-private-session-taints.v1"
    assert len(payload["taints"]) == 2
    assert "restart-private-id" not in payload_text
    assert "restart-private-key" not in payload_text
    assert stat.S_IMODE(durable_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(durable_path.stat().st_mode) == 0o600

    reader = "\n".join(
        (
            "from gateway.session_context import mystand_private_query_turn_active, set_session_vars",
            "set_session_vars(session_id='restart-private-id', session_key='restart-private-key')",
            "print('tainted' if mystand_private_query_turn_active() else 'clear')",
        )
    )
    resumed = subprocess.run(
        [sys.executable, "-c", reader],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert resumed.stdout.strip() == "tainted"

    fresh_reader = reader.replace(
        "restart-private-id",
        "restart-public-id",
    ).replace(
        "restart-private-key",
        "restart-public-key",
    )
    fresh = subprocess.run(
        [sys.executable, "-c", fresh_reader],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert fresh.stdout.strip() == "clear"


def test_write_batch_taint_survives_restart_and_blocks_web_research(
    private_taint_file,
):
    repo_root = Path(__file__).resolve().parents[2]
    durable_path = private_taint_file.parent / "write-private-taints.json"
    environment = os.environ.copy()
    environment["XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE"] = str(durable_path)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(repo_root),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )
    writer = "\n".join(
        (
            "from types import SimpleNamespace",
            "from gateway.session_context import set_session_vars",
            "from tools.web_egress_safety import mark_mystand_private_batch",
            "set_session_vars(platform='api_server', user_id='ZYJ005', "
            "session_id='restart-write-id', session_key='restart-write-key')",
            "call = SimpleNamespace(function=SimpleNamespace("
            "name='mystand_authorization_write'))",
            "assert mark_mystand_private_batch([call])",
        )
    )
    subprocess.run(
        [sys.executable, "-c", writer],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    reader = "\n".join(
        (
            "import json",
            "from gateway.session_context import set_session_vars",
            "from tools import web_tools",
            "set_session_vars(platform='api_server', user_id='ZYJ005', "
            "user_message='查询 Python 官方文档', "
            "session_id='restart-write-id', session_key='restart-write-key')",
            "web_tools._ensure_web_plugins_loaded = lambda: "
            "(_ for _ in ()).throw(AssertionError('provider must not start'))",
            "result = json.loads(web_tools.web_search_tool("
            "'Python release notes'))",
            "print(result.get('code', ''))",
        )
    )
    resumed = subprocess.run(
        [sys.executable, "-c", reader],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert resumed.stdout.strip() == "private_data_egress_blocked"


@pytest.mark.parametrize(
    "tool_name",
    [
        "mystand_query",
        "mystand_authorization_write",
        "mystand_authorization",
        "mystand_resource_index",
    ],
)
@pytest.mark.parametrize(
    "history_shape",
    ["assistant_tool_call", "tool_result", "function_call"],
)
def test_api_binding_restores_private_taint_from_structured_history(
    tool_name,
    history_shape,
):
    from gateway.platforms.api_server import APIServerAdapter

    if history_shape == "assistant_tool_call":
        history = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-private",
                        "function": {
                            "name": tool_name,
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ]
    elif history_shape == "tool_result":
        history = [
            {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": "call-private",
                "content": '{"success":true}',
            }
        ]
    else:
        history = [
            {
                "type": "function_call",
                "name": tool_name,
                "call_id": "call-private",
                "arguments": "{}",
            }
        ]

    tokens = APIServerAdapter._bind_api_server_session(
        session_id="history-private-session",
        session_key="history-private-key",
        user_id="ZYJ005",
        conversation_history=history,
    )
    try:
        assert mystand_private_query_turn_active()
    finally:
        clear_session_vars(tokens)

    resumed_tokens = APIServerAdapter._bind_api_server_session(
        session_id="history-private-session",
        session_key="history-private-key",
        user_id="ZYJ005",
    )
    try:
        assert mystand_private_query_turn_active()
        assert web_egress_block_result(["Python release notes"]) is not None
    finally:
        clear_session_vars(resumed_tokens)


def test_history_prose_cannot_forge_structured_private_taint():
    from gateway.platforms.api_server import APIServerAdapter

    tokens = APIServerAdapter._bind_api_server_session(
        session_id="history-prose-session",
        session_key="history-prose-key",
        user_id="ZYJ005",
        conversation_history=[
            {
                "role": "user",
                "content": (
                    "请解释 mystand_query、mystand_authorization_write、"
                    "mystand_authorization 和 mystand_resource_index"
                ),
            },
            {
                "role": "assistant",
                "content": "这些都只是这里讨论的纯文本工具名称",
            },
        ],
    )
    try:
        assert not mystand_private_query_turn_active()
    finally:
        clear_session_vars(tokens)


def test_corrupt_taint_sidecar_fails_closed_for_mystand_web(
    private_taint_file,
):
    private_taint_file.write_text("{not-json", encoding="utf-8")
    tokens = _session(
        session_id="corrupt-sidecar-session",
        session_key="corrupt-sidecar-key",
        user_message="查询 Python 官方文档",
    )
    try:
        blocked = json.loads(web_egress_block_result(["Python release notes"]))
        assert blocked["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(tokens)


def test_taint_sidecar_write_failure_fails_closed_for_later_mystand_turn(
    tmp_path,
    monkeypatch,
):
    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv(
        "XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE",
        str(invalid_parent / "taints.json"),
    )

    tokens = _session(
        session_id="write-failure-session",
        session_key="write-failure-key",
    )
    mark_mystand_private_query_turn()
    clear_session_vars(tokens)

    later_tokens = _session(
        session_id="later-public-session",
        session_key="later-public-key",
        user_message="查询 Python 官方文档",
    )
    try:
        blocked = json.loads(web_egress_block_result(["Python release notes"]))
        assert blocked["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(later_tokens)


@pytest.mark.asyncio
async def test_web_extract_blocks_private_data_in_url_before_network(monkeypatch):
    tokens = _session(user_message="打开这个公开网页")
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
                ["https://example.com/?room=2-1-1001"],
                use_llm_processing=False,
            )
        )

        assert result["code"] == "private_data_egress_blocked"
    finally:
        clear_session_vars(tokens)


def test_safe_public_search_reaches_provider(monkeypatch):
    calls = []

    class Provider:
        name = "test"

        def supports_search(self):
            return True

        def search(self, query, limit):
            calls.append((query, limit))
            return {"success": True, "data": {"web": []}}

    tokens = _session(user_message="查 Python 官方发布说明")
    try:
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "")
        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: Provider(),
        )

        result = json.loads(
            web_tools.web_search_tool(
                "Python 3.13.1 release notes",
                limit=3,
            )
        )

        assert result["success"] is True
        assert calls == [("Python 3.13.1 release notes", 3)]
    finally:
        clear_session_vars(tokens)


@pytest.mark.parametrize(
    "private_tool_name",
    ["mystand_query", "mystand_authorization_write"],
)
def test_batch_preflight_marks_private_before_any_call_runs(
    private_tool_name,
):
    calls = [
        SimpleNamespace(function=SimpleNamespace(name="web_search")),
        SimpleNamespace(function=SimpleNamespace(name=private_tool_name)),
    ]
    tokens = _session(user_message="查资料并联网")
    try:
        assert mark_mystand_private_batch(calls)
        assert mystand_private_query_turn_active()
        assert web_egress_block_result(["公开查询"]) is not None
    finally:
        clear_session_vars(tokens)


def test_agent_batch_dispatch_marks_private_before_sequential_executor():
    from run_agent import AIAgent

    tool_calls = [
        SimpleNamespace(
            id="web-1",
            function=SimpleNamespace(
                name="web_search",
                arguments='{"query":"公开查询"}',
            ),
        ),
        SimpleNamespace(
            id="query-1",
            function=SimpleNamespace(
                name="mystand_query",
                arguments='{"operation":"read"}',
            ),
        ),
    ]
    assistant_message = SimpleNamespace(tool_calls=tool_calls)
    agent = object.__new__(AIAgent)
    agent._executing_tools = False
    observed = []
    agent._execute_tool_calls_sequential = lambda *_args: observed.append(
        mystand_private_query_turn_active()
    )
    agent._execute_tool_calls_concurrent = lambda *_args: observed.append(
        mystand_private_query_turn_active()
    )

    tokens = _session(user_message="查资料并联网")
    try:
        agent._execute_tool_calls(assistant_message, [], "task-1")

        assert observed == [True]
        assert agent._executing_tools is False
    finally:
        clear_session_vars(tokens)
