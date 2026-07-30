"""Regression tests for dynamic tool staging and finalize-only framing."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.conversation_loop import (
    _contains_raw_tool_protocol_content,
    _prepare_finalize_only_call,
    _reject_finalize_only_protocol_candidate,
)
from agent.chat_completion_helpers import build_api_kwargs
from agent.transports.bedrock import BedrockTransport
from agent.tool_executor import _trusted_preaction_denial
from gateway.session_context import clear_session_vars, set_session_vars
from tools.registry import ToolRegistry
from xiaoban.trusted_runtime import (
    TrustedIdentity,
    activate_turn,
    begin_action,
    begin_turn,
    deactivate_turn,
    finish_action,
)
from xiaoban.trusted_runtime.tool_visibility import (
    filter_dynamic_evidence_api_kwargs,
)
from xiaoban.trusted_runtime.dynamic_completion import (
    dynamic_finalization_mode,
)
from run_agent import AIAgent


IDENTITY = TrustedIdentity(
    account_id="owner-protocol",
    data_scope="mystand",
    source="server_session",
)
PROTOCOL = "dynamic-evidence-v2"


def _binding() -> dict:
    return {
        "user_id": IDENTITY.account_id,
        "session_id": "session-protocol",
        "delivery_id": "delivery-protocol",
        "attempt": 1,
        "message_id": "message-protocol",
        "request_fingerprint": "a" * 64,
        "invocation_fingerprint": "b" * 64,
        "datascope_fingerprint": IDENTITY.datascope_fingerprint,
    }


def _dynamic_turn():
    return begin_turn(
        channel="web",
        user_message="读取目标资料",
        identity=IDENTITY,
        request_id="delivery-protocol",
        message_id="message-protocol",
        evidence_required=True,
        completion_protocol=PROTOCOL,
        completion_binding=_binding(),
    )


def _tools(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_names(payload: dict) -> list[str]:
    return [item["function"]["name"] for item in payload["tools"]]


def _record_found_index(turn) -> None:
    decision = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="index-found",
    )
    assert decision.decision == "allow"
    finish_action(
        turn,
        decision.call.call_id,
        "mystand_resource_index",
        "v1",
        {
            "schema": "mystand.resource-index.complete.v1",
            "ok": True,
            "items": [
                {
                    "resourceUid": "resource-protocol",
                    "safeLabel": "目标资料",
                    "canRead": True,
                }
            ],
            "hasMore": False,
            "nextCursor": "",
        },
    )
    assert turn.index_receipt is not None
    assert turn.index_receipt.status == "found"


def test_dynamic_provider_tools_are_request_local_and_index_first():
    turn = _dynamic_turn()
    original_tools = _tools(
        "mystand_resource_index",
        "mystand_query",
        "mystand_authorization",
        "terminal",
    )
    payload = {
        "model": "test",
        "tools": original_tools,
        "tool_choice": {
            "type": "function",
            "function": {"name": "mystand_query"},
        },
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    assert _tool_names(filtered) == ["mystand_resource_index"]
    assert "tool_choice" not in filtered
    assert _tool_names(payload) == [
        "mystand_resource_index",
        "mystand_query",
        "mystand_authorization",
        "terminal",
    ]
    assert payload["tools"] is original_tools


def test_dynamic_provider_tools_switch_to_reads_after_found_index():
    turn = _dynamic_turn()
    _record_found_index(turn)
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
            "terminal",
        ),
        "tool_choice": {
            "type": "function",
            "function": {"name": "mystand_resource_index"},
        },
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    assert _tool_names(filtered) == [
        "mystand_query",
        "mystand_authorization",
    ]
    assert "tool_choice" not in filtered


def test_dynamic_provider_tools_close_during_finalization():
    turn = _dynamic_turn()
    _record_found_index(turn)
    turn.completion_finalization = "failure"
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        ),
        "tool_choice": "required",
        "parallel_tool_calls": True,
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    assert filtered["tools"] == []
    assert "tool_choice" not in filtered
    assert "parallel_tool_calls" not in filtered


def test_signed_fact_turn_keeps_provider_tools_unchanged():
    signed = begin_turn(
        channel="web",
        user_message="读取签名资料",
        identity=IDENTITY,
        request_id="signed-request",
        message_id="signed-message",
        fact_requirement={"schema": "mystand.fact-requirement.v1"},
    )
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        ),
        "tool_choice": {
            "type": "function",
            "function": {"name": "mystand_query"},
        },
    }

    assert filter_dynamic_evidence_api_kwargs(payload, turn=signed) is payload


def test_dynamic_registry_discards_untrusted_module_hint_before_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_resource_index",
            "mystand_resource_index",
            {"name": "mystand_resource_index", "parameters": {}},
            lambda args: seen.append(dict(args))
            or json.dumps(
                {
                    "ok": True,
                    "items": [
                        {
                            "resourceUid": "resource-protocol",
                            "safeLabel": "目标资料",
                        }
                    ],
                    "hasMore": False,
                }
            ),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_resource_index",
                {
                    "operation": "list_resources",
                    "module_id": "model-guessed-module",
                    "moduleId": "another-model-guess",
                    "query": "目标资料",
                },
            )
        )

        assert result["ok"] is True
        assert seen == [
            {
                "operation": "list_resources",
                "query": "目标资料",
            }
        ]
        assert turn.action_calls[0].arguments == seen[0]

        closed = json.loads(
            registry.dispatch(
                "mystand_resource_index",
                {"query": "再次索引"},
                tool_call_id="index-after-found",
            )
        )
        assert closed == {
            "ok": False,
            "status": 403,
            "code": "dynamic_index_stage_closed",
        }
        assert len(seen) == 1
        assert dynamic_finalization_mode(turn) == ""
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_read_state_machine_blocks_write_before_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_authorization",
            "mystand_authorization",
            {"name": "mystand_authorization", "parameters": {}},
            lambda args: seen.append(dict(args)) or json.dumps({"ok": True}),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_authorization",
                {
                    "operation": "preview_write",
                    "authorization_id": "AUTH-test",
                },
                tool_call_id="dynamic-write",
            )
        )

        assert result == {
            "ok": False,
            "status": 403,
            "code": "write_isolated",
        }
        assert seen == []
        assert turn.action_calls == []
        assert dynamic_finalization_mode(
            turn,
            include_single_preaction=True,
        ) == ""
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_state_machine_blocks_every_hidden_tool_before_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[tuple[str, dict]] = []
        for tool_name in ("mystand_authorization_write", "terminal"):
            registry.register(
                tool_name,
                tool_name,
                {"name": tool_name, "parameters": {}},
                lambda args, name=tool_name: seen.append((name, dict(args)))
                or json.dumps({"ok": True}),
            )

        write_result = json.loads(
            registry.dispatch(
                "mystand_authorization_write",
                {"operation": "preview_write"},
                tool_call_id="hidden-write",
            )
        )
        terminal_result = json.loads(
            registry.dispatch(
                "terminal",
                {"command": "true"},
                tool_call_id="hidden-terminal",
            )
        )

        assert write_result["code"] == "write_isolated"
        assert terminal_result["code"] == "dynamic_tool_stage_closed"
        assert seen == []
        assert turn.action_calls == []
        executor_denial = json.loads(
            _trusted_preaction_denial(
                "memory",
                {"action": "add", "content": "must-not-write"},
                "hidden-inline-memory",
            )
        )
        assert executor_denial["code"] == "dynamic_tool_stage_closed"
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_hidden_tool_fails_closed_if_stage_helper_breaks(monkeypatch):
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "terminal",
            "terminal",
            {"name": "terminal", "parameters": {}},
            lambda args: seen.append(dict(args)) or json.dumps({"ok": True}),
        )

        def broken_stage(_turn):
            raise RuntimeError("stage helper unavailable")

        monkeypatch.setattr(
            "xiaoban.trusted_runtime.tool_visibility."
            "dynamic_evidence_allowed_tool_names",
            broken_stage,
        )
        result = json.loads(
            registry.dispatch(
                "terminal",
                {"command": "true"},
                tool_call_id="hidden-terminal-broken-stage",
            )
        )

        assert result["code"] == "preaction_error"
        assert seen == []
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_finalization_state_blocks_all_tool_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    _record_found_index(turn)
    turn.completion_finalization = "evidence"
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_query",
            "mystand_query",
            {"name": "mystand_query", "parameters": {}},
            lambda args: seen.append(dict(args)) or json.dumps({"ok": True}),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_query",
                {"operation": "read"},
                tool_call_id="read-after-finalize",
            )
        )

        assert result == {
            "ok": False,
            "status": 403,
            "code": "dynamic_finalization_stage_closed",
        }
        assert seen == []
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_signed_registry_keeps_module_hint_bound_and_dispatched():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="signed-message",
    )
    turn = begin_turn(
        channel="web",
        user_message="读取签名资料",
        identity=IDENTITY,
        request_id="signed-request",
        message_id="signed-message",
        fact_requirement={"schema": "mystand.fact-requirement.v1"},
    )
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_resource_index",
            "mystand_resource_index",
            {"name": "mystand_resource_index", "parameters": {}},
            lambda args: seen.append(dict(args))
            or json.dumps(
                {
                    "ok": True,
                    "items": [
                        {
                            "resourceUid": "resource-signed",
                            "moduleId": "signed-module",
                            "safeLabel": "签名资料",
                        }
                    ],
                    "hasMore": False,
                }
            ),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_resource_index",
                {
                    "operation": "list_resources",
                    "module_id": "signed-module",
                },
            )
        )

        assert result["ok"] is True
        assert seen[0]["module_id"] == "signed-module"
        assert turn.action_calls[0].arguments["module_id"] == "signed-module"
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_failure_finalize_view_excludes_raw_tool_trajectory():
    turn = _dynamic_turn()
    denied = begin_action(
        turn,
        "mystand_query",
        "v1",
        {"operation": "read"},
        call_id="preaction-denied",
    )
    assert denied.reason == "missing_index_receipt"
    decision = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="index-failed",
    )
    finish_action(
        turn,
        decision.call.call_id,
        "mystand_resource_index",
        "v1",
        {"ok": False, "status": 503, "code": "private_internal_code"},
    )
    messages = [
        {
            "role": "system",
            "content": "stable policy private_ephemeral_evidence",
        },
        {"role": "user", "content": "读取目标资料"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "mystand_resource_index"}}],
        },
        {
            "role": "tool",
            "content": '{"code":"private_internal_code"}',
        },
    ]
    agent = SimpleNamespace(
        _strict_no_automatic_paid_retry=True,
        max_iterations=3,
    )
    active = activate_turn(turn)
    try:
        selected = _prepare_finalize_only_call(
            agent,
            1,
            messages,
            original_user_message="读取目标资料",
        )
    finally:
        deactivate_turn(active)

    assert selected is True
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
    ]
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "private_internal_code" not in serialized
    assert "private_ephemeral_evidence" not in serialized
    assert "stable policy" not in serialized
    assert "mystand_resource_index" not in serialized
    assert "执行异常（1次）" in serialized
    assert "当前无权读取" not in serialized
    assert "自然中文" in messages[-1]["content"]


def test_raw_protocol_candidate_is_rejected_before_any_cleanup():
    double_bar_dsml = (
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"mystand_resource_index\">\n"
        "<｜｜DSML｜｜parameter name=\"query\">目标资料</｜｜DSML｜｜parameter>\n"
        "</｜｜DSML｜｜invoke>\n"
        "</｜｜DSML｜｜tool_calls>"
    )
    double_bar_dsml += "x" * (272 - len(double_bar_dsml))
    assert len(double_bar_dsml) == 272
    candidates = [
        '<|DSML|function_calls><|DSML|invoke name="mystand_query">',
        double_bar_dsml,
        '<tool_call>{"name":"mystand_query"}</tool_call>',
        '{"tool_calls":[{"function":{"name":"mystand_query"}}]}',
        '{"name":"mystand_query","arguments":{"operation":"read"}}',
        '{"type":"tool_use","name":"mystand_query","input":{}}',
        '{"functionCall":{"name":"mystand_query","args":{}}}',
        "assistant to=tools.mystand_query",
    ]
    assert all(_contains_raw_tool_protocol_content(item) for item in candidates)
    assert not _contains_raw_tool_protocol_content(
        "这次查询没有找到匹配资料，请补充更准确的名称。"
    )
    assert not _contains_raw_tool_protocol_content(
        '{"name":"中海城南一号","arguments":"客户补充说明"}'
    )

    persisted: list[list[dict]] = []
    agent = SimpleNamespace(
        _drop_trailing_empty_response_scaffolding=lambda _messages: None,
        _persist_session=lambda messages, _history: persisted.append(
            list(messages)
        ),
    )
    result = _reject_finalize_only_protocol_candidate(
        agent,
        [{"role": "user", "content": "读取资料"}],
        [],
        api_call_count=2,
        candidate="<think>ignore</think><tool_call>{}</tool_call>",
        finalize_only=True,
    )

    assert result is not None
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["final_response"] is None
    assert result["partial"] is True
    assert "protocol_content_rejected" in result["turn_exit_reason"]
    assert persisted


def test_agent_build_api_kwargs_always_applies_request_local_stage(monkeypatch):
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        )
    }
    monkeypatch.setattr(
        "agent.chat_completion_helpers.build_api_kwargs",
        lambda _agent, _messages: payload,
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        filtered = AIAgent._build_api_kwargs(
            object.__new__(AIAgent),
            [{"role": "user", "content": "读取资料"}],
        )
    finally:
        deactivate_turn(active)

    assert _tool_names(filtered) == ["mystand_resource_index"]
    assert _tool_names(payload) == [
        "mystand_resource_index",
        "mystand_query",
        "mystand_authorization",
    ]


def test_bedrock_filters_canonical_tools_before_wire_conversion():
    turn = _dynamic_turn()
    agent = SimpleNamespace(
        tools=_tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
            "terminal",
        ),
        _ephemeral_tool_choice="",
        api_mode="bedrock_converse",
        _get_transport=lambda: BedrockTransport(),
        _bedrock_region="us-east-1",
        _bedrock_guardrail_config=None,
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=512,
    )
    active = activate_turn(turn)
    try:
        discover = build_api_kwargs(
            agent,
            [{"role": "user", "content": "读取资料"}],
        )
        discover_names = [
            item["toolSpec"]["name"]
            for item in discover["toolConfig"]["tools"]
        ]
        assert discover_names == ["mystand_resource_index"]

        _record_found_index(turn)
        read = build_api_kwargs(
            agent,
            [{"role": "user", "content": "读取资料"}],
        )
        read_names = [
            item["toolSpec"]["name"]
            for item in read["toolConfig"]["tools"]
        ]
        assert read_names == [
            "mystand_query",
            "mystand_authorization",
        ]

        turn.completion_finalization = "failure"
        finalized = build_api_kwargs(
            agent,
            [{"role": "user", "content": "读取资料"}],
        )
        assert "toolConfig" not in finalized
    finally:
        deactivate_turn(active)
