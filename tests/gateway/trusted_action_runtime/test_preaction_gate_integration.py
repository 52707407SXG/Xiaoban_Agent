"""阻断修复 R1-R3 GREEN：真实执行扼点上的 PreAction/PostAction 集成测试。

覆盖修复单 9.3 必须单列的断言：
- deny 时 handler 调用数为 0；
- tool start/complete 与同一个 callId；
- 开放查询无 IndexReceipt 时 handler 调用数为 0；
- registry.dispatch 扼点对可信目录动作默认拒绝；
- Evidence 只含动作合同允许的字段路径；
- tool start 后没有 complete/failed 不得结束为成功；
- My Stand 请求的 SSE 业务 delta 在 Guard 前必须缓冲（既有安全行为锁定）。
"""

import json

from gateway.platforms.api_server import (
    _run_mystand_preexecuted_evidence,
    _should_buffer_stream_deltas,
)
from gateway.session_context import clear_session_vars, set_session_vars
from tools.registry import ToolRegistry
from xiaoban.trusted_runtime import (
    TrustedIdentity,
    activate_turn,
    begin_action,
    begin_turn,
    check_completion,
    deactivate_turn,
    finish_action,
)

IDENTITY = TrustedIdentity(
    account_id="user-a", data_scope="mystand", source="server_session"
)


def _turn(user_message="读取 AUTH-ABC12345", identity=IDENTITY):
    return begin_turn(
        channel="web",
        user_message=user_message,
        identity=identity,
        request_id="req-gate",
        message_id="msg-gate",
    )


def test_preexecuted_allow_path_binds_real_call_id(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tools.mystand_authorization_tool.mystand_authorization_tool_handler",
        lambda args: calls.append(args) or '{"ok":true,"content":"地址3401号"}',
    )
    starts, completes = [], []
    turn = _turn()
    evidence = _run_mystand_preexecuted_evidence(
        "mystand_authorization",
        user_message="读取 AUTH-ABC12345",
        system_prompt="",
        tool_start_callback=lambda cid, name, args: starts.append(cid),
        tool_complete_callback=lambda cid, name, args, content: completes.append(cid),
        trusted_turn=turn,
    )
    assert len(calls) == 1
    assert starts == completes == [turn.action_calls[0].call_id]
    assert evidence[0]["call_id"] == turn.action_calls[0].call_id
    assert turn.action_results[0].status == "success"
    assert turn.index_receipt is not None and turn.index_receipt.status == "found"
    assert turn.evidence and "verifying" in turn.states


def test_preexecuted_deny_means_zero_handler_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tools.mystand_authorization_tool.mystand_authorization_tool_handler",
        lambda args: calls.append(args) or '{"ok":true}',
    )
    turn = _turn(identity=None)  # 关键身份字段缺失：fail closed
    evidence = _run_mystand_preexecuted_evidence(
        "mystand_authorization",
        user_message="读取 AUTH-ABC12345",
        system_prompt="",
        tool_start_callback=lambda *a: None,
        tool_complete_callback=lambda *a: None,
        trusted_turn=turn,
    )
    assert calls == [], "PreAction deny 时 handler 必须零调用"
    assert json.loads(evidence[0]["content"])["code"] == "missing_identity"
    assert turn.pre_action_denials == 1
    assert turn.evidence == []


def test_open_read_without_index_receipt_is_denied_before_execution():
    turn = _turn(user_message="查一下游某今年的结算业绩")
    decision = begin_action(turn, "mystand_query", "v1", {"operation": "read"})
    assert decision.decision == "deny"
    assert decision.reason == "missing_index_receipt"
    assert turn.action_calls == [], "deny 不得登记为已允许调用"


def test_registry_dispatch_gate_denies_without_active_turn():
    tokens = set_session_vars(
        platform="api_server", user_id="user-a", message_id="msg-none"
    )
    try:
        registry = ToolRegistry()
        hits = []
        registry.register(
            "mystand_query",
            "mystand_query",
            {"name": "mystand_query", "parameters": {}},
            lambda args: hits.append(args) or '{"ok":true,"content":"x"}',
        )
        raw = json.loads(registry.dispatch("mystand_query", {"operation": "read"}))
        assert raw["ok"] is False and raw["code"] == "no_active_turn"
        assert hits == []
    finally:
        clear_session_vars(tokens)


def test_registry_dispatch_gate_allows_after_real_index_receipt():
    tokens = set_session_vars(
        platform="api_server", user_id="user-a", message_id="msg-gate"
    )
    turn = _turn(user_message="查一下游某今年的结算业绩")
    token = activate_turn(turn)
    try:
        registry = ToolRegistry()
        hits = []
        registry.register(
            "mystand_query",
            "mystand_query",
            {"name": "mystand_query", "parameters": {}},
            lambda args: hits.append(args) or '{"ok":true,"content":"业绩 100 元"}',
        )
        # 无 IndexReceipt：默认拒绝，handler 零调用。
        raw = json.loads(registry.dispatch("mystand_query", {"operation": "read"}))
        assert raw["code"] == "missing_index_receipt"
        assert hits == []
        # 真实定向读取建立回执后：允许执行并严格绑定。
        allow = begin_action(
            turn, "mystand_authorization", "v1",
            {"operation": "resolve", "resource_uid": "res-demo-1"},
        )
        finish_action(
            turn, allow.call.call_id, "mystand_authorization", "v1",
            '{"ok":true,"content":"档案存在"}',
        )
        raw = json.loads(registry.dispatch("mystand_query", {"operation": "read"}))
        assert raw["ok"] is True
        assert len(hits) == 1
        bound = turn.action_results[-1]
        assert bound.status == "success"
        assert bound.call_id == turn.action_calls[-1].call_id
    finally:
        deactivate_turn(token)
        clear_session_vars(tokens)


def test_evidence_contains_only_contract_field_paths():
    turn = _turn()
    allow = begin_action(
        turn, "mystand_authorization", "v1",
        {"operation": "resolve", "resource_uid": "res-demo-1"},
    )
    finish_action(
        turn, allow.call.call_id, "mystand_authorization", "v1",
        '{"ok":true,"content":"地址3401号","internalNote":"不可出站字段",'
        '"debugTrace":"stack"}',
    )
    assert len(turn.evidence) == 1
    facts = json.loads(turn.evidence[0].allowed_facts)
    assert set(facts.keys()) == {"content"}
    assert "不可出站字段" not in turn.evidence[0].allowed_facts
    assert "stack" not in turn.evidence[0].allowed_facts


def test_tool_start_without_complete_cannot_end_as_success():
    turn = _turn()
    allow = begin_action(
        turn, "mystand_authorization", "v1",
        {"operation": "resolve", "resource_uid": "res-demo-1"},
    )
    assert allow.decision == "allow"
    decision = check_completion("查到了，地址是3401号。", turn)
    assert not decision.allowed
    assert decision.reason == "blocked_no_action_result"
    assert "3401" not in decision.text


def test_mystand_stream_deltas_are_buffered_until_guard_passes():
    # 既有安全行为锁定：My Stand 请求的业务 delta 在 Guard 前不得外发。
    assert _should_buffer_stream_deltas("查一下业主", mystand_request=True) is True
