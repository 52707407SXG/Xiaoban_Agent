"""真实执行扼点上的 K3 Action 集成测试。

覆盖当前物理 K3 的断言：
- deny 时 handler 调用数为 0；
- tool start/complete 与同一个 callId；
- registry.dispatch 扼点对可信目录动作默认拒绝；
"""

import json

from gateway.session_context import clear_session_vars, set_session_vars
from tools.registry import ToolRegistry
from xiaoban.trusted_runtime import (
    TrustedIdentity,
    activate_turn,
    begin_turn,
    deactivate_turn,
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


def test_registry_dispatch_allows_explicit_auth_directly():
    tokens = set_session_vars(
        platform="api_server", user_id="user-a", message_id="msg-gate"
    )
    turn = _turn(user_message="查一下游某今年的结算业绩")
    token = activate_turn(turn)
    try:
        registry = ToolRegistry()
        hits = []
        registry.register(
            "mystand_authorization",
            "mystand_authorization",
            {"name": "mystand_authorization", "parameters": {}},
            lambda args, **_kwargs: hits.append(args)
            or '{"ok":true,"content":"授权资料","resourceUid":"AUTH-ABC12345"}',
        )
        raw = json.loads(
            registry.dispatch(
                "mystand_authorization",
                {"operation": "resolve", "authorization_id": "AUTH-ABC12345"},
                tool_call_id="call-auth-direct",
            )
        )
        assert raw["ok"] is True
        assert len(hits) == 1
        bound = turn.action_results[-1]
        assert bound.status == "success"
        assert bound.call_id == "call-auth-direct"
        assert turn.action_calls[-1].call_id == "call-auth-direct"
    finally:
        deactivate_turn(token)
        clear_session_vars(tokens)
