"""Tests for Xiaoban's server-enforced My Stand AUTH/OUT bridge."""

import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools import mystand_authorization_tool as bridge


@pytest.fixture
def internal_calls(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "session": session,
                "explicit_confirmation": explicit_confirmation,
            }
        )
        return json.dumps(
            {"ok": True, "path": path, "received": payload},
            ensure_ascii=False,
        )

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    return calls


def _call(
    args,
    *,
    platform="api_server",
    user_id="ZYJ005",
    message_id="msg-001",
    session_id="session-001",
    user_message="",
):
    tokens = set_session_vars(
        platform=platform,
        user_id=user_id,
        message_id=message_id,
        session_id=session_id,
        user_message=user_message,
    )
    try:
        return json.loads(bridge.mystand_authorization_tool_handler(args))
    finally:
        clear_session_vars(tokens)


def test_schema_exposes_only_fixed_operations_and_write_actions():
    operation = bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["properties"]["operation"]
    action = bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["properties"]["action"]

    assert operation["enum"] == ["list", "resolve", "preview_write", "commit_write"]
    assert set(action["enum"]) == {
        "note.append-content",
        "property-note.append-text-block",
        "profile-card.update-field",
        "knowledge-graph.add-node",
        "knowledge-graph.update-node",
        "knowledge-graph.add-edge",
    }
    assert bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://127.0.0.1:18081", "http://127.0.0.1:18081"),
        ("http://localhost:18081/", "http://localhost:18081"),
        ("http://[::1]:18081", "http://[::1]:18081"),
        ("https://127.0.0.1:18081", ""),
        ("http://127.0.0.1.evil.example:18081", ""),
        ("http://example.com:18081", ""),
        ("http://user:pass@127.0.0.1:18081", ""),
        ("http://127.0.0.1:18081?redirect=example.com", ""),
    ],
)
def test_api_base_url_is_loopback_http_only(monkeypatch, value, expected):
    monkeypatch.setenv("MYSTAND_XIAOBAN_MYSTAND_API_URL", value)

    assert bridge._api_base_url() == expected


def test_rejects_non_api_or_anonymous_sessions(internal_calls):
    non_api = _call({"operation": "list"}, platform="telegram")
    anonymous = _call({"operation": "list"}, user_id="")

    assert non_api["code"] == "mystand_session_required"
    assert anonymous["code"] == "mystand_session_required"
    assert internal_calls == []


def test_list_uses_only_current_session_identity_and_filters(internal_calls):
    result = _call(
        {
            "operation": "list",
            "query": "知识图谱",
            "source_type": "knowledge-graph",
            "permission": "write",
            "id_type": "internal",
            # Direct handler calls bypass JSON-schema validation. This value
            # must still never become the trusted current user.
            "userId": "ZYJ999",
        }
    )

    assert result["ok"] is True
    assert internal_calls == [
        {
            "path": "/api/xiaoban/internal/authorization/list",
            "payload": {
                "q": "知识图谱",
                "sourceType": "knowledge-graph",
                "permission": "write",
                "idType": "internal",
            },
            "session": {
                "platform": "api_server",
                "user_id": "ZYJ005",
                "message_id": "msg-001",
                "session_id": "session-001",
            },
            "explicit_confirmation": False,
        }
    ]


def test_resolve_passes_auth_id_and_defaults_to_media_summary(internal_calls):
    result = _call(
        {
            "operation": "resolve",
            "authorization_id": "AUTH-ABC123",
        }
    )

    assert result["ok"] is True
    assert internal_calls == [
        {
            "path": "/api/xiaoban/internal/authorization/resolve",
            "payload": {
                "authorizationId": "AUTH-ABC123",
                "mediaMode": "summary",
            },
            "session": {
                "platform": "api_server",
                "user_id": "ZYJ005",
                "message_id": "msg-001",
                "session_id": "session-001",
            },
            "explicit_confirmation": False,
        }
    ]

def test_resolve_passes_resource_uid_from_index_without_guessing_auth(internal_calls):
    result = _call(
        {
            "operation": "resolve",
            "resource_uid": "resource-uid-from-index",
        }
    )

    assert result["ok"] is True
    assert internal_calls[0]["payload"] == {
        "resourceUid": "resource-uid-from-index",
        "mediaMode": "summary",
    }


@pytest.mark.parametrize(
    ("message_id", "session_id"),
    [
        ("", "session-001"),
        ("msg-001", ""),
    ],
)
def test_preview_requires_trusted_write_context(
    internal_calls, message_id, session_id
):
    result = _call(
        {
            "operation": "preview_write",
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.add-node",
            "payload": {"label": "客户需求"},
            "expected_version": "v3",
            "idempotency_key": "write-001",
        },
        message_id=message_id,
        session_id=session_id,
    )

    assert result["code"] == "trusted_write_context_required"
    assert internal_calls == []


def test_preview_passes_fixed_action_payload_version_and_idempotency(internal_calls):
    payload = {"label": "客户需求", "kind": "topic"}
    result = _call(
        {
            "operation": "preview_write",
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.add-node",
            "payload": payload,
            "expected_version": "v3",
            "idempotency_key": "write-001",
        }
    )

    assert result["ok"] is True
    assert internal_calls == [
        {
            "path": "/api/xiaoban/internal/authorization/write/preview",
            "payload": {
                "authorizationId": "AUTH-ABC123",
                "action": "knowledge-graph.add-node",
                "payload": payload,
                "expectedVersion": "v3",
                "idempotencyKey": "write-001",
            },
            "session": {
                "platform": "api_server",
                "user_id": "ZYJ005",
                "message_id": "msg-001",
                "session_id": "session-001",
            },
            "explicit_confirmation": False,
        }
    ]


def test_preview_rejects_unknown_write_action_before_transport(internal_calls):
    result = _call(
        {
            "operation": "preview_write",
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.delete-node",
            "payload": {"nodeId": "node-1"},
            "expected_version": "v3",
            "idempotency_key": "write-001",
        }
    )

    assert result["code"] == "authorization_write_action_not_allowed"
    assert internal_calls == []


@pytest.mark.parametrize(
    "user_message",
    [
        "",
        "先别写",
        "不要确认写入",
        "我还没确认写入",
        "我没有确认写入",
        "我尚未确认写入",
        "确认写入吗？",
        "这不是确认写入",
        "这不算确认写入",
        "我说“确认写入”是什么意思？",
        "确认写入是不是就会立即改资料？",
        "请解释确认写入",
        "按钮文案：确认写入，分析安全问题",
        "如果我说确认写入，你就会修改吗？",
        "引用原话“确认写入”",
        "确认写入，然后把安全问题也分析一下",
    ],
)
def test_commit_rejects_missing_negated_or_question_confirmation(
    internal_calls, user_message
):
    result = _call(
        {
            "operation": "commit_write",
            "preview_token": "preview-token",
            "idempotency_key": "write-001",
            # The model cannot smuggle confirmation through its tool arguments.
            "confirmationPhrase": "确认写入",
            "user_message": "确认写入",
        },
        user_message=user_message,
    )

    assert result["code"] == "explicit_user_confirmation_required"
    assert internal_calls == []


def test_commit_uses_actual_user_confirmation_and_trusted_session_ids(internal_calls):
    result = _call(
        {
            "operation": "commit_write",
            "preview_token": "preview-token",
            "idempotency_key": "write-001",
            "confirmationPhrase": "模型伪造的其他文字",
        },
        message_id="msg-confirm-002",
        session_id="session-001",
        user_message="预览没问题，确认写入",
    )

    assert result["ok"] is True
    assert internal_calls == [
        {
            "path": "/api/xiaoban/internal/authorization/write/commit",
            "payload": {
                "previewToken": "preview-token",
                "idempotencyKey": "write-001",
                "confirmationPhrase": "确认写入",
            },
            "session": {
                "platform": "api_server",
                "user_id": "ZYJ005",
                "message_id": "msg-confirm-002",
                "session_id": "session-001",
            },
            "explicit_confirmation": True,
        }
    ]


def test_post_internal_sends_only_valid_trusted_identity_and_confirmation_headers(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "service-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(
        bridge._post_internal(
            "/api/xiaoban/internal/authorization/write/commit",
            {"previewToken": "preview-token"},
            session={
                "user_id": "ZYJ005\nInjected",
                "message_id": "msg-002",
                "session_id": "session-001",
            },
            explicit_confirmation=True,
        )
    )

    headers = {
        key.lower(): value
        for key, value in captured["request"].header_items()
    }
    assert result == {"ok": True}
    assert captured["timeout"] == 20
    assert headers["authorization"] == "Bearer service-token"
    assert "x-xiaoban-user-id" not in headers
    assert headers["x-xiaoban-message-id"] == "msg-002"
    assert headers["x-xiaoban-session-id"] == "session-001"
    assert headers["x-xiaoban-explicit-confirmation"] == "1"
