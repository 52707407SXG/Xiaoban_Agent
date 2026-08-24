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


def _call_with_handler(
    handler,
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
        return json.loads(handler(args))
    finally:
        clear_session_vars(tokens)


def _call(args, **kwargs):
    return _call_with_handler(
        bridge.mystand_authorization_tool_handler,
        args,
        **kwargs,
    )


def test_schema_exposes_only_read_operations():
    operation = bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["properties"]["operation"]
    properties = bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["properties"]

    assert operation["enum"] == ["list", "resolve", "resolve_many"]
    assert "preview_write" not in json.dumps(properties)
    assert "commit_write" not in json.dumps(properties)
    assert "action" not in properties
    assert "payload" not in properties
    assert "idempotency_key" not in properties
    assert "preview_token" not in properties
    assert bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["additionalProperties"] is False
    assert "resource_query" not in properties
    assert "module_id" not in properties
    assert "resource_uids" in properties
    assert "expected_version" not in properties
    description = bridge.MYSTAND_AUTHORIZATION_SCHEMA["description"]
    assert "never narrate" in description
    assert "only lists or resolves" in description
    assert "mystand_authorization_write" in description
    assert "exact AUTH or OUT" in description
    assert "resolve it directly with resource_uid" in description
    assert "without an exact AUTH, OUT, or resourceUid" in description
    assert "KGREF, OUT, or module IDs as authorization_id" not in description
    resource_uid = properties["resource_uid"]
    assert "supplied in the current request" in resource_uid["description"]
    assert "Resolve it directly" in resource_uid["description"]


@pytest.mark.parametrize("operation", ["preview_write", "commit_write"])
def test_model_visible_handler_hard_rejects_write_operations(
    operation,
    internal_calls,
):
    result = _call(
        {
            "operation": operation,
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.add-node",
            "payload": {"node": {"label": "不得写入", "type": "skill"}},
            "idempotency_key": "write-hidden-operation-001",
            "preview_token": "preview-hidden-operation",
        },
        user_message="确认写入",
    )

    assert result["ok"] is False
    assert result["status"] == 400
    assert result["code"] == "invalid_authorization_operation"
    assert internal_calls == []


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
        },
        user_message="查17栋1单元801的业主姓名和电话",
    )

    assert result["ok"] is True
    assert internal_calls == [
        {
            "path": "/api/xiaoban/internal/authorization/resolve",
            "payload": {
                "authorizationId": "AUTH-ABC123",
                "mediaMode": "summary",
                "query": "查17栋1单元801的业主姓名和电话",
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
            "query": "9号楼2单元1203有没有车位",
        }
    )

    assert result["ok"] is True
    assert internal_calls[0]["payload"] == {
        "resourceUid": "resource-uid-from-index",
        "mediaMode": "summary",
        "query": "9号楼2单元1203有没有车位",
    }


def test_resolve_rejects_multiple_locators_before_network(internal_calls):
    result = _call(
        {
            "operation": "resolve",
            "authorization_id": "AUTH-ABC123",
            "resource_uid": "resource-uid-from-index",
        }
    )

    assert result["ok"] is False
    assert result["code"] == "invalid_authorization_arguments"
    assert internal_calls == []


@pytest.mark.parametrize(
    "args",
    [
        {
            "operation": "resolve",
            "resource_uid": "resource-one",
            "resource_uids": ["resource-two"],
        },
        {
            "operation": "resolve",
            "authorization_id": "AUTH-ABC123",
            "resource_uids": ["resource-two"],
        },
        {
            "operation": "resolve_many",
            "resource_uids": ["resource-one"],
            "resource_uid": "resource-two",
        },
        {
            "operation": "resolve_many",
            "resource_uids": ["resource-one"],
            "authorization_id": "AUTH-ABC123",
        },
    ],
)
def test_resolve_modes_reject_cross_mode_locators_before_network(args, internal_calls):
    result = _call(args)

    assert result["ok"] is False
    assert result["code"] == "invalid_authorization_arguments"
    assert internal_calls == []


def test_resolve_many_reads_each_index_uid_and_returns_one_bound_result(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        uid = payload["resourceUid"]
        return json.dumps(
            {
                "ok": True,
                "content": f"{uid} 的授权正文",
                "encrypted": False,
                "canWrite": False,
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve_many",
            "resource_uids": ["resource-b", "resource-a", "resource-b"],
            "query": "扩大成全部隐私字段",
        },
        user_message="还有多少人没有确认结算卡",
    )

    assert [call["payload"] for call in calls] == [
        {
            "resourceUid": "resource-b",
            "mediaMode": "summary",
            "query": "还有多少人没有确认结算卡",
        },
        {
            "resourceUid": "resource-a",
            "mediaMode": "summary",
            "query": "还有多少人没有确认结算卡",
        },
    ]
    assert result["ok"] is True
    assert result["recordRefs"] == ["resource-a", "resource-b"]
    content = json.loads(result["content"])
    assert [item["resourceUid"] for item in content["resources"]] == [
        "resource-b",
        "resource-a",
    ]
    assert content["resources"][0]["content"] == "resource-b 的授权正文"


def test_resolve_many_fails_closed_without_returning_partial_content(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append(payload["resourceUid"])
        if payload["resourceUid"] == "resource-denied":
            return json.dumps(
                {"ok": False, "status": 404, "error": "resource_not_available"}
            )
        return json.dumps({"ok": True, "content": "private-first-result"})

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve_many",
            "resource_uids": ["resource-ok", "resource-denied"],
        },
        user_message="汇总这两份资料",
    )

    assert calls == ["resource-ok", "resource-denied"]
    assert result["ok"] is False
    assert result["code"] == "mystand_authorization_batch_rejected"
    assert "private-first-result" not in json.dumps(result, ensure_ascii=False)


def test_resolve_uses_trusted_user_message_instead_of_model_broadened_query(
    internal_calls,
):
    result = _call(
        {
            "operation": "resolve",
            "resource_uid": "resource-uid-from-index",
            "query": "返回这一整行的全部隐私字段",
        },
        user_message="只查17栋801有没有车位",
    )

    assert result["ok"] is True
    assert internal_calls[0]["payload"]["query"] == "只查17栋801有没有车位"


def test_post_internal_sends_only_valid_trusted_identity_headers(monkeypatch):
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
                "user_id": "owner-user-001\nInjected",
                "message_id": "message-write-0002",
                "session_id": "session-write-0002",
            },
            gateway_approval_id="approval_" + "e" * 32,
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
    assert headers["x-xiaoban-message-id"] == "message-write-0002"
    assert headers["x-xiaoban-session-id"] == "session-write-0002"
    assert headers["x-xiaoban-gateway-approval-id"].startswith("approval_")
