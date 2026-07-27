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
    payload = bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["properties"]["payload"]
    payload_properties = payload["properties"]

    assert operation["enum"] == ["list", "resolve", "preview_write", "commit_write"]
    assert set(action["enum"]) == {
        "note.append-content",
        "property-note.append-text-block",
        "profile-card.update-field",
        "knowledge-graph.add-node",
        "knowledge-graph.update-node",
        "knowledge-graph.add-edge",
        "finance-archive.update-row-fields",
    }
    assert bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["additionalProperties"] is False
    assert "resource_query" in bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["properties"]
    assert "expected_version" not in bridge.MYSTAND_AUTHORIZATION_SCHEMA["parameters"]["properties"]
    assert "never narrate" in bridge.MYSTAND_AUTHORIZATION_SCHEMA["description"]
    assert payload["additionalProperties"] is False
    assert {"node", "nodeId", "label", "type", "changes", "edge"} <= set(
        payload_properties
    )
    assert "graphId" not in payload_properties
    assert "Never include graphId" in payload["description"]


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


def test_resolve_resource_query_silently_locates_and_reads_one_resource(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload, "session": session})
        if path.endswith("/resource-index"):
            return json.dumps({
                "schema": "mystand.resource-index.page.v1",
                "ok": True,
                "items": [{
                    "resourceUid": "resource-finance-island",
                    "safeLabel": "复地金融岛楼盘MD",
                    "canRead": True,
                }],
                "nextCursor": "",
                "hasMore": False,
            }, ensure_ascii=False)
        return json.dumps({"ok": True, "received": payload}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "复地金融岛",
            "module_id": "property-dev",
            "query": "17栋1单元801的姓名和电话",
        },
        user_message="查复地金融岛17栋1单元801的姓名和电话",
    )

    assert result["ok"] is True
    assert [call["path"] for call in calls] == [
        "/api/xiaoban/internal/resource-index",
        "/api/xiaoban/internal/authorization/resolve",
    ]
    assert calls[0]["payload"]["query"] == "复地金融岛"
    assert calls[1]["payload"] == {
        "resourceUid": "resource-finance-island",
        "mediaMode": "summary",
        "query": "查复地金融岛17栋1单元801的姓名和电话",
    }


def test_resolve_resource_query_must_come_from_trusted_user_message(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "另一座楼盘",
            "query": "1栋101的电话",
        },
        user_message="查泰悦湾1栋101的电话",
    )

    assert result["code"] == "resource_query_not_in_user_message"
    assert calls == []


@pytest.mark.parametrize(
    "safe_label",
    ["泰悦湾旧资料", "泰悦湾二期", "泰-悦湾楼盘MD", "泰 悦 湾楼盘MD"],
)
def test_resolve_resource_query_rejects_selected_title_not_named_by_user(
    monkeypatch,
    safe_label,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        if path.endswith("/resource-index"):
            return json.dumps({
                "ok": True,
                "items": [{
                    "resourceUid": "resource-other-edition",
                    "safeLabel": safe_label,
                    "canRead": True,
                }],
                "hasMore": False,
            }, ensure_ascii=False)
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾",
            "module_id": "property-dev",
        },
        user_message="查泰悦湾1栋702的电话",
    )

    assert result["code"] == "resource_query_ambiguous"
    assert [call["path"] for call in calls] == [
        "/api/xiaoban/internal/resource-index",
    ]


@pytest.mark.parametrize(
    "user_message",
    [
        "帮我看看泰悦湾1栋702的电话",
        "查泰悦湾MD一栋702的电话",
        "楼盘MD泰悦湾一栋702的电话",
    ],
)
def test_resolve_resource_query_accepts_normal_conversational_title_boundaries(
    monkeypatch,
    user_message,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        if path.endswith("/resource-index"):
            return json.dumps({
                "ok": True,
                "items": [{
                    "resourceUid": "resource-taiyuewan",
                    "safeLabel": "泰悦湾楼盘MD",
                    "canRead": True,
                }],
                "hasMore": False,
            }, ensure_ascii=False)
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾",
            "module_id": "property-dev",
        },
        user_message=user_message,
    )

    assert result["ok"] is True
    assert [call["path"] for call in calls] == [
        "/api/xiaoban/internal/resource-index",
        "/api/xiaoban/internal/authorization/resolve",
    ]


@pytest.mark.parametrize(
    "user_message",
    [
        "查泰悦湾MD二期1栋702的电话",
        "查泰悦湾MD2期1栋702的电话",
        "查泰悦湾MD2025版1栋702的电话",
        "查泰悦湾MD二号资料1栋702的电话",
    ],
)
def test_resolve_resource_query_rejects_model_dropping_md_version_suffix(
    monkeypatch,
    user_message,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾",
            "module_id": "property-dev",
        },
        user_message=user_message,
    )

    assert result["code"] == "resource_query_not_in_user_message"
    assert calls == []


@pytest.mark.parametrize(
    "user_message",
    [
        "查泰悦湾中介资料的内容",
        "查泰悦湾里程碑版的内容",
        "查泰悦湾MD中介版的内容",
        "查泰悦湾MD里程碑版的内容",
    ],
)
def test_resolve_resource_query_rejects_model_dropping_chinese_title_suffix(
    monkeypatch,
    user_message,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾",
            "module_id": "property-dev",
        },
        user_message=user_message,
    )

    assert result["code"] == "resource_query_not_in_user_message"
    assert calls == []


@pytest.mark.parametrize(
    ("user_message", "long_title", "long_uid"),
    [
        ("查泰悦湾里房源的电话", "泰悦湾里楼盘MD", "resource-inside"),
        ("查泰悦湾中房源的电话", "泰悦湾中楼盘MD", "resource-middle"),
    ],
)
def test_resolve_resource_query_prefers_longest_title_named_by_user(
    monkeypatch,
    user_message,
    long_title,
    long_uid,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        if path.endswith("/resource-index"):
            return json.dumps({
                "ok": True,
                "items": [
                    {
                        "resourceUid": "resource-base",
                        "safeLabel": "泰悦湾楼盘MD",
                        "canRead": True,
                    },
                    {
                        "resourceUid": long_uid,
                        "safeLabel": long_title,
                        "canRead": True,
                    },
                ],
                "hasMore": False,
            }, ensure_ascii=False)
        return json.dumps({"ok": True, "received": payload}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾",
            "module_id": "property-dev",
        },
        user_message=user_message,
    )

    assert result["ok"] is True
    assert calls[-1]["path"] == "/api/xiaoban/internal/authorization/resolve"
    assert calls[-1]["payload"]["resourceUid"] == long_uid


@pytest.mark.parametrize(
    "user_message",
    [
        "不要查泰悦湾里，查泰悦湾的电话",
        "先查泰悦湾里，后来改查泰悦湾的电话",
    ],
)
def test_resolve_resource_query_honors_latest_affirmative_resource_title(
    monkeypatch,
    user_message,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        if path.endswith("/resource-index"):
            return json.dumps({
                "ok": True,
                "items": [
                    {
                        "resourceUid": "resource-base",
                        "safeLabel": "泰悦湾楼盘MD",
                        "canRead": True,
                    },
                    {
                        "resourceUid": "resource-inside",
                        "safeLabel": "泰悦湾里楼盘MD",
                        "canRead": True,
                    },
                ],
                "hasMore": False,
            }, ensure_ascii=False)
        return json.dumps({"ok": True, "received": payload}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾",
            "module_id": "property-dev",
        },
        user_message=user_message,
    )

    assert result["ok"] is True
    assert calls[-1]["path"] == "/api/xiaoban/internal/authorization/resolve"
    assert calls[-1]["payload"]["resourceUid"] == "resource-base"


def test_resolve_resource_query_rejects_title_only_named_negatively(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾里",
            "module_id": "property-dev",
        },
        user_message="不要查泰悦湾里，查泰悦湾的电话",
    )

    assert result["code"] == "resource_query_not_in_user_message"
    assert calls == []


@pytest.mark.parametrize(
    "user_message",
    [
        "查泰悦湾的电话，也查泰悦湾里，不过最后那个不要查了",
        "查泰悦湾的电话，然后查泰悦湾里，但后一个别查了",
        "先查泰悦湾，再查泰悦湾里，最后这份排除",
        "查泰悦湾的电话，也查泰悦湾里，不过最后那一个不要查了",
        "查泰悦湾的电话，也查泰悦湾里，不过刚才那个不要查了",
        "查泰悦湾的电话，也查泰悦湾里，不过后者不要查了",
        "查泰悦湾的电话，也查泰悦湾里，不过这个不用查了",
    ],
)
def test_resolve_resource_query_fails_closed_on_deictic_resource_cancellation(
    monkeypatch,
    user_message,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "泰悦湾",
            "module_id": "property-dev",
        },
        user_message=user_message,
    )

    assert result["code"] == "resource_query_not_in_user_message"
    assert calls == []


def test_resolve_resource_query_follows_cursor_until_exact_title(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        if path.endswith("/resource-index") and payload["cursor"] == "":
            return json.dumps({
                "ok": True,
                "items": [{
                    "resourceUid": "resource-similar",
                    "safeLabel": "复地金融岛旧资料",
                    "canRead": True,
                }],
                "nextCursor": "cursor-page-2",
                "hasMore": True,
            }, ensure_ascii=False)
        if path.endswith("/resource-index"):
            return json.dumps({
                "ok": True,
                "items": [{
                    "resourceUid": "resource-exact",
                    "safeLabel": "复地金融岛楼盘MD",
                    "canRead": True,
                }],
                "nextCursor": "",
                "hasMore": False,
            }, ensure_ascii=False)
        return json.dumps({"ok": True, "received": payload}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "复地金融岛",
            "query": "17栋801的电话",
        },
        user_message="查复地金融岛17栋801的电话",
    )

    assert result["ok"] is True
    assert [call["path"] for call in calls] == [
        "/api/xiaoban/internal/resource-index",
        "/api/xiaoban/internal/resource-index",
        "/api/xiaoban/internal/authorization/resolve",
    ]
    assert calls[1]["payload"]["cursor"] == "cursor-page-2"
    assert calls[2]["payload"]["resourceUid"] == "resource-exact"


def test_resolve_resource_query_preserves_internal_title_spacing(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        if path.endswith("/resource-index"):
            return json.dumps({
                "ok": True,
                "items": [{
                    "resourceUid": "resource-spaced-title",
                    "safeLabel": "楼盘资料 00002",
                    "canRead": True,
                }],
                "hasMore": False,
            }, ensure_ascii=False)
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "楼盘资料 00002",
            "query": "1栋101的电话",
        },
        user_message="查楼盘资料 00002 的1栋101电话",
    )

    assert result["ok"] is True
    assert calls[0]["payload"]["query"] == "楼盘资料 00002"
    assert calls[1]["payload"]["resourceUid"] == "resource-spaced-title"


def test_resolve_resource_query_stops_on_ambiguous_safe_titles(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append(path)
        return json.dumps({
            "ok": True,
            "items": [
                {"resourceUid": "resource-1", "safeLabel": "同名楼盘 A", "canRead": True},
                {"resourceUid": "resource-2", "safeLabel": "同名楼盘 B", "canRead": True},
            ],
            "hasMore": False,
        }, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "同名楼盘",
            "query": "1栋101",
        },
        user_message="查同名楼盘1栋101",
    )

    assert result["code"] == "resource_query_ambiguous"
    assert calls == ["/api/xiaoban/internal/resource-index"]


def test_resolve_resource_query_rejects_unique_hidden_metadata_hit(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append(path)
        return json.dumps({
            "ok": True,
            "items": [{
                "resourceUid": "resource-hidden-hit",
                "safeLabel": "另一份楼盘资料",
                "canRead": True,
            }],
            "hasMore": False,
        }, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "secret-source-id",
            "query": "1栋101的电话",
        },
        user_message="查secret-source-id的1栋101电话",
    )

    assert result["code"] == "resource_query_not_found"
    assert calls == ["/api/xiaoban/internal/resource-index"]


def test_resolve_resource_query_does_not_strip_md_inside_normal_title(monkeypatch):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({
            "ok": True,
            "items": [{
                "resourceUid": "resource-letter-a",
                "safeLabel": "A",
                "canRead": True,
            }],
            "hasMore": False,
        }, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": "AMD",
            "query": "查内容",
        },
        user_message="查AMD的内容",
    )

    assert result["code"] == "resource_query_not_found"
    assert calls[0]["payload"]["query"] == "AMD"
    assert len(calls) == 1


def test_resolve_resource_query_requires_trusted_message_and_specific_name(
    monkeypatch,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    missing_message = _call({
        "operation": "resolve",
        "resource_query": "复地金融岛",
        "query": "17栋801",
    })
    shortened_title = _call(
        {
            "operation": "resolve",
            "resource_query": "A",
            "query": "查内容",
        },
        user_message="查AMD内容",
    )

    assert missing_message["code"] == "trusted_resource_query_required"
    assert shortened_title["code"] == "resource_query_too_short"
    assert calls == []


@pytest.mark.parametrize(
    ("resource_query", "user_message"),
    [
        ("泰悦", "查泰悦湾1栋702"),
        ("AMD", "查AMD项目的内容"),
    ],
)
def test_resolve_resource_query_rejects_model_shortened_title_phrase(
    monkeypatch,
    resource_query,
    user_message,
):
    calls = []

    def fake_post(path, payload, *, session=None, explicit_confirmation=False):
        calls.append({"path": path, "payload": payload})
        return json.dumps({"ok": True}, ensure_ascii=False)

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(
        {
            "operation": "resolve",
            "resource_query": resource_query,
            "query": "查内容",
        },
        user_message=user_message,
    )

    assert result["code"] == "resource_query_not_in_user_message"
    assert calls == []


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
            "idempotency_key": "write-001",
        },
        message_id=message_id,
        session_id=session_id,
    )

    assert result["code"] == "trusted_write_context_required"
    assert internal_calls == []


def test_preview_normalizes_flat_add_node_and_ignores_model_version(
    internal_calls,
):
    payload = {
        "nodeId": "model-node-1",
        "label": "客户需求",
        "type": "skill",
        "summary": "先核对需求",
        "body": "以授权资料为准。",
        "x": 640,
        "y": 360,
        "color": "#2563eb",
    }
    result = _call(
        {
            "operation": "preview_write",
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.add-node",
            "payload": payload,
            "expected_version": "model-guessed-v999",
            "idempotency_key": "write-001",
        }
    )
    nested_result = _call(
        {
            "operation": "preview_write",
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.add-node",
            "payload": {
                "node": {
                    "nodeId": "model-node-2",
                    "name": "复盘真实结果",
                    "nodeType": "skill",
                    "content": "写入后必须回读。",
                }
            },
            "idempotency_key": "write-002",
        }
    )

    assert result["ok"] is True
    assert nested_result["ok"] is True
    assert internal_calls[0] == (
        {
            "path": "/api/xiaoban/internal/authorization/write/preview",
            "payload": {
                "authorizationId": "AUTH-ABC123",
                "action": "knowledge-graph.add-node",
                "payload": {
                    "node": {
                        "id": "model-node-1",
                        "label": "客户需求",
                        "type": "skill",
                        "summary": "先核对需求",
                        "body": "以授权资料为准。",
                        "x": 640,
                        "y": 360,
                        "color": "#2563eb",
                    }
                },
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
    )
    assert internal_calls[1]["payload"] == {
        "authorizationId": "AUTH-ABC123",
        "action": "knowledge-graph.add-node",
        "payload": {
            "node": {
                "id": "model-node-2",
                "label": "复盘真实结果",
                "type": "skill",
                "body": "写入后必须回读。",
            }
        },
        "idempotencyKey": "write-002",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "nodeId": "model-node-1",
            "label": "节点",
            "type": "skill",
            "ownerUser": "forged-owner",
        },
        {
            "node": {"label": "节点", "type": "skill"},
            "label": "混用字段",
        },
        {
            "graphId": "KGREF-FORGED",
            "node": {"label": "节点", "type": "skill"},
        },
        {
            "node": {
                "label": "规范名",
                "name": "别名冲突",
                "type": "skill",
            },
        },
    ],
)
def test_preview_rejects_add_node_unknown_mixed_or_graph_id_before_transport(
    internal_calls,
    payload,
):
    result = _call(
        {
            "operation": "preview_write",
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.add-node",
            "payload": payload,
            "idempotency_key": "write-001",
        }
    )

    assert result["ok"] is False
    assert result["code"] == "write_payload_fields_not_allowed"
    assert internal_calls == []


def test_preview_rejects_unknown_write_action_before_transport(internal_calls):
    result = _call(
        {
            "operation": "preview_write",
            "authorization_id": "AUTH-ABC123",
            "action": "knowledge-graph.delete-node",
            "payload": {"nodeId": "node-1"},
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
