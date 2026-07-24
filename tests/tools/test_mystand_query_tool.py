"""Tests for Xiaoban's high-level My Stand semantic-query bridge."""

import io
import json
import urllib.error

from gateway.session_context import clear_session_vars, set_session_vars
from tools import mystand_query_tool as bridge

FACT_NEEDS = {
    "owner.name",
    "owner.phone",
    "owner.family",
    "owner.interests",
    "owner.economic",
    "relationship.communication",
    "relationship.followup",
    "property.parking",
    "property.area",
    "property.price.total",
    "property.price.unit",
    "property.rent",
    "document.content",
    "resource.summary",
    "graph.nodes",
    "graph.relations",
}
RESOURCE_TYPES = {
    "note",
    "knowledge-markdown",
    "knowledge-graph",
    "property-note",
    "business-archive",
    "profile-card",
    "property-data",
    "property-md",
    "finance-archive",
}


def _valid_plan():
    return {
        "operation": "read",
        "resource": {
            "name": "复地金融岛楼盘MD",
            "type_hint": "property-md",
        },
        "entities": [
            {"kind": "building", "value": "17", "role": "locator"},
            {"kind": "unit", "value": "1"},
            {"kind": "room", "value": "801"},
        ],
        "fact_needs": ["owner.name", "owner.phone"],
        "mode": "facts",
    }


def _call(
    args,
    *,
    platform="api_server",
    user_id="ZYJ005",
    user_message="查复地金融岛17栋1单元801的业主姓名和电话",
):
    tokens = set_session_vars(
        platform=platform,
        user_id=user_id,
        message_id="msg-1",
        session_id="session-1",
        user_message=user_message,
    )
    try:
        return json.loads(bridge.mystand_query_tool_handler(args))
    finally:
        clear_session_vars(tokens)


def test_contract_is_semantic_and_contains_no_identity_or_internal_id_inputs():
    parameters = bridge.MYSTAND_QUERY_SCHEMA["parameters"]
    properties = parameters["properties"]

    assert set(properties) == {
        "operation",
        "resource",
        "entities",
        "fact_needs",
        "mode",
    }
    assert parameters["additionalProperties"] is False
    assert properties["operation"]["const"] == "read"
    assert set(properties["fact_needs"]["items"]["enum"]) == FACT_NEEDS
    assert set(
        properties["resource"]["properties"]["type_hint"]["enum"]
    ) == RESOURCE_TYPES
    assert set(properties["entities"]["items"]["properties"]["kind"]["enum"]) == {
        "building",
        "unit",
        "room",
        "person",
        "estate",
        "document",
        "topic",
        "time",
    }
    assert "role" not in properties["entities"]["items"]["required"]
    assert "queryText" not in properties
    assert {
        "owner",
        "owner_user",
        "user",
        "user_id",
        "authorization_id",
        "auth_id",
        "resource_uid",
        "source_id",
    }.isdisjoint(properties)


def test_handler_injects_trusted_query_text_and_session_identity_stays_out_of_body(
    monkeypatch,
):
    calls = []

    def fake_post(payload, session):
        calls.append((payload, session))
        return json.dumps(
            {
                "ok": True,
                "facts": [{"predicate": "owner.name", "value": "测试姓名"}],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(_valid_plan())

    assert result["ok"] is True
    assert len(calls) == 1
    payload, session = calls[0]
    assert payload == {
        **_valid_plan(),
        "queryText": "查复地金融岛17栋1单元801的业主姓名和电话",
    }
    assert session == {
        "platform": "api_server",
        "user_id": "ZYJ005",
        "message_id": "msg-1",
        "session_id": "session-1",
    }
    assert {
        "owner",
        "owner_user",
        "user",
        "user_id",
        "authorization_id",
        "resource_uid",
        "source_id",
    }.isdisjoint(payload)


def test_handler_rejects_model_supplied_identity_ids_and_query_text(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args: calls.append(args) or json.dumps({"ok": True}),
    )

    for forbidden in (
        {"owner": "ZYJ999"},
        {"user_id": "ZYJ999"},
        {"authorization_id": "AUTH-forged"},
        {"resource_uid": "forged"},
        {"source_id": "forged"},
        {"queryText": "伪造问题"},
    ):
        result = _call({**_valid_plan(), **forbidden})
        assert result["code"] == "invalid_mystand_query_arguments"

    nested = _valid_plan()
    nested["resource"] = {**nested["resource"], "resource_uid": "forged"}
    assert _call(nested)["code"] == "invalid_mystand_query_arguments"
    assert calls == []


def test_handler_accepts_person_entity_without_guessed_resource_title(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda payload, session: calls.append((payload, session))
        or json.dumps({"ok": True}),
    )
    plan = {
        "operation": "read",
        "entities": [
            {"kind": "person", "value": "汤总", "role": "subject"},
        ],
        "fact_needs": ["owner.family", "owner.interests"],
        "mode": "facts",
    }
    result = _call(
        plan,
        user_message="汤总家里是什么情况，平时喜欢什么？",
    )
    assert result["ok"] is True
    assert calls[0][0] == {
        **plan,
        "queryText": "汤总家里是什么情况，平时喜欢什么？",
    }


def test_handler_rejects_invalid_semantic_enums_and_duplicate_fact_needs(
    monkeypatch,
):
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *_args: json.dumps({"ok": True}),
    )

    invalid_type = _valid_plan()
    invalid_type["resource"] = {
        "name": "资料",
        "type_hint": "unknown",
    }
    assert _call(invalid_type)["code"] == "invalid_mystand_query_arguments"

    invalid_entity = _valid_plan()
    invalid_entity["entities"] = [{"kind": "authorization", "value": "AUTH-x"}]
    assert _call(invalid_entity)["code"] == "invalid_mystand_query_arguments"

    invalid_fact = _valid_plan()
    invalid_fact["fact_needs"] = ["owner.name", "owner.name"]
    assert _call(invalid_fact)["code"] == "invalid_mystand_query_arguments"

    invalid_fact_type = _valid_plan()
    invalid_fact_type["fact_needs"] = [{"predicate": "owner.name"}]
    assert _call(invalid_fact_type)["code"] == "invalid_mystand_query_arguments"


def test_handler_requires_authenticated_api_session_and_trusted_user_message(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args: calls.append(args) or json.dumps({"ok": True}),
    )

    assert _call(_valid_plan(), platform="telegram")["code"] == (
        "mystand_session_required"
    )
    assert _call(_valid_plan(), user_id="")["code"] == "mystand_session_required"
    assert _call(_valid_plan(), user_message="")["code"] == (
        "trusted_query_text_required"
    )
    assert calls == []


def test_internal_post_uses_only_trusted_session_headers(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"ok":true,"status":"matched"}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    payload = {
        **_valid_plan(),
        "queryText": "可信原始问题",
    }
    result = json.loads(
        bridge._post_internal(
            payload,
            {
                "user_id": "ZYJ005",
                "message_id": "msg-1",
                "session_id": "session-1",
            },
        )
    )

    request = captured["request"]
    headers = {key.lower(): value for key, value in request.header_items()}
    assert result == {"ok": True, "status": "matched"}
    assert captured["timeout"] == 20
    assert request.full_url.endswith("/api/xiaoban/internal/query")
    assert headers["x-xiaoban-user-id"] == "ZYJ005"
    assert headers["x-xiaoban-message-id"] == "msg-1"
    assert headers["x-xiaoban-session-id"] == "session-1"
    assert json.loads(request.data.decode("utf-8")) == payload


def test_internal_post_rejects_oversized_result(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b"x" * (bridge._MAX_RESPONSE_BYTES + 1)

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(
        bridge.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )

    assert result["status"] == 413
    assert result["code"] == "mystand_query_result_too_large"


def test_http_409_synthesizes_safe_clarification_and_candidates(monkeypatch):
    upstream = {
        "ok": False,
        "code": "resource_query_ambiguous",
        "clarification": "找到两个楼盘，请补充完整名称。",
        "candidates": [
            {
                "safeLabel": "复地金融岛楼盘MD",
                "resourceType": "property-md",
                "resourceUid": "resource-secret-uid",
                "authorizationId": "AUTH-secret",
                "ownerUser": "ZYJ999",
                "sourceId": "source-secret",
            },
            {
                "displayName": "复地金融岛楼盘数据",
                "typeHint": "property-data",
                "internalId": "internal-secret",
            },
        ],
        "details": {
            "resourceUid": "nested-resource-secret",
            "owner": "nested-owner-secret",
        },
    }

    def raise_conflict(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps(upstream).encode("utf-8")),
        )

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", raise_conflict)
    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )
    serialized = json.dumps(result, ensure_ascii=False)

    assert result == {
        "ok": False,
        "status": 409,
        "code": "resource_query_ambiguous",
        "error": "找到多项可能资料，需要补充信息。",
        "clarification": "找到两个楼盘，请补充完整名称。",
        "candidates": [
            {"label": "复地金融岛楼盘MD", "type": "property-md"},
            {"label": "复地金融岛楼盘数据", "type": "property-data"},
        ],
    }
    for secret in (
        "resource-secret-uid",
        "AUTH-secret",
        "ZYJ999",
        "source-secret",
        "internal-secret",
        "nested-resource-secret",
        "nested-owner-secret",
    ):
        assert secret not in serialized


def test_http_409_without_candidates_does_not_invent_multiple_matches(
    monkeypatch,
):
    upstream = {
        "ok": False,
        "code": "resource_needs_clarification",
        "clarification": "没有找到唯一且可供小伴读取的资料，请补充资料名称。",
        "candidates": [],
    }

    def raise_conflict(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps(upstream).encode("utf-8")),
        )

    monkeypatch.setattr(
        bridge,
        "_api_base_url",
        lambda: "http://127.0.0.1:18081",
    )
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", raise_conflict)

    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )

    assert result["status"] == 409
    assert result.get("candidates", []) == []
    assert result["error"] == (
        "没有找到唯一且可供小伴读取的资料，请补充资料名称。"
    )
    assert "多项" not in result["error"]


def test_http_404_limits_clarification_and_drops_internal_identifiers(monkeypatch):
    upstream = {
        "code": "resource_query_not_found",
        "details": {
            "clarification": "请选择 AUTH-secret 后继续。",
            "candidates": [
                {
                    "name": "候选资料",
                    "type": "note",
                    "uid": "hidden-uid",
                }
            ],
        },
    }

    def raise_not_found(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(json.dumps(upstream).encode("utf-8")),
        )

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", raise_not_found)
    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )

    assert result["status"] == 404
    assert result["code"] == "resource_query_not_found"
    assert "clarification" not in result
    assert result["candidates"] == [{"name": "候选资料", "type": "note"}]
    assert "AUTH-secret" not in json.dumps(result, ensure_ascii=False)
    assert "hidden-uid" not in json.dumps(result, ensure_ascii=False)
