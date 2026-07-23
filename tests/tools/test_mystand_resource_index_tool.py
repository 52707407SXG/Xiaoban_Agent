"""Tests for Xiaoban's read-only My Stand resource-index bridge."""

import json

from gateway.session_context import clear_session_vars, set_session_vars
from tools import mystand_resource_index_tool as bridge


def _call(args, *, platform="api_server", user_id="ZYJ005"):
    tokens = set_session_vars(platform=platform, user_id=user_id)
    try:
        return json.loads(bridge.mystand_resource_index_tool_handler(args))
    finally:
        clear_session_vars(tokens)


def test_schema_is_read_only_and_cannot_accept_owner():
    parameters = bridge.MYSTAND_RESOURCE_INDEX_SCHEMA["parameters"]
    assert parameters["properties"]["operation"]["const"] == "list_resources"
    assert "owner_user" not in parameters["properties"]
    assert parameters["additionalProperties"] is False


def test_current_session_identity_is_the_only_owner_source(monkeypatch):
    calls = []

    def fake_post(payload, user_id):
        calls.append((payload, user_id))
        return json.dumps({"ok": True})

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call({
        "operation": "list_resources",
        "module_id": "property-notes",
        "status": "all",
        "cursor": "cursor-1",
        "limit": 20,
        "owner_user": "ZYJ999",
    })

    assert result["ok"] is True
    assert calls == [({
        "operation": "list_resources",
        "moduleId": "property-notes",
        "status": "all",
        "cursor": "cursor-1",
        "limit": 20,
    }, "ZYJ005")]


def test_rejects_non_api_and_invalid_arguments(monkeypatch):
    monkeypatch.setattr(bridge, "_post_internal", lambda *_args: json.dumps({"ok": True}))
    assert _call({"operation": "list_resources"}, platform="telegram")["code"] == "mystand_session_required"
    assert _call({"operation": "write"})["code"] == "invalid_resource_index_operation"
    assert _call({"operation": "list_resources", "limit": 0})["code"] == "invalid_resource_index_limit"
    assert _call({"operation": "list_resources", "status": "deleted"})["code"] == "invalid_resource_index_status"
