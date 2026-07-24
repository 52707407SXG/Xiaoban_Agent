"""Tests for the write-only My Stand authorization wrapper."""

import json

import pytest

from gateway.session_context import (
    clear_session_vars,
    mystand_private_query_turn_active,
    set_session_vars,
)
from tools import mystand_authorization_write_tool as bridge


def test_schema_exposes_only_write_operations():
    parameters = bridge.MYSTAND_AUTHORIZATION_WRITE_SCHEMA["parameters"]

    assert parameters["properties"]["operation"]["enum"] == [
        "preview_write",
        "commit_write",
    ]
    assert "resource" in parameters["properties"]
    assert "authorization_id" not in parameters["properties"]
    assert "expected_version" not in parameters["properties"]
    assert "query" not in parameters["properties"]
    assert "resource_uid" not in parameters["properties"]
    assert parameters["additionalProperties"] is False


def test_handler_hard_rejects_read_operations_and_extra_fields(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )
    monkeypatch.setattr(
        bridge,
        "mystand_authorization_tool_handler",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    read_result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {"operation": "resolve"}
        )
    )
    extra_result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource_uid": "forged",
            }
        )
    )

    assert read_result["code"] == "authorization_read_not_allowed"
    assert extra_result["code"] == "invalid_authorization_write_arguments"
    assert calls == []


def test_handler_delegates_valid_write_operations(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )

    def fake_handler(args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps({"ok": True})

    monkeypatch.setattr(
        bridge,
        "mystand_authorization_tool_handler",
        fake_handler,
    )
    args = {
        "operation": "commit_write",
        "preview_token": "preview-token",
        "idempotency_key": "idem-1",
    }
    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            args,
            task_id="task-1",
        )
    )

    assert result["ok"] is True
    assert calls == [(args, {"task_id": "task-1"})]


def test_failed_commit_result_injects_immediate_integrity_notice(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )
    monkeypatch.setattr(
        bridge,
        "mystand_authorization_tool_handler",
        lambda *_args, **_kwargs: json.dumps(
            {
                "ok": False,
                "status": 409,
                "code": "authorization_write_conflict",
                "error": "版本冲突",
            },
            ensure_ascii=False,
        ),
    )

    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "commit_write",
                "preview_token": "preview-token",
                "idempotency_key": "idem-failed-commit",
            }
        )
    )

    assert result["ok"] is False
    assert "本次写入没有成功" in result["integrity_notice"]
    assert "禁止向用户声称已经写入" in result["integrity_notice"]


def test_argument_failure_also_injects_immediate_integrity_notice(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )

    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "城南一号业主特征卡",
                    "type_hint": "profile-card",
                },
                "action": "profile-card.update-field",
                "payload": {"fields": {"思维特征": "测试"}},
            }
        )
    )

    assert result["code"] == "authorization_argument_missing"
    assert "本次写入没有成功" in result["integrity_notice"]
    assert "禁止向用户声称已经写入" in result["integrity_notice"]


def test_preview_success_reminds_model_that_nothing_was_committed(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *_args, **_kwargs: json.dumps(
            {
                "ok": True,
                "status": 200,
                "previewToken": "preview-token",
            }
        ),
    )

    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "城南一号业主特征卡",
                    "type_hint": "profile-card",
                },
                "action": "profile-card.update-field",
                "payload": {"fields": {"思维特征": "测试"}},
                "idempotency_key": "idem-preview-only",
            }
        )
    )

    assert result["ok"] is True
    assert "只是预览" in result["integrity_notice"]
    assert "没有写入" in result["integrity_notice"]


@pytest.mark.parametrize("operation", ["preview_write", "commit_write"])
def test_handler_taints_before_preview_or_commit_processing(
    operation,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "XIAOBAN_MYSTAND_PRIVATE_TAINT_FILE",
        str(tmp_path / "private-taints.json"),
    )
    observed = []

    def observe_commit(_args, **_kwargs):
        observed.append(("commit", mystand_private_query_turn_active()))
        return '{"ok":true}'

    def observe_preview(_path, _payload, **_kwargs):
        observed.append(("preview", mystand_private_query_turn_active()))
        return '{"ok":true}'

    monkeypatch.setattr(
        bridge,
        "mystand_authorization_tool_handler",
        observe_commit,
    )
    monkeypatch.setattr(bridge, "_post_internal", observe_preview)
    tokens = set_session_vars(
        platform="api_server",
        user_id="ZYJ005",
        user_message="更新这份资料",
        message_id="msg-write-taint",
        session_id="session-write-taint",
        session_key="key-write-taint",
    )
    try:
        if operation == "commit_write":
            args = {
                "operation": "commit_write",
                "preview_token": "preview-token",
                "idempotency_key": "idem-commit-taint",
            }
        else:
            args = {
                "operation": "preview_write",
                "resource": {
                    "name": "汤总房源笔记",
                    "type_hint": "property-note",
                },
                "action": "property-note.append-text-block",
                "payload": {"content": "跟进记录"},
                "idempotency_key": "idem-preview-taint",
            }

        result = json.loads(
            bridge.mystand_authorization_write_tool_handler(args)
        )

        assert result["ok"] is True
        assert observed == [
            (
                "commit" if operation == "commit_write" else "preview",
                True,
            )
        ]
    finally:
        clear_session_vars(tokens)


def test_preview_resolves_semantic_resource_without_auth_id(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append(
            (path, payload, kwargs)
        ) or json.dumps({"ok": True}),
    )
    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "汤总房源笔记",
                    "type_hint": "property-note",
                },
                "action": "property-note.append-text-block",
                "payload": {
                    "archiveId": "M000001:FYBG1",
                    "documentId": "owner-info",
                    "blockId": "new-block",
                    "content": "跟进记录",
                },
                "idempotency_key": "idem-preview-1",
            }
        )
    )
    assert result["ok"] is True
    assert calls == [(
        "/api/xiaoban/internal/authorization/write/preview",
        {
            "resource": {
                "name": "汤总房源笔记",
                "typeHint": "property-note",
            },
            "action": "property-note.append-text-block",
            "payload": {
                "archiveId": "M000001:FYBG1",
                "documentId": "owner-info",
                "blockId": "new-block",
                "content": "跟进记录",
            },
            "idempotencyKey": "idem-preview-1",
        },
        {
            "session": {
                "platform": "api_server",
                "user_id": "ZYJ005",
                "message_id": "msg-001",
                "session_id": "session-001",
            },
        },
    )]


def test_preview_rejects_auth_id_and_type_mismatch(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )
    auth_result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "authorization_id": "AUTH-forged",
            }
        )
    )
    assert auth_result["code"] == "invalid_authorization_write_arguments"
    mismatch = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "资料",
                    "type_hint": "knowledge-graph",
                },
                "action": "property-note.append-text-block",
                "payload": {},
                "idempotency_key": "idem-preview-2",
            }
        )
    )
    assert mismatch["code"] == "invalid_write_resource"


def test_schema_tells_model_idempotency_key_is_required_for_preview():
    parameters = bridge.MYSTAND_AUTHORIZATION_WRITE_SCHEMA["parameters"]

    key_description = parameters["properties"]["idempotency_key"].get("description", "")
    assert "REQUIRED for preview_write" in key_description
    assert "unique" in key_description
    assert "retrying the same preview" in key_description
    assert "Never reuse" in key_description
    tool_description = bridge.MYSTAND_AUTHORIZATION_WRITE_SCHEMA["description"]
    assert "preview_write REQUIRES idempotency_key" in tool_description


def test_preview_still_hard_fails_closed_without_idempotency_key(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "mark_mystand_private_query_turn",
        lambda: None,
    )
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-001",
            "session_id": "session-001",
        },
    )
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "{}",
    )

    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "城南一号2栋10楼特征卡",
                    "type_hint": "profile-card",
                },
                "action": "profile-card.update-field",
                "payload": {"fields": {"思维特征": "测试"}},
            }
        )
    )

    assert result["ok"] is False
    assert result["code"] == "authorization_argument_missing"
    assert "idempotency_key" in result["error"]
    assert result["integrity_notice"]
    assert calls == [], "缺少 idempotency_key 时不得触达 My Stand 写入预览接口"
