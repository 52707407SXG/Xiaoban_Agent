"""Focused contract tests for My Stand's preview/approve/commit write tool."""

import hashlib
import json

import pytest

from tools import mystand_authorization_write_tool as bridge


SESSION = {
    "platform": "api_server",
    "user_id": "owner-user-001",
    "message_id": "message-write-0001",
    "session_id": "session-write-0001",
}


def _preview_token_hash(token: str) -> str:
    serialized = json.dumps(token, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _receipt(*, token: str, idempotency_key: str, approval_id: str) -> dict:
    return {
        "ok": True,
        "status": 200,
        "receiptVersion": "authorization-write-receipt-v2",
        "verified": True,
        "audit": {"recorded": True, "auditId": "audit-write-0001"},
        "recovery": {
            "recoveryId": "auth-recovery-" + "d" * 32,
            "retentionDays": 7,
            "expiresAt": "2026-08-30T12:00:00.000Z",
        },
        "confirmationId": approval_id,
        "confirmationMode": "separate-user-confirmation",
        "action": "note.append-content",
        "target": {"noteId": "note-1"},
        "expectedVersion": "note-v1",
        "nextVersion": "note-v2",
        "idempotencyKey": idempotency_key,
        "requestFingerprint": "a" * 64,
        "previewTokenHash": _preview_token_hash(token),
        "changeDigest": "c" * 64,
        "committedAt": "2026-08-23T12:00:00.000Z",
    }


def _args(**overrides) -> dict:
    return {
        "operation": "preview_and_apply",
        "resource": {"name": "小伴写入测试笔记", "type_hint": "note"},
        "action": "note.append-content",
        "payload": {"content": "新增一段"},
        "idempotency_key": "write-note-idempotency-0001",
        **overrides,
    }


@pytest.fixture(autouse=True)
def trusted_session(monkeypatch):
    monkeypatch.setattr(bridge, "mark_mystand_private_query_turn", lambda: None)
    monkeypatch.setattr(bridge, "_current_session", lambda: dict(SESSION))


def test_schema_exposes_one_exact_approval_flow():
    parameters = bridge.MYSTAND_AUTHORIZATION_WRITE_SCHEMA["parameters"]
    assert parameters["properties"]["operation"]["enum"] == ["preview_and_apply"]
    assert set(parameters["required"]) == {
        "operation",
        "resource",
        "action",
        "payload",
        "idempotency_key",
    }
    assert "preview_token" not in parameters["properties"]
    assert "execute_safe_write" not in json.dumps(parameters)


@pytest.mark.parametrize(
    "session",
    [
        {**SESSION, "platform": "telegram"},
        {**SESSION, "user_id": ""},
    ],
)
def test_requires_authenticated_api_session(monkeypatch, session):
    calls = []
    monkeypatch.setattr(bridge, "_current_session", lambda: dict(session))
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args()))

    assert result["code"] == "mystand_session_required"
    assert calls == []


@pytest.mark.parametrize("missing_key", ["message_id", "session_id"])
def test_requires_trusted_write_context(monkeypatch, missing_key):
    calls = []
    session = {**SESSION, missing_key: ""}
    monkeypatch.setattr(bridge, "_current_session", lambda: session)
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args()))

    assert result["code"] == "trusted_write_context_required"
    assert calls == []


def test_approved_change_previews_then_commits_exact_plan(monkeypatch):
    calls = []
    token = "preview-token-write-0001"
    approval_id = "approval_" + "b" * 32

    def post(path, payload, **kwargs):
        calls.append((path, payload, kwargs))
        if path.endswith("/preview"):
            return json.dumps({
                "ok": True,
                "previewToken": token,
                "action": "note.append-content",
                "target": {"noteId": "note-1"},
                "expectedVersion": "note-v1",
                "preview": {
                    "title": "追加未加密笔记",
                    "before": "原文",
                    "after": "原文\n新增一段",
                },
                "displayPreview": {
                    "title": "追加未加密笔记",
                    "action": "note.append-content",
                    "target": "小伴写入测试笔记",
                    "before": "原文",
                    "after": "原文\n新增一段",
                    "changeType": "update",
                    "summary": "追加一段",
                    "recoveryDays": 7,
                },
            })
        if path.endswith("/commit"):
            return json.dumps(_receipt(
                token=token,
                idempotency_key="write-note-idempotency-0001",
                approval_id=approval_id,
            ))
        raise AssertionError(path)

    approvals = []
    monkeypatch.setattr(bridge, "_post_internal", post)
    monkeypatch.setattr(
        bridge,
        "request_gateway_action_approval",
        lambda **kwargs: approvals.append(kwargs) or {
            "approved": True,
            "choice": "once",
            "approval_id": approval_id,
            "message": None,
        },
    )

    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args()))

    assert result["ok"] is True
    assert result["verified"] is True
    assert result["recovery"]["retentionDays"] == 7
    assert [item[0].rsplit("/", 1)[-1] for item in calls] == ["preview", "commit"]
    assert calls[1][2]["gateway_approval_id"] == approval_id
    assert approvals[0]["choices"] == ("once", "deny")
    assert approvals[0]["metadata"]["approval_kind"] == "authorization-write"
    assert approvals[0]["metadata"]["write_preview"]["before"] == "原文"
    assert approvals[0]["metadata"]["write_preview"]["after"].endswith("新增一段")


def test_denied_change_is_cancelled_without_commit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append((path, payload, kwargs)) or json.dumps({
            "ok": True,
            "previewToken": "preview-token-denied",
            "action": "note.append-content",
            "target": {"noteId": "note-1"},
            "expectedVersion": "note-v1",
            "preview": {"title": "追加笔记", "before": "原文", "after": "新文"},
        }),
    )
    monkeypatch.setattr(
        bridge,
        "request_gateway_action_approval",
        lambda **_kwargs: {
            "approved": False,
            "choice": "deny",
            "approval_id": "approval_" + "e" * 32,
            "message": "用户拒绝了当前操作，操作没有执行。",
        },
    )

    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args()))

    assert result["ok"] is False
    assert result["code"] == "authorization_write_user_denied"
    assert [item[0].rsplit("/", 1)[-1] for item in calls] == ["preview", "cancel"]


def test_preview_failure_never_requests_approval(monkeypatch):
    approvals = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *_args, **_kwargs: json.dumps({
            "ok": False,
            "status": 404,
            "code": "write_resource_not_available",
            "error": "资料不存在",
        }),
    )
    monkeypatch.setattr(
        bridge,
        "request_gateway_action_approval",
        lambda **kwargs: approvals.append(kwargs),
    )

    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args()))

    assert result["ok"] is False
    assert result["code"] == "write_resource_not_available"
    assert approvals == []


@pytest.mark.parametrize("operation", ["execute_safe_write", "preview_write", "commit_write"])
def test_legacy_or_same_turn_operations_are_not_available(operation):
    result = json.loads(bridge.mystand_authorization_write_tool_handler({
        "operation": operation,
    }))
    assert result["ok"] is False
    assert result["code"] == "authorization_read_not_allowed"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "graphId": "forged-internal-id",
            "node": {"label": "节点", "type": "skill"},
        },
        {
            "node": {"label": "节点", "type": "skill"},
            "label": "mixed-flat-field",
        },
        {
            "node": {
                "label": "canonical-label",
                "name": "conflicting-alias",
                "type": "skill",
            },
        },
        {
            "node": {
                "label": "节点",
                "type": "skill",
                "ownerUser": "forged-owner",
            },
        },
    ],
)
def test_invalid_payload_is_rejected_before_preview(monkeypatch, payload):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args(
        action="knowledge-graph.add-node",
        resource={"name": "测试图谱", "type_hint": "knowledge-graph"},
        payload=payload,
    )))
    assert result["ok"] is False
    assert result["code"] == "write_payload_fields_not_allowed"
    assert calls == []


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "nodeId": "model-node-1",
                "label": "客户需求",
                "type": "skill",
                "body": "以授权资料为准。",
            },
            {
                "node": {
                    "id": "model-node-1",
                    "label": "客户需求",
                    "type": "skill",
                    "body": "以授权资料为准。",
                },
            },
        ),
        (
            {
                "node": {
                    "nodeId": "model-node-2",
                    "name": "复盘真实结果",
                    "nodeType": "skill",
                    "content": "写入后必须回读。",
                },
            },
            {
                "node": {
                    "id": "model-node-2",
                    "label": "复盘真实结果",
                    "type": "skill",
                    "body": "写入后必须回读。",
                },
            },
        ),
    ],
)
def test_graph_add_normalizes_supported_aliases(monkeypatch, payload, expected):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, body, **kwargs: calls.append((path, body, kwargs))
        or json.dumps({
            "ok": False,
            "status": 404,
            "code": "write_resource_not_available",
            "error": "测试到达服务器",
        }),
    )

    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args(
        action="knowledge-graph.add-node",
        resource={"name": "测试图谱", "type_hint": "knowledge-graph"},
        payload=payload,
    )))

    assert result["code"] == "write_resource_not_available"
    assert calls[0][1]["payload"] == expected


def test_unknown_write_action_is_rejected_before_preview(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args(
        action="knowledge-graph.delete-node",
        resource={"name": "测试图谱", "type_hint": "knowledge-graph"},
        payload={"nodeId": "node-1"},
    )))

    assert result["code"] == "authorization_write_action_not_allowed"
    assert calls == []


def test_note_append_drops_model_internal_id_and_redundant_append_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append((path, payload, kwargs)) or json.dumps({
            "ok": False,
            "status": 404,
            "code": "write_resource_not_available",
            "error": "测试到达服务器",
        }),
    )
    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args(
        payload={"noteId": "model-guessed-id", "mode": "append", "content": "新增一段"},
    )))
    assert result["code"] == "write_resource_not_available"
    assert calls[0][1]["payload"] == {"content": "新增一段"}


def test_property_note_append_leaves_internal_location_to_server(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append((path, payload, kwargs)) or json.dumps({
            "ok": False,
            "status": 404,
            "code": "write_resource_not_available",
            "error": "测试到达服务器",
        }),
    )
    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args(
        action="property-note.append-text-block",
        resource={"name": "仁和春天国际花园 8-2-2204 房源笔记", "type_hint": "property-note"},
        payload={
            "archiveId": "M000006:FYWH1",
            "documentId": "M000006:FYBG1",
            "mode": "append",
            "content": "新增房源判断",
        },
    )))
    assert result["code"] == "write_resource_not_available"
    assert calls[0][1]["payload"] == {"content": "新增房源判断"}


def test_property_note_edit_preserves_exact_structure_operations_and_hides_archive(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append((path, payload, kwargs)) or json.dumps({
            "ok": False,
            "status": 404,
            "code": "write_resource_not_available",
            "error": "测试到达服务器",
        }),
    )
    operations = [
        {
            "op": "delete-block",
            "documentId": "basic-info",
            "blockId": "wrong-line",
            "beforeText": "误写内容。",
        },
        {
            "op": "replace-block-text",
            "documentId": "owner-info",
            "blockId": "owner-current-low-price",
            "beforeText": "当前底价：",
            "afterText": "当前底价：580万元",
        },
    ]
    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args(
        action="property-note.edit-blocks",
        resource={"name": "仁和春天国际花园 8-2-2204 房源笔记", "type_hint": "property-note"},
        payload={"archiveId": "MODEL-MUST-NOT-CHOOSE", "operations": operations},
    )))
    assert result["code"] == "write_resource_not_available"
    assert calls[0][1]["payload"] == {"operations": operations}


def test_receipt_without_recovery_point_is_not_reported_as_success(monkeypatch):
    token = "preview-token-no-recovery"
    approval_id = "approval_" + "f" * 32
    receipt = _receipt(
        token=token,
        idempotency_key="write-note-idempotency-0001",
        approval_id=approval_id,
    )
    receipt.pop("recovery")

    def post(path, _payload, **_kwargs):
        if path.endswith("/preview"):
            return json.dumps({
                "ok": True,
                "previewToken": token,
                "action": "note.append-content",
                "target": {"noteId": "note-1"},
                "expectedVersion": "note-v1",
                "preview": {"title": "追加笔记", "before": "原文", "after": "新文"},
            })
        return json.dumps(receipt)

    monkeypatch.setattr(bridge, "_post_internal", post)
    monkeypatch.setattr(
        bridge,
        "request_gateway_action_approval",
        lambda **_kwargs: {
            "approved": True,
            "choice": "once",
            "approval_id": approval_id,
        },
    )
    result = json.loads(bridge.mystand_authorization_write_tool_handler(_args()))
    assert result["ok"] is False
    assert result["status"] == 502
    assert result["code"] == "authorization_write_receipt_invalid"


def test_other_account_context_cannot_be_model_supplied():
    result = json.loads(bridge.mystand_authorization_write_tool_handler({
        **_args(),
        "owner_user": "other-account",
    }))
    assert result["ok"] is False
    assert result["code"] == "invalid_authorization_write_arguments"
