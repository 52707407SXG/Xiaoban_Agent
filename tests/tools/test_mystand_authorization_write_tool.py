"""Tests for the write-only My Stand authorization wrapper."""

import hashlib
import json

import pytest

from gateway.session_context import (
    clear_session_vars,
    mystand_private_query_turn_active,
    set_session_vars,
)
from tools import mystand_authorization_write_tool as bridge


def _node_preview_token_hash(preview_token: str) -> str:
    """Independently mirror SHA256(UTF-8(JSON.stringify(String(token))))."""
    serialized = json.dumps(
        str(preview_token),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _node_confirmation_id(
    *,
    user_id: str = "ZYJ005",
    session_id: str = "session-001",
    message_id: str = "msg-0001",
) -> str:
    context = f"{user_id}\u001f{session_id}\u001f{message_id}"
    return f"xiaoban-write-{hashlib.sha256(context.encode('utf-8')).hexdigest()}"


def _valid_commit_receipt(
    *,
    preview_token: str = "preview-token",
    idempotency_key: str = "same-key-as-preview",
    user_id: str = "ZYJ005",
    session_id: str = "session-001",
    message_id: str = "msg-0001",
) -> dict:
    return {
        "ok": True,
        "status": 200,
        "receiptVersion": "authorization-write-receipt-v2",
        "verified": True,
        "audit": {
            "recorded": True,
            "auditId": "audit-e4-agent-0001",
        },
        "confirmationId": _node_confirmation_id(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
        ),
        "action": "knowledge-graph.add-node",
        "target": {"graphId": "graph-e4", "nodeId": "node-e4"},
        "expectedVersion": "graph-version-1",
        "nextVersion": "graph-version-2",
        "idempotencyKey": idempotency_key,
        "requestFingerprint": "a" * 64,
        "previewTokenHash": _node_preview_token_hash(preview_token),
        "changeDigest": "c" * 64,
        "committedAt": "2026-08-03T12:00:00.000Z",
    }


def _call_stubbed_commit(
    monkeypatch,
    upstream_result: dict,
    *,
    preview_token: str = "preview-token",
    idempotency_key: str = "same-key-as-preview",
) -> dict:
    monkeypatch.setattr(bridge, "mark_mystand_private_query_turn", lambda: None)
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-0001",
            "session_id": "session-001",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_mystand_authorization_write_operation_handler",
        lambda *_args, **_kwargs: json.dumps(upstream_result),
    )
    return json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "commit_write",
                "preview_token": preview_token,
                "idempotency_key": idempotency_key,
            }
        )
    )


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
    assert "finance-archive" in parameters["properties"]["resource"]["properties"]["type_hint"]["enum"]
    assert "finance-archive.update-row-fields" in parameters["properties"]["action"]["enum"]
    assert parameters["additionalProperties"] is False


def test_schema_exposes_structured_knowledge_graph_node_payloads_without_graph_id():
    payload = bridge.MYSTAND_AUTHORIZATION_WRITE_SCHEMA["parameters"]["properties"]["payload"]
    properties = payload["properties"]
    node = properties["node"]
    changes = properties["changes"]

    assert payload["type"] == "object"
    assert payload["additionalProperties"] is False
    assert node["type"] == "object"
    assert node["additionalProperties"] is False
    assert {
        "id",
        "label",
        "type",
        "summary",
        "body",
        "x",
        "y",
        "color",
        "nodeId",
        "name",
        "nodeType",
        "content",
    } == set(node["properties"])
    assert {
        "nodeId",
        "label",
        "type",
        "summary",
        "body",
        "x",
        "y",
        "color",
    } <= set(properties)
    assert changes["type"] == "object"
    assert changes["minProperties"] == 1
    assert changes["additionalProperties"] is False
    assert {
        "label",
        "type",
        "summary",
        "body",
        "x",
        "y",
        "color",
        "name",
        "nodeType",
        "content",
    } <= set(changes["properties"])
    assert properties["nodeId"]["type"] == "string"
    assert "graphId" not in properties
    assert "Never include graphId" in payload["description"]


def test_add_node_model_payloads_are_normalized_to_canonical_node(monkeypatch):
    calls = []
    monkeypatch.setattr(bridge, "mark_mystand_private_query_turn", lambda: None)
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-graph-add-flat",
            "session_id": "session-graph-add-flat",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append((path, payload, kwargs))
        or '{"ok":true}',
    )

    flat_result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "经纪人经验知识图谱",
                    "type_hint": "knowledge-graph",
                },
                "action": "knowledge-graph.add-node",
                "payload": {
                    "nodeId": "model-node-1",
                    "label": "先核对真实资料",
                    "type": "skill",
                    "summary": "写入前先读取权威来源",
                    "body": "不得根据模型记忆猜测。",
                    "x": 640,
                    "y": 360,
                    "color": "#2563eb",
                },
                "idempotency_key": "graph-add-flat-preview-0001",
            }
        )
    )
    nested_result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "经纪人经验知识图谱",
                    "type_hint": "knowledge-graph",
                },
                "action": "knowledge-graph.add-node",
                "payload": {
                    "node": {
                        "nodeId": "model-node-2",
                        "name": "复盘真实结果",
                        "nodeType": "skill",
                        "content": "写入后必须回读。",
                    }
                },
                "idempotency_key": "graph-add-nested-preview-0002",
            }
        )
    )

    assert flat_result["ok"] is True
    assert nested_result["ok"] is True
    assert calls[0][1]["payload"] == {
        "node": {
            "id": "model-node-1",
            "label": "先核对真实资料",
            "type": "skill",
            "summary": "写入前先读取权威来源",
            "body": "不得根据模型记忆猜测。",
            "x": 640,
            "y": 360,
            "color": "#2563eb",
        }
    }
    assert "graphId" not in calls[0][1]["payload"]
    assert calls[1][1]["payload"] == {
        "node": {
            "id": "model-node-2",
            "label": "复盘真实结果",
            "type": "skill",
            "body": "写入后必须回读。",
        }
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
                "label": "节点",
                "type": "skill",
                "graphId": "KGREF-FORGED",
            },
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
def test_add_node_rejects_unknown_mixed_or_graph_id_payload(
    payload,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(bridge, "mark_mystand_private_query_turn", lambda: None)
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-graph-add-rejected",
            "session_id": "session-graph-add-rejected",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args, **kwargs: calls.append((args, kwargs)) or '{"ok":true}',
    )

    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "经纪人经验知识图谱",
                    "type_hint": "knowledge-graph",
                },
                "action": "knowledge-graph.add-node",
                "payload": payload,
                "idempotency_key": "graph-add-rejected-preview-0001",
            }
        )
    )

    assert result["ok"] is False
    assert result["code"] == "write_payload_fields_not_allowed"
    assert calls == []


def test_update_node_requires_canonical_node_id_and_changes(monkeypatch):
    calls = []
    monkeypatch.setattr(bridge, "mark_mystand_private_query_turn", lambda: None)
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "ZYJ005",
            "message_id": "msg-graph-update",
            "session_id": "session-graph-update",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append((path, payload, kwargs))
        or '{"ok":true}',
    )
    base = {
        "operation": "preview_write",
        "resource": {
            "name": "经纪人经验知识图谱",
            "type_hint": "knowledge-graph",
        },
        "action": "knowledge-graph.update-node",
        "idempotency_key": "graph-update-preview-0001",
    }

    valid = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                **base,
                "payload": {
                    "nodeId": "node-1",
                    "changes": {"summary": "仅更新摘要"},
                },
            }
        )
    )
    aliased = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                **base,
                "payload": {
                    "nodeId": "node-1",
                    "changes": {
                        "name": "新名称",
                        "nodeType": "skill",
                        "content": "更新正文",
                        "color": "#2563eb",
                    },
                },
            }
        )
    )
    flat = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                **base,
                "payload": {
                    "nodeId": "node-1",
                    "summary": "不得推测性兼容扁平更新",
                },
            }
        )
    )
    conflict = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                **base,
                "payload": {
                    "nodeId": "node-1",
                    "changes": {
                        "label": "规范名",
                        "name": "别名冲突",
                    },
                },
            }
        )
    )

    assert valid["ok"] is True
    assert aliased["ok"] is True
    assert calls[0][1]["payload"] == {
        "nodeId": "node-1",
        "changes": {"summary": "仅更新摘要"},
    }
    assert "graphId" not in calls[0][1]["payload"]
    assert calls[1][1]["payload"] == {
        "nodeId": "node-1",
        "changes": {
            "label": "新名称",
            "type": "skill",
            "body": "更新正文",
            "color": "#2563eb",
        },
    }
    assert flat["ok"] is False
    assert flat["code"] == "write_payload_fields_not_allowed"
    assert conflict["ok"] is False
    assert conflict["code"] == "write_payload_fields_not_allowed"
    assert len(calls) == 2


def test_finance_archive_write_preview_uses_semantic_resource(monkeypatch):
    calls = []
    monkeypatch.setattr(bridge, "mark_mystand_private_query_turn", lambda: None)
    monkeypatch.setattr(
        bridge,
        "_current_session",
        lambda: {
            "platform": "api_server",
            "user_id": "52707407",
            "message_id": "msg-finance-write",
            "session_id": "session-finance-write",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda path, payload, **kwargs: calls.append((path, payload, kwargs)) or '{"ok":true}',
    )

    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "preview_write",
                "resource": {
                    "name": "游雪梅2026年个人业务档案",
                    "type_hint": "finance-archive",
                },
                "action": "finance-archive.update-row-fields",
                "payload": {
                    "brokerUser": "ZYJ001",
                    "year": 2026,
                    "recordId": "row-1",
                    "changes": {"notes": "待确认备注"},
                },
                "idempotency_key": "finance-write-preview-0001",
            }
        )
    )

    assert result["ok"] is True
    assert calls[0][0] == "/api/xiaoban/internal/authorization/write/preview"
    assert calls[0][1]["resource"]["typeHint"] == "finance-archive"
    assert calls[0][1]["action"] == "finance-archive.update-row-fields"


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
        "_mystand_authorization_write_operation_handler",
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
            "message_id": "msg-0001",
            "session_id": "session-001",
        },
    )

    def fake_handler(args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps(
            _valid_commit_receipt(
                preview_token=args["preview_token"],
                idempotency_key=args["idempotency_key"],
            )
        )

    monkeypatch.setattr(
        bridge,
        "_mystand_authorization_write_operation_handler",
        fake_handler,
    )
    args = {
        "operation": "commit_write",
        "preview_token": "preview-token",
        "idempotency_key": "idem-0001",
    }
    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            args,
            task_id="task-1",
        )
    )

    assert result["ok"] is True
    assert "可以准确说明写入成功" in result["integrity_notice"]
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
        "_mystand_authorization_write_operation_handler",
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

    def observe_commit(commit_args, **_kwargs):
        observed.append(("commit", mystand_private_query_turn_active()))
        return json.dumps(
            _valid_commit_receipt(
                preview_token=commit_args["preview_token"],
                idempotency_key=commit_args["idempotency_key"],
                session_id="session-write-taint",
                message_id="msg-write-taint",
            )
        )

    def observe_preview(_path, _payload, **_kwargs):
        observed.append(("preview", mystand_private_query_turn_active()))
        return '{"ok":true}'

    monkeypatch.setattr(
        bridge,
        "_mystand_authorization_write_operation_handler",
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


def test_schema_requires_one_idempotency_key_across_preview_and_commit():
    parameters = bridge.MYSTAND_AUTHORIZATION_WRITE_SCHEMA["parameters"]

    key_description = parameters["properties"]["idempotency_key"].get("description", "")
    assert "REQUIRED for both preview_write and commit_write" in key_description
    assert "fresh unique key" in key_description
    assert "commit_write MUST reuse exactly the same key" in key_description
    assert "retries of that logical write" in key_description
    assert "Never reuse" in key_description
    tool_description = bridge.MYSTAND_AUTHORIZATION_WRITE_SCHEMA["description"]
    assert "Both preview_write and commit_write REQUIRE idempotency_key" in tool_description
    assert "reuse exactly that same key" in tool_description
    operation_description = parameters["properties"]["operation"]["description"]
    assert "preview_write requires" in operation_description
    assert "commit_write requires preview_token" in operation_description
    assert "same idempotency_key" in operation_description
    assert parameters["required"] == ["operation"]


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


def test_commit_hard_fails_closed_without_preview_idempotency_key(monkeypatch):
    monkeypatch.setattr(bridge, "mark_mystand_private_query_turn", lambda: None)
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
        "_mystand_authorization_write_operation_handler",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "{}",
    )

    result = json.loads(
        bridge.mystand_authorization_write_tool_handler(
            {
                "operation": "commit_write",
                "preview_token": "preview-token",
            }
        )
    )

    assert result["ok"] is False
    assert result["status"] == 400
    assert result["code"] == "authorization_argument_missing"
    assert "idempotency_key" in result["error"]
    assert "禁止向用户声称已经写入" in result["integrity_notice"]
    assert calls == [], "缺少预览使用的 idempotency_key 时不得触达提交接口"


@pytest.mark.parametrize(
    "upstream_result",
    [
        pytest.param(
            {
                "ok": True,
                "status": 200,
                "verified": True,
                "receiptVersion": "authorization-write-receipt-v2",
            },
            id="minimal-three-field-v2-is-not-a-valid-receipt",
        ),
        pytest.param(
            {**_valid_commit_receipt(), "verified": False},
            id="verified-false",
        ),
        pytest.param(
            {
                **_valid_commit_receipt(),
                "receiptVersion": "authorization-write-receipt-v1",
            },
            id="wrong-receipt-version",
        ),
    ],
)
def test_commit_ok_without_verified_v2_receipt_is_forced_to_failure(
    monkeypatch,
    upstream_result,
):
    result = _call_stubbed_commit(monkeypatch, upstream_result)

    assert result["ok"] is False
    assert result["status"] == 502
    assert result["code"] == "authorization_write_receipt_invalid"
    assert "完整且与本次确认绑定" in result["error"]
    assert "verified=true" in result["error"]
    assert "authorization-write-receipt-v2" in result["error"]
    assert "禁止向用户声称已经写入" in result["integrity_notice"]


@pytest.mark.parametrize(
    "missing_field",
    [
        "audit",
        "audit.auditId",
        "confirmationId",
        "action",
        "target",
        "expectedVersion",
        "nextVersion",
        "idempotencyKey",
        "requestFingerprint",
        "previewTokenHash",
        "changeDigest",
        "committedAt",
    ],
)
def test_commit_receipt_missing_required_structure_is_forced_to_failure(
    monkeypatch,
    missing_field,
):
    receipt = _valid_commit_receipt()
    if missing_field == "audit.auditId":
        receipt["audit"].pop("auditId")
    else:
        receipt.pop(missing_field)

    result = _call_stubbed_commit(monkeypatch, receipt)

    assert result["ok"] is False
    assert result["status"] == 502
    assert result["code"] == "authorization_write_receipt_invalid"
    assert "完整且与本次确认绑定" in result["error"]


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        pytest.param(
            "idempotencyKey",
            "different-logical-write-key",
            id="wrong-idempotency-binding",
        ),
        pytest.param(
            "previewTokenHash",
            "d" * 64,
            id="wrong-json-stringified-preview-token-hash",
        ),
        pytest.param(
            "confirmationId",
            f"xiaoban-write-{'e' * 64}",
            id="wrong-confirmation-context-binding",
        ),
    ],
)
def test_commit_receipt_wrong_request_binding_is_forced_to_failure(
    monkeypatch,
    field,
    wrong_value,
):
    receipt = _valid_commit_receipt()
    receipt[field] = wrong_value

    result = _call_stubbed_commit(monkeypatch, receipt)

    assert result["ok"] is False
    assert result["status"] == 502
    assert result["code"] == "authorization_write_receipt_invalid"
    assert "完整且与本次确认绑定" in result["error"]
