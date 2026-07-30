"""Local-only contracts for dynamic-evidence-v2 completion."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _finalize_mystand_egress_result,
    _install_mystand_completion_persistence_guard,
    _mystand_completion_expected_binding,
    _mystand_dynamic_evidence_required,
)
from gateway.platforms.mystand_egress_seal import (
    is_mystand_egress_sealed,
)
from gateway.platforms.true_moa_idempotency import _IdempotencyCache
from gateway.platforms.true_moa_runner import _mystand_index_followup_tool
from tools import mystand_query_tool
from xiaoban.trusted_runtime import (
    EvidenceEnvelope,
    TrustedIdentity,
    activate_turn,
    begin_action,
    begin_turn,
    check_completion,
    deactivate_turn,
    finish_action,
)
from xiaoban.trusted_runtime.true_moa_durable import (
    TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
    TRUE_MOA_OUTCOME_BINDING_SCHEMA,
    TrueMoAOutcomeBindingError,
    project_true_moa_completed_outcome,
)
from xiaoban.trusted_runtime.dynamic_completion import (
    dynamic_finalization_mode,
)
from xiaoban.trusted_runtime.paid_call_policy import (
    SIGNED_MYSTAND_AGENT_POLICY_REVISION,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
)


PROTOCOL = "dynamic-evidence-v2"
DELIVERY_ID = "xbd_" + ("a" * 40)
MESSAGE_ID = "message-v2"
SESSION_ID = "session-v2"
REQUEST_FINGERPRINT = "b" * 64
INVOCATION_FINGERPRINT = "c" * 64
IDENTITY = TrustedIdentity(
    account_id="owner-v2",
    data_scope="mystand",
    source="server_session",
)


def _binding(*, attempt: int = 1) -> dict:
    return {
        "user_id": IDENTITY.account_id,
        "session_id": SESSION_ID,
        "delivery_id": DELIVERY_ID,
        "attempt": attempt,
        "message_id": MESSAGE_ID,
        "request_fingerprint": REQUEST_FINGERPRINT,
        "invocation_fingerprint": INVOCATION_FINGERPRINT,
        "datascope_fingerprint": IDENTITY.datascope_fingerprint,
    }


def _turn(*, attempt: int = 1, evidence_required: bool = False):
    return begin_turn(
        channel="web",
        user_message="这套房有车位吗",
        identity=IDENTITY,
        request_id=DELIVERY_ID,
        message_id=MESSAGE_ID,
        evidence_required=evidence_required,
        completion_protocol=PROTOCOL,
        completion_binding=_binding(attempt=attempt),
    )


def _index_item(
    resource_uid: str,
    safe_label: str,
    *,
    resource_type: str = "property-md",
) -> dict:
    return {
        "resourceUid": resource_uid,
        "moduleId": "property",
        "resourceType": resource_type,
        "parentResourceUid": "",
        "safeLabel": safe_label,
        "encrypted": False,
        "status": "active",
        "locked": False,
        "canRead": True,
        "canWrite": False,
    }


def _record(turn, action_id: str, arguments: dict, payload: dict, call_id: str):
    decision = begin_action(
        turn,
        action_id,
        "v1",
        arguments,
        call_id=call_id,
    )
    assert decision.decision == "allow"
    result = finish_action(
        turn,
        call_id,
        action_id,
        "v1",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )
    assert result is not None
    return result


def _record_index(turn, items: list[dict], *, has_more: bool = False) -> None:
    _record(
        turn,
        "mystand_resource_index",
        {},
        {
            "schema": "mystand.resource-index.complete.v1",
            "ok": True,
            "items": items,
            "hasMore": has_more,
            "nextCursor": "next" if has_more else "",
        },
        "call-index",
    )


def _query_arguments() -> dict:
    return {
        "operation": "read",
        "resource": {
            "name": "中海城南一号2-1-1001",
            "type_hint": "property-md",
        },
        "entities": [],
        "fact_needs": ["property.parking"],
        "mode": "facts",
    }


def _query_payload(
    *,
    resource_uid: str | None = None,
    record_refs: list[str] | None = None,
    facts: list[dict] | None = None,
) -> dict:
    resource = {
        "display_name": "中海城南一号2-1-1001",
        "type": "property-md",
    }
    if resource_uid is not None:
        resource["resourceUid"] = resource_uid
    payload = {
        "schema": "mystand.query-result.v1",
        "ok": True,
        "status": "matched",
        "missing_facts": [],
        "resource": resource,
        "facts": facts or [
            {
                "kind": "property.parking",
                "label": "车位",
                "value": {"available": True},
            }
        ],
        # This raw field must never be projected by the v2 completion.
        "content": "客户电话 13800000000；原始私密正文",
    }
    if record_refs is not None:
        payload["recordRefs"] = record_refs
    return payload


def test_dynamic_parking_projects_only_structured_fact_and_full_receipt():
    turn = _turn()
    _record_index(
        turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        turn,
        "mystand_query",
        _query_arguments(),
        _query_payload(),
        "call-query",
    )

    model_reply = "我查到了这套房的资料，资料里记录的是有车位。"
    decision = check_completion(model_reply, turn)

    assert decision.allowed is True
    assert decision.text == model_reply
    assert "13800000000" not in decision.text
    assert decision.verification is not None
    assert decision.verification["schema"] == (
        "mystand.xiaoban-completion-verification.v2"
    )
    assert decision.verification["completion_kind"] == "evidence-bound"
    assert decision.verification["binding_verified"] is True
    assert decision.verification["semantic_verified"] is False
    assert "verified" not in decision.verification
    assert decision.verification["action_count"] == 2
    assert decision.verification["evidence_count"] == 1
    assert decision.verification["record_refs"] == ["res-selected"]
    assert decision.verification["index_has_more"] is False


def test_explicit_linked_record_refs_are_nonempty_complete_index_subset():
    turn = _turn()
    _record_index(
        turn,
        [
            _index_item("res-linked", "关联房源笔记", resource_type="property-note"),
            _index_item("res-selected", "中海城南一号2-1-1001"),
        ],
    )
    _record(
        turn,
        "mystand_query",
        _query_arguments(),
        _query_payload(
            resource_uid="res-selected",
            record_refs=["res-linked", "res-selected"],
        ),
        "call-query",
    )

    decision = check_completion("ignored", turn)

    assert decision.allowed is True
    assert decision.verification["record_refs"] == [
        "res-linked",
        "res-selected",
    ]
    assert set(decision.verification["record_refs"]).issubset(
        set(turn.index_receipt.matched_resource_refs)
    )

    blocked_turn = _turn()
    locked_link = _index_item(
        "res-linked",
        "关联房源笔记",
        resource_type="property-note",
    )
    locked_link.update({"status": "locked", "locked": True, "canRead": False})
    _record_index(
        blocked_turn,
        [
            locked_link,
            _index_item("res-selected", "中海城南一号2-1-1001"),
        ],
    )
    _record(
        blocked_turn,
        "mystand_query",
        _query_arguments(),
        _query_payload(
            resource_uid="res-selected",
            record_refs=["res-linked", "res-selected"],
        ),
        "call-query",
    )
    blocked = check_completion("ignored", blocked_turn)
    assert blocked.allowed is False
    assert blocked.verification is None

    foreign_turn = _turn()
    _record_index(
        foreign_turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        foreign_turn,
        "mystand_query",
        _query_arguments(),
        _query_payload(
            resource_uid="res-selected",
            record_refs=["res-foreign", "res-selected"],
        ),
        "call-query",
    )
    foreign = check_completion("不能静默丢掉未索引引用", foreign_turn)
    assert foreign.allowed is False
    assert foreign.verification is None

    malformed_turn = _turn()
    _record_index(
        malformed_turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        malformed_turn,
        "mystand_query",
        _query_arguments(),
        _query_payload(
            resource_uid="res-selected",
            record_refs=["res-selected", {"resourceUid": "res-hidden"}],
        ),
        "call-query",
    )
    malformed = check_completion(
        "不能静默忽略格式错误的引用",
        malformed_turn,
    )
    assert malformed.allowed is False
    assert malformed.verification is None


def test_argument_resource_uid_must_match_result_primary_resource():
    turn = _turn()
    _record_index(
        turn,
        [
            _index_item("res-a", "资料甲"),
            _index_item("res-b", "资料乙"),
        ],
    )
    _record(
        turn,
        "mystand_authorization",
        {"operation": "resolve", "resource_uid": "res-a"},
        {
            "ok": True,
            "content": "资料乙的内容",
            "resourceUid": "res-b",
        },
        "call-auth-mismatch",
    )

    decision = check_completion("不能把甲的请求绑定到乙的结果", turn)

    assert decision.allowed is False
    assert decision.verification is None


@pytest.mark.parametrize(
    "facts",
    [
        [
            {
                "kind": "property.parking",
                "label": "车位",
                "value": {"available": True},
            },
            {
                "kind": "property.parking",
                "label": "车位",
                "value": {"available": False},
            },
        ],
        [{"kind": "property.parking", "label": "车位", "value": "有"}],
        [
            {
                "kind": "property.parking",
                "label": "车位",
                "value": {"available": True, "source": "raw"},
            }
        ],
    ],
)
def test_dynamic_gate_does_not_hardcode_module_specific_fact_shapes(facts):
    turn = _turn()
    _record_index(
        turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        turn,
        "mystand_query",
        _query_arguments(),
        _query_payload(facts=facts),
        "call-query",
    )

    model_reply = "我已读取资料，并会按资料原文说明。"
    decision = check_completion(model_reply, turn)

    assert decision.allowed is True
    assert decision.text == model_reply
    assert decision.verification["semantic_verified"] is False


def test_v2_capability_does_not_change_chat_or_unrelated_evidence():
    chat_turn = _turn()
    chat = check_completion("正常聊天回答", chat_turn)
    assert chat.allowed is True
    assert chat.text == "正常聊天回答"
    assert chat.verification is None

    evidence_turn = _turn()
    evidence_turn.interaction_kind = "WORK"
    evidence_turn.evidence.append(
        EvidenceEnvelope(
            evidence_id="web-evidence",
            turn_id=evidence_turn.turn_id,
            call_id="web-call",
            action_id="web_extract",
            datascope_fingerprint=IDENTITY.datascope_fingerprint,
            status="success",
            allowed_facts=json.dumps({"content": "网页原投影"}),
            record_refs=[],
            input_digest="d" * 64,
            output_digest="e" * 64,
            verified_at="1",
            verification_status="verified",
        )
    )
    web = check_completion("模型网页回答", evidence_turn)
    assert web.allowed is True
    assert web.text == "网页原投影"
    assert web.verification is None


def test_v2_authorization_read_is_a_valid_registered_read_chain():
    dynamic_turn = _turn()
    _record_index(
        dynamic_turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        dynamic_turn,
        "mystand_authorization",
        {"operation": "resolve", "resource_uid": "res-selected"},
        {
            "ok": True,
            "content": "legacy raw content",
            "resourceUid": "res-selected",
        },
        "call-auth",
    )
    model_reply = "我已从这份授权资料中读到对应内容。"
    dynamic = check_completion(model_reply, dynamic_turn)
    assert dynamic.allowed is True
    assert dynamic.text == model_reply
    assert dynamic.verification is not None
    assert dynamic.verification["evidence_count"] == 1
    assert dynamic.verification["record_refs"] == ["res-selected"]

    legacy_turn = begin_turn(
        channel="web",
        user_message="读取 AUTH-EXACT123",
        identity=IDENTITY,
        request_id="legacy-request",
        message_id="legacy-message",
    )
    _record_index(
        legacy_turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        legacy_turn,
        "mystand_authorization",
        {"operation": "resolve", "resource_uid": "res-selected"},
        {
            "ok": True,
            "content": "legacy raw content",
            "resourceUid": "res-selected",
        },
        "call-auth",
    )
    legacy = check_completion("模型原文", legacy_turn)
    assert legacy.allowed is True
    assert legacy.text == "legacy raw content"
    assert legacy.verification is None


def test_multiple_registered_reads_share_one_verified_index_chain():
    turn = _turn()
    _record_index(
        turn,
        [
            _index_item("res-a", "资料甲"),
            _index_item("res-b", "资料乙"),
        ],
    )
    for suffix in ("a", "b"):
        _record(
            turn,
            "mystand_authorization",
            {"operation": "resolve", "resource_uid": f"res-{suffix}"},
            {
                "ok": True,
                "content": f"资料{suffix}的真实内容",
                "resourceUid": f"res-{suffix}",
            },
            f"call-auth-{suffix}",
        )

    model_reply = "我已读取两份目标资料，并根据它们一起说明。"
    decision = check_completion(model_reply, turn)

    assert decision.allowed is True
    assert decision.text == model_reply
    assert decision.verification["action_count"] == 3
    assert decision.verification["evidence_count"] == 2
    assert decision.verification["record_refs"] == ["res-a", "res-b"]


def test_ordinary_failed_attempt_does_not_poison_later_valid_read_chain():
    turn = _turn()
    denied = begin_action(
        turn,
        "mystand_query",
        "v1",
        {
            "operation": "read",
            "query_kind": "resource-read",
        },
        call_id="call-denied",
    )
    assert denied.decision == "deny"
    assert denied.reason == "missing_index_receipt"
    _record_index(
        turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        turn,
        "mystand_authorization",
        {"operation": "resolve", "resource_uid": "res-selected"},
        {
            "ok": True,
            "content": "真实读取内容",
            "resourceUid": "res-selected",
        },
        "call-auth-success",
    )

    model_reply = "前一次定位方式不合适，但随后我已从目标资料中读取成功。"
    decision = check_completion(model_reply, turn)

    assert turn.pre_action_denials == 1
    assert decision.allowed is True
    assert decision.text == model_reply
    assert decision.verification["evidence_count"] == 1
    assert decision.verification["action_count"] == 2


def test_query_kind_hint_is_scoped_to_dynamic_protocol():
    dynamic_turn = _turn()
    _record_index(
        dynamic_turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    dynamic = begin_action(
        dynamic_turn,
        "mystand_query",
        "v1",
        {
            **_query_arguments(),
            "query_kind": "resource-read",
        },
        call_id="call-dynamic-query-kind",
    )
    assert dynamic.decision == "allow"

    legacy_turn = begin_turn(
        channel="web",
        user_message="读取资料",
        identity=IDENTITY,
        request_id="request-legacy-query-kind",
        message_id="message-legacy-query-kind",
    )
    _record_index(
        legacy_turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    legacy = begin_action(
        legacy_turn,
        "mystand_query",
        "v1",
        {
            **_query_arguments(),
            "query_kind": "resource-read",
        },
        call_id="call-legacy-query-kind",
    )
    assert legacy.decision == "deny"
    assert legacy.reason == "unbound_fact_query_plan"


def test_failure_bound_reply_requires_finalize_only_runtime_state():
    turn = _turn()
    _record_index(
        turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        turn,
        "mystand_query",
        _query_arguments(),
        {
            "ok": False,
            "status": 404,
            "code": "resource_not_found",
            "error": "没有找到匹配资料",
        },
        "call-query-missing",
    )
    model_reply = "我这次没有找到能够唯一匹配的资料，需要你补充更准确的名称。"

    blocked = check_completion(model_reply, turn)
    assert blocked.allowed is False
    assert blocked.verification is None

    turn.completion_finalization = "failure"
    turn.completion_finalization_output_digest = hashlib.sha256(
        model_reply.encode("utf-8")
    ).hexdigest()
    allowed = check_completion(model_reply, turn)
    assert allowed.allowed is True
    assert allowed.text == model_reply
    assert allowed.verification is not None
    assert allowed.verification["completion_kind"] == "failure-bound"
    assert allowed.verification["semantic_verified"] is False
    assert allowed.verification["evidence_count"] == 0
    assert allowed.verification["action_result_digest"]
    assert allowed.verification["failed_action_count"] == 1
    assert allowed.verification["failure_class"] == "not_found"

    turn.completion_finalization_output_digest = hashlib.sha256(
        "另一个未绑定回复".encode("utf-8")
    ).hexdigest()
    tampered = check_completion(model_reply, turn)
    assert tampered.allowed is False
    assert tampered.verification is None


def test_dynamic_finalization_uses_only_real_dispatched_failure():
    failed = _turn()
    _record(
        failed,
        "mystand_resource_index",
        {},
        {
            "ok": False,
            "status": 503,
            "code": "index_unavailable",
        },
        "call-index-failed",
    )
    assert dynamic_finalization_mode(failed) == "failure"

    denied = _turn()
    first = begin_action(
        denied,
        "mystand_query",
        "v1",
        _query_arguments(),
        call_id="call-denied-first",
    )
    assert first.reason == "missing_index_receipt"
    assert dynamic_finalization_mode(denied) == ""
    assert dynamic_finalization_mode(
        denied,
        include_single_preaction=True,
    ) == ""
    second = begin_action(
        denied,
        "mystand_query",
        "v1",
        _query_arguments(),
        call_id="call-denied-second",
    )
    assert second.reason == "missing_index_receipt"
    assert dynamic_finalization_mode(denied) == ""


def test_completion_attempt_must_be_positive_and_dual_headers_must_match():
    with pytest.raises(ValueError):
        _turn(attempt=0)

    headers = {
        "X-Xiaoban-User-Id": IDENTITY.account_id,
        "X-Xiaoban-Message-Id": MESSAGE_ID,
        "X-Xiaoban-Delivery-Id": DELIVERY_ID,
        "X-Xiaoban-Attempt": "1",
        "X-Xiaoban-Delivery-Attempt": "2",
        "X-Xiaoban-Request-Fingerprint": REQUEST_FINGERPRINT,
        "X-Xiaoban-Invocation-Fingerprint": INVOCATION_FINGERPRINT,
    }
    with pytest.raises(ValueError):
        _mystand_completion_expected_binding(
            headers,
            session_id=SESSION_ID,
        )


def test_dynamic_evidence_requirement_is_explicit_and_protocol_bound():
    assert _mystand_dynamic_evidence_required(
        {"X-Xiaoban-Evidence-Required": "0"},
        completion_protocol=PROTOCOL,
    ) is False
    assert _mystand_dynamic_evidence_required(
        {"X-Xiaoban-Evidence-Required": "1"},
        completion_protocol=PROTOCOL,
    ) is True
    with pytest.raises(ValueError):
        _mystand_dynamic_evidence_required(
            {},
            completion_protocol=PROTOCOL,
        )
    with pytest.raises(ValueError):
        _mystand_dynamic_evidence_required(
            {"X-Xiaoban-Evidence-Required": "true"},
            completion_protocol=PROTOCOL,
        )
    with pytest.raises(ValueError):
        _mystand_dynamic_evidence_required(
            {"X-Xiaoban-Evidence-Required": "1"},
            completion_protocol="",
        )


class _RetryFenceAgent:
    provider = "deepseek"
    model = "deepseek-v4-pro"
    valid_tool_names: set[str] = set()
    tools: list[object] = []
    session_prompt_tokens = 2
    session_completion_tokens = 1
    session_total_tokens = 3
    session_id = SESSION_ID

    def __init__(self) -> None:
        self.ephemeral_system_prompt = ""

    def run_conversation(self, **_kwargs):
        return {
            "final_response": "普通回复",
            "completed": True,
            "failed": False,
            "messages": [],
        }


def _normal_request_headers() -> dict[str, str]:
    return {
        "X-Xiaoban-User-Id": IDENTITY.account_id,
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Memory-Mode": "disabled",
        "X-Xiaoban-Message-Id": MESSAGE_ID,
        "X-Xiaoban-Delivery-Id": DELIVERY_ID,
        SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER: (
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
        ),
    }


@pytest.mark.asyncio
async def test_normal_dynamic_evidence_uses_strict_paid_call_fence(
    monkeypatch,
):
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-test-only"}),
    )
    create_kwargs: dict[str, object] = {}
    headers = _normal_request_headers()
    headers.update(
        {
            "X-Xiaoban-Delivery-Id": DELIVERY_ID,
            "X-Xiaoban-Attempt": "1",
            "X-Xiaoban-Delivery-Attempt": "1",
            "X-Xiaoban-Request-Fingerprint": REQUEST_FINGERPRINT,
            "X-Xiaoban-Invocation-Fingerprint": INVOCATION_FINGERPRINT,
        },
    )

    def _fake_create_agent(**kwargs):
        create_kwargs.update(kwargs)
        return _RetryFenceAgent()

    monkeypatch.setattr(adapter, "_create_agent", _fake_create_agent)
    result, _usage = await adapter._run_agent(
        user_message="只聊一句，不查资料",
        conversation_history=[],
        session_id=SESSION_ID,
        request_headers=headers,
        completion_protocol=PROTOCOL,
        completion_binding=_mystand_completion_expected_binding(
            headers,
            session_id=SESSION_ID,
        ),
    )

    assert result["completed"] is True
    assert create_kwargs["strict_no_automatic_paid_retry"] is True


@pytest.mark.asyncio
async def test_server_work_marker_blocks_zero_call_claim_and_chat_stays_visible(
    monkeypatch,
):
    from gateway.platforms import api_server

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-test-only"}),
    )
    headers = _normal_request_headers()
    headers.update(
        {
            "X-Xiaoban-Attempt": "1",
            "X-Xiaoban-Delivery-Attempt": "1",
            "X-Xiaoban-Request-Fingerprint": REQUEST_FINGERPRINT,
            "X-Xiaoban-Invocation-Fingerprint": INVOCATION_FINGERPRINT,
        },
    )
    binding = _mystand_completion_expected_binding(
        headers,
        session_id=SESSION_ID,
    )
    neutral_message = "请处理这件事"
    preexecuted: list[str] = []

    def _fake_create_agent(**_kwargs):
        agent = _RetryFenceAgent()
        return agent

    def _empty_preexecution(initial_tool_choice, **_kwargs):
        preexecuted.append(initial_tool_choice)
        return []

    monkeypatch.setattr(adapter, "_create_agent", _fake_create_agent)
    monkeypatch.setattr(
        api_server,
        "_run_mystand_preexecuted_evidence",
        _empty_preexecution,
    )

    work_result, _usage = await adapter._run_agent(
        user_message=neutral_message,
        conversation_history=[],
        session_id=SESSION_ID,
        request_headers=headers,
        completion_protocol=PROTOCOL,
        completion_binding=binding,
        dynamic_evidence_required=True,
    )
    work_text = _finalize_mystand_egress_result(
        work_result,
        user_message=neutral_message,
        conversation_history=[],
    )

    assert preexecuted == []
    assert work_result["_mystand_evidence_required"] is True
    assert work_result["_trusted_turn"].interaction_kind == "WORK"
    assert work_text == ""
    assert work_result["completed"] is False
    assert work_result["failed"] is True
    assert work_result["messages"] == []
    assert "_mystand_trusted_verification" not in work_result

    chat_result, _usage = await adapter._run_agent(
        user_message=neutral_message,
        conversation_history=[],
        session_id=SESSION_ID,
        request_headers=headers,
        completion_protocol=PROTOCOL,
        completion_binding=binding,
        dynamic_evidence_required=False,
    )
    chat_text = _finalize_mystand_egress_result(
        chat_result,
        user_message=neutral_message,
        conversation_history=[],
    )

    assert preexecuted == []
    assert chat_result["_mystand_evidence_required"] is False
    assert chat_result["_trusted_turn"].interaction_kind == "CHAT"
    assert chat_text == "普通回复"


@pytest.mark.asyncio
async def test_normal_signed_chat_uses_one_dispatch_per_durable_receipt(
    monkeypatch,
):
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-test-only"}),
    )
    create_kwargs: dict[str, object] = {}

    def _fake_create_agent(**kwargs):
        create_kwargs.update(kwargs)
        return _RetryFenceAgent()

    monkeypatch.setattr(adapter, "_create_agent", _fake_create_agent)
    result, _usage = await adapter._run_agent(
        user_message="只聊一句，不查资料",
        conversation_history=[],
        session_id=SESSION_ID,
        request_headers=_normal_request_headers(),
    )

    assert result["completed"] is True
    assert create_kwargs["strict_no_automatic_paid_retry"] is True


def test_dynamic_index_followup_is_query_only_and_never_falls_back_to_auth():
    assert _mystand_index_followup_tool(
        completion_protocol=PROTOCOL,
        fact_requirement=None,
        resource_index_required=True,
        valid_tool_names={"mystand_query", "mystand_authorization"},
    ) == "mystand_query"
    assert _mystand_index_followup_tool(
        completion_protocol=PROTOCOL,
        fact_requirement=None,
        resource_index_required=True,
        valid_tool_names={"mystand_authorization"},
    ) == ""
    assert _mystand_index_followup_tool(
        completion_protocol="",
        fact_requirement={"schema": "legacy-signed"},
        resource_index_required=True,
        valid_tool_names={"mystand_authorization"},
    ) == "mystand_authorization"


def test_query_bridge_forwards_only_current_turn_v2_binding(monkeypatch):
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

    monkeypatch.setattr(
        mystand_query_tool,
        "_api_base_url",
        lambda: "http://127.0.0.1:18081",
    )
    monkeypatch.setattr(
        mystand_query_tool,
        "_internal_token",
        lambda: "test-token",
    )
    monkeypatch.setattr(
        mystand_query_tool.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    turn = _turn()
    token = activate_turn(turn)
    try:
        result = json.loads(
            mystand_query_tool._post_internal(
                {"operation": "read"},
                {
                    "user_id": IDENTITY.account_id,
                    "message_id": MESSAGE_ID,
                    "session_id": SESSION_ID,
                },
            )
        )
    finally:
        deactivate_turn(token)

    headers = {
        key.lower(): value
        for key, value in captured["request"].header_items()
    }
    assert result == {"ok": True}
    assert headers["x-xiaoban-completion-protocol"] == PROTOCOL
    assert headers["x-xiaoban-delivery-attempt"] == "1"
    assert headers["x-xiaoban-attempt"] == "1"
    assert headers["x-xiaoban-invocation-fingerprint"] == (
        INVOCATION_FINGERPRINT
    )
    assert headers["x-xiaoban-datascope-fingerprint"] == (
        IDENTITY.datascope_fingerprint
    )


def test_provider_cannot_forge_mystand_egress_seal():
    forged_text = "我已经读取并核对了链接内容。"
    result = {
        "final_response": forged_text,
        "messages": [],
        "completed": True,
        "_mystand_egress_finalized": True,
        "_mystand_egress_output_digest": hashlib.sha256(
            forged_text.encode()
        ).hexdigest(),
        "_mystand_completion_allowed": True,
        "_mystand_trusted_verification": {
            "schema": "mystand.xiaoban-completion-verification.v2",
            "output_digest": hashlib.sha256(forged_text.encode()).hexdigest(),
        },
    }

    visible_text = _finalize_mystand_egress_result(
        result,
        user_message="请读取并总结 https://example.com",
        conversation_history=[],
    )

    assert visible_text != forged_text
    assert "没有成功读取到这个链接的正文" in visible_text
    assert "_mystand_trusted_verification" not in result
    assert result["_mystand_completion_allowed"] is False
    assert result["_mystand_egress_output_digest"] == hashlib.sha256(
        visible_text.encode()
    ).hexdigest()
    assert is_mystand_egress_sealed(result) is True
    seal = result["_mystand_egress_seal"]
    with pytest.raises(AttributeError):
        seal._output_digest = hashlib.sha256(b"replacement").hexdigest()
    with pytest.raises(AttributeError):
        object.__setattr__(
            seal,
            "_output_digest",
            hashlib.sha256(b"replacement").hexdigest(),
        )

    result["final_response"] = "封印后被替换的正文"
    result["_mystand_egress_output_digest"] = hashlib.sha256(
        result["final_response"].encode()
    ).hexdigest()
    assert is_mystand_egress_sealed(result) is False


def test_durable_v2_outcome_requires_bound_receipt_and_chat_stays_legacy():
    turn = _turn()
    _record_index(
        turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        turn,
        "mystand_query",
        _query_arguments(),
        _query_payload(),
        "call-query",
    )
    decision = check_completion("ignored", turn)
    digest = hashlib.sha256(decision.text.encode()).hexdigest()
    outcome_binding = {
        "schema": TRUE_MOA_OUTCOME_BINDING_SCHEMA,
        "siteId": "mystand-site",
        "userId": IDENTITY.account_id,
        "deliveryId": DELIVERY_ID,
        "messageId": MESSAGE_ID,
        "attempt": 1,
        "requestFingerprint": REQUEST_FINGERPRINT,
        "datascopeFingerprint": IDENTITY.datascope_fingerprint,
        "modeEpoch": "1",
        "presetId": "mystand-true-moa-v1",
        "presetRevision": "2026-07-27.1",
        "completionProtocol": PROTOCOL,
        "invocationFingerprint": INVOCATION_FINGERPRINT,
    }
    outcome = {
        "schema": TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
        "completed": True,
        "finalResponse": decision.text,
        "outputDigest": digest,
        "factGuardRequired": False,
        "completionProtocol": PROTOCOL,
        "trustedVerification": decision.verification,
    }
    projected = project_true_moa_completed_outcome(
        outcome,
        binding=outcome_binding,
    )
    assert projected["completionProtocol"] == PROTOCOL

    tampered = json.loads(json.dumps(outcome))
    tampered["trustedVerification"]["invocation_fingerprint"] = "0" * 64
    with pytest.raises(TrueMoAOutcomeBindingError):
        project_true_moa_completed_outcome(
            tampered,
            binding=outcome_binding,
        )

    chat_result = {
        "final_response": "普通真 MoA 聊天",
        "messages": [],
        "completed": True,
        "failed": False,
        "_mystand_completion_protocol": PROTOCOL,
        "_trusted_turn": _turn(),
    }
    _finalize_mystand_egress_result(
        chat_result,
        user_message="聊聊天",
        conversation_history=[],
    )
    chat_payload = _IdempotencyCache._completed_outcome_payload(chat_result)
    assert "completionProtocol" not in chat_payload
    assert "trustedVerification" not in chat_payload


def test_dynamic_completion_bypasses_legacy_fixed_tool_group_gate():
    turn = _turn()
    _record_index(
        turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    _record(
        turn,
        "mystand_authorization",
        {"operation": "resolve", "resource_uid": "res-selected"},
        {
            "ok": True,
            "content": "真实读取内容",
            "resourceUid": "res-selected",
        },
        "call-auth",
    )
    model_reply = "我已从目标资料中读取成功，下面按资料内容说明。"
    result = {
        "final_response": model_reply,
        "messages": [],
        "completed": True,
        "failed": False,
        "_mystand_request": True,
        "_mystand_user_id": IDENTITY.account_id,
        "_mystand_request_id": DELIVERY_ID,
        "_mystand_message_id": MESSAGE_ID,
        "_mystand_completion_protocol": PROTOCOL,
        "_mystand_completion_binding": dict(turn.completion_binding),
        "_mystand_required_evidence_groups": [["mystand_query"]],
        "_trusted_turn": turn,
    }

    final_text = _finalize_mystand_egress_result(
        result,
        user_message="读取目标资料",
        conversation_history=[],
    )

    assert final_text == model_reply
    assert result["_mystand_trusted_verification"]["completion_kind"] == (
        "evidence-bound"
    )


def test_zero_call_dynamic_claim_cannot_bypass_legacy_required_group():
    turn = _turn()
    model_reply = "我已经查到资料，答案是肯定的。"
    result = {
        "final_response": model_reply,
        "messages": [],
        "completed": True,
        "failed": False,
        "_mystand_request": True,
        "_mystand_user_id": IDENTITY.account_id,
        "_mystand_request_id": DELIVERY_ID,
        "_mystand_message_id": MESSAGE_ID,
        "_mystand_completion_protocol": PROTOCOL,
        "_mystand_completion_binding": dict(turn.completion_binding),
        "_mystand_required_evidence_groups": [["mystand_query"]],
        "_trusted_turn": turn,
    }

    final_text = _finalize_mystand_egress_result(
        result,
        user_message="读取目标资料",
        conversation_history=[],
    )

    assert final_text != model_reply
    assert "_mystand_trusted_verification" not in result


def test_finalize_only_not_executed_claim_is_blocked_without_legacy_group():
    turn = _turn()
    turn.completion_finalization = "not_executed"
    model_reply = "我已经查到资料，答案是肯定的。"
    result = {
        "final_response": model_reply,
        "messages": [],
        "completed": True,
        "failed": False,
        "_mystand_request": True,
        "_mystand_user_id": IDENTITY.account_id,
        "_mystand_request_id": DELIVERY_ID,
        "_mystand_message_id": MESSAGE_ID,
        "_mystand_completion_protocol": PROTOCOL,
        "_mystand_completion_binding": dict(turn.completion_binding),
        "_trusted_turn": turn,
    }

    final_text = _finalize_mystand_egress_result(
        result,
        user_message="读取目标资料",
        conversation_history=[],
    )

    assert final_text != model_reply
    assert "_mystand_trusted_verification" not in result


def test_zero_call_work_claim_is_not_persisted_as_fixed_fallback():
    turn = _turn(evidence_required=True)
    persisted: list[list[dict]] = []
    agent = SimpleNamespace(
        _persist_session=lambda messages, _history=None: persisted.append(
            list(messages)
        )
    )
    _install_mystand_completion_persistence_guard(agent, turn)

    agent._persist_session(
        [
            {"role": "user", "content": "请处理这件事"},
            {
                "role": "assistant",
                "content": "我已经查到资料，答案是肯定的。",
            },
        ],
        [],
    )

    assert persisted == [[{"role": "user", "content": "请处理这件事"}]]


def test_failed_dynamic_read_delivers_model_failure_with_bound_receipt():
    turn = _turn()
    _record_index(
        turn,
        [_index_item("res-selected", "中海城南一号2-1-1001")],
    )
    query_call = begin_action(
        turn,
        "mystand_query",
        "v1",
        _query_arguments(),
        call_id="call-query-invalid",
    )
    assert query_call.decision == "allow"
    query_result = finish_action(
        turn,
        "call-query-invalid",
        "mystand_query",
        "v1",
        "{}",
    )
    assert query_result is not None
    assert query_result.status == "error"
    turn.completion_finalization = "failure"
    model_reply = "我这次没有读到有效资料，需要你补充更准确的名称。"
    turn.completion_finalization_output_digest = hashlib.sha256(
        model_reply.encode("utf-8")
    ).hexdigest()

    result = {
        "final_response": model_reply,
        "messages": [],
        "completed": True,
        "failed": False,
        "_mystand_request": True,
        "_mystand_user_id": IDENTITY.account_id,
        "_mystand_request_id": DELIVERY_ID,
        "_mystand_message_id": MESSAGE_ID,
        "_mystand_completion_protocol": PROTOCOL,
        "_mystand_completion_binding": dict(turn.completion_binding),
        "_trusted_turn": turn,
    }
    final_text = _finalize_mystand_egress_result(
        result,
        user_message="这套房有车位吗",
        conversation_history=[],
    )

    assert final_text == model_reply
    assert result["_mystand_completion_protocol"] == PROTOCOL
    verification = result["_mystand_trusted_verification"]
    assert verification["completion_kind"] == "failure-bound"
    assert verification["decision"] == "execution_status_bound"
    assert verification["failure_class"] == "error"
    assert verification["failed_action_count"] == 1
    payload = _IdempotencyCache._completed_outcome_payload(result)
    assert payload["trustedVerification"] == verification
