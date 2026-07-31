"""Regression tests for dynamic tool staging and finalize-only framing."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.conversation_loop import (
    _cap_dynamic_evidence_tool_calls,
    _contains_raw_tool_protocol_content,
    _prepare_finalize_only_call,
    _reject_finalize_only_protocol_candidate,
)
from agent.chat_completion_helpers import build_api_kwargs
from agent.transports.bedrock import BedrockTransport
from agent.tool_executor import _trusted_preaction_denial
from gateway.session_context import clear_session_vars, set_session_vars
from gateway.platforms.api_server import (
    APIServerAdapter,
    _run_mystand_preexecuted_evidence,
)
from tools.registry import ToolRegistry
from tools.mystand_query_tool import (
    MYSTAND_QUERY_SCHEMA,
    validate_mystand_semantic_query_plan,
)
from xiaoban.trusted_runtime import (
    TrustedIdentity,
    activate_turn,
    begin_action,
    begin_turn,
    deactivate_turn,
    finish_action,
)
from xiaoban.trusted_runtime.tool_visibility import (
    filter_dynamic_evidence_api_kwargs,
)
from xiaoban.trusted_runtime.dynamic_completion import (
    dynamic_failure_presentation,
    dynamic_finalization_mode,
    dynamic_turn_outcome,
    dynamic_transient_recovery_plan,
    dynamic_transient_recovery_tool_call_valid,
    mark_dynamic_execution_no_progress,
    mark_dynamic_read_no_progress,
    render_dynamic_failure_report,
)
from xiaoban.trusted_runtime.fact_contract import canonical_digest
from xiaoban.trusted_runtime.completion_guard import check_completion
from run_agent import AIAgent


IDENTITY = TrustedIdentity(
    account_id="owner-protocol",
    data_scope="mystand",
    source="server_session",
)
PROTOCOL = "dynamic-evidence-v2"


def _binding() -> dict:
    return {
        "user_id": IDENTITY.account_id,
        "session_id": "session-protocol",
        "delivery_id": "delivery-protocol",
        "attempt": 1,
        "message_id": "message-protocol",
        "request_fingerprint": "a" * 64,
        "invocation_fingerprint": "b" * 64,
        "datascope_fingerprint": IDENTITY.datascope_fingerprint,
    }


def _dynamic_turn(*, business_tools_disabled: bool = False):
    return begin_turn(
        channel="web",
        user_message="读取目标资料",
        identity=IDENTITY,
        request_id="delivery-protocol",
        message_id="message-protocol",
        evidence_required=True,
        completion_protocol=PROTOCOL,
        completion_binding=_binding(),
        business_tools_disabled=business_tools_disabled,
    )


def _tools(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _canonical_query_tool() -> dict:
    return {
        "type": "function",
        "function": json.loads(
            json.dumps(MYSTAND_QUERY_SCHEMA, ensure_ascii=False)
        ),
    }


def _tool_names(payload: dict) -> list[str]:
    return [item["function"]["name"] for item in payload["tools"]]


def _record_found_index(turn) -> None:
    decision = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="index-found",
    )
    assert decision.decision == "allow"
    finish_action(
        turn,
        decision.call.call_id,
        "mystand_resource_index",
        "v1",
        {
            "schema": "mystand.resource-index.complete.v1",
            "ok": True,
            "items": [
                {
                    "resourceUid": "resource-protocol",
                    "safeLabel": "目标资料",
                    "resourceType": "generic-record",
                    "canRead": True,
                    "locked": False,
                }
            ],
            "hasMore": False,
            "nextCursor": "",
        },
    )
    assert turn.index_receipt is not None
    assert turn.index_receipt.status == "found"


def _complete_bound_failure(turn, text: str):
    turn.completion_finalization = "failure"
    turn.completion_finalization_output_digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    return check_completion(text, turn)


def test_complete_index_without_read_enters_bound_natural_failure():
    turn = _dynamic_turn()
    _record_found_index(turn)

    fabricated = check_completion("我已经读完资料，结论正确。", turn)
    assert fabricated.allowed is False
    assert mark_dynamic_read_no_progress(turn) is True
    assert dynamic_finalization_mode(turn) == "failure"

    model_reply = (
        "我完成了资料目录查询，但没有继续读取到能回答问题的内容，"
        "所以这项任务还没有完成。"
    )
    turn.completion_finalization = "failure"
    turn.completion_finalization_output_digest = hashlib.sha256(
        model_reply.encode("utf-8")
    ).hexdigest()
    allowed = check_completion(model_reply, turn)

    assert allowed.allowed is True
    assert allowed.text == model_reply
    assert allowed.reason == "execution_status_bound"
    assert allowed.verification["completion_kind"] == "failure-bound"
    assert allowed.verification["failure_class"] == "no_progress"
    assert allowed.verification["failed_action_count"] == 0
    assert allowed.verification["action_count"] == 1
    assert allowed.verification["evidence_count"] == 0
    assert "output_presentation" not in allowed.verification
    assert allowed.verification["turn_outcome"]["attempt_count"] == 0

    natural_variant = (
        "抱歉，这次我完成了资料目录查询，但没能继续读到可回答的正文。"
        "因此这项任务还没完成。"
    )
    variant = _complete_bound_failure(turn, natural_variant)
    assert variant.allowed is True
    assert variant.text == natural_variant


def test_zero_action_work_enters_bound_natural_failure():
    turn = _dynamic_turn()

    assert mark_dynamic_execution_no_progress(turn) is True
    assert turn.completion_execution_failure == "action_not_dispatched"
    reply = "我这次没有发起实际处理，所以这项任务还没有完成。"
    allowed = _complete_bound_failure(turn, reply)

    assert allowed.allowed is True
    assert allowed.text == reply
    assert allowed.reason == "execution_status_bound"
    assert allowed.verification["failure_class"] == "no_progress"
    assert allowed.verification["action_count"] == 0
    assert allowed.verification["failed_action_count"] == 0

    natural_variant = (
        "这次我还没开始实际处理，因此我暂时无法完成这项任务。"
    )
    variant = _complete_bound_failure(turn, natural_variant)
    assert variant.allowed is True
    assert variant.text == natural_variant


def test_preaction_denial_enters_bound_natural_failure():
    turn = _dynamic_turn()
    denied = begin_action(
        turn,
        "mystand_query",
        "v1",
        {"operation": "read"},
        call_id="read-before-index",
    )
    assert denied.decision == "deny"
    assert denied.reason == "missing_index_receipt"

    assert mark_dynamic_execution_no_progress(turn) is True
    assert turn.completion_execution_failure == "read_precondition_not_met"
    reply = (
        "我这次没能先完成资料定位，所以正文读取没有发起，"
        "这项任务还没有完成。"
    )
    allowed = _complete_bound_failure(turn, reply)

    assert allowed.allowed is True
    assert allowed.verification["action_count"] == 0
    assert allowed.verification["failed_action_count"] == 0


def test_missing_action_result_enters_bound_natural_failure():
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="index-without-result",
    )
    assert started.decision == "allow"

    assert mark_dynamic_execution_no_progress(turn) is True
    assert turn.completion_execution_failure == "action_result_missing"
    reply = (
        "我这次的处理请求已经生成，但没有形成可以确认的结果，"
        "所以这项任务还没有完成。"
    )
    allowed = _complete_bound_failure(turn, reply)

    assert allowed.allowed is True
    assert allowed.verification["action_count"] == 1
    assert allowed.verification["failed_action_count"] == 0


def test_incomplete_index_receipt_enters_bound_natural_failure():
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="index-incomplete-page",
    )
    assert started.decision == "allow"
    result = finish_action(
        turn,
        started.call.call_id,
        "mystand_resource_index",
        "v1",
        {
            "schema": "mystand.resource-index.page.v1",
            "ok": True,
            "items": [
                {
                    "resourceUid": "resource-protocol",
                    "safeLabel": "目标资料",
                    "resourceType": "generic-record",
                    "canRead": True,
                    "locked": False,
                }
            ],
            "hasMore": True,
            "nextCursor": "next-page",
        },
    )
    assert result is not None
    assert result.status == "success"
    assert turn.index_receipt is not None
    assert turn.index_receipt.status == "unavailable"

    assert mark_dynamic_execution_no_progress(turn) is True
    assert turn.completion_execution_failure == "index_incomplete"
    reply = (
        "我这次发起了资料目录查询，但返回的目录不完整，"
        "所以这项任务还没有完成。"
    )
    allowed = _complete_bound_failure(turn, reply)

    assert allowed.allowed is True
    assert allowed.verification["action_count"] == 1
    assert allowed.verification["failed_action_count"] == 0


def test_runtime_failure_reports_ignore_model_wording_for_each_safe_class():
    cases = (
        (
            {"ok": True, "items": [], "hasMore": False, "nextCursor": ""},
            "empty",
        ),
        (
            {"ok": False, "status": 404, "code": "resource_not_found"},
            "not_found",
        ),
        (
            {"ok": False, "status": 403, "code": "permission_denied"},
            "denied",
        ),
        (
            {"ok": False, "status": 409, "code": "ambiguous_resource"},
            "ambiguous",
        ),
        (
            {"ok": False, "status": 503, "code": "provider_timeout"},
            "timeout",
        ),
        (
            {"ok": False, "status": 503, "code": "upstream_unavailable"},
            "unavailable",
        ),
        (
            {
                "ok": False,
                "status": 502,
                "code": "mystand_authorization_transport_failed",
            },
            "unavailable",
        ),
        (
            {"ok": False, "status": 500, "code": "handler_failed"},
            "execution_error",
        ),
        (
            {"ok": False, "status": 500, "code": "not_timeout_related"},
            "execution_error",
        ),
        (
            {
                "ok": False,
                "status": 500,
                "code": "connection_permission_denied",
            },
            "execution_error",
        ),
        (
            {"ok": False, "status": 500, "code": "reconnect_policy_rejected"},
            "execution_error",
        ),
    )
    variants = {
        "empty": (
            "不好意思，本轮我已经发起读取，不过没能拿到有效的内容。"
            "目前我无法给你可靠答复。"
        ),
        "not_found": (
            "抱歉，这次我没找到唯一匹配的记录，"
            "请你补充更具体的信息。"
        ),
        "denied": (
            "这次我无法获得完成任务所需的访问权限，"
            "因此我暂时不能继续回答。"
        ),
        "ambiguous": (
            "这次我不能确定唯一匹配的对象，"
            "请你补充更明确的范围。"
        ),
        "timeout": (
            "抱歉，这次我尝试了查询，但等待结果超时了。"
            "目前我没法确认结果。"
        ),
        "unavailable": (
            "这次我尝试了读取，但连接失败。"
            "目前我没法给你可靠答复。"
        ),
        "execution_error": (
            "不好意思，本轮我尝试了处理，但执行出了问题，"
            "因此这项任务尚未完成。"
        ),
    }
    for index, (payload, expected_reason) in enumerate(cases):
        turn = _dynamic_turn()
        started = begin_action(
            turn,
            "mystand_resource_index",
            "v1",
            {},
            call_id=f"failure-class-{index}",
        )
        assert started.decision == "allow"
        result = finish_action(
            turn,
            started.call.call_id,
            "mystand_resource_index",
            "v1",
            payload,
        )
        assert result is not None
        presentation = dynamic_failure_presentation(turn)
        assert presentation is not None
        assert presentation["failure_reason"] == expected_reason
        allowed = _complete_bound_failure(
            turn,
            "模型候选只是一段没有事实约束的状态话术。",
        )
        assert allowed.allowed is True
        natural = _complete_bound_failure(turn, variants[expected_reason])
        assert natural.allowed is True


@pytest.mark.parametrize(
    ("action_id", "error_code", "arguments"),
    [
        (
            "mystand_resource_index",
            "mystand_resource_index_transport_failed",
            {},
        ),
        (
            "mystand_resource_index",
            "mystand_resource_index_unavailable",
            {},
        ),
        (
            "mystand_query",
            "mystand_query_unavailable",
            {
                "operation": "read",
                "query_kind": "resource-read",
                "resource_uid": "resource-protocol",
            },
        ),
        (
            "mystand_authorization",
            "mystand_authorization_unavailable",
            {
                "operation": "resolve",
                "resource_uid": "resource-protocol",
            },
        ),
    ],
)
def test_unavailable_site_bridge_is_explained_but_not_retried(
    action_id,
    error_code,
    arguments,
):
    turn = _dynamic_turn()
    if action_id != "mystand_resource_index":
        _record_found_index(turn)
    started = begin_action(
        turn,
        action_id,
        "v1",
        arguments,
        call_id=f"{action_id}-unavailable",
    )
    assert started.decision == "allow"
    result = finish_action(
        turn,
        started.call.call_id,
        action_id,
        "v1",
        {
            "ok": False,
            "status": 503,
            "code": error_code,
        },
    )
    assert result is not None
    presentation = dynamic_failure_presentation(turn)
    assert presentation is not None
    assert presentation["failure_reason"] == "unavailable"
    assert dynamic_transient_recovery_plan(turn) is None
    allowed = _complete_bound_failure(
        turn,
        "模型候选没有资格决定失败原因或尝试次数。",
    )
    assert allowed.allowed is True


def test_cancelled_action_accepts_only_bound_natural_failure():
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="cancelled-index",
    )
    assert started.decision == "allow"
    result = finish_action(
        turn,
        started.call.call_id,
        "mystand_resource_index",
        "v1",
        {},
        cancelled=True,
    )
    assert result is not None
    assert result.status == "cancelled"
    assert dynamic_transient_recovery_plan(turn) is None

    reply = (
        "我这次已经发起实际处理，但随后被停止，"
        "所以这项任务还没有完成。"
    )
    allowed = _complete_bound_failure(turn, reply)

    assert allowed.allowed is True
    assert allowed.verification["failure_class"] == "cancelled"
    assert allowed.verification["failed_action_count"] == 1


@pytest.mark.parametrize(
    ("payload", "reply"),
    [
        (
            {"ok": False, "status": 503, "code": "provider_timeout"},
            "抱歉，我这次查询资料时遇到了超时，暂时无法完成你的请求。",
        ),
        (
            {"ok": False, "status": 403, "code": "permission_denied"},
            "我这次没有权限读取相关资料，因此无法继续完成。",
        ),
        (
            {"ok": False, "status": 503, "code": "upstream_unavailable"},
            "我这次读取时服务暂时不可用，所以暂时不能完成这项任务。",
        ),
        (
            {
                "ok": False,
                "status": 502,
                "code": "mystand_query_transport_failed",
            },
            "我这次读取时服务没有响应，任务还没完成。",
        ),
        (
            {"ok": False, "status": 500, "code": "handler_failed"},
            "这次我尝试处理时遇到了问题，因此没能完成。",
        ),
        (
            {"ok": True, "items": [], "hasMore": False, "nextCursor": ""},
            "我这次已经查询，但没有读到可用内容，所以无法给你可靠答复。",
        ),
        (
            {"ok": False, "status": 404, "code": "resource_not_found"},
            "我这次没找到匹配的目标，请你补充更具体的名称。",
        ),
        (
            {"ok": False, "status": 503, "code": "provider_timeout"},
            "我尝试查询了，但一直没等到结果，所以暂时不能给你答复。",
        ),
        (
            {"ok": False, "status": 503, "code": "upstream_unavailable"},
            "我这次读取时服务没有响应，任务还没完成。",
        ),
        (
            {"ok": True, "items": [], "hasMore": False, "nextCursor": ""},
            "我查了，但没有查到可以回答你的内容，所以暂时不能确认。",
        ),
        (
            {"ok": False, "status": 404, "code": "resource_not_found"},
            "我没找到你说的那份资料，需要你再说具体一点。",
        ),
    ],
)
def test_natural_failure_accepts_safe_everyday_paraphrases(payload, reply):
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="natural-paraphrase",
    )
    assert started.decision == "allow"
    finish_action(
        turn,
        started.call.call_id,
        "mystand_resource_index",
        "v1",
        payload,
    )

    assert _complete_bound_failure(turn, reply).allowed is True


def test_cancelled_action_accepts_everyday_paraphrase():
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="cancelled-paraphrase",
    )
    assert started.decision == "allow"
    finish_action(
        turn,
        started.call.call_id,
        "mystand_resource_index",
        "v1",
        {},
        cancelled=True,
    )

    assert _complete_bound_failure(
        turn,
        "这次处理后来被停止了，所以我没能完成。",
    ).allowed is True


def test_transient_read_failure_allows_exactly_one_bounded_recovery():
    turn = _dynamic_turn()
    _record_found_index(turn)
    first = begin_action(
        turn,
        "mystand_authorization",
        "v1",
        {
            "operation": "resolve",
            "resource_uid": "resource-protocol",
        },
        call_id="transient-first",
    )
    assert first.decision == "allow"
    finish_action(
        turn,
        first.call.call_id,
        "mystand_authorization",
        "v1",
        {
            "ok": False,
            "status": 502,
            "code": "mystand_authorization_transport_failed",
        },
    )

    plan = dynamic_transient_recovery_plan(turn)
    assert plan is not None
    assert plan["reason"] == "unavailable"
    assert plan["state"] == "上一次只读处理遇到暂时不可用"
    assert plan["grant"]["schema"] == "xiaoban.recovery-grant.v1"
    assert plan["grant"]["retry_of_event_id"] == "transient-first"
    assert plan["grant"]["max_uses"] == 1
    assert plan["grant"]["allowed_mutation"] == "exact_replay"
    assert plan["grant"]["target_binding"]
    assert plan["grant"]["grant_id"]
    assert plan["safe_scope"] == [
        {
            "resourceUid": "resource-protocol",
            "safeLabel": "目标资料",
            "resourceType": "generic-record",
            "canRead": True,
            "locked": False,
        }
    ]
    assert plan["failed_tool_call"] == {
        "call_id": "transient-first",
        "action_id": "mystand_authorization",
        "version": "v1",
        "arguments": {
            "operation": "resolve",
            "resource_uid": "resource-protocol",
        },
    }
    assert plan["tool_result"] == {
        "ok": False,
        "is_error": True,
        "status": 503,
        "code": "service_unavailable",
        "error": "这次正文读取服务暂时不可用。",
        "retryable": True,
    }
    assert plan["retry"] == {
        "action_id": "mystand_authorization",
        "version": "v1",
        "arguments": {
            "operation": "resolve",
            "resource_uid": "resource-protocol",
        },
        "arguments_digest": canonical_digest(
            {
                "operation": "resolve",
                "resource_uid": "resource-protocol",
            }
        ),
    }
    assert dynamic_transient_recovery_tool_call_valid(
        turn,
        action_id="mystand_authorization",
        arguments={
            "operation": "resolve",
            "resource_uid": "resource-protocol",
        },
    ) is True

    second = begin_action(
        turn,
        "mystand_authorization",
        "v1",
        {
            "operation": "resolve",
            "resource_uid": "resource-protocol",
        },
        call_id="transient-second",
    )
    assert second.decision == "allow"
    finish_action(
        turn,
        second.call.call_id,
        "mystand_authorization",
        "v1",
        {"ok": False, "status": 503, "code": "upstream_unavailable"},
    )

    assert dynamic_transient_recovery_plan(turn) is None


def test_invalid_query_arguments_allow_one_semantic_correction():
    turn = _dynamic_turn()
    _record_found_index(turn)
    mixed_arguments = {
        "operation": "read",
        "query_kind": "resource-read",
        "module_id": "profile",
        "fact_paths": ["resource.summary"],
        "query_args": {},
        "coverage_required": False,
        "resource": {
            "name": "目标特征卡",
            "type_hint": "profile-card",
        },
        "fact_needs": ["document.content", "resource.summary"],
        "mode": "summary",
    }
    first = begin_action(
        turn,
        "mystand_query",
        "v1",
        mixed_arguments,
        call_id="invalid-query-shape",
    )
    assert first.decision == "allow"
    finish_action(
        turn,
        first.call.call_id,
        "mystand_query",
        "v1",
        {
            "ok": False,
            "status": 400,
            "code": "invalid_mystand_query_arguments",
            "error": "本轮正文读取的查询格式与当前阶段不一致。",
            "retryable": True,
        },
    )

    plan = dynamic_transient_recovery_plan(turn)
    assert plan is not None
    assert plan["reason"] == "invalid_arguments"
    assert plan["mode"] == "correct_arguments"
    assert plan["state"] == "正文读取参数混入了当前阶段不允许的字段"
    assert plan["grant"]["schema"] == "xiaoban.recovery-grant.v1"
    assert plan["grant"]["retry_of_event_id"] == "invalid-query-shape"
    assert plan["grant"]["max_uses"] == 1
    assert plan["grant"]["allowed_mutation"] == "schema_only"
    assert plan["grant"]["target_binding"]
    assert plan["safe_scope"] == [
        {
            "safeLabel": "目标资料",
            "resourceType": "generic-record",
            "canRead": True,
            "locked": False,
        }
    ]
    corrected = {
        "operation": "read",
        "resource": {
            "name": "目标特征卡",
            "type_hint": "profile-card",
        },
        "entities": [],
        "fact_needs": ["document.content", "resource.summary"],
        "mode": "summary",
    }
    assert plan["correction"]["arguments"] == corrected
    assert plan["tool_result"]["ok"] is False
    assert plan["tool_result"]["is_error"] is True
    assert plan["tool_result"]["code"] == "invalid_mystand_query_arguments"
    assert dynamic_transient_recovery_tool_call_valid(
        turn,
        action_id="mystand_query",
        arguments=corrected,
    ) is True
    assert dynamic_transient_recovery_tool_call_valid(
        turn,
        action_id="mystand_query",
        arguments=mixed_arguments,
    ) is False
    changed_target = {
        **corrected,
        "resource": {
            "name": "另一张特征卡",
            "type_hint": "profile-card",
        },
    }
    assert dynamic_transient_recovery_tool_call_valid(
        turn,
        action_id="mystand_query",
        arguments=changed_target,
    ) is False
    assert dynamic_transient_recovery_tool_call_valid(
        turn,
        action_id="mystand_authorization",
        arguments=corrected,
    ) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_kind", "resource-read"),
        ("module_id", "profile"),
        ("fact_paths", ["resource.summary"]),
        ("query_args", {}),
        ("coverage_required", False),
    ],
)
def test_semantic_correction_rejects_every_typed_field(field, value):
    arguments = {
        "operation": "read",
        "resource": {"name": "目标特征卡"},
        "fact_needs": ["document.content"],
        field: value,
    }
    with pytest.raises(ValueError):
        validate_mystand_semantic_query_plan(arguments)

    turn = _dynamic_turn()
    _record_found_index(turn)
    failed = begin_action(
        turn,
        "mystand_query",
        "v1",
        {
            "operation": "read",
            "query_kind": "resource-read",
            "module_id": "profile",
            "fact_paths": ["resource.summary"],
            "query_args": {},
            "coverage_required": False,
        },
        call_id=f"strict-semantic-{field}",
    )
    assert failed.decision == "allow"
    finish_action(
        turn,
        failed.call.call_id,
        "mystand_query",
        "v1",
        {
            "ok": False,
            "status": 400,
            "code": "invalid_mystand_query_arguments",
        },
    )
    assert dynamic_transient_recovery_tool_call_valid(
        turn,
        action_id="mystand_query",
        arguments=arguments,
    ) is False


def test_repeated_invalid_query_failure_explains_cause_attempt_and_missing_data():
    turn = _dynamic_turn()
    _record_found_index(turn)
    for index, arguments in enumerate(
        (
            {
                "operation": "read",
                "query_kind": "resource-read",
                "module_id": "profile",
                "fact_paths": ["resource.summary"],
                "query_args": {},
                "coverage_required": False,
                "resource": {
                    "name": "目标特征卡",
                    "type_hint": "profile-card",
                },
                "entities": [],
                "fact_needs": [
                    "document.content",
                    "resource.summary",
                ],
                "mode": "summary",
            },
            {
                "operation": "read",
                "resource": {
                    "name": "目标特征卡",
                    "type_hint": "profile-card",
                },
                "entities": [],
                "fact_needs": ["document.content", "resource.summary"],
                "mode": "summary",
            },
        ),
        start=1,
    ):
        call = begin_action(
            turn,
            "mystand_query",
            "v1",
            arguments,
            call_id=f"invalid-query-{index}",
        )
        assert call.decision == "allow"
        finish_action(
            turn,
            call.call.call_id,
            "mystand_query",
            "v1",
            {
                "ok": False,
                "status": 400,
                "code": "invalid_mystand_query_arguments",
                "error": "本轮正文读取的查询格式与当前阶段不一致。",
            },
        )

    assert dynamic_transient_recovery_plan(turn) is None
    presentation = dynamic_failure_presentation(turn)
    assert presentation is not None
    assert presentation["failure_reason"] == "invalid_arguments"
    assert presentation["failed_attempt_count"] == 2
    assert presentation["recovery_attempted"] is True
    assert presentation["missing"] == "完成请求所需的可靠资料内容"
    reply = (
        "我这次先查询了资料目录，但读取正文时发现参数里混入了当前阶段"
        "不允许的字段。我去掉这些字段后又尝试了一次，还是没拿到这次"
        "要用的资料正文，"
        "所以现在不能根据正文给你可靠建议。"
    )
    allowed = _complete_bound_failure(turn, reply)
    assert allowed.allowed is True
    assert allowed.text == reply
    assert allowed.verification["failure_reason"] == "invalid_arguments"
    assert allowed.verification["recovery_reason"] == "invalid_arguments"
    assert allowed.verification["failed_action_count"] == 2
    outcome = allowed.verification["turn_outcome"]
    assert outcome["attempt_event_ids"] == [
        "invalid-query-1",
        "invalid-query-2",
    ]
    assert outcome["attempt_count"] == 2
    assert outcome["recovery"] == {
        "attempted": True,
        "reason": "invalid_arguments",
    }


@pytest.mark.parametrize(
    "reply",
    [
        (
            "我已经找到了客户特征卡的资料目录。读取正文时，参数里混入"
            "了当前阶段不允许的字段。我去掉这些字段后重试了一次，仍然"
            "没有拿到正文，因此暂时无法根据这张卡给出可靠建议。"
        ),
        (
            "我找到了房源笔记的资料目录，但读取正文时参数里包含当前"
            "阶段不允许的字段。我去掉这些字段后又试了一次，还是没拿到正文，"
            "所以现在不能根据正文给你可靠建议。"
        ),
        (
            "我先找到了业主资料的资料目录，但读取正文时发现参数混入了"
            "当前阶段不允许的字段。去掉这些字段后我又重试了一次，仍未取得正文，"
            "所以没法给出可靠建议。"
        ),
        (
            "我找到了财务账本的资料目录，但读取正文时参数包含当前阶段"
            "不允许的字段。我去掉这些字段后重试了一次，仍没拿到正文，"
            "所以暂时不能给你可靠建议。"
        ),
    ],
)
def test_repeated_invalid_query_does_not_infer_material_from_model_prose(reply):
    turn = _dynamic_turn()
    _record_found_index(turn)
    arguments = (
        {
            "operation": "read",
            "query_kind": "resource-read",
            "module_id": "profile",
            "fact_paths": ["resource.summary"],
            "query_args": {},
            "coverage_required": False,
            "resource": {"name": "目标特征卡"},
            "fact_needs": ["document.content"],
        },
        {
            "operation": "read",
            "resource": {"name": "目标特征卡"},
            "entities": [],
            "fact_needs": ["document.content"],
            "mode": "summary",
        },
    )
    for index, item in enumerate(arguments):
        call = begin_action(
            turn,
            "mystand_query",
            "v1",
            item,
            call_id=f"natural-invalid-{index}",
        )
        assert call.decision == "allow"
        finish_action(
            turn,
            call.call.call_id,
            "mystand_query",
            "v1",
            {
                "ok": False,
                "status": 400,
                "code": "invalid_mystand_query_arguments",
            },
        )

    allowed = _complete_bound_failure(turn, reply)
    assert allowed.allowed is True
    assert allowed.text == reply
    assert allowed.verification["turn_outcome"]["obtained"]["material"] is False
    assert allowed.verification["evidence_count"] == 0


def test_runtime_failure_report_is_tagged_as_system_receipt():
    turn = _dynamic_turn()
    _record_found_index(turn)
    for index, arguments in enumerate(
        (
            {
                "operation": "read",
                "query_kind": "resource-read",
                "module_id": "profile",
                "fact_paths": ["resource.summary"],
                "query_args": {},
                "coverage_required": False,
                "resource": {"name": "目标特征卡"},
                "fact_needs": ["document.content"],
            },
            {
                "operation": "read",
                "resource": {"name": "目标特征卡"},
                "entities": [],
                "fact_needs": ["document.content"],
                "mode": "summary",
            },
        )
    ):
        call = begin_action(
            turn,
            "mystand_query",
            "v1",
            arguments,
            call_id=f"fallback-invalid-{index}",
        )
        assert call.decision == "allow"
        finish_action(
            turn,
            call.call.call_id,
            "mystand_query",
            "v1",
            {
                "ok": False,
                "status": 400,
                "code": "invalid_mystand_query_arguments",
            },
        )

    fallback = _complete_bound_failure(
        turn,
        render_dynamic_failure_report(turn),
    )
    assert fallback.allowed is True
    assert fallback.reason == "execution_status_system_receipt"
    assert "参数混入了当前阶段不允许的字段" in fallback.text
    assert "去掉这些字段后又尝试了一次" in fallback.text
    assert fallback.verification["output_presentation"] == "system-receipt"
    assert fallback.verification["answer_status"] == "incomplete"


@pytest.mark.parametrize(
    "model_text",
    [
        (
            "我这次已经发起实际处理，但执行返回了错误，"
            "所以这项任务还没有完成。"
        ),
        (
            "我这次读取时发现查询格式不符合当前规则，把参数改对后"
            "继续读了一遍，仍没拿到正文，所以不能给你建议。"
        ),
        (
            "我这次读取遇到错误，修正参数后接着查，结果还是失败，"
            "所以这项任务没有完成。"
        ),
    ],
)
def test_failure_prose_never_authenticates_cause_or_retry_count(model_text):
    turn = _dynamic_turn()
    _record_found_index(turn)
    call = begin_action(
        turn,
        "mystand_query",
        "v1",
        {
            "operation": "read",
            "resource": {"name": "任意模块资料"},
            "fact_needs": ["document.content"],
        },
        call_id="one-real-failure",
    )
    assert call.decision == "allow"
    finish_action(
        turn,
        call.call.call_id,
        "mystand_query",
        "v1",
        {
            "ok": False,
            "status": 500,
            "code": "handler_failed",
        },
    )

    decision = _complete_bound_failure(turn, model_text)
    assert decision.allowed is True
    assert decision.reason == "execution_status_bound"
    assert decision.text == model_text
    assert "output_presentation" not in decision.verification
    outcome = decision.verification["turn_outcome"]
    assert outcome["attempt_event_ids"] == ["one-real-failure"]
    assert outcome["attempt_count"] == 1
    assert outcome["recovery"]["attempted"] is False


def test_turn_outcome_uses_physical_event_order_not_call_id_sorting():
    turn = _dynamic_turn()
    _record_found_index(turn)
    first_arguments = {
        "operation": "read",
        "query_kind": "resource-read",
        "resource": {"name": "任意模块资料"},
        "fact_needs": ["document.content"],
    }
    second_arguments = {
        "operation": "read",
        "resource": {"name": "任意模块资料"},
        "entities": [],
        "fact_needs": ["document.content"],
        "mode": "summary",
    }
    for call_id, arguments in (
        ("z-first-physical", first_arguments),
        ("a-second-physical", second_arguments),
    ):
        call = begin_action(
            turn,
            "mystand_query",
            "v1",
            arguments,
            call_id=call_id,
        )
        assert call.decision == "allow"
        finish_action(
            turn,
            call.call.call_id,
            "mystand_query",
            "v1",
            {
                "ok": False,
                "status": 400,
                "code": "invalid_mystand_query_arguments",
            },
        )

    outcome = dynamic_turn_outcome(turn)
    assert outcome is not None
    assert outcome["attempt_event_ids"] == [
        "z-first-physical",
        "a-second-physical",
    ]
    assert outcome["recovery"]["attempted"] is True


@pytest.mark.parametrize(
    ("terminal_payload", "expected_class", "expected_reason"),
    [
        (
            {"ok": False, "status": 504, "code": "provider_timeout"},
            "error",
            "timeout",
        ),
        (
            {"ok": False, "status": 503, "code": "upstream_unavailable"},
            "error",
            "unavailable",
        ),
        (
            {"ok": False, "status": 404, "code": "resource_not_found"},
            "not_found",
            "not_found",
        ),
    ],
)
def test_argument_correction_preserves_terminal_failure_reason(
    terminal_payload,
    expected_class,
    expected_reason,
):
    turn = _dynamic_turn()
    _record_found_index(turn)
    first = begin_action(
        turn,
        "mystand_query",
        "v1",
        {
            "operation": "read",
            "query_kind": "resource-read",
            "module_id": "profile",
            "fact_paths": ["resource.summary"],
            "query_args": {},
            "coverage_required": False,
            "resource": {"name": "目标特征卡"},
            "fact_needs": ["document.content"],
        },
        call_id="ordered-first-invalid",
    )
    assert first.decision == "allow"
    finish_action(
        turn,
        first.call.call_id,
        "mystand_query",
        "v1",
        {
            "ok": False,
            "status": 400,
            "code": "invalid_mystand_query_arguments",
        },
    )
    terminal = begin_action(
        turn,
        "mystand_query",
        "v1",
        {
            "operation": "read",
            "resource": {"name": "目标特征卡"},
            "entities": [],
            "fact_needs": ["document.content"],
            "mode": "summary",
        },
        call_id="ordered-second-terminal",
    )
    assert terminal.decision == "allow"
    finish_action(
        turn,
        terminal.call.call_id,
        "mystand_query",
        "v1",
        terminal_payload,
    )

    presentation = dynamic_failure_presentation(turn)
    assert presentation is not None
    assert presentation["failure_class"] == expected_class
    assert presentation["failure_reason"] == expected_reason
    assert presentation["recovery_reason"] == "invalid_arguments"
    assert "不允许的字段" in presentation["state"]
    allowed = _complete_bound_failure(
        turn,
        "模型候选与运行时失败事实无关。",
    )
    assert allowed.allowed is True
    assert allowed.verification["failure_class"] == expected_class
    assert allowed.verification["failure_reason"] == expected_reason
    assert allowed.verification["recovery_reason"] == "invalid_arguments"


def test_corrected_semantic_query_can_finish_with_evidence_and_advice():
    turn = _dynamic_turn()
    _record_found_index(turn)
    failed = begin_action(
        turn,
        "mystand_query",
        "v1",
        {
            "operation": "read",
            "query_kind": "resource-read",
            "module_id": "profile",
            "fact_paths": ["resource.summary"],
            "query_args": {},
            "coverage_required": False,
            "resource": {"name": "目标资料"},
        },
        call_id="feature-card-bad-shape",
    )
    assert failed.decision == "allow"
    finish_action(
        turn,
        failed.call.call_id,
        "mystand_query",
        "v1",
        {
            "ok": False,
            "status": 400,
            "code": "invalid_mystand_query_arguments",
            "error": "本轮正文读取的查询格式与当前阶段不一致。",
        },
    )
    corrected_arguments = {
        "operation": "read",
        "resource": {"name": "目标资料"},
        "entities": [],
        "fact_needs": ["document.content", "resource.summary"],
        "mode": "summary",
    }
    recovered = begin_action(
        turn,
        "mystand_query",
        "v1",
        corrected_arguments,
        call_id="feature-card-corrected",
    )
    assert recovered.decision == "allow"
    recovered_result = finish_action(
        turn,
        recovered.call.call_id,
        "mystand_query",
        "v1",
        {
            "schema": "mystand.query-result.v1",
            "ok": True,
            "status": "matched",
            "missing_facts": [],
            "resource": {
                "resourceUid": "resource-protocol",
                "display_name": "目标资料",
                "type": "generic-record",
            },
            "recordRefs": ["resource-protocol"],
            "facts": [
                {
                    "kind": "resource.summary",
                    "label": "资料摘要",
                    "value": "沟通重点清晰，下一步适合先核对需求。",
                }
            ],
            "content": "沟通重点清晰，下一步适合先核对需求。",
        },
    )
    assert recovered_result is not None
    assert recovered_result.status == "success"

    for mechanical in (
        "我查到了。",
        "资料已读取。",
        "我拿到资料了，但先不分析。",
    ):
        incomplete = check_completion(mechanical, turn)
        assert incomplete.allowed is True
        assert incomplete.text == (
            "资料已经读取成功，但最终回答没有使用本轮资料中的具体内容，"
            "无法确认它真正完成了你的要求。本次任务仍按未完成处理。"
        )
        assert incomplete.reason == (
            "evidence_answer_incomplete_system_receipt"
        )
        assert incomplete.verification["output_presentation"] == (
            "system-receipt"
        )
        assert incomplete.verification["answer_status"] == "incomplete"

    reply = (
        "资料里写的是“沟通重点清晰，下一步适合先核对需求”。"
        "这说明现在可以继续推进，但还不适合直接替客户下结论；"
        "我的建议是先逐项核对需求，再按确认结果安排下一步跟进。"
    )
    allowed = check_completion(reply, turn)
    assert allowed.allowed is True
    assert allowed.text == reply
    assert allowed.verification["completion_kind"] == "evidence-bound"
    assert allowed.verification["action_count"] == 3
    assert allowed.verification["evidence_count"] == 1
    assert "output_presentation" not in allowed.verification


def test_responses_history_keeps_tools_but_not_system_receipt_as_agent_speech():
    prior = [{"role": "user", "content": "上一轮"}]
    user_message = "查特征卡并给建议"
    rejected = "资料拿到了，建议晚点再说。"
    receipt = (
        "资料已经读取成功，但最终回答没有使用本轮资料中的具体内容，"
        "无法确认它真正完成了你的要求。本次任务仍按未完成处理。"
    )
    tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "query-call",
                "type": "function",
                "function": {
                    "name": "mystand_query",
                    "arguments": '{"operation":"read"}',
                },
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "name": "mystand_query",
        "tool_call_id": "query-call",
        "content": '{"ok":true}',
    }
    result = {
        "messages": [
            *prior,
            {"role": "user", "content": user_message},
            tool_call,
            tool_result,
            {"role": "assistant", "content": rejected},
        ],
        "final_response": receipt,
        "_mystand_trusted_verification": {
            "output_presentation": "system-receipt",
            "answer_status": "incomplete",
        },
    }

    history = APIServerAdapter._build_response_conversation_history(
        prior,
        user_message,
        result,
        receipt,
    )
    assert history == [
        *prior,
        {"role": "user", "content": user_message},
        tool_call,
        tool_result,
    ]
    assert rejected not in json.dumps(history, ensure_ascii=False)
    assert receipt not in json.dumps(history, ensure_ascii=False)
    assert APIServerAdapter._turn_transcript_messages(
        prior,
        user_message,
        result,
    ) == [
        APIServerAdapter._message_response(tool_call),
        APIServerAdapter._message_response(tool_result),
    ]

    result["messages"] = []
    assert APIServerAdapter._build_response_conversation_history(
        prior,
        user_message,
        result,
        receipt,
    ) == [
        *prior,
        {"role": "user", "content": user_message},
    ]


def test_transient_recovery_rejects_target_outside_owner_bound_index():
    turn = _dynamic_turn()
    _record_found_index(turn)
    failed = begin_action(
        turn,
        "mystand_authorization",
        "v1",
        {
            "operation": "resolve",
            "resource_uid": "resource-foreign",
        },
        call_id="transient-foreign-target",
    )
    # PreAction's legacy found-index check alone is not enough; the recovery
    # planner must apply the stronger exact owner-bound membership check.
    assert failed.decision == "allow"
    finish_action(
        turn,
        failed.call.call_id,
        "mystand_authorization",
        "v1",
        {
            "ok": False,
            "status": 502,
            "code": "mystand_authorization_transport_failed",
        },
    )

    assert dynamic_transient_recovery_plan(turn) is None
    assert dynamic_transient_recovery_tool_call_valid(
        turn,
        action_id="mystand_authorization",
        arguments={
            "operation": "resolve",
            "resource_uid": "resource-foreign",
        },
    ) is False


@pytest.mark.parametrize(
    ("can_read", "locked"),
    [(False, False), (True, True)],
)
def test_transient_recovery_rejects_indexed_but_unreadable_target(
    can_read,
    locked,
):
    turn = _dynamic_turn()
    index = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id=f"unreadable-index-{can_read}-{locked}",
    )
    assert index.decision == "allow"
    finish_action(
        turn,
        index.call.call_id,
        "mystand_resource_index",
        "v1",
        {
            "schema": "mystand.resource-index.complete.v1",
            "ok": True,
            "items": [
                {
                    "resourceUid": "resource-unreadable",
                    "safeLabel": "目标资料",
                    "resourceType": "generic-record",
                    "canRead": can_read,
                    "locked": locked,
                }
            ],
            "hasMore": False,
            "nextCursor": "",
        },
    )
    failed = begin_action(
        turn,
        "mystand_authorization",
        "v1",
        {
            "operation": "resolve",
            "resource_uid": "resource-unreadable",
        },
        call_id=f"unreadable-read-{can_read}-{locked}",
    )
    assert failed.decision == "allow"
    finish_action(
        turn,
        failed.call.call_id,
        "mystand_authorization",
        "v1",
        {
            "ok": False,
            "status": 502,
            "code": "mystand_authorization_transport_failed",
        },
    )

    assert dynamic_transient_recovery_plan(turn) is None


def test_transient_index_failure_does_not_start_a_costly_partial_recovery():
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="transient-index",
    )
    assert started.decision == "allow"
    finish_action(
        turn,
        started.call.call_id,
        "mystand_resource_index",
        "v1",
        {"ok": False, "status": 503, "code": "provider_timeout"},
    )

    assert dynamic_transient_recovery_plan(turn) is None


def test_no_progress_mark_requires_index_only_success():
    no_index = _dynamic_turn()
    assert mark_dynamic_read_no_progress(no_index) is False

    with_read = _dynamic_turn()
    _record_found_index(with_read)
    read = begin_action(
        with_read,
        "mystand_query",
        "v1",
        {"operation": "read"},
        call_id="read-dispatched",
    )
    assert read.decision == "allow"
    assert mark_dynamic_read_no_progress(with_read) is False

    rejected = _dynamic_turn()
    _record_found_index(rejected)
    rejected.rejected_cross_account = 1
    assert mark_dynamic_read_no_progress(rejected) is False


def test_dynamic_provider_tools_are_request_local_and_index_first():
    turn = _dynamic_turn()
    original_tools = _tools(
        "mystand_resource_index",
        "mystand_query",
        "mystand_authorization",
        "terminal",
    )
    payload = {
        "model": "test",
        "tools": original_tools,
        "tool_choice": {
            "type": "function",
            "function": {"name": "mystand_query"},
        },
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    assert _tool_names(filtered) == ["mystand_resource_index"]
    assert "tool_choice" not in filtered
    assert _tool_names(payload) == [
        "mystand_resource_index",
        "mystand_query",
        "mystand_authorization",
        "terminal",
    ]
    assert payload["tools"] is original_tools


def test_dynamic_provider_tools_switch_to_reads_after_found_index():
    turn = _dynamic_turn()
    _record_found_index(turn)
    payload = {
        "tools": [
            *_tools("mystand_resource_index"),
            _canonical_query_tool(),
            *_tools("mystand_authorization", "terminal"),
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "mystand_resource_index"},
        },
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    assert _tool_names(filtered) == [
        "mystand_query",
        "mystand_authorization",
    ]
    assert "tool_choice" not in filtered
    assert filtered["parallel_tool_calls"] is False


def test_dynamic_query_schema_is_semantic_only_after_found_index():
    turn = _dynamic_turn()
    _record_found_index(turn)
    query_tool = _canonical_query_tool()
    original_query_tool = json.loads(
        json.dumps(query_tool, ensure_ascii=False)
    )
    payload = {
        "tools": [
            *_tools("mystand_resource_index"),
            query_tool,
            *_tools("mystand_authorization"),
        ],
        "parallel_tool_calls": True,
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    visible_query = next(
        item
        for item in filtered["tools"]
        if item["function"]["name"] == "mystand_query"
    )
    parameters = visible_query["function"]["parameters"]
    assert set(parameters["properties"]) == {
        "operation",
        "resource",
        "entities",
        "fact_needs",
        "mode",
    }
    assert parameters["required"] == [
        "operation",
        "resource",
        "fact_needs",
    ]
    assert parameters["additionalProperties"] is False
    assert "anyOf" not in parameters
    assert {
        "query_kind",
        "module_id",
        "fact_paths",
        "query_args",
        "coverage_required",
    }.isdisjoint(parameters["properties"])
    assert "query_kind" not in visible_query["function"]["description"]
    assert payload["tools"][1] == original_query_tool
    assert payload["parallel_tool_calls"] is True
    assert filtered["parallel_tool_calls"] is False


def test_dynamic_read_batch_executes_only_first_call_before_follow_up():
    turn = _dynamic_turn()
    _record_found_index(turn)
    calls = [
        SimpleNamespace(
            id=f"read-{index}",
            function=SimpleNamespace(
                name="mystand_query",
                arguments=json.dumps(
                    {
                        "operation": "read",
                        "resource": {"name": f"目标资料{index}"},
                        "fact_needs": ["resource.summary"],
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        for index in range(1, 4)
    ]
    active = activate_turn(turn)
    try:
        capped = _cap_dynamic_evidence_tool_calls(calls)
    finally:
        deactivate_turn(active)

    assert [call.id for call in capped] == ["read-1"]
    assert [call.id for call in calls] == ["read-1", "read-2", "read-3"]


def test_dynamic_provider_tools_close_during_finalization():
    turn = _dynamic_turn()
    _record_found_index(turn)
    turn.completion_finalization = "failure"
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        ),
        "tool_choice": "required",
        "parallel_tool_calls": True,
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    assert filtered["tools"] == []
    assert "tool_choice" not in filtered
    assert "parallel_tool_calls" not in filtered


def test_diagnostic_turn_hides_and_rejects_all_business_tools():
    turn = _dynamic_turn(business_tools_disabled=True)
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        ),
        "tool_choice": "required",
        "parallel_tool_calls": True,
    }

    filtered = filter_dynamic_evidence_api_kwargs(payload, turn=turn)

    assert filtered["tools"] == []
    assert "tool_choice" not in filtered
    assert "parallel_tool_calls" not in filtered
    direct_decision = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="diagnostic-direct-preexecution",
    )
    assert direct_decision.decision == "deny"
    assert direct_decision.reason == "business_tools_disabled"

    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    active = activate_turn(turn)
    seen: list[str] = []
    try:
        registry = ToolRegistry()
        for tool_name in (
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        ):
            registry.register(
                tool_name,
                tool_name,
                {"name": tool_name, "parameters": {}},
                lambda _args, name=tool_name: seen.append(name)
                or json.dumps({"ok": True}),
            )
        for index, tool_name in enumerate((
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        )):
            denied = json.loads(
                registry.dispatch(
                    tool_name,
                    {},
                    tool_call_id=f"diagnostic-tool-{index}",
                )
            )
            assert denied["code"] == "business_tools_disabled"
        assert seen == []
        assert turn.action_calls == []
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


@pytest.mark.parametrize(
    "reference_id",
    ["AUTH-ABCDEFG", "OUT-ABCDEFG"],
)
def test_diagnostic_turn_skips_deterministic_reference_preexecution(
    reference_id,
):
    turn = _dynamic_turn(business_tools_disabled=True)
    started: list[str] = []
    completed: list[str] = []

    evidence = _run_mystand_preexecuted_evidence(
        "mystand_authorization",
        user_message=f"刚才查 {reference_id} 为什么失败？",
        system_prompt="",
        tool_start_callback=lambda _call_id, name, _args: (
            started.append(name)
        ),
        tool_complete_callback=lambda _call_id, name, _args, _result: (
            completed.append(name)
        ),
        trusted_turn=turn,
    )

    assert evidence == []
    assert started == []
    assert completed == []
    assert turn.action_calls == []
    assert turn.action_results == []


def test_signed_fact_turn_keeps_provider_tools_unchanged():
    signed = begin_turn(
        channel="web",
        user_message="读取签名资料",
        identity=IDENTITY,
        request_id="signed-request",
        message_id="signed-message",
        fact_requirement={"schema": "mystand.fact-requirement.v1"},
    )
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        ),
        "tool_choice": {
            "type": "function",
            "function": {"name": "mystand_query"},
        },
    }

    assert filter_dynamic_evidence_api_kwargs(payload, turn=signed) is payload


def test_dynamic_registry_discards_untrusted_module_hint_before_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_resource_index",
            "mystand_resource_index",
            {"name": "mystand_resource_index", "parameters": {}},
            lambda args: seen.append(dict(args))
            or json.dumps(
                {
                    "ok": True,
                    "items": [
                        {
                            "resourceUid": "resource-protocol",
                            "safeLabel": "目标资料",
                        }
                    ],
                    "hasMore": False,
                }
            ),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_resource_index",
                {
                    "operation": "list_resources",
                    "module_id": "model-guessed-module",
                    "moduleId": "another-model-guess",
                    "query": "目标资料",
                },
            )
        )

        assert result["ok"] is True
        assert seen == [
            {
                "operation": "list_resources",
                "query": "目标资料",
            }
        ]
        assert turn.action_calls[0].arguments == seen[0]

        closed = json.loads(
            registry.dispatch(
                "mystand_resource_index",
                {"query": "再次索引"},
                tool_call_id="index-after-found",
            )
        )
        assert closed == {
            "ok": False,
            "status": 403,
            "code": "dynamic_index_stage_closed",
        }
        assert len(seen) == 1
        assert dynamic_finalization_mode(turn) == ""
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_read_state_machine_blocks_write_before_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_authorization",
            "mystand_authorization",
            {"name": "mystand_authorization", "parameters": {}},
            lambda args: seen.append(dict(args)) or json.dumps({"ok": True}),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_authorization",
                {
                    "operation": "preview_write",
                    "authorization_id": "AUTH-test",
                },
                tool_call_id="dynamic-write",
            )
        )

        assert result == {
            "ok": False,
            "status": 403,
            "code": "write_isolated",
        }
        assert seen == []
        assert turn.action_calls == []
        assert dynamic_finalization_mode(turn) == ""
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_state_machine_blocks_every_hidden_tool_before_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[tuple[str, dict]] = []
        for tool_name in ("mystand_authorization_write", "terminal"):
            registry.register(
                tool_name,
                tool_name,
                {"name": tool_name, "parameters": {}},
                lambda args, name=tool_name: seen.append((name, dict(args)))
                or json.dumps({"ok": True}),
            )

        write_result = json.loads(
            registry.dispatch(
                "mystand_authorization_write",
                {"operation": "preview_write"},
                tool_call_id="hidden-write",
            )
        )
        terminal_result = json.loads(
            registry.dispatch(
                "terminal",
                {"command": "true"},
                tool_call_id="hidden-terminal",
            )
        )

        assert write_result["code"] == "write_isolated"
        assert terminal_result["code"] == "dynamic_tool_stage_closed"
        assert seen == []
        assert turn.action_calls == []
        executor_denial = json.loads(
            _trusted_preaction_denial(
                "memory",
                {"action": "add", "content": "must-not-write"},
                "hidden-inline-memory",
            )
        )
        assert executor_denial["code"] == "dynamic_tool_stage_closed"
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_hidden_tool_fails_closed_if_stage_helper_breaks(monkeypatch):
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "terminal",
            "terminal",
            {"name": "terminal", "parameters": {}},
            lambda args: seen.append(dict(args)) or json.dumps({"ok": True}),
        )

        def broken_stage(_turn):
            raise RuntimeError("stage helper unavailable")

        monkeypatch.setattr(
            "xiaoban.trusted_runtime.tool_visibility."
            "dynamic_evidence_allowed_tool_names",
            broken_stage,
        )
        result = json.loads(
            registry.dispatch(
                "terminal",
                {"command": "true"},
                tool_call_id="hidden-terminal-broken-stage",
            )
        )

        assert result["code"] == "preaction_error"
        assert seen == []
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_dynamic_finalization_state_blocks_all_tool_dispatch():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="message-protocol",
    )
    turn = _dynamic_turn()
    _record_found_index(turn)
    turn.completion_finalization = "evidence"
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_query",
            "mystand_query",
            {"name": "mystand_query", "parameters": {}},
            lambda args: seen.append(dict(args)) or json.dumps({"ok": True}),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_query",
                {"operation": "read"},
                tool_call_id="read-after-finalize",
            )
        )

        assert result == {
            "ok": False,
            "status": 403,
            "code": "dynamic_finalization_stage_closed",
        }
        assert seen == []
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_signed_registry_keeps_module_hint_bound_and_dispatched():
    tokens = set_session_vars(
        platform="api_server",
        user_id=IDENTITY.account_id,
        message_id="signed-message",
    )
    turn = begin_turn(
        channel="web",
        user_message="读取签名资料",
        identity=IDENTITY,
        request_id="signed-request",
        message_id="signed-message",
        fact_requirement={"schema": "mystand.fact-requirement.v1"},
    )
    active = activate_turn(turn)
    try:
        registry = ToolRegistry()
        seen: list[dict] = []
        registry.register(
            "mystand_resource_index",
            "mystand_resource_index",
            {"name": "mystand_resource_index", "parameters": {}},
            lambda args: seen.append(dict(args))
            or json.dumps(
                {
                    "ok": True,
                    "items": [
                        {
                            "resourceUid": "resource-signed",
                            "moduleId": "signed-module",
                            "safeLabel": "签名资料",
                        }
                    ],
                    "hasMore": False,
                }
            ),
        )

        result = json.loads(
            registry.dispatch(
                "mystand_resource_index",
                {
                    "operation": "list_resources",
                    "module_id": "signed-module",
                },
            )
        )

        assert result["ok"] is True
        assert seen[0]["module_id"] == "signed-module"
        assert turn.action_calls[0].arguments["module_id"] == "signed-module"
    finally:
        deactivate_turn(active)
        clear_session_vars(tokens)


def test_failure_finalize_view_excludes_raw_tool_trajectory():
    turn = _dynamic_turn()
    denied = begin_action(
        turn,
        "mystand_query",
        "v1",
        {"operation": "read"},
        call_id="preaction-denied",
    )
    assert denied.reason == "missing_index_receipt"
    decision = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="index-failed",
    )
    finish_action(
        turn,
        decision.call.call_id,
        "mystand_resource_index",
        "v1",
        {"ok": False, "status": 503, "code": "private_internal_code"},
    )
    messages = [
        {
            "role": "system",
            "content": "stable policy private_ephemeral_evidence",
        },
        {"role": "user", "content": "读取目标资料"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "mystand_resource_index"}}],
        },
        {
            "role": "tool",
            "content": '{"code":"private_internal_code"}',
        },
    ]
    agent = SimpleNamespace(
        _strict_no_automatic_paid_retry=True,
        max_iterations=3,
    )
    active = activate_turn(turn)
    try:
        selected = _prepare_finalize_only_call(
            agent,
            1,
            messages,
            original_user_message="读取目标资料",
        )
    finally:
        deactivate_turn(active)

    assert selected is True
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
    ]
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "private_internal_code" not in serialized
    assert "private_ephemeral_evidence" not in serialized
    assert "stable policy" not in serialized
    assert "mystand_resource_index" not in serialized
    assert "资料读取已发起，但返回了错误" in serialized
    assert "没有可安全确认的更细原因" in serialized
    assert "读取目标资料" in serialized
    assert "推荐句式" not in serialized
    assert "不要增加任何原因" not in serialized
    assert "当前无权读取" not in serialized
    assert "完整执行结果" in messages[-1]["content"]


def test_raw_protocol_candidate_is_rejected_before_any_cleanup():
    double_bar_dsml = (
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"mystand_resource_index\">\n"
        "<｜｜DSML｜｜parameter name=\"query\">目标资料</｜｜DSML｜｜parameter>\n"
        "</｜｜DSML｜｜invoke>\n"
        "</｜｜DSML｜｜tool_calls>"
    )
    double_bar_dsml += "x" * (272 - len(double_bar_dsml))
    assert len(double_bar_dsml) == 272
    candidates = [
        '<|DSML|function_calls><|DSML|invoke name="mystand_query">',
        double_bar_dsml,
        '<tool_call>{"name":"mystand_query"}</tool_call>',
        '{"tool_calls":[{"function":{"name":"mystand_query"}}]}',
        '{"name":"mystand_query","arguments":{"operation":"read"}}',
        '{"type":"tool_use","name":"mystand_query","input":{}}',
        '{"functionCall":{"name":"mystand_query","args":{}}}',
        "assistant to=tools.mystand_query",
    ]
    assert all(_contains_raw_tool_protocol_content(item) for item in candidates)
    assert not _contains_raw_tool_protocol_content(
        "这次查询没有找到匹配资料，请补充更准确的名称。"
    )
    assert not _contains_raw_tool_protocol_content(
        '{"name":"中海城南一号","arguments":"客户补充说明"}'
    )

    persisted: list[list[dict]] = []
    agent = SimpleNamespace(
        _drop_trailing_empty_response_scaffolding=lambda _messages: None,
        _persist_session=lambda messages, _history: persisted.append(
            list(messages)
        ),
    )
    result = _reject_finalize_only_protocol_candidate(
        agent,
        [{"role": "user", "content": "读取资料"}],
        [],
        api_call_count=2,
        candidate="<think>ignore</think><tool_call>{}</tool_call>",
        finalize_only=True,
    )

    assert result is not None
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["final_response"] is None
    assert result["partial"] is True
    assert "protocol_content_rejected" in result["turn_exit_reason"]
    assert persisted


def test_agent_build_api_kwargs_always_applies_request_local_stage(monkeypatch):
    payload = {
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
        )
    }
    monkeypatch.setattr(
        "agent.chat_completion_helpers.build_api_kwargs",
        lambda _agent, _messages: payload,
    )
    turn = _dynamic_turn()
    active = activate_turn(turn)
    try:
        filtered = AIAgent._build_api_kwargs(
            object.__new__(AIAgent),
            [{"role": "user", "content": "读取资料"}],
        )
    finally:
        deactivate_turn(active)

    assert _tool_names(filtered) == ["mystand_resource_index"]
    assert _tool_names(payload) == [
        "mystand_resource_index",
        "mystand_query",
        "mystand_authorization",
    ]


def test_bedrock_filters_canonical_tools_before_wire_conversion():
    turn = _dynamic_turn()
    agent = SimpleNamespace(
        tools=[
            *_tools("mystand_resource_index"),
            _canonical_query_tool(),
            *_tools("mystand_authorization", "terminal"),
        ],
        _ephemeral_tool_choice="",
        api_mode="bedrock_converse",
        _get_transport=lambda: BedrockTransport(),
        _bedrock_region="us-east-1",
        _bedrock_guardrail_config=None,
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=512,
    )
    active = activate_turn(turn)
    try:
        discover = build_api_kwargs(
            agent,
            [{"role": "user", "content": "读取资料"}],
        )
        discover_names = [
            item["toolSpec"]["name"]
            for item in discover["toolConfig"]["tools"]
        ]
        assert discover_names == ["mystand_resource_index"]

        _record_found_index(turn)
        read = build_api_kwargs(
            agent,
            [{"role": "user", "content": "读取资料"}],
        )
        read_names = [
            item["toolSpec"]["name"]
            for item in read["toolConfig"]["tools"]
        ]
        assert read_names == [
            "mystand_query",
            "mystand_authorization",
        ]

        turn.completion_finalization = "failure"
        finalized = build_api_kwargs(
            agent,
            [{"role": "user", "content": "读取资料"}],
        )
        assert "toolConfig" not in finalized
    finally:
        deactivate_turn(active)
