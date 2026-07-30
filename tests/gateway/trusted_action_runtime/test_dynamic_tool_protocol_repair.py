"""Regression tests for dynamic tool staging and finalize-only framing."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.conversation_loop import (
    _contains_raw_tool_protocol_content,
    _prepare_finalize_only_call,
    _reject_finalize_only_protocol_candidate,
)
from agent.chat_completion_helpers import build_api_kwargs
from agent.transports.bedrock import BedrockTransport
from agent.tool_executor import _trusted_preaction_denial
from gateway.session_context import clear_session_vars, set_session_vars
from gateway.platforms.api_server import (
    _run_mystand_preexecuted_evidence,
)
from tools.registry import ToolRegistry
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
    dynamic_transient_recovery_plan,
    dynamic_transient_recovery_tool_call_valid,
    mark_dynamic_execution_no_progress,
    mark_dynamic_read_no_progress,
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
    assert allowed.verification["completion_kind"] == "failure-bound"
    assert allowed.verification["failure_class"] == "no_progress"
    assert allowed.verification["failed_action_count"] == 0
    assert allowed.verification["action_count"] == 1
    assert allowed.verification["evidence_count"] == 0

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
    assert allowed.verification["failure_class"] == "no_progress"
    assert allowed.verification["action_count"] == 0
    assert allowed.verification["failed_action_count"] == 0

    natural_variant = (
        "这次我还没开始实际处理，因此我暂时无法完成这项任务。"
    )
    assert _complete_bound_failure(turn, natural_variant).allowed is True


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


def test_runtime_failure_examples_are_accepted_for_each_safe_class():
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
        allowed = _complete_bound_failure(turn, presentation["example"])
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
    allowed = _complete_bound_failure(turn, presentation["example"])
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


@pytest.mark.parametrize(
    "unbound_tail",
    [
        "但蓝鲸项目没有问题。",
        "但任意对象没有错误。",
        "但甲方状态已确认。",
        "但另一项结果正常。",
    ],
)
def test_natural_failure_rejects_unbound_positive_tail(unbound_tail):
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="unsafe-positive-tail",
    )
    assert started.decision == "allow"
    finish_action(
        turn,
        started.call.call_id,
        "mystand_resource_index",
        "v1",
        {"ok": False, "status": 500, "code": "handler_failed"},
    )
    reply = f"我这次处理失败，所以任务没完成，{unbound_tail}"

    assert _complete_bound_failure(turn, reply).allowed is False


@pytest.mark.parametrize(
    "unbound_tail",
    [
        "但蓝鲸项目失败了。",
        "但张三申请失败了。",
        "但远海计划有问题。",
        "但星河订单处理失败了。",
        "但李四失败了。",
        "但王五的工单被取消了。",
        "我查询的蓝鲸失败了。",
        "我查的张三失败了。",
        "我读取的对象出了问题。",
        "我查询的目标被取消了。",
    ],
)
def test_natural_failure_rejects_unbound_negative_tail(unbound_tail):
    turn = _dynamic_turn()
    started = begin_action(
        turn,
        "mystand_resource_index",
        "v1",
        {},
        call_id="unsafe-negative-tail",
    )
    assert started.decision == "allow"
    finish_action(
        turn,
        started.call.call_id,
        "mystand_resource_index",
        "v1",
        {"ok": False, "status": 500, "code": "handler_failed"},
    )
    reply = f"我这次处理失败，所以任务没完成，{unbound_tail}"

    assert _complete_bound_failure(turn, reply).allowed is False


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

    assert dynamic_transient_recovery_plan(turn) == {
        "reason": "unavailable",
        "state": "上一次只读处理遇到暂时不可用",
        "safe_scope": [
            {
                "resourceUid": "resource-protocol",
                "safeLabel": "目标资料",
                "resourceType": "generic-record",
                "canRead": True,
                "locked": False,
            }
        ],
        "retry": {
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
        },
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


def test_no_progress_failure_rejects_business_facts_and_internal_status():
    turn = _dynamic_turn()
    _record_found_index(turn)
    assert mark_dynamic_read_no_progress(turn) is True
    turn.completion_finalization = "failure"

    for text in (
        "系统提示：缺少动态证据回执，请点击重试。",
        "我没能完成读取，但还有 10 万元未结算。",
        "我已定位到相关资料，任务完成。",
        "我完成了资料目录查询，但没有继续读取到能回答问题的内容，"
        "所以这项任务还没有完成。你的工资已经发放。",
        "我完成了资料目录查询，但没有继续读取到能回答问题的内容，"
        "所以这项任务还没有完成。这处商铺已经出租。",
        "我完成了资料目录查询，但没有继续读取到能回答问题的内容，"
        "所以这项任务还没有完成。记录已经删除。",
        "我完成了资料目录查询，但没有继续读取到能回答问题的内容，"
        "所以这项任务还没有完成。老板已经批准。",
        "我完成了资料目录查询，但服务中断，所以这项任务还没有完成。",
        "我完成了资料目录查询，但权限不足，所以这项任务还没有完成。",
    ):
        turn.completion_finalization_output_digest = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        assert check_completion(text, turn).allowed is False


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
        "tools": _tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
            "terminal",
        ),
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
        assert dynamic_finalization_mode(
            turn,
            include_single_preaction=True,
        ) == ""
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
    assert "实际处理已发起，但执行返回了错误" in serialized
    assert "当前无权读取" not in serialized
    assert "自然中文" in messages[-1]["content"]


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
        tools=_tools(
            "mystand_resource_index",
            "mystand_query",
            "mystand_authorization",
            "terminal",
        ),
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
