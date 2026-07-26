"""阻断修复 R0 RED：暴露执行前门禁与证据绑定的真实绕过。

每条用例都打在真实入口上（出口 Guard、WorkTurn 构建、渠道评估），
不是源码文案检查。在 4a5a0ef 当前实现上必须失败；失败原因标注在每
条用例 docstring 里（行为放行 / 机制缺失）。GREEN 后同一断言必须
不加修改地通过。
"""

import pytest

from xiaoban.trusted_runtime import (
    PLATFORM_CLI,
    TrustedIdentity,
    build_work_turn,
    envelope_from_wechat_event,
    evaluate_channel_answer,
)
from xiaoban.trusted_runtime.completion_guard import check_mystand_final_answer
from xiaoban.trusted_runtime.types import CommandEnvelope

from tests.gateway.trusted_action_runtime import incident_fixtures as fx

IDENTITY = TrustedIdentity(
    account_id="user-a", data_scope="mystand", source="server_session"
)
BUSINESS_MSG = "查一下游某今年的结算业绩"
EVIDENCE_ANSWER = "游某今年结算业绩是 32105.68 元。"


def _result(calls, user_message=BUSINESS_MSG, user_id="user-a"):
    return fx.tool_turn(user_message, calls, user_id=user_id)


# --- R0-1：My Stand WORK 没有 IndexReceipt，业务动作结果被错误采信 ---
def test_red_work_without_index_receipt_must_not_allow():
    # 当前行为：无索引前置，query 成功即 allowed_evidence_backed。
    result = _result(
        [("call_q", "mystand_query", {"operation": "read"}, {"ok": True, "content": "游某 2026 结算业绩 32105.68 元"})]
    )
    decision = check_mystand_final_answer(
        EVIDENCE_ANSWER,
        user_message=BUSINESS_MSG,
        conversation_history=[],
        result=result,
        account_id="user-a",
    )
    assert not decision.allowed, "无 IndexReceipt 的 My Stand WORK 被错误放行"
    assert "32105.68" not in decision.text


# --- R0-2：无证据的纯人名事实（无数字、无 claim 动词）被错误放行 ---
def test_red_pure_person_name_without_evidence_must_block():
    # 当前行为：claim 动词与数字正则都miss，allowed_no_claims 原样放行。
    result = _result((), user_message="滨江一号3栋802的业主是谁")
    decision = check_mystand_final_answer(
        "就是周某本人。",
        user_message="滨江一号3栋802的业主是谁",
        conversation_history=[],
        result=result,
        account_id="user-a",
    )
    assert not decision.allowed, "纯人名业务事实绕过正则 Guard"
    assert "周某" not in decision.text


# --- R0-3：嵌套 payload 中的 owner/actor 与当前身份冲突仍被采信 ---
def test_red_nested_owner_mismatch_must_reject_evidence():
    # 当前行为：只查顶层 accountId/userId/ownerId，嵌套 items[].ownerId 漏检。
    result = _result(
        [("call_n", "mystand_query", {"operation": "read"}, {
            "ok": True,
            "content": "游某 2026 结算业绩 32105.68 元",
            "items": [{"ownerId": "user-b", "resourceUid": "res-demo-9"}],
        })]
    )
    turn = build_work_turn(
        channel="web",
        user_message=BUSINESS_MSG,
        conversation_history=[],
        result=result,
        identity=IDENTITY,
    )
    assert turn.evidence == [], "嵌套越权 owner 的 payload 生成了 Evidence"


# --- R0-4：ok=true 叠加错误状态 / 长文本被当成 success ---
def test_red_ok_true_with_error_status_is_not_success():
    # 当前行为：ok is True 直接 success，不看 status。
    result = _result(
        [("call_e", "mystand_query", {"operation": "read"}, {
            "ok": True, "status": 500,
            "content": "upstream bridge timeout while reading ledger",
        })]
    )
    turn = build_work_turn(
        channel="web", user_message=BUSINESS_MSG, result=result, identity=IDENTITY,
    )
    assert turn.action_results[0].status == "error"
    assert turn.evidence == []


def test_red_long_unstructured_text_is_not_success():
    # 当前行为：无法解析的长文本（>=20 字符）直接 success。
    result = _result(
        [("call_t", "mystand_query", {"operation": "read"}, "业主是周某，月供 5600 元，状态正常")]
    )
    turn = build_work_turn(
        channel="web", user_message=BUSINESS_MSG, result=result, identity=IDENTITY,
    )
    assert turn.action_results[0].status == "error"
    assert turn.evidence == []


# --- R0-5：微信/飞书未绑定身份，回退读取 result 自报身份后放行 ---
def test_red_unbound_wechat_must_not_fall_back_to_result_identity():
    # 当前行为：envelope 无 boundAccountId 时仍用 result._mystand_user_id 放行。
    envelope = envelope_from_wechat_event(
        {
            "requestId": "req-1",
            "messageId": "msg-1",
            "conversationId": "conv-1",
            "fromUser": "wx-stranger",
            "text": BUSINESS_MSG,
        }
    )
    result = _result(
        [("call_q", "mystand_query", {"operation": "read"}, {"ok": True, "content": "游某 2026 结算业绩 32105.68 元"})]
    )
    decision = evaluate_channel_answer(
        envelope,
        final_text=EVIDENCE_ANSWER,
        user_message=BUSINESS_MSG,
        conversation_history=[],
        result=result,
    )
    assert not decision.allowed, "未绑定渠道身份被错误放行"
    assert "32105.68" not in decision.text


# --- R0-6：CLI 请求业务事实仍被放行 ---
def test_red_cli_business_request_has_no_success_path():
    # 当前行为：CLI 直接 allowed=True 原样放行业务回答。
    envelope = CommandEnvelope(
        request_id="cli-1",
        platform=PLATFORM_CLI,
        conversation_id="cli-session",
        message_id="cli-msg-1",
        external_user_ref="local-user",
        text=BUSINESS_MSG,
    )
    decision = evaluate_channel_answer(
        envelope,
        final_text=fx.FABRICATED_ANSWER,
        user_message=BUSINESS_MSG,
        conversation_history=[],
        result=fx.SCENARIO_ZERO_CALL_FABRICATION["result"],
    )
    assert not decision.allowed, "CLI 仍存在业务成功/业务事实路径"
    for token in fx.FABRICATED_TOKENS:
        assert token not in decision.text


# --- R0-7：重复 callId 两次调用，dict 覆盖后错误成功 ---
def test_red_duplicate_call_id_must_block_not_overwrite():
    # 当前行为：calls[call_id] 静默覆盖，后一次调用顶替前一次。
    result = {
        "_mystand_request": True,
        "_mystand_user_id": "user-a",
        "messages": [
            {"role": "user", "content": BUSINESS_MSG},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_dup", "type": "function",
                "function": {"name": "mystand_query", "arguments": '{"operation":"read"}'},
            }]},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_dup", "type": "function",
                "function": {"name": "mystand_query", "arguments": '{"operation":"read"}'},
            }]},
            {"role": "tool", "name": "mystand_query", "tool_call_id": "call_dup",
             "content": '{"ok":true,"content":"游某 2026 结算业绩 32105.68 元"}'},
        ],
    }
    turn = build_work_turn(
        channel="web", user_message=BUSINESS_MSG, result=result, identity=IDENTITY,
    )
    assert turn.evidence == [], "重复 callId 被静默覆盖后错误采信"


# --- R0-8：callId 匹配但 actionId 不一致的回执不得绑定 ---
def test_red_action_id_mismatch_receipt_must_not_bind():
    # 当前行为：只按 callId 绑定，回执 actionId 与调用不一致也采信。
    result = {
        "_mystand_request": True,
        "_mystand_user_id": "user-a",
        "messages": [
            {"role": "user", "content": BUSINESS_MSG},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_m", "type": "function",
                "function": {"name": "mystand_query", "arguments": '{"operation":"read"}'},
            }]},
            {"role": "tool", "name": "mystand_authorization", "tool_call_id": "call_m",
             "content": '{"ok":true,"content":"游某 2026 结算业绩 32105.68 元"}'},
        ],
    }
    turn = build_work_turn(
        channel="web", user_message=BUSINESS_MSG, result=result, identity=IDENTITY,
    )
    assert turn.evidence == [], "actionId 不一致的回执被错误绑定"


# --- R0-9：上一回合成功 evidence 不得被本轮失败复用（生命周期隔离）---
def test_red_previous_turn_evidence_cannot_be_reused():
    # 当前行为：机制缺失——没有生命周期级 turn 隔离 API（ImportError 即 RED）。
    from xiaoban.trusted_runtime.turns import (  # noqa: F401
        begin_action,
        begin_turn,
        finish_action,
    )

    turn_a = begin_turn(
        channel="web",
        user_message=BUSINESS_MSG,
        identity=IDENTITY,
        request_id="req-a",
        message_id="msg-a",
    )
    decision_a = begin_action(turn_a, "mystand_authorization", "v1", {"operation": "resolve", "resource_uid": "res-demo-1"})
    assert decision_a.decision == "allow"
    finish_action(turn_a, decision_a.call.call_id, "mystand_authorization", "v1", '{"ok":true,"content":"游某 2026 结算业绩 32105.68 元"}')
    assert turn_a.evidence

    turn_b = begin_turn(
        channel="web",
        user_message="那周某今年业绩呢？",
        identity=IDENTITY,
        request_id="req-b",
        message_id="msg-b",
    )
    decision_b = begin_action(turn_b, "mystand_authorization", "v1", {"operation": "resolve", "resource_uid": "res-demo-2"})
    finish_action(turn_b, decision_b.call.call_id, "mystand_authorization", "v1", '{"ok":false,"status":500,"error":"internal"}')
    assert turn_b.evidence == [], "上一回合 evidence 污染了本轮"


# --- R0-10：执行异常/超时回执被当成 success ---
def test_red_exception_receipt_is_not_success():
    # 当前行为：'{"error": "Tool execution failed: ..."}' 超 20 字符即 success。
    result = _result(
        [("call_x", "mystand_query", {"operation": "read"},
          '{"error": "Tool execution failed: TimeoutError: read timed out"}')]
    )
    turn = build_work_turn(
        channel="web", user_message=BUSINESS_MSG, result=result, identity=IDENTITY,
    )
    assert turn.action_results[0].status == "error"
    assert turn.evidence == []


# --- R0-13：没有真实 PostAction Verify 却产出 verified Evidence ---
def test_red_unverified_payload_must_not_be_marked_verified():
    # 当前行为：长文本归 success 后无条件 verification_status="verified"。
    result = _result(
        [("call_v", "mystand_query", {"operation": "read"}, "调试输出：读取过程中部分字段缺失，但文本足够长")]
    )
    turn = build_work_turn(
        channel="web", user_message=BUSINESS_MSG, result=result, identity=IDENTITY,
    )
    assert turn.evidence == []
    assert all(
        item.verification_status != "verified" for item in turn.evidence
    )


# --- R0-14：result 自报 _mystand_user_id 不得冒充服务端身份 ---
def test_red_result_self_reported_identity_is_not_trusted():
    # 当前行为：account_id 为空时回退读取 result._mystand_user_id。
    result = _result(
        [("call_q", "mystand_query", {"operation": "read"}, {"ok": True, "content": "游某 2026 结算业绩 32105.68 元"})],
        user_id="user-b",
    )
    decision = check_mystand_final_answer(
        EVIDENCE_ANSWER,
        user_message=BUSINESS_MSG,
        conversation_history=[],
        result=result,
        account_id="",
    )
    assert not decision.allowed, "result 自报身份被当成服务端身份放行"
    assert "32105.68" not in decision.text


# --- R0-15：未知 action / 无 output schema 的回执不得采信 ---
def test_red_unknown_action_payload_must_fail_closed():
    # 当前行为：任意工具名 + ok=true 即 success 并生成 Evidence。
    result = _result(
        [("call_u", "mystand_secret_dump", {}, {"ok": True, "content": "全部客户电话 13800001111"})]
    )
    turn = build_work_turn(
        channel="web", user_message=BUSINESS_MSG, result=result, identity=IDENTITY,
    )
    assert turn.action_results[0].status == "error"
    assert turn.evidence == []
    decision = check_mystand_final_answer(
        "客户电话是 13800001111。",
        user_message=BUSINESS_MSG,
        conversation_history=[],
        result=result,
        account_id="user-a",
    )
    assert not decision.allowed
    assert "13800001111" not in decision.text
