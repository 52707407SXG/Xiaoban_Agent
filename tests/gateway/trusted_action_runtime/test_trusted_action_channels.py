"""波次 1 GREEN：web / 微信模拟 / 飞书模拟共享同一个 Runtime 的合同测试。

三个渠道必须得到相同的身份结论、CHAT/WORK 结论、证据结论与
CompletionDecision；CLI 不得拥有独立业务成功路径。
"""

from xiaoban.trusted_runtime import (
    TrustedIdentity,
    build_work_turn,
    check_completion,
    envelope_from_feishu_event,
    envelope_from_wechat_event,
    evaluate_channel_answer,
    identity_from_envelope,
)
from xiaoban.trusted_runtime.completion_guard import check_mystand_final_answer
from xiaoban.trusted_runtime.types import INTERACTION_WORK

from tests.gateway.trusted_action_runtime import incident_fixtures as fx

ACCOUNT = "user-a"


def _web_decision(scenario):
    return check_mystand_final_answer(
        scenario["answer"],
        user_message=scenario["user_message"],
        conversation_history=scenario.get("conversation_history") or [],
        result=scenario["result"],
        channel="web",
        account_id=ACCOUNT,
        request_id="req-1",
        message_id="msg-1",
    )


def _wechat_decision(scenario):
    envelope = envelope_from_wechat_event(
        {
            "requestId": "req-1",
            "messageId": "msg-1",
            "conversationId": "conv-1",
            "fromUser": "wx-open-id-demo",
            "text": scenario["user_message"],
            "boundAccountId": ACCOUNT,
        }
    )
    assert identity_from_envelope(envelope).account_id == ACCOUNT
    return evaluate_channel_answer(
        envelope,
        final_text=scenario["answer"],
        user_message=scenario["user_message"],
        conversation_history=scenario.get("conversation_history") or [],
        result=scenario["result"],
    )


def _feishu_decision(scenario):
    envelope = envelope_from_feishu_event(
        {
            "requestId": "req-1",
            "messageId": "msg-1",
            "chatId": "chat-1",
            "openId": "ou-demo",
            "text": scenario["user_message"],
            "boundAccountId": ACCOUNT,
        }
    )
    assert identity_from_envelope(envelope).account_id == ACCOUNT
    return evaluate_channel_answer(
        envelope,
        final_text=scenario["answer"],
        user_message=scenario["user_message"],
        conversation_history=scenario.get("conversation_history") or [],
        result=scenario["result"],
    )


def _assert_same_semantics(scenario):
    decisions = [
        _web_decision(scenario),
        _wechat_decision(scenario),
        _feishu_decision(scenario),
    ]
    signatures = [(d.allowed, d.text, d.reason) for d in decisions]
    assert signatures[0] == signatures[1] == signatures[2], (
        "三个渠道的业务结果、权限、证据或错误语义不一致: " + repr(signatures)
    )
    return decisions[0]


def test_channels_share_fabrication_block_semantics():
    decision = _assert_same_semantics(fx.SCENARIO_ZERO_CALL_FABRICATION)
    assert not decision.allowed


def test_channels_share_failure_status_semantics():
    decision = _assert_same_semantics(fx.SCENARIO_ALL_TOOLS_FAILED)
    assert not decision.allowed
    assert decision.text == fx.ERROR_MESSAGE


def test_channels_share_denied_semantics():
    scenario = dict(fx.SCENARIO_RESULT_STATUSES)
    scenario["result"] = fx.result_status_turn(
        fx.SCENARIO_RESULT_STATUSES["cases"]["denied"]["receipt"]
    )
    decision = _assert_same_semantics(scenario)
    assert not decision.allowed
    assert decision.text == fx.DENIED_MESSAGE


def test_channels_share_evidence_backed_success_semantics():
    decision = _assert_same_semantics(fx.SCENARIO_EVIDENCE_BACKED_ANSWER)
    assert decision.allowed
    assert decision.text == fx.SCENARIO_EVIDENCE_BACKED_ANSWER["answer"]


def test_channels_share_chat_semantics_without_rewrite():
    decision = _assert_same_semantics(fx.SCENARIO_PLAIN_CHAT)
    assert decision.allowed
    assert decision.text == fx.SCENARIO_PLAIN_CHAT["answer"]


def test_channels_share_cross_account_rejection_semantics():
    decision = _assert_same_semantics(fx.SCENARIO_CROSS_ACCOUNT_EVIDENCE)
    assert not decision.allowed


def test_unbound_channel_identity_fails_closed():
    envelope = envelope_from_wechat_event(
        {"messageId": "msg-9", "fromUser": "wx-stranger", "text": "查业主"}
    )
    assert identity_from_envelope(envelope) is None


# --- B7：三渠道共享同一个 Runtime 的完整核心产物，不只是合成 decision ---
def _channel_products(scenario):
    products = []
    identities = {
        "web": TrustedIdentity(
            account_id=ACCOUNT, data_scope="mystand", source="server_session"
        ),
        "weixin": identity_from_envelope(
            envelope_from_wechat_event(
                {
                    "requestId": "req-1",
                    "messageId": "msg-1",
                    "conversationId": "conv-1",
                    "fromUser": "wx-open-id-demo",
                    "text": scenario["user_message"],
                    "boundAccountId": ACCOUNT,
                }
            )
        ),
        "feishu": identity_from_envelope(
            envelope_from_feishu_event(
                {
                    "requestId": "req-1",
                    "messageId": "msg-1",
                    "chatId": "chat-1",
                    "openId": "ou-demo",
                    "text": scenario["user_message"],
                    "boundAccountId": ACCOUNT,
                }
            )
        ),
    }
    for channel in ("web", "weixin", "feishu"):
        identity = identities[channel]
        turn = build_work_turn(
            channel=channel,
            user_message=scenario["user_message"],
            conversation_history=scenario.get("conversation_history") or [],
            result=scenario["result"],
            identity=identity,
            request_id="req-1",
            message_id="msg-1",
        )
        decision = check_completion(scenario["answer"], turn)
        receipt = turn.index_receipt
        products.append(
            {
                "account": (identity.account_id, identity.data_scope),
                "kind": turn.interaction_kind,
                "receipt": (receipt.status if receipt else None),
                "actions": [call.action_id for call in turn.action_calls],
                "bound": sorted(
                    result.call_id for result in turn.action_results
                )
                == sorted(call.call_id for call in turn.action_calls),
                "statuses": [item.status for item in turn.action_results],
                "facts": sorted(item.allowed_facts for item in turn.evidence),
                "refs": sorted(
                    ref for item in turn.evidence for ref in item.record_refs
                ),
                "decision": (decision.allowed, decision.text, decision.reason),
            }
        )
    return products


def test_channels_share_full_runtime_products_for_success():
    products = _channel_products(fx.SCENARIO_EVIDENCE_BACKED_ANSWER)
    assert products[0] == products[1] == products[2]
    product = products[0]
    assert product["kind"] == INTERACTION_WORK
    assert product["bound"] is True
    assert product["statuses"] == ["success"]
    assert product["facts"], "三渠道必须共享相同 Evidence 字段路径"
    assert product["decision"][0] is True


def test_channels_share_full_runtime_products_for_failure():
    products = _channel_products(fx.SCENARIO_ALL_TOOLS_FAILED)
    assert products[0] == products[1] == products[2]
    product = products[0]
    assert product["statuses"] == ["error"]
    assert product["facts"] == []
    assert product["decision"] == (False, fx.ERROR_MESSAGE, "blocked_no_evidence")
