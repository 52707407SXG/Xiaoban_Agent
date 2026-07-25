"""波次 1 GREEN：Trusted Action Runtime 核心机制单元测试。

覆盖：callId 严格绑定、ActionResult 全状态、跨回合/跨账号/伪造证据排除、
真实状态机、IndexReceipt、CHAT/WORK 分类与事实绑定。
"""

from xiaoban.trusted_runtime import (
    PLATFORM_CLI,
    TrustedIdentity,
    build_work_turn,
    check_completion,
    classify_interaction,
)
from xiaoban.trusted_runtime.completion_guard import (
    ERROR_MESSAGE,
    FACT_MISMATCH_MESSAGE,
    NO_EVIDENCE_MESSAGE,
    VERIFICATION_BLOCK_MESSAGE,
)

from tests.gateway.trusted_action_runtime import incident_fixtures as fx

IDENTITY = TrustedIdentity(
    account_id="user-a", data_scope="mystand", source="server_session"
)


def _turn(scenario_result, user_message="帮我看看滨江一号3栋802的业主和月供"):
    return build_work_turn(
        channel="web",
        user_message=user_message,
        conversation_history=[],
        result=scenario_result,
        identity=IDENTITY,
    )


def test_action_result_binds_strictly_to_current_turn_call_id():
    turn = _turn(fx.SCENARIO_FORGED_CALL_ID["result"])
    assert turn.action_calls == []
    assert turn.action_results == []
    assert turn.orphaned_receipts == 1
    assert turn.evidence == []


def test_cross_turn_call_id_never_enters_current_evidence():
    turn = _turn(
        fx.SCENARIO_CROSS_TURN_CALL_ID["result"],
        user_message=fx.SCENARIO_CROSS_TURN_CALL_ID["user_message"],
    )
    assert turn.action_calls == []
    assert turn.evidence == []


def test_cross_account_receipt_is_rejected_even_when_successful():
    turn = _turn(fx.SCENARIO_CROSS_ACCOUNT_EVIDENCE["result"])
    assert turn.rejected_cross_account == 1
    assert turn.evidence == []


def test_result_statuses_are_classified_distinctly():
    cases = fx.SCENARIO_RESULT_STATUSES["cases"]
    expected = {
        "empty": "empty",
        "error": "error",
        "denied": "denied",
        "ambiguous": "ambiguous",
        "not_found": "not_found",
    }
    for name, case in cases.items():
        turn = _turn(fx.result_status_turn(case["receipt"]))
        assert [item.status for item in turn.action_results] == [expected[name]]


def test_verifying_state_only_after_real_post_action_verify():
    turn = _turn(fx.SCENARIO_FORGED_CALL_ID["result"])
    assert "verifying" not in turn.states

    turn_ok = _turn(fx.SCENARIO_EVIDENCE_BACKED_ANSWER["result"])
    assert "executing" in turn_ok.states
    assert "verifying" in turn_ok.states
    assert turn_ok.states.index("verifying") > turn_ok.states.index("executing")


def test_terminal_state_blocked_on_guard_rejection():
    turn = _turn(fx.SCENARIO_ZERO_CALL_FABRICATION["result"])
    decision = check_completion(fx.FABRICATED_ANSWER, turn)
    assert not decision.allowed
    turn.enter("blocked")
    turn.terminal_reason = decision.reason
    assert turn.state == "blocked"
    assert turn.terminal_reason.startswith("blocked_")


def test_index_receipt_reflects_real_index_outcome():
    turn = _turn(fx.SCENARIO_INDEX_OK_BUSINESS_FAILED["result"])
    assert turn.index_receipt is not None
    assert turn.index_receipt.status == "found"
    assert turn.index_receipt.matched_resource_refs

    chat_turn = _turn(
        fx.SCENARIO_PLAIN_CHAT["result"],
        user_message=fx.SCENARIO_PLAIN_CHAT["user_message"],
    )
    assert chat_turn.index_receipt.status == "no_internal_resource_needed"


def test_chat_and_work_classification_fail_closed_to_work():
    assert classify_interaction("今天有点累，陪我聊两句", []) == "CHAT"
    assert classify_interaction("小张这个月提成多少", []) == "WORK"
    # 无法可靠区分时默认 WORK（本回合调用了业务工具）
    assert (
        classify_interaction("嗯", [], used_business_tools=True) == "WORK"
    )


def test_business_facts_must_exist_in_current_evidence():
    turn = _turn(fx.SCENARIO_EVIDENCE_BACKED_ANSWER["result"])
    decision = check_completion(fx.SCENARIO_EVIDENCE_BACKED_ANSWER["answer"], turn)
    assert decision.allowed

    mismatch_turn = _turn(fx.SCENARIO_FACT_MISMATCH["result"])
    mismatch = check_completion(fx.SCENARIO_FACT_MISMATCH["answer"], mismatch_turn)
    assert not mismatch.allowed
    assert mismatch.text == FACT_MISMATCH_MESSAGE


def test_verification_claim_requires_real_verified_evidence():
    turn = _turn(
        fx.SCENARIO_FAKE_REVERIFICATION["result"],
        user_message=fx.SCENARIO_FAKE_REVERIFICATION["user_message"],
    )
    decision = check_completion(fx.SCENARIO_FAKE_REVERIFICATION["answer"], turn)
    assert not decision.allowed
    assert decision.text == VERIFICATION_BLOCK_MESSAGE
    assert decision.reason == "blocked_verification_claim"


def test_no_action_call_means_no_query_claim():
    turn = _turn(fx.SCENARIO_ZERO_CALL_FABRICATION["result"])
    decision = check_completion(fx.FABRICATED_ANSWER, turn)
    assert not decision.allowed
    assert decision.reason == "blocked_no_action_call"
    assert decision.text == NO_EVIDENCE_MESSAGE


def test_failed_action_cannot_be_rephrased_as_success():
    turn = _turn(fx.SCENARIO_ALL_TOOLS_FAILED["result"])
    decision = check_completion(fx.FABRICATED_ANSWER, turn)
    assert not decision.allowed
    assert decision.text == ERROR_MESSAGE


def test_cli_platform_has_no_independent_business_path():
    from xiaoban.trusted_runtime import evaluate_channel_answer
    from xiaoban.trusted_runtime.types import CommandEnvelope

    envelope = CommandEnvelope(
        request_id="cli-1",
        platform=PLATFORM_CLI,
        conversation_id="cli-session",
        message_id="cli-msg-1",
        external_user_ref="local-user",
        text=fx.SCENARIO_ZERO_CALL_FABRICATION["user_message"],
    )
    decision = evaluate_channel_answer(
        envelope,
        final_text=fx.FABRICATED_ANSWER,
        user_message=envelope.text,
        result=fx.SCENARIO_ZERO_CALL_FABRICATION["result"],
    )
    assert decision.reason == "not_mystand"
