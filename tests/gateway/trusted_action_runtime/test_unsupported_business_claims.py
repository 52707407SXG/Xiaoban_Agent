"""波次 0 RED：在真实运行出口重放事故结构。

出口是被测的真实函数 ``gateway.platforms.api_server._guard_evidence_backed_response``
（My Stand 非流式与流式缓冲响应的共同出口闸），不是提示词字符串检查。

波次 0 基线预期：标注 RED 的用例在基线上失败（出口放行无证据业务陈述）；
标注 回归 的用例在基线上已通过，用来防止波次 1 误伤正常行为。
"""

import pytest

from gateway.platforms.api_server import _guard_evidence_backed_response

from tests.gateway.trusted_action_runtime import incident_fixtures as fx


def _egress(scenario):
    return _guard_evidence_backed_response(
        scenario["answer"],
        user_message=scenario["user_message"],
        conversation_history=scenario.get("conversation_history") or [],
        result=scenario["result"],
    )


def _assert_blocked(guarded: str, answer: str, forged_tokens=fx.FABRICATED_TOKENS):
    assert guarded != answer, "出口原样放行了无证据业务陈述"
    for token in forged_tokens:
        assert token not in guarded, f"编造事实 {token!r} 泄漏到最终回答"
    assert "call_" not in guarded, "出口向用户暴露了内部调用 ID"


# --- RED：事故场景 1，零工具调用却给出人名、金额、状态 ---
def test_red_zero_call_fabrication_is_blocked():
    guarded = _egress(fx.SCENARIO_ZERO_CALL_FABRICATION)
    _assert_blocked(guarded, fx.SCENARIO_ZERO_CALL_FABRICATION["answer"])


# --- RED：事故场景 2，工具全部失败后仍编造业务结果 ---
def test_red_all_tools_failed_cannot_be_turned_into_success():
    guarded = _egress(fx.SCENARIO_ALL_TOOLS_FAILED)
    assert guarded == fx.ERROR_MESSAGE


# --- RED：事故场景 3，索引成功不能冒充业务动作成功 ---
def test_red_index_success_does_not_prove_business_result():
    scenario = fx.SCENARIO_INDEX_OK_BUSINESS_FAILED
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"], scenario["forged_tokens"])


# --- RED：事故场景 4，连续追问不得补充新假事实 ---
def test_red_followup_cannot_add_new_fabricated_facts():
    scenario = fx.SCENARIO_FOLLOWUP_NEW_FABRICATION
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"], scenario["forged_tokens"])


# --- RED：事故场景 5，没有新的真实验证不得声称"重新核验了" ---
def test_red_fake_reverification_is_blocked():
    scenario = fx.SCENARIO_FAKE_REVERIFICATION
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"], scenario["forged_tokens"])
    assert "核验" not in guarded or "没有" in guarded or "未" in guarded


# --- RED：事故场景 6，自然业务问法绕过旧正则后仍必须拦截 ---
def test_red_natural_phrasing_without_markers_is_blocked():
    scenario = fx.SCENARIO_NATURAL_PHRASING_BYPASS
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"], scenario["forged_tokens"])


# --- RED：事故场景 7，旧历史事实不得污染当前回合 ---
def test_red_stale_history_facts_do_not_enter_current_turn():
    scenario = fx.SCENARIO_STALE_HISTORY_POLLUTION
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"], scenario["forged_tokens"])


# --- RED：事故场景 8，ActionResult 各失败状态不得转成业务成功 ---
@pytest.mark.parametrize(
    "status_name,expected_message",
    [
        ("error", fx.ERROR_MESSAGE),
        ("denied", fx.DENIED_MESSAGE),
        ("ambiguous", fx.AMBIGUOUS_MESSAGE),
        ("not_found", fx.NOT_FOUND_MESSAGE),
    ],
)
def test_red_failed_result_status_never_becomes_success(status_name, expected_message):
    scenario = fx.SCENARIO_RESULT_STATUSES
    result = fx.result_status_turn(scenario["cases"][status_name]["receipt"])
    guarded = _guard_evidence_backed_response(
        scenario["answer"],
        user_message=scenario["user_message"],
        conversation_history=[],
        result=result,
    )
    assert guarded == expected_message


def test_red_empty_result_status_never_becomes_success():
    scenario = fx.SCENARIO_RESULT_STATUSES
    result = fx.result_status_turn(scenario["cases"]["empty"]["receipt"])
    guarded = _guard_evidence_backed_response(
        scenario["answer"],
        user_message=scenario["user_message"],
        conversation_history=[],
        result=result,
    )
    _assert_blocked(guarded, scenario["answer"])


# --- RED：事故场景 9a，伪造 callId 的回执不得成为证据 ---
def test_red_forged_call_id_receipt_is_not_evidence():
    scenario = fx.SCENARIO_FORGED_CALL_ID
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"])


# --- RED：事故场景 9b，跨回合 callId 的证据不得进入本轮 ---
def test_red_cross_turn_call_id_is_not_current_evidence():
    scenario = fx.SCENARIO_CROSS_TURN_CALL_ID
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"], scenario["forged_tokens"])


# --- RED：事故场景 9c，跨账号 evidence 不得进入本轮 ---
def test_red_cross_account_evidence_is_rejected():
    scenario = fx.SCENARIO_CROSS_ACCOUNT_EVIDENCE
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"])


# --- RED：回答数值与本轮证据不一致必须拦截 ---
def test_red_answer_facts_must_match_current_evidence():
    scenario = fx.SCENARIO_FACT_MISMATCH
    guarded = _egress(scenario)
    _assert_blocked(guarded, scenario["answer"], scenario["forged_tokens"])


# --- 回归：正常闲聊不得被误伤成生硬工作流 ---
def test_regression_plain_chat_is_not_rewritten():
    scenario = fx.SCENARIO_PLAIN_CHAT
    assert _egress(scenario) == scenario["answer"]


# --- 回归：本轮真实证据支持的业务回答必须放行 ---
def test_regression_evidence_backed_answer_passes():
    scenario = fx.SCENARIO_EVIDENCE_BACKED_ANSWER
    assert _egress(scenario) == scenario["answer"]


# --- 回归：无权限语义只来自 ActionResult，不沿用模型补充建议 ---
def test_regression_denied_result_uses_runtime_message():
    scenario = fx.SCENARIO_HONEST_DENIAL_ADMISSION
    assert _egress(scenario) == fx.DENIED_MESSAGE
