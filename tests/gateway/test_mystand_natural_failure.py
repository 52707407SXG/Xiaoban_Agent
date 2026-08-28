"""Regression tests for natural-language My Stand failure delivery."""

from gateway.platforms.api_server import _finalize_mystand_egress_result


def _provider_failure() -> dict:
    return {
        "schema": "xiaoban.agent-failure.v1",
        "kind": "fatal",
        "code": "provider_call_failed",
        "phase": "provider_call",
        "reason": "private provider diagnostic",
        "retryable": True,
    }


def test_empty_failed_turn_gets_natural_language_reply():
    result = {
        "completed": False,
        "failed": True,
        "final_response": "",
        "failure": _provider_failure(),
    }

    visible = _finalize_mystand_egress_result(
        result,
        user_message="查一下天气",
    )

    assert "这次任务没有完成" in visible
    assert "模型服务调用失败" in visible
    assert "private provider diagnostic" not in visible
    assert result["final_response"] == visible


def test_existing_natural_reply_is_preserved():
    result = {
        "completed": False,
        "failed": True,
        "final_response": "这次查询没有成功，我正在说明原因。",
        "failure": _provider_failure(),
    }

    visible = _finalize_mystand_egress_result(
        result,
        user_message="查一下天气",
    )

    assert visible == "这次查询没有成功，我正在说明原因。"


def test_stopped_turn_gets_brief_acknowledgement():
    result = {
        "completed": False,
        "stopped": True,
        "final_response": "",
    }

    visible = _finalize_mystand_egress_result(
        result,
        user_message="停止",
    )

    assert visible == "好的，当前任务已经停止。"
