import copy
import json

from gateway.platforms.true_moa_stop_projection import (
    _stopped_chat_completion_response,
)


def test_normal_stop_projection_preserves_completed_call_accounting():
    execution_id = "a" * 32
    completed_call = {
        "callId": f"{execution_id}:call:000001",
        "ordinal": 1,
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "role": "agent",
        "startedAtMs": 100,
        "endedAtMs": 120,
        "status": "completed",
        "inputTokens": 11,
        "outputTokens": 4,
        "totalTokens": 15,
        "cachedInputTokens": 2,
        "usageStatus": "reported",
        "costUsd": 0.0042,
        "costStatus": "reported",
        "costSource": "provider_usage",
    }
    running_call = {
        "callId": f"{execution_id}:call:000002",
        "ordinal": 2,
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "role": "agent",
        "startedAtMs": 130,
        "endedAtMs": None,
        "status": "running",
        "inputTokens": None,
        "outputTokens": None,
        "totalTokens": None,
        "cachedInputTokens": None,
        "usageStatus": "unavailable",
    }
    ledger = {
        "schema": "mystand.agent-call-usage.v1",
        "executionId": execution_id,
        "status": "running",
        "calls": [completed_call, running_call],
    }
    original = copy.deepcopy(ledger)

    result, usage = _stopped_chat_completion_response(
        (
            {
                "final_response": "PRIVATE_LATE_RESULT",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "PRIVATE_LATE_TRANSCRIPT",
                    }
                ],
                "completed": True,
                "_agent_call_usage": ledger,
            },
            {
                "input_tokens": 11,
                "output_tokens": 4,
                "total_tokens": 15,
                "agent_calls": ledger,
            },
        )
    )

    assert "PRIVATE_LATE" not in json.dumps(
        {"result": result, "usage": usage},
        ensure_ascii=False,
    )
    assert result["final_response"] == ""
    assert result["messages"] == []
    assert result["interrupted"] is True
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 4
    assert usage["total_tokens"] == 15

    projected = usage["agent_calls"]
    assert result["_agent_call_usage"] == projected
    assert projected["status"] == "cancelled"
    assert projected["calls"][0] == completed_call
    assert projected["calls"][1]["status"] == "timed_out"
    assert (
        projected["calls"][1]["errorCategory"]
        == "completion_stopped"
    )
    assert ledger == original
