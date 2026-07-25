from types import SimpleNamespace

from agent.chat_completion_helpers import _apply_ephemeral_tool_choice


def test_ephemeral_tool_choice_is_forced_once_then_consumed():
    agent = SimpleNamespace(_ephemeral_tool_choice="mystand_authorization")
    first = _apply_ephemeral_tool_choice(
        agent,
        {"tools": [{"type": "function"}]},
    )
    second = _apply_ephemeral_tool_choice(
        agent,
        {"tools": [{"type": "function"}]},
    )

    assert first["tool_choice"] == {
        "type": "function",
        "function": {"name": "mystand_authorization"},
    }
    assert "tool_choice" not in second
    assert agent._ephemeral_tool_choice == ""
