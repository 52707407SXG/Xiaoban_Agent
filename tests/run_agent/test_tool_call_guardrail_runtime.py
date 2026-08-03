"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import sanitize_api_messages
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("xiaoban_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    completes = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_complete_callback = (
        lambda *a, **k: completes.append((a, k))
    )
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert messages[0]["_xiaoban_tool_result"]["continuation"]["guardrail"][
        "code"
    ] == "repeated_exact_failure_warning"
    assert "repeated_exact_failure_warning" not in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    completes = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_complete_callback = (
        lambda *a, **k: completes.append((a, k))
    )
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == [(("c-block", "web_search", args), {})]
    assert len(completes) == 1
    assert completes[0][0][:3] == ("c-block", "web_search", args)
    streamed_result = json.loads(completes[0][0][3])
    assert streamed_result["status"] == "failed"
    assert streamed_result["phase"] == "preflight"
    assert [event[0][0] for event in progress] == [
        "tool.started",
        "tool.completed",
    ]
    assert progress[-1][1]["is_error"] is True
    assert progress[-1][1]["phase"] == "preflight"
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


@pytest.mark.parametrize(
    "executor_name",
    ["_execute_tool_calls_sequential", "_execute_tool_calls_concurrent"],
)
def test_after_call_guidance_uses_trusted_sidecar_without_exposing_not_found_raw(
    executor_name,
):
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    private_result = json.dumps(
        {"ok": False, "status": 404, "private": "HIDE-ME"}
    )

    with patch("run_agent.handle_function_call", return_value=private_result):
        getattr(agent, executor_name)(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "HIDE-ME" in messages[0]["content"]
    assert "repeated_exact_failure_warning" not in messages[0]["content"]
    continuation = messages[0]["_xiaoban_tool_result"]["continuation"]
    assert continuation["kind"] == "tool_guardrail"
    assert continuation["guardrail"]["code"] == (
        "repeated_exact_failure_warning"
    )

    provider_messages = sanitize_api_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c-warn",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps(args),
                        },
                    }
                ],
            },
            messages[0],
        ]
    )
    projected = json.loads(provider_messages[-1]["content"])
    assert projected["outcome"] == "not_found"
    assert projected["modelError"] == {"code": "not_found"}
    assert projected["continuation"] == continuation
    assert "HIDE-ME" not in provider_messages[-1]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    provider_messages = sanitize_api_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c-recover",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps({"command": "bad-3"}),
                        },
                    }
                ],
            },
            messages[0],
        ]
    )
    content = provider_messages[-1]["content"]
    assert content.count("same_tool_failure_warning") == 1
    assert content.count("Do not switch to text-only replies") == 1
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [
        ("c-block", "web_search", blocked_args),
        ("c-allow", "web_search", allowed_args),
    ]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [
        ("tool.started", "web_search", blocked_args, {}),
        ("tool.started", "web_search", allowed_args, {}),
    ]
    assert len(completed_events) == 2
    assert completed_events[0][1] == "web_search"
    assert completed_events[0][3]["is_error"] is True
    assert completed_events[0][3]["phase"] == "preflight"
    assert completed_events[1][3]["is_error"] is False


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("xiaoban_cli.plugins.get_pre_tool_call_block_message", return_value="plugin policy"),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_messages = [m for m in result["messages"] if m.get("role") == "tool"]
    assert any(
        message.get("_xiaoban_tool_result", {})
        .get("continuation", {})
        .get("guardrail", {})
        .get("code")
        == "repeated_exact_failure_warning"
        for message in tool_messages
    )
    third_sample = agent.client.chat.completions.create.call_args_list[2].kwargs[
        "messages"
    ]
    projected_tool_results = [
        json.loads(message["content"])
        for message in third_sample
        if message.get("role") == "tool"
    ]
    latest_projection = projected_tool_results[-1]
    assert latest_projection["continuation"]["guardrail"]["code"] == (
        "repeated_exact_failure_warning"
    )
    assert json.dumps(latest_projection).count(
        "repeated_exact_failure_warning"
    ) == 1


def test_config_enabled_hard_stop_run_conversation_returns_failed_human_outcome():
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 2
    assert result["api_calls"] == 3
    assert result["api_calls"] < agent.max_iterations
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["error"] == "tool execution did not produce a usable result"
    assert result["completed"] is False
    assert result["failed"] is True
    # L4 leaves business-task interpretation with the model. The generic loop
    # guardrail may halt repeated physical calls, but it no longer fabricates a
    # separate whole-turn outcome/lineage receipt.
    assert "tool_outcome" not in result
    assert "连续 2 次采用同一处理方式" in result["final_response"]
    assert "仍缺少可靠结果" in result["final_response"]
    assert result["guardrail"]["code"] == "repeated_exact_failure_block"
    assert result["guardrail"]["tool_name"] == "web_search"

    assistant_tool_calls = [m for m in result["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    for assistant_msg in assistant_tool_calls:
        call_ids = [tc["id"] for tc in assistant_msg["tool_calls"]]
        following_results = [m for m in result["messages"] if m.get("role") == "tool" and m.get("tool_call_id") in call_ids]
        assert len(following_results) == len(call_ids)


def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "仍缺少可靠结果" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )
