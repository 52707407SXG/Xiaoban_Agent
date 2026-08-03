"""One canonical table for the ToolResult contract at the executor boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import (
    append_trusted_steer_to_tool_message,
    sanitize_api_messages,
)
from agent.prompt_builder import STEER_MARKER_OPEN
from agent.tool_executor import _append_canonical_tool_result
from run_agent import AIAgent
from xiaoban_state import SessionDB


def test_canonical_append_owns_terminal_callback_metadata():
    messages = []
    callbacks = []

    def _complete(call_id, tool_name, tool_args, tool_result, metadata):
        # The transcript commit is the truth source: observers must never see
        # a terminal callback before its canonical sidecar exists.
        committed = messages[-1]["_xiaoban_tool_result"]
        assert all(committed[key] == value for key, value in metadata.items())
        callbacks.append((call_id, tool_name, tool_args, tool_result, metadata))

    agent = SimpleNamespace(
        _current_request_id="request-e1a",
        _current_turn_id="turn-e1a",
        tool_complete_callback=_complete,
    )
    stored = _append_canonical_tool_result(
        agent,
        messages,
        "mystand_example_read",
        '{"private":"stored-only"}',
        "call-e1a",
        dispatch_state="dispatched",
        outcome_hint="unknown",
        trusted_fields={
            "recordRefs": ["private-record-ref"],
            "continuation": {"private": "must-not-reach-lifecycle"},
        },
        function_args={"privateArg": "legacy-callback-only"},
        callback_result="PRIVATE_LEGACY_RESULT",
        emit_terminal_callback=True,
    )

    assert len(callbacks) == 1
    call_id, tool_name, tool_args, tool_result, metadata = callbacks[0]
    assert (call_id, tool_name) == ("call-e1a", "mystand_example_read")
    assert tool_args == {"privateArg": "legacy-callback-only"}
    assert tool_result == "PRIVATE_LEGACY_RESULT"
    assert metadata == {
        "schema": "xiaoban.tool-result.v1",
        "requestId": "request-e1a",
        "turnId": "turn-e1a",
        "callId": "call-e1a",
        "toolName": "mystand_example_read",
        "dispatchState": "dispatched",
        "outcome": "unknown",
        "retrySafe": False,
    }
    assert stored["_xiaoban_tool_result"]["recordRefs"] == [
        "private-record-ref"
    ]
    assert "recordRefs" not in metadata
    assert "continuation" not in metadata


def test_canonical_terminal_callback_keeps_legacy_four_argument_contract():
    callbacks = []
    agent = SimpleNamespace(
        _current_request_id="request-legacy",
        _current_turn_id="turn-legacy",
        tool_complete_callback=lambda call_id, name, args, result: callbacks.append(
            (call_id, name, args, result)
        ),
    )

    _append_canonical_tool_result(
        agent,
        [],
        "web_search",
        "ok",
        "call-legacy",
        dispatch_state="dispatched",
        function_args={"query": "hello"},
        callback_result="ok",
        emit_terminal_callback=True,
    )

    assert callbacks == [
        ("call-legacy", "web_search", {"query": "hello"}, "ok")
    ]


def test_finance_aggregate_result_uses_the_same_canonical_tool_result_path():
    agent = SimpleNamespace(
        _current_request_id="request-finance",
        _current_turn_id="turn-finance",
    )
    messages = []
    result = {
        "schema": "mystand.query-result.v1",
        "ok": True,
        "facts": [
            {
                "path": "finance.performance.rank",
                "year": 2026,
                "rank": 3,
            }
        ],
        "coverage": {"complete": True},
    }

    stored = _append_canonical_tool_result(
        agent,
        messages,
        "mystand_query",
        json.dumps(result, ensure_ascii=False),
        "call-finance",
        dispatch_state="dispatched",
    )

    assert stored["_xiaoban_tool_result"] == {
        "schema": "xiaoban.tool-result.v1",
        "requestId": "request-finance",
        "turnId": "turn-finance",
        "callId": "call-finance",
        "toolName": "mystand_query",
        "dispatchState": "dispatched",
        "outcome": "success",
        "retrySafe": False,
    }
    provider_message = sanitize_api_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-finance",
                        "type": "function",
                        "function": {
                            "name": "mystand_query",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            stored,
        ]
    )[-1]
    projected = json.loads(provider_message["content"])
    assert projected["outcome"] == "success"
    assert projected["modelResult"] == result


@pytest.mark.parametrize(
    (
        "raw_result",
        "dispatch_state",
        "outcome_hint",
        "expected_outcome",
        "expected_projection_key",
        "must_hide_raw",
    ),
    [
        pytest.param(
            json.dumps({"ok": True, "rows": [{"id": "visible-1"}]}),
            "dispatched",
            None,
            "success",
            "modelResult",
            False,
            id="success-dispatched",
        ),
        pytest.param(
            json.dumps({}),
            "dispatched",
            None,
            "success",
            "modelResult",
            False,
            id="unmarked-empty-shape-is-not-guessed",
        ),
        pytest.param(
            json.dumps({"ok": True, "status": "empty", "rows": []}),
            "dispatched",
            None,
            "empty",
            "modelResult",
            False,
            id="empty-dispatched",
        ),
        pytest.param(
            json.dumps({"ok": False, "status": 404, "resource": "private-404"}),
            "dispatched",
            None,
            "not_found",
            "modelError",
            True,
            id="not-found-dispatched",
        ),
        pytest.param(
            json.dumps({"ok": False, "status": 403, "resource": "private-403"}),
            "not_dispatched",
            None,
            "denied",
            "modelError",
            True,
            id="denied-before-dispatch",
        ),
        pytest.param(
            json.dumps({"ok": False, "status": 403, "resource": "private-handler-403"}),
            "dispatched",
            None,
            "denied",
            "modelError",
            True,
            id="denied-by-handler-after-dispatch",
        ),
        pytest.param(
            json.dumps({"ok": False, "status": "failed", "code": "upstream_error"}),
            "dispatched",
            None,
            "failed",
            "modelError",
            False,
            id="failed-dispatched",
        ),
        pytest.param(
            json.dumps({"status": "ambiguous", "writeMayHaveLanded": True}),
            "dispatched",
            None,
            "unknown",
            "modelError",
            True,
            id="unknown-dispatched",
        ),
        pytest.param(
            json.dumps({"status": "cancelled", "error": "not started"}),
            "not_dispatched",
            None,
            "cancelled",
            "modelError",
            True,
            id="cancelled-before-dispatch",
        ),
        pytest.param(
            json.dumps({"status": "cancelled", "error": "interrupted"}),
            "dispatched",
            None,
            "cancelled",
            "modelError",
            True,
            id="cancelled-after-dispatch",
        ),
        pytest.param(
            json.dumps({"status": "cancelled", "error": "terminal fence"}),
            "dispatched",
            "unknown",
            "unknown",
            "modelError",
            True,
            id="result-fence-after-dispatch-is-unknown",
        ),
    ],
)
def test_one_canonical_tool_result_table_reaches_the_model(
    raw_result,
    dispatch_state,
    outcome_hint,
    expected_outcome,
    expected_projection_key,
    must_hide_raw,
):
    agent = SimpleNamespace(
        _current_request_id="request-1",
        _current_task_id="request-1",
        _current_turn_id="turn-1",
    )
    messages = []

    stored = _append_canonical_tool_result(
        agent,
        messages,
        "mystand_example_read",
        raw_result,
        "call-1",
        dispatch_state=dispatch_state,
        outcome_hint=outcome_hint,
    )

    # Raw history remains available to existing private audit/session consumers.
    assert messages == [stored]
    assert stored["content"] == raw_result
    metadata = stored["_xiaoban_tool_result"]
    assert metadata == {
        "schema": "xiaoban.tool-result.v1",
        "requestId": "request-1",
        "turnId": "turn-1",
        "callId": "call-1",
        "toolName": "mystand_example_read",
        "dispatchState": dispatch_state,
        "outcome": expected_outcome,
        "retrySafe": False,
    }

    # The API copy carries the canonical, bounded model projection under the
    # original call id; the internal marker never reaches a strict provider.
    provider_messages = sanitize_api_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "mystand_example_read",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            stored.copy(),
        ]
    )
    projected_message = provider_messages[-1]
    assert "_xiaoban_tool_result" not in projected_message
    assert projected_message["tool_call_id"] == "call-1"
    projected = json.loads(projected_message["content"])
    assert projected["outcome"] == expected_outcome
    assert projected["dispatchState"] == dispatch_state
    assert expected_projection_key in projected
    other_key = "modelError" if expected_projection_key == "modelResult" else "modelResult"
    assert other_key not in projected
    if must_hide_raw:
        assert raw_result not in projected_message["content"]


def test_classification_uses_handler_result_before_output_persistence():
    agent = SimpleNamespace(
        _current_request_id="request-1",
        _current_turn_id="turn-1",
    )
    messages = []
    stored = _append_canonical_tool_result(
        agent,
        messages,
        "mystand_example_read",
        "<persisted-output>large result moved aside</persisted-output>",
        "call-persisted",
        dispatch_state="dispatched",
        classification_result=json.dumps(
            {"ok": False, "status": "failed", "code": "upstream_error"}
        ),
    )

    assert stored["_xiaoban_tool_result"]["outcome"] == "failed"


def test_projection_failure_never_falls_back_to_private_raw(monkeypatch):
    secret = "PRIVATE-RAW-RESULT-MUST-NOT-LEAK"
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.project_tool_result_for_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "demo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "demo",
            "content": secret,
            "tool_call_id": "call-1",
            "_xiaoban_tool_result": {
                "schema": "xiaoban.tool-result.v1",
                "requestId": "request-1",
                "turnId": "turn-1",
                "callId": "call-1",
                "toolName": "demo",
                "dispatchState": "dispatched",
                "outcome": "success",
                "retrySafe": False,
            },
        },
    ]
    append_trusted_steer_to_tool_message(messages[-1], "失败后改成只读")

    provider_messages = sanitize_api_messages(messages)
    projected_message = provider_messages[-1]
    projected_json, _, _ = projected_message["content"].partition(
        f"\n\n{STEER_MARKER_OPEN}"
    )
    projected = json.loads(projected_json)
    assert secret not in projected_message["content"]
    assert "_xiaoban_tool_result" not in projected_message
    assert projected["dispatchState"] == "dispatched"
    assert projected["outcome"] == "unknown"
    assert projected["retrySafe"] is False
    assert projected_message["content"].count(STEER_MARKER_OPEN) == 1
    assert projected_message["content"].count("失败后改成只读") == 1


@pytest.mark.parametrize(
    ("raw_result", "dispatch_state", "outcome_hint", "must_hide_raw"),
    [
        pytest.param(
            json.dumps({"ok": True, "visible": "SUCCESS-RAW"}),
            "dispatched",
            None,
            False,
            id="success",
        ),
        pytest.param(
            json.dumps({"status": "failed", "code": "FAILED-RAW"}),
            "dispatched",
            None,
            False,
            id="failed",
        ),
        pytest.param(
            json.dumps({"ok": False, "status": 403, "private": "DENIED-RAW"}),
            "not_dispatched",
            None,
            True,
            id="denied",
        ),
        pytest.param(
            json.dumps({"status": "ambiguous", "private": "UNKNOWN-RAW"}),
            "dispatched",
            None,
            True,
            id="unknown",
        ),
        pytest.param(
            json.dumps({"status": "cancelled", "private": "CANCELLED-RAW"}),
            "not_dispatched",
            None,
            True,
            id="cancelled",
        ),
    ],
)
def test_runtime_trusted_steer_survives_private_projection(
    raw_result,
    dispatch_state,
    outcome_hint,
    must_hide_raw,
):
    agent = SimpleNamespace(
        _current_request_id="request-steer",
        _current_task_id="request-steer",
        _current_turn_id="turn-steer",
    )
    messages = []
    stored = _append_canonical_tool_result(
        agent,
        messages,
        "mystand_example_read",
        raw_result,
        "call-steer",
        dispatch_state=dispatch_state,
        outcome_hint=outcome_hint,
    )
    append_trusted_steer_to_tool_message(stored, "改成只读")

    assert stored["content"].count("改成只读") == 1
    assert stored["_xiaoban_trusted_steer"]

    provider_messages = sanitize_api_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-steer",
                        "type": "function",
                        "function": {
                            "name": "mystand_example_read",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            stored.copy(),
        ]
    )
    projected_message = provider_messages[-1]
    assert "_xiaoban_tool_result" not in projected_message
    assert "_xiaoban_trusted_steer" not in projected_message
    if must_hide_raw:
        assert raw_result not in projected_message["content"]
    assert projected_message["content"].count(STEER_MARKER_OPEN) == 1
    assert projected_message["content"].count("改成只读") == 1


def test_runtime_trusted_steer_survives_multimodal_projection_once():
    stored = {
        "role": "tool",
        "name": "computer_use",
        "tool_call_id": "call-image",
        "content": [
            {"type": "text", "text": "screen captured"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        ],
        "_xiaoban_tool_result": {
            "schema": "xiaoban.tool-result.v1",
            "requestId": "request-image",
            "turnId": "turn-image",
            "callId": "call-image",
            "toolName": "computer_use",
            "dispatchState": "dispatched",
            "outcome": "success",
            "retrySafe": False,
        },
    }
    append_trusted_steer_to_tool_message(stored, "只看当前页面")

    provider_messages = sanitize_api_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-image",
                        "type": "function",
                        "function": {"name": "computer_use", "arguments": "{}"},
                    }
                ],
            },
            stored,
        ]
    )

    projected_message = provider_messages[-1]
    assert "_xiaoban_trusted_steer" not in projected_message
    assert isinstance(projected_message["content"], list)
    rendered = "\n".join(
        str(part.get("text", ""))
        for part in projected_message["content"]
        if isinstance(part, dict)
    )
    assert rendered.count(STEER_MARKER_OPEN) == 1
    assert rendered.count("只看当前页面") == 1
    assert "screen captured" in rendered


def test_multimodal_projection_bounds_nested_text_and_preserves_small_image():
    huge_text = "A" * 150_000 + "PRIVATE-TAIL-MUST-BE-TRUNCATED"
    small_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AA"},
    }
    stored = {
        "role": "tool",
        "name": "computer_use",
        "tool_call_id": "call-large-image",
        "content": [
            {"type": "text", "text": huge_text},
            small_image,
        ],
        "_xiaoban_tool_result": {
            "schema": "xiaoban.tool-result.v1",
            "requestId": "request-large-image",
            "turnId": "turn-large-image",
            "callId": "call-large-image",
            "toolName": "computer_use",
            "dispatchState": "dispatched",
            "outcome": "success",
            "retrySafe": False,
        },
    }

    projected_message = sanitize_api_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-large-image",
                        "type": "function",
                        "function": {
                            "name": "computer_use",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            stored,
        ]
    )[-1]

    assert isinstance(projected_message["content"], list)
    header = json.loads(projected_message["content"][0]["text"])
    projected_parts = projected_message["content"][1:]
    assert header["truncated"] is True
    assert header["modelResult"] == {
        "contentType": "multimodal",
        "parts": 2,
        "originalParts": 2,
    }
    assert projected_parts[-1] == small_image
    assert "PRIVATE-TAIL-MUST-BE-TRUNCATED" not in json.dumps(
        projected_parts,
        ensure_ascii=False,
    )
    assert len(
        json.dumps(projected_parts, ensure_ascii=False, separators=(",", ":"))
    ) <= 100_000


def test_canonical_tool_result_survives_session_flush_and_replay(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_db.create_session(session_id="session-e1", source="api")
        agent = object.__new__(AIAgent)
        agent._session_db = session_db
        agent._session_db_created = True
        agent.session_id = "session-e1"
        agent._last_flushed_db_idx = 0
        agent._persist_user_message_idx = None
        agent._persist_user_message_override = None
        agent._persist_user_message_timestamp = None
        agent._current_request_id = "request-e1"
        agent._current_turn_id = "turn-e1"

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-e1",
                        "type": "function",
                        "function": {
                            "name": "mystand_example_read",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ]
        stored = _append_canonical_tool_result(
            agent,
            messages,
            "mystand_example_read",
            json.dumps({"status": "ambiguous", "private": "never-retry"}),
            "call-e1",
            dispatch_state="dispatched",
            outcome_hint="unknown",
        )
        AIAgent._flush_messages_to_session_db(agent, messages, [])

        replayed = session_db.get_messages_as_conversation("session-e1")
        assert len(replayed) == 2
        assert replayed[1]["_xiaoban_tool_result"] == stored["_xiaoban_tool_result"]
        assert replayed[1]["_xiaoban_tool_result"]["outcome"] == "unknown"
        assert replayed[1]["_xiaoban_tool_result"]["retrySafe"] is False

        provider_message = sanitize_api_messages(replayed)[1]
        projected = json.loads(provider_message["content"])
        assert projected["outcome"] == "unknown"
        assert projected["modelError"] == {"code": "unknown"}
        assert "never-retry" not in provider_message["content"]
    finally:
        session_db.close()


@pytest.mark.parametrize("writer", ["replace", "compact"])
def test_canonical_tool_result_survives_session_rewrite_paths(tmp_path, writer):
    session_db = SessionDB(db_path=tmp_path / f"{writer}.db")
    canonical = {
        "schema": "xiaoban.tool-result.v1",
        "requestId": "request-rewrite",
        "turnId": "turn-rewrite",
        "callId": "call-rewrite",
        "toolName": "mystand_example_read",
        "dispatchState": "dispatched",
        "outcome": "denied",
        "retrySafe": False,
    }
    transcript = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-rewrite",
                    "type": "function",
                    "function": {
                        "name": "mystand_example_read",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"private":"must-stay-private"}',
            "tool_call_id": "call-rewrite",
            "tool_name": "mystand_example_read",
            "_xiaoban_tool_result": canonical,
        },
    ]
    try:
        session_db.create_session(session_id="session-rewrite", source="api")
        if writer == "replace":
            session_db.replace_messages("session-rewrite", transcript)
        else:
            session_db.archive_and_compact("session-rewrite", transcript)

        replayed = session_db.get_messages_as_conversation("session-rewrite")
        assert replayed[1]["_xiaoban_tool_result"] == canonical
        projected = sanitize_api_messages(replayed)[1]
        assert "must-stay-private" not in projected["content"]
        assert json.loads(projected["content"])["outcome"] == "denied"
    finally:
        session_db.close()


def test_session_replay_drops_mismatched_tool_result_sidecar(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "mismatch.db")
    try:
        session_db.create_session(session_id="session-mismatch", source="api")
        session_db.append_message(
            "session-mismatch",
            role="tool",
            content="private raw",
            tool_call_id="call-real",
            tool_name="mystand_example_read",
            tool_result={
                "schema": "xiaoban.tool-result.v1",
                "requestId": "request-mismatch",
                "turnId": "turn-mismatch",
                "callId": "call-forged",
                "toolName": "mystand_example_read",
                "dispatchState": "dispatched",
                "outcome": "success",
                "retrySafe": True,
            },
        )
        replayed = session_db.get_messages_as_conversation(
            "session-mismatch"
        )
        assert "_xiaoban_tool_result" not in replayed[0]
    finally:
        session_db.close()


@pytest.mark.parametrize(
    ("empty_field", "empty_value"),
    [
        pytest.param(field, value, id=f"{field}-{kind}")
        for field in ("requestId", "turnId", "callId", "toolName")
        for kind, value in (("empty", ""), ("blank", "   "))
    ],
)
def test_session_replay_drops_canonical_sidecar_with_empty_identity(
    tmp_path,
    empty_field,
    empty_value,
):
    session_db = SessionDB(
        db_path=tmp_path / f"empty-{empty_field}-{len(empty_value)}.db"
    )
    canonical = {
        "schema": "xiaoban.tool-result.v1",
        "requestId": "request-real",
        "turnId": "turn-real",
        "callId": "call-real",
        "toolName": "mystand_example_read",
        "dispatchState": "dispatched",
        "outcome": "success",
        "retrySafe": False,
    }
    canonical[empty_field] = empty_value
    try:
        session_db.create_session(
            session_id=f"session-empty-{empty_field}",
            source="api",
        )
        session_db.append_message(
            f"session-empty-{empty_field}",
            role="tool",
            content="private raw",
            tool_call_id=(
                empty_value if empty_field == "callId" else "call-real"
            ),
            tool_name=(
                empty_value
                if empty_field == "toolName"
                else "mystand_example_read"
            ),
            tool_result=canonical,
        )

        replayed = session_db.get_messages_as_conversation(
            f"session-empty-{empty_field}"
        )
        assert "_xiaoban_tool_result" not in replayed[0]
    finally:
        session_db.close()
