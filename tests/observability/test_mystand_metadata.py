from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from xiaoban.observability.mystand_metadata import (
    MetadataValidationError,
    MystandMetadataTrace,
)


def test_metadata_log_contains_only_closed_schema(caplog):
    sentinel = "13800138000 secret@example.com Bearer-very-secret"
    trace = MystandMetadataTrace(
        secret="trace-secret",
        site_id="site-a",
        user_id="alice",
        trace_id="a" * 32,
    )
    with caplog.at_level(logging.INFO, logger="xiaoban.mystand.metadata"):
        payload = trace.emit(
            "request_completed",
            status="completed",
            duration_ms=12,
            tool_count=1,
            memory_enabled=True,
            memory_hit_count=2,
            input_tokens=3,
            output_tokens=4,
            total_tokens=7,
            provider="openai",
            model="gpt-5.2",
        )

    assert set(payload) == {
        "event", "timestamp_ms", "trace_id", "account_scope", "status",
        "duration_ms", "tool_count", "memory_enabled", "memory_hit_count",
        "input_tokens", "output_tokens", "total_tokens", "provider", "model",
    }
    assert "alice" not in caplog.text
    assert "site-a" not in caplog.text
    assert sentinel not in caplog.text


def test_unknown_content_and_tool_payload_fields_are_rejected():
    trace = MystandMetadataTrace(
        secret="trace-secret",
        site_id="site-a",
        user_id="alice",
        trace_id="b" * 32,
    )
    for forbidden in ("prompt", "output", "reasoning", "tool_args", "tool_result", "attachment"):
        with pytest.raises(MetadataValidationError):
            trace.emit("request_started", status="accepted", **{forbidden: "sentinel"})


def test_account_scope_is_stable_but_not_shared_between_accounts():
    first = MystandMetadataTrace("secret", "site-a", "alice", trace_id="c" * 32)
    same = MystandMetadataTrace("secret", "site-a", "alice", trace_id="d" * 32)
    other = MystandMetadataTrace("secret", "site-a", "bob", trace_id="e" * 32)

    assert first.account_scope == same.account_scope
    assert first.account_scope != other.account_scope


def test_langfuse_hooks_are_inert_for_mystand(monkeypatch):
    import plugins.observability.langfuse as langfuse

    calls = []
    monkeypatch.setattr(langfuse, "_get_langfuse", lambda: calls.append("client") or object())
    tokens = set_session_vars(source="mystand")
    try:
        langfuse.on_pre_llm_request(request_messages=[{"role": "user", "content": "secret"}])
        langfuse.on_post_llm_call(assistant_response="secret")
        langfuse.on_pre_tool_call(tool_name="terminal", args={"cmd": "secret"})
        langfuse.on_post_tool_call(tool_name="terminal", result="secret")
    finally:
        clear_session_vars(tokens)

    assert calls == []


def test_langfuse_hooks_remain_inert_in_propagated_delegate_thread(monkeypatch):
    import plugins.observability.langfuse as langfuse
    from tools.thread_context import propagate_context_to_thread

    calls = []
    monkeypatch.setattr(langfuse, "_get_langfuse", lambda: calls.append("client") or object())

    def child_hooks():
        langfuse.on_pre_llm_request(request_messages=[{"role": "user", "content": "secret"}])
        langfuse.on_post_tool_call(tool_name="web_search", result="secret")

    tokens = set_session_vars(source="mystand")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(propagate_context_to_thread(child_hooks)).result(timeout=5)
    finally:
        clear_session_vars(tokens)

    assert calls == []


@pytest.mark.parametrize(
    ("tool_name", "result", "expected"),
    [
        ("web_search", {"success": True, "items": []}, False),
        ("web_search", {"success": False, "error": "upstream unavailable"}, True),
        ("terminal", {"exit_code": 7, "output": ""}, True),
        ("terminal", {"exit_code": 0, "output": "ok"}, False),
    ],
)
def test_mystand_tool_result_status_uses_real_result(tool_name, result, expected):
    from gateway.platforms.api_server import _mystand_tool_result_failed

    assert _mystand_tool_result_failed(tool_name, result) is expected
