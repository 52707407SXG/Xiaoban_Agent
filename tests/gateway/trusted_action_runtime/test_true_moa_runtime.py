"""Focused deterministic tests for the provider-agnostic true-MoA core.

No test in this module imports a provider client or performs a network call.
"""

from __future__ import annotations

import html
import json
import re
import threading
import time
from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from xiaoban.trusted_runtime.true_moa import (
    DEEPSEEK_ADVISOR_SLOT,
    FINAL_EXECUTOR_SLOT,
    KIMI_ADVISOR_SLOT,
    MODE_EPOCH_HEADER,
    MOA_PRESET_ID_HEADER,
    MOA_PRESET_REVISION_HEADER,
    REASONING_MODE_HEADER,
    StrictAdvisorResult,
    TRUE_MOA_ADVISOR_SLOTS,
    TRUE_MOA_FINAL_CALL_LIMIT,
    TRUE_MOA_MODE,
    TRUE_MOA_PRESET_ID,
    TRUE_MOA_PRESET_REVISION,
    TRUE_MOA_TOTAL_CALL_LIMIT,
    TrueMoACancelController,
    TrueMoAContractError,
    TrueMoAExecutionError,
    TrueMoASnapshot,
    TrueMoAUsageLedger,
    build_minimal_advisor_messages,
    run_true_moa_advisors,
    validate_true_moa_headers,
)


def _snapshot(epoch: str = "7") -> TrueMoASnapshot:
    return TrueMoASnapshot(
        mode=TRUE_MOA_MODE,
        mode_epoch=epoch,
        preset_id=TRUE_MOA_PRESET_ID,
        preset_revision=TRUE_MOA_PRESET_REVISION,
    )


def _headers(epoch: str = "7") -> dict[str, str]:
    return {
        REASONING_MODE_HEADER: TRUE_MOA_MODE,
        MODE_EPOCH_HEADER: epoch,
        MOA_PRESET_ID_HEADER: TRUE_MOA_PRESET_ID,
        MOA_PRESET_REVISION_HEADER: TRUE_MOA_PRESET_REVISION,
    }


def _slot_receipts(ledger) -> dict[str, dict]:
    return {item["slotId"]: item for item in ledger.to_dict()["slots"]}


def test_header_contract_accepts_only_exact_moa_snapshot_or_clean_normal():
    snapshot = validate_true_moa_headers(
        {key.lower(): value for key, value in _headers("42").items()},
        mystand_request=True,
        api_authenticated=True,
    )
    assert snapshot == _snapshot("42")

    assert (
        validate_true_moa_headers(
            {},
            mystand_request=True,
            api_authenticated=True,
        )
        is None
    )
    assert (
        validate_true_moa_headers(
            {REASONING_MODE_HEADER: "normal"},
            mystand_request=True,
            api_authenticated=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("headers", "mystand_request", "api_authenticated", "code"),
    [
        (
            {
                REASONING_MODE_HEADER: "normal",
                MODE_EPOCH_HEADER: "1",
            },
            True,
            True,
            "normal_mode_cannot_carry_moa_metadata",
        ),
        (
            {REASONING_MODE_HEADER: "legacy-moa"},
            True,
            True,
            "unsupported_reasoning_mode",
        ),
        (
            {
                REASONING_MODE_HEADER: TRUE_MOA_MODE,
                MOA_PRESET_ID_HEADER: TRUE_MOA_PRESET_ID,
                MOA_PRESET_REVISION_HEADER: TRUE_MOA_PRESET_REVISION,
            },
            True,
            True,
            "invalid_mode_epoch",
        ),
        (
            _headers("-1"),
            True,
            True,
            "invalid_mode_epoch",
        ),
        (
            _headers("01"),
            True,
            True,
            "invalid_mode_epoch",
        ),
        (
            {**_headers(), MOA_PRESET_ID_HEADER: "client-selected-models"},
            True,
            True,
            "invalid_true_moa_preset_id",
        ),
        (
            {**_headers(), MOA_PRESET_REVISION_HEADER: "latest"},
            True,
            True,
            "invalid_true_moa_preset_revision",
        ),
        (
            _headers(),
            False,
            True,
            "true_moa_requires_authenticated_mystand",
        ),
        (
            _headers(),
            True,
            False,
            "true_moa_requires_authenticated_mystand",
        ),
    ],
)
def test_header_contract_fails_closed(
    headers,
    mystand_request,
    api_authenticated,
    code,
):
    with pytest.raises(TrueMoAContractError) as caught:
        validate_true_moa_headers(
            headers,
            mystand_request=mystand_request,
            api_authenticated=api_authenticated,
        )
    assert caught.value.code == code


def test_minimal_input_excludes_privileged_roles_attachments_and_secrets():
    messages = build_minimal_advisor_messages(
        [
            {"type": "text", "text": "当前问题 api_key=sk-current-secret-123456"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 100},
            },
        ],
        [
            {"role": "system", "content": "内部系统提示绝不能外发"},
            {"role": "user", "content": "更早、应被相邻窗口裁掉"},
            {"role": "tool", "content": "跨账号客户明文"},
            {"role": "assistant", "content": "最近回答 password=supersecret"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "最近问题"},
                    {"type": "file", "file_id": "private-file-id"},
                ],
            },
        ],
    )

    assert [message.role for message in messages] == ["assistant", "user", "user"]
    serialized = "\n".join(message.content for message in messages)
    assert "内部系统提示" not in serialized
    assert "跨账号客户明文" not in serialized
    assert "更早" not in serialized
    assert "private-file-id" not in serialized
    assert "base64" not in serialized
    assert "supersecret" not in serialized
    assert "sk-current-secret" not in serialized
    assert serialized.count("[REDACTED]") == 2


def test_fixed_two_advisors_run_once_in_parallel_without_tools_or_shared_input():
    rendezvous = threading.Barrier(2, timeout=1)
    lock = threading.Lock()
    calls: Counter[str] = Counter()
    message_views: dict[str, tuple] = {}
    tool_views: dict[str, tuple] = {}

    def strict_caller(*, slot, messages, tools, dispatch_callback, **_kwargs):
        dispatch_callback()
        with lock:
            calls[slot.slot_id] += 1
            message_views[slot.slot_id] = messages
            tool_views[slot.slot_id] = tools
        rendezvous.wait()
        with pytest.raises(TypeError):
            messages[0] = messages[0]
        with pytest.raises(FrozenInstanceError):
            messages[0].content = "mutated"
        return StrictAdvisorResult(
            content=f"建议 <{slot.slot_id}>",
            usage={
                "input_tokens": 10,
                "output_tokens": 3,
                "cached_input_tokens": 0,
            },
            cost_usd=0.01,
            cost_status="reported",
            cost_source="fake-provider",
        )

    controller = TrueMoACancelController()
    bundle = run_true_moa_advisors(
        _snapshot(),
        current_question="请分析方案",
        conversation_history=[{"role": "assistant", "content": "相邻回答"}],
        strict_caller=strict_caller,
        cancel_controller=controller,
        timeout_seconds=1,
    )

    expected_ids = {slot.slot_id for slot in TRUE_MOA_ADVISOR_SLOTS}
    assert calls == Counter({slot_id: 1 for slot_id in expected_ids})
    assert set(message_views) == expected_ids
    assert all(tools == () for tools in tool_views.values())
    first, second = (message_views[slot.slot_id] for slot in TRUE_MOA_ADVISOR_SLOTS)
    assert first is not second
    assert all(left is not right for left, right in zip(first, second))
    assert "<advisor-kimi-k3>" not in bundle.guidance
    assert "&lt;advisor-kimi-k3&gt;" in bundle.guidance

    ledger = bundle.ledger.to_dict()
    receipts = _slot_receipts(bundle.ledger)
    assert ledger["status"] == "advisors_completed"
    assert [item["slotId"] for item in ledger["slots"]] == [
        KIMI_ADVISOR_SLOT.slot_id,
        DEEPSEEK_ADVISOR_SLOT.slot_id,
        FINAL_EXECUTOR_SLOT.slot_id,
    ]
    for slot in TRUE_MOA_ADVISOR_SLOTS:
        receipt = receipts[slot.slot_id]
        assert receipt["status"] == "completed"
        assert receipt["inputTokens"] == 10
        assert receipt["outputTokens"] == 3
        assert receipt["totalTokens"] == 13
        assert receipt["cachedInputTokens"] == 0
        assert receipt["costUsd"] == 0.01
    assert [call["slotId"] for call in ledger["calls"]] == [
        KIMI_ADVISOR_SLOT.slot_id,
        DEEPSEEK_ADVISOR_SLOT.slot_id,
    ]
    assert all(call["status"] == "completed" for call in ledger["calls"])
    assert receipts[FINAL_EXECUTOR_SLOT.slot_id]["status"] == "not_started"

    # Advisor completion is a stage boundary, not the request terminal state.
    assert controller.state == "running"
    assert not controller.is_set
    controller.cancel()
    assert controller.state == "cancelled"
    assert controller.is_set


def test_any_advisor_failure_closes_peer_and_waits_for_all_dispatched_calls():
    rendezvous = threading.Barrier(2, timeout=1)
    peer_closed = threading.Event()
    exited: set[str] = set()
    calls: Counter[str] = Counter()
    lock = threading.Lock()

    def strict_caller(*, slot, cancel_controller, dispatch_callback, **_kwargs):
        dispatch_callback()
        with lock:
            calls[slot.slot_id] += 1
        if slot == DEEPSEEK_ADVISOR_SLOT:
            cancel_controller.register_cancel_callback(slot.slot_id, peer_closed.set)
        rendezvous.wait()
        if slot == KIMI_ADVISOR_SLOT:
            with lock:
                exited.add(slot.slot_id)
            raise RuntimeError("sensitive-provider-detail")
        assert peer_closed.wait(1)
        with lock:
            exited.add(slot.slot_id)
        raise TimeoutError("transport closed after peer failure")

    with pytest.raises(TrueMoAExecutionError) as caught:
        run_true_moa_advisors(
            _snapshot(),
            current_question="失败测试",
            conversation_history=[],
            strict_caller=strict_caller,
            timeout_seconds=1,
        )

    expected_ids = {slot.slot_id for slot in TRUE_MOA_ADVISOR_SLOTS}
    assert calls == Counter({slot_id: 1 for slot_id in expected_ids})
    assert peer_closed.is_set()
    # The execution exception is observable only after the termination barrier.
    assert exited == expected_ids
    assert caught.value.category == "provider_error"
    assert "sensitive-provider-detail" not in str(caught.value)
    receipts = _slot_receipts(caught.value.ledger)
    assert receipts[KIMI_ADVISOR_SLOT.slot_id]["status"] == "failed"
    assert receipts[DEEPSEEK_ADVISOR_SLOT.slot_id]["status"] == "cancelled"
    assert (
        receipts[DEEPSEEK_ADVISOR_SLOT.slot_id]["errorCategory"]
        == "cascade_after_provider_error"
    )


def test_timeout_invokes_transport_close_callbacks_and_waits_for_exit():
    rendezvous = threading.Barrier(2, timeout=1)
    callback_events = {
        slot.slot_id: threading.Event() for slot in TRUE_MOA_ADVISOR_SLOTS
    }
    exited: set[str] = set()
    lock = threading.Lock()

    def strict_caller(*, slot, cancel_controller, dispatch_callback, **_kwargs):
        dispatch_callback()
        close_event = callback_events[slot.slot_id]
        cancel_controller.register_cancel_callback(slot.slot_id, close_event.set)
        rendezvous.wait()
        assert close_event.wait(1)
        with lock:
            exited.add(slot.slot_id)
        return StrictAdvisorResult(content="late timeout output")

    started = time.monotonic()
    with pytest.raises(TrueMoAExecutionError) as caught:
        run_true_moa_advisors(
            _snapshot(),
            current_question="超时测试",
            conversation_history=[],
            strict_caller=strict_caller,
            timeout_seconds=0.05,
        )
    elapsed = time.monotonic() - started

    expected_ids = {slot.slot_id for slot in TRUE_MOA_ADVISOR_SLOTS}
    assert caught.value.category == "advisor_timeout"
    assert elapsed < 0.5
    assert all(event.is_set() for event in callback_events.values())
    assert exited == expected_ids
    receipts = _slot_receipts(caught.value.ledger)
    statuses = {receipts[slot_id]["status"] for slot_id in expected_ids}
    assert "timed_out" in statuses
    assert statuses <= {"timed_out", "cancelled"}


def test_cancel_before_start_dispatches_zero_advisors():
    controller = TrueMoACancelController()
    controller.cancel()
    calls = 0

    def strict_caller(**_kwargs):
        nonlocal calls
        calls += 1
        return StrictAdvisorResult(content="must not run")

    with pytest.raises(TrueMoAExecutionError) as caught:
        run_true_moa_advisors(
            _snapshot(),
            current_question="不要派发",
            conversation_history=[],
            strict_caller=strict_caller,
            cancel_controller=controller,
        )

    assert calls == 0
    assert caught.value.category == "cancelled"


def test_running_cancel_closes_both_calls_and_late_results_cannot_escape():
    controller = TrueMoACancelController()
    both_started = threading.Event()
    callback_events = {
        slot.slot_id: threading.Event() for slot in TRUE_MOA_ADVISOR_SLOTS
    }
    started_ids: set[str] = set()
    exited_ids: set[str] = set()
    lock = threading.Lock()
    outcome: list[object] = []

    def strict_caller(*, slot, cancel_controller, dispatch_callback, **_kwargs):
        dispatch_callback()
        close_event = callback_events[slot.slot_id]
        cancel_controller.register_cancel_callback(slot.slot_id, close_event.set)
        with lock:
            started_ids.add(slot.slot_id)
            if len(started_ids) == 2:
                both_started.set()
        assert close_event.wait(1)
        with lock:
            exited_ids.add(slot.slot_id)
        if slot == KIMI_ADVISOR_SLOT:
            error = TimeoutError("transport closed after user stop")
            error.usage = {
                "input_tokens": 9,
                "output_tokens": 2,
                "total_tokens": 11,
                "cached_input_tokens": 0,
            }
            error.cost_usd = 0.04
            error.cost_status = "reported"
            error.cost_source = "fake-late-usage"
            error.content = "private late result"
            raise error
        raise OSError("socket closed after user stop")

    def run_wave():
        try:
            outcome.append(
                run_true_moa_advisors(
                    _snapshot(),
                    current_question="运行中取消",
                    conversation_history=[],
                    strict_caller=strict_caller,
                    cancel_controller=controller,
                    timeout_seconds=1,
                )
            )
        except Exception as exc:  # captured for assertions in the test thread
            outcome.append(exc)

    thread = threading.Thread(target=run_wave, daemon=True)
    thread.start()
    assert both_started.wait(1)
    controller.cancel()
    thread.join(1)

    expected_ids = {slot.slot_id for slot in TRUE_MOA_ADVISOR_SLOTS}
    assert not thread.is_alive()
    assert exited_ids == expected_ids
    assert all(event.is_set() for event in callback_events.values())
    assert len(outcome) == 1
    assert isinstance(outcome[0], TrueMoAExecutionError)
    assert outcome[0].category == "cancelled"
    assert "private late result" not in json.dumps(
        outcome[0].ledger.to_dict(),
        ensure_ascii=False,
    )
    receipts = _slot_receipts(outcome[0].ledger)
    assert all(
        receipts[slot_id]["status"] == "cancelled" for slot_id in expected_ids
    )
    kimi_receipt = receipts[KIMI_ADVISOR_SLOT.slot_id]
    assert kimi_receipt["usageStatus"] == "reported"
    assert kimi_receipt["totalTokens"] == 11
    assert kimi_receipt["cachedInputTokens"] == 0
    assert kimi_receipt["costUsd"] == 0.04


def test_usage_and_cost_are_fill_once_after_terminal_status():
    ledger = TrueMoAUsageLedger(_snapshot())
    ledger.start_slot(KIMI_ADVISOR_SLOT)
    ledger.finish_slot(
        KIMI_ADVISOR_SLOT,
        status="completed",
        usage={
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
            "cached_input_tokens": 0,
        },
        cost_usd=0.01,
        cost_status="reported",
        cost_source="first-provider-receipt",
    )
    ledger.finish_slot(
        KIMI_ADVISOR_SLOT,
        status="cancelled",
        usage={
            "input_tokens": 999,
            "output_tokens": 999,
            "total_tokens": 1998,
            "cached_input_tokens": 999,
        },
        cost_usd=9.99,
        cost_status="estimated",
        cost_source="late-overwrite",
    )

    ledger.start_slot(DEEPSEEK_ADVISOR_SLOT)
    ledger.finish_slot(
        DEEPSEEK_ADVISOR_SLOT,
        status="cancelled",
        error_category="terminal_fence",
    )
    ledger.finish_slot(
        DEEPSEEK_ADVISOR_SLOT,
        status="completed",
        usage={
            "input_tokens": 7,
            "output_tokens": 2,
            "total_tokens": 9,
            "cached_input_tokens": 2,
        },
        cost_usd=0.02,
        cost_status="reported",
        cost_source="late-actual-receipt",
    )

    receipts = _slot_receipts(ledger)
    first = receipts[KIMI_ADVISOR_SLOT.slot_id]
    assert first["status"] == "completed"
    assert first["inputTokens"] == 10
    assert first["outputTokens"] == 3
    assert first["totalTokens"] == 13
    assert first["costUsd"] == 0.01
    assert first["costSource"] == "first-provider-receipt"
    late = receipts[DEEPSEEK_ADVISOR_SLOT.slot_id]
    assert late["status"] == "cancelled"
    assert late["errorCategory"] == "terminal_fence"
    assert late["usageStatus"] == "reported"
    assert late["totalTokens"] == 9
    assert late["cachedInputTokens"] == 2
    assert late["costUsd"] == 0.02


@pytest.mark.parametrize(
    ("cache_fields", "expected_cached", "expected_status"),
    [
        ({"prompt_cache_hit_tokens": 4}, 4, "reported"),
        ({"cached_prompt_tokens": 5}, 5, "reported"),
        ({"prompt_tokens_details": {"cached_tokens": 6}}, 6, "reported"),
        ({"prompt_cache_hit_tokens": 0}, 0, "reported"),
        ({}, None, "partial"),
        ({"prompt_cache_hit_tokens": True}, None, "partial"),
    ],
)
def test_cache_split_requires_a_trusted_nonnegative_integer(
    cache_fields,
    expected_cached,
    expected_status,
):
    ledger = TrueMoAUsageLedger(_snapshot())
    ledger.start_slot(DEEPSEEK_ADVISOR_SLOT)
    call_id = ledger.start_advisor_call(DEEPSEEK_ADVISOR_SLOT)
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
        **cache_fields,
    }
    ledger.finish_advisor_call(
        call_id,
        status="completed",
        usage=usage,
    )
    ledger.finish_slot(
        DEEPSEEK_ADVISOR_SLOT,
        status="completed",
        usage=usage,
    )

    call = ledger.to_dict()["calls"][0]
    assert call["cachedInputTokens"] == expected_cached
    assert call["usageStatus"] == expected_status


@pytest.mark.parametrize(
    "result",
    [
        StrictAdvisorResult(content=""),
        StrictAdvisorResult(content="   \n\t"),
        StrictAdvisorResult(content={"text": "not plaintext"}),
        StrictAdvisorResult(content="tries a tool", tool_calls=[{"name": "write"}]),
        {"content": "mapping tries a tool", "tool_calls": [{"name": "read"}]},
    ],
)
def test_malformed_or_tool_call_advisor_output_fails_whole_wave(result):
    rendezvous = threading.Barrier(2, timeout=1)
    calls: Counter[str] = Counter()
    lock = threading.Lock()

    def strict_caller(*, slot, dispatch_callback, **_kwargs):
        dispatch_callback()
        with lock:
            calls[slot.slot_id] += 1
        rendezvous.wait()
        return result

    with pytest.raises(TrueMoAExecutionError) as caught:
        run_true_moa_advisors(
            _snapshot(),
            current_question="畸形输出",
            conversation_history=[],
            strict_caller=strict_caller,
            timeout_seconds=1,
        )

    expected_ids = {slot.slot_id for slot in TRUE_MOA_ADVISOR_SLOTS}
    assert calls == Counter({slot_id: 1 for slot_id in expected_ids})
    assert caught.value.category in {
        "advisor_content_empty",
        "advisor_content_not_string",
        "advisor_returned_tool_calls",
    }
    assert caught.value.ledger.to_dict()["status"] == "failed"


def test_malformed_advisor_result_preserves_reported_usage_and_cost():
    rendezvous = threading.Barrier(2, timeout=1)

    def strict_caller(*, dispatch_callback, **_kwargs):
        dispatch_callback()
        rendezvous.wait()
        return StrictAdvisorResult(
            content="tool attempts are forbidden",
            tool_calls=[{"name": "forbidden"}],
            usage={
                "input_tokens": 12,
                "output_tokens": 4,
                "total_tokens": 16,
                "cached_input_tokens": 3,
            },
            cost_usd=0.02,
            cost_status="reported",
            cost_source="fake-provider",
        )

    with pytest.raises(TrueMoAExecutionError) as caught:
        run_true_moa_advisors(
            _snapshot(),
            current_question="保留已产生的用量",
            conversation_history=[],
            strict_caller=strict_caller,
            timeout_seconds=1,
        )

    receipts = _slot_receipts(caught.value.ledger)
    for slot in TRUE_MOA_ADVISOR_SLOTS:
        receipt = receipts[slot.slot_id]
        assert receipt["usageStatus"] == "reported"
        assert receipt["inputTokens"] == 12
        assert receipt["outputTokens"] == 4
        assert receipt["totalTokens"] == 16
        assert receipt["cachedInputTokens"] == 3
        assert receipt["costUsd"] == 0.02


def test_output_is_redacted_bounded_escaped_and_absent_from_ledger():
    rendezvous = threading.Barrier(2, timeout=1)
    secret = "sk-do-not-leak-123456789"

    def strict_caller(*, slot, dispatch_callback, **_kwargs):
        dispatch_callback()
        rendezvous.wait()
        return StrictAdvisorResult(
            content=f"<danger>{slot.slot_id}</danger> api_key={secret} " + "x" * 500,
            usage={"total_tokens": 9},
        )

    bundle = run_true_moa_advisors(
        _snapshot(),
        current_question="净化测试",
        conversation_history=[],
        strict_caller=strict_caller,
        timeout_seconds=1,
        output_max_chars=80,
    )

    assert secret not in bundle.guidance
    assert "[REDACTED]" in bundle.guidance
    assert "<danger>" not in bundle.guidance
    assert "&lt;danger&gt;" in bundle.guidance
    assert "[TRUNCATED]" in bundle.guidance
    payloads = re.findall(
        r"<advisor [^>]+>(.*?)</advisor>",
        bundle.guidance,
        flags=re.DOTALL,
    )
    assert len(payloads) == 2
    assert all(len(html.unescape(payload)) <= 80 for payload in payloads)

    ledger_text = json.dumps(bundle.ledger.to_dict(), ensure_ascii=False)
    assert secret not in ledger_text
    assert "danger" not in ledger_text
    assert "净化测试" not in ledger_text


def test_fixed_preset_has_two_advisors_and_at_most_eight_final_calls():
    assert len(TRUE_MOA_ADVISOR_SLOTS) == 2
    assert TRUE_MOA_FINAL_CALL_LIMIT == 8
    assert TRUE_MOA_TOTAL_CALL_LIMIT == 10
    assert (
        len(TRUE_MOA_ADVISOR_SLOTS) + TRUE_MOA_FINAL_CALL_LIMIT
        == TRUE_MOA_TOTAL_CALL_LIMIT
    )

    ledger = TrueMoAUsageLedger(
        _snapshot(),
        wave_id="8" * 32,
    )
    for index in range(TRUE_MOA_FINAL_CALL_LIMIT):
        ledger.start_final_call(f"final-{index}")
    with pytest.raises(
        RuntimeError,
        match="final provider call limit exceeded",
    ):
        ledger.start_final_call("final-over-limit")
    assert len(ledger.to_dict()["calls"]) == TRUE_MOA_FINAL_CALL_LIMIT
