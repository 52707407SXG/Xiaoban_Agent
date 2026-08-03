"""Canonical physical K3 call/result binding and DataScope tests."""

import json

import pytest

from xiaoban.trusted_runtime import (
    TrustedIdentity,
    begin_action,
    begin_turn,
    finish_action,
)


IDENTITY = TrustedIdentity(
    account_id="user-a",
    data_scope="mystand",
    source="server_session",
)


def _turn(seed: str = "canonical", identity=IDENTITY):
    return begin_turn(
        channel="web",
        user_message="读取 AUTH-ABC12345",
        identity=identity,
        request_id=f"request-{seed}",
        message_id=f"message-{seed}",
    )


def _start(turn, *, call_id="call-read"):
    decision = begin_action(
        turn,
        "mystand_authorization",
        "v1",
        {"operation": "resolve", "authorization_id": "AUTH-ABC12345"},
        call_id=call_id,
    )
    assert decision.decision == "allow"
    return decision.call


def test_result_requires_a_current_bound_call_id():
    turn = _turn("orphan")
    result = finish_action(
        turn,
        "forged-call",
        "mystand_authorization",
        "v1",
        {"ok": True, "content": "PRIVATE"},
    )
    assert result is None
    assert turn.action_calls == []
    assert turn.action_results == []
    assert turn.orphaned_receipts == 1
    assert "verifying" not in turn.states


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"ok": True, "content": ""}, "empty"),
        ({"ok": False, "status": 500, "code": "failed"}, "error"),
        ({"ok": False, "status": 403, "code": "denied"}, "denied"),
        ({"ok": False, "status": 409, "code": "ambiguous"}, "ambiguous"),
        ({"ok": False, "status": 404, "code": "missing"}, "not_found"),
    ],
)
def test_result_statuses_are_classified_distinctly(payload, expected):
    turn = _turn(expected)
    call = _start(turn, call_id=f"call-{expected}")
    result = finish_action(
        turn,
        call.call_id,
        call.action_id,
        call.version,
        payload,
    )
    assert result is not None
    assert result.status == expected
    assert "executing" in turn.states
    assert "verifying" in turn.states


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": True, "status": 500, "content": "PRIVATE"},
        {"ok": True, "status": "500", "content": "PRIVATE"},
        {"ok": True, "error": "bridge failure", "content": "PRIVATE"},
        {"ok": True, "code": "read_failed", "content": "PRIVATE"},
    ],
)
def test_contradictory_handler_receipts_fail_closed(payload):
    turn = _turn("contradiction")
    call = _start(turn, call_id="call-contradiction")
    result = finish_action(
        turn,
        call.call_id,
        call.action_id,
        call.version,
        payload,
    )
    assert result is not None
    assert result.status == "error"


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": True, "ownerUser": "user-b", "content": "PRIVATE"},
        {"ok": True, "teamId": "team-b", "content": "PRIVATE"},
        {"ok": True, "content": "PRIVATE", "items": [{"ownerId": "user-b"}]},
    ],
)
def test_server_identity_and_scope_mismatch_is_recorded(payload):
    turn = _turn("scope")
    call = _start(turn, call_id="call-scope")
    result = finish_action(
        turn,
        call.call_id,
        call.action_id,
        call.version,
        json.dumps(payload),
    )
    assert result is not None
    assert turn.rejected_cross_account == 1


def test_duplicate_call_id_invalidates_the_first_result():
    turn = _turn("duplicate")
    call = _start(turn, call_id="call-duplicate")
    first = finish_action(
        turn,
        call.call_id,
        call.action_id,
        call.version,
        {"ok": True, "content": "first"},
    )
    assert first is not None and first.status == "success"

    duplicate = begin_action(
        turn,
        call.action_id,
        call.version,
        call.arguments,
        call_id=call.call_id,
    )
    assert duplicate.decision == "deny"
    assert duplicate.reason == "duplicate_call_id"
    assert any(
        item.call_id == call.call_id and item.status == "error"
        for item in turn.action_results
    )


def test_action_id_mismatch_is_orphaned():
    turn = _turn("mismatch")
    decision = begin_action(
        turn,
        "mystand_query",
        "v1",
        {"operation": "read", "resource": {"name": "目标资料"}},
        call_id="call-mismatch",
    )
    assert decision.decision == "allow"
    result = finish_action(
        turn,
        decision.call.call_id,
        "mystand_authorization",
        "v1",
        {"ok": True, "content": "PRIVATE"},
    )
    assert result is None
    assert turn.orphaned_receipts == 1


@pytest.mark.parametrize("operation", ["preview_write", "commit_write"])
def test_read_lifecycle_never_claims_write_execution(operation):
    turn = _turn(f"write-{operation}")
    decision = begin_action(
        turn,
        "mystand_authorization",
        "v1",
        {"operation": operation, "authorization_id": "AUTH-ABC12345"},
        call_id=f"call-{operation}",
    )
    assert decision.decision == "deny"
    assert decision.reason == "write_isolated"
    assert turn.action_calls == []


def test_results_are_isolated_between_turns():
    turn_a = _turn("a")
    call_a = _start(turn_a, call_id="call-a")
    finish_action(
        turn_a,
        call_a.call_id,
        call_a.action_id,
        call_a.version,
        {"ok": True, "content": "turn-a"},
    )

    turn_b = _turn("b")
    call_b = _start(turn_b, call_id="call-b")
    finish_action(
        turn_b,
        call_b.call_id,
        call_b.action_id,
        call_b.version,
        {"ok": False, "status": 503, "code": "unavailable"},
    )
    assert all(item.call_id != "call-a" for item in turn_b.action_results)
