"""Adversarial contract tests for trusted chat approval and steer controls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import (
    APIServerAdapter,
    _ChatControlBridge,
    _ChatControlConflict,
)
from gateway.config import PlatformConfig
from gateway.platforms.true_moa_idempotency import _IdempotencyCache
from tools import approval as approval_module
from xiaoban.trusted_runtime.protocol_contract import (
    TRUSTED_RUNTIME_CONTRACT_DIGEST,
    TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
    TRUSTED_RUNTIME_CONTRACT_REVISION,
    TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
)


def _queue_approval(session_key: str, approval_id: str):
    entry = approval_module._ApprovalEntry({"approvalId": approval_id})
    with approval_module._lock:
        approval_module._gateway_queues.setdefault(session_key, []).append(entry)
    return entry


@pytest.fixture(autouse=True)
def _clear_approval_queues():
    yield
    with approval_module._lock:
        entries = [
            entry
            for queue in approval_module._gateway_queues.values()
            for entry in queue
        ]
        approval_module._gateway_queues.clear()
    for entry in entries:
        entry.result = "deny"
        entry.event.set()


def test_exact_approval_resolution_never_falls_back_to_fifo():
    session_key = "chat-control-exact"
    first = _queue_approval(session_key, "approval-first")
    second = _queue_approval(session_key, "approval-second")
    frame_observations = []

    resolved = approval_module.resolve_gateway_approval_exact(
        session_key,
        "approval-first",
        "once",
        before_unblock=lambda _data: frame_observations.append(first.event.is_set()),
    )

    assert resolved == 1
    assert frame_observations == [False]
    assert first.event.is_set() is True
    assert first.result == "once"
    assert second.event.is_set() is False
    assert approval_module.resolve_gateway_approval_exact(
        session_key,
        "approval-first",
        "once",
    ) == 0
    assert second.event.is_set() is False


@pytest.mark.asyncio
async def test_control_fingerprint_locates_only_exact_active_agent_ref():
    cache = _IdempotencyCache(max_items=8, ttl_seconds=30)
    key = "mystand:scoped-owner"
    internal_fingerprint = "a" * 64
    request_fingerprint = "b" * 64
    agent_ref = [object(), False, None]
    started = asyncio.Event()
    release = asyncio.Event()

    async def _compute():
        started.set()
        await release.wait()
        return {"completed": True}

    task = asyncio.create_task(cache.get_or_set(
        key,
        internal_fingerprint,
        _compute,
        agent_ref=agent_ref,
        control_fingerprint=request_fingerprint,
    ))
    await started.wait()
    try:
        assert cache.active_agent_ref(key, request_fingerprint) is agent_ref
        assert cache.active_agent_ref(key, "c" * 64) is None
        assert cache.active_agent_ref("mystand:other-owner", request_fingerprint) is None
    finally:
        release.set()
        await task
    assert cache.active_agent_ref(key, request_fingerprint) is None


def _make_bridge(*, session_key: str = "chat-control-bridge"):
    emitted = []
    lifecycle_lock = threading.Lock()
    open_tool_calls = {
        "call-current": ("mystand_query", ("delivery-current", "turn-current")),
    }

    class Agent:
        def __init__(self):
            self.messages = []

        def steer(self, message):
            self.messages.append(message)
            return True

    agent = Agent()
    agent_ref = [agent, False, None]
    bridge = _ChatControlBridge(
        request_id="delivery-current",
        approval_session_key=session_key,
        lifecycle_lock=lifecycle_lock,
        started_turn_getter=lambda: {
            "type": "turn.started",
            "requestId": "delivery-current",
            "turnId": "turn-current",
        },
        open_tool_calls=open_tool_calls,
        agent_ref=agent_ref,
        emit=lambda event_name, payload: emitted.append((event_name, payload)),
    )
    return bridge, emitted, open_tool_calls, agent


def _notify_current_approval(bridge, approval_id: str):
    bridge.approval_notify({
        "approvalId": approval_id,
        "requestId": "delivery-current",
        "turnId": "turn-current",
        "callId": "call-current",
        "command": "PRIVATE COMMAND MUST NEVER ENTER A RECEIPT",
        "description": "PRIVATE ARGS MUST NEVER ENTER A RECEIPT",
    })


def test_duplicate_click_replays_receipt_without_approving_next_item():
    session_key = "chat-control-replay"
    bridge, emitted, _open, _agent = _make_bridge(session_key=session_key)
    first = _queue_approval(session_key, "approval-first")
    second = _queue_approval(session_key, "approval-second")
    _notify_current_approval(bridge, "approval-first")

    receipt = bridge.respond(
        control_id="control-same",
        approval_id="approval-first",
        choice="once",
    )
    replay = bridge.respond(
        control_id="control-same",
        approval_id="approval-first",
        choice="once",
    )

    assert replay == receipt
    assert receipt["choice"] == "once"
    assert first.result == "once"
    assert second.event.is_set() is False
    assert [name for name, _payload in emitted] == [
        "approval.request",
        "approval.responded",
    ]


def test_stale_approval_and_control_id_payload_conflict_fail_closed():
    session_key = "chat-control-conflict"
    bridge, _emitted, _open, _agent = _make_bridge(session_key=session_key)
    _queue_approval(session_key, "approval-current")
    _notify_current_approval(bridge, "approval-current")

    with pytest.raises(_ChatControlConflict) as stale:
        bridge.respond(
            control_id="control-stale",
            approval_id="approval-missing",
            choice="once",
        )
    assert stale.value.code == "approval_not_pending"

    bridge.respond(
        control_id="control-conflict",
        approval_id="approval-current",
        choice="once",
    )
    with pytest.raises(_ChatControlConflict) as conflict:
        bridge.respond(
            control_id="control-conflict",
            approval_id="approval-current",
            choice="deny",
        )
    assert conflict.value.code == "control_id_conflict"


def test_pending_approval_is_closed_before_same_turn_steer_is_accepted():
    session_key = "chat-control-steer"
    bridge, emitted, _open, agent = _make_bridge(session_key=session_key)
    entry = _queue_approval(session_key, "approval-pending")
    _notify_current_approval(bridge, "approval-pending")
    private_message = "PRIVATE SUPPLEMENT BODY MUST NOT ENTER RECEIPT"

    receipt = bridge.steer(
        control_id="control-steer",
        message=private_message,
        approval_id="approval-pending",
    )

    assert entry.event.is_set() is True
    assert entry.result == "deny"
    assert agent.messages == [private_message]
    assert [name for name, _payload in emitted] == [
        "approval.request",
        "approval.responded",
        "steer.accepted",
    ]
    assert emitted[1][1]["choice"] == "deny"
    assert receipt["approvalId"] == "approval-pending"
    assert receipt["messageDigest"] == hashlib.sha256(
        private_message.encode("utf-8")
    ).hexdigest()
    receipt_wire = json.dumps(receipt, ensure_ascii=False)
    assert private_message not in receipt_wire
    assert "PRIVATE COMMAND" not in receipt_wire
    assert "PRIVATE ARGS" not in receipt_wire


def test_steer_uses_ecmascript_trim_for_cross_runtime_digest():
    bridge, _emitted, _open_tool_calls, agent = _make_bridge(
        session_key="chat-control-ecmascript-trim"
    )
    nel_message = "\u0085private supplement\u0085"

    nel_receipt = bridge.steer(
        control_id="control-steer-nel",
        message=nel_message,
    )
    trimmed_receipt = bridge.steer(
        control_id="control-steer-es-whitespace",
        message="\ufeff\u3000trimmed supplement\u3000\ufeff",
    )

    assert agent.messages == [nel_message, "trimmed supplement"]
    assert nel_receipt["approvalId"] == ""
    assert trimmed_receipt["approvalId"] == ""
    assert nel_receipt["messageDigest"] == hashlib.sha256(
        nel_message.encode("utf-8")
    ).hexdigest()
    assert trimmed_receipt["messageDigest"] == hashlib.sha256(
        b"trimmed supplement"
    ).hexdigest()


def test_unique_control_limit_is_bounded_but_exact_replay_survives():
    bridge, _emitted, _open_tool_calls, agent = _make_bridge(
        session_key="chat-control-bounded-receipts"
    )
    receipts = [
        bridge.steer(
            control_id=f"control-steer-{index}",
            message=f"supplement {index}",
        )
        for index in range(8)
    ]

    assert len(agent.messages) == 8
    assert bridge.steer(
        control_id="control-steer-0",
        message="supplement 0",
    ) == receipts[0]
    assert len(agent.messages) == 8
    with pytest.raises(_ChatControlConflict) as limited:
        bridge.steer(
            control_id="control-steer-9",
            message="supplement 9",
        )
    assert limited.value.code == "chat_control_limit_reached"
    assert len(agent.messages) == 8


def test_pending_approval_limit_is_bounded_but_exact_notify_replay_survives():
    bridge, emitted, _open_tool_calls, _agent = _make_bridge(
        session_key="chat-control-bounded-pending"
    )
    for index in range(8):
        _notify_current_approval(bridge, f"approval-pending-{index}")

    _notify_current_approval(bridge, "approval-pending-0")
    assert len(emitted) == 8
    with pytest.raises(_ChatControlConflict) as limited:
        _notify_current_approval(bridge, "approval-pending-8")
    assert limited.value.code == "chat_control_limit_reached"
    assert len(emitted) == 8


def test_late_steer_after_tool_terminal_is_rejected():
    bridge, _emitted, open_tool_calls, agent = _make_bridge()
    open_tool_calls.clear()

    with pytest.raises(_ChatControlConflict) as late:
        bridge.steer(
            control_id="control-late",
            message="late body",
        )

    assert late.value.code == "steer_not_active"
    assert agent.messages == []


def test_pending_approval_steer_rejects_missing_or_changed_slot():
    session_key = "chat-control-steer-slot"
    bridge, emitted, _open, agent = _make_bridge(session_key=session_key)
    current = _queue_approval(session_key, "approval-current")
    _notify_current_approval(bridge, "approval-current")

    with pytest.raises(_ChatControlConflict) as missing:
        bridge.steer(
            control_id="control-missing-slot",
            message="supplement",
        )
    assert missing.value.code == "steer_approval_id_required"

    with pytest.raises(_ChatControlConflict) as changed:
        bridge.steer(
            control_id="control-old-slot",
            message="supplement",
            approval_id="approval-old",
        )
    assert changed.value.code == "steer_approval_changed"
    assert current.event.is_set() is False
    assert agent.messages == []
    assert [name for name, _payload in emitted] == ["approval.request"]


def test_steer_rejects_multiple_pending_approvals_or_unbound_open_tools():
    session_key = "chat-control-steer-ambiguous"
    bridge, _emitted, open_tool_calls, agent = _make_bridge(
        session_key=session_key
    )
    first = _queue_approval(session_key, "approval-first")
    second = _queue_approval(session_key, "approval-second")
    _notify_current_approval(bridge, "approval-first")
    _notify_current_approval(bridge, "approval-second")

    with pytest.raises(_ChatControlConflict) as approvals:
        bridge.steer(
            control_id="control-ambiguous-approval",
            message="supplement",
            approval_id="approval-first",
        )
    assert approvals.value.code == "steer_approval_ambiguous"
    assert first.event.is_set() is False
    assert second.event.is_set() is False
    assert agent.messages == []

    approval_module.resolve_gateway_approval_exact(
        session_key,
        "approval-first",
        "deny",
    )
    approval_module.resolve_gateway_approval_exact(
        session_key,
        "approval-second",
        "deny",
    )
    bridge.close()

    plain_bridge, _plain_events, plain_open, plain_agent = _make_bridge(
        session_key="chat-control-multiple-tools"
    )
    plain_open["call-other"] = (
        "mystand_query",
        ("delivery-current", "turn-current"),
    )
    with pytest.raises(_ChatControlConflict) as tools:
        plain_bridge.steer(
            control_id="control-ambiguous-tool",
            message="supplement",
        )
    assert tools.value.code == "steer_tool_ambiguous"
    assert plain_agent.messages == []


def test_combined_steer_targets_exact_approval_call_and_preserves_other_call():
    session_key = "chat-control-steer-independent-calls"
    bridge, emitted, open_tool_calls, agent = _make_bridge(
        session_key=session_key
    )
    open_tool_calls["call-other"] = (
        "mystand_query",
        ("delivery-current", "turn-current"),
    )
    selected = _queue_approval(session_key, "approval-selected")
    other = _queue_approval(session_key, "approval-other")
    _notify_current_approval(bridge, "approval-selected")
    bridge.approval_notify({
        "approvalId": "approval-other",
        "requestId": "delivery-current",
        "turnId": "turn-current",
        "callId": "call-other",
    })

    receipt = bridge.steer(
        control_id="control-exact-call",
        message="supplement for selected call",
        approval_id="approval-selected",
    )

    assert selected.result == "deny"
    assert selected.event.is_set() is True
    assert other.event.is_set() is False
    assert agent.messages == ["supplement for selected call"]
    assert receipt["approvalId"] == "approval-selected"
    assert receipt["event"]["callId"] == "call-current"
    assert [name for name, _payload in emitted] == [
        "approval.request",
        "approval.request",
        "approval.responded",
        "steer.accepted",
    ]


def test_fatal_close_emits_fail_closed_response_for_each_pending_approval():
    session_key = "chat-control-fatal-close"
    bridge, emitted, _open, _agent = _make_bridge(session_key=session_key)
    _queue_approval(session_key, "approval-fatal")
    _notify_current_approval(bridge, "approval-fatal")

    bridge.close()

    assert [name for name, _payload in emitted] == [
        "approval.request",
        "approval.responded",
    ]
    closed = emitted[-1][1]
    assert closed["approvalId"] == "approval-fatal"
    assert closed["choice"] == "deny"
    assert closed["status"] == "completed"
    assert closed["controlId"].startswith("control_system_close_")
    assert closed["summary"] == "运行已结束，审批等待已安全关闭。"


def _control_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(
        enabled=True,
        extra={"key": "sk-secret"},
    ))


def _control_headers(
    delivery_id: str,
    *,
    user: str = "alice",
    fingerprint: str | None = None,
) -> dict[str, str]:
    return {
        "Authorization": "Bearer sk-secret",
        "X-Xiaoban-Site-Id": "mystand-test-site",
        "X-Xiaoban-User-Id": user,
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Memory-Mode": "disabled",
        "X-Xiaoban-Session-Key": f"session-{user}",
        "X-Xiaoban-Session-Id": f"session-{user}",
        "X-Xiaoban-Message-Id": f"message-{delivery_id}",
        "X-Xiaoban-Attempt": "1",
        "X-Xiaoban-Delivery-Id": delivery_id,
        "X-Xiaoban-Delivery-Attempt": "1",
        "X-Xiaoban-Request-Fingerprint": fingerprint or hashlib.sha256(
            f"request:{delivery_id}".encode("utf-8")
        ).hexdigest(),
        TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER: TRUSTED_RUNTIME_CONTRACT_REVISION,
        TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER: TRUSTED_RUNTIME_CONTRACT_DIGEST,
    }


def _control_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/approval",
        adapter._handle_chat_completion_approval,
    )
    app.router.add_post(
        "/v1/chat/completions/steer",
        adapter._handle_chat_completion_steer,
    )
    return app


@pytest.mark.asyncio
async def test_control_endpoint_binds_owner_and_request_fingerprint(
    monkeypatch,
    tmp_path,
):
    adapter = _control_adapter()
    cache = _IdempotencyCache(
        durable_path=str(tmp_path / "chat-controls.sqlite"),
    )
    monkeypatch.setattr("gateway.platforms.api_server._idem_cache", cache)
    delivery_id = "xbd_" + "1" * 40
    headers = _control_headers(delivery_id)
    scoped_key = adapter._scoped_idempotency_key(headers, delivery_id)
    release = asyncio.Event()
    started = asyncio.Event()

    class Bridge:
        def respond(self, **kwargs):
            return {"ok": True, "status": "accepted", **kwargs}

    agent_ref = [object(), False, None, Bridge()]

    async def _compute():
        started.set()
        await release.wait()
        return {"completed": True}

    active = asyncio.create_task(cache.get_or_set(
        scoped_key,
        "a" * 64,
        _compute,
        agent_ref=agent_ref,
        control_fingerprint=headers["X-Xiaoban-Request-Fingerprint"],
    ))
    await started.wait()
    body = {
        "idempotency_key": delivery_id,
        "controlId": "control-owner",
        "approvalId": "approval-owner",
        "choice": "once",
    }
    try:
        async with TestClient(TestServer(_control_app(adapter))) as client:
            wrong_owner = await client.post(
                "/v1/chat/completions/approval",
                headers=_control_headers(delivery_id, user="bob"),
                json=body,
            )
            wrong_fingerprint = await client.post(
                "/v1/chat/completions/approval",
                headers=_control_headers(delivery_id, fingerprint="f" * 64),
                json=body,
            )
            accepted = await client.post(
                "/v1/chat/completions/approval",
                headers=headers,
                json=body,
            )
            accepted_body = await accepted.json()
        assert wrong_owner.status == 409
        assert wrong_fingerprint.status == 409
        assert accepted.status == 202
        assert accepted_body == {
            "ok": True,
            "status": "accepted",
            "control_id": "control-owner",
            "approval_id": "approval-owner",
            "choice": "once",
        }
    finally:
        release.set()
        await active


@pytest.mark.asyncio
async def test_steer_http_exact_approval_id_closes_toctou_window(
    monkeypatch,
    tmp_path,
):
    adapter = _control_adapter()
    cache = _IdempotencyCache(
        durable_path=str(tmp_path / "chat-steer-http.sqlite"),
    )
    monkeypatch.setattr("gateway.platforms.api_server._idem_cache", cache)
    delivery_id = "xbd_" + "6" * 40
    headers = _control_headers(delivery_id)
    session_key = headers["X-Xiaoban-Session-Key"]
    emitted = []
    open_tool_calls = {
        "call-http": ("mystand_query", (delivery_id, "7" * 16)),
    }

    class Agent:
        def __init__(self):
            self.messages = []

        def steer(self, message):
            self.messages.append(message)
            return True

    agent = Agent()
    agent_ref = [agent, False, None]
    bridge = _ChatControlBridge(
        request_id=delivery_id,
        approval_session_key=session_key,
        lifecycle_lock=threading.Lock(),
        started_turn_getter=lambda: {
            "type": "turn.started",
            "requestId": delivery_id,
            "turnId": "7" * 16,
        },
        open_tool_calls=open_tool_calls,
        agent_ref=agent_ref,
        emit=lambda event_name, payload: emitted.append((event_name, payload)),
    )
    agent_ref.append(bridge)
    entry = _queue_approval(session_key, "approval-http")
    bridge.approval_notify({
        "approvalId": "approval-http",
        "requestId": delivery_id,
        "turnId": "7" * 16,
        "callId": "call-http",
    })
    scoped_key = adapter._scoped_idempotency_key(headers, delivery_id)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _compute():
        started.set()
        await release.wait()
        return {"completed": True}

    active = asyncio.create_task(cache.get_or_set(
        scoped_key,
        "8" * 64,
        _compute,
        agent_ref=agent_ref,
        control_fingerprint=headers["X-Xiaoban-Request-Fingerprint"],
    ))
    await started.wait()
    private_message = "PRIVATE HTTP STEER BODY"
    base_body = {
        "idempotency_key": delivery_id,
        "message": private_message,
    }
    try:
        async with TestClient(TestServer(_control_app(adapter))) as client:
            missing = await client.post(
                "/v1/chat/completions/steer",
                headers=headers,
                json={**base_body, "controlId": "control-http-missing"},
            )
            changed = await client.post(
                "/v1/chat/completions/steer",
                headers=headers,
                json={
                    **base_body,
                    "controlId": "control-http-changed",
                    "approvalId": "approval-old",
                },
            )
            accepted = await client.post(
                "/v1/chat/completions/steer",
                headers=headers,
                json={
                    **base_body,
                    "controlId": "control-http-accepted",
                    "approvalId": "approval-http",
                },
            )
            missing_body = await missing.json()
            changed_body = await changed.json()
            accepted_body = await accepted.json()

            open_tool_calls["call-http-other"] = (
                "mystand_query",
                (delivery_id, "7" * 16),
            )
            ambiguous = await client.post(
                "/v1/chat/completions/steer",
                headers=headers,
                json={
                    **base_body,
                    "controlId": "control-http-ambiguous",
                },
            )
            ambiguous_body = await ambiguous.json()

        assert missing.status == 409
        assert missing_body["error"]["code"] == "steer_approval_id_required"
        assert changed.status == 409
        assert changed_body["error"]["code"] == "steer_approval_changed"
        assert entry.event.is_set() is True
        assert entry.result == "deny"
        assert accepted.status == 202
        assert accepted_body["messageDigest"] == hashlib.sha256(
            private_message.encode("utf-8")
        ).hexdigest()
        assert private_message not in json.dumps(accepted_body, ensure_ascii=False)
        assert agent.messages == [private_message]
        assert ambiguous.status == 409
        assert ambiguous_body["error"]["code"] == "steer_tool_ambiguous"
    finally:
        release.set()
        await active


@pytest.mark.asyncio
async def test_stream_emits_approval_response_before_tool_terminal(
    monkeypatch,
    tmp_path,
):
    from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

    adapter = _control_adapter()
    cache = _IdempotencyCache(
        durable_path=str(tmp_path / "chat-control-stream.sqlite"),
    )
    monkeypatch.setattr("gateway.platforms.api_server._idem_cache", cache)
    delivery_id = "xbd_" + "2" * 40
    headers = _control_headers(delivery_id)
    turn_id = "3" * 16
    call_id = "call-control"
    approval_id = "approval-stream"
    approval_ready = asyncio.Event()
    private_command = "PRIVATE COMMAND MUST NOT ENTER SSE"

    class Agent:
        def steer(self, _message):
            return True

    async def _mock_run_agent(**kwargs):
        kwargs["agent_ref"][0] = Agent()
        kwargs["tool_progress_callback"](
            "turn.started",
            delivery_id,
            turn_id,
            None,
        )
        kwargs["tool_start_callback"](
            call_id,
            "mystand_query",
            {"query": private_command},
        )
        bridge = kwargs["agent_ref"][3]
        entry = _queue_approval(bridge.approval_session_key, approval_id)
        bridge.approval_notify({
            "approvalId": approval_id,
            "requestId": delivery_id,
            "turnId": turn_id,
            "callId": call_id,
            "command": private_command,
            "description": private_command,
        })
        approval_ready.set()
        while not entry.event.is_set():
            await asyncio.sleep(0.01)
        assert entry.result == "once"
        kwargs["tool_complete_callback"](
            call_id,
            "mystand_query",
            {"query": private_command},
            {"ok": True},
            {
                "schema": "xiaoban.tool-result.v1",
                "requestId": delivery_id,
                "turnId": turn_id,
                "callId": call_id,
                "toolName": "mystand_query",
                "dispatchState": "dispatched",
                "outcome": "success",
                "retrySafe": False,
            },
        )
        kwargs["stream_delta_callback"]("最终答复。")
        ledger = AgentCallUsageLedger(provider="test", model="test")
        ledger.set_status("completed")
        return (
            {
                "final_response": "最终答复。",
                "completed": True,
                "failed": False,
                "partial": False,
                "interrupted": False,
                "messages": [],
            },
            {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "agent_calls": ledger.to_dict(),
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post(
        "/v1/chat/completions/approval",
        adapter._handle_chat_completion_approval,
    )
    request_body = {
        "model": "test",
        "messages": [{"role": "user", "content": "run controlled tool"}],
        "stream": True,
    }
    async with TestClient(TestServer(app)) as client:
        with monkeypatch.context() as context:
            context.setattr(adapter, "_run_agent", _mock_run_agent)
            stream_task = asyncio.create_task(client.post(
                "/v1/chat/completions",
                headers=headers,
                json=request_body,
            ))
            await asyncio.wait_for(approval_ready.wait(), timeout=2)
            response = await client.post(
                "/v1/chat/completions/approval",
                headers=headers,
                json={
                    "idempotency_key": delivery_id,
                    "controlId": "control-stream",
                    "approvalId": approval_id,
                    "choice": "once",
                },
            )
            receipt = await response.json()
            stream_response = await stream_task
            stream_body = await stream_response.text()

    assert response.status == 202
    assert receipt["event"]["type"] == "approval.responded"
    assert receipt["event"]["status"] == "completed"
    assert receipt["event"]["requestId"] == delivery_id
    assert receipt["event"]["turnId"] == turn_id
    assert receipt["event"]["callId"] == call_id
    assert receipt["event"]["approvalId"] == approval_id
    assert private_command not in json.dumps(receipt, ensure_ascii=False)
    request_event = next(
        payload
        for name, payload in (
            (
                line[len("event: xiaoban."):],
                json.loads(stream_body.splitlines()[index + 1][len("data: "):]),
            )
            for index, line in enumerate(stream_body.splitlines()[:-1])
            if line.startswith("event: xiaoban.")
            and stream_body.splitlines()[index + 1].startswith("data: ")
        )
        if name == "approval.request"
    )
    assert request_event["choices"] == ["once", "session", "deny"]
    assert request_event["status"] == "running"
    request_index = stream_body.index("event: xiaoban.approval.request")
    responded_index = stream_body.index("event: xiaoban.approval.responded")
    terminal_index = stream_body.index(
        f'"toolCallId": "{call_id}", "status": "completed"'
    )
    assert request_index < responded_index < terminal_index
    assert private_command not in stream_body
    assert stream_body.count("data: [DONE]") == 1


@pytest.mark.asyncio
async def test_fatal_stream_closes_pending_approval_before_terminals(
    monkeypatch,
    tmp_path,
):
    from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger
    from tools import approval as approval_module

    adapter = _control_adapter()
    cache = _IdempotencyCache(
        durable_path=str(tmp_path / "chat-control-fatal-stream.sqlite"),
    )
    monkeypatch.setattr("gateway.platforms.api_server._idem_cache", cache)
    delivery_id = "xbd_" + "4" * 40
    headers = _control_headers(delivery_id)
    turn_id = "5" * 16
    call_id = "call-fatal-control"
    private_command = "PRIVATE FATAL COMMAND MUST NOT ENTER SSE"
    terminal_callback = {}
    original_unregister = approval_module.unregister_gateway_notify

    def racing_unregister(session_key):
        original_unregister(session_key)
        terminal_callback["emit"]()

    monkeypatch.setattr(
        approval_module,
        "unregister_gateway_notify",
        racing_unregister,
    )

    async def _mock_run_agent(**kwargs):
        kwargs["tool_progress_callback"](
            "turn.started", delivery_id, turn_id, None,
        )
        kwargs["tool_start_callback"](
            call_id,
            "mystand_query",
            {"query": private_command},
        )
        bridge = kwargs["agent_ref"][3]
        _queue_approval(bridge.approval_session_key, "approval-fatal-stream")
        bridge.approval_notify({
            "approvalId": "approval-fatal-stream",
            "requestId": delivery_id,
            "turnId": turn_id,
            "callId": call_id,
            "command": private_command,
            "description": private_command,
        })
        terminal_callback["emit"] = lambda: kwargs["tool_complete_callback"](
            call_id,
            "mystand_query",
            {"query": private_command},
            {"ok": False},
            {
                "schema": "xiaoban.tool-result.v1",
                "requestId": delivery_id,
                "turnId": turn_id,
                "callId": call_id,
                "toolName": "mystand_query",
                "dispatchState": "dispatched",
                "outcome": "failed",
                "retrySafe": False,
            },
        )
        ledger = AgentCallUsageLedger(provider="test", model="test")
        ledger.set_status("failed")
        return (
            {
                "final_response": "",
                "completed": False,
                "failed": True,
                "partial": False,
                "interrupted": False,
                "messages": [],
            },
            {
                "input_tokens": 1,
                "output_tokens": 0,
                "total_tokens": 1,
                "agent_calls": ledger.to_dict(),
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        with monkeypatch.context() as context:
            context.setattr(adapter, "_run_agent", _mock_run_agent)
            response = await asyncio.wait_for(client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "fatal"}],
                    "stream": True,
                },
            ), timeout=2)
            stream_body = await asyncio.wait_for(response.text(), timeout=2)

    request_index = stream_body.index("event: xiaoban.approval.request")
    responded_index = stream_body.index("event: xiaoban.approval.responded")
    tool_terminal_index = stream_body.index(
        f'"toolCallId": "{call_id}", "status": "failed"'
    )
    turn_terminal_index = stream_body.index('"type": "turn.failed"')
    assert request_index < responded_index < tool_terminal_index < turn_terminal_index
    assert "control_system_close_" in stream_body
    assert private_command not in stream_body
    assert stream_body.count("data: [DONE]") == 1
