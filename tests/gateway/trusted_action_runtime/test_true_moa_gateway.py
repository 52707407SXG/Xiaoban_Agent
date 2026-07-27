"""Gateway integration contracts for My Stand's fixed true-MoA mode.

Every provider-facing boundary in this module is replaced with a deterministic
fake.  These tests must never perform a network request or consume paid tokens.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path
import queue
import subprocess
import sys
import textwrap
import threading
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from xiaoban.trusted_runtime.true_moa import (
    DEEPSEEK_ADVISOR_SLOT,
    FINAL_EXECUTOR_SLOT,
    KIMI_ADVISOR_SLOT,
    MODE_EPOCH_HEADER,
    MOA_PRESET_ID_HEADER,
    MOA_PRESET_REVISION_HEADER,
    REASONING_MODE_HEADER,
    StrictAdvisorResult,
    TRUE_MOA_MODE,
    TRUE_MOA_PRESET_ID,
    TRUE_MOA_PRESET_REVISION,
    TRUE_MOA_USAGE_SCHEMA,
    validate_true_moa_headers,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-test-only"}),
    )


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions",
        adapter._handle_chat_completions,
    )
    return app


def _mystand_headers(
    key: str,
    *,
    mode: str = TRUE_MOA_MODE,
    epoch: str = "17",
    preset_id: str = TRUE_MOA_PRESET_ID,
    preset_revision: str = TRUE_MOA_PRESET_REVISION,
) -> dict[str, str]:
    return {
        "Authorization": "Bearer sk-test-only",
        "Idempotency-Key": key,
        "X-Xiaoban-Site-Id": "mystand-test-site",
        "X-Xiaoban-User-Id": "test-user",
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Memory-Mode": "disabled",
        "X-Xiaoban-Session-Key": "gateway-test-channel",
        "X-Xiaoban-Session-Id": "gateway-test-session",
        "X-Xiaoban-Message-Id": f"message-{key}",
        "X-Xiaoban-Attempt": "1",
        "X-Xiaoban-Request-Fingerprint": hashlib.sha256(
            f"request:{key}".encode(),
        ).hexdigest(),
        REASONING_MODE_HEADER: mode,
        MODE_EPOCH_HEADER: epoch,
        MOA_PRESET_ID_HEADER: preset_id,
        MOA_PRESET_REVISION_HEADER: preset_revision,
    }


def _normal_direct_headers() -> dict[str, str]:
    return {
        "X-Xiaoban-User-Id": "test-user",
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Memory-Mode": "disabled",
        "X-Xiaoban-Message-Id": "normal-direct-message",
        REASONING_MODE_HEADER: "normal",
    }


class _FakeFinalAgent:
    provider = "deepseek"
    model = "deepseek-v4-pro"
    valid_tool_names: set[str] = set()
    tools: list[object] = []
    session_prompt_tokens = 17
    session_completion_tokens = 7
    session_total_tokens = 24
    session_estimated_cost_usd = 0.03
    session_cost_status = "reported"
    session_cost_source = "fake-final-agent"
    session_id = "gateway-test-session"

    def __init__(self) -> None:
        self.run_calls: list[dict[str, object]] = []
        self.interrupt_calls: list[str] = []
        self.ephemeral_system_prompt = ""

    def run_conversation(self, **kwargs):
        self.run_calls.append(kwargs)
        return {
            "final_response": "fake final synthesis",
            "completed": True,
            "failed": False,
            "messages": [],
        }

    def interrupt(self, reason: str) -> None:
        self.interrupt_calls.append(reason)


def test_normal_fresh_subprocess_does_not_import_true_moa_or_fan_out():
    code = textwrap.dedent(
        """
        import asyncio
        import sys

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        TRUE_MOA = "xiaoban.trusted_runtime.true_moa"
        TRUE_MOA_PROVIDERS = "xiaoban.trusted_runtime.true_moa_providers"
        assert TRUE_MOA not in sys.modules
        assert TRUE_MOA_PROVIDERS not in sys.modules

        class FakeAgent:
            provider = "deepseek"
            model = "deepseek-v4-pro"
            valid_tool_names = set()
            session_prompt_tokens = 2
            session_completion_tokens = 1
            session_total_tokens = 3
            session_id = "fresh-normal"

            def __init__(self):
                self.calls = 0

            def run_conversation(self, **_kwargs):
                self.calls += 1
                return {
                    "final_response": "normal",
                    "completed": True,
                    "messages": [],
                }

        async def main():
            adapter = APIServerAdapter(
                PlatformConfig(enabled=True, extra={"key": "fake-only"})
            )
            created = []

            def create_agent(**_kwargs):
                agent = FakeAgent()
                created.append(agent)
                return agent

            adapter._create_agent = create_agent
            app = web.Application()
            app.router.add_post(
                "/v1/chat/completions",
                adapter._handle_chat_completions,
            )
            async with TestClient(TestServer(app)) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer fake-only",
                        "X-Xiaoban-Reasoning-Mode": "normal",
                    },
                    json={
                        "model": "xiaoban-agent",
                        "messages": [
                            {"role": "user", "content": "normal request"}
                        ],
                    },
                )
                payload = await response.json()
            assert response.status == 200
            assert payload["choices"][0]["message"]["content"] == "normal"
            assert len(created) == 1
            assert created[0].calls == 1
            assert TRUE_MOA not in sys.modules
            assert TRUE_MOA_PROVIDERS not in sys.modules

        asyncio.run(main())
        """
    )
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, (
        f"fresh normal subprocess failed\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("header_name", "bad_value", "expected_code"),
    [
        (
            REASONING_MODE_HEADER,
            "moa-template",
            "unsupported_reasoning_mode",
        ),
        (
            MODE_EPOCH_HEADER,
            "-1",
            "invalid_mode_epoch",
        ),
        (
            MOA_PRESET_ID_HEADER,
            "client-selected-preset",
            "invalid_true_moa_preset_id",
        ),
        (
            MOA_PRESET_REVISION_HEADER,
            "latest",
            "invalid_true_moa_preset_revision",
        ),
    ],
)
async def test_each_true_moa_header_error_fails_before_agent_or_provider(
    monkeypatch,
    header_name,
    bad_value,
    expected_code,
):
    adapter = _adapter()
    create_agent = MagicMock(
        side_effect=AssertionError("invalid headers reached agent creation"),
    )
    monkeypatch.setattr(adapter, "_create_agent", create_agent)

    provider_calls: list[str] = []
    fake_provider_module = types.ModuleType(
        "xiaoban.trusted_runtime.true_moa_providers",
    )

    def _provider_must_not_run(**_kwargs):
        provider_calls.append("called")
        raise AssertionError("invalid headers reached a provider")

    fake_provider_module.strict_advisor_call = _provider_must_not_run
    monkeypatch.setitem(
        sys.modules,
        "xiaoban.trusted_runtime.true_moa_providers",
        fake_provider_module,
    )

    headers = _mystand_headers(f"invalid-{uuid.uuid4().hex}")
    headers[header_name] = bad_value
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "messages": [{"role": "user", "content": "must fail closed"}],
            },
        )
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == expected_code
    create_agent.assert_not_called()
    assert provider_calls == []


@pytest.mark.parametrize(
    ("header_name", "changed_value"),
    [
        (REASONING_MODE_HEADER, "normal"),
        (MODE_EPOCH_HEADER, "18"),
        (MOA_PRESET_ID_HEADER, "other-preset"),
        (MOA_PRESET_REVISION_HEADER, "other-revision"),
    ],
)
def test_mode_epoch_and_preset_headers_are_bound_into_idempotency_fingerprint(
    header_name,
    changed_value,
):
    key = f"fingerprint-{uuid.uuid4().hex}"
    headers = _mystand_headers(key)
    body = {
        "model": "xiaoban-agent",
        "messages": [{"role": "user", "content": "same durable request"}],
    }

    baseline = APIServerAdapter._chat_idempotency_fingerprint(body, headers)
    changed_headers = {**headers, header_name: changed_value}
    changed = APIServerAdapter._chat_idempotency_fingerprint(
        body,
        changed_headers,
    )

    assert changed != baseline


@pytest.mark.asyncio
async def test_http_passes_one_frozen_true_moa_snapshot_into_run_agent():
    adapter = _adapter()
    captured: dict[str, object] = {}

    async def _fake_run_agent(**kwargs):
        captured.update(kwargs)
        return (
            {
                "final_response": "snapshot accepted",
                "completed": True,
                "messages": [],
            },
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    headers = _mystand_headers(f"snapshot-{uuid.uuid4().hex}", epoch="23")
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(
            adapter,
            "_run_agent",
            side_effect=_fake_run_agent,
        ) as run_agent:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "xiaoban-agent",
                    "messages": [{"role": "user", "content": "use this snapshot"}],
                },
            )
            await response.read()

    assert response.status == 200
    run_agent.assert_awaited_once()
    snapshot = captured["true_moa_snapshot"]
    assert snapshot.mode == TRUE_MOA_MODE
    assert snapshot.mode_epoch == "23"
    assert snapshot.preset_id == TRUE_MOA_PRESET_ID
    assert snapshot.preset_revision == TRUE_MOA_PRESET_REVISION
    with pytest.raises(FrozenInstanceError):
        snapshot.mode_epoch = "24"


@pytest.mark.asyncio
async def test_gateway_runs_two_fake_advisors_and_one_fake_final_with_one_ledger(
    monkeypatch,
):
    from xiaoban.trusted_runtime import true_moa_providers

    adapter = _adapter()
    headers = _mystand_headers(f"wave-{uuid.uuid4().hex}", epoch="31")
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )

    advisor_calls: Counter[str] = Counter()
    calls_lock = threading.Lock()

    def _fake_strict_advisor(*, slot, tools, **_kwargs):
        assert tools == ()
        with calls_lock:
            advisor_calls[slot.slot_id] += 1
        if slot == KIMI_ADVISOR_SLOT:
            return StrictAdvisorResult(
                content="fake kimi advice",
                usage={"input_tokens": 11, "output_tokens": 3},
                cost_usd=0.01,
                cost_status="reported",
                cost_source="fake-kimi",
            )
        assert slot == DEEPSEEK_ADVISOR_SLOT
        return StrictAdvisorResult(
            content="fake deepseek advice",
            usage={
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "total_tokens": 18,
            },
            cost_usd=0.02,
            cost_status="reported",
            cost_source="fake-deepseek",
        )

    monkeypatch.setattr(
        true_moa_providers,
        "strict_advisor_call",
        _fake_strict_advisor,
    )

    final_agent = _FakeFinalAgent()
    create_kwargs: dict[str, object] = {}

    def _fake_create_agent(**kwargs):
        create_kwargs.update(kwargs)
        final_agent.ephemeral_system_prompt = str(
            kwargs.get("ephemeral_system_prompt") or "",
        )
        return final_agent

    monkeypatch.setattr(adapter, "_create_agent", _fake_create_agent)
    result, usage = await adapter._run_agent(
        user_message="compare two safe options",
        conversation_history=[
            {"role": "assistant", "content": "adjacent safe context"},
        ],
        session_id="gateway-test-session",
        gateway_session_key="gateway-test-channel",
        request_headers=headers,
        agent_ref=[None, False, None],
        true_moa_snapshot=snapshot,
    )

    assert advisor_calls == Counter(
        {
            KIMI_ADVISOR_SLOT.slot_id: 1,
            DEEPSEEK_ADVISOR_SLOT.slot_id: 1,
        },
    )
    assert len(final_agent.run_calls) == 1
    assert create_kwargs["strict_no_automatic_paid_retry"] is True
    assert "[MY STAND TRUE MOA - UNTRUSTED ADVISORY CONTEXT]" in str(
        final_agent.ephemeral_system_prompt,
    )
    assert usage["input_tokens"] == 41
    assert usage["output_tokens"] == 15
    assert usage["total_tokens"] == 56

    ledger = usage["true_moa"]
    assert result["_true_moa_usage"] == ledger
    assert ledger["schema"] == TRUE_MOA_USAGE_SCHEMA
    assert ledger["mode"] == TRUE_MOA_MODE
    assert ledger["modeEpoch"] == "31"
    assert ledger["presetId"] == TRUE_MOA_PRESET_ID
    assert ledger["presetRevision"] == TRUE_MOA_PRESET_REVISION
    assert ledger["status"] == "completed"
    assert [slot["slotId"] for slot in ledger["slots"]] == [
        KIMI_ADVISOR_SLOT.slot_id,
        DEEPSEEK_ADVISOR_SLOT.slot_id,
        FINAL_EXECUTOR_SLOT.slot_id,
    ]
    assert all(slot["status"] == "completed" for slot in ledger["slots"])
    assert all(slot["usageStatus"] == "reported" for slot in ledger["slots"])
    assert len({slot["callId"] for slot in ledger["slots"]}) == 3


@pytest.mark.asyncio
async def test_advisor_failure_closes_gateway_before_final_or_tools(
    monkeypatch,
):
    from xiaoban.trusted_runtime import true_moa_providers

    adapter = _adapter()
    headers = _mystand_headers(
        f"advisor-failure-{uuid.uuid4().hex}",
        epoch="32",
    )
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    rendezvous = threading.Barrier(2, timeout=1)
    advisor_calls: Counter[str] = Counter()
    lock = threading.Lock()

    def _fake_advisor(*, slot, **_kwargs):
        with lock:
            advisor_calls[slot.slot_id] += 1
        rendezvous.wait()
        if slot == KIMI_ADVISOR_SLOT:
            raise RuntimeError("sanitized fake provider failure")
        return StrictAdvisorResult(
            content="PRIVATE_LATE_ADVISOR",
            usage={"input_tokens": 5, "output_tokens": 2},
        )

    monkeypatch.setattr(
        true_moa_providers,
        "strict_advisor_call",
        _fake_advisor,
    )
    final_agent = _FakeFinalAgent()
    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **_kwargs: final_agent,
    )
    tool_start = MagicMock()
    tool_complete = MagicMock()

    result, usage = await adapter._run_agent(
        user_message="advisor failure must close the wave",
        conversation_history=[],
        session_id="advisor-failure-session",
        request_headers=headers,
        agent_ref=[None, False, None],
        tool_start_callback=tool_start,
        tool_complete_callback=tool_complete,
        true_moa_snapshot=snapshot,
    )

    assert advisor_calls == Counter(
        {
            KIMI_ADVISOR_SLOT.slot_id: 1,
            DEEPSEEK_ADVISOR_SLOT.slot_id: 1,
        },
    )
    assert final_agent.run_calls == []
    tool_start.assert_not_called()
    tool_complete.assert_not_called()
    assert result["failed"] is True
    assert "PRIVATE_LATE_ADVISOR" not in json.dumps(
        {"result": result, "usage": usage},
        ensure_ascii=False,
    )
    receipts = {
        item["slotId"]: item
        for item in usage["true_moa"]["slots"]
    }
    assert receipts[FINAL_EXECUTOR_SLOT.slot_id]["status"] == "not_started"


@pytest.mark.asyncio
async def test_normal_gateway_uses_only_final_agent_and_has_no_moa_usage(
    monkeypatch,
):
    from xiaoban.trusted_runtime import true_moa_providers

    adapter = _adapter()
    advisor_calls = 0

    def _provider_must_not_run(**_kwargs):
        nonlocal advisor_calls
        advisor_calls += 1
        raise AssertionError("normal mode fanned out to an advisor")

    monkeypatch.setattr(
        true_moa_providers,
        "strict_advisor_call",
        _provider_must_not_run,
    )
    final_agent = _FakeFinalAgent()
    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **_kwargs: final_agent,
    )

    result, usage = await adapter._run_agent(
        user_message="normal request",
        conversation_history=[],
        session_id="normal-direct-session",
        request_headers=_normal_direct_headers(),
        true_moa_snapshot=None,
    )

    assert result["final_response"] == "fake final synthesis"
    assert len(final_agent.run_calls) == 1
    assert advisor_calls == 0
    assert usage == {
        "input_tokens": 17,
        "output_tokens": 7,
        "total_tokens": 24,
    }
    assert "_true_moa_usage" not in result
    assert "true_moa" not in usage


def test_preexecuted_stop_after_index_never_dispatches_authorization(
    monkeypatch,
):
    from gateway.platforms.api_server import (
        CompletionStoppedError,
        _run_mystand_preexecuted_evidence,
    )
    from xiaoban.trusted_runtime.true_moa import TrueMoACancelController

    controller = TrueMoACancelController()
    calls: list[str] = []

    def _index(_args):
        calls.append("index")
        controller.cancel()
        return '{"ok":true,"items":[]}'

    def _authorization(_args):
        calls.append("authorization")
        return '{"ok":true}'

    monkeypatch.setattr(
        "tools.mystand_resource_index_tool.mystand_resource_index_tool_handler",
        _index,
    )
    monkeypatch.setattr(
        "tools.mystand_authorization_tool.mystand_authorization_tool_handler",
        _authorization,
    )

    with pytest.raises(CompletionStoppedError):
        _run_mystand_preexecuted_evidence(
            "mystand_authorization",
            user_message="读取 AUTH-ABC12345",
            system_prompt="",
            tool_start_callback=lambda *_args: None,
            tool_complete_callback=lambda *_args: None,
            terminal_controller=controller,
        )

    assert calls == ["index"]


def test_preexecuted_cancel_wins_before_trusted_preaction(monkeypatch):
    from gateway.platforms.api_server import (
        CompletionStoppedError,
        _run_mystand_preexecuted_evidence,
    )
    from xiaoban.trusted_runtime import TrustedIdentity, begin_turn
    from xiaoban.trusted_runtime.true_moa import TrueMoACancelController

    controller = TrueMoACancelController()
    original_gate = controller.try_begin_dispatch
    handler_calls: list[str] = []
    starts = MagicMock()
    completes = MagicMock()
    turn = begin_turn(
        channel="web",
        user_message="读取 AUTH-ABC12345",
        identity=TrustedIdentity(
            account_id="user-a",
            data_scope="mystand",
            source="server_session",
        ),
        request_id="req-preaction-race",
        message_id="msg-preaction-race",
    )

    def _cancel_before_gate(key):
        controller.cancel()
        return original_gate(key)

    monkeypatch.setattr(
        controller,
        "try_begin_dispatch",
        _cancel_before_gate,
    )
    monkeypatch.setattr(
        "tools.mystand_resource_index_tool.mystand_resource_index_tool_handler",
        lambda _args: handler_calls.append("index") or '{"ok":true}',
    )
    monkeypatch.setattr(
        "tools.mystand_authorization_tool.mystand_authorization_tool_handler",
        lambda _args: handler_calls.append("authorization") or '{"ok":true}',
    )

    with pytest.raises(CompletionStoppedError):
        _run_mystand_preexecuted_evidence(
            "mystand_authorization",
            user_message="读取 AUTH-ABC12345",
            system_prompt="",
            tool_start_callback=starts,
            tool_complete_callback=completes,
            trusted_turn=turn,
            terminal_controller=controller,
        )

    assert handler_calls == []
    assert turn.action_calls == []
    assert turn.action_results == []
    starts.assert_not_called()
    completes.assert_not_called()


@pytest.mark.asyncio
async def test_stop_tombstone_wins_completion_commit_without_text_leak():
    from gateway.platforms.api_server import _IdempotencyCache

    cache = _IdempotencyCache(max_items=8, ttl_seconds=30)
    key = "scope:terminal-race"
    agent_ref = [None, False, None]
    ledger = {
        "schema": TRUE_MOA_USAGE_SCHEMA,
        "status": "completed",
        "slots": [
            {
                "slotId": FINAL_EXECUTOR_SLOT.slot_id,
                "role": "final_executor",
                "status": "completed",
                "inputTokens": 7,
                "outputTokens": 3,
                "totalTokens": 10,
                "usageStatus": "reported",
                "costUsd": 0.02,
            },
        ],
    }

    async def _compute():
        assert cache.stop(key) is True
        return (
            {
                "final_response": "PRIVATE_LATE_RESULT",
                "messages": [
                    {"role": "assistant", "content": "PRIVATE_LATE_TRANSCRIPT"},
                ],
                "completed": True,
                "_true_moa_usage": ledger,
            },
            {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "true_moa": ledger,
            },
        )

    result, usage = await cache.get_or_set(
        key,
        "fingerprint",
        _compute,
        agent_ref=agent_ref,
    )
    serialized = json.dumps(
        {"result": result, "usage": usage},
        ensure_ascii=False,
    )

    assert "PRIVATE_LATE" not in serialized
    assert result["final_response"] == ""
    assert result["messages"] == []
    assert result["interrupted"] is True
    assert usage["total_tokens"] == 10
    assert usage["true_moa"]["status"] == "cancelled"
    final_slot = usage["true_moa"]["slots"][0]
    assert final_slot["status"] == "cancelled"
    assert final_slot["errorCategory"] == "terminal_fence_after_stop"
    assert final_slot["costUsd"] == 0.02
    await asyncio.sleep(0)
    state, cached = cache.result_state(key)
    assert state == "stopped"
    assert cached == (result, usage)
    assert "PRIVATE_LATE" not in json.dumps(cached, ensure_ascii=False)
    assert cached[1]["true_moa"]["slots"][0]["costUsd"] == 0.02


@pytest.mark.asyncio
async def test_stopped_completion_usage_endpoint_recovers_actual_receipt(
    monkeypatch,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"usage-recovery-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="40")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    cache = api_server._IdempotencyCache(max_items=8, ttl_seconds=30)
    ledger = {
        "schema": TRUE_MOA_USAGE_SCHEMA,
        "status": "completed",
        "slots": [
            {
                "slotId": FINAL_EXECUTOR_SLOT.slot_id,
                "role": "final_executor",
                "status": "completed",
                "inputTokens": 11,
                "outputTokens": 5,
                "totalTokens": 16,
                "usageStatus": "reported",
                "costUsd": 0.04,
            },
        ],
    }

    async def _compute():
        assert cache.stop(scoped_key) is True
        return (
            {
                "final_response": "PRIVATE_STOPPED_TEXT",
                "messages": [
                    {"role": "assistant", "content": "PRIVATE_STOPPED_TEXT"}
                ],
                "_true_moa_usage": ledger,
            },
            {
                "input_tokens": 11,
                "output_tokens": 5,
                "total_tokens": 16,
                "true_moa": ledger,
            },
        )

    await cache.get_or_set(
        scoped_key,
        "fingerprint",
        _compute,
        agent_ref=[None, False, None],
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert payload["status"] == "cancelled"
    assert payload["final"] is True
    assert payload["usage"]["slots"][0]["totalTokens"] == 16
    assert payload["usage"]["slots"][0]["costUsd"] == 0.04
    assert "PRIVATE_STOPPED_TEXT" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_final_commit_wins_later_stop_without_rewriting_completion():
    from gateway.platforms.api_server import _IdempotencyCache
    from xiaoban.trusted_runtime.true_moa import TrueMoACancelController

    cache = _IdempotencyCache(max_items=8, ttl_seconds=30)
    key = "scope:completion-wins"
    controller = TrueMoACancelController()
    agent_ref = [MagicMock(), False, controller]

    async def _compute():
        assert controller.try_commit_final("final-response:test") is True
        assert cache.stop(key) is False
        return (
            {
                "final_response": "committed response",
                "messages": [],
                "completed": True,
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    result, usage = await cache.get_or_set(
        key,
        "fingerprint",
        _compute,
        agent_ref=agent_ref,
    )
    await asyncio.sleep(0)

    assert result["final_response"] == "committed response"
    assert usage["total_tokens"] == 2
    assert controller.state == "completed"
    state, cached = cache.result_state(key)
    assert state == "completed"
    assert cached[0]["final_response"] == "committed response"


@pytest.mark.asyncio
async def test_final_route_preflight_fails_before_any_advisor_call(monkeypatch):
    from xiaoban.trusted_runtime import true_moa_providers

    adapter = _adapter()
    headers = _mystand_headers(f"preflight-{uuid.uuid4().hex}", epoch="41")
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    provider_calls = 0

    def _provider_must_not_run(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("route drift spent advisor tokens")

    bad_agent = _FakeFinalAgent()
    bad_agent.provider = "openai"
    monkeypatch.setattr(
        true_moa_providers,
        "strict_advisor_call",
        _provider_must_not_run,
    )
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: bad_agent)

    result, usage = await adapter._run_agent(
        user_message="must preflight",
        conversation_history=[],
        session_id="preflight-session",
        request_headers=headers,
        agent_ref=[None, False, None],
        true_moa_snapshot=snapshot,
    )

    assert provider_calls == 0
    assert result["failed"] is True
    receipts = {
        item["slotId"]: item
        for item in usage["true_moa"]["slots"]
    }
    assert receipts[KIMI_ADVISOR_SLOT.slot_id]["status"] == "not_started"
    assert receipts[DEEPSEEK_ADVISOR_SLOT.slot_id]["status"] == "not_started"
    assert receipts[FINAL_EXECUTOR_SLOT.slot_id]["status"] == "failed"


@pytest.mark.asyncio
async def test_post_advisor_setup_failure_returns_complete_partial_ledger(
    monkeypatch,
):
    from xiaoban.trusted_runtime import true_moa_providers

    adapter = _adapter()
    headers = _mystand_headers(f"setup-fail-{uuid.uuid4().hex}", epoch="42")
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )

    def _fake_advisor(*, slot, **_kwargs):
        return StrictAdvisorResult(
            content=f"advice from {slot.slot_id}",
            usage={"input_tokens": 6, "output_tokens": 2},
        )

    monkeypatch.setattr(
        true_moa_providers,
        "strict_advisor_call",
        _fake_advisor,
    )
    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **_kwargs: _FakeFinalAgent(),
    )
    monkeypatch.setattr(
        adapter,
        "_bind_api_server_session",
        MagicMock(side_effect=RuntimeError("local setup failed")),
    )

    result, usage = await adapter._run_agent(
        user_message="preserve advisor receipts",
        conversation_history=[],
        session_id="setup-failure-session",
        request_headers=headers,
        agent_ref=[None, False, None],
        true_moa_snapshot=snapshot,
    )

    assert result["failed"] is True
    receipts = {
        item["slotId"]: item
        for item in usage["true_moa"]["slots"]
    }
    for slot in (KIMI_ADVISOR_SLOT, DEEPSEEK_ADVISOR_SLOT):
        assert receipts[slot.slot_id]["status"] == "completed"
        assert receipts[slot.slot_id]["usageStatus"] == "reported"
        assert receipts[slot.slot_id]["totalTokens"] == 8
    assert receipts[FINAL_EXECUTOR_SLOT.slot_id]["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected_reason", "expected_code"),
    [
        (
            {
                "final_response": "",
                "completed": False,
                "failed": True,
            },
            "error",
            "agent_incomplete",
        ),
        (
            {
                "final_response": "partial text must not escape",
                "completed": False,
                "partial": True,
                "failed": True,
            },
            "length",
            "output_truncated",
        ),
        (
            {
                "final_response": "late stopped text must not escape",
                "completed": False,
                "failed": True,
                "interrupted": True,
            },
            "error",
            "completion_stopped",
        ),
    ],
)
async def test_sse_failed_true_moa_never_emits_success_stop(
    result,
    expected_reason,
    expected_code,
):
    from aiohttp import web

    adapter = _adapter()
    stream_q: queue.Queue = queue.Queue()
    stream_q.put(None)
    usage = {
        "input_tokens": 8,
        "output_tokens": 2,
        "total_tokens": 10,
        "true_moa": {
            "schema": TRUE_MOA_USAGE_SCHEMA,
            "status": "failed",
            "slots": [],
        },
    }

    async def _finished():
        return result, usage

    task = asyncio.create_task(_finished())
    await asyncio.sleep(0)
    response = AsyncMock(spec=web.StreamResponse)
    response.prepare = AsyncMock()
    response.write = AsyncMock()
    request = MagicMock()
    request.headers = {}

    with patch(
        "gateway.platforms.api_server.web.StreamResponse",
        return_value=response,
    ):
        await adapter._write_sse_chat_completion(
            request,
            "cmpl-failed-moa",
            "xiaoban-agent",
            1,
            stream_q,
            task,
        )

    wire = b"".join(
        call.args[0] for call in response.write.await_args_list
    ).decode()
    finish_chunks = [
        json.loads(line[6:])
        for line in wire.splitlines()
        if line.startswith("data: {")
        and '"finish_reason"' in line
    ]
    terminal_chunks = [
        chunk
        for chunk in finish_chunks
        if chunk["choices"][0]["finish_reason"] is not None
    ]
    assert len(terminal_chunks) == 1
    finish = terminal_chunks[0]
    assert finish["choices"][0]["finish_reason"] == expected_reason
    assert '"finish_reason": "stop"' not in wire
    assert "event: xiaoban.error" in wire
    assert f'"code": "{expected_code}"' in wire
    assert "partial text must not escape" not in wire
    assert "late stopped text must not escape" not in wire
