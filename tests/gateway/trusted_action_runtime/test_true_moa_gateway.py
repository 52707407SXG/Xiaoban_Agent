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
import sqlite3
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
    TRUE_MOA_FINAL_CALL_LIMIT,
    TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS,
    TRUE_MOA_PRESET_ID,
    TRUE_MOA_PRESET_REVISION,
    TRUE_MOA_USAGE_SCHEMA,
    TrueMoAUsageLedger,
    validate_true_moa_headers,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTCOME_KEYS = {"test-v1": b"\x31" * 32}


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
    session_cached_input_tokens = 2
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
        ledger = getattr(self, "_true_moa_usage_ledger", None)
        if ledger is not None:
            call_id = ledger.start_final_call(
                f"fake-final-{len(self.run_calls)}",
            )
            ledger.finish_final_call(
                call_id,
                status="completed",
                usage={
                    "input_tokens": self.session_prompt_tokens,
                    "output_tokens": self.session_completion_tokens,
                    "total_tokens": self.session_total_tokens,
                    "cached_input_tokens": self.session_cached_input_tokens,
                },
                cost_usd=self.session_estimated_cost_usd,
                cost_status=self.session_cost_status,
                cost_source=self.session_cost_source,
            )
        return {
            "final_response": "fake final synthesis",
            "completed": True,
            "failed": False,
            "messages": [],
        }

    def interrupt(self, reason: str) -> None:
        self.interrupt_calls.append(reason)


def _completed_usage(
    snapshot,
    *,
    wave_id: str = "f" * 32,
) -> dict:
    ledger = TrueMoAUsageLedger(snapshot, wave_id=wave_id)
    for slot, input_tokens, output_tokens in (
        (KIMI_ADVISOR_SLOT, 2, 1),
        (DEEPSEEK_ADVISOR_SLOT, 3, 1),
    ):
        ledger.start_slot(slot)
        advisor_call_id = ledger.start_advisor_call(slot)
        slot_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cached_input_tokens": 0,
        }
        ledger.finish_advisor_call(
            advisor_call_id,
            status="completed",
            usage=slot_usage,
        )
        ledger.finish_slot(
            slot,
            status="completed",
            usage=slot_usage,
        )
    ledger.start_slot(FINAL_EXECUTOR_SLOT)
    final_call_id = ledger.start_final_call(f"fake-final-{wave_id[:8]}")
    final_usage = {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
        "cached_input_tokens": 0,
    }
    ledger.finish_final_call(
        final_call_id,
        status="completed",
        usage=final_usage,
    )
    ledger.finish_slot(
        FINAL_EXECUTOR_SLOT,
        status="completed",
        usage=final_usage,
    )
    ledger.set_wave_status("completed")
    return ledger.to_dict()


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
async def test_http_passes_one_frozen_true_moa_snapshot_into_run_agent(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    captured: dict[str, object] = {}
    cache = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(tmp_path / "snapshot-ledger.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)

    async def _fake_run_agent(**kwargs):
        captured.update(kwargs)
        usage_ledger = _completed_usage(
            kwargs["true_moa_snapshot"],
            wave_id="2" * 32,
        )
        return (
            {
                "final_response": "snapshot accepted",
                "completed": True,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": usage_ledger,
            },
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "true_moa": usage_ledger,
            },
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
    cache._durable.close()


@pytest.mark.asyncio
async def test_true_moa_missing_outcome_key_fails_before_any_agent_dispatch(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "missing-key.sqlite"),
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    run_agent = AsyncMock(
        side_effect=AssertionError("missing outcome key reached provider path"),
    )
    monkeypatch.setattr(adapter, "_run_agent", run_agent)
    headers = _mystand_headers(f"missing-key-{uuid.uuid4().hex}")

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "messages": [{"role": "user", "content": "must preflight"}],
            },
        )
        payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "true_moa_outcome_key_unavailable"
    run_agent.assert_not_awaited()
    cache._durable.close()


@pytest.mark.asyncio
async def test_completed_http_outcome_replays_after_restart_recovers_and_acks(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    path = tmp_path / "completed-outcome.sqlite"
    raw_key = f"completed-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="51")
    body = {
        "model": "xiaoban-agent",
        "messages": [{"role": "user", "content": "sealed completion"}],
    }
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    completed_usage = _completed_usage(
        snapshot,
        wave_id="5" * 32,
    )
    dispatches = 0

    async def _fake_run_agent(**_kwargs):
        nonlocal dispatches
        dispatches += 1
        return (
            {
                "final_response": "sealed exact answer",
                "completed": True,
                "failed": False,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            },
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "true_moa": completed_usage,
            },
        )

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
    cache = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions",
        adapter._handle_chat_completions,
    )
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )

    async with TestClient(TestServer(app)) as client:
        first = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=body,
        )
        first_payload = await first.json()
        assert first.status == 200
        assert (
            first_payload["choices"][0]["message"]["content"]
            == "sealed exact answer"
        )
        outcome_id = first_payload["xiaoban"]["outcome_id"]
        assert first_payload["xiaoban"]["output_digest"] == hashlib.sha256(
            b"sealed exact answer"
        ).hexdigest()
        assert dispatches == 1

        cache._durable.close()
        restarted = api_server._IdempotencyCache(
            durable_path=str(path),
            outcome_keys=_OUTCOME_KEYS,
        )
        monkeypatch.setattr(api_server, "_idem_cache", restarted)

        replay = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=body,
        )
        replay_payload = await replay.json()
        assert replay.status == 200
        assert (
            replay_payload["choices"][0]["message"]["content"]
            == "sealed exact answer"
        )
        assert replay_payload["xiaoban"]["outcome_id"] == outcome_id
        assert dispatches == 1

        recovered = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        recovered_payload = await recovered.json()
        assert recovered.status == 200
        assert recovered_payload["status"] == "completed"
        assert recovered_payload["settlementBlocked"] is False
        assert recovered_payload["outcomeStatus"] == "sealed"
        assert recovered_payload["outcomeId"] == outcome_id
        assert (
            recovered_payload["outcome"]["finalResponse"]
            == "sealed exact answer"
        )
        assert recovered_payload["usage"] == completed_usage
        assert dispatches == 1

        wrong_message_headers = {
            **headers,
            "X-Xiaoban-Message-Id": "different-message",
        }
        wrong_message = await client.post(
            "/v1/chat/completions/usage",
            headers=wrong_message_headers,
            json={"idempotency_key": raw_key},
        )
        assert wrong_message.status == 409
        assert (await wrong_message.json())["error"]["code"] == (
            "true_moa_outcome_binding_invalid"
        )

        wrong_fingerprint_headers = {
            **headers,
            "X-Xiaoban-Request-Fingerprint": hashlib.sha256(
                b"different-request"
            ).hexdigest(),
        }
        wrong_fingerprint = await client.post(
            "/v1/chat/completions/usage",
            headers=wrong_fingerprint_headers,
            json={"idempotency_key": raw_key},
        )
        assert wrong_fingerprint.status == 409

        foreign_headers = {
            **headers,
            "X-Xiaoban-User-Id": "foreign-user",
        }
        foreign = await client.post(
            "/v1/chat/completions/usage",
            headers=foreign_headers,
            json={"idempotency_key": raw_key},
        )
        assert foreign.status == 404

        wrong_attempt_headers = {
            **headers,
            "X-Xiaoban-Attempt": "2",
        }
        wrong_attempt = await client.post(
            "/v1/chat/completions/usage",
            headers=wrong_attempt_headers,
            json={"idempotency_key": raw_key},
        )
        assert wrong_attempt.status == 404

        acknowledged = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={
                "action": "ack",
                "idempotency_key": raw_key,
                "outcome_id": outcome_id,
            },
        )
        acknowledged_payload = await acknowledged.json()
        assert acknowledged.status == 200
        assert acknowledged_payload["status"] == "acknowledged"
        assert acknowledged_payload["outcomeStatus"] == "acked"
        assert acknowledged_payload["usage"] == completed_usage

        repeat_ack = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={
                "action": "ack",
                "idempotency_key": raw_key,
                "outcome_id": outcome_id,
            },
        )
        assert repeat_ack.status == 200
        assert (await repeat_ack.json())["status"] == (
            "already_acknowledged"
        )

        after_ack = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        after_ack_payload = await after_ack.json()
        assert after_ack.status == 200
        assert after_ack_payload["status"] == "acknowledged"
        assert after_ack_payload["outcomeStatus"] == "acked"
        assert "outcome" not in after_ack_payload
        assert after_ack_payload["usage"] == completed_usage

        with sqlite3.connect(path) as connection:
            cleared = connection.execute(
                """
                SELECT length(outcome_nonce), length(outcome_ciphertext),
                       length(usage_json)
                FROM true_moa_idempotency
                """
            ).fetchone()
        assert cleared[0] == 0
        assert cleared[1] == 0
        assert cleared[2] > 0
        restarted._durable.close()


@pytest.mark.asyncio
async def test_tampered_completed_outcome_fails_closed_without_redispatch(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    path = tmp_path / "tampered-http.sqlite"
    raw_key = f"tampered-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="52")
    body = {
        "model": "xiaoban-agent",
        "messages": [{"role": "user", "content": "tamper test"}],
    }
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    completed_usage = _completed_usage(
        snapshot,
        wave_id="6" * 32,
    )
    dispatches = 0

    async def _fake_run_agent(**_kwargs):
        nonlocal dispatches
        dispatches += 1
        return (
            {
                "final_response": "tamper protected answer",
                "completed": True,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            },
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "true_moa": completed_usage,
            },
        )

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
    cache = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions",
        adapter._handle_chat_completions,
    )
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        initial = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=body,
        )
        assert initial.status == 200
        assert dispatches == 1
        cache._durable.close()
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                UPDATE true_moa_idempotency
                SET outcome_ciphertext = zeroblob(32)
                """
            )
        restarted = api_server._IdempotencyCache(
            durable_path=str(path),
            outcome_keys=_OUTCOME_KEYS,
        )
        monkeypatch.setattr(api_server, "_idem_cache", restarted)

        replay = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=body,
        )
        replay_payload = await replay.json()
        assert replay.status == 409
        assert replay_payload["error"]["code"] == (
            "true_moa_outcome_binding_invalid"
        )
        assert dispatches == 1

        recovery = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        recovery_payload = await recovery.json()
        assert recovery.status == 409
        assert recovery_payload["error"]["code"] == (
            "true_moa_outcome_binding_invalid"
        )
        assert dispatches == 1
        restarted._durable.close()


@pytest.mark.asyncio
async def test_completed_sse_buffers_until_sealed_and_emits_outcome_receipt(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    delivery_id = "xbd_" + ("7" * 40)
    headers = _mystand_headers("stream-sealed", epoch="53")
    headers.pop("Idempotency-Key")
    headers["X-Xiaoban-Delivery-Id"] = delivery_id
    headers["X-Xiaoban-Delivery-Attempt"] = headers[
        "X-Xiaoban-Attempt"
    ]
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    completed_usage = _completed_usage(
        snapshot,
        wave_id="7" * 32,
    )
    dispatches = 0

    async def _fake_run_agent(**kwargs):
        nonlocal dispatches
        dispatches += 1
        callback = kwargs.get("stream_delta_callback")
        if callback is not None:
            callback("UNSEALED_PRIVATE_DELTA")
        return (
            {
                "final_response": "SSE sealed exact answer",
                "completed": True,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            },
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "true_moa": completed_usage,
            },
        )

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "sse-sealed.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions",
        adapter._handle_chat_completions,
    )
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "sealed SSE completion"}
                ],
            },
        )
        wire = await response.text()
        assert response.status == 200
        assert "UNSEALED_PRIVATE_DELTA" not in wire
        assert "SSE sealed exact answer" in wire
        assert "event: xiaoban.moa.outcome" in wire
        assert "event: xiaoban.moa.usage" in wire
        assert wire.index("event: xiaoban.moa.outcome") < wire.index(
            "event: xiaoban.moa.usage"
        )
        outcome_event = wire.split(
            "event: xiaoban.moa.outcome\ndata: ",
            1,
        )[1].split("\n\n", 1)[0]
        outcome_payload = json.loads(outcome_event)
        assert len(outcome_payload["outcomeId"]) == 64
        assert outcome_payload["outputDigest"] == hashlib.sha256(
            b"SSE sealed exact answer"
        ).hexdigest()
        assert dispatches == 1

        recovered = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": delivery_id},
        )
        recovered_payload = await recovered.json()
        assert recovered.status == 200
        assert (
            recovered_payload["outcome"]["finalResponse"]
            == "SSE sealed exact answer"
        )
        assert recovered_payload["outcomeId"] == outcome_payload["outcomeId"]
        assert dispatches == 1
    cache._durable.close()


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

    def _fake_strict_advisor(*, slot, tools, dispatch_callback, **_kwargs):
        assert tools == ()
        dispatch_callback()
        with calls_lock:
            advisor_calls[slot.slot_id] += 1
        if slot == KIMI_ADVISOR_SLOT:
            return StrictAdvisorResult(
                content="fake kimi advice",
                usage={
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "cached_input_tokens": 0,
                },
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
                "prompt_cache_hit_tokens": 4,
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
    assert final_agent.max_iterations == TRUE_MOA_FINAL_CALL_LIMIT
    assert final_agent.max_tokens == TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS
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
    assert [call["cachedInputTokens"] for call in ledger["calls"]] == [
        0,
        4,
        2,
    ]
    assert ledger["slots"][-1]["cachedInputTokens"] == 2


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

    def _fake_advisor(*, slot, dispatch_callback, **_kwargs):
        dispatch_callback()
        with lock:
            advisor_calls[slot.slot_id] += 1
        rendezvous.wait()
        if slot == KIMI_ADVISOR_SLOT:
            raise RuntimeError("sanitized fake provider failure")
        return StrictAdvisorResult(
            content="PRIVATE_LATE_ADVISOR",
            usage={
                "input_tokens": 5,
                "output_tokens": 2,
                "cached_input_tokens": 0,
            },
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
                "cachedInputTokens": 0,
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
async def test_stop_before_create_is_scope_bound_and_stops_without_dispatch(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    path = tmp_path / "stop-before-create.sqlite"
    raw_key = f"stop-before-create-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="39")
    target_key = adapter._scoped_idempotency_key(headers, raw_key)
    body = {
        "model": "xiaoban-agent",
        "messages": [{"role": "user", "content": "must never dispatch"}],
    }
    cache = api_server._IdempotencyCache(
        max_items=16,
        ttl_seconds=30,
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    run_agent = AsyncMock(
        side_effect=AssertionError("pre-stopped create reached provider path"),
    )
    monkeypatch.setattr(adapter, "_run_agent", run_agent)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions",
        adapter._handle_chat_completions,
    )
    app.router.add_post(
        "/v1/chat/completions/stop",
        adapter._handle_stop_idempotent_chat_completion,
    )
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )

    async with TestClient(TestServer(app)) as client:
        # A valid stop for another site, owner, attempt, or raw delivery key
        # gets its own pre-create fence and cannot mutate this target scope.
        foreign_stops = (
            ({**headers, "X-Xiaoban-Site-Id": "foreign-site"}, raw_key),
            ({**headers, "X-Xiaoban-User-Id": "foreign-user"}, raw_key),
            ({**headers, "X-Xiaoban-Attempt": "2"}, raw_key),
            (headers, f"{raw_key}-foreign"),
        )
        for foreign_headers, foreign_key in foreign_stops:
            foreign_scoped_key = adapter._scoped_idempotency_key(
                foreign_headers,
                foreign_key,
            )
            assert foreign_scoped_key != target_key
            foreign = await client.post(
                "/v1/chat/completions/stop",
                headers=foreign_headers,
                json={"idempotency_key": foreign_key},
            )
            assert foreign.status == 202
            assert cache.durable_record(target_key) is None

        stopped = await client.post(
            "/v1/chat/completions/stop",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        assert stopped.status == 202
        stopped_record = cache.durable_record(target_key)
        assert stopped_record["state"] == "stopped"
        assert stopped_record["fingerprint"] == ""
        assert stopped_record["usage"] is None

        create = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=body,
        )
        create_payload = await create.json()
        assert create.status == 409
        assert create_payload["error"]["code"] == "completion_stopped"
        run_agent.assert_not_awaited()

        usage_response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        usage_payload = await usage_response.json()
        assert usage_response.status == 200
        assert usage_payload == {
            "ok": True,
            "status": "stopped_before_start",
            "final": True,
            "usage": None,
            "outcomeStatus": "none",
        }

    cache._durable.close()


@pytest.mark.asyncio
async def test_create_restart_stop_fences_late_completion_and_keeps_usage(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server
    from xiaoban.trusted_runtime.true_moa_durable import (
        TrueMoAOutcomeBindingError,
    )

    adapter = _adapter()
    path = tmp_path / "restart-stop-late-completion.sqlite"
    raw_key = f"restart-stop-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="41")
    body = {
        "model": "xiaoban-agent",
        "messages": [{"role": "user", "content": "late completion"}],
    }
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = adapter._chat_idempotency_fingerprint(body, headers)
    outcome_binding = adapter._true_moa_outcome_binding(
        headers,
        snapshot=snapshot,
        delivery_id=raw_key,
    )

    ledger = TrueMoAUsageLedger(snapshot, wave_id="e" * 32)
    for slot, input_tokens, output_tokens in (
        (KIMI_ADVISOR_SLOT, 4, 2),
        (DEEPSEEK_ADVISOR_SLOT, 6, 3),
    ):
        ledger.start_slot(slot)
        call_id = ledger.start_advisor_call(slot)
        call_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cached_input_tokens": 0,
        }
        ledger.finish_advisor_call(
            call_id,
            status="completed",
            usage=call_usage,
        )
        ledger.finish_slot(
            slot,
            status="completed",
            usage=call_usage,
        )
    ledger.set_wave_status("advisors_completed")
    ledger.start_slot(FINAL_EXECUTOR_SLOT)
    final_call_id = ledger.start_final_call("restart-late-final")
    dispatched_usage = ledger.to_dict()

    final_usage = {
        "input_tokens": 13,
        "output_tokens": 5,
        "total_tokens": 18,
        "cached_input_tokens": 2,
    }
    ledger.finish_final_call(
        final_call_id,
        status="completed",
        usage=final_usage,
        cost_usd=0.05,
        cost_status="reported",
        cost_source="fake-late-provider",
    )
    ledger.finish_slot(
        FINAL_EXECUTOR_SLOT,
        status="completed",
        usage=final_usage,
        cost_usd=0.05,
        cost_status="reported",
        cost_source="fake-late-provider",
    )
    ledger.set_wave_status("completed")
    completed_usage = ledger.to_dict()

    original = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    compute_started = asyncio.Event()
    release_late_completion = asyncio.Event()
    late_text = "PRIVATE_LATE_RESTART_RESULT"

    async def _late_compute():
        original.persist_usage(
            scoped_key,
            fingerprint,
            dispatched_usage,
        )
        compute_started.set()
        await release_late_completion.wait()
        # Provider usage can arrive after the durable stop fence. It must
        # still merge for settlement, while the visible answer stays fenced.
        original.persist_usage(
            scoped_key,
            fingerprint,
            completed_usage,
        )
        return (
            {
                "final_response": late_text,
                "messages": [{"role": "assistant", "content": late_text}],
                "completed": True,
                "failed": False,
                "_mystand_request": True,
                "_mystand_egress_finalized": True,
                "_mystand_egress_output_digest": hashlib.sha256(
                    late_text.encode(),
                ).hexdigest(),
                "_true_moa_usage": completed_usage,
            },
            {
                "input_tokens": 23,
                "output_tokens": 10,
                "total_tokens": 33,
                "true_moa": completed_usage,
            },
        )

    late_task = asyncio.create_task(
        original.get_or_set(
            scoped_key,
            fingerprint,
            _late_compute,
            agent_ref=[None, False, None],
            durable=True,
            outcome_binding=outcome_binding,
        )
    )
    await compute_started.wait()
    before_restart = original.durable_record(scoped_key)
    assert before_restart["state"] == "running"
    assert before_restart["usage"]["calls"][-1]["status"] == "running"

    # Release the single-process durable lock just as a real old process
    # exits, then create the replacement process over the same SQLite file.
    original._durable.close()
    restarted = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", restarted)
    interrupted = restarted.durable_record(scoped_key)
    assert interrupted["state"] == "interrupted"

    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/stop",
        adapter._handle_stop_idempotent_chat_completion,
    )
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        stop_response = await client.post(
            "/v1/chat/completions/stop",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        assert stop_response.status == 202
        assert restarted.durable_record(scoped_key)["state"] == "stopped"

        pending_usage = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        pending_payload = await pending_usage.json()
        assert pending_usage.status == 200
        assert pending_payload["status"] == "settlement_blocked"
        assert pending_payload["terminalState"] == "stopped"
        assert pending_payload["settlementBlocked"] is True
        assert pending_payload["outcomeStatus"] == "none"
        assert pending_payload["usage"]["calls"][-1]["status"] == "running"
        assert pending_payload["usage"]["calls"][0]["totalTokens"] == 6

        release_late_completion.set()
        with pytest.raises(
            TrueMoAOutcomeBindingError,
            match="terminal fence",
        ):
            await late_task

        stopped_record = restarted.durable_record(scoped_key)
        assert stopped_record["state"] == "stopped"
        assert stopped_record["usage"] == completed_usage
        assert stopped_record["outcomeState"] == "none"

        recovered = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        recovered_payload = await recovered.json()
        assert recovered.status == 200
        assert recovered_payload["status"] == "cancelled"
        assert recovered_payload["terminalState"] == "stopped"
        assert recovered_payload["settlementBlocked"] is False
        assert recovered_payload["outcomeStatus"] == "none"
        assert recovered_payload["usage"] == completed_usage
        assert recovered_payload["usage"]["calls"][-1]["totalTokens"] == 18
        assert recovered_payload["usage"]["calls"][-1]["costUsd"] == 0.05
        assert "outcome" not in recovered_payload
        assert late_text not in json.dumps(
            recovered_payload,
            ensure_ascii=False,
        )

    assert late_text.encode() not in path.read_bytes()
    restarted._durable.close()


@pytest.mark.asyncio
async def test_stopped_completion_usage_endpoint_recovers_actual_receipt(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"usage-recovery-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="40")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    cache = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(tmp_path / "usage-ledger.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    usage_ledger = TrueMoAUsageLedger(snapshot, wave_id="a" * 32)
    for slot, input_tokens, output_tokens in (
        (KIMI_ADVISOR_SLOT, 2, 1),
        (DEEPSEEK_ADVISOR_SLOT, 3, 1),
    ):
        usage_ledger.start_slot(slot)
        advisor_call_id = usage_ledger.start_advisor_call(slot)
        usage_ledger.finish_advisor_call(
            advisor_call_id,
            status="completed",
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cached_input_tokens": 0,
            },
        )
        usage_ledger.finish_slot(
            slot,
            status="completed",
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cached_input_tokens": 0,
            },
        )
    usage_ledger.start_slot(FINAL_EXECUTOR_SLOT)
    final_call_id = usage_ledger.start_final_call("usage-recovery-call")
    usage_ledger.finish_final_call(
        final_call_id,
        status="completed",
        usage={
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
            "cached_input_tokens": 3,
        },
        cost_usd=0.04,
        cost_status="reported",
        cost_source="fake-final",
    )
    usage_ledger.finish_slot(
        FINAL_EXECUTOR_SLOT,
        status="completed",
        usage={
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
            "cached_input_tokens": 3,
        },
        cost_usd=0.04,
        cost_status="reported",
        cost_source="fake-final",
    )
    usage_ledger.set_wave_status("completed")
    ledger = usage_ledger.to_dict()

    async def _compute():
        assert cache.stop(scoped_key, durable=True) is True
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
        durable=True,
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
    recovered_final_call = payload["usage"]["calls"][-1]
    assert recovered_final_call["totalTokens"] == 16
    assert recovered_final_call["costUsd"] == 0.04
    assert "PRIVATE_STOPPED_TEXT" not in json.dumps(payload, ensure_ascii=False)
    cache._durable.close()


@pytest.mark.asyncio
async def test_usage_endpoint_blocks_unknown_cache_split(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"unknown-cache-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="43")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    ledger = TrueMoAUsageLedger(snapshot, wave_id="b" * 32)
    for slot, usage in (
        (
            KIMI_ADVISOR_SLOT,
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
                "cache_read_input_tokens": 0,
            },
        ),
        (
            DEEPSEEK_ADVISOR_SLOT,
            {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
            },
        ),
    ):
        ledger.start_slot(slot)
        call_id = ledger.start_advisor_call(slot)
        ledger.finish_advisor_call(call_id, status="completed", usage=usage)
        ledger.finish_slot(slot, status="completed", usage=usage)
    ledger.start_slot(FINAL_EXECUTOR_SLOT)
    final_usage = {
        "prompt_tokens": 13,
        "completion_tokens": 4,
        "total_tokens": 17,
        "prompt_cache_hit_tokens": 2,
    }
    final_call_id = ledger.start_final_call("unknown-cache-final")
    ledger.finish_final_call(
        final_call_id,
        status="completed",
        usage=final_usage,
    )
    ledger.finish_slot(
        FINAL_EXECUTOR_SLOT,
        status="completed",
        usage=final_usage,
    )
    ledger.set_wave_status("completed")
    usage = ledger.to_dict()
    assert [call["usageStatus"] for call in usage["calls"]] == [
        "reported",
        "partial",
        "reported",
    ]

    cache = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(tmp_path / "unknown-cache.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    outcome_binding = adapter._true_moa_outcome_binding(
        headers,
        snapshot=snapshot,
        delivery_id=raw_key,
    )

    async def _compute():
        result = {
            "final_response": "answer",
            "messages": [],
            "completed": True,
            "_mystand_request": True,
            "_true_moa_usage": usage,
        }
        api_server._finalize_mystand_egress_result(
            result,
            user_message="unknown cache split",
            conversation_history=[],
        )
        return (
            result,
            {
                "input_tokens": 26,
                "output_tokens": 9,
                "total_tokens": 35,
                "true_moa": usage,
            },
        )

    await cache.get_or_set(
        scoped_key,
        "fingerprint",
        _compute,
        agent_ref=[None, False, None],
        durable=True,
        outcome_binding=outcome_binding,
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
    assert payload["status"] == "settlement_blocked"
    assert payload["terminalState"] == "completed"
    assert payload["usage"]["calls"][1]["cachedInputTokens"] is None
    cache._durable.close()


@pytest.mark.asyncio
async def test_usage_endpoint_surfaces_completed_callback_crash_gap(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"outcome-crash-gap-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="44")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    usage = _completed_usage(snapshot, wave_id="c" * 32)
    fingerprint = "crash-gap-fingerprint"
    cache = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(tmp_path / "outcome-crash-gap.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert cache._durable.claim(
        scoped_key,
        fingerprint,
        kind="execution",
    ) == "missing"
    # This is the exact durable state left by a hard process exit between the
    # completed usage callback and the atomic completed-outcome seal.
    cache.persist_usage(scoped_key, fingerprint, usage)
    crash_record = cache.durable_record(scoped_key)
    assert crash_record["state"] == "running"
    assert crash_record["usage"]["status"] == "completed"
    assert crash_record["outcomeState"] == "none"

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
    assert payload["status"] == "settlement_blocked"
    assert payload["terminalState"] == "running"
    assert payload["final"] is True
    assert payload["usage"]["status"] == "completed"
    assert payload["outcomeStatus"] == "none"
    assert payload["settlementBlocked"] is True
    assert payload["errorCode"] == "true_moa_outcome_unavailable"
    assert "outcome" not in payload
    cache._durable.close()


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

    def _fake_advisor(*, slot, dispatch_callback, **_kwargs):
        dispatch_callback()
        return StrictAdvisorResult(
            content=f"advice from {slot.slot_id}",
            usage={
                "input_tokens": 6,
                "output_tokens": 2,
                "cached_input_tokens": 0,
            },
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
