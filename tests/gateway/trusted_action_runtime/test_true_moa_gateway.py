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
import time
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.mystand_egress_seal import (
    seal_mystand_egress_projection,
)
from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger
from xiaoban.trusted_runtime.paid_call_policy import (
    SIGNED_MYSTAND_AGENT_POLICY_REVISION,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
)
from xiaoban.trusted_runtime.protocol_contract import (
    TRUSTED_RUNTIME_CONTRACT_DIGEST,
    TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
    TRUSTED_RUNTIME_CONTRACT_REVISION,
    TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
)
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
        TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER:
            TRUSTED_RUNTIME_CONTRACT_REVISION,
        TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER:
            TRUSTED_RUNTIME_CONTRACT_DIGEST,
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


@pytest.mark.asyncio
async def test_mystand_contract_mismatch_stops_before_provider_dispatch():
    adapter = _adapter()
    headers = _mystand_headers("contract-mismatch")
    headers[TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER] = "0" * 64

    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_run_agent") as run_agent:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "xiaoban-agent",
                    "messages": [{"role": "user", "content": "不应调用模型"}],
                },
            )
            payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == (
        "xiaoban_trusted_runtime_contract_mismatch"
    )
    run_agent.assert_not_called()


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
        self.persisted_sessions: list[str] = []
        self.saved_trajectories: list[str] = []
        self.persistence_raise_in: set[str] = set()

    def run_conversation(self, **kwargs):
        self.run_calls.append(kwargs)
        ledger = getattr(self, "_true_moa_usage_ledger", None)
        if ledger is not None:
            call_id = ledger.start_final_call(
                f"fake-final-{len(self.run_calls)}",
            )
            ledger.mark_dispatched(call_id)
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

    def _drop_trailing_empty_response_scaffolding(self, _messages) -> None:
        if "drop_scaffolding" in self.persistence_raise_in:
            raise RuntimeError("fake scaffold cleanup failure")

    def _save_trajectory(self, messages, *_args) -> None:
        if "save_trajectory" in self.persistence_raise_in:
            raise RuntimeError("fake trajectory failure")
        self.saved_trajectories.append(json.dumps(messages, ensure_ascii=False))

    def _persist_session(self, messages, *_args) -> None:
        if "persist_session" in self.persistence_raise_in:
            raise RuntimeError("fake session failure")
        self.persisted_sessions.append(json.dumps(messages, ensure_ascii=False))


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
        ledger.mark_dispatched(advisor_call_id)
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
    ledger.mark_dispatched(final_call_id)
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


def _failed_advisor_timeout_ledger(
    snapshot,
    *,
    wave_id: str,
) -> tuple[TrueMoAUsageLedger, str]:
    ledger = TrueMoAUsageLedger(snapshot, wave_id=wave_id)
    ledger.set_wave_status("running")
    ledger.start_slot(KIMI_ADVISOR_SLOT)
    call_id = ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    ledger.mark_dispatched(call_id)
    ledger.finish_slot(
        KIMI_ADVISOR_SLOT,
        status="timed_out",
        error_category="advisor_timeout",
    )
    ledger.set_wave_status("failed")
    ledger.terminate_unfinished(
        status="cancelled",
        error_category="cascade_after_advisor_timeout",
        preserve_running_calls=True,
    )
    assert ledger.to_dict()["calls"][0]["status"] == "running"
    return ledger, call_id


def _sealed_mystand_result(payload: dict) -> dict:
    result = dict(payload)
    final_text = str(result.get("final_response") or "")
    result["_mystand_egress_finalized"] = True
    result["_mystand_egress_output_digest"] = hashlib.sha256(
        final_text.encode("utf-8"),
    ).hexdigest()
    seal_mystand_egress_projection(result)
    return result


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
            _sealed_mystand_result({
                "final_response": "snapshot accepted",
                "completed": True,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": usage_ledger,
            }),
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
            _sealed_mystand_result({
                "final_response": "sealed exact answer",
                "completed": True,
                "failed": False,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            }),
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
            _sealed_mystand_result({
                "final_response": "tamper protected answer",
                "completed": True,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            }),
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
            _sealed_mystand_result({
                "final_response": "SSE sealed exact answer",
                "completed": True,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            }),
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
@pytest.mark.parametrize("stream", [True, False])
async def test_true_moa_diagnostic_tool_mode_reaches_stream_and_nonstream_runner(
    monkeypatch,
    tmp_path,
    stream,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    delivery_id = "xbd_" + ("a1" if stream else "a2") * 20
    headers = _mystand_headers(
        f"diagnostic-{stream}",
        epoch="74" if stream else "75",
    )
    headers.update({
        "Idempotency-Key": delivery_id,
        "X-Xiaoban-Delivery-Id": delivery_id,
        "X-Xiaoban-Delivery-Attempt": headers["X-Xiaoban-Attempt"],
        "X-Xiaoban-Completion-Protocol": "dynamic-evidence-v2",
        "X-Xiaoban-Evidence-Required": "0",
        "X-Xiaoban-Business-Tool-Mode": "disabled",
        "X-Xiaoban-Invocation-Fingerprint": "a" * 64,
    })
    if stream:
        headers.pop("Idempotency-Key")
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    completed_usage = _completed_usage(
        snapshot,
        wave_id=("a" if stream else "b") * 32,
    )
    mock_run = AsyncMock(
        return_value=(
            _sealed_mystand_result({
                "final_response": "上一轮没有形成可用结果。",
                "completed": True,
                "failed": False,
                "messages": [],
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            }),
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "true_moa": completed_usage,
            },
        ),
    )
    monkeypatch.setattr(adapter, "_run_agent", mock_run)
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / f"moa-diagnostic-{stream}.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "stream": stream,
                "messages": [{
                    "role": "user",
                    "content": (
                        "刚才查 OUT-ABCDEFG 为什么失败？"
                        "不要索引，直接读库"
                    ),
                }],
            },
        )
        await response.read()
    cache._durable.close()

    assert response.status == 200
    mock_run.assert_awaited_once()
    runner_kwargs = mock_run.await_args.kwargs
    assert runner_kwargs["business_tools_disabled"] is True
    assert runner_kwargs["dynamic_evidence_required"] is False
    assert runner_kwargs["true_moa_snapshot"] == snapshot


@pytest.mark.asyncio
async def test_normal_signed_sse_has_distinct_usage_recovery_and_no_redispatch(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    delivery_id = "xbd_" + ("8" * 40)
    headers = _mystand_headers("normal-stream")
    headers.pop("Idempotency-Key")
    headers.pop(MODE_EPOCH_HEADER)
    headers.pop(MOA_PRESET_ID_HEADER)
    headers.pop(MOA_PRESET_REVISION_HEADER)
    headers[REASONING_MODE_HEADER] = "normal"
    headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
        SIGNED_MYSTAND_AGENT_POLICY_REVISION
    )
    headers["X-Xiaoban-Delivery-Id"] = delivery_id
    headers["X-Xiaoban-Delivery-Attempt"] = headers[
        "X-Xiaoban-Attempt"
    ]
    ledger = AgentCallUsageLedger(
        provider="fake-provider",
        model="fake-model",
        execution_id="8" * 32,
    )
    call_id = ledger.start_call()
    ledger.mark_dispatched(call_id)
    ledger.finish_call(
        call_id,
        status="completed",
        usage={
            "input_tokens": 9,
            "output_tokens": 3,
            "total_tokens": 12,
            "cached_input_tokens": 0,
        },
    )
    ledger.set_status("completed")
    completed_usage = ledger.to_dict()
    dispatches = 0

    async def _fake_run_agent(**_kwargs):
        nonlocal dispatches
        dispatches += 1
        return (
            {
                "final_response": "normal durable answer",
                "completed": True,
                "failed": False,
                "messages": [],
                "_mystand_request": True,
                "_agent_call_usage": completed_usage,
            },
            {
                "input_tokens": 9,
                "output_tokens": 3,
                "total_tokens": 12,
                "agent_calls": completed_usage,
            },
        )

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "normal-sse.sqlite"),
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
                    {"role": "user", "content": "normal signed completion"}
                ],
            },
        )
        wire = await response.text()
        assert response.status == 200
        assert "event: xiaoban.agent.usage" in wire
        assert "event: xiaoban.moa.usage" not in wire

        recovered = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": delivery_id},
        )
        recovered_payload = await recovered.json()
        assert recovered.status == 200
        assert recovered_payload["usage"] == completed_usage
        assert recovered_payload["settlementBlocked"] is False

        replay = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "normal signed completion"}
                ],
            },
        )
        await replay.text()
        assert dispatches == 1
    cache._durable.close()


@pytest.mark.parametrize("revision", [None, "stale-policy"])
@pytest.mark.asyncio
async def test_normal_signed_http_rejects_unbound_billing_policy_before_agent(
    monkeypatch,
    revision,
):
    adapter = _adapter()
    create_calls = 0
    headers = _mystand_headers(f"normal-policy-{revision or 'missing'}")
    headers.pop(MODE_EPOCH_HEADER)
    headers.pop(MOA_PRESET_ID_HEADER)
    headers.pop(MOA_PRESET_REVISION_HEADER)
    headers[REASONING_MODE_HEADER] = "normal"
    delivery_id = "xbd_" + ("6" * 40)
    headers["Idempotency-Key"] = delivery_id
    headers["X-Xiaoban-Delivery-Id"] = delivery_id
    headers["X-Xiaoban-Delivery-Attempt"] = headers[
        "X-Xiaoban-Attempt"
    ]
    if revision is not None:
        headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = revision

    def create_agent(**_kwargs):
        nonlocal create_calls
        create_calls += 1
        raise AssertionError("Agent construction crossed policy gate")

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "stream": False,
                "messages": [
                    {"role": "user", "content": "normal signed completion"}
                ],
            },
        )
        payload = await response.json()

    assert response.status == 502
    assert create_calls == 0
    assert payload["error"]["code"] == "agent_incomplete"
    assert "true_moa_usage" not in payload["error"]["xiaoban"]
    usage = payload["error"]["xiaoban"]["agent_call_usage"]
    assert usage["schema"] == "mystand.agent-call-usage.v1"
    assert usage["status"] == "failed"
    assert usage["calls"] == []


@pytest.mark.asyncio
async def test_normal_durable_http_requires_stable_delivery_before_agent(
    monkeypatch,
):
    adapter = _adapter()
    create_calls = 0
    headers = _mystand_headers("normal-policy-missing-delivery")
    headers.pop(MODE_EPOCH_HEADER)
    headers.pop(MOA_PRESET_ID_HEADER)
    headers.pop(MOA_PRESET_REVISION_HEADER)
    headers[REASONING_MODE_HEADER] = "normal"
    headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
        SIGNED_MYSTAND_AGENT_POLICY_REVISION
    )

    def create_agent(**_kwargs):
        nonlocal create_calls
        create_calls += 1
        raise AssertionError("Agent construction crossed delivery gate")

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "stream": False,
                "messages": [
                    {"role": "user", "content": "durable completion"}
                ],
            },
        )
        payload = await response.json()

    assert response.status == 400
    assert create_calls == 0
    assert payload["error"]["code"] == (
        "mystand_delivery_identity_required"
    )


def test_normal_delivery_id_is_bound_into_idempotency_fingerprint():
    delivery_id = "xbd_" + ("4" * 40)
    headers = _mystand_headers("normal-delivery-fingerprint")
    headers.pop(MODE_EPOCH_HEADER)
    headers.pop(MOA_PRESET_ID_HEADER)
    headers.pop(MOA_PRESET_REVISION_HEADER)
    headers[REASONING_MODE_HEADER] = "normal"
    headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
        SIGNED_MYSTAND_AGENT_POLICY_REVISION
    )
    headers["Idempotency-Key"] = delivery_id
    headers["X-Xiaoban-Delivery-Id"] = delivery_id
    headers["X-Xiaoban-Delivery-Attempt"] = headers[
        "X-Xiaoban-Attempt"
    ]
    body = {
        "model": "xiaoban-agent",
        "stream": False,
        "messages": [
            {"role": "user", "content": "normal signed completion"}
        ],
    }

    baseline = APIServerAdapter._chat_idempotency_fingerprint(body, headers)
    changed = APIServerAdapter._chat_idempotency_fingerprint(
        body,
        {
            **headers,
            "X-Xiaoban-Delivery-Id": "xbd_" + ("5" * 40),
        },
    )

    assert changed != baseline


@pytest.mark.parametrize(
    ("identity_drift", "expected_code"),
    [
        ("idempotency_key", "mystand_idempotency_identity_conflict"),
        ("delivery_attempt", "mystand_delivery_attempt_conflict"),
    ],
)
@pytest.mark.asyncio
async def test_normal_nonstream_rejects_conflicting_durable_identity_before_agent(
    monkeypatch,
    tmp_path,
    identity_drift,
    expected_code,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    delivery_id = "xbd_" + ("3" * 40)
    headers = _mystand_headers(f"normal-identity-{identity_drift}")
    headers.pop(MODE_EPOCH_HEADER)
    headers.pop(MOA_PRESET_ID_HEADER)
    headers.pop(MOA_PRESET_REVISION_HEADER)
    headers[REASONING_MODE_HEADER] = "normal"
    headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
        SIGNED_MYSTAND_AGENT_POLICY_REVISION
    )
    headers["Idempotency-Key"] = delivery_id
    headers["X-Xiaoban-Delivery-Id"] = delivery_id
    headers["X-Xiaoban-Delivery-Attempt"] = headers[
        "X-Xiaoban-Attempt"
    ]
    if identity_drift == "idempotency_key":
        headers["Idempotency-Key"] = "xbd_" + ("2" * 40)
    else:
        headers["X-Xiaoban-Delivery-Attempt"] = "2"
    dispatches = 0

    async def _fake_run_agent(**_kwargs):
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("Agent crossed durable identity gate")

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / f"{identity_drift}.sqlite"),
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "stream": False,
                "messages": [
                    {"role": "user", "content": "normal signed completion"}
                ],
            },
        )
        payload = await response.json()
    cache._durable.close()

    assert response.status == 400
    assert payload["error"]["code"] == expected_code
    assert dispatches == 0


@pytest.mark.asyncio
async def test_normal_nonstream_delivery_drift_cannot_replay_prior_outcome(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    first_delivery_id = "xbd_" + ("7" * 40)
    headers = _mystand_headers("normal-delivery-replay")
    headers.pop(MODE_EPOCH_HEADER)
    headers.pop(MOA_PRESET_ID_HEADER)
    headers.pop(MOA_PRESET_REVISION_HEADER)
    headers[REASONING_MODE_HEADER] = "normal"
    headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
        SIGNED_MYSTAND_AGENT_POLICY_REVISION
    )
    headers["Idempotency-Key"] = first_delivery_id
    headers["X-Xiaoban-Delivery-Id"] = first_delivery_id
    headers["X-Xiaoban-Delivery-Attempt"] = headers[
        "X-Xiaoban-Attempt"
    ]
    ledger = AgentCallUsageLedger(
        provider="deepseek",
        model="deepseek-v4-pro",
        execution_id="7" * 32,
    )
    call_id = ledger.start_call()
    ledger.mark_dispatched(call_id)
    ledger.finish_call(
        call_id,
        status="completed",
        usage={
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "cached_input_tokens": 0,
        },
    )
    ledger.set_status("completed")
    completed_usage = ledger.to_dict()
    dispatches = 0

    async def _fake_run_agent(**_kwargs):
        nonlocal dispatches
        dispatches += 1
        return (
            {
                "final_response": f"answer-for:{first_delivery_id}",
                "completed": True,
                "failed": False,
                "messages": [],
                "_mystand_request": True,
                "_agent_call_usage": completed_usage,
            },
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
                "agent_calls": completed_usage,
            },
        )

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "normal-delivery-replay.sqlite"),
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    body = {
        "model": "xiaoban-agent",
        "stream": False,
        "messages": [
            {"role": "user", "content": "normal signed completion"}
        ],
    }
    async with TestClient(TestServer(_app(adapter))) as client:
        initial = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=body,
        )
        initial_payload = await initial.json()
        drifted = await client.post(
            "/v1/chat/completions",
            headers={
                **headers,
                "X-Xiaoban-Delivery-Id": "xbd_" + ("8" * 40),
            },
            json=body,
        )
        drifted_payload = await drifted.json()
    cache._durable.close()

    assert initial.status == 200
    assert first_delivery_id in initial_payload["choices"][0]["message"]["content"]
    assert drifted.status == 400
    assert drifted_payload["error"]["code"] == (
        "mystand_idempotency_identity_conflict"
    )
    assert dispatches == 1


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
async def test_normal_signed_legacy_direct_uses_old_path_without_paid_ledger(
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
    create_kwargs: dict[str, object] = {}

    def create_agent(**kwargs):
        create_kwargs.update(kwargs)
        return final_agent

    monkeypatch.setattr(adapter, "_create_agent", create_agent)

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
    assert usage["input_tokens"] == 17
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 24
    assert create_kwargs["strict_no_automatic_paid_retry"] is False
    assert "_true_moa_usage" not in result
    assert "_agent_call_usage" not in result
    assert "agent_calls" not in usage
    assert "true_moa" not in usage


@pytest.mark.asyncio
async def test_normal_signed_legacy_http_needs_no_durable_headers(
    monkeypatch,
):
    adapter = _adapter()
    final_agent = _FakeFinalAgent()
    create_kwargs: dict[str, object] = {}
    headers = _mystand_headers("normal-legacy-http")
    headers.pop("Idempotency-Key")
    headers.pop(MODE_EPOCH_HEADER)
    headers.pop(MOA_PRESET_ID_HEADER)
    headers.pop(MOA_PRESET_REVISION_HEADER)
    headers[REASONING_MODE_HEADER] = "normal"

    def create_agent(**kwargs):
        create_kwargs.update(kwargs)
        return final_agent

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "xiaoban-agent",
                "stream": False,
                "messages": [
                    {"role": "user", "content": "legacy normal completion"}
                ],
            },
        )
        payload = await response.json()

    assert response.status == 200
    assert len(final_agent.run_calls) == 1
    assert create_kwargs["strict_no_automatic_paid_retry"] is False
    assert "agent_call_usage" not in payload.get("xiaoban", {})


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
        "calls": [
            {
                "callId": "advisor-running",
                "slotId": KIMI_ADVISOR_SLOT.slot_id,
                "role": "advisor",
                "status": "running",
                "usageStatus": "unavailable",
            },
            {
                "callId": "final-running",
                "slotId": FINAL_EXECUTOR_SLOT.slot_id,
                "role": "final_executor",
                "status": "running",
                "usageStatus": "unavailable",
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
    calls = {
        item["callId"]: item
        for item in usage["true_moa"]["calls"]
    }
    assert calls["advisor-running"]["status"] == "running"
    assert calls["final-running"]["status"] == "cancelled"
    assert (
        calls["final-running"]["errorCategory"]
        == "terminal_fence_after_stop"
    )
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
async def test_same_process_stopped_usage_drain_is_not_terminalized_as_orphan(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"same-process-stop-drain-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="45")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    cache = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(tmp_path / "same-process-stop-drain.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )

    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    ledger = TrueMoAUsageLedger(snapshot, wave_id="d" * 32)
    running_calls = {}
    for slot in (KIMI_ADVISOR_SLOT, DEEPSEEK_ADVISOR_SLOT):
        ledger.start_slot(slot)
        running_calls[slot] = ledger.start_advisor_call(slot)
        ledger.mark_dispatched(running_calls[slot])
    compute_started = asyncio.Event()
    release_outer_request = asyncio.Event()

    async def _compute():
        cache.persist_usage(scoped_key, fingerprint, ledger.to_dict())
        compute_started.set()
        await release_outer_request.wait()
        ledger.set_wave_status("cancelled", notify=False)
        ledger.terminate_unfinished(
            status="cancelled",
            error_category="cancelled",
            preserve_running_calls=True,
            notify=False,
        )
        pending_usage = ledger.to_dict()
        cache.persist_usage(scoped_key, fingerprint, pending_usage)
        return (
            {
                "final_response": "",
                "messages": [],
                "completed": False,
                "failed": True,
                "interrupted": True,
                "_true_moa_usage": pending_usage,
            },
            {"true_moa": pending_usage},
        )

    outer_task = asyncio.create_task(
        cache.get_or_set(
            scoped_key,
            fingerprint,
            _compute,
            agent_ref=[None, False, None],
            durable=True,
        )
    )
    await compute_started.wait()
    assert cache.stop(scoped_key, durable=True) is True
    release_outer_request.set()
    await outer_task
    await asyncio.sleep(0)

    # This reproduces the real failure window: the outer asyncio request owner
    # is already gone, but the same process still owns both provider workers
    # and must retain their exact usage receipts.
    assert not cache._inflight
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        pending_response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        pending_payload = await pending_response.json()
        assert pending_response.status == 202
        assert pending_payload["status"] == "stopped_draining"
        assert pending_payload["final"] is False
        assert pending_payload["terminalState"] == "stopped"
        assert pending_payload["settlementBlocked"] is True
        assert all(
            call["status"] == "running"
            for call in pending_payload["usage"]["calls"]
        )

        def _finish_receipt(slot, index):
            usage = {
                "input_tokens": 10 + index,
                "output_tokens": 2 + index,
                "total_tokens": 12 + (index * 2),
                "cached_input_tokens": 0,
            }
            ledger.finish_advisor_call(
                running_calls[slot],
                status="completed",
                usage=usage,
                error_category="completed_after_stop",
                notify=False,
            )
            ledger.finish_slot(
                slot,
                status="cancelled",
                usage=usage,
                error_category="late_result_after_terminal",
                notify=False,
            )

        _finish_receipt(KIMI_ADVISOR_SLOT, 1)
        cache.persist_usage(scoped_key, fingerprint, ledger.to_dict())
        partial_response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        partial_payload = await partial_response.json()
        assert partial_response.status == 202
        assert partial_payload["final"] is False
        partial_calls = {
            call["slotId"]: call
            for call in partial_payload["usage"]["calls"]
        }
        assert (
            partial_calls[KIMI_ADVISOR_SLOT.slot_id]["usageStatus"]
            == "reported"
        )
        assert (
            partial_calls[DEEPSEEK_ADVISOR_SLOT.slot_id]["status"]
            == "running"
        )

        _finish_receipt(DEEPSEEK_ADVISOR_SLOT, 2)
        cache.persist_usage(scoped_key, fingerprint, ledger.to_dict())

        settled_response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        settled_payload = await settled_response.json()
        assert settled_response.status == 200
        assert settled_payload["status"] == "cancelled"
        assert settled_payload["settlementBlocked"] is False
        assert all(
            call["status"] == "completed"
            and call["usageStatus"] == "reported"
            for call in settled_payload["usage"]["calls"]
        )

    cache._durable.close()


@pytest.mark.asyncio
async def test_same_process_interrupted_usage_drain_is_not_recovered_as_orphan(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"same-process-interrupted-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="61")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "same-process-interrupted.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert cache._durable.claim(
        scoped_key,
        fingerprint,
        kind="execution",
    ) == "missing"
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    ledger = TrueMoAUsageLedger(snapshot, wave_id="5" * 32)
    ledger.set_wave_status("running")
    ledger.start_slot(KIMI_ADVISOR_SLOT)
    call_id = ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    ledger.mark_dispatched(call_id)
    cache.persist_usage(scoped_key, fingerprint, ledger.to_dict())
    cache._durable.set_state(scoped_key, state="interrupted")
    assert cache.has_active_usage_drain(scoped_key) is True

    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        payload = await response.json()

    assert response.status == 202
    assert payload["status"] == "interrupted_draining"
    assert payload["final"] is False
    assert payload["settlementBlocked"] is True
    record = cache.durable_record(scoped_key)
    assert record["state"] == "interrupted"
    assert record["usage"]["calls"][0]["status"] == "running"
    assert record["usage"]["calls"][0]["endedAtMs"] is None
    cache._durable.close()


@pytest.mark.asyncio
async def test_same_process_interrupted_zero_call_inflight_stays_nonterminal(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"same-process-zero-call-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="64")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "same-process-zero-call.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    ledger = TrueMoAUsageLedger(snapshot, wave_id="8" * 32)
    ledger.set_wave_status("running")
    compute_started = asyncio.Event()
    release_compute = asyncio.Event()

    async def compute():
        cache.persist_usage(
            scoped_key,
            fingerprint,
            ledger.to_dict(),
        )
        compute_started.set()
        await release_compute.wait()
        return (
            {
                "final_response": "",
                "messages": [],
                "completed": False,
                "failed": True,
                "_true_moa_usage": ledger.to_dict(),
            },
            {"true_moa": ledger.to_dict()},
        )

    outer = asyncio.create_task(
        cache.get_or_set(
            scoped_key,
            fingerprint,
            compute,
            durable=True,
        )
    )
    await compute_started.wait()
    cache._durable.set_state(scoped_key, state="interrupted")
    assert cache.has_active_usage_drain(scoped_key) is False

    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        payload = await response.json()

    assert response.status == 202
    assert payload["status"] == "interrupted_draining"
    assert payload["final"] is False
    assert payload["settlementBlocked"] is True
    record = cache.durable_record(scoped_key)
    assert record["state"] == "interrupted"
    assert record["usage"]["status"] == "running"
    assert record["usage"]["calls"] == []

    release_compute.set()
    await outer
    cache._durable.close()


@pytest.mark.asyncio
async def test_restart_failed_advisor_timeout_terminalizes_orphaned_call(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"restart-failed-drain-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="65")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    path = tmp_path / "restart-failed-drain.sqlite"
    original = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert original._durable.claim(
        scoped_key,
        fingerprint,
        kind="execution",
    ) == "missing"
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    ledger, call_id = _failed_advisor_timeout_ledger(
        snapshot,
        wave_id="9" * 32,
    )
    original._durable.save_usage(
        scoped_key,
        fingerprint,
        ledger.to_dict(),
        state="failed",
    )
    original._durable.close()

    restarted = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    before_recovery = restarted.durable_record(scoped_key)
    assert before_recovery["state"] == "failed"
    assert before_recovery["usage"]["status"] == "failed"
    assert before_recovery["usage"]["calls"][0]["status"] == "running"
    monkeypatch.setattr(api_server, "_idem_cache", restarted)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        payload = await response.json()
        repeated = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        repeated_payload = await repeated.json()

    assert response.status == 200
    assert repeated.status == 200
    assert repeated_payload == payload
    assert payload["final"] is True
    assert payload["terminalState"] == "failed"
    assert payload["usage"]["status"] == "failed"
    assert payload["settlementBlocked"] is True
    call = payload["usage"]["calls"][0]
    assert call["status"] == "timed_out"
    assert call["usageStatus"] == "unavailable"
    assert call["endedAtMs"] is not None
    assert (
        call["errorCategory"]
        == "agent_restart_outcome_unknown"
    )
    assert all(
        call[field] is None
        for field in (
            "inputTokens",
            "outputTokens",
            "totalTokens",
            "cachedInputTokens",
        )
    )
    assert "outcome" not in payload
    assert "finalResponse" not in json.dumps(payload)
    fenced_ended_at = call["endedAtMs"]

    late_usage = {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
        "cached_input_tokens": 2,
    }
    ledger.finish_advisor_call(
        call_id,
        status="completed",
        usage=late_usage,
        error_category="late_provider_result",
        cost_usd=0.007,
        cost_status="reported",
        cost_source="fake-provider",
    )
    ledger.finish_slot(
        KIMI_ADVISOR_SLOT,
        status="timed_out",
        usage=late_usage,
        error_category="advisor_timeout",
        cost_usd=0.007,
        cost_status="reported",
        cost_source="fake-provider",
    )
    restarted.persist_usage(
        scoped_key,
        fingerprint,
        ledger.to_dict(),
    )
    late_record = restarted.durable_record(scoped_key)
    late_call = late_record["usage"]["calls"][0]
    assert late_record["state"] == "failed"
    assert late_call["status"] == "timed_out"
    assert late_call["endedAtMs"] == fenced_ended_at
    assert (
        late_call["errorCategory"]
        == "agent_restart_outcome_unknown"
    )
    assert late_call["usageStatus"] == "reported"
    assert late_call["inputTokens"] == 12
    assert late_call["outputTokens"] == 4
    assert late_call["totalTokens"] == 16
    assert late_call["cachedInputTokens"] == 2
    assert late_call["costUsd"] == 0.007
    restarted._durable.close()


@pytest.mark.asyncio
async def test_restart_retries_running_usage_after_old_lease_expires(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"restart-live-lease-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="67")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    path = tmp_path / "restart-live-lease.sqlite"
    original = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert original._durable.claim(
        scoped_key,
        fingerprint,
        kind="execution",
    ) == "missing"
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    ledger = TrueMoAUsageLedger(snapshot, wave_id="b" * 32)
    ledger.set_wave_status("running")
    ledger.start_slot(KIMI_ADVISOR_SLOT)
    call_id = ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    ledger.mark_dispatched(call_id)
    original.persist_usage(scoped_key, fingerprint, ledger.to_dict())
    lease_before_restart = original._durable.usage_drain_lease(scoped_key)
    assert lease_before_restart["leaseUntilMs"] > int(time.time() * 1000)

    # Model a hard process death: its timer cannot heartbeat or release, while
    # SQLite retains the still-live lease until the original expiry.
    with original._usage_drains_lock:
        timers = list(original._usage_drain_heartbeat_timers.values())
        original._usage_drain_heartbeat_timers.clear()
    for timer in timers:
        timer.cancel()
    original._durable.close()

    restarted = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    before_recovery = restarted.durable_record(scoped_key)
    assert before_recovery["state"] == "running"
    assert before_recovery["usage"]["calls"][0]["status"] == "running"
    monkeypatch.setattr(api_server, "_idem_cache", restarted)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )

    async with TestClient(TestServer(app)) as client:
        waiting = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        waiting_payload = await waiting.json()
        assert waiting.status == 202
        assert waiting_payload["status"] == "running_draining"
        assert waiting_payload["final"] is False

        with restarted._durable._lock, restarted._durable._connect() as connection:
            connection.execute(
                """
                UPDATE true_moa_usage_drain_leases
                SET lease_until_ms = 0
                """
            )
            connection.commit()

        recovered = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        recovered_payload = await recovered.json()
        repeated = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        repeated_payload = await repeated.json()

    assert recovered.status == 200
    assert repeated.status == 200
    assert repeated_payload == recovered_payload
    assert recovered_payload["final"] is True
    assert recovered_payload["terminalState"] == "failed"
    assert recovered_payload["settlementBlocked"] is True
    call = recovered_payload["usage"]["calls"][0]
    assert call["status"] == "timed_out"
    assert call["usageStatus"] == "unavailable"
    assert call["errorCategory"] == "agent_restart_outcome_unknown"
    lease_after_recovery = restarted._durable.usage_drain_lease(scoped_key)
    assert lease_after_recovery["generation"] == 2
    assert lease_after_recovery["leaseUntilMs"] == 0
    restarted._durable.close()


@pytest.mark.asyncio
async def test_same_process_failed_usage_drain_is_not_recovered_as_orphan(
    monkeypatch,
    tmp_path,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = f"same-process-failed-drain-{uuid.uuid4().hex}"
    headers = _mystand_headers(raw_key, epoch="66")
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / "same-process-failed-drain.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert cache._durable.claim(
        scoped_key,
        fingerprint,
        kind="execution",
    ) == "missing"
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    ledger, _call_id = _failed_advisor_timeout_ledger(
        snapshot,
        wave_id="a" * 32,
    )
    cache.persist_usage(scoped_key, fingerprint, ledger.to_dict())
    assert cache.has_active_usage_drain(scoped_key) is True
    assert cache.durable_record(scoped_key)["state"] == "failed"

    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        payload = await response.json()

    assert response.status == 202
    assert payload["status"] == "failed_draining"
    assert payload["final"] is False
    assert payload["settlementBlocked"] is True
    record = cache.durable_record(scoped_key)
    assert record["state"] == "failed"
    assert record["usage"]["status"] == "failed"
    assert record["usage"]["calls"][0]["status"] == "running"
    assert record["usage"]["calls"][0]["endedAtMs"] is None
    cache._durable.close()


@pytest.mark.parametrize("topology", ["normal", "true_moa"])
@pytest.mark.parametrize("dispatched", [False, True])
@pytest.mark.asyncio
async def test_restart_usage_recovery_terminalizes_orphan_without_stop(
    monkeypatch,
    tmp_path,
    topology,
    dispatched,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = (
        f"restart-orphan-{topology}-{int(dispatched)}-"
        f"{uuid.uuid4().hex}"
    )
    headers = _mystand_headers(raw_key, epoch="62")
    if topology == "normal":
        headers.pop(MODE_EPOCH_HEADER)
        headers.pop(MOA_PRESET_ID_HEADER)
        headers.pop(MOA_PRESET_REVISION_HEADER)
        headers[REASONING_MODE_HEADER] = "normal"
        headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
        )
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    path = tmp_path / f"restart-orphan-{topology}-{dispatched}.sqlite"
    original = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert original._durable.claim(
        scoped_key,
        fingerprint,
        kind="execution",
    ) == "missing"

    if topology == "normal":
        ledger = AgentCallUsageLedger(
            provider="deepseek",
            model="deepseek-v4-pro",
            execution_id="6" * 32,
        )
        call_id = ledger.start_call()
    else:
        snapshot = validate_true_moa_headers(
            headers,
            mystand_request=True,
            api_authenticated=True,
        )
        ledger = TrueMoAUsageLedger(snapshot, wave_id="6" * 32)
        ledger.set_wave_status("running")
        ledger.start_slot(KIMI_ADVISOR_SLOT)
        call_id = ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    if dispatched:
        ledger.mark_dispatched(call_id)
    original._durable.save_usage(
        scoped_key,
        fingerprint,
        ledger.to_dict(),
        state="running",
    )
    original._durable.close()

    restarted = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    before_recovery = restarted.durable_record(scoped_key)
    assert before_recovery["state"] == "interrupted"
    assert before_recovery["usage"]["calls"][0]["status"] == (
        "running" if dispatched else "reserved"
    )
    monkeypatch.setattr(api_server, "_idem_cache", restarted)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        payload = await response.json()
        repeated = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        repeated_payload = await repeated.json()

    assert response.status == 200
    assert repeated.status == 200
    assert repeated_payload == payload
    assert payload["final"] is True
    assert payload["terminalState"] == "failed"
    assert payload["usage"]["status"] == "failed"
    assert payload["settlementBlocked"] is dispatched
    call = payload["usage"]["calls"][0]
    assert call["status"] == (
        "timed_out" if dispatched else "not_dispatched"
    )
    assert call["usageStatus"] == "unavailable"
    assert call["endedAtMs"] is not None
    assert call["errorCategory"] == (
        "agent_restart_outcome_unknown"
        if dispatched
        else "provider_dispatch_fence_closed"
    )
    assert all(
        call[field] is None
        for field in (
            "inputTokens",
            "outputTokens",
            "totalTokens",
            "cachedInputTokens",
        )
    )
    assert "costUsd" not in call
    if topology == "true_moa":
        slot = payload["usage"]["slots"][0]
        assert slot["status"] == "failed"
        assert slot["endedAtMs"] is not None
        assert (
            slot["errorCategory"]
            == "agent_restart_outcome_unknown"
        )
    durable = restarted.durable_record(scoped_key)
    assert durable["state"] == "failed"
    assert durable["usage"] == payload["usage"]
    fenced_usage = durable["usage"]
    fenced_call = fenced_usage["calls"][0]
    fenced_ended_at = fenced_call["endedAtMs"]

    if not dispatched:
        ledger.mark_dispatched(call_id)
    late_usage = {
        "input_tokens": 9,
        "output_tokens": 3,
        "total_tokens": 12,
        "cached_input_tokens": 1,
    }
    if topology == "normal":
        ledger.finish_call(
            call_id,
            status="completed",
            usage=late_usage,
            error_category="late_provider_result",
            cost_usd=0.006,
            cost_status="reported",
            cost_source="fake-provider",
        )
    else:
        ledger.finish_advisor_call(
            call_id,
            status="completed",
            usage=late_usage,
            error_category="late_provider_result",
            cost_usd=0.006,
            cost_status="reported",
            cost_source="fake-provider",
        )
        ledger.finish_slot(
            KIMI_ADVISOR_SLOT,
            status="completed",
            usage=late_usage,
            error_category="late_provider_result",
            cost_usd=0.006,
            cost_status="reported",
            cost_source="fake-provider",
        )

    if not dispatched:
        with pytest.raises(ValueError):
            restarted.persist_usage(
                scoped_key,
                fingerprint,
                ledger.to_dict(),
            )
        assert restarted.durable_record(scoped_key)["usage"] == fenced_usage
    else:
        restarted.persist_usage(
            scoped_key,
            fingerprint,
            ledger.to_dict(),
        )
        late_record = restarted.durable_record(scoped_key)
        assert late_record["state"] == "failed"
        late_call = late_record["usage"]["calls"][0]
        assert late_call["status"] == "timed_out"
        assert late_call["endedAtMs"] == fenced_ended_at
        assert (
            late_call["errorCategory"]
            == "agent_restart_outcome_unknown"
        )
        assert late_call["usageStatus"] == "reported"
        assert late_call["inputTokens"] == 9
        assert late_call["outputTokens"] == 3
        assert late_call["totalTokens"] == 12
        assert late_call["cachedInputTokens"] == 1
        assert late_call["costUsd"] == 0.006
        assert late_call["costStatus"] == "reported"
        assert late_call["costSource"] == "fake-provider"
        if topology == "true_moa":
            late_slot = late_record["usage"]["slots"][0]
            assert late_slot["status"] == "failed"
            assert (
                late_slot["errorCategory"]
                == "agent_restart_outcome_unknown"
            )
            assert late_slot["usageStatus"] == "reported"
    restarted._durable.close()


@pytest.mark.parametrize("topology", ["normal", "true_moa"])
@pytest.mark.parametrize("phase", ["before_first_call", "between_calls"])
@pytest.mark.asyncio
async def test_restart_usage_recovery_fails_nonterminal_ledger_without_active_call(
    monkeypatch,
    tmp_path,
    topology,
    phase,
):
    from gateway.platforms import api_server

    adapter = _adapter()
    raw_key = (
        f"restart-inactive-gap-{topology}-{phase}-"
        f"{uuid.uuid4().hex}"
    )
    headers = _mystand_headers(raw_key, epoch="63")
    if topology == "normal":
        headers.pop(MODE_EPOCH_HEADER)
        headers.pop(MOA_PRESET_ID_HEADER)
        headers.pop(MOA_PRESET_REVISION_HEADER)
        headers[REASONING_MODE_HEADER] = "normal"
        headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
        )
    scoped_key = adapter._scoped_idempotency_key(headers, raw_key)
    fingerprint = hashlib.sha256(raw_key.encode()).hexdigest()
    path = tmp_path / f"inactive-gap-{topology}-{phase}.sqlite"
    original = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert original._durable.claim(
        scoped_key,
        fingerprint,
        kind="execution",
    ) == "missing"

    if topology == "normal":
        ledger = AgentCallUsageLedger(
            provider="deepseek",
            model="deepseek-v4-pro",
            execution_id="7" * 32,
        )
        if phase == "between_calls":
            call_id = ledger.start_call()
            ledger.mark_dispatched(call_id)
            ledger.finish_call(
                call_id,
                status="completed",
                usage={
                    "input_tokens": 8,
                    "output_tokens": 3,
                    "total_tokens": 11,
                    "cached_input_tokens": 1,
                },
                cost_usd=0.004,
                cost_status="reported",
                cost_source="fake-provider",
            )
    else:
        snapshot = validate_true_moa_headers(
            headers,
            mystand_request=True,
            api_authenticated=True,
        )
        ledger = TrueMoAUsageLedger(snapshot, wave_id="7" * 32)
        ledger.set_wave_status("running")
        if phase == "between_calls":
            for index, slot in enumerate(
                (KIMI_ADVISOR_SLOT, DEEPSEEK_ADVISOR_SLOT),
                start=1,
            ):
                usage = {
                    "input_tokens": 8 + index,
                    "output_tokens": 2 + index,
                    "total_tokens": 10 + (2 * index),
                    "cached_input_tokens": 1,
                }
                ledger.start_slot(slot)
                call_id = ledger.start_advisor_call(slot)
                ledger.mark_dispatched(call_id)
                ledger.finish_advisor_call(
                    call_id,
                    status="completed",
                    usage=usage,
                    cost_usd=0.004 * index,
                    cost_status="reported",
                    cost_source="fake-provider",
                )
                ledger.finish_slot(
                    slot,
                    status="completed",
                    usage=usage,
                    cost_usd=0.004 * index,
                    cost_status="reported",
                    cost_source="fake-provider",
                )
            ledger.set_wave_status("advisors_completed")
    before_usage = ledger.to_dict()
    expected_call_count = (
        0
        if phase == "before_first_call"
        else 1
        if topology == "normal"
        else 2
    )
    assert len(before_usage["calls"]) == expected_call_count
    assert all(
        call["status"] == "completed"
        and call["usageStatus"] == "reported"
        for call in before_usage["calls"]
    )
    original._durable.save_usage(
        scoped_key,
        fingerprint,
        before_usage,
        state="running",
    )
    original._durable.close()

    restarted = api_server._IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    before_recovery = restarted.durable_record(scoped_key)
    assert before_recovery["state"] == "interrupted"
    assert before_recovery["usage"] == before_usage
    monkeypatch.setattr(api_server, "_idem_cache", restarted)
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions/usage",
        adapter._handle_chat_completion_usage,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        payload = await response.json()
        repeated = await client.post(
            "/v1/chat/completions/usage",
            headers=headers,
            json={"idempotency_key": raw_key},
        )
        repeated_payload = await repeated.json()

    assert response.status == 200
    assert repeated.status == 200
    assert repeated_payload == payload
    assert payload["status"] == "failed"
    assert payload["final"] is True
    assert payload["terminalState"] == "failed"
    assert payload["settlementBlocked"] is False
    assert payload["usage"]["status"] == "failed"
    assert payload["usage"]["calls"] == before_usage["calls"]
    if topology == "true_moa":
        assert payload["usage"]["slots"] == before_usage["slots"]
    assert payload["outcomeStatus"] == "none"
    assert "outcome" not in payload
    assert "outcomeId" not in payload
    assert "finalResponse" not in json.dumps(payload)
    recovered = restarted.durable_record(scoped_key)
    assert recovered["state"] == "failed"
    assert recovered["usage"] == payload["usage"]
    restarted._durable.close()


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
        ledger.mark_dispatched(call_id)
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
    ledger.mark_dispatched(final_call_id)
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
            _sealed_mystand_result({
                "final_response": late_text,
                "messages": [{"role": "assistant", "content": late_text}],
                "completed": True,
                "failed": False,
                "_mystand_request": True,
                "_true_moa_usage": completed_usage,
            }),
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

    # Release the single-process durable lock while the old provider callback
    # is still alive, then create the replacement process over the same
    # SQLite file.  The persistent usage-drain lease must prevent the
    # replacement from falsely declaring that live callback interrupted.
    original._durable.close()
    restarted = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    monkeypatch.setattr(api_server, "_idem_cache", restarted)
    interrupted = restarted.durable_record(scoped_key)
    assert interrupted["state"] == "running"

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
        assert pending_usage.status == 202
        assert pending_payload["status"] == "stopped_draining"
        assert pending_payload["final"] is False
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
        assert stopped_record["usage"]["status"] == "cancelled"
        assert stopped_record["usage"]["calls"][-1]["status"] == "completed"
        assert stopped_record["usage"]["calls"][-1]["totalTokens"] == 18
        assert (
            stopped_record["usage"]["calls"][-1]["usageStatus"]
            == "reported"
        )
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
        assert recovered_payload["usage"]["status"] == "cancelled"
        assert (
            recovered_payload["usage"]["calls"][-1]["status"]
            == "completed"
        )
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
        usage_ledger.mark_dispatched(advisor_call_id)
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
    usage_ledger.mark_dispatched(final_call_id)
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
        ledger.mark_dispatched(call_id)
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
    ledger.mark_dispatched(final_call_id)
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
    path = tmp_path / "outcome-crash-gap.sqlite"
    cache = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
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
    cache._durable.close()

    restarted = api_server._IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    restarted_record = restarted.durable_record(scoped_key)
    assert restarted_record["state"] == "interrupted"
    assert restarted_record["usage"]["status"] == "completed"
    monkeypatch.setattr(api_server, "_idem_cache", restarted)
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
    assert payload["terminalState"] == "interrupted"
    assert payload["final"] is True
    assert payload["usage"]["status"] == "completed"
    assert payload["outcomeStatus"] == "none"
    assert payload["settlementBlocked"] is True
    assert payload["errorCode"] == "true_moa_outcome_unavailable"
    assert "outcome" not in payload
    assert restarted.durable_record(scoped_key)["state"] == "interrupted"
    restarted._durable.close()


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


class _BlockingFinalAgent(_FakeFinalAgent):
    """One deterministic fixture for timeout, interrupt, and late-exit races."""

    def __init__(self, behavior: str = "ignore") -> None:
        super().__init__()
        self.behavior = behavior
        self.provider_started = threading.Event()
        self.release_provider = threading.Event()
        self.run_exited = threading.Event()
        self.interrupt_started = threading.Event()
        self.release_interrupt = threading.Event()
        self.interrupt_exited = threading.Event()
        self.late_text = "PRIVATE_LATE_FINAL_TIMEOUT_OUTPUT"
        self.worker_commit_result: bool | None = None
        self.context: tuple[str, str, str] = ("", "", "")

    def run_conversation(self, **kwargs):
        from gateway.session_context import get_session_env
        from xiaoban.trusted_runtime.turns import current_turn

        self.run_calls.append(kwargs)
        turn = current_turn()
        self.context = (
            get_session_env("XIAOBAN_SESSION_USER_ID"),
            get_session_env("XIAOBAN_SESSION_MESSAGE_ID"),
            str(getattr(turn, "request_id", "") or ""),
        )
        if self.behavior == "worker_commit":
            self.worker_commit_result = (
                self._true_moa_cancel_controller.try_commit_final(
                    f"gateway-final-handoff:{turn.request_id}",
                )
            )
        if self.behavior != "late_error":
            call_id = self._true_moa_usage_ledger.start_final_call("blocked")
            self._true_moa_usage_ledger.mark_dispatched(call_id)
        self.provider_started.set()
        assert self.release_provider.wait(1)
        if self.behavior == "late_error":
            self.run_exited.set()
            raise RuntimeError("PRIVATE_LATE_PROVIDER_FAILURE")
        try:
            self._true_moa_usage_ledger.finish_final_call(
                call_id,
                status="completed",
                usage={
                    "input_tokens": 19,
                    "output_tokens": 5,
                    "total_tokens": 24,
                    "cached_input_tokens": 3,
                },
            )
        finally:
            self.run_exited.set()
        return {
            "final_response": self.late_text,
            "completed": True,
            "failed": False,
            "messages": [{"role": "assistant", "content": self.late_text}],
        }

    def interrupt(self, reason: str) -> None:
        self.interrupt_calls.append(reason)
        if self.behavior == "release":
            self.release_provider.set()
        elif self.behavior == "raise":
            raise RuntimeError("fake interrupt failure")
        elif self.behavior == "block":
            self.interrupt_started.set()
            self.release_interrupt.wait(1)
            self.interrupt_exited.set()


def _gateway_case(monkeypatch, *, agent=None, epoch="60"):
    from xiaoban.trusted_runtime import true_moa_providers

    adapter = _adapter()
    headers = _mystand_headers(f"case-{uuid.uuid4().hex}", epoch=epoch)
    snapshot = validate_true_moa_headers(
        headers,
        mystand_request=True,
        api_authenticated=True,
    )
    monkeypatch.setattr(
        true_moa_providers,
        "strict_advisor_call",
        lambda *, slot, dispatch_callback, **_kwargs: (
            dispatch_callback(),
            StrictAdvisorResult(content=f"safe {slot.slot_id}"),
        )[1],
    )
    final_agent = agent or _FakeFinalAgent()
    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **_kwargs: final_agent,
    )
    return adapter, headers, snapshot, final_agent


async def _run_gateway_case(case, **kwargs):
    adapter, headers, snapshot, _agent = case
    return await adapter._run_agent(
        user_message=kwargs.pop("user_message", "deterministic test"),
        conversation_history=kwargs.pop("conversation_history", []),
        session_id=kwargs.pop("session_id", "compact-gateway-test"),
        request_headers=headers,
        agent_ref=kwargs.pop("agent_ref", [None, False, None]),
        true_moa_snapshot=snapshot,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_mode", ["normal", TRUE_MOA_MODE])
async def test_diagnostic_runner_never_resolves_or_dispatches_business_tools(
    monkeypatch,
    reasoning_mode,
):
    from gateway.platforms import api_server
    from gateway.platforms.api_server import (
        _mystand_completion_expected_binding,
    )
    from run_agent import AIAgent
    from tools.registry import ToolRegistry
    from xiaoban.trusted_runtime.turns import current_turn

    tool_names = (
        "mystand_resource_index",
        "mystand_query",
        "mystand_authorization",
    )
    tool_definitions = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {}},
        }
        for name in tool_names
    ]
    provider_payloads: list[dict[str, object]] = []
    handler_seen: list[str] = []
    registry_codes: list[str] = []
    turn_snapshots: list[dict[str, object]] = []
    advisor_tools: list[tuple[object, ...]] = []

    class DiagnosticProbeAgent(_FakeFinalAgent):
        valid_tool_names = set(tool_names)

        def __init__(self) -> None:
            super().__init__()
            self.tools = list(tool_definitions)

        def run_conversation(self, **kwargs):
            turn = current_turn()
            assert turn is not None
            provider_payload = AIAgent._build_api_kwargs(
                object.__new__(AIAgent),
                [{"role": "user", "content": "诊断上一轮"}],
            )
            provider_payloads.append(provider_payload)

            registry = ToolRegistry()
            for name in tool_names:
                registry.register(
                    name,
                    name,
                    {"name": name, "parameters": {}},
                    lambda _args, tool_name=name: (
                        handler_seen.append(tool_name)
                        or json.dumps({"ok": True})
                    ),
                )
            for index, name in enumerate(tool_names):
                denied = json.loads(
                    registry.dispatch(
                        name,
                        {},
                        tool_call_id=f"diagnostic-runner-{index}",
                    )
                )
                registry_codes.append(str(denied.get("code") or ""))
            turn_snapshots.append({
                "business_tools_disabled": turn.business_tools_disabled,
                "action_calls": list(turn.action_calls),
            })

            generic_ledger = getattr(
                self,
                "_paid_call_usage_ledger",
                None,
            )
            if (
                generic_ledger is not None
                and getattr(self, "_true_moa_usage_ledger", None) is None
            ):
                call_id = generic_ledger.start_call()
                generic_ledger.mark_dispatched(call_id)
                generic_ledger.finish_call(
                    call_id,
                    status="completed",
                    usage={
                        "input_tokens": self.session_prompt_tokens,
                        "output_tokens": self.session_completion_tokens,
                        "total_tokens": self.session_total_tokens,
                        "cached_input_tokens": (
                            self.session_cached_input_tokens
                        ),
                    },
                )
            result = super().run_conversation(**kwargs)
            result["final_response"] = "上一轮没有形成可用结果。"
            return result

    monkeypatch.setattr(
        "agent.chat_completion_helpers.build_api_kwargs",
        lambda _agent, _messages: {
            "tools": list(tool_definitions),
            "tool_choice": "required",
            "parallel_tool_calls": True,
        },
    )
    resolve_initial_choice = MagicMock(
        side_effect=AssertionError(
            "diagnostic turn parsed an initial business tool",
        ),
    )
    run_preexecuted = MagicMock(
        side_effect=AssertionError(
            "diagnostic turn entered deterministic preexecution",
        ),
    )
    monkeypatch.setattr(
        api_server,
        "_resolve_mystand_initial_tool_choice",
        resolve_initial_choice,
    )
    monkeypatch.setattr(
        api_server,
        "_run_mystand_preexecuted_evidence",
        run_preexecuted,
    )

    probe_agent = DiagnosticProbeAgent()
    if reasoning_mode == TRUE_MOA_MODE:
        from xiaoban.trusted_runtime import true_moa_providers

        case = _gateway_case(
            monkeypatch,
            agent=probe_agent,
            epoch="76",
        )
        adapter, headers, snapshot, _agent = case

        def fake_advisor(*, slot, tools, dispatch_callback, **_kwargs):
            advisor_tools.append(tuple(tools))
            dispatch_callback()
            return StrictAdvisorResult(content=f"safe {slot.slot_id}")

        monkeypatch.setattr(
            true_moa_providers,
            "strict_advisor_call",
            fake_advisor,
        )
    else:
        adapter = _adapter()
        headers = _mystand_headers("diagnostic-normal-runner")
        headers.pop(MODE_EPOCH_HEADER)
        headers.pop(MOA_PRESET_ID_HEADER)
        headers.pop(MOA_PRESET_REVISION_HEADER)
        headers[REASONING_MODE_HEADER] = "normal"
        headers[SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER] = (
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
        )
        snapshot = None
        monkeypatch.setattr(
            adapter,
            "_create_agent",
            lambda **_kwargs: probe_agent,
        )

    delivery_id = "xbd_" + (
        "b1" if reasoning_mode == "normal" else "b2"
    ) * 20
    headers.update({
        "Idempotency-Key": delivery_id,
        "X-Xiaoban-Delivery-Id": delivery_id,
        "X-Xiaoban-Delivery-Attempt": headers["X-Xiaoban-Attempt"],
        "X-Xiaoban-Completion-Protocol": "dynamic-evidence-v2",
        "X-Xiaoban-Evidence-Required": "0",
        "X-Xiaoban-Business-Tool-Mode": "disabled",
        "X-Xiaoban-Invocation-Fingerprint": "b" * 64,
    })
    completion_binding = _mystand_completion_expected_binding(
        headers,
        session_id="gateway-test-session",
    )

    result, usage = await adapter._run_agent(
        user_message=(
            "刚才查 AUTH-ABCDEFG 和 OUT-ABCDEFG 为什么失败？"
            "不要索引，直接读库"
        ),
        conversation_history=[],
        session_id="gateway-test-session",
        gateway_session_key="gateway-test-channel",
        request_headers=headers,
        agent_ref=[None, False, None],
        completion_protocol="dynamic-evidence-v2",
        completion_binding=completion_binding,
        dynamic_evidence_required=False,
        business_tools_disabled=True,
        true_moa_snapshot=snapshot,
    )

    assert result["completed"] is True
    assert result["final_response"] == "上一轮没有形成可用结果。"
    assert resolve_initial_choice.call_count == 0
    assert run_preexecuted.call_count == 0
    assert len(provider_payloads) == 1
    assert provider_payloads[0]["tools"] == []
    assert "tool_choice" not in provider_payloads[0]
    assert "parallel_tool_calls" not in provider_payloads[0]
    assert registry_codes == ["business_tools_disabled"] * len(tool_names)
    assert handler_seen == []
    assert turn_snapshots == [{
        "business_tools_disabled": True,
        "action_calls": [],
    }]
    if reasoning_mode == TRUE_MOA_MODE:
        assert advisor_tools == [(), ()]
        assert usage["true_moa"]["status"] == "completed"
    else:
        assert advisor_tools == []
        assert usage["agent_calls"]["status"] == "completed"


@pytest.mark.asyncio
async def test_dynamic_v2_true_moa_no_tool_chat_keeps_legacy_outcome(
    monkeypatch,
):
    from gateway.platforms.api_server import (
        _mystand_completion_expected_binding,
    )
    from gateway.platforms.true_moa_idempotency import _IdempotencyCache

    case = _gateway_case(monkeypatch)
    _adapter_instance, headers, _snapshot, _agent = case
    delivery_id = "xbd_" + ("9" * 40)
    headers.update(
        {
            "Idempotency-Key": delivery_id,
            "X-Xiaoban-Delivery-Id": delivery_id,
            "X-Xiaoban-Attempt": "1",
            "X-Xiaoban-Delivery-Attempt": "1",
            "X-Xiaoban-Completion-Protocol": "dynamic-evidence-v2",
            "X-Xiaoban-Invocation-Fingerprint": "8" * 64,
        }
    )
    completion_binding = _mystand_completion_expected_binding(
        headers,
        session_id="gateway-test-session",
    )

    result, usage = await _run_gateway_case(
        case,
        user_message="只聊一句，不查资料",
        session_id="gateway-test-session",
        completion_protocol="dynamic-evidence-v2",
        completion_binding=completion_binding,
    )

    assert result["completed"] is True
    assert result["final_response"] == "fake final synthesis"
    assert usage["true_moa"]["status"] == "completed"
    assert "_mystand_completion_protocol" not in result
    assert "_mystand_trusted_verification" not in result
    payload = _IdempotencyCache._completed_outcome_payload(result)
    assert "completionProtocol" not in payload
    assert "trustedVerification" not in payload


def _final_slot(usage: dict) -> dict:
    return {
        item["slotId"]: item for item in usage["true_moa"]["slots"]
    }[FINAL_EXECUTOR_SLOT.slot_id]


def _assert_closed(result, usage, *, slot, wave, error):
    assert {
        "text": result["final_response"],
        "messages": result["messages"],
        "completed": result["completed"],
        "failed": result["failed"],
        "error": result["error"],
        "wave": usage["true_moa"]["status"],
        "slot": _final_slot(usage)["status"],
    } == {
        "text": "",
        "messages": [],
        "completed": False,
        "failed": True,
        "error": error,
        "wave": wave,
        "slot": slot,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["cleanup", "durable_final"])
async def test_terminal_persistence_failures_stay_fail_closed(
    monkeypatch,
    failure_kind,
):
    case = _gateway_case(monkeypatch)
    final_agent = case[3]
    callback_payloads = []
    callback = None
    if failure_kind == "cleanup":
        final_agent.persistence_raise_in = {
            "drop_scaffolding",
            "save_trajectory",
            "persist_session",
        }
    else:
        def callback(payload):
            callback_payloads.append(payload)
            if _final_slot({"true_moa": payload})["status"] == "completed":
                raise RuntimeError("fake durable final ledger failure")

    result, usage = await _run_gateway_case(
        case,
        paid_call_usage_callback=callback,
    )
    assert final_agent.saved_trajectories == []
    assert final_agent.persisted_sessions == []
    assert final_agent._true_moa_cancel_controller.state == "completed"
    if failure_kind == "cleanup":
        assert result["final_response"] == "fake final synthesis"
        assert result["completed"] is True
        assert usage["true_moa"]["status"] == "completed"
        assert [x.split(":", 1)[0] for x in result["cleanup_errors"]] == [
            "drop_trailing_scaffolding",
            "save_trajectory",
            "persist_session",
        ]
    else:
        assert callback_payloads
        _assert_closed(
            result,
            usage,
            slot="completed",
            wave="failed",
            error="true MoA final settlement failed",
        )
        assert final_agent._defer_true_moa_final_commit is True


@pytest.mark.asyncio
@pytest.mark.parametrize("guard_kind", ["fact", "authorization"])
async def test_only_guarded_public_transcript_is_persisted(
    monkeypatch,
    guard_kind,
):
    from gateway.platforms import api_server
    from xiaoban.trusted_runtime import completion_guard

    raw_text = f"PRIVATE_RAW_{guard_kind.upper()}_TEXT"
    raw_tool = f"PRIVATE_RAW_{guard_kind.upper()}_TOOL"
    public_text = f"PUBLIC_GUARDED_{guard_kind.upper()}_ANSWER"
    case = _gateway_case(monkeypatch)
    adapter, _headers, _snapshot, agent = case
    tool_name = (
        "mystand_resource_index"
        if guard_kind == "fact"
        else "mystand_authorization"
    )
    agent.valid_tool_names = {tool_name}
    monkeypatch.setattr(
        api_server,
        "_run_mystand_preexecuted_evidence",
        lambda *_args, **_kwargs: [{
            "call_id": "trusted-read",
            "name": tool_name,
            "args": {},
            "content": json.dumps({"ok": True, "privateRaw": raw_tool}),
        }],
    )
    if guard_kind == "fact":
        guard_inputs = []

        def guard(text, _turn):
            guard_inputs.append(str(text))
            return types.SimpleNamespace(
                allowed=True,
                text=public_text,
                reason="projected_test",
                verification=None,
            )

        monkeypatch.setattr(completion_guard, "check_completion", guard)
    else:
        monkeypatch.setattr(
            api_server,
            "_resolve_mystand_initial_tool_choice",
            lambda *_args, **_kwargs: tool_name,
        )
        monkeypatch.setattr(
            completion_guard,
            "check_mystand_final_answer",
            lambda *_args, **_kwargs: types.SimpleNamespace(
                allowed=True,
                text=public_text,
                reason="projected_test",
                verification=None,
            ),
        )

    def raw_final(**kwargs):
        result = _FakeFinalAgent.run_conversation(agent, **kwargs)
        result.update({
            "final_response": raw_text,
            "messages": [
                {"role": "user", "content": "当前可信问题"},
                {"role": "tool", "content": raw_tool},
                {"role": "assistant", "content": raw_text},
            ],
        })
        return result

    agent.run_conversation = raw_final
    run_kwargs = {
        "user_message": "当前可信问题",
        "conversation_history": [
            {"role": "user", "content": "可信旧问题"},
            {"role": "assistant", "content": "可信旧回答"},
            {"role": "tool", "content": "PRIVATE_OLD_TOOL_BYTES"},
        ],
    }
    if guard_kind == "fact":
        run_kwargs["fact_requirement"] = {
            "schema": "mystand.fact-requirement.v1",
            "fact_kind": "collection",
            "module_id": "finance-ledger",
        }
    result, usage = await _run_gateway_case(case, **run_kwargs)
    persisted = "\n".join(
        agent.saved_trajectories + agent.persisted_sessions,
    )
    assert result["final_response"] == public_text
    assert result["completed"] is True
    assert usage["true_moa"]["status"] == "completed"
    assert all(x in persisted for x in (
        public_text,
        "当前可信问题",
        "可信旧问题",
        "可信旧回答",
    ))
    assert all(x not in persisted for x in (
        raw_text,
        raw_tool,
        "PRIVATE_OLD_TOOL_BYTES",
    ))
    assert raw_text not in json.dumps(result, ensure_ascii=False, default=str)
    assert raw_tool not in json.dumps(result, ensure_ascii=False, default=str)
    if guard_kind == "fact":
        assert guard_inputs == [raw_text, public_text]
        assert result["messages"] == []
    else:
        assert result["_mystand_egress_finalized"] is True
        assert result["_mystand_egress_output_digest"] == hashlib.sha256(
            public_text.encode(),
        ).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("guard_kind", ["fact", "egress"])
async def test_guard_projection_is_watchdog_bounded_and_turn_isolated(
    monkeypatch,
    guard_kind,
):
    from gateway.platforms import api_server
    from xiaoban.trusted_runtime import completion_guard, true_moa

    case = _gateway_case(monkeypatch)
    adapter, _headers, _snapshot, agent = case
    monkeypatch.setattr(true_moa, "TRUE_MOA_FINAL_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(
        true_moa,
        "TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS",
        0.03,
    )
    tool_name = (
        "mystand_resource_index"
        if guard_kind == "fact"
        else "mystand_authorization"
    )
    agent.valid_tool_names = {tool_name}
    original_turns, projected_turns = [], []
    started, release, finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    monkeypatch.setattr(
        api_server,
        "_run_mystand_preexecuted_evidence",
        lambda *_args, **kwargs: (
            original_turns.append(kwargs["trusted_turn"])
            or [{"name": tool_name, "content": '{"ok":true}'}]
        ),
    )
    late_text = "PRIVATE_LATE_GUARD_TEXT"
    if guard_kind == "fact":
        def blocking_guard(_text, turn):
            projected_turns.append(turn)
            started.set()
            release.wait(1)
            finished.set()
            return types.SimpleNamespace(
                allowed=True,
                text=late_text,
                reason="test",
                verification=None,
            )

        monkeypatch.setattr(
            completion_guard,
            "check_completion",
            blocking_guard,
        )
    else:
        monkeypatch.setattr(
            api_server,
            "_resolve_mystand_initial_tool_choice",
            lambda *_args, **_kwargs: tool_name,
        )

        def blocking_guard(_text, **kwargs):
            projected_turns.append(kwargs["result"]["_trusted_turn"])
            started.set()
            release.wait(1)
            finished.set()
            return types.SimpleNamespace(
                allowed=True,
                text=late_text,
                reason="test",
                verification=None,
            )

        monkeypatch.setattr(
            completion_guard,
            "check_mystand_final_answer",
            blocking_guard,
        )
    kwargs = {"user_message": "受保护事实请求"}
    if guard_kind == "fact":
        kwargs["fact_requirement"] = {
            "schema": "mystand.fact-requirement.v1",
            "fact_kind": "collection",
            "module_id": "finance-ledger",
        }
    before = time.monotonic()
    result, usage = await _run_gateway_case(case, **kwargs)
    assert time.monotonic() - before < 0.5
    assert started.is_set()
    _assert_closed(
        result,
        usage,
        slot="timed_out",
        wave="failed",
        error="true MoA final executor timed out",
    )
    assert len(original_turns) == len(projected_turns) == 1
    assert projected_turns[0] is not original_turns[0]
    original = original_turns[0]
    state = (original.state, list(original.states), original.terminal_reason)
    release.set()
    assert finished.wait(1)
    time.sleep(0.02)
    assert (original.state, original.states, original.terminal_reason) == state
    assert late_text not in json.dumps(
        {"result": result, "usage": usage},
        ensure_ascii=False,
    )
    assert agent.saved_trajectories == agent.persisted_sessions == []


@pytest.mark.asyncio
async def test_cancel_before_final_dispatch_never_persists(monkeypatch):
    from xiaoban.trusted_runtime import true_moa_providers

    case = _gateway_case(monkeypatch)
    advisor = MagicMock(side_effect=AssertionError("advisor dispatched"))
    monkeypatch.setattr(true_moa_providers, "strict_advisor_call", advisor)
    result, usage = await _run_gateway_case(
        case,
        agent_ref=[None, True, None],
    )
    assert result["interrupted"] is True
    assert usage["true_moa"]["status"] == "cancelled"
    assert case[3].run_calls == []
    assert case[3].saved_trajectories == case[3].persisted_sessions == []
    advisor.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "behavior",
    ["release", "ignore", "raise", "worker_commit", "block"],
)
async def test_final_deadline_fences_every_worker_and_interrupt_race(
    monkeypatch,
    behavior,
):
    from xiaoban.trusted_runtime import true_moa

    agent = _BlockingFinalAgent(behavior)
    case = _gateway_case(monkeypatch, agent=agent)
    monkeypatch.setattr(true_moa, "TRUE_MOA_FINAL_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(
        true_moa,
        "TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS",
        0.03,
    )
    before = time.monotonic()
    result, usage = await _run_gateway_case(case)
    assert time.monotonic() - before < 0.5
    assert agent.provider_started.is_set()
    assert len(agent.interrupt_calls) == 1
    _assert_closed(
        result,
        usage,
        slot="timed_out",
        wave="failed",
        error="true MoA final executor timed out",
    )
    assert agent.late_text not in json.dumps(
        {"result": result, "usage": usage},
        default=str,
    )
    assert agent.saved_trajectories == agent.persisted_sessions == []
    assert agent._true_moa_cancel_controller.state == "failed"
    if behavior == "release":
        assert agent.run_exited.is_set()
        assert agent.context[0] == "test-user"
        assert agent.context[1] == case[1]["X-Xiaoban-Message-Id"]
        assert agent.context[2].startswith("mystand-req-")
    elif behavior == "worker_commit":
        assert agent.worker_commit_result is False
    elif behavior == "block":
        assert agent.interrupt_started.wait(1)
        assert not agent.interrupt_exited.is_set()
    else:
        assert not agent.run_exited.is_set()
    agent.release_interrupt.set()
    agent.release_provider.set()
    assert agent.run_exited.wait(1)
    if behavior == "block":
        assert agent.interrupt_exited.wait(1)
    late = {"true_moa": agent._true_moa_usage_ledger.to_dict()}
    assert _final_slot(late)["status"] == "timed_out"
    assert _final_slot(late)["totalTokens"] == 24
    assert agent.late_text not in json.dumps(late)
    assert agent.saved_trajectories == agent.persisted_sessions == []


@pytest.mark.asyncio
async def test_final_deadline_includes_preexecuted_trusted_evidence(monkeypatch):
    from gateway.platforms import api_server
    from xiaoban.trusted_runtime import true_moa

    case = _gateway_case(monkeypatch)
    agent = case[3]
    agent.valid_tool_names = {"mystand_resource_index"}
    started, release, exited = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def blocking_evidence(*_args, **_kwargs):
        from gateway.session_context import get_session_env
        from xiaoban.trusted_runtime.turns import current_turn

        assert get_session_env("XIAOBAN_SESSION_USER_ID") == "test-user"
        assert current_turn() is not None
        started.set()
        assert release.wait(1)
        exited.set()
        return [{"name": "mystand_resource_index", "content": '{"ok":true}'}]

    monkeypatch.setattr(true_moa, "TRUE_MOA_FINAL_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(
        true_moa,
        "TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS",
        0.03,
    )
    monkeypatch.setattr(
        api_server,
        "_resolve_mystand_initial_tool_choice",
        lambda *_args, **_kwargs: "mystand_resource_index",
    )
    monkeypatch.setattr(
        api_server,
        "_run_mystand_preexecuted_evidence",
        blocking_evidence,
    )
    result, usage = await _run_gateway_case(case)
    assert started.is_set()
    _assert_closed(
        result,
        usage,
        slot="timed_out",
        wave="failed",
        error="true MoA final executor timed out",
    )
    assert agent.run_calls == []
    release.set()
    assert exited.wait(1)
    assert agent.run_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_status", ["running", "timed_out"])
async def test_durable_callback_cannot_hold_final_watchdog(
    monkeypatch,
    blocked_status,
):
    from xiaoban.trusted_runtime import true_moa

    agent = _BlockingFinalAgent()
    case = _gateway_case(monkeypatch, agent=agent)
    monkeypatch.setattr(true_moa, "TRUE_MOA_FINAL_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(
        true_moa,
        "TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS",
        0.03,
    )
    started, release = threading.Event(), threading.Event()

    def callback(payload):
        if _final_slot({"true_moa": payload})["status"] == blocked_status:
            started.set()
            release.wait(1)

    before = time.monotonic()
    result, usage = await _run_gateway_case(
        case,
        paid_call_usage_callback=callback,
    )
    assert time.monotonic() - before < 0.5
    assert started.is_set()
    _assert_closed(
        result,
        usage,
        slot="timed_out",
        wave="failed",
        error="true MoA final settlement failed",
    )
    assert agent.saved_trajectories == agent.persisted_sessions == []
    release.set()
    agent.release_provider.set()
    if blocked_status == "running":
        assert not agent.provider_started.is_set()
        assert not agent.run_exited.is_set()
    else:
        assert agent.run_exited.wait(1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_kind", "expected_slot", "expected_wave"),
    [
        ("timed_out", "timed_out", "failed"),
        ("cancelled", "cancelled", "cancelled"),
        ("failed", "failed", "failed"),
    ],
)
async def test_terminal_callback_failure_is_attempted_once(
    monkeypatch,
    terminal_kind,
    expected_slot,
    expected_wave,
):
    from xiaoban.trusted_runtime import true_moa

    if terminal_kind == "timed_out":
        agent = _BlockingFinalAgent()
    else:
        agent = _FakeFinalAgent()
        if terminal_kind == "cancelled":
            def terminal_result(**_kwargs):
                raise KeyboardInterrupt("PRIVATE_CANCEL_CALLBACK")
        else:
            def terminal_result(**_kwargs):
                return {
                    "final_response": "PRIVATE_FAILED_TEXT",
                    "completed": False,
                    "failed": True,
                    "messages": [{
                        "role": "assistant",
                        "content": "PRIVATE_FAILED_TEXT",
                    }],
                }
        agent.run_conversation = terminal_result
    case = _gateway_case(monkeypatch, agent=agent)
    if terminal_kind == "timed_out":
        monkeypatch.setattr(
            true_moa,
            "TRUE_MOA_FINAL_TIMEOUT_SECONDS",
            0.03,
        )
        monkeypatch.setattr(
            true_moa,
            "TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS",
            0.03,
        )
    callback_count = 0

    def callback(payload):
        nonlocal callback_count
        if _final_slot({"true_moa": payload})["status"] == expected_slot:
            callback_count += 1
            raise RuntimeError("fake terminal durable failure")

    result, usage = await _run_gateway_case(
        case,
        paid_call_usage_callback=callback,
    )
    assert callback_count == 1
    _assert_closed(
        result,
        usage,
        slot=expected_slot,
        wave=expected_wave,
        error="true MoA final settlement failed",
    )
    assert agent.saved_trajectories == agent.persisted_sessions == []
    if terminal_kind == "timed_out":
        agent.release_provider.set()
        assert agent.run_exited.wait(1)
        late = {"true_moa": agent._true_moa_usage_ledger.to_dict()}
        assert _final_slot(late)["status"] == "timed_out"
        assert _final_slot(late)["totalTokens"] == 24


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["preflight", "turn_setup"])
async def test_early_terminal_callback_block_is_bounded(
    monkeypatch,
    failure_stage,
):
    from xiaoban.trusted_runtime import true_moa

    case = _gateway_case(monkeypatch)
    adapter = case[0]
    monkeypatch.setattr(
        true_moa,
        "TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS",
        0.03,
    )
    if failure_stage == "preflight":
        monkeypatch.setattr(
            adapter,
            "_create_agent",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("PRIVATE_PREFLIGHT_FAILURE"),
            ),
        )
    else:
        monkeypatch.setattr(
            adapter,
            "_bind_api_server_session",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("PRIVATE_SETUP_FAILURE"),
            ),
        )
    started, release = threading.Event(), threading.Event()

    def callback(payload):
        if _final_slot({"true_moa": payload})["status"] == "failed":
            started.set()
            release.wait(1)

    before = time.monotonic()
    result, usage = await _run_gateway_case(
        case,
        paid_call_usage_callback=callback,
    )
    assert time.monotonic() - before < 0.5
    assert started.is_set()
    _assert_closed(
        result,
        usage,
        slot="failed",
        wave="failed",
        error="true MoA final settlement failed",
    )
    release.set()


def test_stop_returns_before_a_blocking_agent_interrupt():
    from gateway.platforms import api_server
    from xiaoban.trusted_runtime.true_moa import TrueMoACancelController

    controller = TrueMoACancelController()
    started, release, finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    class Agent:
        def interrupt(self, _reason):
            started.set()
            release.wait(1)
            finished.set()

    agent_ref = [Agent(), False, controller]
    before = time.monotonic()
    assert api_server._cancel_chat_agent_ref(agent_ref, "test stop") is True
    assert time.monotonic() - before < 0.1
    assert controller.state == "cancelled"
    assert agent_ref[1] is True
    assert started.wait(1) and not finished.is_set()
    release.set()
    assert finished.wait(1)


@pytest.mark.parametrize("durable_accepts", [False, True])
def test_durable_stop_fence_wins_before_local_cancel(durable_accepts):
    from gateway.platforms import api_server
    from xiaoban.trusted_runtime.true_moa import TrueMoACancelController

    cache = api_server._IdempotencyCache(max_items=4, ttl_seconds=30)
    key, fingerprint = "durable-stop", "fingerprint"
    controller = TrueMoACancelController()
    interrupted = threading.Event()
    order = []

    class Agent:
        def interrupt(self, _reason):
            if durable_accepts:
                assert key in cache._stopped
            order.append("interrupt")
            interrupted.set()

    def mark_stopped(actual):
        assert actual == key
        assert controller.state == "running"
        assert key not in cache._stopped
        order.append("durable")
        return durable_accepts

    agent_ref = [Agent(), False, controller]
    cache._inflight[(key, fingerprint)] = object()
    cache._agent_refs[(key, fingerprint)] = agent_ref
    cache._durable = types.SimpleNamespace(mark_stopped=mark_stopped)
    assert cache.stop(key, durable=True) is durable_accepts
    if durable_accepts:
        assert controller.state == "cancelled"
        assert agent_ref[1] is True
        assert key in cache._stopped
        assert interrupted.wait(1)
        assert order == ["durable", "interrupt"]
    else:
        assert controller.state == "running"
        assert agent_ref[1] is False
        assert not interrupted.wait(0.05)
        assert order == ["durable"]
        assert key not in cache._stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "state", "interrupted"),
    [(KeyboardInterrupt, "cancelled", True), (SystemExit, "failed", False)],
)
async def test_final_worker_baseexception_stays_inside_request(
    monkeypatch,
    error_type,
    state,
    interrupted,
):
    agent = _FakeFinalAgent()

    def raise_error(**_kwargs):
        raise error_type("PRIVATE_BASEEXCEPTION")

    agent.run_conversation = raise_error
    result, usage = await _run_gateway_case(
        _gateway_case(monkeypatch, agent=agent),
    )
    assert result["interrupted"] is interrupted
    assert agent._true_moa_cancel_controller.state == state
    _assert_closed(
        result,
        usage,
        slot=state,
        wave=state,
        error=(
            "completion stopped"
            if interrupted
            else "true MoA final executor failed"
        ),
    )
    assert "PRIVATE_BASEEXCEPTION" not in json.dumps(
        {"result": result, "usage": usage},
    )
    assert agent.saved_trajectories == agent.persisted_sessions == []


@pytest.mark.asyncio
async def test_preexecuted_tool_keyboardinterrupt_becomes_cancelled(monkeypatch):
    from gateway.platforms import api_server
    from tools import mystand_resource_index_tool

    case = _gateway_case(monkeypatch)
    agent = case[3]
    agent.valid_tool_names = {"mystand_authorization"}
    monkeypatch.setattr(
        api_server,
        "_resolve_mystand_initial_tool_choice",
        lambda *_args, **_kwargs: "mystand_authorization",
    )

    def stop_tool(_args):
        raise KeyboardInterrupt("PRIVATE_TOOL_INTERRUPT")

    monkeypatch.setattr(
        mystand_resource_index_tool,
        "mystand_resource_index_tool_handler",
        stop_tool,
    )
    result, usage = await _run_gateway_case(
        case,
        user_message="读取 AUTH-ABCDEF1",
    )
    _assert_closed(
        result,
        usage,
        slot="cancelled",
        wave="cancelled",
        error="completion stopped",
    )
    assert result["interrupted"] is True
    assert agent.run_calls == []
    assert agent._true_moa_cancel_controller.state == "cancelled"


@pytest.mark.asyncio
async def test_stop_first_beats_late_worker_error(monkeypatch):
    agent = _BlockingFinalAgent("late_error")
    agent_ref = [None, False, None]
    task = asyncio.create_task(
        _run_gateway_case(
            _gateway_case(monkeypatch, agent=agent),
            agent_ref=agent_ref,
        ),
    )
    assert await asyncio.to_thread(agent.provider_started.wait, 1)
    assert agent_ref[2].cancel() is True
    agent.release_provider.set()
    result, usage = await task
    _assert_closed(
        result,
        usage,
        slot="cancelled",
        wave="cancelled",
        error="completion stopped",
    )
    assert result["interrupted"] is True
    assert agent._true_moa_cancel_controller.state == "cancelled"
    assert "PRIVATE_LATE_PROVIDER_FAILURE" not in json.dumps(
        {"result": result, "usage": usage},
    )
