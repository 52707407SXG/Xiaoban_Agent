"""Deterministic restart and monotonicity gates for the true-MoA ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from gateway.platforms.api_server import (
    IdempotencyConflictError,
    _IdempotencyCache,
)
from xiaoban.trusted_runtime.true_moa import (
    DEEPSEEK_ADVISOR_SLOT,
    FINAL_EXECUTOR_SLOT,
    KIMI_ADVISOR_SLOT,
    TRUE_MOA_MODE,
    TRUE_MOA_PRESET_ID,
    TRUE_MOA_PRESET_REVISION,
    StrictAdvisorResult,
    TrueMoAExecutionError,
    TrueMoASnapshot,
    TrueMoAUsageLedger,
    run_true_moa_advisors,
)
from xiaoban.trusted_runtime import true_moa_durable, true_moa_providers
from xiaoban.trusted_runtime.true_moa_durable import (
    TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
    TRUE_MOA_DURABLE_MAX_CALLS,
    TRUE_MOA_DURABLE_MAX_FINAL_CALLS,
    TRUE_MOA_OUTCOME_BINDING_SCHEMA,
    TRUE_MOA_OUTCOME_MAX_TEXT_BYTES,
    TrueMoAOutcomeBindingError,
    TrueMoAOutcomeUnavailableError,
    TrueMoADurableStore,
    project_true_moa_usage,
)
from xiaoban.trusted_runtime.types import TrustedIdentity


_OUTCOME_KEYS = {"test-v1": b"\x11" * 32}


def _snapshot(epoch: str = "1") -> TrueMoASnapshot:
    return TrueMoASnapshot(
        mode=TRUE_MOA_MODE,
        mode_epoch=epoch,
        preset_id=TRUE_MOA_PRESET_ID,
        preset_revision=TRUE_MOA_PRESET_REVISION,
    )


def _fingerprint(seed: str = "request") -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _running_ledger(
    *,
    epoch: str = "1",
    wave_id: str = "a" * 32,
) -> TrueMoAUsageLedger:
    ledger = TrueMoAUsageLedger(_snapshot(epoch), wave_id=wave_id)
    ledger.set_wave_status("running")
    return ledger


def _finish_advisor(
    ledger: TrueMoAUsageLedger,
    slot,
    *,
    input_tokens: int,
    output_tokens: int,
) -> None:
    ledger.start_slot(slot)
    call_id = ledger.start_advisor_call(slot)
    ledger.finish_advisor_call(
        call_id,
        status="completed",
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cached_input_tokens": 0,
        },
        cost_usd=0.01,
        cost_status="reported",
        cost_source="fake-provider",
    )
    ledger.finish_slot(
        slot,
        status="completed",
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cached_input_tokens": 0,
        },
        cost_usd=0.01,
        cost_status="reported",
        cost_source="fake-provider",
    )


def _close_cache(cache: _IdempotencyCache) -> None:
    durable = getattr(cache, "_durable", None)
    if durable is not None:
        durable.close()


def _completed_ledger(
    *,
    epoch: str = "1",
    wave_id: str = "a" * 32,
) -> dict:
    ledger = _running_ledger(epoch=epoch, wave_id=wave_id)
    _finish_advisor(
        ledger,
        KIMI_ADVISOR_SLOT,
        input_tokens=5,
        output_tokens=2,
    )
    _finish_advisor(
        ledger,
        DEEPSEEK_ADVISOR_SLOT,
        input_tokens=7,
        output_tokens=3,
    )
    ledger.start_slot(FINAL_EXECUTOR_SLOT)
    call_id = ledger.start_final_call(f"final-{wave_id[:8]}")
    final_usage = {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "cached_input_tokens": 2,
    }
    ledger.finish_final_call(
        call_id,
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


def _outcome_binding(
    *,
    epoch: str = "1",
    site_id: str = "mystand-site",
    user_id: str = "owner-user",
    delivery_id: str = "xbd_" + ("a" * 40),
    message_id: str = "message-1",
    attempt: int = 1,
    request_fingerprint: str | None = None,
) -> dict:
    return {
        "schema": TRUE_MOA_OUTCOME_BINDING_SCHEMA,
        "siteId": site_id,
        "userId": user_id,
        "deliveryId": delivery_id,
        "messageId": message_id,
        "attempt": attempt,
        "requestFingerprint": (
            request_fingerprint or _fingerprint("request-binding")
        ),
        "datascopeFingerprint": TrustedIdentity(
            account_id=user_id,
            data_scope="mystand",
            source="server_session",
        ).datascope_fingerprint,
        "modeEpoch": epoch,
        "presetId": TRUE_MOA_PRESET_ID,
        "presetRevision": TRUE_MOA_PRESET_REVISION,
    }


def _completed_outcome(
    text: str = "owner-visible completed answer",
    *,
    binding: dict | None = None,
    fact_guard_required: bool = False,
) -> dict:
    digest = hashlib.sha256(text.encode()).hexdigest()
    outcome = {
        "schema": TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
        "completed": True,
        "finalResponse": text,
        "outputDigest": digest,
        "factGuardRequired": fact_guard_required,
    }
    if fact_guard_required:
        assert binding is not None
        outcome["trustedVerification"] = {
            "schema": "mystand.xiaoban-fact-verification.v1",
            "verified": True,
            "delivery_id": binding["deliveryId"],
            "message_id": binding["messageId"],
            "attempt": binding["attempt"],
            "request_fingerprint": binding["requestFingerprint"],
            "datascope_fingerprint": binding["datascopeFingerprint"],
            "output_digest": digest,
            "decision": "projected_evidence",
        }
    return outcome


def test_store_merges_late_snapshots_without_usage_or_state_regression(
    tmp_path: Path,
):
    store = TrueMoADurableStore(str(tmp_path / "ledger.sqlite"))
    key = "scoped-delivery"
    fingerprint = _fingerprint()
    assert store.claim(key, fingerprint, kind="execution") == "missing"

    ledger = _running_ledger()
    ledger.start_slot(KIMI_ADVISOR_SLOT)
    kimi_call_id = ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    old_snapshot = ledger.to_dict()
    ledger.finish_advisor_call(
        kimi_call_id,
        status="completed",
        usage={
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
            "cached_input_tokens": 2,
        },
    )
    ledger.finish_slot(
        KIMI_ADVISOR_SLOT,
        status="completed",
        usage={
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
            "cached_input_tokens": 2,
        },
    )
    _finish_advisor(
        ledger,
        DEEPSEEK_ADVISOR_SLOT,
        input_tokens=11,
        output_tokens=4,
    )
    new_snapshot = ledger.to_dict()

    store.save_usage(key, fingerprint, new_snapshot, state="running")
    store.save_usage(key, fingerprint, old_snapshot, state="running")
    store.mark_stopped(key)
    store.save_usage(key, fingerprint, old_snapshot, state="running")

    recovered = store.get(key)
    assert recovered is not None
    assert recovered["state"] == "stopped"
    calls = {
        item["slotId"]: item
        for item in recovered["usage"]["calls"]
    }
    assert calls[KIMI_ADVISOR_SLOT.slot_id]["totalTokens"] == 10
    assert calls[KIMI_ADVISOR_SLOT.slot_id]["status"] == "completed"
    assert calls[DEEPSEEK_ADVISOR_SLOT.slot_id]["totalTokens"] == 15
    assert calls[DEEPSEEK_ADVISOR_SLOT.slot_id]["status"] == "completed"
    store.close()


def test_stopped_orphan_fence_accepts_late_failed_usage_without_reopening(
    tmp_path: Path,
):
    store = TrueMoADurableStore(str(tmp_path / "stopped-orphan.sqlite"))
    key = "scoped-stopped-orphan"
    fingerprint = _fingerprint("stopped-orphan")
    assert store.claim(key, fingerprint, kind="execution") == "missing"

    ledger = _running_ledger(wave_id="b" * 32)
    ledger.start_slot(KIMI_ADVISOR_SLOT)
    call_id = ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    store.save_usage(
        key,
        fingerprint,
        ledger.to_dict(),
        state="running",
    )
    assert store.mark_stopped(key) is True
    assert store.terminalize_stopped_running_calls(key) is True

    fenced = store.get(key)
    assert fenced is not None
    assert fenced["state"] == "stopped"
    assert fenced["usage"]["status"] == "cancelled"
    assert fenced["usage"]["calls"][0]["status"] == "timed_out"
    assert fenced["usage"]["calls"][0]["usageStatus"] == "unavailable"

    late_usage = {
        "input_tokens": 9,
        "output_tokens": 2,
        "total_tokens": 11,
        "cached_input_tokens": 1,
    }
    ledger.finish_advisor_call(
        call_id,
        status="failed",
        usage=late_usage,
        error_category="late_malformed_result_after_terminal",
    )
    ledger.finish_slot(
        KIMI_ADVISOR_SLOT,
        status="cancelled",
        usage=late_usage,
        error_category="late_malformed_result_after_terminal",
    )
    ledger.set_wave_status("cancelled")
    store.save_usage(
        key,
        fingerprint,
        ledger.to_dict(),
        state="stopped",
    )

    recovered = store.get(key)
    assert recovered is not None
    call = recovered["usage"]["calls"][0]
    assert recovered["state"] == "stopped"
    assert recovered["usage"]["status"] == "cancelled"
    assert call["status"] == "timed_out"
    assert call["usageStatus"] == "reported"
    assert call["totalTokens"] == 11
    assert call["errorCategory"] == "late_malformed_result_after_terminal"
    store.close()


def test_mark_stopped_accepts_restart_state_but_not_finished_terminals(
    tmp_path: Path,
):
    store = TrueMoADurableStore(str(tmp_path / "atomic-stop.sqlite"))

    interrupted_key = "restart-interrupted"
    interrupted_fingerprint = _fingerprint(interrupted_key)
    assert (
        store.claim(
            interrupted_key,
            interrupted_fingerprint,
            kind="execution",
        )
        == "missing"
    )
    store.set_state(interrupted_key, state="running")
    store.set_state(interrupted_key, state="interrupted")
    assert store.mark_stopped(interrupted_key) is True
    assert store.get(interrupted_key)["state"] == "stopped"
    assert store.mark_stopped(interrupted_key) is True

    for terminal in ("completed", "failed"):
        key = f"already-{terminal}"
        assert (
            store.claim(
                key,
                _fingerprint(key),
                kind="execution",
            )
            == "missing"
        )
        store.set_state(key, state=terminal)
        assert store.mark_stopped(key) is False
        assert store.get(key)["state"] == terminal

    store.close()


def test_mark_stopped_applies_capacity_only_to_missing_rows(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        true_moa_durable,
        "TRUE_MOA_DURABLE_MAX_ROWS",
        2,
    )
    store = TrueMoADurableStore(str(tmp_path / "stop-capacity.sqlite"))
    assert store.mark_stopped("first-stop") is True
    assert store.mark_stopped("second-stop") is True

    # Existing stoppable rows remain mutable at capacity.
    assert store.mark_stopped("first-stop") is True
    with pytest.raises(
        RuntimeError,
        match="durable ledger capacity exhausted",
    ):
        store.mark_stopped("third-stop")
    assert store.get("third-stop") is None
    store.close()


def test_store_accepts_dynamic_advisor_subset_and_keeps_fixed_call_order(
    tmp_path: Path,
):
    store = TrueMoADurableStore(str(tmp_path / "dynamic-calls.sqlite"))
    key = "dynamic-advisor-calls"
    fingerprint = _fingerprint(key)
    assert store.claim(key, fingerprint, kind="execution") == "missing"

    ledger = _running_ledger(wave_id="f" * 32)
    ledger.start_slot(DEEPSEEK_ADVISOR_SLOT)
    ledger.start_advisor_call(DEEPSEEK_ADVISOR_SLOT)
    deepseek_only = ledger.to_dict()
    store.save_usage(key, fingerprint, deepseek_only, state="running")

    ledger.start_slot(KIMI_ADVISOR_SLOT)
    ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    both_advisors = ledger.to_dict()
    store.save_usage(key, fingerprint, both_advisors, state="running")
    store.save_usage(key, fingerprint, deepseek_only, state="running")

    recovered = store.get(key)
    assert recovered is not None
    assert [
        call["slotId"] for call in recovered["usage"]["calls"]
    ] == [
        KIMI_ADVISOR_SLOT.slot_id,
        DEEPSEEK_ADVISOR_SLOT.slot_id,
    ]
    store.close()


def test_store_fill_once_upgrades_unknown_cache_split_without_regression(
    tmp_path: Path,
):
    store = TrueMoADurableStore(str(tmp_path / "cache-fill-once.sqlite"))
    key = "cache-fill-once"
    fingerprint = _fingerprint(key)
    assert store.claim(key, fingerprint, kind="execution") == "missing"

    ledger = _running_ledger(wave_id="1" * 32)
    ledger.start_slot(DEEPSEEK_ADVISOR_SLOT)
    call_id = ledger.start_advisor_call(DEEPSEEK_ADVISOR_SLOT)
    base_usage = {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }
    ledger.finish_advisor_call(
        call_id,
        status="completed",
        usage=base_usage,
    )
    unknown_cache = ledger.to_dict()
    assert unknown_cache["calls"][0]["usageStatus"] == "partial"
    assert unknown_cache["calls"][0]["cachedInputTokens"] is None
    forged_reported = {
        **unknown_cache,
        "calls": [dict(unknown_cache["calls"][0])],
    }
    forged_reported["calls"][0]["usageStatus"] = "reported"
    with pytest.raises(
        ValueError,
        match="reported true MoA usage is incomplete",
    ):
        project_true_moa_usage(forged_reported)
    forged_inconsistent_total = {
        **unknown_cache,
        "calls": [dict(unknown_cache["calls"][0])],
    }
    forged_inconsistent_total["calls"][0]["totalTokens"] = 15
    with pytest.raises(
        ValueError,
        match="inconsistent true MoA token total",
    ):
        project_true_moa_usage(forged_inconsistent_total)

    ledger.finish_advisor_call(
        call_id,
        status="completed",
        usage={
            **base_usage,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    )
    reported_cache = ledger.to_dict()
    store.save_usage(key, fingerprint, unknown_cache, state="running")
    store.save_usage(key, fingerprint, reported_cache, state="running")
    store.save_usage(key, fingerprint, unknown_cache, state="running")

    recovered = store.get(key)
    assert recovered is not None
    call = recovered["usage"]["calls"][0]
    assert call["usageStatus"] == "reported"
    assert call["cachedInputTokens"] == 4
    store.close()


@pytest.mark.parametrize(
    ("wave_status", "receipt_status", "durable_state"),
    [
        ("cancelled", "cancelled", "stopped"),
        ("failed", "failed", "failed"),
    ],
)
def test_interrupted_delivery_accepts_late_terminal_usage_callback(
    tmp_path: Path,
    wave_status: str,
    receipt_status: str,
    durable_state: str,
):
    cache = _IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(tmp_path / f"late-{durable_state}.sqlite"),
    )
    key = f"late-{durable_state}"
    fingerprint = _fingerprint(key)
    assert cache._durable.claim(
        key,
        fingerprint,
        kind="execution",
    ) == "missing"
    cache._durable.set_state(key, state="running")

    ledger = _running_ledger(wave_id="2" * 32)
    ledger.start_slot(DEEPSEEK_ADVISOR_SLOT)
    call_id = ledger.start_advisor_call(DEEPSEEK_ADVISOR_SLOT)
    cache.persist_usage(key, fingerprint, ledger.to_dict())

    # The SSE asyncio wrapper can be interrupted while the provider thread
    # is still unwinding.  Its outcome is provisional until that thread's
    # trusted usage callback supplies a terminal ledger.
    cache._durable.set_state(key, state="interrupted")
    usage = {
        "input_tokens": 13,
        "output_tokens": 5,
        "total_tokens": 18,
        "cached_input_tokens": 3,
    }
    ledger.finish_advisor_call(
        call_id,
        status=receipt_status,
        usage=usage,
        error_category="late_terminal_after_disconnect",
    )
    ledger.finish_slot(
        DEEPSEEK_ADVISOR_SLOT,
        status=receipt_status,
        usage=usage,
        error_category="late_terminal_after_disconnect",
    )
    ledger.set_wave_status(wave_status)
    cache.persist_usage(key, fingerprint, ledger.to_dict())

    recovered = cache._durable.get(key)
    assert recovered is not None
    assert recovered["state"] == durable_state
    call = recovered["usage"]["calls"][0]
    assert call["status"] == receipt_status
    assert call["inputTokens"] == 13
    assert call["outputTokens"] == 5
    assert call["totalTokens"] == 18
    assert call["cachedInputTokens"] == 3
    assert call["usageStatus"] == "reported"

    # The opposite scheduling order is also legal: a late wrapper
    # cancellation cannot regress an already known provider terminal.
    cache._durable.set_state(key, state="interrupted")
    assert cache._durable.get(key)["state"] == durable_state
    _close_cache(cache)


def test_real_advisor_wave_persists_advisors_completed_state(
    tmp_path: Path,
):
    store = TrueMoADurableStore(str(tmp_path / "advisor-wave.sqlite"))
    key = "advisor-wave"
    fingerprint = _fingerprint("advisor-wave")
    assert store.claim(key, fingerprint, kind="execution") == "missing"

    def _persist(usage):
        store.save_usage(
            key,
            fingerprint,
            usage,
            state="running",
        )

    ledger = TrueMoAUsageLedger(
        _snapshot("9"),
        wave_id="d" * 32,
        on_change=_persist,
    )

    def _fake_caller(*, slot, dispatch_callback, **_kwargs):
        dispatch_callback()
        return StrictAdvisorResult(
            content=f"safe advice from {slot.slot_id}",
            usage={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "cached_input_tokens": 0,
            },
        )

    bundle = run_true_moa_advisors(
        _snapshot("9"),
        current_question="test",
        conversation_history=[],
        strict_caller=_fake_caller,
        usage_ledger=ledger,
    )
    assert bundle.ledger.to_dict()["status"] == "advisors_completed"
    recovered = store.get(key)
    assert recovered["state"] == "running"
    assert recovered["usage"]["status"] == "advisors_completed"
    assert all(
        call["status"] == "completed"
        for call in recovered["usage"]["calls"]
    )
    store.close()


@pytest.mark.parametrize("failure_stage", ["credentials", "client"])
def test_predispatch_provider_failure_persists_zero_actual_calls(
    monkeypatch,
    tmp_path: Path,
    failure_stage: str,
):
    store = TrueMoADurableStore(
        str(tmp_path / f"predispatch-{failure_stage}.sqlite")
    )
    key = f"predispatch-{failure_stage}-failure"
    fingerprint = _fingerprint(key)
    assert store.claim(key, fingerprint, kind="execution") == "missing"

    def _persist(usage):
        store.save_usage(
            key,
            fingerprint,
            usage,
            state="failed" if usage["status"] == "failed" else "running",
        )

    ledger = TrueMoAUsageLedger(
        _snapshot("10"),
        wave_id="e" * 32,
        on_change=_persist,
    )

    def _fail_credentials(*_args, **_kwargs):
        raise true_moa_providers.StrictAdvisorProviderError(
            "forced_credentials_unavailable"
        )

    if failure_stage == "credentials":
        monkeypatch.setattr(
            true_moa_providers,
            "_fixed_credentials",
            _fail_credentials,
        )
    else:
        monkeypatch.setattr(
            true_moa_providers,
            "_fixed_credentials",
            lambda provider, **_kwargs: {
                "api_key": "fake-key",
                "base_url": (
                    "https://api.kimi.com/coding"
                    if provider == "kimi-coding"
                    else "https://api.deepseek.com/v1"
                ),
            },
        )

        def _fail_client(*_args, **_kwargs):
            raise RuntimeError("forced client construction failure")

        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client",
            _fail_client,
        )
        monkeypatch.setattr("openai.OpenAI", _fail_client)

    with pytest.raises(TrueMoAExecutionError):
        run_true_moa_advisors(
            _snapshot("10"),
            current_question="must not dispatch",
            conversation_history=[],
            strict_caller=true_moa_providers.strict_advisor_call,
            usage_ledger=ledger,
        )

    recovered = store.get(key)
    assert recovered is not None
    assert recovered["state"] == "failed"
    assert recovered["usage"]["status"] == "failed"
    assert recovered["usage"]["calls"] == []
    assert all(
        slot["status"] in {"failed", "cancelled"}
        for slot in recovered["usage"]["slots"][:2]
    )
    store.close()


@pytest.mark.asyncio
async def test_restart_replay_preserves_calls_and_never_recomputes(
    tmp_path: Path,
):
    path = tmp_path / "restart.sqlite"
    key = "restart-delivery"
    fingerprint = _fingerprint("same")
    first = TrueMoADurableStore(str(path))
    assert first.claim(key, fingerprint, kind="execution") == "missing"
    first.set_state(key, state="running")
    ledger = _running_ledger(epoch="2", wave_id="b" * 32)
    _finish_advisor(
        ledger,
        KIMI_ADVISOR_SLOT,
        input_tokens=5,
        output_tokens=2,
    )
    ledger.start_slot(DEEPSEEK_ADVISOR_SLOT)
    ledger.start_advisor_call(DEEPSEEK_ADVISOR_SLOT)
    first.save_usage(
        key,
        fingerprint,
        ledger.to_dict(),
        state="running",
    )
    first.close()

    cache = _IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
    )
    compute_count = 0

    async def _must_not_compute():
        nonlocal compute_count
        compute_count += 1
        raise AssertionError("restart replay dispatched duplicate work")

    result, usage = await cache.get_or_set(
        key,
        fingerprint,
        _must_not_compute,
        durable=True,
    )
    assert compute_count == 0
    assert result["interrupted"] is True
    assert result["final_response"] == ""
    assert usage["true_moa"]["status"] == "failed"
    kimi_call = usage["true_moa"]["calls"][0]
    deepseek_call = usage["true_moa"]["calls"][1]
    assert kimi_call["totalTokens"] == 7
    assert kimi_call["status"] == "completed"
    assert deepseek_call["status"] == "failed"
    assert deepseek_call["errorCategory"] == "agent_restart_outcome_unknown"

    with pytest.raises(IdempotencyConflictError):
        await cache.get_or_set(
            key,
            _fingerprint("changed"),
            _must_not_compute,
            durable=True,
        )
    assert compute_count == 0
    _close_cache(cache)


@pytest.mark.asyncio
async def test_stop_before_start_survives_restart_and_binds_first_fingerprint(
    tmp_path: Path,
):
    path = tmp_path / "stop-before-start.sqlite"
    key = "stop-before-start"
    first = _IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
    )
    assert first.stop(key, durable=True) is True
    _close_cache(first)

    second = _IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
    )
    compute_count = 0

    async def _must_not_compute():
        nonlocal compute_count
        compute_count += 1
        return ({}, {})

    result, _usage = await second.get_or_set(
        key,
        _fingerprint("first"),
        _must_not_compute,
        durable=True,
    )
    assert compute_count == 0
    assert result["interrupted"] is True
    state, cached = second.result_state(key)
    assert state == "stopped"
    assert cached is None

    with pytest.raises(IdempotencyConflictError):
        await second.get_or_set(
            key,
            _fingerprint("different"),
            _must_not_compute,
            durable=True,
        )
    assert compute_count == 0
    _close_cache(second)


@pytest.mark.asyncio
async def test_normal_idempotency_never_creates_a_durable_row(tmp_path: Path):
    path = tmp_path / "normal-isolation.sqlite"
    cache = _IdempotencyCache(
        max_items=8,
        ttl_seconds=30,
        durable_path=str(path),
    )
    key = "normal-mode-key"
    calls = 0

    async def _compute():
        nonlocal calls
        calls += 1
        return (
            {"final_response": "normal answer", "completed": True},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    first = await cache.get_or_set(key, _fingerprint(), _compute)
    second = await cache.get_or_set(key, _fingerprint(), _compute)
    assert first == second
    assert calls == 1
    assert cache._durable.get(key) is None
    _close_cache(cache)


@pytest.mark.asyncio
async def test_unavailable_durable_store_does_not_break_normal_mode(
    tmp_path: Path,
):
    cache = _IdempotencyCache(
        durable_path=str(tmp_path),
    )
    assert cache.durable_ready is False
    assert cache._durable_error is not None

    calls = 0

    async def _compute():
        nonlocal calls
        calls += 1
        return ("normal", {"total_tokens": 0})

    result = await cache.get_or_set(
        "normal-key",
        _fingerprint(),
        _compute,
    )
    assert result[0] == "normal"
    assert calls == 1


def test_durable_files_never_contain_scope_or_prompt_plaintext(tmp_path: Path):
    path = tmp_path / "plaintext-free.sqlite"
    store = TrueMoADurableStore(str(path))
    secret_scope = "owner-user-secret-delivery"
    secret_prompt = "customer-private-prompt-must-not-persist"
    fingerprint = _fingerprint(secret_prompt)
    assert store.claim(
        secret_scope,
        fingerprint,
        kind="execution",
    ) == "missing"
    ledger = _running_ledger(epoch="3", wave_id="c" * 32)
    _finish_advisor(
        ledger,
        KIMI_ADVISOR_SLOT,
        input_tokens=2,
        output_tokens=1,
    )
    store.save_usage(
        secret_scope,
        fingerprint,
        ledger.to_dict(),
        state="running",
    )
    store.close()

    raw = b"".join(
        candidate.read_bytes()
        for candidate in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
            Path(f"{path}.lock"),
        )
        if candidate.exists()
    )
    assert secret_scope.encode() not in raw
    assert secret_prompt.encode() not in raw


def test_outcome_key_is_a_separate_fail_closed_preflight(tmp_path: Path):
    store = TrueMoADurableStore(str(tmp_path / "no-outcome-key.sqlite"))
    assert store.outcome_ready is False
    key = "missing-outcome-key"
    fingerprint = _fingerprint(key)
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    with pytest.raises(TrueMoAOutcomeUnavailableError):
        store.save_completed_outcome(
            key,
            fingerprint,
            _completed_ledger(),
            _completed_outcome(),
            binding=_outcome_binding(),
        )
    assert store.get(key)["state"] == "claimed"
    store.close()


def test_sealed_outcome_survives_ttl_until_owner_ack_and_leaves_no_plaintext(
    tmp_path: Path,
):
    path = tmp_path / "sealed.sqlite"
    key = "sealed-owner-delivery"
    fingerprint = _fingerprint(key)
    binding = _outcome_binding()
    text = "PRIVATE_OWNER_RESULT_不可明文落盘"
    usage = _completed_ledger()
    outcome = _completed_outcome(text)
    store = TrueMoADurableStore(
        str(path),
        outcome_keys=_OUTCOME_KEYS,
        active_outcome_key_id="test-v1",
        outcome_ttl_seconds=1,
    )
    assert store.outcome_ready is True
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    outcome_id = store.save_completed_outcome(
        key,
        fingerprint,
        usage,
        outcome,
        binding=binding,
    )
    recovered = store.recover_completed_outcome(key, binding=binding)
    assert recovered["finalResponse"] == text
    assert recovered["outcomeId"] == outcome_id
    assert recovered["retentionOverdue"] is False
    store.close()

    raw = b"".join(
        candidate.read_bytes()
        for candidate in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
            Path(f"{path}.lock"),
        )
        if candidate.exists()
    )
    assert text.encode() not in raw
    assert binding["siteId"].encode() not in raw
    assert binding["userId"].encode() not in raw

    store = TrueMoADurableStore(
        str(path),
        outcome_keys=_OUTCOME_KEYS,
        active_outcome_key_id="test-v1",
        outcome_ttl_seconds=1,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE true_moa_idempotency
            SET outcome_expires_at_ms = 1
            """
        )
    overdue = store.recover_completed_outcome(key, binding=binding)
    assert overdue["finalResponse"] == text
    assert overdue["retentionOverdue"] is True
    assert store.acknowledge_completed_outcome(
        key,
        binding=binding,
        outcome_id=outcome_id,
    ) == "acknowledged"
    record = store.get(key)
    assert record["state"] == "completed"
    assert record["outcomeState"] == "acked"
    assert record["usage"] == usage
    with pytest.raises(TrueMoAOutcomeUnavailableError):
        store.recover_completed_outcome(key, binding=binding)
    assert store.acknowledge_completed_outcome(
        key,
        binding=binding,
        outcome_id=outcome_id,
    ) == "already_acknowledged"
    foreign_binding = _outcome_binding(user_id="foreign-owner")
    with pytest.raises(TrueMoAOutcomeBindingError):
        store.acknowledge_completed_outcome(
            key,
            binding=foreign_binding,
            outcome_id=outcome_id,
        )
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT length(outcome_nonce), length(outcome_ciphertext),
                   outcome_receipt, length(usage_json)
            FROM true_moa_idempotency
            """
        ).fetchone()
    assert row[0] == 0
    assert row[1] == 0
    assert row[2] == outcome_id
    assert row[3] > 0
    store.close()


def test_fact_guard_outcome_round_trips_its_bound_verification(
    tmp_path: Path,
):
    binding = _outcome_binding(epoch="7")
    usage = _completed_ledger(epoch="7", wave_id="7" * 32)
    outcome = _completed_outcome(
        "事实核验后的唯一可见答案",
        binding=binding,
        fact_guard_required=True,
    )
    store = TrueMoADurableStore(
        str(tmp_path / "fact-outcome.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert store.claim(
        "fact-delivery",
        _fingerprint("fact-delivery"),
        kind="execution",
    ) == "missing"
    store.save_completed_outcome(
        "fact-delivery",
        _fingerprint("fact-delivery"),
        usage,
        outcome,
        binding=binding,
    )
    recovered = store.recover_completed_outcome(
        "fact-delivery",
        binding=binding,
    )
    assert recovered["factGuardRequired"] is True
    assert recovered["trustedVerification"] == (
        outcome["trustedVerification"]
    )
    store.close()


@pytest.mark.parametrize(
    "changed_field",
    [
        "siteId",
        "userId",
        "deliveryId",
        "messageId",
        "attempt",
        "requestFingerprint",
        "modeEpoch",
    ],
)
def test_sealed_outcome_rejects_cross_binding_replay(
    tmp_path: Path,
    changed_field: str,
):
    key = f"cross-binding-{changed_field}"
    fingerprint = _fingerprint(key)
    binding = _outcome_binding()
    store = TrueMoADurableStore(
        str(tmp_path / f"{changed_field}.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    store.save_completed_outcome(
        key,
        fingerprint,
        _completed_ledger(),
        _completed_outcome(),
        binding=binding,
    )
    changed = dict(binding)
    changed[changed_field] = {
        "siteId": "other-site",
        "userId": "other-user",
        "deliveryId": "xbd_" + ("b" * 40),
        "messageId": "message-2",
        "attempt": 2,
        "requestFingerprint": _fingerprint("other-request"),
        "modeEpoch": "2",
    }[changed_field]
    if changed_field == "userId":
        changed["datascopeFingerprint"] = TrustedIdentity(
            account_id=changed["userId"],
            data_scope="mystand",
            source="server_session",
        ).datascope_fingerprint
    with pytest.raises(TrueMoAOutcomeBindingError):
        store.recover_completed_outcome(key, binding=changed)
    store.close()


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("outcome_nonce", sqlite3.Binary(b"\x00" * 12)),
        ("outcome_ciphertext", sqlite3.Binary(b"\x00" * 32)),
        ("outcome_binding_digest", "0" * 64),
    ],
)
def test_sealed_outcome_rejects_envelope_tampering(
    tmp_path: Path,
    column: str,
    replacement,
):
    key = f"tamper-{column}"
    fingerprint = _fingerprint(key)
    binding = _outcome_binding()
    path = tmp_path / f"{column}.sqlite"
    store = TrueMoADurableStore(
        str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    store.save_completed_outcome(
        key,
        fingerprint,
        _completed_ledger(),
        _completed_outcome(),
        binding=binding,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"UPDATE true_moa_idempotency SET {column} = ?",
            (replacement,),
        )
    with pytest.raises(TrueMoAOutcomeBindingError):
        store.recover_completed_outcome(key, binding=binding)
    store.close()


def test_sealed_outcome_rejects_row_swap(tmp_path: Path):
    path = tmp_path / "row-swap.sqlite"
    store = TrueMoADurableStore(
        str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    rows = []
    for index, wave in enumerate(("8" * 32, "9" * 32), start=1):
        key = f"row-swap-{index}"
        fingerprint = _fingerprint(key)
        binding = _outcome_binding(
            delivery_id="xbd_" + (str(index) * 40),
            message_id=f"message-{index}",
            request_fingerprint=_fingerprint(f"request-{index}"),
        )
        assert store.claim(key, fingerprint, kind="execution") == "missing"
        store.save_completed_outcome(
            key,
            fingerprint,
            _completed_ledger(wave_id=wave),
            _completed_outcome(f"private row {index}"),
            binding=binding,
        )
        rows.append((key, fingerprint, binding))
    columns = (
        "outcome_key_id, outcome_nonce, outcome_ciphertext, "
        "outcome_receipt, outcome_binding_digest"
    )
    with sqlite3.connect(path) as connection:
        first = connection.execute(
            f"""
            SELECT {columns}
            FROM true_moa_idempotency WHERE fingerprint = ?
            """,
            (rows[0][1],),
        ).fetchone()
        second = connection.execute(
            f"""
            SELECT {columns}
            FROM true_moa_idempotency WHERE fingerprint = ?
            """,
            (rows[1][1],),
        ).fetchone()
        assignments = ", ".join(
            f"{name.strip()} = ?"
            for name in columns.split(",")
        )
        connection.execute(
            f"""
            UPDATE true_moa_idempotency SET {assignments}
            WHERE fingerprint = ?
            """,
            (*second, rows[0][1]),
        )
        connection.execute(
            f"""
            UPDATE true_moa_idempotency SET {assignments}
            WHERE fingerprint = ?
            """,
            (*first, rows[1][1]),
        )
    for key, _fingerprint_value, binding in rows:
        with pytest.raises(TrueMoAOutcomeBindingError):
            store.recover_completed_outcome(key, binding=binding)
    store.close()


def test_outcome_key_rotation_reads_old_key_and_fails_when_removed(
    tmp_path: Path,
):
    path = tmp_path / "rotation.sqlite"
    key = "rotation-delivery"
    fingerprint = _fingerprint(key)
    binding = _outcome_binding()
    old_key = b"\x21" * 32
    new_key = b"\x22" * 32
    store = TrueMoADurableStore(
        str(path),
        outcome_keys={"old": old_key},
        active_outcome_key_id="old",
    )
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    store.save_completed_outcome(
        key,
        fingerprint,
        _completed_ledger(),
        _completed_outcome("rotation-safe"),
        binding=binding,
    )
    store.close()

    store = TrueMoADurableStore(
        str(path),
        outcome_keys={"new": new_key, "old": old_key},
        active_outcome_key_id="new",
    )
    assert store.recover_completed_outcome(
        key,
        binding=binding,
    )["finalResponse"] == "rotation-safe"
    store.close()

    store = TrueMoADurableStore(
        str(path),
        outcome_keys={"new": new_key},
        active_outcome_key_id="new",
    )
    with pytest.raises(TrueMoAOutcomeUnavailableError):
        store.recover_completed_outcome(key, binding=binding)
    store.close()


def test_completed_callback_is_provisional_until_outcome_atomic_commit(
    tmp_path: Path,
):
    path = tmp_path / "completion-linearization.sqlite"
    key = "completion-linearization"
    fingerprint = _fingerprint(key)
    cache = _IdempotencyCache(
        durable_path=str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert cache._durable.claim(
        key,
        fingerprint,
        kind="execution",
    ) == "missing"
    cache.persist_usage(
        key,
        fingerprint,
        _completed_ledger(),
    )
    record = cache._durable.get(key)
    assert record["state"] == "running"
    assert record["usage"]["status"] == "completed"
    assert record["outcomeState"] == "none"
    _close_cache(cache)

    reopened = TrueMoADurableStore(
        str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    record = reopened.get(key)
    assert record["state"] == "interrupted"
    assert record["usage"]["status"] == "completed"
    assert record["outcomeState"] == "none"
    reopened.close()


def test_first_outcome_encryption_binds_transactionally_merged_usage(
    tmp_path: Path,
):
    path = tmp_path / "merged-aad.sqlite"
    key = "merged-aad"
    fingerprint = _fingerprint(key)
    binding = _outcome_binding()
    ledger = _running_ledger(wave_id="c" * 32)
    ledger.start_slot(KIMI_ADVISOR_SLOT)
    kimi_call = ledger.start_advisor_call(KIMI_ADVISOR_SLOT)
    provisional = ledger.to_dict()
    kimi_usage = {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
        "cached_input_tokens": 0,
    }
    ledger.finish_advisor_call(
        kimi_call,
        status="completed",
        usage=kimi_usage,
    )
    ledger.finish_slot(
        KIMI_ADVISOR_SLOT,
        status="completed",
        usage=kimi_usage,
    )
    _finish_advisor(
        ledger,
        DEEPSEEK_ADVISOR_SLOT,
        input_tokens=7,
        output_tokens=3,
    )
    ledger.start_slot(FINAL_EXECUTOR_SLOT)
    final_call = ledger.start_final_call("merged-aad-final")
    final_usage = {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "cached_input_tokens": 2,
    }
    ledger.finish_final_call(
        final_call,
        status="completed",
        usage=final_usage,
    )
    ledger.finish_slot(
        FINAL_EXECUTOR_SLOT,
        status="completed",
        usage=final_usage,
    )
    ledger.set_wave_status("completed")
    completed = ledger.to_dict()

    store = TrueMoADurableStore(
        str(path),
        outcome_keys=_OUTCOME_KEYS,
    )
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    store.save_usage(
        key,
        fingerprint,
        provisional,
        state="running",
    )
    outcome_id = store.save_completed_outcome(
        key,
        fingerprint,
        completed,
        _completed_outcome("merged usage survives recovery"),
        binding=binding,
    )
    recovered = store.recover_completed_outcome(key, binding=binding)
    assert recovered["outcomeId"] == outcome_id
    assert recovered["finalResponse"] == "merged usage survives recovery"
    assert store.get(key)["usage"] == completed

    drifted = json.loads(json.dumps(completed))
    drifted["calls"][-1]["costUsd"] = 0.123
    drifted["calls"][-1]["costStatus"] = "reported"
    drifted["calls"][-1]["costSource"] = "late-cost"
    with pytest.raises(
        TrueMoAOutcomeBindingError,
        match="usage cannot change",
    ):
        store.save_completed_outcome(
            key,
            fingerprint,
            drifted,
            _completed_outcome("merged usage survives recovery"),
            binding=binding,
        )
    assert store.recover_completed_outcome(
        key,
        binding=binding,
    )["outcomeId"] == outcome_id
    store.close()


def test_completed_outcome_rejects_oversized_text(tmp_path: Path):
    store = TrueMoADurableStore(
        str(tmp_path / "oversized.sqlite"),
        outcome_keys=_OUTCOME_KEYS,
    )
    key = "oversized"
    fingerprint = _fingerprint(key)
    assert store.claim(key, fingerprint, kind="execution") == "missing"
    oversized = "界" * ((TRUE_MOA_OUTCOME_MAX_TEXT_BYTES // 3) + 1)
    with pytest.raises(ValueError, match="invalid true MoA completed outcome"):
        store.save_completed_outcome(
            key,
            fingerprint,
            _completed_ledger(),
            _completed_outcome(oversized),
            binding=_outcome_binding(),
        )
    assert store.get(key)["state"] == "claimed"
    store.close()


def test_durable_projection_accepts_ten_total_calls_and_rejects_ninth_final():
    ledger = _running_ledger(wave_id="d" * 32)
    _finish_advisor(
        ledger,
        KIMI_ADVISOR_SLOT,
        input_tokens=1,
        output_tokens=1,
    )
    _finish_advisor(
        ledger,
        DEEPSEEK_ADVISOR_SLOT,
        input_tokens=1,
        output_tokens=1,
    )
    ledger.start_slot(FINAL_EXECUTOR_SLOT)
    usage = {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "cached_input_tokens": 0,
    }
    for index in range(TRUE_MOA_DURABLE_MAX_FINAL_CALLS):
        call_id = ledger.start_final_call(f"bounded-{index}")
        ledger.finish_final_call(
            call_id,
            status="completed",
            usage=usage,
        )
    ledger.finish_slot(
        FINAL_EXECUTOR_SLOT,
        status="completed",
        usage={
            "input_tokens": TRUE_MOA_DURABLE_MAX_FINAL_CALLS,
            "output_tokens": TRUE_MOA_DURABLE_MAX_FINAL_CALLS,
            "total_tokens": TRUE_MOA_DURABLE_MAX_FINAL_CALLS * 2,
            "cached_input_tokens": 0,
        },
    )
    ledger.set_wave_status("completed")
    projected = project_true_moa_usage(ledger.to_dict())
    assert len(projected["calls"]) == TRUE_MOA_DURABLE_MAX_CALLS

    forged = json.loads(json.dumps(projected))
    extra = dict(forged["calls"][-1])
    extra["callId"] = (
        f"{forged['waveId']}:{FINAL_EXECUTOR_SLOT.slot_id}:bounded-over"
    )
    forged["calls"].append(extra)
    with pytest.raises(ValueError, match="invalid true MoA durable calls"):
        project_true_moa_usage(forged)
