"""Deterministic checks for the fixed true-MoA provider boundary."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from xiaoban.trusted_runtime.true_moa import (
    DEEPSEEK_ADVISOR_SLOT,
    KIMI_ADVISOR_SLOT,
    TRUE_MOA_ADVISOR_INPUT_MAX_BYTES,
    TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS,
    TRUE_MOA_FINAL_INPUT_MAX_BYTES,
    TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS,
    AdvisorMessage,
    TrueMoACostCapError,
    TrueMoACancelController,
    enforce_true_moa_dispatch_budget,
)
from xiaoban.trusted_runtime import true_moa_providers as providers


def _messages():
    return (
        AdvisorMessage(role="assistant", content="相邻回答"),
        AdvisorMessage(role="user", content="当前问题"),
    )


def test_deepseek_advisor_is_one_fixed_toolless_no_retry_call(monkeypatch):
    captured = {"create_calls": 0, "dispatches": 0}

    class _Completions:
        def create(self, **kwargs):
            captured["create_calls"] += 1
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="独立建议",
                            tool_calls=None,
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                    prompt_cache_hit_tokens=4,
                ),
            )

    class _Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=_Completions())
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        providers,
        "_fixed_credentials",
        lambda *_args, **_kwargs: {
            "api_key": "fake-deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    monkeypatch.setattr("openai.OpenAI", _Client)

    result = providers.strict_advisor_call(
        slot=DEEPSEEK_ADVISOR_SLOT,
        messages=_messages(),
        tools=(),
        timeout_seconds=9,
        cancel_controller=TrueMoACancelController(),
        dispatch_callback=lambda: captured.__setitem__(
            "dispatches",
            captured["dispatches"] + 1,
        ),
    )

    assert captured["client"]["max_retries"] == 0
    assert captured["client"]["timeout"] == 9
    assert captured["request"]["model"] == "deepseek-v4-pro"
    assert captured["request"]["tools"] == []
    assert captured["request"]["stream"] is False
    assert (
        captured["request"]["max_tokens"]
        == TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS
    )
    assert captured["create_calls"] == 1
    assert captured["dispatches"] == 1
    assert result.content == "独立建议"
    assert result.usage["total_tokens"] == 18
    assert result.usage["prompt_cache_hit_tokens"] == 4


def test_kimi_advisor_is_one_fixed_toolless_no_retry_call(monkeypatch):
    captured = {
        "create_calls": 0,
        "stream_calls": 0,
        "dispatches": 0,
    }

    class _Stream:
        def __enter__(self):
            captured["stream_entered"] = True
            return self

        def get_final_message(self):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Kimi 独立建议")],
                usage=SimpleNamespace(
                    input_tokens=0,
                    output_tokens=1996,
                    cache_read_input_tokens=212,
                    cache_creation_input_tokens=0,
                ),
            )

        def __exit__(self, *_args):
            captured["stream_closed"] = True

    class _Messages:
        def create(self, **_kwargs):
            captured["create_calls"] += 1
            raise AssertionError("Kimi advisor must use the streaming helper")

        def stream(self, **kwargs):
            captured["stream_calls"] += 1
            captured["request"] = kwargs
            return _Stream()

    class _Client:
        def __init__(self):
            self.messages = _Messages()

        def close(self):
            captured["closed"] = True

    client = _Client()
    monkeypatch.setattr(
        providers,
        "_fixed_credentials",
        lambda *_args, **_kwargs: {
            "api_key": "fake-kimi-key",
            "base_url": "https://api.kimi.com/coding",
        },
    )

    def _build_client(api_key, base_url, **kwargs):
        captured["client"] = {
            "api_key": api_key,
            "base_url": base_url,
            **kwargs,
        }
        return client

    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        _build_client,
    )

    result = providers.strict_advisor_call(
        slot=KIMI_ADVISOR_SLOT,
        messages=_messages(),
        tools=(),
        timeout_seconds=8,
        cancel_controller=TrueMoACancelController(),
        dispatch_callback=lambda: captured.__setitem__(
            "dispatches",
            captured["dispatches"] + 1,
        ),
    )

    assert captured["client"]["max_retries"] == 0
    assert captured["client"]["timeout"] == 8
    assert captured["request"]["model"] == "k3"
    assert captured["request"]["tools"] == []
    assert (
        captured["request"]["max_tokens"]
        == TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS
    )
    assert captured["create_calls"] == 0
    assert captured["stream_calls"] == 1
    assert captured["dispatches"] == 1
    assert result.content == "Kimi 独立建议"
    assert result.usage == {
        "input_tokens": 212,
        "output_tokens": 1996,
        "total_tokens": 2208,
        "cache_read_input_tokens": 212,
    }
    assert captured["stream_entered"] is True
    assert captured["stream_closed"] is True
    assert captured["closed"] is True


def test_kimi_stop_drains_final_usage_without_reading_late_text(monkeypatch):
    stream_started = threading.Event()
    release_response = threading.Event()
    controller = TrueMoACancelController()
    outcome = []
    captured = {
        "dispatches": 0,
        "content_reads": 0,
        "stream_closed": 0,
        "client_closed": 0,
    }

    class _Response:
        usage = SimpleNamespace(
            input_tokens=17,
            output_tokens=6,
            cache_read_input_tokens=2,
            cache_creation_input_tokens=3,
        )

        @property
        def content(self):
            captured["content_reads"] += 1
            raise AssertionError("late Kimi text must not be read after stop")

    class _Stream:
        def __enter__(self):
            return self

        def get_final_message(self):
            stream_started.set()
            assert release_response.wait(1)
            return _Response()

        def __exit__(self, *_args):
            captured["stream_closed"] += 1

    class _Messages:
        def stream(self, **_kwargs):
            return _Stream()

    class _Client:
        def __init__(self):
            self.messages = _Messages()

        def close(self):
            captured["client_closed"] += 1

    monkeypatch.setattr(
        providers,
        "_fixed_credentials",
        lambda *_args, **_kwargs: {
            "api_key": "fake-kimi-key",
            "base_url": "https://api.kimi.com/coding",
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda *_args, **_kwargs: _Client(),
    )

    def _call():
        try:
            providers.strict_advisor_call(
                slot=KIMI_ADVISOR_SLOT,
                messages=_messages(),
                tools=(),
                timeout_seconds=1,
                cancel_controller=controller,
                dispatch_callback=lambda: captured.__setitem__(
                    "dispatches",
                    captured["dispatches"] + 1,
                ),
            )
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    assert stream_started.wait(1)
    controller.cancel()
    release_response.set()
    worker.join(1)

    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], providers.StrictAdvisorCancelled)
    assert outcome[0].usage == {
        "input_tokens": 22,
        "output_tokens": 6,
        "total_tokens": 28,
        "cache_read_input_tokens": 2,
    }
    assert captured == {
        "dispatches": 1,
        "content_reads": 0,
        "stream_closed": 1,
        "client_closed": 1,
    }


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("kimi-coding", "https://proxy.example/v1"),
        ("deepseek", "https://proxy.example/v1"),
    ],
)
def test_fixed_preset_rejects_endpoint_drift(monkeypatch, provider, base_url):
    monkeypatch.setattr(
        "xiaoban_cli.auth.resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": _provider,
            "api_key": "fake-key",
            "base_url": base_url,
        },
    )
    expected = (
        ("https://api.kimi.com", "/coding")
        if provider == "kimi-coding"
        else ("https://api.deepseek.com", "/v1")
    )
    with pytest.raises(
        providers.StrictAdvisorProviderError,
        match="fixed_endpoint_mismatch",
    ):
        providers._fixed_credentials(
            provider,
            expected_origin=expected[0],
            expected_path=expected[1],
        )


def test_running_cancel_fences_output_but_waits_for_usage_receipt(monkeypatch):
    request_started = threading.Event()
    release_response = threading.Event()
    controller = TrueMoACancelController()
    outcome = []
    dispatches = []
    closed = []

    class _Completions:
        def create(self, **_kwargs):
            request_started.set()
            assert release_response.wait(1)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="late", tool_calls=None)
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=1),
                ),
            )

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        providers,
        "_fixed_credentials",
        lambda *_args, **_kwargs: {
            "api_key": "fake-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    monkeypatch.setattr("openai.OpenAI", _Client)

    def _call():
        try:
            providers.strict_advisor_call(
                slot=DEEPSEEK_ADVISOR_SLOT,
                messages=_messages(),
                tools=(),
                timeout_seconds=1,
                cancel_controller=controller,
                dispatch_callback=lambda: dispatches.append("deepseek"),
            )
        except Exception as exc:
            outcome.append(exc)

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    assert request_started.wait(1)
    controller.cancel()
    thread.join(0.05)
    assert thread.is_alive(), (
        "an already-dispatched request must retain its exact usage receipt"
    )
    release_response.set()
    thread.join(1)

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert dispatches == ["deepseek"]
    assert isinstance(outcome[0], providers.StrictAdvisorCancelled)
    assert outcome[0].usage == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 1},
    }
    assert closed == [True]


def test_cancel_winning_atomic_dispatch_gate_means_zero_provider_calls(
    monkeypatch,
):
    controller = TrueMoACancelController()
    create_calls = 0
    dispatch_calls = 0

    class _Completions:
        def create(self, **_kwargs):
            nonlocal create_calls
            create_calls += 1
            raise AssertionError("cancelled advisor reached provider dispatch")

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

        def close(self):
            pass

    monkeypatch.setattr(
        providers,
        "_fixed_credentials",
        lambda *_args, **_kwargs: {
            "api_key": "fake-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    monkeypatch.setattr("openai.OpenAI", _Client)
    original_gate = controller.try_begin_dispatch

    def _cancel_then_claim(key):
        controller.cancel()
        return original_gate(key)

    def _record_dispatch():
        nonlocal dispatch_calls
        dispatch_calls += 1

    monkeypatch.setattr(controller, "try_begin_dispatch", _cancel_then_claim)

    with pytest.raises(
        providers.StrictAdvisorCancelled,
        match="advisor_cancelled_before_dispatch",
    ):
        providers.strict_advisor_call(
            slot=DEEPSEEK_ADVISOR_SLOT,
            messages=_messages(),
            tools=(),
            timeout_seconds=1,
            cancel_controller=controller,
            dispatch_callback=_record_dispatch,
        )

    assert create_calls == 0
    assert dispatch_calls == 0


def test_cancel_during_durable_reservation_never_reaches_provider(
    monkeypatch,
):
    controller = TrueMoACancelController()
    callback_started = threading.Event()
    release_callback = threading.Event()
    create_calls = []
    outcome = []

    class _Completions:
        def create(self, **_kwargs):
            create_calls.append("deepseek")
            raise AssertionError("late durable callback reached provider")

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

        def close(self):
            pass

    monkeypatch.setattr(
        providers,
        "_fixed_credentials",
        lambda *_args, **_kwargs: {
            "api_key": "fake-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    monkeypatch.setattr("openai.OpenAI", _Client)

    def _reserve():
        callback_started.set()
        assert release_callback.wait(1)

    def _call():
        try:
            providers.strict_advisor_call(
                slot=DEEPSEEK_ADVISOR_SLOT,
                messages=_messages(),
                tools=(),
                timeout_seconds=1,
                cancel_controller=controller,
                dispatch_callback=_reserve,
            )
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    assert callback_started.wait(1)
    assert controller.fail() is True
    release_callback.set()
    worker.join(1)

    assert not worker.is_alive()
    assert create_calls == []
    assert len(outcome) == 1
    assert isinstance(outcome[0], providers.StrictAdvisorCancelled)


@pytest.mark.parametrize(
    "slot",
    [KIMI_ADVISOR_SLOT, DEEPSEEK_ADVISOR_SLOT],
)
def test_advisor_input_cap_rejects_before_credentials_or_dispatch(
    monkeypatch,
    slot,
):
    credentials = []
    dispatches = []
    monkeypatch.setattr(
        providers,
        "_fixed_credentials",
        lambda *_args, **_kwargs: credentials.append("resolved"),
    )
    oversized = (
        AdvisorMessage(
            role="user",
            content="界" * TRUE_MOA_ADVISOR_INPUT_MAX_BYTES,
        ),
    )

    with pytest.raises(
        TrueMoACostCapError,
        match="true_moa_input_byte_cap_exceeded",
    ):
        providers.strict_advisor_call(
            slot=slot,
            messages=oversized,
            tools=(),
            timeout_seconds=1,
            cancel_controller=TrueMoACancelController(),
            dispatch_callback=lambda: dispatches.append("dispatched"),
        )

    assert credentials == []
    assert dispatches == []


@pytest.mark.parametrize(
    ("role", "input_limit", "output_limit"),
    [
        (
            "advisor",
            TRUE_MOA_ADVISOR_INPUT_MAX_BYTES,
            TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS,
        ),
        (
            "final_executor",
            TRUE_MOA_FINAL_INPUT_MAX_BYTES,
            TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS,
        ),
    ],
)
def test_fixed_cost_cap_accepts_exact_byte_boundary_and_rejects_next_byte(
    role,
    input_limit,
    output_limit,
):
    payload = {
        "max_tokens": output_limit,
        "messages": [{"role": "user", "content": ""}],
    }
    base_size = enforce_true_moa_dispatch_budget(
        role=role,
        payload=payload,
    )
    payload["messages"][0]["content"] = "x" * (input_limit - base_size)
    assert enforce_true_moa_dispatch_budget(
        role=role,
        payload=payload,
    ) == input_limit
    payload["messages"][0]["content"] += "x"
    with pytest.raises(
        TrueMoACostCapError,
        match="true_moa_input_byte_cap_exceeded",
    ):
        enforce_true_moa_dispatch_budget(
            role=role,
            payload=payload,
        )


@pytest.mark.parametrize(
    ("role", "output_limit"),
    [
        ("advisor", TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS),
        ("final_executor", TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS),
    ],
)
def test_fixed_cost_cap_requires_explicit_bounded_output_tokens(
    role,
    output_limit,
):
    assert enforce_true_moa_dispatch_budget(
        role=role,
        payload={"messages": [], "max_tokens": output_limit},
    ) > 0
    with pytest.raises(
        TrueMoACostCapError,
        match="true_moa_output_token_cap_exceeded",
    ):
        enforce_true_moa_dispatch_budget(
            role=role,
            payload={"messages": [], "max_tokens": output_limit + 1},
        )
    with pytest.raises(
        TrueMoACostCapError,
        match="true_moa_output_token_cap_exceeded",
    ):
        enforce_true_moa_dispatch_budget(
            role=role,
            payload={"messages": []},
        )
