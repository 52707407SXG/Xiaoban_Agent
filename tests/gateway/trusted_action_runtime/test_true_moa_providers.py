"""Deterministic checks for the fixed true-MoA provider boundary."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from xiaoban.trusted_runtime.true_moa import (
    DEEPSEEK_ADVISOR_SLOT,
    KIMI_ADVISOR_SLOT,
    AdvisorMessage,
    TrueMoACancelController,
)
from xiaoban.trusted_runtime import true_moa_providers as providers


def _messages():
    return (
        AdvisorMessage(role="assistant", content="相邻回答"),
        AdvisorMessage(role="user", content="当前问题"),
    )


def test_deepseek_advisor_is_one_fixed_toolless_no_retry_call(monkeypatch):
    captured = {"create_calls": 0}

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
    )

    assert captured["client"]["max_retries"] == 0
    assert captured["client"]["timeout"] == 9
    assert captured["request"]["model"] == "deepseek-v4-pro"
    assert captured["request"]["tools"] == []
    assert captured["request"]["stream"] is False
    assert captured["create_calls"] == 1
    assert result.content == "独立建议"
    assert result.usage["total_tokens"] == 18


def test_kimi_advisor_is_one_fixed_toolless_no_retry_call(monkeypatch):
    captured = {"create_calls": 0}

    class _Messages:
        def create(self, **kwargs):
            captured["create_calls"] += 1
            captured["request"] = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Kimi 独立建议")],
                usage=SimpleNamespace(input_tokens=13, output_tokens=5),
            )

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
    )

    assert captured["client"]["max_retries"] == 0
    assert captured["client"]["timeout"] == 8
    assert captured["request"]["model"] == "k3"
    assert captured["request"]["tools"] == []
    assert captured["create_calls"] == 1
    assert result.content == "Kimi 独立建议"
    assert result.usage == {"input_tokens": 13, "output_tokens": 5}
    assert captured["closed"] is True


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


def test_running_cancel_invokes_socket_abort_before_caller_exits(monkeypatch):
    request_started = threading.Event()
    socket_aborted = threading.Event()
    controller = TrueMoACancelController()
    outcome = []

    class _Completions:
        def create(self, **_kwargs):
            request_started.set()
            assert socket_aborted.wait(1)
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
                ),
            )

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
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.force_close_tcp_sockets",
        lambda _client: (socket_aborted.set() or 1),
    )

    def _call():
        try:
            providers.strict_advisor_call(
                slot=DEEPSEEK_ADVISOR_SLOT,
                messages=_messages(),
                tools=(),
                timeout_seconds=1,
                cancel_controller=controller,
            )
        except Exception as exc:
            outcome.append(exc)

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    assert request_started.wait(1)
    controller.cancel()
    thread.join(1)

    assert not thread.is_alive()
    assert socket_aborted.is_set()
    assert len(outcome) == 1
    assert isinstance(outcome[0], providers.StrictAdvisorCancelled)
    assert outcome[0].usage == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


def test_cancel_winning_atomic_dispatch_gate_means_zero_provider_calls(
    monkeypatch,
):
    controller = TrueMoACancelController()
    create_calls = 0

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
        )

    assert create_calls == 0
