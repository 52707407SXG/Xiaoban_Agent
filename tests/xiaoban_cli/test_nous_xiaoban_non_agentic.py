"""Tests for the Nous-Xiaoban-3/4 non-agentic warning detector.

Prior to this check, the warning fired on any model whose name contained
``"xiaoban"`` anywhere (case-insensitive). That false-positived on unrelated
local Modelfiles such as ``xiaoban-brain:qwen3-14b-ctx16k`` — a tool-capable
Qwen3 wrapper that happens to live under the "xiaoban" tag namespace.

``is_nous_xiaoban_non_agentic`` should only match the actual Nous Research
Xiaoban-3 / Xiaoban-4 chat family.
"""

from __future__ import annotations

import pytest

from xiaoban_cli.model_switch import (
    _XIAOBAN_MODEL_WARNING,
    _check_xiaoban_model_warning,
    is_nous_xiaoban_non_agentic,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "NousResearch/Xiaoban-3-Llama-3.1-70B",
        "NousResearch/Xiaoban-3-Llama-3.1-405B",
        "xiaoban-3",
        "Xiaoban-3",
        "xiaoban-4",
        "xiaoban-4-405b",
        "xiaoban_4_70b",
        "openrouter/xiaoban3:70b",
        "openrouter/nousresearch/xiaoban-4-405b",
        "NousResearch/Xiaoban3",
        "xiaoban-3.1",
    ],
)
def test_matches_real_nous_xiaoban_chat_models(model_name: str) -> None:
    assert is_nous_xiaoban_non_agentic(model_name), (
        f"expected {model_name!r} to be flagged as Nous Xiaoban 3/4"
    )
    assert _check_xiaoban_model_warning(model_name) == _XIAOBAN_MODEL_WARNING


@pytest.mark.parametrize(
    "model_name",
    [
        # Kyle's local Modelfile — qwen3:14b under a custom tag
        "xiaoban-brain:qwen3-14b-ctx16k",
        "xiaoban-brain:qwen3-14b-ctx32k",
        "xiaoban-honcho:qwen3-8b-ctx8k",
        # Plain unrelated models
        "qwen3:14b",
        "qwen3-coder:30b",
        "qwen2.5:14b",
        "claude-opus-4-6",
        "anthropic/claude-sonnet-4.5",
        "gpt-5",
        "openai/gpt-4o",
        "google/gemini-2.5-flash",
        "deepseek-chat",
        # Non-chat Xiaoban models we don't warn about
        "xiaoban-llm-2",
        "xiaoban2-pro",
        "nous-xiaoban-2-mistral",
        # Edge cases
        "",
        "xiaoban",  # bare "xiaoban" isn't the 3/4 family
        "xiaoban-brain",
        "brain-xiaoban-3-impostor",  # "3" not preceded by /: boundary
    ],
)
def test_does_not_match_unrelated_models(model_name: str) -> None:
    assert not is_nous_xiaoban_non_agentic(model_name), (
        f"expected {model_name!r} NOT to be flagged as Nous Xiaoban 3/4"
    )
    assert _check_xiaoban_model_warning(model_name) == ""


def test_none_like_inputs_are_safe() -> None:
    assert is_nous_xiaoban_non_agentic("") is False
    # Defensive: the helper shouldn't crash on None-ish falsy input either.
    assert _check_xiaoban_model_warning("") == ""
