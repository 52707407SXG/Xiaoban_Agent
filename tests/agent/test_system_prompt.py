"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo
from datetime import datetime

from agent.system_prompt import build_system_prompt_parts, _build_temporal_context_block


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False, context_length=None):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


class TestTemporalContextBlock:
    def test_uses_user_local_time_and_raw_schedule_guidance(self):
        block = _build_temporal_context_block(
            datetime(2026, 6, 24, 22, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
        )

        assert "Current user default city/timezone: 成都/北京时间" in block
        assert "Current user local time: 2026-06-24 22:10:00 CST+0800" in block
        assert "Today window: 2026-06-24 00:00:00 CST+0800 to 2026-06-25 00:00:00 CST+0800" in block
        assert "Tonight window: 2026-06-24 18:00:00 CST+0800 to 2026-06-25 12:00:00 CST+0800" in block
        assert "今天, 最新, 当下, 现在, 最近, 后面, 接下来, 下一场, 今晚" in block
        assert "use raw source rows" in block
        assert "ET/EDT/EST/PT/CT/MT/BST/UTC/local/venue-local" in block


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestXiaobanOperatingPolicy:
    def test_xiaoban_operating_policy_is_stable_prompt_layer(self):
        stable = _stable_prompt(_make_agent())

        assert "# Xiaoban operating policy" in stable
        assert "Uploaded files" in stable
        assert "Yuan Laoshi" in stable
        assert "Did I answer the newest user message" in stable
        assert "Visual and media evidence" in stable
        assert "Do not invent scenes, people, actions" in stable
        assert "Live sports and current scores" in stable
        assert "search result snippet gives a fresh score/status" in stable
        assert "Do not bury a usable live/final score behind vague uncertainty" in stable
        assert "Clean web research workflow" in stable
        assert "internal evidence ledger" in stable
        assert "Do not present a stale snippet" in stable
        assert "Link and article reading" in stable
        assert "mp.weixin.qq.com are especially tool-first" in stable
        assert "use mystand_parse when available" in stable
        assert "If no tool was called or the tool result is empty" in stable
        assert "# Xiaoban agentic workflow principles" in stable
        assert "latest user message and this-turn evidence override old topic drift" in stable
        assert "anchor the reasoning to the user's default Beijing time" in stable

    def test_deepseek_gets_xiaoban_execution_discipline(self):
        agent = _make_agent(
            valid_tool_names=["web_search"],
            model="deepseek-v4-pro",
            _tool_use_enforcement="auto",
        )

        stable = _stable_prompt(agent)

        assert "# Tool-use enforcement" in stable
        assert "# Xiaoban execution discipline for DeepSeek-class models" in stable
        assert "Do not fabricate results" in stable
        assert "Uploaded-file, current-time, and explicit correction turns are hard pivots" in stable

    def test_non_deepseek_does_not_get_deepseek_specific_block(self):
        agent = _make_agent(
            valid_tool_names=["web_search"],
            model="claude-sonnet-4",
            _tool_use_enforcement="auto",
        )

        stable = _stable_prompt(agent)

        assert "# Xiaoban operating policy" in stable
        assert "# Xiaoban execution discipline for DeepSeek-class models" not in stable


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)
