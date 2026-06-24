from types import SimpleNamespace
from unittest.mock import patch

from xiaoban_cli.config import (
    format_managed_message,
    get_managed_system,
    recommended_update_command,
)
from xiaoban_cli.main import cmd_update
from tools.skills_hub import OptionalSkillSource


def test_get_managed_system_homebrew(monkeypatch):
    monkeypatch.setenv("XIAOBAN_MANAGED", "homebrew")

    assert get_managed_system() == "Homebrew"
    assert recommended_update_command() == "brew upgrade xiaoban-agent"


def test_format_managed_message_homebrew(monkeypatch):
    monkeypatch.setenv("XIAOBAN_MANAGED", "homebrew")

    message = format_managed_message("update Xiaoban-Agent")

    assert "managed by Homebrew" in message
    assert "brew upgrade xiaoban-agent" in message


def test_recommended_update_command_defaults_to_xiaoban_update(monkeypatch):
    monkeypatch.delenv("XIAOBAN_MANAGED", raising=False)

    # Also short-circuit the .managed marker path — CI runners may have an
    # ambient ~/.xiaoban/.managed if a prior test left XIAOBAN_HOME pointing
    # somewhere with that marker, which would make get_managed_update_command()
    # return "Update your Nix flake input ..." instead of falling through to
    # detect_install_method().
    with patch("xiaoban_cli.config.get_managed_update_command", return_value=None), \
         patch("xiaoban_cli.config.detect_install_method", return_value="git"):
        assert recommended_update_command() == "xiaoban update"


def test_cmd_update_blocks_managed_homebrew(monkeypatch, capsys):
    monkeypatch.setenv("XIAOBAN_MANAGED", "homebrew")

    with patch("xiaoban_cli.main.subprocess.run") as mock_run:
        cmd_update(SimpleNamespace())

    assert not mock_run.called
    captured = capsys.readouterr()
    assert "managed by Homebrew" in captured.err
    assert "brew upgrade xiaoban-agent" in captured.err


def test_optional_skill_source_honors_env_override(monkeypatch, tmp_path):
    optional_dir = tmp_path / "optional-skills"
    optional_dir.mkdir()
    monkeypatch.setenv("XIAOBAN_OPTIONAL_SKILLS", str(optional_dir))

    source = OptionalSkillSource()

    assert source._optional_dir == optional_dir
