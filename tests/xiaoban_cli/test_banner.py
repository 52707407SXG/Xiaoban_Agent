"""Tests for the Xiaoban startup banner."""

from unittest.mock import patch

from rich.console import Console

import xiaoban_cli.banner as banner


def test_display_toolset_name_strips_legacy_suffix():
    assert banner._display_toolset_name("homeassistant_tools") == "homeassistant"
    assert banner._display_toolset_name("honcho_tools") == "honcho"
    assert banner._display_toolset_name("web_tools") == "web"


def test_display_toolset_name_preserves_clean_names():
    assert banner._display_toolset_name("browser") == "browser"
    assert banner._display_toolset_name("file") == "file"
    assert banner._display_toolset_name("terminal") == "terminal"


def test_display_toolset_name_handles_empty():
    assert banner._display_toolset_name("") == "unknown"
    assert banner._display_toolset_name(None) == "unknown"


def test_build_welcome_banner_uses_claude_style_xiaoban_card():
    with (
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(banner, "get_latest_release_tag", return_value=None),
    ):
        console = Console(record=True, force_terminal=False, color_system=None, width=160)
        banner.build_welcome_banner(
            console=console,
            model="deepseek/deepseek-v4-pro",
            cwd="/tmp/project",
            tools=[
                {"function": {"name": "web_search"}},
                {"function": {"name": "read_file"}},
            ],
            get_toolset_for_tool=lambda name: {
                "web_search": "web_tools",
                "read_file": "file",
            }.get(name),
        )

    output = console.export_text()
    assert f"Xiaoban v{banner.VERSION}" in output
    assert "Welcome back!" in output
    assert "My Stand" in output
    assert "██║╚████╔╝██║" in output
    assert "deepseek-v4-pro · API" in output
    assert "/tmp/project" in output
    assert "Tips for getting started" in output
    assert "Recent activity" in output
    assert "No recent activity" in output
    assert "Available Tools" not in output
    assert "Available Skills" not in output
    assert "homeassistant" not in output
    assert "web_tools" not in output
    assert ("Xiao" + " ban") not in output
    assert "Xiaoban-Agent" not in output
    assert output.count("Xiaoban") <= 2


def test_build_welcome_banner_title_is_hyperlinked_to_release():
    """Panel title is wrapped in an OSC-8 hyperlink to the GitHub release."""
    import io

    banner._latest_release_cache = None
    tag_url = (
        "v2026.4.23",
        "https://github.com/52707407SXG/Xiaoban_Agent/releases/tag/v2026.4.23",
    )

    buf = io.StringIO()
    with (
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(banner, "get_latest_release_tag", return_value=tag_url),
    ):
        console = Console(file=buf, force_terminal=True, color_system="truecolor", width=160)
        banner.build_welcome_banner(
            console=console,
            model="x",
            cwd="/tmp",
            session_id="abc123",
            tools=[{"function": {"name": "read_file"}}],
            get_toolset_for_tool=lambda n: "file",
        )

    raw = buf.getvalue()
    assert f"Xiaoban v{banner.VERSION}" in raw, "Version label missing from title"
    assert "\x1b]8;" in raw, "OSC-8 hyperlink not emitted"
    assert "releases/tag/v2026.4.23" in raw, "Release URL missing from banner output"


def test_build_welcome_banner_title_falls_back_when_no_tag():
    """Without a resolvable tag, the panel title renders as plain text."""
    import io

    banner._latest_release_cache = None
    buf = io.StringIO()
    with (
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(banner, "get_latest_release_tag", return_value=None),
    ):
        console = Console(file=buf, force_terminal=True, color_system="truecolor", width=160)
        banner.build_welcome_banner(
            console=console,
            model="x",
            cwd="/tmp",
            session_id="abc123",
            tools=[{"function": {"name": "read_file"}}],
            get_toolset_for_tool=lambda n: "file",
        )

    raw = buf.getvalue()
    assert f"Xiaoban v{banner.VERSION}" in raw, "Version label missing from title"
    assert "\x1b]8;" not in raw, "OSC-8 hyperlink should not be emitted without a tag"
