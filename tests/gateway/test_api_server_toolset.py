"""Tests for xiaoban-api-server toolset and API server tool availability."""
from unittest.mock import patch, MagicMock

import pytest


from toolsets import resolve_toolset, get_toolset, validate_toolset


class TestReadOnlyFileToolset:
    def test_file_readonly_contains_no_write_tools(self):
        tools = resolve_toolset("file_readonly")

        assert "read_file" in tools
        assert "search_files" in tools
        assert "write_file" not in tools
        assert "patch" not in tools
        assert "terminal" not in tools
        assert "execute_code" not in tools


class TestMystandAuthorizationToolset:
    def test_toolset_contains_only_server_enforced_bridge(self):
        assert resolve_toolset("mystand_authorization") == [
            "mystand_authorization"
        ]
        assert validate_toolset("mystand_authorization")


class TestXiaobanApiServerToolset:
    """Tests for the xiaoban-api-server toolset definition."""

    def test_toolset_exists(self):
        ts = get_toolset("xiaoban-api-server")
        assert ts is not None

    def test_toolset_validates(self):
        assert validate_toolset("xiaoban-api-server")

    def test_toolset_includes_web_tools(self):
        tools = resolve_toolset("xiaoban-api-server")
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_toolset_includes_core_tools(self):
        tools = resolve_toolset("xiaoban-api-server")
        expected = [
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze", "image_generate",
            "execute_code", "delegate_task",
            "todo", "memory", "session_search", "cronjob",
        ]
        for tool in expected:
            assert tool in tools, f"Missing expected tool: {tool}"

    def test_toolset_includes_browser_tools(self):
        tools = resolve_toolset("xiaoban-api-server")
        for tool in ["browser_navigate", "browser_snapshot", "browser_click",
                      "browser_type", "browser_scroll", "browser_back",
                      "browser_press"]:
            assert tool in tools, f"Missing browser tool: {tool}"

    def test_toolset_includes_homeassistant_tools(self):
        tools = resolve_toolset("xiaoban-api-server")
        for tool in ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"]:
            assert tool in tools, f"Missing HA tool: {tool}"

    def test_toolset_excludes_clarify(self):
        tools = resolve_toolset("xiaoban-api-server")
        assert "clarify" not in tools

    def test_toolset_excludes_send_message(self):
        tools = resolve_toolset("xiaoban-api-server")
        assert "send_message" not in tools

    def test_toolset_excludes_text_to_speech(self):
        tools = resolve_toolset("xiaoban-api-server")
        assert "text_to_speech" not in tools


class TestApiServerPlatformConfig:
    def test_platforms_dict_includes_api_server(self):
        from xiaoban_cli.tools_config import PLATFORMS
        assert "api_server" in PLATFORMS
        assert PLATFORMS["api_server"]["default_toolset"] == "xiaoban-api-server"


class TestApiServerAdapterToolset:
    def test_header_value_falls_back_to_case_insensitive_dict_lookup(self):
        from gateway.platforms.api_server import APIServerAdapter

        headers = {"x-xiaoban-toolset-policy": "mystand-broker-basic"}

        assert APIServerAdapter._header_value(headers, "X-Xiaoban-Toolset-Policy") == "mystand-broker-basic"

    def test_mystand_policies_are_explicit_and_exclude_server_mutation_tools(self):
        from gateway.platforms.api_server import APIServerAdapter

        basic = APIServerAdapter._toolsets_for_request_policy("mystand-broker-basic")
        research = APIServerAdapter._toolsets_for_request_policy("mystand-broker-research")
        owner = APIServerAdapter._toolsets_for_request_policy("mystand-owner")
        owner_research = APIServerAdapter._toolsets_for_request_policy("mystand-owner-research")

        assert basic == ["web", "mystand_parser", "mystand_authorization"]
        assert research == [
            "web",
            "mystand_parser",
            "mystand_authorization",
            "delegation",
        ]
        assert owner == basic
        assert owner_research == research
        for toolsets in (basic, research, owner, owner_research):
            assert "terminal" not in toolsets
            assert "file" not in toolsets
            assert "file_readonly" not in toolsets
            assert "skills" not in toolsets
            assert "memory" not in toolsets
            assert "session_search" not in toolsets

    @pytest.mark.parametrize("policy", ["", "mystand-owner-typo", "unknown", "  "])
    def test_present_unknown_or_blank_mystand_policy_is_rejected(self, policy):
        from gateway.platforms.api_server import APIServerAdapter, InvalidToolsetPolicy

        with pytest.raises(InvalidToolsetPolicy):
            APIServerAdapter._toolsets_for_request_headers(
                {"x-xiaoban-toolset-policy": policy}
            )

    def test_missing_policy_header_preserves_non_mystand_api_configuration(self):
        from gateway.platforms.api_server import APIServerAdapter

        assert APIServerAdapter._toolsets_for_request_headers({}) is None

    def test_mystand_user_header_without_policy_is_rejected(self):
        from gateway.platforms.api_server import APIServerAdapter, InvalidToolsetPolicy

        with pytest.raises(InvalidToolsetPolicy):
            APIServerAdapter._toolsets_for_request_headers(
                {"X-Xiaoban-User-Id": "ZYJ001"}
            )

    @pytest.mark.parametrize(
        "policy",
        [
            "mystand-broker-basic",
            "mystand-broker-research",
            "mystand-owner",
            "mystand-owner-research",
        ],
    )
    def test_resolved_mystand_tools_exclude_dangerous_capabilities(self, policy):
        from gateway.platforms.api_server import APIServerAdapter
        from toolsets import resolve_multiple_toolsets

        toolsets = APIServerAdapter._toolsets_for_request_headers(
            {
                "X-Xiaoban-Toolset-Policy": policy,
                "X-Xiaoban-User-Id": "ZYJ001",
            }
        )
        resolved = set(resolve_multiple_toolsets(toolsets))
        forbidden = {
            "terminal",
            "process",
            "read_terminal",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "execute_code",
            "cronjob",
            "computer_use",
            "skill_manage",
            "memory",
            "session_search",
        }
        assert resolved.isdisjoint(forbidden)

    def test_mystand_policy_requires_authenticated_user_identity(self):
        from gateway.platforms.api_server import APIServerAdapter, InvalidToolsetPolicy

        with pytest.raises(InvalidToolsetPolicy):
            APIServerAdapter._toolsets_for_request_headers(
                {"X-Xiaoban-Toolset-Policy": "mystand-owner"}
            )

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reads_config_toolsets(self):
        """API server resolves toolsets from config like all other platforms."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # No platform_toolsets override — should fall back to xiaoban-api-server default
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert isinstance(toolsets, list)
            assert len(toolsets) > 0
            assert call_kwargs.kwargs.get("platform") == "api_server"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_respects_config_override(self):
        """User can override API server toolsets via platform_toolsets in config.yaml."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # User overrides with just web and terminal
            mock_config.return_value = {
                "platform_toolsets": {"api_server": ["web", "terminal"]}
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert sorted(toolsets) == ["terminal", "web"]

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_respects_mystand_basic_override(self):
        """My Stand broker accounts get web/parser plus the AUTH-gated bridge."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            mock_config.return_value = {
                "platform_toolsets": {"api_server": ["xiaoban-api-server"]}
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent(
                enabled_toolsets_override=[
                    "web",
                    "mystand_parser",
                    "mystand_authorization",
                ]
            )

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            assert call_kwargs.kwargs.get("enabled_toolsets") == [
                "mystand_authorization",
                "mystand_parser",
                "web",
            ]
