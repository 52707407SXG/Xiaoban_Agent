"""Tests for explicit /learn skill distillation."""

from agent.learn_prompt import _AUTHORING_STANDARDS, build_learn_prompt


class TestBuildLearnPrompt:
    def test_embeds_user_request_verbatim(self):
        req = "the My Stand upgrade workflow, focus on version backfill"
        prompt = build_learn_prompt(req)

        assert req in prompt

    def test_includes_authoring_standards(self):
        for req in ["", "https://example.com/docs", "what we just did"]:
            assert _AUTHORING_STANDARDS in build_learn_prompt(req)

    def test_requires_explicit_permission_boundary(self):
        prompt = build_learn_prompt("make a skill from today's work")

        assert "explicitly asked Xiaoban" in prompt
        assert "permission boundary" in prompt
        assert "do not learn unrelated private context" in prompt

    def test_instructs_skill_manage_save_or_stage(self):
        prompt = build_learn_prompt("learn the thing")

        assert "skill_manage(action=\"create\")" in prompt
        assert "created or staged for approval" in prompt

    def test_references_source_gathering_tools(self):
        prompt = build_learn_prompt("learn from a local folder and a URL")

        for tool in ("read_file", "search_files", "web_extract"):
            assert tool in prompt

    def test_separates_sources_from_requirements(self):
        prompt = build_learn_prompt(
            "https://api.example.com/docs focus on auth, skip deprecated endpoints"
        )
        low = prompt.lower()

        assert "focus on auth, skip deprecated endpoints" in prompt
        assert "sources" in low
        assert "requirements" in low
        assert "do not fetch only the first source" in low

    def test_empty_request_falls_back_to_conversation(self):
        prompt = build_learn_prompt("")

        assert "conversation" in prompt.lower()
        assert "skill_manage" in prompt

    def test_whitespace_only_matches_empty(self):
        assert build_learn_prompt("   \n  ") == build_learn_prompt("")

    def test_author_and_description_rules_are_xiaoban_specific(self):
        std = _AUTHORING_STANDARDS

        assert "author: always the literal value `Xiaoban`" in std
        assert "<=60 chars" in std
        assert "metadata.xiaoban.tags" in std


class TestLearnRegistryWiring:
    def test_learn_is_registered_and_resolves(self):
        from xiaoban_cli.commands import resolve_command

        cmd = resolve_command("learn")
        assert cmd is not None
        assert cmd.name == "learn"

    def test_learn_is_available_to_gateway(self):
        from xiaoban_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

        assert "learn" in GATEWAY_KNOWN_COMMANDS
        assert not resolve_command("learn").cli_only

    def test_learn_is_in_tools_and_skills_category(self):
        from xiaoban_cli.commands import resolve_command

        assert resolve_command("learn").category == "Tools & Skills"
