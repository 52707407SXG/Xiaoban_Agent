from __future__ import annotations

from xiaoban_constants import reset_xiaoban_home_override, set_xiaoban_home_override

from agent.auto_skill_context import collect_matching_skill_context


def _write_skill(tmp_path, *, body: str) -> None:
    skill_dir = tmp_path / "skills" / "research" / "web-research"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def test_collects_matching_auto_context_skill(tmp_path):
    _write_skill(
        tmp_path,
        body="""---
name: web-research
description: Structured web research.
metadata:
  xiaoban:
    requires_toolsets: [web, skills]
    auto_context:
      patterns:
        - "世界杯.*比分"
      max_chars: 4000
---

# Web Research

Keep an evidence ledger and resolve conflicting sources before answering.
""",
    )
    token = set_xiaoban_home_override(tmp_path)
    try:
        context = collect_matching_skill_context(
            "小伴，帮我查一下世界杯加拿大瑞士比分",
            available_tools={"web_search", "skill_view"},
            available_toolsets={"web", "skills"},
        )
    finally:
        reset_xiaoban_home_override(token)

    assert 'auto_loaded_skill name="web-research"' in context
    assert "Keep an evidence ledger" in context
    assert "世界杯.*比分" in context


def test_returns_empty_when_pattern_does_not_match(tmp_path):
    _write_skill(
        tmp_path,
        body="""---
name: web-research
description: Structured web research.
metadata:
  xiaoban:
    auto_context:
      patterns:
        - "世界杯.*比分"
---

# Web Research
""",
    )
    token = set_xiaoban_home_override(tmp_path)
    try:
        context = collect_matching_skill_context("闲聊一句", available_tools={"skill_view"})
    finally:
        reset_xiaoban_home_override(token)

    assert context == ""


def test_respects_required_toolsets(tmp_path):
    _write_skill(
        tmp_path,
        body="""---
name: web-research
description: Structured web research.
metadata:
  xiaoban:
    requires_toolsets: [web]
    auto_context:
      patterns:
        - "最新"
---

# Web Research
""",
    )
    token = set_xiaoban_home_override(tmp_path)
    try:
        context = collect_matching_skill_context(
            "最新消息",
            available_tools={"skill_view"},
            available_toolsets={"skills"},
        )
    finally:
        reset_xiaoban_home_override(token)

    assert context == ""


def test_orders_auto_context_by_priority(tmp_path):
    base = tmp_path / "skills" / "research"
    first = base / "first"
    second = base / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "SKILL.md").write_text(
        """---
name: first
description: Low priority.
metadata:
  xiaoban:
    auto_context:
      priority: 1
      patterns: ["比分"]
---

# First
""",
        encoding="utf-8",
    )
    (second / "SKILL.md").write_text(
        """---
name: second
description: High priority.
metadata:
  xiaoban:
    auto_context:
      priority: 99
      patterns: ["比分"]
---

# Second
""",
        encoding="utf-8",
    )

    token = set_xiaoban_home_override(tmp_path)
    try:
        context = collect_matching_skill_context("比分")
    finally:
        reset_xiaoban_home_override(token)

    assert context.index('auto_loaded_skill name="second"') < context.index(
        'auto_loaded_skill name="first"'
    )
