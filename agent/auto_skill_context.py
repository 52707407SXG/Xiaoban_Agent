"""Auto-load task skills into the current turn context.

This is a lightweight routing layer for web/API chat surfaces where the model
must not rely on remembering to call ``skill_view`` before using generic tools.
Skills opt in through SKILL.md frontmatter:

metadata:
  xiaoban:
    auto_context:
      patterns:
        - "(?i)world cup.*score"
      max_chars: 6000
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

from agent.skill_utils import (
    extract_skill_conditions,
    get_all_skills_dirs,
    get_disabled_skill_names,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_environment,
    skill_matches_platform,
)

logger = logging.getLogger(__name__)

DEFAULT_SKILL_CONTEXT_MAX_CHARS = 6000
MAX_AUTO_SKILLS_PER_TURN = 3


def _string_set(values: Iterable[Any] | None) -> set[str]:
    if not values:
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def _conditions_match(
    conditions: dict[str, Any],
    available_tools: set[str] | None,
    available_toolsets: set[str] | None,
) -> bool:
    if available_tools is None and available_toolsets is None:
        return True

    tools = available_tools or set()
    toolsets = available_toolsets or set()

    for toolset in conditions.get("fallback_for_toolsets", []):
        if str(toolset) in toolsets:
            return False
    for tool in conditions.get("fallback_for_tools", []):
        if str(tool) in tools:
            return False
    for toolset in conditions.get("requires_toolsets", []):
        if str(toolset) not in toolsets:
            return False
    for tool in conditions.get("requires_tools", []):
        if str(tool) not in tools:
            return False
    return True


def _skill_identity(skill_file: Path, skills_dir: Path) -> tuple[str, str]:
    rel_path = skill_file.relative_to(skills_dir)
    parts = rel_path.parts
    if len(parts) >= 2:
        name = parts[-2]
        category = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
        return name, category
    return skill_file.parent.name, "general"


def _auto_context_spec(frontmatter: dict[str, Any]) -> dict[str, Any] | None:
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return None
    xiaoban = metadata.get("xiaoban")
    if not isinstance(xiaoban, dict):
        return None
    spec = xiaoban.get("auto_context") or xiaoban.get("auto_load")
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, list):
        return {"patterns": spec}
    if isinstance(spec, str):
        return {"patterns": [spec]}
    return None


def _patterns_from_spec(spec: dict[str, Any]) -> list[str]:
    raw = spec.get("patterns") or spec.get("triggers") or spec.get("regex")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return []


def _pattern_matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error as exc:
        logger.warning("Invalid auto_context skill pattern %r: %s", pattern, exc)
        return False


def _frontmatter_name(frontmatter: dict[str, Any], fallback: str) -> str:
    return str(frontmatter.get("name") or fallback).strip() or fallback


def collect_matching_skill_context(
    user_message: Any,
    *,
    available_tools: Iterable[Any] | None = None,
    available_toolsets: Iterable[Any] | None = None,
    max_skills: int = MAX_AUTO_SKILLS_PER_TURN,
) -> str:
    """Return context for auto-matched skills, or an empty string.

    The returned text is injected into the current user turn only. It does not
    mutate the cached system prompt or the persisted conversation transcript.
    """

    text = str(user_message or "").strip()
    if not text:
        return ""

    tool_names = _string_set(available_tools)
    toolset_names = _string_set(available_toolsets)
    disabled = get_disabled_skill_names()
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()

    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                raw = skill_file.read_text(encoding="utf-8")
                frontmatter, _body = parse_frontmatter(raw)
                skill_name, category = _skill_identity(skill_file, skills_dir)
                public_name = _frontmatter_name(frontmatter, skill_name)
                if public_name in seen or skill_name in seen:
                    continue
                if public_name in disabled or skill_name in disabled:
                    continue
                if not skill_matches_platform(frontmatter):
                    continue
                if not skill_matches_environment(frontmatter):
                    continue
                if not _conditions_match(
                    extract_skill_conditions(frontmatter),
                    tool_names,
                    toolset_names,
                ):
                    continue

                spec = _auto_context_spec(frontmatter)
                if not spec:
                    continue
                matched_pattern = ""
                for pattern in _patterns_from_spec(spec):
                    if _pattern_matches(pattern, text):
                        matched_pattern = pattern
                        break
                if not matched_pattern:
                    continue

                max_chars = spec.get("max_chars", DEFAULT_SKILL_CONTEXT_MAX_CHARS)
                try:
                    max_chars = int(max_chars)
                except (TypeError, ValueError):
                    max_chars = DEFAULT_SKILL_CONTEXT_MAX_CHARS
                max_chars = max(1000, min(max_chars, 12000))
                try:
                    priority = int(spec.get("priority", 0))
                except (TypeError, ValueError):
                    priority = 0
                seen.add(public_name)
                seen.add(skill_name)
                candidates.append(
                    (
                        priority,
                        "\n".join(
                            [
                                f'<auto_loaded_skill name="{public_name}" category="{category}">',
                                f"Matched current user message by pattern: {matched_pattern}",
                                "Follow this skill before generic search or browsing. If the skill defines source priority, pitfalls, or verification steps, treat that as the workflow for this turn.",
                                raw.strip()[:max_chars],
                                "</auto_loaded_skill>",
                            ]
                        ),
                    )
                )
            except Exception as exc:
                logger.debug("Failed to inspect auto_context skill %s: %s", skill_file, exc)

    if not candidates:
        return ""
    sections = [
        section
        for _priority, section in sorted(candidates, key=lambda item: item[0], reverse=True)[:max_skills]
    ]
    return (
        "## Auto-loaded task skill context\n"
        "The following skill(s) matched the current user message. Use them as the clean task workflow before relying on generic search results.\n\n"
        + "\n\n".join(sections)
    )
