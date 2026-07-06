#!/usr/bin/env python3
"""Shared prompt builder for Xiaoban's explicit ``/learn`` command.

``/learn`` has no separate engine. It rewrites the user's request into one
normal agent turn: gather the named sources, author one reusable skill, and save
or stage it through ``skill_manage``. The explicit command is the permission
boundary; ordinary chat must not auto-create skills.
"""

from __future__ import annotations


_AUTHORING_STANDARDS = """\
Follow Xiaoban skill-authoring standards:

Frontmatter:
- name: lowercase-hyphenated, <=64 chars, no spaces.
- description: one sentence, <=60 chars, ends with a period. State the
  capability, not the implementation. Do not repeat the skill name. Avoid
  marketing words such as powerful, comprehensive, seamless, advanced, robust.
  If the description contains a colon, wrap the value in double quotes. Count
  it before saving; trim anything over 60 chars.
- version: 0.1.0
- author: always the literal value `Xiaoban`. Never use the host username,
  git config, OS account, or any probed identity.
- platforms: include [macos], [linux], or [windows] only when the skill uses
  OS-bound primitives. Prefer portable Python/pathlib/psutil first.
- metadata.xiaoban.tags: a few Capitalized, Relevant, Tags.

Body order:
1. "# <Human Title>" plus a 2-3 sentence intro: what it does, what it does not
   do, and the dependency stance.
2. "## When to Use" with concrete trigger phrases.
3. "## Prerequisites" with exact env vars, credentials, setup steps.
4. "## How to Run" with the canonical invocation through Xiaoban tools.
5. "## Quick Reference" with a flat command/API/config list.
6. "## Procedure" with numbered, copy-paste-exact steps.
7. "## Pitfalls" with known limits and failure modes.
8. "## Verification" with one check that proves the skill worked.

Xiaoban-tool framing:
- Reference available tools by name in backticks: `terminal`, `read_file`,
  `write_file`, `search_files`, `patch`, `web_extract`, `web_search`,
  `vision_analyze`, `browser_navigate`, `delegate_task`, `image_generate`,
  `text_to_speech`, `cronjob`, `memory`, `skill_view`, `execute_code`,
  `skill_manage`.
- Say `read_file` instead of cat/head/tail, `search_files` instead of
  grep/rg/find/ls, `patch` instead of sed/awk edits, and `web_extract` instead
  of curl scraping. Third-party CLIs are fine inside scripts, but prose should
  frame them as invoked through `terminal`.
- Larger scripts belong in `scripts/`; references in `references/`; templates
  in `templates/`. Add them with `skill_manage(action="write_file")`.

Quality bar:
- Use exact commands, URLs, function signatures, config keys, and file paths
  only when they appear in the gathered sources. Do not invent APIs or flags.
- Keep the skill tight and reusable. Do not paste full source docs.
- Do not create router/index/hub skills that only point at other skills.
- If `skill_manage` stages the write for approval, report the pending approval
  state instead of saying the skill is installed.
"""


def build_learn_prompt(user_request: str) -> str:
    """Build the normal-turn instruction used by CLI and gateway ``/learn``."""
    req = (user_request or "").strip()
    if not req:
        req = (
            "the workflow we just went through in this conversation; review the "
            "steps taken and distill only that requested workflow into one "
            "reusable skill"
        )

    return (
        "[/learn] The user explicitly asked Xiaoban to create or stage a "
        "reusable skill from the request below. This explicit request is the "
        "permission boundary; do not learn unrelated private context.\n\n"
        f"THE REQUEST:\n{req}\n\n"
        "The request may mix SOURCES to gather (directories, files, URLs, "
        "'what we just did', pasted notes) and REQUIREMENTS that shape the "
        "skill (focus, exclusions, name, scope, audience). Treat every part of "
        "the request as load-bearing. Prose after a path or URL is authoring "
        "guidance, not incidental text.\n\n"
        "Do this:\n"
        "1. Gather every named source with available tools: `read_file` and "
        "`search_files` for local paths, `web_extract` for URLs, current "
        "conversation history when the user references the work just done, and "
        "pasted text as-is. If a source cannot be read, state that gap in the "
        "final result.\n"
        "2. Apply every requirement, focus, and exclusion from the request. Do "
        "not fetch only the first source and ignore trailing instructions.\n"
        "3. Author exactly one SKILL.md and save or stage it through "
        "`skill_manage(action=\"create\")`. Pick a sensible category. If a "
        "non-trivial helper file is needed, add it with `skill_manage` "
        "write_file and reference it by relative path.\n\n"
        f"{_AUTHORING_STANDARDS}\n\n"
        "When done, tell the user the skill name, category, whether it was "
        "created or staged for approval, and one concise summary of what it "
        "captured."
    )
