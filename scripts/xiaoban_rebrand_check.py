#!/usr/bin/env python3
"""Check Xiaoban first-release branding surfaces.

This is not the final deep namespace migration. It protects the surfaces that
users and the model see first: README, package metadata, config examples,
primary prompt identity, CLI entry, Xiaoban layer, and toolset names.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SURFACE_PATHS = [
    "README.md",
    "README.zh-CN.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".env.example",
    "cli-config.yaml.example",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "setup-xiaoban.sh",
    "bin/xiaoban",
    "docs/server-install.md",
    "docs/production-readiness.md",
    "docs/release-checklist.md",
    "docs/systemd",
    "website/docs/index.mdx",
    "website/docs/getting-started/quickstart.md",
    "website/docs/getting-started/installation.md",
    "xiaoban",
    "agent/prompt_builder.py",
    "toolsets.py",
    "tests/test_toolsets.py",
]

FORBIDDEN_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"Xiaoban Agent",
        r"Xiaoban CLI",
        r"You are Xiaoban",
        r"xiaoban-agent",
        r"Xiaoban auto-inject",
        r"send and receive emails as Xiaoban",
        r"xiaoban-cli",
        r"xiaoban-feishu",
        r"xiaoban-weixin",
        r"xiaoban-wecom",
        r"xiaoban-qqbot",
        r"xiaoban setup",
        r"xiaoban tools",
        r"xiaoban model",
        r"xiaoban chat",
        r"xiaoban gateway",
        r"xiaoban update",
    ]
]

XIAOBAN_HOME_ALLOWED_PHRASES = (
    "legacy runtime compatibility",
    "inherited runtime compatibility",
)


def iter_files(path: Path):
    if path.is_dir():
        yield from (item for item in path.rglob("*") if item.is_file())
    elif path.is_file():
        yield path


def main() -> int:
    failures: list[str] = []
    for rel in SURFACE_PATHS:
        for path in iter_files(REPO_ROOT / rel):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(text):
                    failures.append(f"{path.relative_to(REPO_ROOT)}: {pattern.pattern}")
            if "XIAOBAN_HOME" in text and not any(
                phrase in text for phrase in XIAOBAN_HOME_ALLOWED_PHRASES
            ):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}: XIAOBAN_HOME must be labeled as inherited runtime compatibility"
                )
    if failures:
        print("xiaoban-rebrand-check failed")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("xiaoban-rebrand-check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
