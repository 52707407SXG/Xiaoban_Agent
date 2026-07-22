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
        r"Hermes Agent",
        r"Hermes CLI",
        r"You are Hermes",
        r"hermes-agent",
        r"Hermes auto-inject",
        r"send and receive emails as Hermes",
        r"hermes-cli",
        r"hermes-feishu",
        r"hermes-weixin",
        r"hermes-wecom",
        r"hermes-qqbot",
        r"hermes setup",
        r"hermes tools",
        r"hermes model",
        r"hermes chat",
        r"hermes gateway",
        r"hermes update",
    ]
]

HERMES_HOME_ALLOWED_PHRASES = (
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
            if "HERMES_HOME" in text and not any(
                phrase in text for phrase in HERMES_HOME_ALLOWED_PHRASES
            ):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}: HERMES_HOME must be labeled as inherited runtime compatibility"
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
