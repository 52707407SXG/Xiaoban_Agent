#!/usr/bin/env python3
"""Bulk-rewrite user-facing text assets from upstream branding to Xiaoban."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".md",
    ".mdx",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".css",
    ".scss",
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".sh",
    ".service",
    ".nix",
    ".rb",
    ".rs",
    ".lock",
}

SKIP_PARTS = {
    ".git",
    "node_modules",
    "__pycache__",
}

SKIP_FILES = {
    "xiaoban_rebrand_check.py",
    "xiaoban_brand_rewrite.py",
    "xiaoban_text_rebrand.py",
}

REPLACEMENTS = [
    ("NousResearch/xiaoban-agent", "52707407SXG/Xiaoban_Agent"),
    ("nousresearch/xiaoban-agent", "52707407SXG/xiaoban-agent"),
    ("xiaoban-agent.nousresearch.com", "github.com/52707407SXG/Xiaoban_Agent"),
    ("Xiaoban Agent", "Xiaoban-Agent"),
    ("XiaobanAgent", "XiaobanAgent"),
    ("Xiaoban CLI", "Xiaoban CLI"),
    ("Xiaoban Desktop", "Xiaoban Desktop"),
    ("Xiaoban WebUI", "Xiaoban WebUI"),
    ("xiaoban-agent", "xiaoban-agent"),
    ("xiaoban setup", "xiaoban setup"),
    ("xiaoban update", "xiaoban update"),
    ("xiaoban gateway", "xiaoban gateway"),
    ("xiaoban dashboard", "xiaoban dashboard"),
    ("xiaoban skills", "xiaoban skills"),
    ("xiaoban tools", "xiaoban tools"),
    ("xiaoban model", "xiaoban model"),
    ("xiaoban chat", "xiaoban chat"),
    ("xiaoban logs", "xiaoban logs"),
    ("xiaoban version", "xiaoban version"),
]


def iter_files(roots: list[Path]):
    for root in roots:
        base = root if root.is_absolute() else REPO_ROOT / root
        if base.is_file():
            candidates = [base]
        else:
            candidates = (path for path in base.rglob("*") if path.is_file())
        for path in candidates:
            if SKIP_PARTS.intersection(path.parts):
                continue
            if path.name in SKIP_FILES:
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            yield path


def rewrite(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    new_text = text
    for old, new in REPLACEMENTS:
        new_text = new_text.replace(old, new)
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    roots = [Path(arg) for arg in argv[1:]] or [
        Path("skills"),
        Path("optional-skills"),
        Path("plugins"),
        Path("docs"),
        Path("README.md"),
        Path("README.zh-CN.md"),
        Path("AGENTS.md"),
        Path("CONTRIBUTING.md"),
        Path("SECURITY.md"),
    ]
    changed = []
    for path in iter_files(roots):
        if rewrite(path):
            changed.append(path.relative_to(REPO_ROOT))
    for path in changed:
        print(path)
    print(f"rewritten={len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
