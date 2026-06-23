#!/usr/bin/env python3
"""Rewrite user-visible Hermes branding in Python comments and strings.

This intentionally leaves identifiers such as ``hermes_cli`` intact so the
runtime chassis keeps working while public surfaces become Xiaoban.
"""

from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path


REPLACEMENTS = [
    ("hermes-agent", "xiaoban-agent"),
    ("Hermes Agent", "Xiaoban-Agent"),
    ("Hermes CLI", "Xiaoban CLI"),
    ("Hermes WebUI", "Xiaoban WebUI"),
    ("Hermes Desktop", "Xiaoban Desktop"),
    ("Hermes desktop", "Xiaoban desktop"),
    ("Hermes app", "Xiaoban app"),
    ("Hermes App", "Xiaoban App"),
    ("Hermes process", "Xiaoban runtime process"),
    ("Hermes processes", "Xiaoban runtime processes"),
    ("Hermes itself", "Xiaoban runtime"),
    ("Hermes checkout", "Xiaoban checkout"),
    ("Hermes repository", "Xiaoban repository"),
    ("official Hermes", "official Xiaoban"),
    ("Hermes URL", "Xiaoban URL"),
    ("Hermes log", "Xiaoban log"),
    ("Hermes logs", "Xiaoban logs"),
    ("Hermes behaves", "Xiaoban behaves"),
    ("Hermes only", "Xiaoban only"),
    ("Hermes isn't", "Xiaoban isn't"),
    ("with Hermes", "with Xiaoban"),
    ("using Hermes", "using Xiaoban"),
    ("for Hermes", "for Xiaoban"),
    ("from Hermes", "from Xiaoban"),
    ("to Hermes", "to Xiaoban"),
    ("Hermes ", "Xiaoban "),
    (" Hermes", " Xiaoban"),
    ("`hermes`", "`xiaoban`"),
    ("`hermes ", "`xiaoban "),
    ("hermes setup", "xiaoban setup"),
    ("hermes version", "xiaoban version"),
    ("hermes update", "xiaoban update"),
    ("hermes logs", "xiaoban logs"),
    ("hermes gateway", "xiaoban gateway"),
    ("hermes dashboard", "xiaoban dashboard"),
]


SKIP_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
}

SKIP_FILES = {
    "prompt.py",
    "xiaoban_brand_rewrite.py",
    "xiaoban_rebrand_check.py",
    "xiaoban_smoke.py",
}


def rewrite_text(value: str) -> str:
    for old, new in REPLACEMENTS:
        value = value.replace(old, new)
    return value


def rewrite_python(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    reader = io.StringIO(raw).readline
    tokens = []
    changed = False
    for tok in tokenize.generate_tokens(reader):
        if tok.type in {tokenize.STRING, tokenize.COMMENT}:
            new_value = rewrite_text(tok.string)
            if new_value != tok.string:
                tok = tokenize.TokenInfo(tok.type, new_value, tok.start, tok.end, tok.line)
                changed = True
        tokens.append(tok)
    if not changed:
        return False
    path.write_text(tokenize.untokenize(tokens), encoding="utf-8")
    return True


def iter_python_files(paths: list[Path]):
    for base in paths:
        if base.is_file() and base.suffix == ".py":
            if base.name not in SKIP_FILES:
                yield base
            continue
        for path in base.rglob("*.py"):
            if SKIP_PARTS.intersection(path.parts):
                continue
            if path.name in SKIP_FILES:
                continue
            yield path


def main(argv: list[str]) -> int:
    roots = [Path(arg) for arg in argv[1:]] or [Path("agent"), Path("hermes_cli")]
    changed = []
    for path in iter_python_files(roots):
        if rewrite_python(path):
            changed.append(str(path))
    for path in changed:
        print(path)
    print(f"rewritten={len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
