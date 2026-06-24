#!/usr/bin/env python3
"""Rewrite user-visible Xiaoban branding in Python comments and strings.

This intentionally leaves identifiers such as ``xiaoban_cli`` intact so the
runtime chassis keeps working while public surfaces become Xiaoban.
"""

from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path


REPLACEMENTS = [
    ("xiaoban-agent", "xiaoban-agent"),
    ("Xiaoban Agent", "Xiaoban-Agent"),
    ("Xiaoban CLI", "Xiaoban CLI"),
    ("Xiaoban WebUI", "Xiaoban WebUI"),
    ("Xiaoban Desktop", "Xiaoban Desktop"),
    ("Xiaoban desktop", "Xiaoban desktop"),
    ("Xiaoban app", "Xiaoban app"),
    ("Xiaoban App", "Xiaoban App"),
    ("Xiaoban process", "Xiaoban runtime process"),
    ("Xiaoban processes", "Xiaoban runtime processes"),
    ("Xiaoban itself", "Xiaoban runtime"),
    ("Xiaoban checkout", "Xiaoban checkout"),
    ("Xiaoban repository", "Xiaoban repository"),
    ("official Xiaoban", "official Xiaoban"),
    ("Xiaoban URL", "Xiaoban URL"),
    ("Xiaoban log", "Xiaoban log"),
    ("Xiaoban logs", "Xiaoban logs"),
    ("Xiaoban behaves", "Xiaoban behaves"),
    ("Xiaoban only", "Xiaoban only"),
    ("Xiaoban isn't", "Xiaoban isn't"),
    ("with Xiaoban", "with Xiaoban"),
    ("using Xiaoban", "using Xiaoban"),
    ("for Xiaoban", "for Xiaoban"),
    ("from Xiaoban", "from Xiaoban"),
    ("to Xiaoban", "to Xiaoban"),
    ("Xiaoban ", "Xiaoban "),
    (" Xiaoban", " Xiaoban"),
    ("`xiaoban`", "`xiaoban`"),
    ("`xiaoban ", "`xiaoban "),
    ("xiaoban setup", "xiaoban setup"),
    ("xiaoban version", "xiaoban version"),
    ("xiaoban update", "xiaoban update"),
    ("xiaoban logs", "xiaoban logs"),
    ("xiaoban gateway", "xiaoban gateway"),
    ("xiaoban dashboard", "xiaoban dashboard"),
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
    roots = [Path(arg) for arg in argv[1:]] or [Path("agent"), Path("xiaoban_cli")]
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
