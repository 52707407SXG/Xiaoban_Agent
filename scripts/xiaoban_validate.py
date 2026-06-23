#!/usr/bin/env python3
"""Run Xiaoban first-release local validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    python = sys.executable
    run([python, "scripts/xiaoban_rebrand_check.py"])
    run([python, "scripts/xiaoban_smoke.py"])
    run([python, "-m", "xiaoban.cli", "--version"])
    print("xiaoban-validate ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

