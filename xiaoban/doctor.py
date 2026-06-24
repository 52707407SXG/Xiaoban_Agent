"""Dependency-light Xiaoban doctor surface.

This doctor is intentionally import-safe: it can run before PyYAML and the full
inherited runtime dependencies are installed. The full runtime can still provide
deeper checks after installation, but this command proves the deploy candidate
has the right Xiaoban home, entrypoint, and server-facing basics.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _status(ok: bool, label: str, detail: str = "") -> None:
    mark = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"[{mark}] {label}{suffix}")


def _xiaoban_home() -> Path:
    value = os.environ.get("XIAOBAN_HOME") or os.environ.get("XIAOBAN_HOME") or str(Path.home() / ".xiaoban")
    return Path(value).expanduser()


def run_doctor(argv: list[str] | None = None) -> int:
    argv = argv or []
    should_fix = "--fix" in argv
    home = _xiaoban_home()
    inherited_home = os.environ.get("XIAOBAN_HOME")
    repo = _repo_root()
    bin_path = repo / "bin" / "xiaoban"

    print("Xiaoban Doctor")
    print("==============")
    print(f"XIAOBAN_HOME: {home}")
    if inherited_home:
        print(f"XIAOBAN_HOME: {inherited_home} (inherited runtime compatibility)")
    print(f"Repository: {repo}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    _status(sys.version_info >= (3, 9), "Python runtime", "3.11+ recommended for full gateway runtime")
    _status(bin_path.exists(), "Repository launcher", str(bin_path))
    _status(os.access(bin_path, os.X_OK), "Repository launcher executable", "./bin/xiaoban")

    installed = shutil.which("xiaoban")
    installed_ok = bool(installed) and str(installed).startswith(str(repo))
    _status(
        installed_ok,
        "Installed entrypoint xiaoban",
        installed or "not on PATH yet",
    )

    if should_fix:
        home.mkdir(parents=True, exist_ok=True)
    _status(home.exists(), "Xiaoban state directory", "created" if should_fix and home.exists() else str(home))
    _status((repo / "docs" / "systemd" / "xiaoban-agent.service.example").exists(), "systemd example")
    _status((repo / "xiaoban" / "mmcc" / "fixtures").exists(), "MMCC fixtures")
    _status((repo / "scripts" / "xiaoban_server_smoke.py").exists(), "server smoke")

    print()
    print("Default My Stand channels: cli, web-desktop-pet, webhook, weixin, feishu, wecom, qqbot")
    print("Legacy runtime adapters are opt-in only.")
    if not installed:
        print("Tip: install with `pip install -e .` to expose the `xiaoban` command on PATH.")
    if not home.exists() and not should_fix:
        print("Tip: run `xiaoban doctor --fix` to create the Xiaoban state directory.")
    return 0


__all__ = ["run_doctor"]
