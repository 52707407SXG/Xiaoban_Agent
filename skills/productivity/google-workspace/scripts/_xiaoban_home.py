"""Resolve XIAOBAN_HOME for standalone skill scripts.

Skill scripts may run outside the Xiaoban process (e.g. system Python,
nix env, CI) where ``xiaoban_constants`` is not importable.  This module
provides the same ``get_xiaoban_home()`` and ``display_xiaoban_home()``
contracts as ``xiaoban_constants`` without requiring it on ``sys.path``.

When ``xiaoban_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``xiaoban_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``XIAOBAN_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from xiaoban_constants import display_xiaoban_home as display_xiaoban_home
    from xiaoban_constants import get_xiaoban_home as get_xiaoban_home
except (ModuleNotFoundError, ImportError):

    def get_xiaoban_home() -> Path:
        """Return the Xiaoban home directory (default: ~/.xiaoban).

        Mirrors ``xiaoban_constants.get_xiaoban_home()``."""
        val = os.environ.get("XIAOBAN_HOME", "").strip()
        return Path(val) if val else Path.home() / ".xiaoban"

    def display_xiaoban_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``xiaoban_constants.display_xiaoban_home()``."""
        home = get_xiaoban_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)
