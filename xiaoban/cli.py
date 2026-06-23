"""Xiaoban command-line entry point.

The runtime still reuses a mature CLI implementation internally, but
the public command and process identity are Xiaoban.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    os.environ.setdefault("XIAOBAN_AGENT_MODE", "1")
    os.environ.setdefault("XIAOBAN_COMMAND_NAME", "xiaoban")

    if any(arg in {"--version", "-V", "version"} for arg in sys.argv[1:]):
        from hermes_cli import __release_date__, __version__

        print(f"Xiaoban-Agent v{__version__} ({__release_date__})")
        return

    from hermes_cli.main import main as runtime_main

    runtime_main()


if __name__ == "__main__":
    main()
