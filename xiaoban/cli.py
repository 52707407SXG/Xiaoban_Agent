"""Xiaoban command-line entry point.

The runtime still reuses a mature CLI implementation internally, but
the public command and process identity are Xiaoban.
"""

from __future__ import annotations

import os
import sys


def _bridge_xiaoban_home() -> None:
    """Make the Xiaoban public home variable drive the inherited runtime."""

    xiaoban_home = os.environ.get("XIAOBAN_HOME", "").strip()
    if xiaoban_home and not os.environ.get("XIAOBAN_HOME"):
        # XIAOBAN_HOME is inherited runtime compatibility for the upstream chassis.
        os.environ["XIAOBAN_HOME"] = xiaoban_home


def _print_gateway_help() -> None:
    print(
        """usage: xiaoban gateway [-h] {run,start,stop,restart,status,install,uninstall,list,setup} ...

Manage the Xiaoban My Stand gateway.

Default My Stand channels:
  web-desktop-pet, webhook, weixin, feishu, wecom, qqbot, cli

commands:
  run          Run gateway in foreground. systemd should use: gateway run --replace --accept-hooks
  start        Start the installed background service
  stop         Stop the background service
  restart      Restart the background service
  status       Show gateway status
  install      Install gateway as a background service
  uninstall    Uninstall gateway service
  list         List profiles and gateway status
  setup        Configure My Stand target channels

Legacy runtime adapters such as Telegram, Discord, Slack, WhatsApp, Signal,
Matrix, HomeAssistant and DingTalk are opt-in only and are not part of the
default My Stand deployment surface.
"""
    )


def _print_gateway_run_help() -> None:
    print(
        """usage: xiaoban gateway run [-h] [-v] [-q] [--replace] [--force] [--no-supervise] [--accept-hooks]

Run the Xiaoban gateway in the foreground.

Recommended systemd ExecStart:
  python -m xiaoban.cli gateway run --replace --accept-hooks

options:
  -h, --help       show this help message and exit
  -v, --verbose    increase stderr log verbosity
  -q, --quiet      suppress stderr log output
  --replace        replace any existing gateway instance
  --force          start even when a supervisor already exists
  --no-supervise   opt out of s6 supervised redirect
  --accept-hooks   allow configured startup hooks in non-interactive service mode
"""
    )


def _handle_lightweight_surfaces(argv: list[str]) -> bool:
    if argv[:1] == ["doctor"]:
        from xiaoban.doctor import run_doctor

        raise SystemExit(run_doctor(argv[1:]))
    if argv[:2] == ["gateway", "run"] and any(arg in {"-h", "--help"} for arg in argv[2:]):
        _print_gateway_run_help()
        return True
    if argv[:1] == ["gateway"] and (len(argv) == 1 or any(arg in {"-h", "--help"} for arg in argv[1:])):
        _print_gateway_help()
        return True
    return False


def main() -> None:
    os.environ.setdefault("XIAOBAN_AGENT_MODE", "1")
    os.environ.setdefault("XIAOBAN_COMMAND_NAME", "xiaoban")
    _bridge_xiaoban_home()

    if any(arg in {"--version", "-V", "version"} for arg in sys.argv[1:]):
        from xiaoban_cli import __release_date__, __version__

        print(f"Xiaoban v{__version__} ({__release_date__})")
        return
    if _handle_lightweight_surfaces(sys.argv[1:]):
        return

    from xiaoban_cli.main import main as runtime_main

    runtime_main()


if __name__ == "__main__":
    main()
