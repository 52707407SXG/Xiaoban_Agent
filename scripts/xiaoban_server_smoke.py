#!/usr/bin/env python3
"""No-key server smoke for Xiaoban-Agent.

This script is intentionally offline. It verifies that a freshly pulled server
checkout can run the Xiaoban entrypoints and local contracts without real
provider keys, channel tokens, My Stand host APIs, or production services.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xiaoban.connectors import InMemoryDurableReceiveStore, NormalizedInboundEvent
from xiaoban.identity import ChannelIdentity, InMemoryIdentityDirectory, MyStandUserIdentity
from xiaoban.mmcc.loader import load_manifest
from xiaoban.webhook import verify_event_center_signature
from xiaoban.webhook.event_center import canonical_json_bytes


def _run(*cmd: str, env: dict[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=merged_env,
    )
    return completed.stdout.strip()


def _check_cli() -> None:
    expected = "Xiaoban-Agent v"
    assert expected in _run(sys.executable, "-m", "xiaoban.cli", "--version")
    assert expected in _run(sys.executable, "bin/xiaoban", "--version")
    assert expected in _run("./bin/xiaoban", "--version")

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'xiaoban = "xiaoban.cli:main"' in pyproject

    path_env = str(REPO_ROOT / "bin") + os.pathsep + os.environ.get("PATH", "")
    assert expected in _run("xiaoban", "--version", env={"PATH": path_env})

    installed = shutil.which("xiaoban")
    if installed and str(Path(installed).resolve()).startswith(str(REPO_ROOT.resolve())):
        assert expected in _run(installed, "--version")


def _check_config_surface() -> None:
    config_text = (REPO_ROOT / "cli-config.yaml.example").read_text(encoding="utf-8")
    assert "web: [xiaoban-webhook]" in config_text
    assert "weixin: [xiaoban-weixin]" in config_text
    assert "feishu: [xiaoban-feishu]" in config_text
    assert "telegram: [xiaoban-telegram]" not in config_text
    assert "legacy runtime compatibility XIAOBAN_HOME" in config_text


def _check_xiaoban_home_and_user_surfaces() -> None:
    with tempfile.TemporaryDirectory(prefix="xiaoban-home-smoke-") as tmp:
        home = Path(tmp) / "state"
        doctor = _run("./bin/xiaoban", "doctor", env={"XIAOBAN_HOME": str(home)})
        assert f"XIAOBAN_HOME: {home}" in doctor
        assert "~/.xiaoban" not in doctor
        assert "xiaoban doctor" not in doctor
        assert "Installed entrypoint xiaoban" in doctor
        assert "Default My Stand channels" in doctor

    gateway_help = _run("./bin/xiaoban", "gateway", "--help")
    assert "web-desktop-pet" in gateway_help
    assert "weixin" in gateway_help
    assert "feishu" in gateway_help
    assert "Legacy runtime adapters" in gateway_help
    assert "Telegram, Discord" not in gateway_help.split("Default My Stand channels:", 1)[-1].split("commands:", 1)[0]

    gateway_run_help = _run("./bin/xiaoban", "gateway", "run", "--help")
    assert "gateway run --replace --accept-hooks" in gateway_run_help
    assert "--host" not in gateway_run_help
    assert "--port" not in gateway_run_help


def _check_systemd_execstart() -> None:
    service = REPO_ROOT / "docs" / "systemd" / "xiaoban-agent.service.example"
    text = service.read_text(encoding="utf-8")
    match = re.search(r"^ExecStart=(.+)$", text, flags=re.MULTILINE)
    assert match, "systemd ExecStart missing"
    execstart = match.group(1)
    assert "python -m xiaoban.cli gateway run --replace --accept-hooks" in execstart
    assert " --host " not in execstart and " --port " not in execstart
    assert "Environment=XIAOBAN_HOME=/var/lib/xiaoban-agent" in text
    assert "Environment=XIAOBAN_HOME=/var/lib/xiaoban-agent" in text


def _check_mmcc_fixtures() -> None:
    fixtures = sorted((REPO_ROOT / "xiaoban" / "mmcc" / "fixtures").glob("*.mmcc.json"))
    assert fixtures, "MMCC fixtures missing"
    manifests = [load_manifest(path) for path in fixtures]
    assert {manifest.module_id for manifest in manifests} == {
        "event-center",
        "help-center",
        "works-processing",
    }
    assert sum(len(manifest.tools) for manifest in manifests) == 0
    assert all(manifest.context_providers for manifest in manifests)


def _check_event_center_signature() -> None:
    payload = {
        "siteId": "site-1",
        "eventId": "evt-smoke",
        "eventType": "event.due",
        "userId": "52707407",
        "occurredAt": "2026-06-24T10:00:00+08:00",
    }
    body = canonical_json_bytes(payload)
    timestamp = "2026-06-24T10:00:00+08:00"
    secret = "server-smoke-secret"
    signature = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("utf-8") + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    assert verify_event_center_signature(body, timestamp=timestamp, signature=signature, secret=secret)
    assert not verify_event_center_signature(body, timestamp=timestamp, signature="bad", secret=secret)


def _check_dev_stores() -> None:
    durable_source = (REPO_ROOT / "xiaoban" / "connectors" / "durable_receive.py").read_text(encoding="utf-8")
    identity_source = (REPO_ROOT / "xiaoban" / "identity" / "store.py").read_text(encoding="utf-8")
    assert "dev/smoke" in durable_source
    assert "dev/smoke" in identity_source

    identity = InMemoryIdentityDirectory()
    user = MyStandUserIdentity(
        site_id="site-1",
        user_id="52707407",
        display_name="刚哥",
        role="owner",
        is_owner=True,
        capabilities=frozenset({"help.read"}),
        scopes=frozenset({"company"}),
    )
    channel_identity = ChannelIdentity(
        channel="web",
        channel_account_id="desktop-pet",
        external_chat_id="conversation-1",
        external_user_id="52707407",
    )
    identity.add_user(user)
    identity.bind_channel("site-1", "52707407", channel_identity)
    assert identity.resolve_channel(channel_identity) == user

    durable = InMemoryDurableReceiveStore()
    event = NormalizedInboundEvent(
        connector="web-desktop-pet",
        channel_identity=channel_identity,
        message_id="msg-1",
        conversation_id="conversation-1",
        client_ts="2026-06-24T10:00:00+08:00",
        kind="text",
        text="health check",
    )
    assert durable.accept(event).status == "accepted"
    assert durable.accept(event).status == "duplicate"


def main() -> int:
    _check_cli()
    _check_config_surface()
    _check_xiaoban_home_and_user_surfaces()
    _check_systemd_execstart()
    _check_mmcc_fixtures()
    _check_event_center_signature()
    _check_dev_stores()
    print("xiaoban-server-smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
