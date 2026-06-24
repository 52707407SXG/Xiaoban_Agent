# Xiaoban-Agent Release Checklist

Use this checklist before asking the server CLI to pull and validate a release.

## Local Required Checks

```bash
python3 scripts/xiaoban_validate.py
python3 scripts/xiaoban_server_smoke.py
python3 -m xiaoban.cli --version
python3 bin/xiaoban --version
./bin/xiaoban --version
```

Optional when dev dependencies are installed:

```bash
pip install -r requirements-dev.txt
pytest tests/xiaoban
```

## Branding

- User-facing identity says Xiaoban-Agent / 小伴.
- Internal architecture may say Xiaoban runtime fork or chassis.
- Public docs must not present Xiaoban as generic Xiaoban.
- `XIAOBAN_HOME` may appear only as legacy runtime compatibility.

## Default Channels

Default My Stand channels:

- `cli`
- `web`
- `webhook`
- `weixin`
- `feishu`
- `wecom`
- `qqbot`

Legacy runtime adapters are opt-in only and must not be presented as the default
My Stand deployment surface.

## MMCC

- Xiaoban-Agent is an MMCC consumer.
- My Stand Core owns module discovery, permissions, data boundaries, and module
  manifest resolution.
- Runtime auto-discovery of module tools remains off in this fix package.
- Seed fixtures must keep `agent.tools` empty until the user approves the first
  read-only experiment.

## Server Handoff

Send the server operator:

- Commit hash.
- This checklist.
- `docs/server-install.md`.
- No-key smoke commands.
- Known production blockers.

Do not provide real provider keys or channel secrets in the handoff.
