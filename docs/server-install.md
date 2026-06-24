# Xiaoban-Agent Server Install

This document is the server-side baseline for `f4db0f5-fix1`. It is for a
My Stand dev or staging host that pulls Xiaoban-Agent from GitHub and validates
the runtime fork before any production deployment.

## Boundaries

- Do not deploy to production from this document alone.
- Do not put real model keys, channel secrets, or My Stand module tokens in Git.
- Do not restore an existing `xiaoban-agent.service` blindly.
- Do not expose a gateway directly to the public internet before My Stand host
  authentication, durable stores, and connector signature checks are wired.

## Recommended Layout

```bash
sudo mkdir -p /opt/xiaoban-agent
sudo chown "$USER":"$USER" /opt/xiaoban-agent
git clone git@github.com:52707407SXG/Xiaoban-Agent.git /opt/xiaoban-agent
cd /opt/xiaoban-agent
```

Recommended runtime state:

```txt
/opt/xiaoban-agent        code checkout
/var/lib/xiaoban-agent    XIAOBAN_HOME, config, sessions, state
/var/log/xiaoban-agent    service logs
/etc/xiaoban-agent        environment files owned by the server operator
```

`XIAOBAN_HOME` is the user-facing home variable. `XIAOBAN_HOME` may still appear
inside the inherited runtime as legacy runtime compatibility, but it is not the
primary My Stand operator interface.

## Versions

- Python: 3.11 or newer.
- Node: 24 LTS is recommended for the current My Stand server environment.
- Git and ripgrep should be available on the host.

## Install Dependencies

```bash
cd /opt/xiaoban-agent
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt
```

For minimal no-key validation, the editable install is optional because
`./bin/xiaoban` can run from the repository root. The venv is still recommended
for service isolation.

## Environment

Create `/etc/xiaoban-agent/xiaoban.env`:

```bash
XIAOBAN_HOME=/var/lib/xiaoban-agent
XIAOBAN_HOME=/var/lib/xiaoban-agent
XIAOBAN_LOG_DIR=/var/log/xiaoban-agent
MYSTAND_XIAOBAN_CHANNEL_DEBUG=false
MYSTAND_XIAOBAN_ALLOW_UNVERIFIED_TEST_MESSAGES=false
MYSTAND_XIAOBAN_CHANNEL_STORE_FILE=/var/lib/xiaoban-agent/channel-store.jsonl
```

Add provider and channel keys only on the target server, never in Git:

```bash
DEEPSEEK_API_KEY=
MYSTAND_XIAOBAN_WECHAT_TOKEN=
MYSTAND_XIAOBAN_FEISHU_VERIFICATION_TOKEN=
```

During the transition period, set both `XIAOBAN_HOME` and `XIAOBAN_HOME` to the
same path. `XIAOBAN_HOME` is the operator-facing variable; `XIAOBAN_HOME` is
inherited runtime compatibility.

## No-Key Smoke

Run these before any service install:

```bash
cd /opt/xiaoban-agent
. .venv/bin/activate
python scripts/xiaoban_validate.py
python scripts/xiaoban_server_smoke.py
python -m xiaoban.cli --version
python bin/xiaoban --version
./bin/xiaoban --version
pytest tests/xiaoban
```

`pytest tests/xiaoban` requires `pip install -r requirements-dev.txt`.

## Health Check Contract

For this fix package, the offline health check is:

```bash
python scripts/xiaoban_server_smoke.py
```

When the gateway is later enabled behind the My Stand backend proxy, the HTTP
health endpoint must stay local-only or private-network-only until My Stand
session authentication and durable stores are wired.

The current service start command is:

```bash
python -m xiaoban.cli gateway run --replace --accept-hooks
```

Do not use `gateway --host ... --port ...`; the current runtime expects the
`gateway run` subcommand for a foreground service process.

## systemd Example

Use `docs/systemd/xiaoban-agent.service.example` as a template. Copy it only
after reviewing paths and environment files:

```bash
sudo cp docs/systemd/xiaoban-agent.service.example /etc/systemd/system/xiaoban-agent.service
sudo systemctl daemon-reload
sudo systemctl start xiaoban-agent
sudo systemctl status xiaoban-agent
```

Do not overwrite an existing service without backing it up first.

## Logs

Recommended log locations:

```txt
/var/log/xiaoban-agent/service.log
/var/log/xiaoban-agent/error.log
/var/log/xiaoban-agent/audit.jsonl
```

Production logs must redact provider keys, access tokens, module tokens,
customer records, and raw source snippets.

## Start, Stop, Restart

```bash
sudo systemctl start xiaoban-agent
sudo systemctl stop xiaoban-agent
sudo systemctl restart xiaoban-agent
sudo journalctl -u xiaoban-agent -n 200 --no-pager
```

## Rollback

1. Record the current commit:
   ```bash
   git -C /opt/xiaoban-agent rev-parse --short HEAD
   ```
2. Stop the service:
   ```bash
   sudo systemctl stop xiaoban-agent
   ```
3. Check out the previous approved commit:
   ```bash
   git -C /opt/xiaoban-agent checkout <previous-approved-commit>
   ```
4. Reinstall dependencies if needed:
   ```bash
   cd /opt/xiaoban-agent
   . .venv/bin/activate
   pip install -e .
   ```
5. Run smoke, then start:
   ```bash
   python scripts/xiaoban_server_smoke.py
   sudo systemctl start xiaoban-agent
   ```

Rollback must not delete `XIAOBAN_HOME`, logs, or audit files unless the My Stand
operator explicitly approves data removal.
