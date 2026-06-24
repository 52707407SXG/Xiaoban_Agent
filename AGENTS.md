# Xiaoban-Agent Development Guide

This repository is the My Stand native Xiaoban-Agent.

## Core Direction

Do not rebuild Xiaoban from scratch.

Xiaoban-Agent is a Xiaoban-runtime fork:

- Keep the mature runtime chassis: loop, providers, tools, skills, gateway,
  memory, cron, logs, approvals, session recovery.
- Add the My Stand native layer under `xiaoban/`.
- Patch stable runtime extension points only when needed.
- Avoid broad rewrites of the runtime core.

User-facing identity is Xiaoban / 站小伴 / 小伴. Xiaoban is only the internal
runtime lineage and compatibility layer.

## My Stand Hard Rules

- Web, WeChat, Feishu, WeCom, QQ, webhook, and CLI are Connectors only.
- Business state belongs in the server-side Agent runtime, not in a frontend
  chat box.
- My Stand module tools must enter through MMCC v0.1.
- Do not create a separate Xiaoban plugin protocol beside MMCC.
- Tool names must be `moduleId.toolName`.
- Write/external tools require stable `message_id` or `idempotencyKey`.
- Connectors must normalize messages and map identity before runtime logic.
- ContextBuilder must use authorized context providers, not dump all module
  data into the prompt.
- Xiaoban may use readonly source clues to understand My Stand features, but it
  must not reveal raw source code, secrets, private data, or internal paths to
  ordinary users.

## Engineering Rules

- Prefer localized additions under `xiaoban/`.
- Preserve upstream-compatible runtime imports such as `xiaoban_cli` until a
  deliberate namespace migration is planned.
- Keep user-visible docs, prompts, CLI output, and setup text Xiaoban-branded.
- Keep secrets out of Git, prompts, logs, fixtures, and tests.
- Do not hardcode production domains, server paths, or account ids.
- Do not SSH or deploy from this repository task unless the user explicitly
  asks for a deployment task.
- Validate after each meaningful change.

## Validation

Run:

```bash
python3 scripts/xiaoban_validate.py
```

This covers the first-release branding gate, MMCC fixtures, ToolRegistry
adapter, ContextBuilder authorization, identity scoping, durable receive,
Event Center webhook contracts, security guards, and CLI version identity.

Full upstream test coverage can be enabled after a project venv with test
dependencies is available.

## Key Docs

- `XIAOBAN_TRANSFORMATION_TRACKER.md`
- `docs/xiaoban-xiaoban-fork-strategy.md`
- `docs/mmcc-v0.1-agent-plugin-alignment.md`

