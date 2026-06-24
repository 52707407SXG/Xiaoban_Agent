# Contributing To Xiaoban-Agent

Xiaoban-Agent is a My Stand native Agent built on a mature Xiaoban runtime fork.

The main contribution rule is simple: preserve the runtime chassis, and make My
Stand integration cleaner.

## Priorities

1. Fix real runtime bugs with focused tests.
2. Improve My Stand MMCC v0.1 module integration.
3. Improve identity mapping, durable receive, TaskState, memory, and logs.
4. Improve Connector reliability for Web, WeChat, Feishu, WeCom, QQ, and
   webhook.
5. Improve provider routing for text, vision, embedding, and fallback models.
6. Improve security approval, source disclosure protection, and audit.
7. Improve documentation and deployability.

## Plugin And Module Policy

My Stand modules must expose tools and context through
`Mystand Module Capability Contract v0.1`.

Do not add a second Xiaoban-only plugin protocol.

Xiaoban tools should be registered from module manifests as:

```txt
moduleId.toolName
```

Tool calls must pass through ToolRegistry or SubagentGateway. Subagents,
Connectors, and channel adapters must not bypass this path.

## Runtime Policy

Keep runtime changes small and testable:

- Add My Stand-specific logic under `xiaoban/`.
- Patch prompt assembly, tool registry, connector normalization, provider
  defaults, and approval gates only where needed.
- Do not rename every internal package in one pass.
- Do not break upstream-compatible imports before a migration plan exists.

## Validation

Before handing off a change, run:

```bash
python3 scripts/xiaoban_validate.py
```

If you touch upstream runtime code and the full test environment is available,
also run the relevant tests with:

```bash
scripts/run_tests.sh <test-path>
```

If the full test environment is unavailable, say that clearly in the handoff.

## Secrets

Never commit real keys, tokens, cookies, API secrets, database paths, or server
private paths. Examples must use empty variables or obvious placeholders.

