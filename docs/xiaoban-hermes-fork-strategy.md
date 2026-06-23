# Xiaoban Hermes Runtime Fork Strategy

Date: 2026-06-24

## Core Decision

Xiaoban-Agent is not a new lightweight Agent built from scratch.

It is a My Stand native Agent built on a mature Hermes runtime fork:

- Hermes provides the runtime chassis: conversation loop, tools, skills,
  memory, model providers, gateway, cron, logs, approvals, and session
  recovery.
- Xiaoban provides the My Stand native identity, module capability contract,
  site/user context, real-estate workflow policy, permission gates, and reply
  style.

The first public version of this repository must already identify as
Xiaoban-Agent. A pure upstream runtime baseline must not be pushed as the first
visible release.

## What We Keep

- Agent loop and tool-calling lifecycle.
- Provider abstraction for text, vision, embedding, and fallback models.
- Gateway/channel runtime.
- Skill and plugin infrastructure.
- Memory/search/session infrastructure.
- Cron and long-running task primitives.
- Approval and safety infrastructure.
- Existing testable runtime modules where they remain useful.

## What We Change

- Public command becomes `xiaoban`.
- Public docs, setup, help text, and model-facing identity become Xiaoban.
- Default system identity says Xiaoban is the My Stand native Agent.
- My Stand module tools enter through MMCC v0.1, not ad hoc imports.
- My Stand source can be used as readonly understanding context, not exposed as
  raw code to ordinary users.
- External channels map to My Stand identities before business logic.

## What We Do Not Change Yet

Some internal names such as `hermes_cli`, `HERMES_HOME`, and old import paths
remain compatibility surfaces for now. They are not user-facing identity. A
deep namespace migration can be done later after the runtime is stable.

This is deliberate: changing every internal identifier in one pass would create
unnecessary breakage risk.

## Upgrade Policy

Xiaoban should keep tracking useful upstream runtime improvements.

Preferred approach:

- Keep My Stand-specific code localized under `xiaoban/`.
- Patch runtime extension points only where needed.
- Avoid broad rewrites of mature runtime logic.
- Record upstream commit references.
- Re-run Xiaoban smoke and rebrand checks after each upstream merge.

If an upstream feature is useful, adapt it into Xiaoban rather than freezing
forever or re-implementing it from scratch.

## Validation Gate

Before a release:

```bash
python3 scripts/xiaoban_validate.py
```

This currently verifies:

- first-release Xiaoban branding surfaces
- MMCC manifest loading and validation
- ToolRegistry adapter behavior
- ContextBuilder authorization filtering
- identity and memory scoping contracts
- durable receive duplicate detection
- Event Center webhook normalization and signature handling
- source disclosure and approval guards
- CLI version reports `Xiaoban-Agent`

