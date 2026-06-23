# Xiaoban Agent Plugin Alignment: MMCC v0.1

Date: 2026-06-24

## Hard Rule

Xiaoban-Agent must use `Mystand Module Capability Contract v0.1`
(`MMCC v0.1`) as the single My Stand module/Agent integration contract.

Do not create a second isolated Xiaoban plugin protocol.

MMCC borrows the MCP idea of tool discovery, tool invocation, and context
boundaries, but My Stand internal modules do not all need to become MCP
servers. A future MCP bridge can adapt MMCC outward.

## Roles

My Stand Core is the host/mainboard. It owns module discovery, installation
state, permissions, event center, data boundaries, and base services.

My Stand modules are pluggable packages. Each module exposes a manifest with
capabilities, permissions, data scopes, events, Agent tools, and context
providers.

Xiaoban-Agent is the server-side brain plus ToolRegistry, ContextBuilder,
SubagentGateway, and Connector entry points. It must not bypass the My Stand
host and directly read/write module internals.

Web, WeChat, Feishu, WeCom, QQ, CLI, and webhook are Connectors. They normalize
messages, map identity, and return delivery responses. They do not own business
logic.

## Manifest Fields Xiaoban Consumes

Every module should gradually provide:

- `contractVersion`
- `moduleId`
- `packageName`
- `version`
- `displayName`
- `owner`
- `status`
- `category`
- `surfaces`
- `capabilities.provides`
- `capabilities.requires`
- `permissions`
- `dataScopes`
- `events.emits`
- `events.subscribes`
- `agent.tools`
- `agent.contextProviders`
- `migrations`

## ToolRegistry Contract

Only tools declared in `manifest.agent.tools` are registered.

Tool names must be:

```txt
moduleId.toolName
```

Examples:

```txt
event-center.create_event
works-processing.generate_xiaohongshu_post
help-center.search
```

Each call must validate:

- trusted My Stand user identity
- role
- company/team/scope
- installed and enabled module state
- capability grant
- data scope
- side effect level
- idempotency for writes
- approval requirement

Write and external side-effect tools must carry `message_id` or
`idempotencyKey` so retries cannot duplicate M豆 charges, events, tasks, or
records.

## ContextBuilder Contract

Xiaoban must not dump all module data into the prompt.

It should build context from:

- installed modules
- enabled capabilities
- current user
- current site/company/team
- channel identity
- conversation
- task state
- memory hits
- current module/page hints
- authorized `agent.contextProviders`

Each context provider result must include source module, provider id,
capability, scope, timestamp, and freshness. Raw secrets, tokens, and unrelated
module data must not enter the model context.

## SubagentGateway Contract

Subagents cannot bypass ToolRegistry.

All subagent tool use must flow through SubagentGateway and be audited with:

- `conversation_id`
- `message_id`
- `task_id`
- parent agent
- subagent id
- `moduleId`
- `toolName`
- input summary
- output summary
- status
- duration
- error

## Connector Contract

Connectors must generate or preserve stable:

- `conversation_id`
- `message_id`
- `client_ts`
- channel id
- channel account/app id
- external user id
- external chat id

The service must persist receive state before runtime execution. A repeated
message id must not execute a side-effect twice.

## Current Xiaoban Implementation

The first Xiaoban layer currently includes:

- `xiaoban/mmcc/manifest.py`
- `xiaoban/mmcc/loader.py`
- `xiaoban/mmcc/validator.py`
- `xiaoban/mmcc/policy.py`
- `xiaoban/mmcc/registry_adapter.py`
- `xiaoban/mmcc/context_builder.py`
- fixture manifests for help center, event center, and works processing
- smoke coverage in `scripts/xiaoban_smoke.py`

## Interface Gaps Needed From My Stand Core

The Agent side expects My Stand Core to provide or confirm:

- installed module list and enabled state
- manifest lookup path or API
- trusted session user identity
- role/team/company scope resolution
- capability grant lookup
- module API token format and expiry policy
- event center delivery signature contract
- audit log sink for tool calls
- idempotency result lookup for module writes

Until those are provided by the host, Xiaoban should keep using fixtures and
dry-run gateway handlers for local validation.

