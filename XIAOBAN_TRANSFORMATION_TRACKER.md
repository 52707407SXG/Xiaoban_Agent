# Xiaoban-Agent Transformation Tracker

Date: 2026-06-24

## Non-Negotiable Rule

The first GitHub version of this repository must not be a pure Xiaoban baseline.

Xiaoban is the runtime chassis, but the first pushed version must already be a
My Stand native Xiaoban build with:

- Xiaoban identity and launch path.
- My Stand module capability contract alignment.
- My Stand permission and context boundaries.
- Tests proving the first Xiaoban layers work.
- Documentation that clearly says this is Xiaoban-Agent, powered by a Xiaoban
  runtime fork, not upstream Xiaoban itself.

Upstream Xiaoban commit is recorded only for traceability:

```txt
5ecf3bf0e0726b8b33682bb5c3aad9679b7b5be4
```

## Source Plans

- `/Users/saix/Documents/Codex/2026-06-18/ssh-root-47-237-254-194-2/work/xiaoban-xiaoban-transformation-plan-20260623.md`
- `/Users/saix/Documents/Codex/2026-06-18/ssh-root-47-237-254-194-2/work/xiaoban-agent-mmcc-v0.1-plugin-alignment-plan.md`

## Architecture Target

Xiaoban-Agent is a My Stand native server-side Agent.

Web, WeChat, Feishu, WeCom, QQ, CLI, and webhook are only connectors. They do
not own business logic. The server-side Agent owns:

- durable receive
- conversation log
- task state
- memory
- context building
- tool registry
- module permission checks
- audit log
- outbound delivery state

## Phase Queue

### Phase 0: Clean First Version Boundary

Status: in progress, first pass complete.

Goal: the repository can be pushed as an initial Xiaoban version, not as raw
Xiaoban.

Actions:

- [x] Keep Xiaoban source as runtime foundation.
- [x] Add Xiaoban package and docs.
- [x] Add `xiaoban` CLI entry point.
- [x] Add visible Xiaoban README/installer notes before first push.
- [x] Keep upstream commit in docs only.
- [x] Replace user-facing README with Xiaoban-Agent README.
- [x] Rename top-level package metadata to `xiaoban-agent`.
- [x] Add `bin/xiaoban`.
- [x] Rename developer setup script to `setup-xiaoban.sh`.
- [x] Replace default toolset IDs from `xiaoban-*` to `xiaoban-*`.
- [x] `python3 -m xiaoban.cli --version` shows `Xiaoban-Agent`.
- [x] Core docs/config/package surfaces use `xiaoban` commands.
- [x] Add `scripts/xiaoban_rebrand_check.py` to prevent first-release surfaces
  from regressing to Xiaoban branding.
- [x] Add `scripts/xiaoban_validate.py` as the first-release validation gate.
- [x] Rewrite default seeded SOUL identity as Xiaoban/My Stand native identity.
- [x] Change CLI help examples and parser program name to `xiaoban`.
- [x] Replace root governance docs with Xiaoban-specific `AGENTS.md`,
  `CONTRIBUTING.md`, and `SECURITY.md`.
- [x] Rebrand user-facing skill/plugin/doc text assets away from upstream
  runtime identity.
- [x] Add smoke assertions that preserve the key identity relationship:
  Xiaoban is the user-facing Agent; Xiaoban is only the runtime chassis.

Verification:

- [ ] `git log` before first push does not contain a public "pure Xiaoban baseline"
  release commit.
- [x] `xiaoban` package exists.
- [x] First pushed README says Xiaoban-Agent clearly.
- [x] Core identity files do not contain `You are Xiaoban`.
- [x] `xiaoban-cli`, `xiaoban-feishu`, `xiaoban-weixin`, `xiaoban-wecom`,
  `xiaoban-qqbot`, and `xiaoban-api-server` toolsets resolve.

Validation run:

```txt
python3 -m py_compile toolsets.py tests/test_toolsets.py bin/xiaoban xiaoban/cli.py xiaoban_cli/main.py agent/prompt_builder.py scripts/xiaoban_smoke.py
python3 scripts/xiaoban_smoke.py
python3 -c "from toolsets import TOOLSETS, resolve_toolset; ..."
```

Observed:

```txt
Xiaoban-Agent v0.17.0 (2026.6.19)
xiaoban-smoke ok
xiaoban-toolsets ok
xiaoban-rebrand-check ok
xiaoban-validate ok
```

### Phase 1: MMCC v0.1 Core

Status: first pass complete, integration pending.

Goal: My Stand module tools enter Xiaoban through one unified contract.

Actions:

- [x] Add MMCC manifest model.
- [x] Add manifest loader for JSON/YAML.
- [x] Add validator for contractVersion, moduleId, tools, permissions,
  dataScopes, and contextProviders.
- [x] Add committed fixtures for help-center, event-center, and works-processing.

Verification:

- [x] Valid manifests pass.
- [x] Wrong contractVersion fails.
- [x] Tool without requiredCapability fails.
- [x] Write tool without idempotency policy fails.
- [x] Fixture manifests load and register module-prefixed tools.

Validation run:

```txt
python3 scripts/xiaoban_smoke.py
python3 -c "from pathlib import Path; from xiaoban.mmcc.loader import load_manifest; ..."
```

Observed:

```txt
xiaoban-smoke ok
['event-center', 'help-center', 'works-processing']
['event-center.create_event', 'event-center.read_due_events', 'help-center.read_article', 'help-center.search', 'works-processing.generate_xiaohongshu_post', 'works-processing.read_project']
```

### Phase 2: ToolRegistry Adapter

Status: first pass complete, real My Stand gateway pending.

Goal: Xiaoban discovers module tools as `moduleId.toolName`.

Actions:

- [x] Convert MMCC `agent.tools` into Xiaoban-compatible tool schema.
- [x] Bind each tool to capability, sideEffect, dataScope, idempotency.
- [x] Route execution through a My Stand module tool gateway stub.
- [x] Deny direct import of My Stand module internals.
- [x] Register MMCC tools into a ToolRegistry-compatible registry.

Verification:

- [x] Registered tool names are prefixed with moduleId.
- [x] Disabled modules disappear.
- [x] Unauthorized users cannot invoke a known toolName.
- [x] Write actions without message_id/idempotencyKey are rejected.
- [x] Registered dry-run handler calls `gateway_call(moduleId, toolName, args)`.

Validation run:

```txt
python3 scripts/xiaoban_smoke.py
python3 -c "from tools.registry import ToolRegistry; from xiaoban.mmcc.registry_adapter import register_mmcc_tools; ..."
```

Observed:

```txt
xiaoban-smoke ok
tool-registry-adapter ok
enabled-module-filter ok
```

### Phase 3: ContextBuilder Adapter

Status: first pass complete, real provider execution pending.

Goal: Xiaoban only sees authorized module summaries.

Actions:

- [x] Build context from site, user, channel, installed modules, enabled
  capabilities, conversation, task state, memory, and contextProviders.
- [x] Call only authorized contextProviders.
- [x] Keep prompt compact; do not dump all module data.
- [x] Build a compact context-provider index with source, capability,
  permission roles/scopes, timestamp, and freshness.

Verification:

- [x] Current module context appears when authorized.
- [x] Unauthorized module context is absent.
- [x] Context output contains source, permission scope, timestamp, and freshness.

Validation run:

```txt
python3 scripts/xiaoban_smoke.py
python3 -c "from xiaoban.mmcc.context_builder import build_mmcc_context_index; ..."
```

Observed:

```txt
xiaoban-smoke ok
context-index ok
context-permission-scope ok
```

### Phase 4: Identity And Memory Boundary

Status: first pass complete, persistent store and My Stand API pending.

Goal: Xiaoban recognizes people and keeps memory scoped by user/site.

Actions:

- [x] Add My Stand user identity model.
- [x] Add channel identity mapping interface.
- [x] Add site owner context.
- [x] Add per-user memory namespace contract.
- [x] Add company/person graph lookup interface.
- [x] Add in-memory identity directory for smoke and connector tests.

Verification:

- [x] Same external channel user maps to the same My Stand user.
- [x] Zhang San's memory is not injected into Li Si's conversation.
- [x] Owner channel policy is enforced outside the website.
- [x] Unknown external channel identity resolves to `None` before business logic.

Validation run:

```txt
python3 scripts/xiaoban_smoke.py
python3 -c "from xiaoban.identity import ChannelIdentity, InMemoryIdentityDirectory, MyStandUserIdentity; ..."
```

Observed:

```txt
xiaoban-smoke ok
identity-directory ok
owner-channel-policy ok
person-graph ok
```

### Phase 5: Channel And Durable Receive

Status: contract first pass complete, real platform adapters pending.

Goal: Xiaoban channels become Xiaoban connectors, not fake adapters.

Actions:

- [x] Keep Feishu, Weixin, WeCom, QQ, webhook.
- [x] Disable irrelevant platforms by Xiaoban default.
- [x] Add stable conversation_id/message_id contract.
- [x] Add duplicate delivery state contract.
- [x] Add normalized inbound event contract.
- [x] Add stable delivery key with channel account id to avoid multi-app collisions.

Verification:

- [x] Duplicate message_id does not repeat writes or billing.
- [x] Unmapped external identity is refused or asked to bind.
- [x] Connector returns status without leaking runtime internals.

Validation run:

```txt
python3 scripts/xiaoban_smoke.py
python3 -c "from xiaoban.connectors import InMemoryDurableReceiveStore, NormalizedInboundEvent; ..."
```

Observed:

```txt
xiaoban-smoke ok
durable-receive ok
connector-response ok
```

### Phase 6: My Stand Event Center

Status: first pass complete, live My Stand callback pending.

Goal: My Stand events can notify Xiaoban and route to allowed channels.

Actions:

- [x] Add Event Center webhook normalizer.
- [x] Add HMAC signature contract.
- [x] Add event_id/message_id contract for dedupe.
- [x] Add delivery draft builder.

Verification:

- [x] Valid signed event creates one delivery draft.
- [x] Duplicate event returns existing status.
- [x] Invalid signature is rejected.

Validation run:

```txt
python3 scripts/xiaoban_smoke.py
python3 -c "from xiaoban.webhook import verify_event_center_signature, normalize_event_center_payload, build_event_delivery_draft; ..."
```

Observed:

```txt
xiaoban-smoke ok
event-center-webhook ok
event-center-dedupe ok
```

### Phase 7: Security And Approval

Status: first pass complete, runtime integration pending.

Goal: Xiaoban is useful but cannot乱来.

Actions:

- [x] Add source disclosure guard.
- [x] Add module token scope guard.
- [x] Add high-risk action approval model.
- [x] Add owner/admin policy gates.

Verification:

- [x] Ordinary user cannot ask Xiaoban to paste source code.
- [x] No raw API key/token enters prompt or logs.
- [x] Delete/export/permission/mass-message/deploy actions require confirmation.

Validation run:

```txt
python3 scripts/xiaoban_smoke.py
python3 -c "from xiaoban.security import classify_action; ..."
```

Observed:

```txt
xiaoban-smoke ok
security-approval ok
```

### Phase 8: Full Delivery

Status: in progress.

Goal: first GitHub push is a verified Xiaoban-Agent build.

Actions:

- Run targeted Xiaoban tests after every phase.
- Run touched Xiaoban regression tests.
- Run final smoke 3 times.
- Update README, SECURITY, install docs, and release checklist.
- Add `docs/xiaoban-xiaoban-fork-strategy.md`.
- Add `docs/mmcc-v0.1-agent-plugin-alignment.md`.
- Push to `52707407SXG/Xiaoban-Agent`.

Verification:

- Test output is recorded in the handoff.
- GitHub first pushed version is Xiaobanized.
- No server deployment is performed from this window.

## Interfaces Needed From My Stand Chief Engineer

- Installed module discovery location/API.
- Manifest storage format and runtime API.
- Module enabled/disabled source of truth.
- Role/capability grant format.
- Scope calculation rules for `self/team/company/public`.
- Tool gateway HTTP/RPC request and response schema.
- Tool error code standard.
- ContextProvider invocation schema.
- Event Center webhook schema and signature algorithm.
- External identity bind/unbind storage and flow.
- Billing/M豆 idempotency rules.

## Forbidden Work

- Do not SSH.
- Do not deploy production.
- Do not push a pure Xiaoban baseline as the first Xiaoban version.
- Do not add a second plugin protocol beside MMCC v0.1.
- Do not hardcode production account, domain, or server paths.
- Do not put secrets in Git, prompt, docs, logs, or tests.
- Do not let connector code own My Stand business logic.
- Do not let Xiaoban directly read/write SQLite, WAL, or file-storage.
