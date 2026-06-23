# Xiaoban-Agent Security Policy

Xiaoban-Agent is designed for My Stand deployments that may contain sensitive
customer, company, finance, source-code, and module data.

## Trust Model

Xiaoban-Agent is a server-side Agent runtime. Web, WeChat, Feishu, WeCom, QQ,
webhook, and CLI are only Connectors.

Trusted identity must come from My Stand server-side session or a verified
external channel binding. Frontend-provided `role`, `userId`, `workspaceId`,
or `companyId` must not be treated as authority.

## Boundaries

- Operating-system or container isolation is the real boundary for code and
  shell execution.
- Tool approval, output redaction, prompt rules, and pattern guards are useful
  safety layers, not hard containment.
- My Stand module tools must be authorized through MMCC capability checks.
- Raw source code, secrets, private data, and internal paths must not be
  disclosed to ordinary users.

## Module Tool Safety

Every module tool call must validate:

- trusted user identity
- role
- company/team/scope
- capability grant
- data scope
- side effect level
- module enabled state
- approval requirement
- idempotency for writes

Write/external side-effect tools must carry `message_id` or `idempotencyKey`.
Retries must not duplicate M豆 charges, event creation, task creation, or
record writes.

## Secrets

Secrets must not enter:

- Git history
- system prompts
- conversation logs
- debug responses
- fixtures
- tests
- screenshots or public docs

Doctor/setup output may show only `has_key=true/false` style status.

## Reporting

For now, report vulnerabilities privately to the repository owner. Do not open
public issues containing secrets, reproduction tokens, private My Stand data,
or server paths.

Useful reports include:

- affected file/path
- reproduction steps
- expected and actual behavior
- whether the issue crosses a trust boundary
- whether a Connector, ToolRegistry, ContextBuilder, or MMCC policy check is
  involved

