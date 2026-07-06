# Xiaoban Version Backfill From Hermes 0.18

Date: 2026-07-06

## Baseline

- Xiaoban before this pass: 0.17.0, service `xiaoban-agent.service`, health endpoint `http://127.0.0.1:8787/health`.
- Hermes reference: official stable tag `v2026.7.1`, release `Hermes Agent v0.18.0 (2026.7.1)`.
- Backfill name: `站小伴功能回填`.

## Audit Result

Already present in Xiaoban:

- Persistent `/goal` workflow and completion-contract style continuation.
- Background delegation via `delegate_task`.
- Existing opt-in `mixture_of_agents` toolset.
- Skill management tools: `skills_list`, `skill_view`, `skill_manage`.

Useful Hermes 0.18 features selected for this pass:

- Pre-final verification discipline, added as stable Xiaoban prompt policy.
- Explicit `/learn` command, added to CLI and gateway so the pet/web window can ask Xiaoban to create a reusable skill.
- MoA boundary guidance, keeping multi-model review as opt-in advanced mode rather than default chat behavior.

Deferred:

- Full Hermes provider-level MoA runtime and model picker integration. Xiaoban already has an opt-in MoA tool; making MoA a default/selectable provider would add latency, cost, routing, privacy, and accounting risk.
- Hermes desktop journey/learning graph UI. My Stand does not need this in the default user window yet.
- Low-level verification-stop code loop. Xiaoban gets the safer prompt-level verification rule now; runtime stop-loop can be evaluated separately for code-only workflows.

## Implemented Plan

1. Add stable prompt policy for quiet verification before final answers.
2. Add stable prompt policy that skills are created only on explicit user request.
3. Add stable prompt policy that MoA is advanced opt-in review mode.
4. Add `agent.learn_prompt.build_learn_prompt`.
5. Register `/learn` in the shared slash command registry.
6. Wire `/learn` in CLI and gateway dispatch.
7. Tighten `skill_manage` tool schema so it no longer invites automatic skill creation after complex tasks.
8. Update MoA toolset description to reflect opt-in/privacy boundary.
9. Bump Xiaoban to 0.17.1 for this backfill patch.

## Verification Plan

- Static import/compile for touched Python modules.
- Targeted tests for system prompt policy, `/learn` prompt, and command registry.
- Gateway health check after reinstall and service restart.
- Confirm package/service version reports 0.17.1.

## Verification Results

- `python -m py_compile` passed for touched Python modules.
- `pytest tests/agent/test_learn_prompt.py tests/agent/test_system_prompt.py tests/xiaoban_cli/test_commands.py tests/tools/test_skill_manager_tool.py -q`: 271 passed, dependency warnings only.
- `uv pip install --python .venv/bin/python -e .` updated installed metadata from 0.17.0 to 0.17.1.
- `systemctl restart xiaoban-agent.service` completed; service is active/running.
- `curl http://127.0.0.1:8787/health`: `{"status":"ok","platform":"xiaoban-agent","version":"0.17.1"}`.
- `xiaoban version`: `Xiaoban v0.17.1 (2026.7.6)`.
- My Stand API remained healthy: `curl http://127.0.0.1:18081/healthz` returned `{"ok":true}`.
- Protected capability endpoint remained locked: `http://127.0.0.1:8787/v1/capabilities` returned 401 without credentials.
