# Langfuse Observability Plugin

This plugin ships bundled with Xiaoban but is **opt-in** — it only loads when
you explicitly enable it.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
xiaoban tools  # → Langfuse Observability

# Manual
pip install langfuse
xiaoban plugins enable observability/langfuse
```

## Required credentials

Set these in `~/.xiaoban/.env` (or via `xiaoban tools`):

```bash
XIAOBAN_LANGFUSE_PUBLIC_KEY=pk-lf-...
XIAOBAN_LANGFUSE_SECRET_KEY=sk-lf-...
XIAOBAN_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

## Verify

```bash
xiaoban plugins list                 # observability/langfuse should show "enabled"
xiaoban chat -q "hello"              # then check Langfuse for a "Xiaoban turn" trace
```

## Optional tuning

```bash
XIAOBAN_LANGFUSE_ENV=production       # environment tag
XIAOBAN_LANGFUSE_RELEASE=v1.0.0       # release tag
XIAOBAN_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
XIAOBAN_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
XIAOBAN_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## Disable

```bash
xiaoban plugins disable observability/langfuse
```
