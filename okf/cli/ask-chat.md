---
type: CLI Command
title: Ask & answer commands
description: Ask a question or hold a chat over the graph memory, review history, and run the context-compilation pipeline as explicit, inspectable steps.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/cli.py
tags: [tirzah, cli, ask, chat, context]
timestamp: 2026-06-19T00:00:00Z
---

# Ask & answer

- **`ask`** — answer a single query over the [graph](../concepts/graph-memory.md).
  `--retrieval-mode {direct,agentic,deep}` selects the
  [strategy](../concepts/retrieval-modes.md); `--adapter` / `--model` choose the
  [answer model](../modules/adapters.md) (`mock`, `ollama_http`, `ollama_cli`,
  `hoglah`, `kiro_cli`, `claude_cli`, `codex_cli`, `google_cli`, `grok_cli`); `--json` emits the full result (answer,
  used node ids, retrieval status, process trace).
- **`chat`** — a multi-turn conversation, persisted as a [session](../concepts/sessions-and-continuity.md)
  of exchanges.
- **`history`** — review prior exchanges.
- **`compile-context` / `render-context` / `build-prompt`** — the
  [context-compilation](../concepts/context-compilation.md) pipeline as discrete
  steps, so the assembled, role-tagged, budgeted context is inspectable before it
  reaches a model.
- **`show-tree` / `queue-recent`** — view a document's node tree / recent queue
  activity.

Each `ask`/`chat` records a [restart-state snapshot](../concepts/sessions-and-continuity.md)
viewable via the [governance & sessions](governance-sessions.md) commands.
