---
type: CLI Index
title: Tirzah CLI
description: The `tirzah` command-line interface (~70 commands) grouped by purpose — ask/answer, ingest & documents, retrieval tools, backfills & queue, and governance & sessions.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/cli.py
tags: [tirzah, cli]
timestamp: 2026-06-19T00:00:00Z
---

# CLI (`tirzah`)

The CLI is `tirzah` (legacy `mnemosyne` alias). `--config` is a **global** flag
(before the subcommand). The commands fall into five groups:

- **[Ask & answer](ask-chat.md)** — `ask`, `chat`, `history`, and the explicit
  context steps `compile-context` / `render-context` / `build-prompt`.
- **[Ingest & documents](ingest.md)** — `init`, `serve`, `db-ping`, `ingest-one`/
  `ingest-folder`, `rebuild-*`, the inbox (`process-inbox`), `list-docs`/`show-doc`,
  and LLM-proposal review (`list-proposed-ingestions` / `promote-ingestion` /
  `reject-ingestion`).
- **[Retrieval tools](retrieval-tools.md)** — `search-nodes`, `node-context`,
  `graph-edges`, `expand-*`, `graph-explore`, and the semantic-candidate /
  semantic-edge commands.
- **[Backfills & queue](backfill-queue.md)** — `backfill-*`, the embedding/profile
  backfill queue, `queue-status`, `memory-health`, `embedding-smoke`.
- **[Governance & sessions](governance-sessions.md)** — agent identities, trust,
  policies, process objects/runs, sessions, active documents, continuity, and
  endorsement/review.

Every command runs against the [storage](../modules/storage.md) selected by
[config](../modules/config.md); answer commands use the
[retrieval modes](../concepts/retrieval-modes.md) and [adapters](../modules/adapters.md).
