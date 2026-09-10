---
type: CLI Command
title: Ingest & document commands
description: Initialise and serve, check the database, ingest files or folders into the graph, rebuild documents non-destructively, drive the inbox queue, and inspect documents.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/cli.py
tags: [tirzah, cli, ingest, documents]
timestamp: 2026-06-19T00:00:00Z
---

# Ingest & documents

- **`init` / `serve` / `db-ping`** — initialise storage/indexes, run the
  [web app](../modules/web.md), and verify the MongoDB connection.
- **`ingest-one` / `ingest-folder`** — parse a file/folder into the
  `source_root → section → chunk` [graph](../concepts/graph-memory.md) via the
  [ingestion pipeline](../modules/ingestion.md).
- **`rebuild-document` / `rebuild-by-label`** — re-ingest non-destructively: a new
  ingestion epoch supersedes the prior tree rather than deleting it.
  `--diff-only` patches only changed sections/chunks (stable node ids);
  `--compare` prints the structural diff without applying.
- **`show-tree --compare`** — inspect the active tree and a rebuild diff against
  the archived source.
- **`enqueue-inbox` / `process-next` / `process-inbox`** — the ingestion **inbox
  queue**: stage sources, then process them (one or all).
- **`list-docs` / `show-doc`** — list ingested documents and inspect one.
- **`list-proposed-ingestions` / `promote-ingestion` / `reject-ingestion`** —
  review-gate for `ingestion_adapter: llm` trees (`pending_review` until
  promoted; rejected trees stay out of retrieval).

Embeddings are generated during ingestion (per the configured
[embedding adapter](../modules/adapters.md)); embedding an existing corpus at scale
uses the [backfill commands](backfill-queue.md).
