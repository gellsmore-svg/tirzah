---
type: Module
title: Ingestion
description: Parsing sources into provenance-aware node trees, the ingestion worker and inbox queue, and the embedding/profile backfill pipelines (batched, resumable).
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/ingestion/parser.py
tags: [tirzah, ingestion, parsing, backfill, embeddings]
timestamp: 2026-06-19T00:00:00Z
---

# Ingestion (`ingestion/`, `models/`)

Turns source documents into the [graph memory](../concepts/graph-memory.md):

- **`parser.py`** — deterministic heading/paragraph parsing into the
  `source_root → source_section → source_chunk` tree (the default mock-adapter
  baseline). Optional `ingestion_adapter: llm` proposes the same tree via a
  local Ollama model or an external coding CLI (Kiro, Claude, Codex, Gemini, Grok);
  proposed trees stay `pending_review` until `promote-ingestion`.
- **`worker.py`** — the ingestion worker: commits a parsed tree, embedding each
  node via the configured [embedding adapter](adapters.md), and writes provenance/
  epoch metadata. Drives the inbox queue (`process-inbox` / `process-next`).
- **`files.py`, `dates.py`, `activity.py`** — file discovery, date parsing, and
  human-readable activity logging.
- **`embedding_backfill.py`** — batched, **resumable** embedding/profile backfill
  jobs (queue → process), for embedding a large corpus after the fact rather than
  blocking ingestion. Surfaced as `backfill-embeddings` / `backfill-profiles` and
  the `queue-`/`process-` variants in the [backfill CLI](../cli/backfill-queue.md).
- **`models/ingestion.py`** — the ingestion data records.

Ingestion is selected via `runtime.ingestion_adapter` and shares the
[adapter boundary](adapters.md) with the CLI, rebuild, and web paths.
