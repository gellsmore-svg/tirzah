---
type: CLI Command
title: Backfills & queue
description: Batched, resumable maintenance — backfill metadata, structural edges, and embeddings/profiles over a large corpus; queue and process the backfill and semantic-candidate jobs; check queue status and memory health; smoke-test embeddings.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/cli.py
tags: [tirzah, cli, backfill, queue, embeddings, maintenance]
timestamp: 2026-06-19T00:00:00Z
---

# Backfills & queue

Maintenance over an existing [graph](../concepts/graph-memory.md), batched and
resumable (see the [ingestion module](../modules/ingestion.md)):

- **Metadata / structure** — `backfill-source-metadata`, `backfill-schema-metadata`,
  `backfill-structural-graph-edges`, `graph-status`.
- **Embeddings / profiles** — `backfill-embeddings` / `backfill-profiles` (run a
  bounded batch), and the durable job variants: `queue-embedding-backfill` /
  `queue-profile-backfill` → `process-embedding-backfill` / `process-profile-backfill`,
  with `embedding-backfill-jobs` / `profile-backfill-jobs` to list them. Use these
  to embed a large corpus after the fact (the bulk path for
  [hybrid/semantic](../concepts/hybrid-and-semantic.md)).
- **Semantic candidate batches** — `enqueue-semantic-candidates` /
  `enqueue-vector-semantic-candidates` / `enqueue-profile-semantic-candidates`
  (+ the `-batch` forms) to generate edge candidates at scale.
  `enqueue-contradiction-candidates` / `enqueue-contradiction-batch` queue
  conservative `contradicts` candidates the same way.
- **Health** — `queue-status`, `memory-health`, and `embedding-smoke` (verify an
  [embedding adapter](../modules/adapters.md) produces a vector).
