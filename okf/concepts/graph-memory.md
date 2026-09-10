---
type: Concept
title: Graph memory
description: Sources are ingested into hierarchical, provenance-aware trees of nodes (source_root → source_section → source_chunk) with verbatim text; rebuilds open a new ingestion epoch and mark prior trees superseded rather than deleting them.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/db/schema.py
tags: [tirzah, graph, provenance, ingestion, nodes]
timestamp: 2026-06-19T00:00:00Z
---

# Graph memory

Tirzah's memory is a **graph of nodes in MongoDB**, not a blob of text. Ingesting a
document builds a hierarchical tree:

- **`source_root`** → **`source_section`** → **`source_chunk`**, with the source
  text preserved **verbatim** at the leaves.
- Every node carries **provenance**: source path, checksum, labels, ingestion
  epoch, and endorsement/status fields.
- Beyond the tree, nodes can be linked by **graph edges** (structural and
  semantic) — see [retrieval modules](../modules/retrieval.md) and the
  `graph-edges` / `expand-graph-paths` / `graph-explore`
  [tools](../cli/retrieval-tools.md). `graph-explore` (and `/graph`) renders a
  one- or two-hop neighborhood with relation types and provenance.

**Rebuilds are non-destructive:** re-ingesting a source opens a **new ingestion
epoch** and marks the prior tree `superseded` rather than deleting it, so history
and provenance survive. Nodes may also carry an **embedding** (a vector profile)
used by [hybrid & semantic retrieval](hybrid-and-semantic.md); embeddings are
generated at ingest or via a backfill.

This graph is what every [retrieval mode](retrieval-modes.md) navigates and what
[context compilation](context-compilation.md) assembles into a prompt. The schema
and store live in the [storage module](../modules/storage.md).
