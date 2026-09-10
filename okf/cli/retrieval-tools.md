---
type: CLI Command
title: Retrieval tools
description: Direct, read-only access to the query layer — search nodes, fetch a node's context, walk graph edges, expand by proximity or path, and find/review/create semantic candidates and edges.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/cli.py
tags: [tirzah, cli, retrieval, search, graph, semantic]
timestamp: 2026-06-19T00:00:00Z
---

# Retrieval tools

Direct access to the [retrieval module](../modules/retrieval.md):

- **`search-nodes`** — lexical (and, with embeddings,
  [hybrid](../concepts/hybrid-and-semantic.md)) node search. `--origin-after` /
  `--origin-before` filter on source origin dates. `--trust-ranking` /
  `--trust-profile` opt into trust/temporal re-ranking.
- **`set-origin-date`** — operator correction of a document origin date, stamped
  onto active nodes with provenance.
- **`set-node-summary`** — provenance-tagged node summary used when context
  budget skips a record.
- **`node-context`** — a node plus its document, parent, and children.
- **`graph-edges` / `expand-proximity` / `expand-graph-paths`** — inspect edges and
  traverse the [graph](../concepts/graph-memory.md) from a node.
- **`graph-explore`** — one- or two-hop neighborhood as text or Mermaid
  (`--format mermaid`), with relation types and provenance. Optional
  `--endorsement` / `--identity-id` filters. Web viewer: `/graph`.
- **`labels`** — the label vocabulary.
- **Semantic candidates** — `semantic-candidates` (label-based), and the
  embedding-based `vector-semantic-candidates` / `profile-semantic-candidates`
  (node-to-node similarity used to propose edges).
- **Contradiction candidates** — `contradiction-candidates` previews conservative
  `contradicts` pairs (high similarity plus shared labels, date delta, and/or
  conflict wording). Queue with `enqueue-contradiction-candidates` /
  `enqueue-contradiction-batch`; review with the same semantic-edge commands.
- **Semantic edges** — `semantic-edge-candidates` (`--relation-type contradicts`
  to filter), `review-semantic-edge-candidate`, `create-semantic-edge`: the
  human-reviewed path from a similarity candidate to a durable graph edge.

These mirror the primitives the [deep-retrieval agent](../concepts/deep-retrieval.md)
uses internally, exposed for direct inspection. Batch enqueuing of semantic-candidate
generation lives in the [backfill & queue](backfill-queue.md) commands.
