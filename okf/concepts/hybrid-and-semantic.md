---
type: Concept
title: Hybrid & semantic retrieval
description: A deterministic hybrid ranker blends min-max-normalised lexical score with query-vector cosine similarity; semantic_search reaches nodes by meaning even with no shared keywords. Both require a real embedding adapter and degrade safely to lexical.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/retrieval/queries.py
tags: [tirzah, retrieval, hybrid, semantic, embeddings, vector]
timestamp: 2026-06-19T00:00:00Z
---

# Hybrid & semantic retrieval

Two ways the [graph](graph-memory.md) is ranked beyond plain lexical search
(`retrieval/queries.py`, ADR-020):

- **Hybrid ranking** (`hybrid_rank`) — a deterministic coarse ranker: it keeps a
  candidate clearing the lexical OR vector floor, then ranks by min-max-normalised
  lexical score blended with query-vector cosine similarity (component scores
  and `match_source` exposed). `search_nodes` unions lexical hits with
  embedding-similar nodes (Atlas `$vectorSearch` when `vector_search_index` is
  set, otherwise a bounded cosine scan), so meaning-only matches can surface
  without shared keywords. Controlled by `runtime.hybrid_search_enabled` (**on
  by default**); it only engages with a real (non-mock) embedding adapter and
  degrades to lexical otherwise.
- **Semantic search** (`query_embedding_candidate_nodes`) — pure meaning-based
  retrieval: it ranks embedded nodes by cosine similarity to a query embedding,
  reaching nodes that share **no keywords** with the query (unlike the lexically
  gated `keyword_search`/`hybrid_search`). Exposed as the `semantic_search`
  primitive in [deep retrieval](deep-retrieval.md).

Both rely on node [embeddings](graph-memory.md) produced by an
[embedding adapter](../modules/adapters.md) (e.g. bge-small via `local_command`, or
Ollama models). Query and node embeddings must be **comparable** (same model +
dimensions) to be blended.
