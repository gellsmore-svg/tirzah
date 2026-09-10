---
type: Module
title: Retrieval
description: The read-only query layer — search_nodes and the hybrid ranker, context compilation, graph traversal, and the deep-retrieval agent loop, plus trust/temporal signals.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/retrieval/queries.py
tags: [tirzah, retrieval, queries, deep, trust]
timestamp: 2026-06-19T00:00:00Z
---

# Retrieval (`retrieval/`)

- **`queries.py`** — the read-only query functions over the
  [graph](../concepts/graph-memory.md): `search_nodes` (lexical, with an optional
  `query_embedding` for [hybrid](../concepts/hybrid-and-semantic.md) ranking),
  `hybrid_rank` / `merge_candidate_pools` / `attach_query_similarity`,
  budget-aware skip summaries in `render_context_document`,
  `query_embedding_candidate_nodes` (semantic search), `node_context` /
  `expand_graph_paths` (neighbourhood + traversal), and the
  [context-compilation](../concepts/context-compilation.md) helpers.
- **`deep.py`** — the [deep-retrieval](../concepts/deep-retrieval.md) agent: the
  fixed validated `PRIMITIVES` menu, `validate_primitive_call` / `run_primitive`,
  `DeepRetrievalSession` (exclusion + stop signals), `run_deep_retrieval`
  (orchestrator), `make_planner` / `make_triager` (LLM seams over the
  [answer adapter](adapters.md)), and `synthesize_answer` / `run_deep_answer`.
- **`trust.py`** — trust/temporal weighting signals (exposed for
  [governance](../concepts/governance.md) inspection; not yet in default ranking).

These functions are read-only and authoritative; the LLM never queries Mongo
directly. Consumed by the [sessions](sessions.md) answer pipeline and the
[retrieval-tools CLI](../cli/retrieval-tools.md).
