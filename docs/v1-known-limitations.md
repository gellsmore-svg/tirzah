# V1 Known Limitations

Date: 2026-06-15

Tirzah V1 is complete as a local memory workbench: ingestion, inspection, retrieval, sessions, active documents, generated-output review, semantic-edge review, governance listings, and readable activity logs are implemented across CLI and web surfaces. The limits below describe what remains scaffolded or post-V1, so release and checklist language stays precise.

This file incorporates the durable findings from the June 14 review artifacts now filed under `docs/reviews/`.

## Persistence And Recovery

- Ingestion and rebuild writes use best-effort rollback/restore behavior rather than MongoDB transactions or a two-phase commit.
- Embeddings are generated during node insertion for normal ingestion paths. Larger deployments should prefer explicit profile backfill and stronger interruption recovery.
- Maintenance paths are being batched incrementally, but not all rebuild/backfill operations are optimized for large corpora.

## Retrieval Quality

- Primary retrieval is still lexical and locally reranked; vector/profile signals exist but are not yet a full hybrid lexical+vector retrieval path. A deterministic hybrid ranker (`hybrid_rank` in `retrieval/queries.py`, ADR-020) is wired into **both** the direct and agentic retrieval modes — `runtime.hybrid_search_enabled` is now **on by default**, active only with a real (non-mock) embedding adapter and degrading safely to lexical otherwise (so it is harmless under the default mock adapter). A third retrieval mode, **`deep`** (ADR-020, `retrieval/deep.py`), is now selectable (`retrieval_mode: deep`): a Python-orchestrated agent loop over a fixed validated primitive menu — plan → execute → gate/shortlist → triage → deterministic stop → synthesise. The menu now includes **`semantic_search`** — pure meaning-based (vector) retrieval for free-text queries (`query_embedding_candidate_nodes`), which reaches relevant nodes that share **no keywords** with the query, unlike the lexically-gated `keyword_search`/`hybrid_search`. Deep mode and `semantic_search` have been validated end-to-end against the real `mnemosyne_dev` corpus (fully embedded `ams_domain`) with a real local model; a token estimator and a frontier synthesis adapter remain post-V1.
- Trust and temporal diagnostics are exposed for inspection, but they do not yet affect default ranking.
- Relevance gating remains V1-scaffold depth for broad prompts; stronger thresholds and per-node "why included" explanations are post-V1 work.

## Continuity

- Sessions, exchanges, active documents, used nodes, and context metadata are persisted.
- Initial first-class prompt-iteration records are persisted in `session_continuity` and exposed through CLI/API inspection plus the Ask workspace continuity panel. Each record now also captures a bounded summary of considered-but-not-included ("skipped") chunks, surfaced in the continuity text and the `restart-render` view. Full final context package display, unresolved follow-ups, and follow-up prompt seeding remain post-V1 work.

## Governance

- Agent identities, process objects, policies, process runs, and trust profiles are seedable and inspectable.
- Governance is observational in V1. Process steps, approvals, and behavioral expectations are not automatically enforced.

## Ingestion Intelligence

- The V1 ingestion baseline is deterministic heading/paragraph parsing through the mock ingestion adapter, selected by `runtime.ingestion_adapter: mock`.
- CLI, rebuild, worker, and web processing share an ingestion adapter boundary. `ingestion_adapter: llm` is an opt-in, review-gated chunker (local Ollama or an external coding CLI); proposed trees are `pending_review` until `promote-ingestion`. Relation extraction and richer metadata generation remain post-V1 work.

## Operations And Naming

- The preferred package and CLI name is `tirzah`; the `mnemosyne` command and some historical defaults remain for compatibility during the rename transition.
- The CLI and web API expose broad operator surfaces, but some modules are intentionally monolithic at V1 and need post-V1 refactoring.
- Most automated tests are fast unit tests with fakes. A `real_mongo` pytest marker now identifies tests that exercise real MongoDB collections, but a broader real-Mongo integration profile is still desirable.
