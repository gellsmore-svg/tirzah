# Requirements Index

Last updated: 2026-09-11

This file is a compact implementation index for `LLM_Memory_Architecture_Requirements_v0.3.md`.

**Authoritative document.** `docs/consolidated-requirements-and-design.md` is the single review entry point for product requirements and design. `docs/current-product-requirements-and-design.md` (2026-05-29) is a historical input that was folded into it; where the two differ, the consolidated document wins.

Current product-level requirements and implementation design are consolidated in `docs/consolidated-requirements-and-design.md`. That document should be used as the review entry point when assessing the present UI, answer behavior, ingestion transparency, retrieval guardrails, governance direction, repository rebuild requirements, and near-term product design. Code organization boundaries are tracked in `docs/code-module-boundaries.md`. Filed review artifacts live in `docs/reviews/`; their actionable findings are folded into `docs/v1-known-limitations.md`, `docs/improvements-and-enhancements.md`, and the consolidated design.

## Ingestion

| Area | Requirement IDs | Implementation Notes |
|---|---|---|
| Watched ingestion folder | REQ-ING-01, REQ-ING-02 | Polling is acceptable; archive only after successful commit. |
| Gemma chunking | REQ-ING-03 to REQ-ING-06 | Gemma chooses chunks, proximity, relationships, and one-or-more trees. |
| Context labels | REQ-ING-06, REQ-ING-07 | User-declared document label overrides effective context across derived trees; keep Gemma labels too. |
| Graph edges | REQ-ING-04, REQ-ING-05, REQ-ING-11 | First `graph_edges` collection persists ingestion relation hints when they reference known node keys; `get_graph_edges` exposes bounded single-hop lookup, `expand_proximity` ranks one-hop neighbors, and `expand_graph_paths` provides bounded multi-hop path expansion. Relation inference and richer path scoring remain deferred. |
| Provenance | REQ-ING-08, Section 6 | Chunk-level, three tiers. |
| Source references | REQ-ING-09 | Nodes must link to document ID, version, and storage path. |
| Transactional ingestion | REQ-ING-10 to REQ-ING-12 | Stage first; commit only after complete coherent Gemma processing. |

## Failure Handling

| Area | Requirement IDs | Implementation Notes |
|---|---|---|
| Failure types | REQ-FAI-01 | Parsing, processing, database. |
| Retry queue | REQ-FAI-02 | Default three attempts. |
| Error logging | REQ-FAI-03 | Include document ID, point, description, timestamp, attempt. |
| Dead letter folder | REQ-FAI-04 | Distinct from ingestion and archive. |
| No partial active corpus writes | REQ-FAI-05 | Clear staging on failure. |

## Consolidation And Semantic Map

| Area | Requirement IDs | Implementation Notes |
|---|---|---|
| REM process | REQ-CON-01, REQ-CON-02 | Scheduled background process. |
| Semantic map | REQ-CON-03 to REQ-CON-05, REQ-SEM-01 to REQ-SEM-04 | Sense clusters, polysemy support, snapshots, synonym expansion. **Not built; REQ-SEM-01 to REQ-SEM-04 are unsatisfied.** Query expansion uses an interim synonym table (`retrieval/reformulate.py`, replaceable via `runtime.query_synonyms`) that is a stand-in, not the semantic map. |
| Low-confidence edge review | REQ-CON-06 | The 5.0 default threshold in the design (DQ-004) was never implemented. Low-confidence edges are reviewed through the semantic-edge candidate queue (label-overlap, embedding-similarity and contradiction sources); every candidate needs explicit human acceptance. |
| Embedding pre-clustering | REQ-CON-07 | Gemma confirms candidate clusters after embedding search. |

## Retrieval And Context Compilation

| Area | Requirement IDs | Implementation Notes |
|---|---|---|
| Navigable retrieval | REQ-RET-01, REQ-RET-02 | Iterative traversal, not one-shot search. |
| Fallbacks | REQ-RET-03, REQ-RET-04 | Direct document read, then optional web search queued for ingestion. |
| Context filtering | REQ-RET-05, REQ-RET-06 | Context labels and provenance tiers influence ranking. |
| Query modes | REQ-RET-07 | Qualitative semantic and quantitative direct lookup. |
| Traversal scoring | REQ-RET-08, REQ-RET-09 | Path and chunk usage feedback; chunk score never decreases. |
| Context document | REQ-CTX-01 to REQ-CTX-05 | Structured context plus self-confidence/questioning instruction. |

## Session Continuity And Endorsement

| Area | Requirement IDs | Implementation Notes |
|---|---|---|
| Continuity types | REQ-SCO-01 | Thought, process, session. |
| Session ingestion | REQ-SCO-02, REQ-SCO-03 | Session IDs and project/session clusters. |
| Prompt iteration records | REQ-SCO-01 to REQ-SCO-03 | Initial `session_continuity` records persist the latest/recent prompt cycle per session with prompt, domains, exchange ID, focus/used/active nodes, controller decision, evidence summary, answer preview, and process-trace summary. CLI/API/work-mode panel inspection exists; richer record expansion and prompt seeding remain open. |
| Continuity-critical flag | REQ-SCO-04, REQ-SCO-05 | Flag, not label type; must survive compression. |
| Restart state | REQ-SCO-06, REQ-SCO-07 | Graph node is source of truth; `.restart.md` is rendered view. **Open:** `tirzah restart-render` renders session-continuity records, but the working-copy `.restart.md` is hand-maintained notes rather than a render of a graph node. |
| Endorsement | REQ-END-01 to REQ-END-05 | Natural language endorsement, chunk-level writes, clarify if ambiguous. |
| Active document registry | REQ-ADR-01 to REQ-ADR-03 | Needed for endorsement and retrieval resolution. |

## Interface And Non-Functional

| Area | Requirement IDs | Implementation Notes |
|---|---|---|
| Local web UI | REQ-UI-01 to REQ-UI-06 | Stage 2; tabbed session threads. |
| Practical integrations | Inferred application requirement | See `docs/practical-applications.md`; Tirzah should expose memory APIs usable by web importers, coding agents, CLI workflows, voice transcript tools, and FOSS clients. |
| Local operation | REQ-NFR-01 | Cloud APIs optional only. |
| Async pipelines | REQ-NFR-02 | Ingestion/consolidation do not block UI. |
| Hardware baseline | REQ-NFR-03 | 32GB RAM target, GTX 3060 mobile, 8GB initial constraint noted. |
| Storage | REQ-NFR-04 | MongoDB adjacency-list graph. |
| Evaluation | REQ-NFR-05, REQ-NFR-06 | Compare to brute force; expect cold-start underperformance. |

## Capabilities Shipped 2026-09-10 (Requirement Mapping)

These capabilities shipped without requirement ids. Each is mapped here to the existing requirements it serves. None has a dedicated REQ id of its own; the mapping is recorded so a capability is not mistaken for a newly satisfied requirement.

| Capability | Requirement IDs | Status |
|---|---|---|
| Origin dates as a filter and ranking axis | REQ-ING-08, REQ-ING-09; consolidated *Chronological Corpus Processing* | Implemented. Operator-reviewed dates survive rebuilds; `origin_date` is indexed. |
| Targeted (diff) rebuild | REQ-ING-09, REQ-END-02, REQ-END-05, REQ-FAI-05; consolidated *Repository Refresh* | Implemented: content-first node matching, reviewed edges preserved, endorsement reset when content changes, non-destructive rollback. |
| Per-model prompt budgets | REQ-CTX-01, REQ-CTX-03 | Implemented for direct, agentic and deep modes, resolved against the model selected for the request. |
| Budget-skip summaries | REQ-CTX-01; PRD *LLM Transparency* | Implemented. Summary provenance is re-stamped when a rebuild regenerates the text. |
| Contradiction candidates | REQ-CON-02; DQ-005 | Implemented as review candidates only. Admission requires disagreement evidence and local-model confirmation before queueing; see DQ-005. |
| Multi-format ingestion (HTML, code, JSON, YAML, CSV) | REQ-ING-03; consolidated *Source Authority* | Implemented. Parser transformations are reported in the ingestion activity log (`source_analysis`). |
| Graph exploration (CLI and web) | REQ-RET-01, REQ-RET-02, REQ-UI-01 to REQ-UI-06 | Implemented as an inspection view; node and branch caps are reported as exclusions. |
