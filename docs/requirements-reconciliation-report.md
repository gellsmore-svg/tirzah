# Requirements Reconciliation Report

Date: 2026-05-18

Status: pre-repair report. No repair implementation is authorised by this report.

Reviewer inputs:

- A re-read of the requirements, design notes, project docs, and implementation.
- An independent code review against the implementation.

## 1. Intended Architecture

Mnemosyne is intended to sit between the user and a final reasoning LLM.

The intended request flow is:

1. The user submits a request.
2. The request is passed first to a memory-agent LLM, initially Gemma.
3. Gemma is acting as a memory/context retrieval agent, not as the final reasoning LLM.
4. Gemma receives the original user prompt, role instructions, session/continuity context, and instructions for interacting with MongoDB as a semantic memory interface.
5. Gemma queries MongoDB iteratively. It may use qualitative semantic retrieval, quantitative direct lookup, graph traversal, proximity expansion, provenance filters, context labels, active document registry data, and source-document fallback.
6. Gemma decides whether retrieved results are sufficient. If not, it performs further retrieval or fallback.
7. Gemma compiles a context corpus/document from the selected memory content, preserving provenance and traceability.
8. The compiled context corpus plus the original user request are passed to a separate final thinking LLM.
9. The final thinking LLM performs the final reasoning/answer generation.
10. Outputs, session state, and continuity-critical material are then stored or queued according to the requirements.

The requirements do not authorise silent stripping, deletion, deduplication of document content, rewriting, compression, or summarisation of source documents as a content-cleanup strategy. Duplicate or overlapping content should be preserved, mapped, labelled, scored, referenced, versioned, or provenance-tracked.

## 2. Requirement Sources Reviewed

Authoritative / project requirement sources:

- `LLM_Memory_Architecture_Requirements_v0.3.md`
- `Mnemosyne_Technical_Design_v0.1.md`
- `docs/requirements-index.md`
- `docs/architecture-decisions.md`
- `docs/project-brief.md`
- `docs/build-roadmap.md`
- `meta/open-questions.md`
- `docs/source-documents.md`

Implementation files reviewed:

- `src/mnemosyne/sessions/interaction.py`
- `src/mnemosyne/retrieval/queries.py`
- `src/mnemosyne/adapters/answer.py`
- `src/mnemosyne/adapters/mock.py`
- `src/mnemosyne/models/ingestion.py`
- `src/mnemosyne/db/repositories.py`
- `src/mnemosyne/db/indexes.py`
- `src/mnemosyne/db/queue.py`
- `src/mnemosyne/ingestion/files.py`
- `src/mnemosyne/ingestion/parser.py`
- `src/mnemosyne/ingestion/worker.py`
- `src/mnemosyne/cli.py`
- `src/mnemosyne/web/app.py`
- `src/mnemosyne/sessions/registry.py`
- `src/mnemosyne/sessions/exchanges.py`
- related tests under `tests/`

Operational state checked:

- Git status is clean after corrective source-preservation commit.
- Latest pushed commit at time of report: `78dfebc Restore source-preserving ingestion`.
- Live Mongo corpus was restored to source-preserving derived state:
  - memory corpus: 357 nodes, Project Gutenberg text present;
  - AMS corpus: 1,868 documents, 175,687 `ams_domain` nodes, source-derived heading-only sections present.

## 3. Current Implementation Summary

The current implementation is a useful prototype scaffold, but it is not yet the intended Mnemosyne architecture.

Current ingestion:

- Reads markdown/plain text directly.
- Uses `runtime.ingestion_adapter` through the ingestion adapter factory, with deterministic `MockIngestionAdapter` heading/paragraph parsing as the shipped V1 default.
- Writes documents, one tree, and source-derived nodes to MongoDB.
- Copies accepted sources to archive.
- Rejects duplicate source files by SHA-256 checksum.
- Provides queue, retry, dead-letter, and inspection commands.
- Stores source path, checksum, archive path, adapter name, labels, and endorsement label on nodes.

Current retrieval and answering:

- Direct mode selects one focus node using regex/lexical search and local heuristic ranking.
- Context compilation expands from a focus node to ancestors, siblings, and descendants by parent/child hierarchy.
- Agentic mode performs one planner LLM call, executes up to three tool calls, packages top tool output, then calls the answer adapter.
- The same configured model/adapter may be used for planner and answer.
- The planner does not receive iterative tool outputs and decide whether to continue.
- The available tools are limited to `search_nodes`, `compile_context`, and `list_documents`.

Current web UI:

- Provides a local FastAPI/static UI surface.
- Supports ask/search/session/queue/job operations.
- Does not provide true tabbed session workflow or streaming WebSocket responses as specified.

Current missing layers:

- Gemma-driven ingestion is not implemented.
- Typed weighted graph edges are not implemented.
- Proximity scoring is not implemented.
- Semantic map and REM consolidation are not implemented.
- Active document registry exists as a session-scoped used-node/document skeleton and drives narrow direct reference resolution, but does not yet drive broad retrieval, endorsement, or restart state.
- Restart state node and generated `.restart.md` are not implemented.
- Explicit review controls can update node endorsement labels and provenance; natural-language endorsement detection is not implemented.
- Used-node usage feedback is partially implemented; full traversal path scoring and unused-path decay are not implemented.
- Output ingestion exists as conservative queue-to-graph insertion of unreviewed LLM answer documents; explicit review can mark output nodes, but semantic relation extraction and natural-language endorsement detection are not implemented.

## 4. Drift From Requirements

| Area | Requirement says | Implementation does | Problem | Recommended action |
|---|---|---|---|---|
| Memory-agent role | Gemma is a retrieval/context agent that queries memory and compiles context before final reasoning (`REQ-RET-01`, `REQ-RET-02`, `REQ-CTX-01`) | Agentic mode makes one planner call, executes tools once, then calls answer adapter | Not iterative; Gemma is not actually navigating until sufficient context | Replace one-shot planner with memory-agent loop before answer step |
| Final reasoning LLM separation | Retrieval layer and reasoning layer are distinct; reasoning LLM may be Gemma or external (`System Overview`, `REQ-NFR-01`) | Same runtime adapter/model can act as planner and final answer model | Conflates memory agent and final thinker | Split config and orchestration into memory-agent model and final-thinking model roles |
| Ingestion chunking | Gemma determines chunking strategy, relationships, context labels, proximity (`REQ-ING-03` to `REQ-ING-06`) | Deterministic mock parser splits headings/paragraphs | Acceptable as scaffold only; not requirement-compliant ingestion | Mark as scaffold; introduce real ingestion adapter boundary and Gemma structured call before claiming Stage 1 compliance |
| Transactional ingestion | Stage raw data, commit only after complete coherent processing; no partial active corpus writes (`REQ-ING-10`, `REQ-FAI-05`) | Sequential inserts into documents, trees, nodes without Mongo transaction/staging rollback | Partial writes can persist on mid-commit failure | Add staging collection or Mongo transaction boundary |
| Versioning | New document version creates new trees; old trees linked and not deleted (`REQ-ING-12`) | `rebuild_document` deletes old nodes/trees and inserts replacements | Destructive replacement violates versioning requirement and would lose endorsements/scores | Replace default rebuild with versioned tree insertion; keep destructive replace only as explicit maintenance command if approved |
| Source content preservation | Source document must remain traceable; no requirement authorises stripping/removal | Current code now preserves source text; previous implementation stripped Gutenberg envelope and removed empty sections before being corrected | Previous behavior was unsupported; current state restored | Keep preservation as rule; ask before any source transformation |
| Duplicate content | Duplicate files rejected by checksum per user decision; overlapping content should not be silently removed | File-level checksum rejection exists; no intra-document content removal now | File-level duplicate rejection is approved, but future content dedupe must not be assumed | Keep checksum rejection; treat content overlap via labels/provenance/versioning, not deletion |
| Graph structure | Mongo stores graph nodes with typed weighted edges and proximity (`REQ-ING-04`, `REQ-ING-05`, `REQ-ING-11`) | Nodes have parent/child hierarchy only; no `relations`, no confidence, no proximity | Retrieval cannot traverse intended semantic graph | Add schema fields before further architecture claims |
| Qualitative retrieval | Semantic/vector query plus semantic-map expansion (`REQ-RET-07`, `REQ-SEM-04`, design 7.3) | Regex text search with Python scoring | Not semantic retrieval | Keep as temporary diagnostic; implement semantic retrieval path when vector/search decision is resolved |
| Quantitative retrieval | Direct lookup by node/document/tree ID (`REQ-RET-07`) | Node/document lookup exists partially | Tree lookup and tool exposure are incomplete | Add explicit quantitative tools for document, tree, node, source read |
| Iterative traversal | Gemma assesses results and repeats until sufficient or depth limit (`REQ-RET-01`, design 7.5) | Tool results are not fed back to planner | Core architecture missing | Implement loop: plan, execute, observe, continue/done |
| Cold-start fallback | Fall back to direct source document read; optionally web search and queue ingestion (`REQ-RET-03`, `REQ-RET-04`) | If no focus node, answer adapter receives prompt-only context | Does not read source refs or queue web content | Add source-read fallback and web-fetch queue only with explicit design approval |
| Provenance weighting | Provenance tiers surfaced and weighted in retrieval (`REQ-RET-06`, Section 6) | Endorsement labels are stored but not meaningfully used by rankers | LLM is told to prefer endorsed content, but retrieval does not enforce it | Define requirement-backed provenance scoring and implement transparently |
| Traversal scoring | Used chunks/paths increase; unused paths decrease; chunk scores never decrease (`REQ-RET-08`, `REQ-RET-09`) | Used non-rejected nodes now increment `usage_score` and update `last_used_at` when exchanges are saved; no traversal path score or unused-path decrease exists | First chunk usage signal exists, but full traversal feedback remains incomplete | Add path scoring and traversed-unused path decay later; do not treat node usage alone as full requirement compliance |
| Context document format | Gemma compiles structured context with source, provenance, chunk content/summary, relationships, confidence, adjacent/full-doc availability (`REQ-CTX-01` to `REQ-CTX-04`, design 7.8) | Markdown context around selected hierarchy records; no relationships/confidence/sufficiency structure | Current context document is useful but not the specified compiled corpus | Define/implement specified JSON/hybrid corpus after agent loop |
| Self-confidence instruction | Every reasoning call includes instruction to assess confidence and surface relevant questions (`REQ-CTX-05`) | Current default instruction only says answer using context and say when insufficient | Missing standard confidence/questioning instruction | Add exact standard reasoning instruction once final-thinking call is separated |
| Output ingestion | Reasoning LLM text outputs queued for ingestion (`REQ-OUT-01`, `REQ-OUT-02`) | Answer text is queued, then can be inserted as unreviewed `generated_output` / `llm_answer` graph documents with exchange/session provenance | Useful first pass, but no relationship extraction or REM consolidation | Add relation extraction and review workflows before treating output memory as fully integrated |
| Endorsement | Natural language endorsement detection and chunk-level writes (`REQ-END-01` to `REQ-END-05`) | Explicit CLI/API review can set endorsement labels and provenance; natural-language detection is absent | Manual review works, but language-driven endorsement flow is still missing | Add natural-language detection and ambiguity handling on top of explicit controls |
| Active document registry | Session registry of referenced/produced/ingested docs (`REQ-ADR-01` to `REQ-ADR-03`) | Session registry exists for documents referenced by used nodes and output-ingested documents; direct retrieval uses it for prompts such as "this document", with document-root fallback when no topical node matches | Still not a broad retrieval/fallback/endorsement driver | Connect registry to source fallback, endorsement workflows, and broader retrieval controls |
| Session continuity | Thought/process/session continuity, restart state node, rendered `.restart.md` (`REQ-SCO-01` to `REQ-SCO-07`) | Sessions/exchanges only; `.restart.md` is manually edited | Restart file is not rendered from graph source of truth | Treat current file as manual stopgap; implement restart state node later |
| Web interface | Local UI with tabbed sessions and no direct DB interaction exposed to user (`REQ-UI-01` to `REQ-UI-06`) | UI exposes search/focus/operator DB-like controls; sessions are selector-like, not full tabs | Useful dev surface but not final intended UX | Keep as dev UI; do not mistake it for requirement-complete UI |
| Streaming | FastAPI WebSocket streams token output (`design 10.4`) | POST request waits for whole answer | Not implemented | Defer or implement explicitly in UI stage |
| Model runtime | Design specifies HF Transformers + llama-cpp-python adapter path, with all LLM calls behind adapter | Uses Ollama CLI/HTTP answer adapter; `model_adapter` mostly unused | Accepted as practical Stage 1 deviation, but architecture docs need clarity | Document as temporary runtime adapter; keep model-role abstraction |

## 5. Invented or Unsupported Behaviour

Current unsupported or assumption-based behavior:

- One-shot "agentic" planner. The requirements specify iterative graph navigation by Gemma, not one planner pass followed by answer generation.
- Hardcoded lexical fallback search terms and ranking boosts, including special terms such as `system`, `purpose`, `concept`, `function`, and `role`.
- Demoting `source_root` and empty nodes as generic ranking policy. This may be useful, but it is not stated in requirements as a substitute for Gemma retrieval judgement.
- Python-side query weighting/scoring that is not the specified traversal scoring mechanism. The required scoring concerns used traversal paths and chunks after context compilation, not arbitrary lexical relevance weights.
- Prompt token budget enforcement and character-budget context skipping as a design driver. Token efficiency is a goal, but the requirements do not authorise removing required retrieval behavior or source content to save tokens.
- Diagnostic `mock` answer path as a user-facing adapter. Useful for tests, but not part of intended user interaction architecture.
- `rebuild-document` / `rebuild-by-label` destructive replacement of existing trees and nodes. This conflicts with versioning unless treated as an explicitly approved maintenance-only operation.
- Bulk folder import as a convenience path. Useful, but not in the primary watched-folder ingestion requirement.
- Extra labels such as `external_corpus`, `public_domain`, `memory_reference`, `ams_domain`, `imported_domain`, and `research_corpus`. These are useful metadata additions but should remain descriptive labels, not hidden retrieval policy.
- `rejected` endorsement label. The requirements define three tiers; `rejected` is not specified as a fourth provenance state.

Previously introduced and now corrected:

- Project Gutenberg boilerplate stripping.
- Title inference by rewriting text into a Markdown heading.
- Suppression/removal of heading-only empty sections.

These were not requirement-backed and have been reverted in code and live Mongo state.

Special attention items:

- Document stripping: not currently implemented after correction; do not reintroduce without approval.
- Deduplication: file-level SHA-256 duplicate rejection is approved by prior user decision and documented in ADR-016. Content-level deduplication is not approved.
- Summarisation: current root/document summary is a mechanical preview. Requirements call for chunk summaries generated during ingestion, but do not authorise replacing source content with summaries.
- Content removal: current source-derived corpus restored; destructive rebuild remains a risk.
- Weighting instead of retrieval results: current ranking is heuristic and not the same as required traversal scoring. It must not be treated as fulfilling `REQ-RET-08` / `REQ-RET-09`.
- Gemma role: current implementation still risks treating Gemma as both planner and final answer model. The corrected target must separate memory-agent and final-thinking roles.

## 6. Corrected Target Design

Corrected target flow:

1. UI/API receives user request and session ID.
2. Orchestration creates a memory retrieval task containing:
   - original user request;
   - memory-agent system instructions;
   - current session state;
   - continuity-critical items;
   - active document registry;
   - available Mongo memory tools and schemas.
3. Memory-agent Gemma starts an iterative loop.
4. In each iteration, Gemma may call read-only memory tools such as:
   - semantic/qualitative search;
   - direct node/document/tree lookup;
   - graph expansion by relations;
   - proximity expansion;
   - source document read via provenance reference;
   - list active/session documents;
   - inspect previous retrieval results.
5. Mongo returns structured results with provenance, labels, confidence scores, source references, and available expansion options.
6. Gemma decides whether to continue retrieving, expand context, inspect source, or stop.
7. When sufficient, Gemma emits a compiled context corpus, not a final answer.
8. The compiled context corpus includes:
   - original user request;
   - retrieved chunks/content;
   - source document references;
   - provenance tier / endorsement label;
   - relationships and confidence scores where available;
   - adjacent/full-document availability;
   - context sufficiency assessment;
   - any open questions or limitations.
9. The final thinking LLM receives:
   - standard reasoning system instruction including confidence/question surfacing;
   - original user request;
   - compiled context corpus from Gemma.
10. Final thinking LLM generates the user-facing answer.
11. Exchange, used nodes, traversal paths, output, and restart/session state are stored or queued according to requirements.

Corrected design constraints:

- Source documents are preserved.
- Duplicate or overlapping content is not removed silently.
- Any transformation is explicit and requirement-backed.
- Ranking/weighting is transparent and tied to stated requirements.
- The memory agent may rank/select results, but hardcoded ranking must not replace the agentic retrieval loop.
- Destructive maintenance actions require explicit user approval and must not masquerade as requirement-compliant versioning.

## 7. Open Questions

1. Should the next repair implement only the iterative memory-agent loop using the current regex/hierarchy tools as temporary tools, or should it first add the missing graph/proximity schema fields required for proper traversal?
2. Should `rebuild-document` be removed from normal commands, renamed to `force-replace-document`, or kept but blocked behind an explicit destructive confirmation flag?
3. Should current lexical ranking remain only as a low-level tool result ordering hint, with Gemma making final memory selection, or should it be minimized further?
4. Is `rejected` intended to become a fourth provenance state, or should it be removed/renamed outside the three-tier trust model?
5. For Stage 1, is deterministic chunking acceptable as a temporary scaffold if clearly labelled non-compliant with `REQ-ING-03`, or should all new ingestion wait for a Gemma structured-ingestion adapter?
6. Should Ollama remain the temporary local adapter despite the technical design naming HF Transformers / llama-cpp-python, provided the model adapter boundary is cleaned up?
7. Should the development web UI continue exposing search/focus/operator tools as a console view, while the user-facing conversation flow hides direct DB operations behind Gemma?

## 8. Proposed Repair Plan

Do not execute this plan until approved.

1. Freeze source-preserving behavior as a project rule.
   - Add tests that fail if `.txt` source content is rewritten, stripped, or newline-normalized during ingestion.
   - Document that source transformation requires explicit requirement/user approval.

2. Reclassify current implementation honestly.
   - Mark deterministic chunking, regex search, direct context compilation, and current web UI as scaffold/dev surfaces.
   - Update roadmap/status docs so missing requirements are visible rather than implied complete.

3. Disable or quarantine destructive rebuild behavior.
   - Rename or gate `rebuild-document` and `rebuild-by-label`.
   - Add a non-destructive versioned ingest path aligned with `REQ-ING-12`.

4. Separate model roles.
   - Add memory-agent/retrieval model config separate from final-thinking model config.
   - Ensure Gemma memory-agent output is not treated as the final user answer.

5. Replace one-shot agentic planner with an iterative memory-agent loop.
   - Add loop state: original prompt, tool results so far, selected context so far, sufficiency status.
   - Add explicit `continue` / `done` control from Gemma.
   - Feed tool results back to Gemma each iteration.
   - Stop on `done`, max iterations, or explicit failure.

6. Expand read-only Mongo memory tools before adding write behavior.
   - `search_nodes`
   - `get_node`
   - `get_document`
   - `get_tree`
   - `get_node_context`
   - `read_source_excerpt`
   - `list_active_documents` once registry exists
   - later: relation/proximity traversal tools

7. Define compiled context corpus schema.
   - Align with technical design 7.8.
   - Include provenance, source references, selected content, relationship metadata, confidence/sufficiency, and limitations.
   - Preserve traceability from every context item back to Mongo/source.

8. Add missing graph schema fields without using them prematurely.
   - `relations`
   - `proximity`
   - `usage_score`
   - `traversal_score`
   - `continuity_critical`
   - per-node `summary`
   - Use empty/default values initially if Gemma ingestion is not ready.

9. Implement source-document fallback.
   - If memory-agent cannot find sufficient graph context, allow it to request source reads through provenance references.
   - Do not silently dump whole documents; expose source-read choices and provenance in the trace.

10. Revisit ranking only after the memory-agent loop exists.
    - Treat rankers as tool-side ordering hints, not as replacement for Gemma result selection.
    - Any weighting must cite the requirement it implements, especially provenance weighting and traversal scoring.

11. Add active document registry and session continuity foundations.
    - Required before endorsement, cold-start fallback, and restart-state correctness can be completed.

12. Only then implement final answer orchestration.
    - Pass original request plus Gemma-compiled context corpus to final thinking LLM.
    - Include required confidence/questioning system instruction.

No implementation changes should proceed until the intended repair order is approved.
