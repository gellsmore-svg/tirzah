# Improvements and Enhancements

**Date:** 2026-06-15
**Status:** Living post-V1 proposal document. This captures concrete, actionable improvements and enhancements to existing V1 features that would meaningfully strengthen Tirzah as a local memory and retrieval layer.

All proposals are evaluated against Tirzah's core principles (see `docs/consolidated-requirements-and-design.md`):
- Source authority and full provenance preservation at the chunk level.
- Transparency (human-readable activity first; raw data available on demand).
- Local-first operation with HTTP limited to the human UI and optional final answer calls.
- Appropriate human review gates for endorsement and semantic relationships in the current phase.
- Quality over premature optimization or automation.

This document complements (and does not duplicate) the staged roadmap in `docs/build-roadmap.md`, known gaps listed in the V1 reconciliation, the "Required next improvements" section of the consolidated requirements, `docs/open-questions.md`, and `docs/development-plan.md`. Filed review artifacts in `docs/reviews/` are supporting evidence; actionable findings belong here, in known limitations, or in the consolidated design.

## How to Use This Document
- Items are grouped by functional area.
- Each entry includes a short description, rationale tied to current implementation, suggested approach notes, rough priority, and links to related code or documents.
- "High" items offer strong leverage on existing V1 foundations (adapters, MemoryStore, review queues, process runs, active documents, deterministic scaffolding) with contained scope.
- Use this list when planning slices after the current V1.x release line.

---

## 1. Ingestion Quality & Source Fidelity

### 1.1 Optional LLM-Assisted Chunking Adapter (Review-Gated)
**Description:** Add a real ingestion adapter implementation behind the existing runtime adapter boundary that proposes hierarchical structure and summaries using a local model, while always writing the full original source text and requiring explicit or implicit endorsement before the new tree becomes preferred for retrieval.

**Rationale:** Current ingestion is still deterministic by default (`runtime.ingestion_adapter: mock`, `MockIngestionAdapter`, and heading-based parsing in `src/tirzah/adapters/mock.py`). A runtime factory now exists (`src/tirzah/adapters/ingestion.py`) so CLI, rebuild, worker, and web processing can share a future implementation without changing entry points. The roadmap and development plan explicitly call for LLM-assisted ingestion as a post-V1 step. Poor chunk boundaries are a primary source of retrieval quality problems.

**Approach notes:**
- Keep `MockIngestionAdapter` as the default for reproducibility and tests.
- Add new implementations through `ingestion_adapter()` and require explicit config selection.
- New adapter produces candidate `IngestedNode` trees with proposed `summary`, `relations`, and `proximity` scaffolds already present in the schema.
- On commit, mark the tree with `ingestion_kind: "llm_proposed"` and require an operator review step (or batch) before promoting it to active status for a document.
- Always preserve the exact source text in `source_chunk` nodes.
- Provide `tirzah compare-rebuild <document_id>` or similar that shows side-by-side deterministic vs proposed trees.

**Priority:** High — **implemented (opt-in).** `ingestion_adapter: llm` proposes
verbatim `source_root`/`source_section`/`source_chunk` trees via
`ingestion_model_adapter` (`ollama_http` / `ollama_cli` / `kiro_cli` /
`claude_cli` / `codex_cli` / `google_cli` / `grok_cli` / `hoglah` / `mock`), commits them as
`ingestion_kind: llm_proposed` / `pending_review`, and
keeps mock as the default. Remaining work: side-by-side `compare-rebuild`,
review-gated relation promotion from the proposal, and LLM-assisted summaries
that are themselves endorsed.
**Related:** `src/tirzah/adapters/`, `src/tirzah/db/repositories.py:commit_ingestion` and `rebuild_document`, `docs/development-plan.md`, `docs/build-roadmap.md` (Stage 3+ gaps).

### 1.2 Source Diffing and Targeted Rebuilds
**Description:** On `rebuild-document` or label-based rebuild, compute a structural diff between the archived source and the current active tree and allow operators to accept only changed sections instead of a full tree replacement.

**Rationale:** Rebuilds currently create a new epoch and mark the entire prior tree `superseded` (see `rebuild_document` in repositories and the guarded `--force-replace` path in CLI). For large documents (e.g. the 175k-node AMS corpus), full rebuilds are heavy even when only small edits occurred.

**Approach notes:**
- Store or compute a lightweight section-level checksum or hash tree during ingestion.
- Offer `--diff-only` mode that proposes only affected subtrees.
- Update provenance on changed nodes while preserving stable node identities where possible.
- Surface the diff in both CLI (`show-tree --compare`) and web developer view.

**Priority:** High
**Related:** `src/tirzah/cli.py:rebuild_document_from_existing_source`, `src/tirzah/db/repositories.py:rebuild_document`, `ingestion_epoch` handling, `docs/build-roadmap.md` (version comparison gap).

### 1.3 Richer Metadata Extraction and Chronological Intelligence
**Description:** Strengthen and make first-class the existing date analysis (`src/tirzah/ingestion/dates.py`) so that origin dates, document type, and other lightweight metadata become reliable signals for retrieval ranking, session continuity, and graph edges.

**Rationale:** Chronological source-date extraction exists and is used in folder planning, but is not deeply integrated into search, trust diagnostics, or context ordering beyond basic fields.

**Approach notes:**
- Persist extracted `origin_date` and confidence on the document and root node.
- Expose date-range filters in `search-nodes` and the memory-agent tool surface.
- Use recency as a first-class (but secondary) signal in direct retrieval ranking and trust/temporal diagnostics.
- Allow operators to correct dates via CLI/web with full provenance.

**Priority:** Medium
**Related:** `src/tirzah/ingestion/dates.py`, `src/tirzah/retrieval/trust.py`, `src/tirzah/cli.py` chronological helpers, retrieval queries.

### 1.4 Support for Additional Source Formats with Structure Preservation
**Description:** Extend beyond `.md`/`.txt` (current `SUPPORTED_SUFFIXES`) to handle HTML (with heading structure), common code files (with function/class boundaries), and lightweight structured formats while still storing the original text faithfully.

**Rationale:** The project already has a large Markdown corpus and an AMS research corpus. Future users will bring code, exported notes, and web captures. Treating everything as flat text loses valuable structure that the tree model is designed to capture.

**Approach notes:**
- Keep parser pluggable (similar to adapters).
- New formats must still produce `source_root` / `source_section` / `source_chunk` hierarchy plus full original text.
- Add format-specific labels (e.g. `code_python`, `html_export`).
- Provide safe stripping of front-matter, navigation chrome, etc., with the raw source always archived.

**Priority:** Medium
**Related:** `src/tirzah/ingestion/parser.py`, `src/tirzah/ingestion/files.py`, web upload paths.

---

## 2. Retrieval Precision & Context Construction

### 2.1 Hybrid Lexical + Vector Search with Real Embeddings
**Description:** When nodes have populated embeddings (via the existing profile/embedding backfill machinery), use them for candidate generation and re-ranking alongside current lexical + near-match logic. Support Mongo `$vectorSearch` when available, with a local fallback path.

**Rationale:** Embeddings and profile backfill jobs are implemented and exercised, indexes exist (`embedding.model` + `dimensions`), but search (`search_nodes` in `src/tirzah/retrieval/queries.py`) remains primarily lexical with temporary ordering hints. The architecture decision log and roadmap anticipated vector use.

**Approach notes:**
- Add an `embedding_similarity` stage in candidate collection (respecting `runtime.embedding_adapter` constraints).
- Provide a clear `hybrid_score` or diagnostic breakdown in traces and activity logs.
- Make vector usage optional and gated by config + presence of embeddings (graceful degradation to current behavior).
- Surface "this result came from vector similarity" in developer traces and the plain activity log.

**Priority:** High
**Related:** `src/tirzah/retrieval/queries.py`, `src/tirzah/db/indexes.py`, embedding backfill modules, `config.runtime`, Mongo vector search open question.

### 2.2 Budget-Aware Hierarchical Summarization and Skip Metadata
**Description:** When a character or token budget forces omission of large subtrees, automatically include short, provenance-tagged summaries of skipped sections (leveraging the `summary` field that already exists on nodes and `IngestedNode`).

**Rationale:** `render_context_document` and `compile_context` already track included/skipped under budget, but large documents (common in the AMS corpus) can result in abrupt cuts. The schema has had `summary` fields since early design.

**Approach notes:**
- Prefer node-stored summaries; fall back to lightweight extractive or model-generated ones only when missing (and mark them as derived).
- Always surface the decision in the rendered context and the structured `context_document`.
- Allow operators to pre-generate or edit summaries with full audit trail.

**Priority:** High
**Related:** `src/tirzah/retrieval/queries.py:render_context_document`, `compile_context`, `src/tirzah/models/ingestion.py`, activity reports.

### 2.3 Stronger Near-Match, Typo Tolerance, and Query Reformulation
**Description:** Improve the existing near-match fallback (already present after empty or weak searches) with better fuzzy matching, synonym/light stemming hints, and an explicit "reformulated query" step visible in traces.

**Rationale:** Current implementation has a useful fallback scaffold, but the consolidated requirements explicitly call out "less dependence on lexical regex candidate collection" as a required next improvement.

**Approach notes:**
- Extract the reformulation logic into a reusable stage that both direct and agentic paths can call.
- Record the original vs. expanded/reformulated terms in the query assembly artifact and activity log.
- Make the threshold and candidate expansion limits configurable per retrieval request.

**Priority:** Medium-High
**Related:** `src/tirzah/retrieval/queries.py`, `src/tirzah/sessions/interaction.py` query assembly, consolidated requirements "Required next improvements".

### 2.4 Configurable, Model-Aware Context and Token Budgets
**Description:** Move beyond the current global `retrieval.context_char_budget` / `prompt_token_budget` (simple 4-char approximation) to per-model or per-adapter profiles with optional real tokenizer integration.

**Rationale:** Different models (gemma3:1b vs larger) have very different practical context windows. The current approximation and single global budget are known limitations for prompt quality.

**Approach notes:**
- Add a `model_profiles` section in config.
- Optional dependency on `tiktoken` or equivalent for accurate counting (behind an extra).
- Expose the effective budget and truncation decisions clearly in `build-prompt` output and answer traces.

**Priority:** Medium
**Related:** `src/tirzah/retrieval/queries.py:estimate_tokens`, `build_prompt_envelope`, config, web model selector.

---

## 3. Semantic Graph & Relationship Intelligence

### 3.1 Actionable Trust and Temporal Weighting in Ranking
**Description:** Promote the existing trust/temporal diagnostic machinery (`src/tirzah/retrieval/trust.py`) from purely explanatory signals into a configurable, secondary ranking component in direct retrieval (and available to the memory-agent).

**Rationale:** Diagnostics for recency, frequency, stability, verification, etc. are computed and attached to traces, but the roadmap and governance schema plan note that actual ranking effects are not yet implemented. This is low-hanging fruit on existing work.

**Approach notes:**
- Add a `trust_weighting_profile` override per session or per ask.
- Apply a bounded, explainable boost/penalty after the primary provenance + usage ordering.
- Always show the before/after effect in developer traces and a compact version in the readable activity log.
- Keep the current conservative default (diagnostics-only) until operators opt in.

**Priority:** High
**Related:** `src/tirzah/retrieval/trust.py`, `src/tirzah/db/governance.py`, `docs/governance-schema-plan.md`, known V1 gaps around trust/temporal ranking.

### 3.2 Contradiction and Conflict Detection Candidates
**Description:** Extend the semantic candidate machinery to surface potential `contradicts` relations (explicitly called for in the architecture decisions) using a combination of embedding distance, temporal signals, and shared-entity cues.

**Rationale:** ADR-005 and open decisions listed "Contradictions as first-class edges" as desirable early. The current candidate system already handles `related_to` and label-overlap evidence.

**Approach notes:**
- New candidate source type with conservative generation (high bar to avoid noise).
- Same review/accept/reject workflow as other semantic edges, with explicit "contradicts" relation_type.
- Include both nodes' dates and provenance in the candidate record for easy human judgment.

**Priority:** Medium
**Related:** `src/tirzah/db/repositories.py` semantic edge code, `graph_edges`, `semantic_edge_candidates`, ADR table.

### 3.3 Lightweight Graph Visualization and Exploration
**Description:** Add a simple (text or SVG) graph neighborhood viewer in the web UI developer mode and a richer `graph-explore` CLI command that renders one-hop and two-hop neighborhoods with relation types and provenance.

**Rationale:** Graph edges and proximity/path expansion exist, but inspection is purely tabular/JSON. Visual or structured neighborhood views would dramatically improve an operator's ability to understand and curate the semantic layer.

**Approach notes:**
- Start with ASCII-art or Mermaid diagram output in CLI (`--format mermaid`).
- Web: a read-only force-directed or tree+links pane using vanilla JS (no new heavy deps).
- Respect identity scoping and endorsement filters.

**Priority:** Medium
**Related:** `expand_proximity`, `expand_graph_paths`, `graph_edges_for_node` in retrieval, web browse/ingestion tabs.

---

## 4. Agentic Memory & Interaction Quality

### 4.1 More Robust Memory-Agent Planner (JSON Mode + Stronger Validation)
**Description:** Improve the current agentic loop (`retrieval_mode: agentic`) with better JSON-mode enforcement, schema validation of tool calls before execution, and richer repair guidance when the planner produces malformed output.

**Rationale:** The current implementation already does iteration, repair instructions, and traces (see `src/tirzah/sessions/interaction.py` and `docs/agentic-retrieval-process.md`). The README notes that "the planner can choose broad or lossy search text" and that a "stricter JSON planner mode would reduce malformed planner output".

**Approach notes:**
- Make `memory_agent_ollama_format: json` (already configurable) the strongly preferred path when the model supports it.
- Add a small Pydantic or JSON-Schema validator for the exact tool call shape before any execution.
- Improve the repair section in the next planner prompt with concrete examples of prior failures.
- Expose planner "thinking" vs final tool calls more clearly even when `ollama_hide_thinking` is on.

**Priority:** High
**Related:** `src/tirzah/sessions/interaction.py`, agentic-retrieval-process.md, runtime config for ollama_format, README findings.

### 4.2 Expand Allowed Read-Only Tools for the Memory Agent
**Description:** Add a small number of high-value read-only tools that the memory agent can call without increasing its authority: date-range search, trust-diagnostic lookup for a node, recent exchanges in the current conversation domain, and a "search by label set" helper.

**Rationale:** The current allowed tool list is already fairly rich, but practical agent behavior is limited by what it can discover. Adding narrow, safe tools improves the memory agent's ability to do the job the architecture intends.

**Approach notes:**
- All new tools must be read-only and go through the same validation + observation recording path.
- Document the exact tool contract in the agent system prompt and in docs.
- Provide deterministic mock implementations for tests.

**Priority:** Medium
**Related:** interaction.py tool surface, `src/tirzah/retrieval/queries.py`, agentic process doc.

### 4.3 Streaming Answer Generation (Web + Optional CLI)
**Description:** Support streaming from the answer adapter (starting with the Ollama CLI path where feasible) and surface incremental output in the web Ask workspace with proper activity log updates.

**Rationale:** The web UI and CLI are already the primary interaction surfaces. Long answers without progress feel broken for real work. Streaming was explicitly noted as future in the development plan.

**Approach notes:**
- Keep the full exchange persistence and activity report at the end.
- Web: use Server-Sent Events or chunked responses.
- CLI: optional `--stream` flag that prints tokens as they arrive while still capturing the full trace.
- Clearly mark in traces that streaming was used.

**Priority:** Medium
**Related:** web/app.py answer endpoints, adapters/answer.py, sessions/interaction.py, development-plan.md.

---

## 5. Session Continuity & Long-Term Memory

### 5.1 Last Prompt Iteration Records and Continuity Panel
**Status:** Partially implemented after V1. Initial `session_continuity` records are written on saved answers and exposed through `tirzah session-continuity`, `/api/session-continuity`, and the Ask workspace continuity panel. Remaining work is unresolved follow-up capture, rejected/full-context expansion, and using the record to seed future prompts.

**Description:** Implement the "last prompt iteration record" concept described in the consolidated requirements: a dedicated continuity artifact per session (or conversation domain) that captures submitted prompt, interpreted intent, retrieved chunks, context package, answer, and unresolved follow-ups.

**Rationale:** This is called out as open implementation work in the consolidated design. Active documents provide a skeleton, but richer thread continuity is needed for the "persistent working memory" vision (Stage 5).

**Approach notes:**
- Store as a bounded recent history collection or embedded in the session document.
- Keep CLI (`session-continuity <session-id>`) and work-mode web panel inspection lightweight while expanding the underlying record detail.
- Use the record to seed follow-up prompts ("continue from last...") and to improve active document vocabulary.

**Priority:** High
**Related:** `src/tirzah/sessions/`, consolidated-requirements-and-design.md (Last Prompt Iteration Record section), active_documents.py.

### 5.2 Restart State as First-Class Graph Nodes
**Description:** Move beyond the current `.restart.md` renderer convention (ADR-009) to proper restart-state nodes in the graph, linked from sessions or conversation domains, that can be compiled into context like any other memory.

**Rationale:** The architecture treats restart state as important for long-running work. Current `.restart.md` is explicitly "rendered state, not source of truth."

**Approach notes:**
- Create a `restart_state` node type with strong provenance back to the exchanges and active documents that produced it.
- Provide `tirzah build-restart <session>` and automatic updates on exchange save (configurable).
- Make restart state visible to the memory agent and direct retrieval with appropriate endorsement rules.

**Priority:** Medium
**Related:** ADR-009, sessions/exchanges, `.restart.md` mentions in README, Stage 5 in build-roadmap.

### 5.3 Active Document Curation and Manual Promotion Controls
**Description:** Give operators explicit controls to pin, unpin, promote, or temporarily exclude documents and specific nodes from a session's active set, with full provenance on the changes.

**Rationale:** Active documents are currently populated automatically from `used_node_ids`. This is useful but can be noisy; users need a way to curate the "this document / this thread" context deliberately.

**Approach notes:**
- New CLI commands: `pin-document`, `active-document-edit`.
- Web: simple list with pin/star and "exclude for this session" toggles in developer and (light) work mode.
- Record changes as process steps or endorsement-adjacent events.

**Priority:** Medium
**Related:** `src/tirzah/sessions/active_documents.py`, interaction.py active doc scoping, web UI.

---

## 6. Governance, Trust & Review Workflows

### 6.1 Natural-Language Hints for Generated-Output Review
**Description:** When ingesting LLM output (`output_ingestion`), run a lightweight, local, read-only analysis pass that proposes likely endorsement labels or key claims, presented to the operator as suggestions only (never automatic).

**Rationale:** Output ingestion + explicit review (`review-generated-output`, `endorse-node`) is implemented, but the current path is purely manual. Natural-language endorsement detection is listed as future in the roadmap.

**Approach notes:**
- The analysis must not write endorsement labels itself.
- Present suggestions in the review command output and web review UI with "accept suggestion" as an explicit operator action.
- Keep all provenance pointing back to the originating exchange.

**Priority:** Medium
**Related:** `src/tirzah/sessions/output_ingestion.py`, `src/tirzah/sessions/endorsements.py`, build-roadmap gaps.

### 6.2 Process Enforcement Scaffolding
**Description:** Begin enforcing simple process objects for high-stakes flows (e.g. "any generated output that will be endorsed must go through output-ingestion + review").

**Rationale:** Process objects, runs, and governance are already seeded and used for tracing. Automatic enforcement was deliberately left out of V1.

**Approach notes:**
- Start with advisory mode that logs violations into the process run.
- Later add hard blocks behind a config flag.
- Document the minimal process catalog in `label_definitions` style.

**Priority:** Low-to-Medium (foundational for later agent write autonomy)
**Related:** `src/tirzah/db/governance.py`, process runs in interaction and worker, `docs/governance-schema-plan.md`.

---

## 7. Transparency, Diagnostics & Explainability

### 7.1 Unified, Searchable Activity History
**Description:** Provide a first-class way to search and browse past activity reports (ingestion + answer flows) across sessions without dropping to raw Mongo queries.

**Rationale:** Activity logs and reports are excellent for individual interactions, but there is no good way to answer "show me every time we discussed X topic" or "what happened during last week's ingestion run" from the tools.

**Approach notes:**
- Add `tirzah activity-log --query "..." --since ...` and equivalent API.
- Index key fields from the activity report documents.
- Web: a searchable history tab in developer mode.

**Priority:** Medium
**Related:** `src/tirzah/sessions/activity_reports.py`, ingestion/activity.py, retrieval traces collection.

### 7.2 Per-Node "Why Was This Included?" Inspector
**Description:** From any rendered context or answer exchange, allow drilling into a specific node to see the full set of signals (provenance, usage, trust diagnostics, graph proximity, active document match, lexical score, vector score, etc.) that contributed to its selection or exclusion.

**Rationale:** Traces already contain rich data, but it is scattered. A focused inspector would dramatically improve operator trust and debugging.

**Approach notes:**
- New `node-diagnostic <node_id> --exchange <id>` or similar command.
- Web node inspector panel.
- Must work even for nodes that were ultimately skipped under budget.

**Priority:** High (leverage on existing diagnostics)
**Related:** retrieval/trust, queries, activity reports, web developer tools.

---

## 8. Web UI / Developer Experience

### 8.1 Live Job Progress and Ingestion Dashboard
**Description:** Replace or augment the current polling/refresh model in the Ingestion tab with visible progress for inbox processing, profile backfills, and output ingestion jobs.

**Rationale:** The web UI already exposes job controls, but feedback is coarse. Operators have to manually refresh or switch to CLI for visibility.

**Approach notes:**
- Use existing process-run and job collections as the source of truth.
- Simple Server-Sent Events or short-polling from the static JS.
- Show per-job success/retry/dead-letter counts and links to the affected documents or nodes.

**Priority:** Medium
**Related:** web/app.py ingestion endpoints, embedding_backfill, output_ingestion, process runs.

### 8.2 Visual Document Tree and Node Inspector
**Description:** In Browse/developer mode, render the document tree as an expandable outline (with endorsement and label badges) and allow clicking a node to see full text, provenance, graph neighbors, and usage history in a side panel.

**Rationale:** `show-tree` and `node-context` exist in CLI and are powerful. The web browse surface is currently weaker for deep inspection.

**Approach notes:**
- Keep it vanilla JS + CSS (no new frameworks).
- Support search/filter within the tree.
- Link directly to "compile context from here", "endorse", "semantic candidates", etc.

**Priority:** Medium-High
**Related:** web static files, retrieval queries, sessions/endorsements.

---

## 9. Performance, Scalability & Operations

### 9.1 Background Scheduling for Recurring Jobs
**Description:** Introduce lightweight in-process scheduling (APScheduler was chosen in ADR-007) for periodic profile backfill sweeps, embedding maintenance, or low-priority cleanup, controllable from config.

**Rationale:** Most background work is currently driven by explicit CLI or web "process-*" commands. This works for a developer but is fragile for longer-running operator use.

**Approach notes:**
- Keep all jobs idempotent and resumable (current design is already good here).
- Expose scheduler status via `tirzah scheduler-status`.
- Default: off, so existing explicit control remains the normal path.

**Priority:** Low-to-Medium
**Related:** ADR-007, ingestion/embedding_backfill, db/queue patterns.

### 9.2 Cursor-Based Pagination and Bounded Scans for Large Corpora
**Description:** Add proper cursor or keyset pagination to list/search commands that can return very large result sets (current `limit` + sort is fine for small results but brittle at AMS scale).

**Rationale:** With 175k+ nodes, commands like broad searches or label listings will hit practical limits.

**Approach notes:**
- Introduce `--cursor` / `--next` style output that includes the pagination token.
- Apply consistently to CLI, API, and web listing paths.
- Document the stability guarantees (or lack thereof) when concurrent writes occur.

**Priority:** Medium
**Related:** `list_documents`, `search_nodes`, various `list_*` in governance and sessions, web aggregation queries.

---

## 10. Extensibility, Integration & Ecosystem

### 10.1 Stable Context Envelope + Minimal MCP / Tool Server
**Description:** Publish a stable, versioned "context envelope" format (building on the existing `context_document` and `build-prompt` output) and optionally expose a small local MCP-compatible or JSON-RPC tool surface so external coding agents and CLI tools can request compiled context without going through the full ask flow.

**Rationale:** The project already intends to be a memory backend for coding agents and other tools (see practical-applications.md and integration minimums in the roadmap). The current `ask` / `build-prompt` surface is good but not the easiest for agent integration.

**Approach notes:**
- Define the envelope in a small schema document.
- Provide a `--format envelope` or dedicated `get-context-envelope` command.
- The tool surface should be read-only and go through the same identity + budget logic.

**Priority:** High (ecosystem leverage)
**Related:** `build-prompt`, render context, `src/tirzah/sessions/interaction.py`, `docs/practical-applications.md`, roadmap integration minimums.

### 10.2 Export / Import for Subgraphs and Sessions
**Description:** Add commands to export a document (or set of nodes + edges) as a portable bundle (with sources + provenance) and to import it into another Tirzah instance or as a linked domain.

**Rationale:** Useful for sharing curated memory, creating test fixtures from real data, and multi-machine workflows. The archive already contains the raw sources.

**Approach notes:**
- Bundle format can be a directory or a single `.tar.gz` with a manifest.
- Must preserve all endorsement, review, and graph edge provenance.
- Provide a "dry-run import" that shows label and domain collisions.

**Priority:** Medium
**Related:** archive paths, rebuild logic, domains/registry.py.

### 10.3 Kafka-Backed Hoglah Delivery Adapter
**Status:** Future feature; details blocked until Hoglah defines its Kafka capability surface.

**Description:** Add optional Tirzah support for routing Hoglah-bound answer and embedding jobs through Kafka once Hoglah exposes Kafka producers/consumers or a compatible transport contract.

**Rationale:** Hoglah is already the durability boundary for queued model work. Kafka could provide a higher-throughput, multi-process, observable delivery layer for Hoglah-backed deployments while preserving Tirzah's local-first submitter role.

**Approach notes:**
- Treat Kafka as an optional Hoglah transport, not as a replacement for Tirzah's Mongo-backed memory store or review queues.
- Keep Tirzah as a pure submitter: Tirzah should publish/consume only the minimal Hoglah job/result envelope that Hoglah specifies, while the Hoglah daemon owns execution, retry semantics, and any Ollama/worker calls.
- Preserve existing `hoglah_delivery: poll` and `hoglah_delivery: callback` behavior as the default, simpler local modes.
- Require explicit config for Kafka bootstrap servers, topic names, consumer group IDs, and delivery timeouts; do not enable Kafka implicitly.
- Keep the local memory interface boundary intact: Kafka transport must remain local/operator-configured IPC for memory-critical answer/embedding paths, not a hosted-service dependency.
- Add smoke tests only after Hoglah's Kafka contract is stable, using a local single-node Kafka broker.

**Priority:** Low-to-Medium until Hoglah Kafka support exists; can become Medium if multi-worker Hoglah deployment becomes a near-term need.
**Related:** `src/tirzah/adapters/hoglah_runtime.py`, Hoglah runtime config, README "Routing via Hoglah", local Kafka operator setup.

---

## 11. Evaluation, Testing & Benchmarking

### 11.1 Brute-Force Baseline Comparison Harness
**Description:** Add an optional evaluation mode that can run the same query through Tirzah retrieval vs. a simple "load top-N documents or chunks" baseline and produce side-by-side token usage + answer quality traces (for manual inspection).

**Rationale:** Explicitly listed as part of Stage 3 minimum build and evaluation needs in the requirements. Currently there is no built-in way to quantify the value of the graph memory approach.

**Approach notes:**
- Must be offline and local.
- Do not require external hosted models for the baseline itself.
- Output should be reproducible bundles that can live in `tests/fixtures` or a new `evals/` area.

**Priority:** Medium
**Related:** roadmap Stage 3, open-questions.md (evaluation items), test suite.

### 11.2 Expanded Deterministic Test Corpus and Regression Fixtures
**Description:** Grow the V1 smoke fixture into a small but representative set of documents that exercise tree shape, labels, semantic candidates, active documents, and retrieval edge cases. Use it for golden-context regression tests.

**Rationale:** The current test suite is strong on unit coverage; golden end-to-end retrieval behavior on realistic (but small) data would catch regressions in ranking, budgeting, and context construction as the system evolves.

**Approach notes:**
- Keep everything public-domain or synthetic.
- Tests assert on structural properties ("X ancestors were included", "no rejected nodes", "budget respected") rather than exact token strings.

**Priority:** Medium
**Related:** `tests/fixtures/`, `test_retrieval_queries.py`, `test_interaction.py`, v1-release-candidate-smoke.md.

---

## 12. Documentation & Project Hygiene

### 12.1 Living "Memory Health" Report Command
**Status:** Partially implemented in V1.2. `tirzah memory-health` and `GET /api/memory-health` now report corpus totals, profile and embedding coverage, endorsement distribution, queue counts, and attention items.

**Description:** Extend the implemented memory-health report with a first-class web panel, recent failure-rate checks, semantic edge count vs. candidate backlog, and documents with unusually high superseded tree counts.

**Rationale:** Operators working with large imported corpora (AMS, external) need an at-a-glance view of the state of the memory layer without writing custom queries.

**Approach notes:**
- Keep it fast and safe to run frequently.
- Preserve both human text and machine JSON outputs.
- Surface deeper actionable follow-ups ("Run `queue-profile-backfill --label ams_domain`", "Inspect unusually large superseded trees").

**Priority:** Medium (high operational value)
**Related:** existing status/inspection commands, governance listings, profile backfill job queries.

### 12.2 Improved Contributor and Operator Onboarding
**Description:** Expand `CONTRIBUTING.md` and add a short "Operator Quickstart" that walks through a complete local setup (Mongo + optional Ollama worker + first corpus + first ask) with verification commands.

**Rationale:** The project has excellent design documentation but the barrier to a first successful real-world run (especially with real embeddings and Ollama) remains higher than ideal for a local-first tool.

**Priority:** Medium
**Related:** README, existing smoke docs, config.example.yaml comments.

---

## Prioritization Guidance

**Recommended near-term post-V1 focus areas (high leverage):**
- 1.1 LLM-assisted chunking (review-gated)
- 2.1 Hybrid vector search
- 3.1 Actionable trust/temporal ranking
- 4.1 Robust planner for agentic
- 5.1 Last prompt iteration / continuity records
- 7.2 Per-node "why included" inspector
- 10.1 Stable context envelope for external agents

**Items that strengthen foundations for later stages (Stages 4–7):**
- Semantic graph improvements (3.x)
- Process enforcement and governance (6.2)
- Restart state nodes (5.2)
- Evaluation harness (11.1)

Treat this list as input to design slices rather than a commitment. Any change that would weaken source authority, reduce transparency, or bypass review gates for endorsement/relationships should be discussed against the principles first.

## Maintenance
- Update this document when a proposal is implemented (move to "Implemented" or remove and note the version).
- Add new items only when they represent clear strengthening of existing features rather than entirely new product directions (use the lifecycle and roadmap docs for the latter).
- Keep the date current on significant revisions.
