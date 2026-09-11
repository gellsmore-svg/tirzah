# Consolidated Requirements And Design

Date: 2026-06-15

Status: canonical working product/design document. This document consolidates the current Tirzah requirements, implemented design, inferred decisions, and open work into one review entry point. The older detailed documents remain useful supporting records, but this file should be the first place to look when deciding what to build next. The product was renamed from Mnemosyne to Tirzah because the previous name conflicted with another memory-oriented GitHub project and was harder to use consistently.

Review artifacts from the June 14 implementation audits are filed under `docs/reviews/`. Their findings are treated as evidence behind this document, `docs/v1-known-limitations.md`, and `docs/improvements-and-enhancements.md`; those living documents are the active task and product sources.

Version boundary: V1 is defined in `docs/build-roadmap.md` as the local memory workbench release. V1 is intentionally narrower than the full cognitive-memory design: it prioritizes dependable local ingestion, provenance, retrieval, transparent answer flow, review controls, and session continuity, while leaving LLM-driven ingestion, REM consolidation, semantic-map sense clusters, automatic endorsement, and full traversal feedback for later releases.

## Product Intent

Tirzah is a local-first memory engine for long-running LLM work. It preserves source material, builds document and semantic graph structures, retrieves evidence for questions, records continuity, and explains its own behavior in language a human can inspect.

Tirzah should not become a closed chatbot. It should become a memory backend that can serve:

- the current local web UI;
- CLI workflows;
- coding-agent support;
- web import and temporary internet context tools;
- voice transcript capture;
- future FOSS agent or knowledge-management clients.

The durable memory system owns source preservation, provenance, graph storage, retrieval, context construction, review state, and endorsement. External tools may help capture, transcribe, browse, or execute actions, but they should not bypass memory governance.

## Relationship To Mahalath

Tirzah and Mahalath should be treated as related local projects that use LLMs as recursive semantic workers, not merely answer generators.

Tirzah preserves what matters: source continuity, provenance, durable memory, retrieval state, reviewed relationships, and future context.

Mahalath clarifies what it means: ontology, interpretive structure, semantic refinement, and conceptual precision.

Together they form an external semantic-reasoning layer. The purpose is to reduce drift, raise precision, and translate human language into richer machine-native context without pretending the machine is conscious.

This relationship does not imply immediate code integration. The current implementation direction remains to treat Mahalath as a linked local corpus/domain first, then evaluate which ontology, process, or semantic-clarification structures should enrich Tirzah.

## Core Principles

### Source Authority

Source documents are the authority. Repository state should be disposable and reproducible from preserved sources plus reviewed generated artifacts.

The system must preserve source text and provenance. It must not silently rewrite, compress, summarize, or clean source material during ingestion unless that transformed material is explicitly stored as derived content.

### Memory As Cognitive Infrastructure

Memory is not passive storage queried by cognition. Memory should participate in:

- semantic weighting;
- contextual interpretation;
- procedural enforcement;
- identity-aware retrieval;
- trust evaluation;
- continuity of reasoning;
- collaboration and review.

The current implementation is still scaffolded, but future design should treat memory as active cognitive infrastructure rather than a flat RAG store.

### Transparency First

The user should be able to understand what happened without reading Python, raw JSON, or database records.

For every important flow, the system should expose:

- what was attempted;
- what information was used;
- which Python functions and LLM calls were involved;
- what decisions were made;
- what failed or was omitted;
- what the system saved for continuity.

JSON traces may remain available, but the default surface should read like a clear application log.

### Quality Before Speed

For the foreseeable future, quality is more important than performance. The system should prefer better interpretation, context construction, relationship extraction, provenance, and reviewability even when processing takes longer.

Optimization comes after the high-quality baseline is understood.

### Local Memory Interface Boundary

HTTP is allowed for the human web interface and may be used for an optional final hosted answer-model call. HTTP must not be used for ingestion, retrieval, memory-agent tool orchestration, Python memory tools, or embedding generation for repository memory.

Python remains the stateful orchestrator for memory work. The memory-agent LLM may request tool calls, but Python validates those requests and executes local Python functions directly. Ingestion and retrieval model/tool calls must use local non-HTTP execution paths.

## User-Facing Requirements

### Ask Workspace

The Ask workspace is the primary experimentation surface. It must support real use, not just debugging.

Requirements:

- Prompt, Response, and Activity Log should be visible side by side on desktop.
- Prompt / Trace is supporting detail and should sit below the main row.
- The Ask button should live directly in or below the prompt panel.
- The activity log should start updating as soon as the request begins.
- The returned timeline should show each Python step and LLM handoff in order.
- LLM handoffs should expand to show the human-readable prompt/context package sent to the model.
- Raw JSON should be collapsed behind technical details.
- Low-intent prompts such as `hello` must not retrieve arbitrary corpus material.

### E-Paper Display Mode

The UI must support a Dasung-style 60Hz e-paper display mode.

Requirements:

- high-contrast light panels;
- no dark trace block;
- moderate text scale;
- proportional side-by-side columns;
- low visual texture;
- predictable layout at common desktop browser widths.

The current URL is:

```text
http://127.0.0.1:8765/?display=epaper
```

### Workspace Separation

Operational controls must not compete with question-answering.

The web UI should remain separated into:

- Ask: prompt, answer, activity log, and model selection for normal work.
- Browse: node search, documents, exchange history.
- Ingestion: source staging, inbox processing, semantic-edge review, ingest jobs.

### Work Mode And Developer Mode

The UI must work as a clean LLM wrapper client, not only as a developer console.

Default work mode should show only what a user needs for normal work:

- session selection;
- prompt;
- Ask button;
- model choice;
- response;
- plain activity log.

Developer mode should be available through a toggle. It may reveal:

- Browse and Ingestion tabs;
- focus-node override;
- adapter selection;
- retrieval-mode override;
- raw process trace;
- technical JSON report;
- review/debug controls.

Human transparency is not developer-only. The readable activity log remains visible in normal work mode. Developer mode is for raw diagnostics and operational controls, not for basic understanding.

### Prompt Processing Pipeline

Tirzah should provide a structured prompt processing pipeline between the user interface and the target answer model.

This is a core product distinction from generic wrapper tools such as Ollama's chat interface. The user prompt should not be passed straight to a model with only a thin adapter around it. Once submitted, the prompt should move through deliberate local stages before any answer model receives it.

Required stages:

- receive the submitted prompt and selected session/domain state;
- classify prompt intent, including low-intent, normal conversation, repository question, active-document reference, process request, and ingestion/review operation;
- select or propose a process when a process is relevant;
- resolve project domain and conversation domain;
- decide whether repository memory should be used;
- ask the memory-agent/controller to gather or reject context when agentic retrieval is active;
- validate tool calls, context use, budget, and provenance locally;
- assemble the final answer context package;
- produce a readable activity log while the pipeline is running;
- persist the prompt cycle, selected context, answer, and unresolved continuation items.

The answer model should receive a complete context package, not raw internal state. The package should make the controller decision, evidence, constraints, and expected answer behaviour clear.

The implemented front-end boundary now begins with a recursive Cairn planning
wrapper (`tirzah.planning.recursive`). It creates a first-pass versioned plan,
invokes the existing retrieval/answer pipeline, and revises the same plan from the
resulting evidence. Revisions are bounded, persisted separately from graph memory,
and exposed to the work-mode UI. Later information can revise a persisted plan
through the plan revision API. Python validates the complete plan on every
revision and retains authority over tools, side effects, budgets, and stopping.

The target modular boundary is a dedicated prompt-pipeline module. The current implementation still spreads this across `sessions`, `retrieval`, and adapter calls. That is acceptable as a scaffold, but should be refactored so prompt intake, process selection, context strategy, package assembly, LLM handoff, and continuity persistence are testable as separate steps.

### Product Naming

The previous repository/package name was Mnemosyne. The target product name is Tirzah.

Rename requirements:

- GitHub repository should be renamed from `mnemosyne` to `tirzah`.
- Python package, CLI command, UI labels, config names, and documentation should move to Tirzah naming.
- The old `mnemosyne` CLI command and Python import path may remain as temporary compatibility paths during the transition.
- Compatibility paths must warn or document that Tirzah is the preferred name.
- The rename should be done as a dedicated implementation slice, separate from feature work.

### Project And Conversation Domains

Memory should support explicit working boundaries.

A project domain groups source material, conversation domains, processes, reviewed relationships, current assumptions, and continuity records for a project.

Examples:

```text
project_domain: tirzah
project_domain: mahalath
project_domain: rs5_clause_collection
```

A conversation domain groups memory that belongs to a particular conversation thread or work session. A project domain may contain one or more conversation domains. Conversation-domain memory should not automatically become project-domain memory; promotion should be explicit or governed by a process.

Required fields for future nodes/exchanges:

- project domain;
- conversation domain;
- source conversation/session ID where applicable;
- promotion status: conversation-only, project-candidate, project-accepted, rejected.

Implementation status:

- Project-domain and conversation-domain registries exist under `src/tirzah/domains`.
- Domain collections are indexed.
- Saved exchanges carry `project_domain_id` and `conversation_domain_id`.
- `/api/ask` accepts optional project/conversation domain IDs.
- The default project domain is `tirzah`.
- The default conversation domain follows the session ID.

Open implementation items:

- domain selection/listing controls in the UI;
- CLI/API commands for browsing domains;
- domain fields on nodes and future continuity records;
- promotion rules from conversation-domain memory into project-domain memory.

### Process Objects

Processes must become first-class product elements.

A process is an invokable fixed or semi-fixed procedure with:

- name;
- purpose;
- trigger rules;
- ordered steps;
- optional steps;
- required evidence;
- allowed tools;
- stopping conditions;
- exception rules;
- human approval points;
- activity-log output.

The LLM wrapper may use processes in three ways:

- always apply a process for selected prompt types;
- propose a process when relevance appears likely;
- follow a process explicitly requested by the user, while explaining if the process appears unsuitable.

Python remains responsible for validating process availability, enforcing required steps, recording exceptions, and preserving the process trace. The LLM may select, request, or execute steps through the local tool interface, but it does not silently bypass process rules.

### Last Prompt Iteration Record

Each prompt cycle should save a continuity record containing:

- submitted prompt;
- interpreted intent;
- selected project domain;
- selected conversation domain;
- retrieved chunks;
- rejected chunks;
- selected process, if any;
- LLM tool calls;
- final context package;
- answer;
- unresolved follow-up items.

The UI should expose the last used chunks/context records in a simple continuity panel so a user can understand and continue the prior thread.

Implementation status: partial. Exchange records carry domain IDs, and answer-save now writes a `session_continuity` prompt-iteration record with prompt, domains, exchange ID, focus/used/active node IDs, prompt budget, context metadata, controller decision, evidence summary, answer preview, and process-trace summary. `tirzah session-continuity`, `/api/session-continuity`, and the Ask workspace continuity panel expose the latest/recent continuity state. Rejected-chunk expansion, full final context package display, unresolved follow-up capture, and use of these records to seed follow-up prompts remain open.

### Mahalath Integration

Mahalath should first be treated as a linked local corpus/tool source rather than merged into the codebase.

Initial integration requirements:

- import Mahalath source material as a project domain or linked external domain;
- build text similarity profiles for Mahalath content;
- expose Mahalath documents/nodes in retrieval and review surfaces;
- inspect Mahalath ontology material for possible process/domain enrichment;
- only reuse Mahalath code after the corpus and ontology value is understood.

## Answer And Retrieval Requirements

### Direct Retrieval

Direct retrieval should be conservative. It should search repository memory only when the prompt appears to be a substantive repository question, an active-document reference, or a focused-node request.

It currently supports:

See `docs/improvements-and-enhancements.md` (Retrieval Precision & Context Construction and related sections) for concrete proposals to strengthen the current direct retrieval implementation.

- exact focus-node use;
- active-document scoping for references such as `this document`;
- lexical and near-match search;
- deterministic intent classification for empty, low-intent, active-document-reference, generic, and repository-query prompts;
- a minimum direct context match score before broad corpus search can select a node;
- a `controller_decision` trace/prompt object that explicitly marks direct retrieval as a deterministic scaffold and names the target owner as the memory-agent/controller;
- source-file fallback for active documents;
- no-context handling for low-intent conversational prompts.

Required next improvements:

- better explanation when retrieval is skipped;
- less dependence on lexical regex candidate collection.

### Agentic Retrieval

Agentic retrieval separates the memory-agent from the final answer model.

The memory-agent should not answer the user. It should iteratively gather memory by calling read-only Python tools, inspect observations, and stop when context is sufficient, clearly insufficient, or no useful read-only call remains.

Agentic answers emit the same `controller_decision` concept used by direct mode, but with `mode: agentic` and `current_owner: memory_agent_controller`. The memory-agent can propose this decision when it stops, including action, reason, and confidence; the runtime validates and enriches that proposal with actual tool counts, context inclusion counts, and retrieval status. If the proposed action conflicts with actual tool output, such as claiming repository context was used when no context records were included, the runtime corrects the action and records the correction. Controller decisions also carry validation issues for malformed action, missing reason, impossible context use, or invalid confidence. Direct and agentic final answer prompts include a compact Controller Decision section before retrieved context/tool results so the answer model sees the controller context strategy. This keeps UI/reporting aligned with the target architecture while the deterministic direct path remains a scaffold.

Current allowed memory-agent tools:

- `search_nodes`;
- `compile_context`;
- `get_node_context`;
- `get_document`;
- `get_document_tree`;
- `get_graph_edges`;
- `expand_proximity`;
- `expand_graph_paths`;
- `semantic_candidates`;
- `list_active_documents`;
- `list_documents`.

Tirzah validates memory-agent tool calls, executes them through the Python runtime, records observations, and feeds compact summaries back into later planner iterations. If the LLM makes an invalid call, the tool layer returns an instructional error with usage guidance and a repair instruction so the next iteration can recover.

Failed tool-call guidance is preserved in memory-agent history and repeated in a dedicated repair-guidance section of the next planner prompt. The user-facing activity log also summarizes these failures in plain language so recovery is visible without reading the raw JSON trace.

When the memory-agent stops, it may return a bounded `context_proposal` containing selected node IDs, rationale, and organization hints. Tirzah treats this as a retrieval-controller proposal, not unchecked authority. The Python runtime validates node IDs, enforces budgets, ignores invented IDs, and uses the proposal only to prioritize matching context records.

### Query Assembly

Python builds a deterministic query assembly artifact used by both direct retrieval and agentic prompts.

Current fields include:

- lexical terms;
- exact phrases;
- named anchors;
- bounded near-match terms;
- fallback probe suggestions;
- active-document vocabulary hints.

This is the current bridge between deterministic retrieval, typo tolerance, UI diagnostics, and LLM planner guidance. It is not a substitute for embeddings, graph traversal, or semantic-map retrieval.

### Context Construction

Context construction must produce a bounded, inspectable package for the final answer model.

Current context package includes:

- rendered Markdown context for current local models;
- structured `context_document` metadata;
- structured evidence summary with direct/agentic evidence counts, included node IDs, source fallback state, and source documents;
- included/skipped records;
- used node IDs;
- query diagnostics;
- selected tool outputs;
- context proposal metadata when present.

The final model should see source evidence before diagnostics. Diagnostics help the model understand retrieval behavior, but source content must remain primary.

## Ingestion Requirements

### Current Ingestion

Current ingestion is deterministic scaffold ingestion for Markdown and text files. It preserves source text, archives accepted files by checksum, rejects duplicates, writes documents/trees/nodes to MongoDB, and records queue/job state.

Direct and queued ingestion now return both a structured `activity_report` and a plain `activity_log`. The log summarizes source path/type, checksum, duplicate or failure status, selected origin date and candidate count, adapter used, node and relationship-hint counts, repository writes, archive/processed paths, and operator-facing recovery notes. Completed queue jobs persist the readable log in the stored job result. The web inbox processor aggregates per-job logs into an `Inbox Processing Activity Log` and shows the plain text first.

The Ingestion tab now gives staged source files chronology context before processing: selected origin date, date source, date-candidate count, and origin-date ordering are visible from the inbox browser.

The Ingestion tab also exposes recent ingestion epochs and ingestion process runs through `/api/ingestion/status`, making the latest/current epoch and its dated-document coverage visible without requiring direct database inspection.

Current ingestion is not yet the target LLM-assisted semantic ingestion pipeline.

### Target Ingestion Pipeline

The target ingestion flow is:

```text
document
  -> source analysis
  -> metadata/date extraction
  -> semantic analysis
  -> chunk generation
  -> relationship analysis
  -> identity relevance analysis
  -> sensitivity analysis
  -> trust scoring
  -> temporal analysis
  -> process tagging
  -> graph placement
  -> semantic weighting
  -> storage
```

Every ingestion run should produce a human-readable activity log covering:

- source analysis;
- metadata and earliest credible date;
- concepts/entities detected;
- nodes created or updated;
- relationships detected or proposed;
- LLM calls made and why;
- repository writes;
- failures, retries, and review requirements.

### Repository Refresh

The repository should be treated as reproducible from source documents and reviewed generated artifacts.

Required operating model:

```text
Source Documents
  -> Fresh Ingestion
  -> Semantic Analysis
  -> Relationship Construction
  -> Inference Generation
  -> Repository Build
```

The system needs a controlled rebuild workflow that can:

- clear disposable generated repository content;
- re-ingest all source content;
- rebuild semantic relationships;
- rebuild inferred content;
- rebuild supporting node structures;
- tag runs with ingestion epochs;
- compare or supersede earlier generated content.

Current rebuild behavior uses ingestion epochs as the versioning surface. There are two rebuild modes with different guarantees:

- **Full rebuild (default, `mode=full`).** Inserts a new active tree/node set and marks earlier tree/node records as `superseded` rather than deleting them. Node ids change. Human-reviewed semantic edges are carried onto the new node with identical content (same content hash and structural label); edges whose endpoint has no such counterpart stay on the superseded node and are flagged `needs_review`.
- **Targeted rebuild (`rebuild-document --diff-only`, `mode=diff`).** Mutates the active tree in place and keeps node ids for matched nodes. Matching is content-first (content hash, then position, then title), so a kept id always holds the same text it held before. When a kept node's content does change, its endorsement, usage score, `last_used_at` and continuity-critical flag reset (prior values kept in `content_change_history`), and an operator summary moves into summary history. Machine-generated graph edges for the tree are deleted and regenerated; human-reviewed edges are never deleted, and are flagged `needs_review` when an endpoint changed or was superseded. Removed nodes are marked `superseded`.

Both modes merge source metadata rather than replacing it: an operator-reviewed origin date survives, `origin_date_history` and operator date candidates are kept, every active node is restamped so the document has one chronology, the stored checksum is recomputed from the file actually read (the previous one goes to `source.previous_checksums`), and origin-date or checksum changes are reported in `source_changes`. Rollback on failure is non-destructive (only rows the failed write added are deleted; every prior row is restored in place) but is not a MongoDB transaction; see `docs/v1-known-limitations.md`.

Normal search and document-tree views prefer active records. Explicit epoch comparison, audit browsing of superseded trees, and garbage collection remain open work.

### Chronological Corpus Processing

Large historical corpora, especially AMS / Relational Substrate / RS5 Clause material, should be ingestible in likely historical order.

Earliest credible origin date priority:

1. Explicit date inside document content.
2. Date embedded in filename.
3. Original file creation date, when preserved by the source acquisition path.
4. Original file modification date, when preserved by the source acquisition path.

The earliest credible origin date should generally win, because the goal is to reconstruct the development of ideas over time.

Current ingestion records now store source-date candidates plus the selected `origin_date` and `origin_date_source` in document source metadata. The implemented extractor covers explicit content date markers, filename dates, filesystem creation/change dates, and filesystem modification dates. Filesystem dates are weak fallback evidence: on Linux `ctime` is inode change time, and web-staged uploads usually reflect import time rather than authorship time. CLI folder ingestion now builds a chronological source plan ordered by selected origin date and then path, reports unreadable files as rejected source-plan entries instead of aborting the whole batch, and reports the first planned sources in `source_order` so operators can inspect ordering before treating a run as authoritative. UI comparison across dated corpus runs remains open work.

## Graph, Semantics, And Review Requirements

### Graph Model

The graph must support:

- structural parent/child `contains` edges;
- reviewed semantic edges;
- conceptual relationships;
- causal relationships;
- contradiction relationships;
- reinforcement relationships;
- hierarchy relationships;
- dependency relationships;
- process relationships.

Relationship records should support:

- weight;
- confidence;
- trust;
- temporal relevance;
- provenance;
- identity visibility.

Current implementation includes structural edge backfill, read-only edge lookup, one-hop proximity expansion, bounded path expansion, semantic-candidate diagnostics, candidate queueing, and reviewed semantic-edge promotion.

Automated semantic relation extraction is still deferred.

### Generated Output Review

Saved LLM answers are queued as pending output-ingestion work. They can be processed into unreviewed generated-output nodes and later reviewed.

Generated output must not become trusted source memory automatically.

Review states include:

- unreviewed;
- implicit endorsed;
- explicit endorsed;
- rejected.

Future work should infer candidate endorsements and relationships from generated output, but writes must remain reviewable.

## Governance And Identity Requirements

### Identity Layers

Agents should have identity beyond prompt text.

Identity should define:

- semantic scope;
- trusted corpus;
- exclusions;
- behavioral expectations;
- process obligations;
- access permissions;
- weighting preferences;
- governance rules.

The target architecture includes:

- shared identity layer;
- domain identity layer;
- restricted identity layer.

Current implementation has read-only identity/governance records, seeded defaults, CLI/API listing and lookup, memory-agent identity prompt summaries, and agentic search exclusions.

### Process Enforcement

Processes should become enforceable semantic objects.

Target process objects should support:

- mandatory acknowledgement;
- execution tracking;
- procedural validation;
- step enforcement;
- escalation logic;
- audit trails;
- exception proposals;
- approval pathways.

Current process-run persistence is observational. Answer and ingestion flows create/update process runs where possible, but process rules are not yet enforced.

### Trust And Temporal Weighting

Trust and relevance should not be static.

Each semantic object should support:

- creation timestamp;
- last verified timestamp;
- last accessed timestamp;
- recency weighting;
- frequency weighting;
- confidence weighting;
- contextual persistence weighting.

Trust/temporal diagnostics are visible in retrieval traces. By default they do not affect ranking. With `runtime.trust_ranking_enabled` (opt-in, off by default) they re-rank the pre-truncation candidate pool as a bounded secondary signal, scored from the stored node (explicit trust score, verification flags, origin/created dates). The seeded `default_balanced` profile uses a 1825-day decay half-life so recency participates; undated nodes get neutral recency, and each ranked row reports `temporal_decay_active`.

## Internet-Assisted Reasoning Requirements

Internet content may be used as temporary context when local memory is insufficient, but it must not automatically become durable memory.

Target workflow:

```text
User Question
  -> Initial Reasoning
  -> Determine Internet Needed
  -> Internet Retrieval
  -> Context Enrichment
  -> Additional LLM Pass
  -> Final Response
```

Promotion states:

- Temporary Context: used only for the current answer.
- Candidate Knowledge: stored separately for review.
- Permanent Knowledge: ingested only after review criteria are satisfied.

This protects the durable repository from unstable, low-quality, or unreviewed web material.

## Implemented Runtime Design

Current runtime shape:

- Python package under `src/tirzah`;
- local MongoDB persistence;
- FastAPI backend;
- static HTML/CSS/JS frontend;
- CLI commands for ingestion, retrieval, graph inspection, governance, sessions, process runs, and history;
- local answer adapters, including mock and Ollama CLI;
- direct and agentic retrieval modes;
- separate memory-agent model configuration;
- process traces, activity reports, and activity logs.

Current answer flow:

1. UI or CLI submits query, session, focus node, adapter/model, and retrieval mode.
2. Python records the prompt and starts an `answer_query` process run where possible.
3. Low-intent prompts are routed to no-context direct answering.
4. Direct or agentic retrieval gathers context.
5. Invalid memory-agent tool calls return usage and repair guidance.
6. Memory-agent may propose context ordering.
7. Python validates, budgets, and assembles final context.
8. Answer model receives the final prompt/context package.
9. Exchange is saved.
10. Used-node scoring, active-document tracking, and output-ingestion queueing run.
11. API returns answer, process trace, activity report, and plain-English activity log.

## Open Risks And Gaps

- Ingestion is still deterministic scaffold ingestion, not LLM-assisted semantic ingestion.
- Full repository refresh/rebuild is not yet a product workflow.
- Direct retrieval still needs explicit intent classification and relevance thresholds.
- Agentic retrieval is read-only and does not yet perform full semantic-map traversal.
- Context documents are scaffolded and do not yet match the full target schema.
- Trust/temporal diagnostics are explanatory only.
- Process runs are observational only; process enforcement is not active.
- Internet-assisted reasoning and candidate knowledge promotion are planned but not implemented.
- Real server-pushed per-step streaming is not implemented; the UI shows immediate client-side milestones and then the returned trace.
- Ingestion logs currently explain deterministic ingestion runs; they do not yet include LLM ingestion call traces because LLM-assisted ingestion is not implemented.
- Product naming remains unresolved because `Tirzah` collides with existing AI memory projects.

## Near-Term Priorities

1. Extend the ingestion UI test bench around readable logs, run inspection, and epoch visibility.
2. Add controlled repository refresh/rebuild with ingestion epochs.
3. Introduce the semantic substrate and reviewed relationship candidates needed for higher-quality ingestion.
4. Continue moving context strategy from deterministic direct retrieval toward the memory-agent/controller interface.
5. Expand context-document structure toward the target schema.
6. Begin higher-quality semantic ingestion using local LLM calls.
7. Continue UI refinement through real use on standard and e-paper displays.
8. Defer internet temporary context and candidate-knowledge boundaries until semantic candidates and provenance are stronger.

## Supporting Documents

- `docs/current-product-requirements-and-design.md`: recent product/UI requirements and implemented behavior.
- `docs/lifecycle-next-phase-requirements.md`: next-phase transparency, quality, repository refresh, internet, chronology, and multi-corpus requirements.
- `docs/requirements-design-addendum.md`: implementation decisions and inferred requirements accumulated during development.
- `docs/agentic-retrieval-process.md`: detailed current Python/LLM sequence for agentic retrieval and final answer generation.
- `docs/governance-schema-plan.md`: planned identity, governance, process, and trust schemas.
- `docs/tirzah-cognitive-architecture-draft.md`: forward-looking governed cognitive architecture concept.
- `docs/practical-applications.md`: possible application lanes and FOSS integration criteria.
- `docs/development-plan.md`: active ingestion-first implementation sequence covering ingestion epochs, non-destructive rebuilds, chronology, ingestion logs, UI test-bench work, semantic substrate, LLM-assisted ingestion, and ranking/trust integration.
- `docs/improvements-and-enhancements.md`: living post-V1 list of enhancements to strengthen current features (hybrid search, trust effects in ranking, agentic planner robustness, continuity records, context envelopes, per-node diagnostics, etc.).
