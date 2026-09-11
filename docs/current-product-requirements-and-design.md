# Current Product Requirements And Design

Date: 2026-05-29

Status: historical input, not authoritative. This document was folded into `docs/consolidated-requirements-and-design.md`, which is the single review entry point (see `docs/requirements-index.md`); where the two differ, the consolidated document wins. It predates the rename from Mnemosyne to Tirzah and has been updated only for naming.

## Product Intent

Tirzah is a local-first memory engine. It should preserve source material, build a graph of documents and semantic objects, retrieve context for questions, and explain what it did in language a human can inspect without reading code or JSON.

The system is not intended to be a single closed chatbot. It should become a memory backend that can serve a web UI, CLI agents, coding support, web importers, voice transcript tools, and future FOSS integrations.

## Current Product Requirements

### Human Experimentation UI

- The default screen must support real experimentation with prompt, response, activity log, and process trace visible together.
- The Ask workspace must place Prompt, Response, and Activity Log side by side on desktop. Prompt / Trace is supporting detail and should sit below the primary row.
- The UI must support an e-paper display mode for Dasung-style monitors, optimized for low refresh, high contrast, moderate text scale, proportional side-by-side panels, and clear task separation.
- Operational controls must not compete with the question-answering workflow.
- The UI must separate:
  - Ask: prompt, answer, activity log, trace.
  - Browse: node search, recent documents, exchange history.
  - Ingestion: source staging, inbox processing, semantic-edge review, ingest jobs.
- The user-facing activity view must be plain English. JSON may exist as an expandable technical detail, but it must not be the normal way to understand what happened.
- The activity log should read as an operator timeline, not a forensic report. It should update as soon as a request starts, then show each Python step and LLM handoff in order when the run completes.
- LLM handoff entries should expand to show a human-readable copy of the prompt/context payload sent to the model.
- Browser caching must not leave the user unknowingly on an old static layout after UI changes.

### Answer Behavior

- Low-intent conversational prompts such as `hello` must not retrieve arbitrary corpus material.
- When no useful memory context is available, the system should answer from the submitted prompt with an explicit no-context runtime fact rather than inventing repository relevance.
- Direct retrieval should be conservative enough that generic prompts do not become accidental corpus searches.
- If context is used, the activity log must explain:
  - what question was understood;
  - what context was selected;
  - why retrieval did or did not use repository memory;
  - what model/adapter generated the answer;
  - what system functions ran.

### Ingestion And Repository Rebuild

- Source documents are the authority. Runtime repository content should be reproducible from source documents.
- The system needs a clear mechanism to clear and rebuild generated repository content from source material.
- Ingestion should favor quality over speed while the baseline is being established.
- Ingestion must preserve source text and provenance instead of silently cleaning, summarizing, or rewriting it.
- Every ingestion job should produce an understandable activity log covering source analysis, semantic processing, repository actions, LLM activity, and system actions.
- Current direct and queued ingestion now return a structured activity report plus a plain activity log; completed queue jobs retain that log, and the Ingestion tab shows the plain inbox-processing log first so operators do not have to read JSON for normal inspection.
- The Ingestion tab shows staged source origin dates, origin-date evidence, and date-candidate counts so chronological corpus ordering can be inspected before processing.
- The Ingestion tab shows recent ingestion epochs, dated-document coverage, and recent ingestion process runs so the operator can see which epoch is current and whether ingestion workflows completed or blocked.
- Chronological corpus ingestion must determine the earliest credible origin date by priority:
  1. explicit date inside document;
  2. date embedded in filename;
  3. original file creation date, when preserved by the source acquisition path;
  4. original file modification date, when preserved by the source acquisition path.
- Filesystem timestamps are weak fallback evidence because staged uploads and copied files can reflect import time rather than authorship time.
- Large corpora such as AMS / Relational Substrate / RS5 Clause material should be ingestible in likely historical order.

### Retrieval And Context Construction

- Retrieval must evolve beyond flat lexical search or profile-based similarity search into graph-aware, identity-aware, governed context construction.
- The current deterministic lexical/fuzzy sidecar is an interim aid for predictable tests, typo tolerance, and cold-start behavior.
- Query assembly should remain a shared contract between deterministic Python retrieval and LLM planner calls.
- Retrieval should support:
  - exact document/node lookup;
  - lexical and near-match search;
  - active document scoping;
  - graph edge lookup;
  - one-hop proximity expansion;
  - bounded graph path expansion;
  - reviewed semantic candidate inspection;
  - context budget enforcement.
- Retrieval traces should expose enough detail for the user to understand why context was chosen.
- If an LLM calls the Python tool interface incorrectly, Python should return an instructional error that explains how to repair the call, allowing the LLM to recover in the next iteration.

### Product Vocabulary And Implementation Detail

- User-facing language must name the product element before naming the current implementation detail.
- A text similarity profile is a product element. An embedding vector is the current technical representation used to describe that profile.
- A source document is not the same thing as a parsed text chunk.
- A semantic relationship is not the same thing as a graph edge row.
- An agent identity is not the same thing as prompt text.
- A process obligation is not the same thing as a checklist item.
- A memory state is not the same thing as chat history.
- A trust assessment is not the same thing as a numeric score.
- UI labels, reports, requirements, and design notes should use the product element name first. Technical names are acceptable when the implementation detail itself is being discussed.

### Profile Backfill Operation

- Text similarity profile backfill must be controllable by static configuration and visible operator controls.
- Model-backed profile generation must use a local non-HTTP transport for ingestion and retrieval memory operations. The current bridge is `runtime.embedding_adapter: local_command`, which calls `runtime.profile_command` over stdin/stdout JSON.
- The system must expose profile-adapter readiness before a backfill starts, including blocked HTTP-backed adapters and `local_command` without a configured command.
- The current default recommendation is 25 nodes per batch and 10 web batches per run.
- The system should expose the recovery behavior of profile jobs plainly: node writes are saved individually, while the job cursor is saved after a completed batch.
- If a profile job is interrupted mid-batch, the operator should requeue it. Missing-profile jobs skip profiles already written during replay; forced jobs may rebuild the interrupted batch.
- Later work should add dynamic batch adjustment based on observed local model throughput and error rate.

### LLM Transparency

- LLM calls are product-visible events, not hidden implementation details.
- HTTP is allowed for the human web interface and may be used for an optional final hosted answer-model call. It must not be used for ingestion, retrieval, memory-agent tool orchestration, Python memory tools, or repository text similarity profile generation.
- Each answer should expose:
  - memory-agent planner calls, if agentic mode is used;
  - final answer model call;
  - adapter/model identity;
  - prompt/trace details;
  - success or blocked state.
- JSON traces may remain available for debugging, but the default report must be readable like a good application log.
- Current web behavior provides immediate client-side running milestones and then renders the returned process trace as a step log. True server-pushed per-step streaming remains a later pipeline/API change.

### Governance, Identity, And Process

- Memory is an active part of cognition, not passive storage.
- Agent identity should influence retrieval scope, trusted corpus, exclusions, weighting preferences, and process obligations.
- The system should support shared, domain, and restricted identity layers.
- Process objects should become enforceable semantic objects over time, with execution tracking, exception proposals, and audit trails.
- Current process-run persistence is observational scaffolding, not full enforcement.

### Internet-Assisted Reasoning

- Internet content should be usable as temporary context.
- Internet-derived content must not automatically become repository memory.
- Promotion states should be:
  - temporary context;
  - candidate knowledge;
  - permanent knowledge after review/ingestion criteria.

## Current Implemented Design

### Runtime Shape

- Python package under `src/tirzah`.
- Local MongoDB persistence.
- FastAPI backend.
- Static HTML/CSS/JS web UI.
- CLI commands for ingestion, retrieval, graph inspection, governance lookup, sessions, and process runs.
- Local answer adapters, including stub and Ollama CLI.

### Ask Flow

1. The web UI submits query, session, optional focus node, adapter, model, and retrieval mode.
2. Python starts an `answer_query` process run when possible.
3. Low-intent conversational prompts without an explicit focus node are forced onto a no-context direct path, even if agentic mode was requested, so greetings cannot trigger arbitrary memory-agent search.
4. Direct mode either:
   - uses a provided focus node;
   - scopes active-document references to active documents;
   - searches the corpus for substantive queries;
   - bypasses retrieval for low-intent greetings;
   - falls back to no-context prompting when no memory context is appropriate.
5. Agentic mode calls a memory-agent planner that can request read-only retrieval tools.
6. Python executes allowed retrieval tools and records observations. Invalid tool calls return usage guidance instead of opaque errors.
7. When stopping, the memory-agent may propose selected node IDs and context organization. Python validates and budgets that proposal before final assembly.
8. Python assembles a bounded context document and final prompt.
9. The answer adapter generates the final response.
10. The exchange is saved, used nodes are scored, active documents are updated, and generated output is queued for reviewable ingestion.
11. The API returns:
   - answer;
   - structured process trace;
   - machine-facing `activity_report`;
   - plain-English `activity_log`.

### Web UI Layout

- `Ask` tab:
  - Prompt panel.
  - Response panel.
  - Activity Log panel with plain-English log and collapsed JSON.
  - Prompt / Trace panel with detailed process trace, visually relegated below the main Prompt / Response / Activity row.
  - Standard display mode uses a desktop workbench layout.
  - E-paper display mode keeps the Ask panels side by side with proportional columns, no dark trace panel, and reduced visual texture.
- `Browse` tab:
  - node search;
  - recent documents;
  - history filters.
- `Ingestion` tab:
  - upload/stage source files;
  - browse inbox;
  - process inbox;
  - review semantic-edge candidates;
  - inspect recent jobs.

### Retrieval Guardrail

The current retrieval guardrail treats exact low-intent conversational prompts such as `hello`, `hi`, `thanks`, and `ok` as no-context prompts when no focus node is supplied. This prevents arbitrary corpus hits from being presented as relevant memory. Agentic mode is also forced onto this no-context path for these prompts so the memory-agent fallback cannot run a pointless `search_nodes` call.

This is only the first guardrail. A better next step is an intent classifier or deterministic retrieval threshold that distinguishes:

- conversational prompt;
- operational command;
- repository question;
- active-document reference;
- internet-needed question;
- ingestion request.

## Open Design Risks

- The UI may still need a more disciplined information hierarchy after users exercise real workflows.
- E-paper mode is a first hardware-specific layout; it still needs direct testing on the Dasung screen for font size, browser zoom, and refresh behavior.
- Direct retrieval can still over-match generic substantive prompts because there is no strict relevance threshold yet.
- Agentic retrieval is read-only and scaffolded; it does not yet perform full semantic graph traversal or governed process enforcement.
- Ingestion is deterministic chunking, not the desired LLM-assisted semantic ingestion pipeline.
- Activity logs explain current behavior but are not yet rich ingestion logs.
- Repository refresh/rebuild is not yet a full reproducible product workflow.

## Near-Term Design Decisions

1. Treat the Ask tab as the primary human experimentation surface.
2. Keep ingestion and review in a separate operational workspace.
3. Support both Standard and E-paper display modes instead of treating side-by-side desktop layout as universally correct.
4. Prefer no-context answers over weak accidental retrieval for low-intent prompts.
5. Add explicit relevance thresholds before direct retrieval can select a context node for generic prompts.
6. Build ingestion logs as human-readable application logs first, with JSON only as a technical substrate.
7. Continue using deterministic tests for retrieval behavior before introducing heavier semantic ranking.
