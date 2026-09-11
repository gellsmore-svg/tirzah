# Development Plan

Date: 2026-05-30

Status: active implementation plan. This plan reconciles the current requirements around ingestion quality, real-world human testing, and human-readable transparency.

## Guiding Priorities

1. Ingestion quality comes first. Poor ingestion creates poor memory, poor retrieval, and misleading confidence.
2. Human usability must improve alongside ingestion, because the UI is the main test bench for real use.
3. Human-readable transparency is required for every major step. JSON may exist for machines, but it must not be the normal path for understanding what happened.

## Agreed Strategy

The next development phase should build an ingestion-quality spine while preserving the answer/retrieval transparency already implemented.

The system should move in this order:

1. Preserve the current working baseline and rename the product/repo/package from Mnemosyne to Tirzah in a dedicated slice.
2. Keep the Ask UI usable as a normal LLM wrapper client, with developer/debug controls behind a toggle.
3. Add project-domain and conversation-domain fields so memory has explicit working boundaries.
4. Add a prompt processing pipeline so prompt intake, context strategy, model handoff, and continuity persistence are deliberate local stages.
5. Add last prompt iteration records so thread continuity is inspectable.
6. Add process objects and one working answer/retrieval process before expanding process coverage.
6. Ingestion epoch and non-destructive rebuild foundation.
7. Chronological source-date extraction.
8. Human-readable ingestion activity logs.
9. UI support for ingestion runs, epochs, and log inspection.
10. Text similarity profile interface and semantic candidate substrate.
11. LLM-assisted ingestion adapter.
12. Reviewed semantic relationship promotion.
13. Ranking, trust, and temporal integration.

Internet-assisted reasoning, live server-pushed streaming, and process enforcement should not be pulled forward until ingestion epochs, semantic candidates, and reviewable provenance are stronger. Those features depend on trust and graph foundations that are not ready yet.

## Immediate Product-Shape Slices

### Slice 0A: Tirzah Rename

Goal:

- Rename the GitHub repository, product labels, package/module references, CLI command, documentation, and config examples from Mnemosyne to Tirzah.

Required behavior:

- Keep temporary compatibility for the old `mnemosyne` CLI command and Python import path.
- Document Tirzah as the preferred name.
- Avoid mixing the rename with ingestion/retrieval behavior changes.

### Slice 0B: Work-First Ask UI

Goal:

- Make the Ask workspace usable as a clean LLM wrapper client for real work.

Default work mode:

- session selector;
- prompt;
- Ask button;
- model selector;
- response;
- human-readable activity log.

Developer mode:

- Browse and Ingestion tabs;
- focus-node override;
- adapter selection;
- retrieval-mode override;
- raw prompt/trace panel;
- technical JSON reports;
- review/debug controls.

Human transparency remains visible in work mode. Raw diagnostics and operational controls are developer-mode features.

### Slice 0C: Domains And Continuity

Implementation status: initial domain foundation is implemented. Project and conversation domain registries exist under `src/tirzah/domains`, domain collections are indexed, saved exchanges now carry `project_domain_id` and `conversation_domain_id`, and the default conversation domain follows the session ID. Last prompt iteration records remain open.

Goal:

- Add explicit project-domain and conversation-domain concepts.
- Save the last prompt iteration record for continuity.

Required fields:

- project domain;
- conversation domain;
- retrieved chunks;
- skipped/rejected chunks;
- process used;
- final context package;
- unresolved follow-up items.

### Slice 0D: Prompt Processing Pipeline

Goal:

- Add a dedicated local prompt-processing boundary between the UI/CLI and the target answer model.

Why:

- Generic wrapper tools tend to pass a submitted prompt directly to a target model with limited structured handling.
- Tirzah needs a deliberate stage where the prompt is interpreted, domains are resolved, processes are selected or rejected, repository memory use is decided, context is gathered, and the final answer package is assembled before the answer model sees anything.

Required stages:

- prompt intake;
- intent classification;
- project/conversation domain resolution;
- process selection or rejection;
- repository-memory use decision;
- memory-agent/controller handoff where enabled;
- context package assembly;
- answer-model handoff preparation;
- readable activity-log events;
- prompt cycle persistence.

Implementation direction:

- Create `src/tirzah/prompt_pipeline`.
- Move orchestration currently spread through `sessions.interaction` toward small pipeline steps.
- Keep model adapters as execution endpoints, not owners of prompt strategy.
- Keep Python responsible for validation, local tool execution, budgeting, provenance, persistence, and error recovery.
- Let the memory-agent/controller make context strategy decisions through the local interface when agentic retrieval is active.

### Slice 0E: Process Objects

Goal:

- Add fixed process definitions that can be invoked by the LLM wrapper, user request, or runtime trigger.

Initial process:

- answer with context construction.

Required behavior:

- Python validates process availability and required steps.
- The LLM may request or operate process steps through local tools.
- Activity logs explain process use in plain language.

### Slice 0F: Mahalath Linked Domain

Goal:

- Treat Mahalath as a linked local corpus/domain before considering code integration.

Initial work:

- inspect Mahalath source/ontology structure;
- ingest representative Mahalath content under a separate project domain;
- build text similarity profiles;
- evaluate whether its ontology/process material should enrich Tirzah.

## Slice 1: Ingestion Epoch Foundation

Goal:

- Stamp documents, trees, nodes, and node provenance with an `ingestion_epoch`.
- Allow CLI ingestion and maintenance rebuild commands to accept an explicit epoch ID.
- Generate a conservative date-based default epoch when none is provided.

Why:

- Later non-destructive rebuilds need a way to distinguish old and new derived structures.
- Chronological corpus builds need run-level provenance.
- Human ingestion logs need to say which ingestion run produced which objects.

Non-goal:

- This slice does not make rebuild non-destructive by itself. It creates the metadata surface required for that change.

## Slice 2: Non-Destructive Rebuild

Implementation status: versioned rebuild insertion is implemented; epoch comparison and explicit garbage collection remain open.

Goal:

- Replace destructive tree/node deletion with versioned insertion.
- Mark previous tree/node sets as superseded rather than deleting them.
- Preserve endorsements, usage scores, active-document references, and reviewed semantic edges.

Required behavior:

- Rebuild should create a new active tree under a new epoch.
- Previous trees/nodes should remain queryable for audit unless explicitly garbage-collected.
- Retrieval should prefer active epoch records by default.

Implemented behavior:

- `rebuild-document` and `rebuild-by-label` no longer require destructive replacement for normal operation.
- Previous document trees and nodes are marked `superseded`.
- New trees and nodes are inserted as `active` under the selected ingestion epoch.
- Normal search and document-tree views exclude superseded nodes by default.

## Slice 3: Chronological Source-Date Extraction

Implementation status: source-date extraction and persistence are implemented for direct CLI ingestion, queued ingestion, and maintenance rebuilds. CLI folder ingestion now orders sources by selected origin date, then path.

Goal:

- Determine earliest credible source origin date for each document.

Priority order:

1. Explicit date inside document content.
2. Date embedded in filename.
3. Original file creation date, when preserved by the source acquisition path.
4. Original file modification date, when preserved by the source acquisition path.

Output:

- Store date candidates and the chosen date with rationale.
- Use chronology for corpus build ordering, especially AMS / RS / RS5 collections.

Implemented behavior:

- The ingestion date analyzer records explicit content dates, filename dates, filesystem creation/change dates, and filesystem modification dates as structured candidates.
- Filesystem candidates are treated as weak fallback evidence because web uploads and copied files can reflect import time rather than authorship time.
- The selected origin date and rationale are stored in document `source` metadata.
- Retrieval document serialization exposes `origin_date` and `origin_date_source`.
- `ingest-folder` builds a chronological source plan and reports the first 20 planned sources in `source_order`.
- `ingest-folder` and the Ingestion tab preserve unreadable supported files as explicit error rows instead of failing the entire folder/listing.

## Slice 4: Human-Readable Ingestion Logs

Implementation status: initial direct and queued ingestion activity logs are implemented. Logs are attached alongside structured ingestion activity reports, web inbox processing displays the plain operator log by default, completed queue jobs retain their readable log, and the inbox browser shows origin-date ordering context. LLM-assisted ingestion logs remain open until the ingestion adapter exists.

Goal:

- Give every ingestion operation a readable activity report comparable to answer activity logs.

The log should explain:

- source detected;
- checksum and duplicate status;
- date candidates and chosen date;
- adapter used;
- chunks/nodes created;
- relationship hints detected or proposed;
- repository writes;
- failures, retries, and review requirements.

Implemented behavior:

- Direct CLI ingestion returns `activity_report` and `activity_log` for successful and duplicate-rejected files.
- Queue worker ingestion returns `activity_report` and `activity_log` for successful, rejected, retrying, failed, and missing-source jobs.
- Completed queue jobs persist their ingestion activity log inside the stored job result for later review.
- Web inbox processing aggregates per-job logs into an `Inbox Processing Activity Log` and shows that human-readable log in the Ingestion tab instead of defaulting to raw JSON.

## Slice 5: Ingestion UI Test Bench

Goal:

- Make the Ingestion tab usable for real-world corpus testing.

Required views:

- current epoch;
- recent ingestion runs;
- readable ingestion log;
- source file status;
- created document/tree/node counts;
- candidate relationships awaiting review;
- comparison between epochs when available.

Implemented behavior:

- The inbox browser shows selected origin date, origin-date source, and date-candidate count for staged files, ordered by origin date and then path.
- Recent jobs expose the persisted readable ingestion log for completed queue jobs.
- `/api/ingestion/status` and the Ingestion tab now summarize recent ingestion epochs and ingestion process runs, including the current/latest epoch by update time and dated-document coverage per epoch.

## Slice 6: Semantic Substrate

Goal:

- Add a text similarity profile interface behind a stub-capable adapter boundary.
- Store the current embedding vector representation without immediately changing ranking behavior.
- Generate semantic candidates into the existing review queue.

Constraint:

- Semantic writes remain candidate/review based. The memory-agent should not gain autonomous write authority.

Status: Phase A implemented (stub-capable text similarity profile substrate); Phase B partially implemented (profile-based semantic candidate diagnostics and review-queue enqueueing).

Phase A implemented behavior:

- A dependency-free profile adapter boundary lives in `tirzah.adapters.embedding`, with a deterministic `MockEmbeddingAdapter` and an `embedding_adapter(config)` factory selected by `runtime.embedding_adapter` / `runtime.embedding_dimensions`.
- HTTP-backed `ollama_http` and `ollama_powershell` profile adapters were built as temporary diagnostics, but they are not compliant for ingestion or retrieval memory operations. The default remains `mock`.
- A non-HTTP `local_command` profile adapter is available behind the same boundary. It calls a configured local executable over stdin/stdout JSON, so a local model runner can be plugged in without routing ingestion or retrieval memory operations through HTTP.
- Runtime and ingestion status now expose profile-adapter readiness, including HTTP-policy blocking and missing `runtime.profile_command` for `local_command`.
- The stub adapter derives a bounded, unit-norm embedding vector representation from a SHA-256 expansion of the node text, so the same text always produces the same profile representation without any external model or network call.
- During direct CLI ingestion and queued worker ingestion (both via `commit_ingestion`), and during maintenance rebuilds (`rebuild_document`), each node is annotated before commit with an `embedding` field holding adapter/model name, dimensions, embedding vector representation, and the source text hash.
- Ingestion activity reports/logs now surface the profiled node count, profile model, and dimensions.
- This phase intentionally does not change search ranking, answer behavior, or memory-agent write authority.
- Stored text similarity profiles can now be read by operator CLI/API/web controls to propose pending semantic-edge candidates, distinguished from label-overlap candidates by `candidate_source: embedding_similarity` and similarity/model/dimension metadata.
- `contradicts` candidates (`candidate_source: contradiction_signals`) reuse that review queue. Generation requires disagreement evidence (a negation/refutation term on one side only, over shared claim wording) with embedding similarity in [0.6, 0.97); shared semantic labels and origin-date delta only break ties. Candidate rows stamp both nodes' dates and provenance.
- `tirzah enqueue-profile-semantic-batch` (compatibility alias: `enqueue-vector-semantic-batch`), `/api/review/enqueue-vector-semantic-batch`, and the matching Ingestion-tab controls can queue profile-derived candidates across a bounded focus-node scope. Repeated `--exclude-node-key` filters let operators skip templated focus rows such as concept-visit `section-1` headers, and dry runs preview candidates plus duplicate skips without writing pending rows. This exists to make larger profile baselines reviewable without changing the review rule: candidates still require explicit acceptance before they become semantic relationships.
- Profile backfill recommendations are operator-configurable through `runtime.profile_backfill_recommended_batch_limit` and `runtime.profile_backfill_web_max_batches`, defaulting to 25 nodes and 10 web batches per run.
- Profile backfill jobs expose their recovery behavior: node writes are saved individually, but the job cursor is saved after a completed batch. If interrupted mid-batch, requeue the job; missing-profile jobs skip profiles already written during replay, while forced jobs may rebuild the interrupted batch.

Phase A remaining work:

- Choose and configure the actual local model-backed profile command, then validate candidate quality against a refreshed corpus.
- Only then consider any retrieval/ranking use of text similarity profiles, kept candidate/review based.

## Slice 7: LLM-Assisted Ingestion

Goal:

- Introduce a quality-first local LLM ingestion adapter.

The adapter should produce:

- semantic chunks;
- concept/entity observations;
- relationship hints;
- sensitivity/trust notes;
- source-date observations;
- structured ingestion log entries.

Writes should remain reviewable until trust, identity, and governance controls are stronger.

## Slice 8: Ranking And Trust Integration

Goal:

- Move away from increasingly hardcoded lexical ranking.
- Consolidate ranking into a requirement-cited scoring module.
- Integrate reviewed semantic relationships, trust, temporal relevance, usage, and active-document context.

Constraint:

- Each ranking factor must be explainable in the activity report.

## Documentation Maintenance

Before or during this phase:

- Mark stale historical reports as superseded where they no longer describe current behavior.
- Close the open documentation question around the `rejected` endorsement state.
- Keep `docs/consolidated-requirements-and-design.md` as the canonical product/design entry point.
- Keep this file as the active implementation sequence.
