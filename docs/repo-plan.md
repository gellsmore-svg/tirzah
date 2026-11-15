# Repository Plan

Last updated: 2026-05-17

## Proposed Initial Tree

```text
Mnemosyne/
  README.md
  pyproject.toml
  config.example.yaml
  docs/
    project-brief.md
    source-documents.md
    requirements-index.md
    architecture-decisions.md
    build-roadmap.md
    product-direction.md
    repo-plan.md
  src/
    mnemosyne/
      __init__.py
      cli.py
      config.py
      models/
      adapters/
      db/
      queue/
      ingestion/
      retrieval/
      consolidation/
      sessions/
      web/
  tests/
  data/
    ingest/
    archive/
    dead_letter/
    staging/
```

## Python Package Boundaries

| Package | Responsibility |
|---|---|
| `config` | Load and validate `config.yaml`. |
| `cli` | Developer commands for ingestion, worker execution, status checks, and database inspection. |
| `models` | Pydantic/domain models for documents, trees, nodes, sessions, queue jobs, semantic clusters. |
| `adapters` | Model adapter interface and concrete Gemma/mock adapters. |
| `db` | Mongo client, collection access, indexes, repository helpers. |
| `queue` | Queue adapter interface and MongoDB-backed queue. |
| `ingestion` | Folder watcher, parser, staging, Gemma processing, transactional commit, archive/dead-letter handling. |
| `retrieval` | Query formation, traversal, context compilation. |
| `consolidation` | REM candidate discovery, clustering, semantic map updates. |
| `sessions` | Session records, active document registry, restart state, endorsement. |
| `web` | FastAPI app and static UI. |

## First Implementation Slice

Build Stage 0 and the smallest Stage 1 ingestion path:

1. Create package scaffold.
2. Add config model and example config.
3. Add Pydantic schemas for documents, trees, nodes, queue jobs, and ingestion result.
4. Add mock model adapter returning deterministic structured ingestion output.
5. Add markdown/plaintext parser.
6. Add Mongo queue adapter and repositories.
7. Add ingestion worker command.
8. Add tests around parser, queue state transitions, and commit/failure paths.
9. Add a CLI command that ingests one file into the real local MongoDB and prints inserted IDs.
10. Add checksum duplicate rejection and source archive copying.
11. Seed MongoDB label definitions with key, scope, and description fields.
12. Add inbox enqueue/process commands with processed and duplicate file movement.
13. Add bounded retry handling and queue inspection commands.
14. Add schema-versioned document/tree/node records and schema metadata backfill.
15. Add deterministic hierarchical mock ingestion and tree inspection.
16. Add first retrieval commands for document listing, document inspection, and node search.
17. Add document/date scoped retrieval and first node context expansion.
18. Add role-tagged context compilation with ancestor, sibling, and descendant expansion.
19. Add Markdown context rendering with character-budget include/skip metadata.
20. Add prompt-envelope construction with response reservation and token estimates.
21. Add ask/chat interaction commands, mock answer adapter, and exchange persistence.
22. Add optional Ollama CLI answer adapter for local model interaction.
23. Add FastAPI/static web UI over search, ask, and history.
24. Add web operator controls for queue status, process-inbox, and recent jobs.

## Risks To Manage Early

- Gemma structured JSON reliability.
- MongoDB transaction support in the local deployment.
- Whether local MongoDB vector search is available.
- Chunk granularity exploding node counts.
- Endorsement labels accidentally over-trusting content.
- Hidden complexity in "one document may produce many trees".
