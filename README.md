# Tirzah

**Tirzah gives a local model a memory of your documents that stays faithful
to the source.** Rather than loading whole documents into a prompt, it
ingests them into a provenance-aware graph in MongoDB and compiles just the
relevant, structured context for each question. Every answer can be traced
back to the exact source text it used.

It runs entirely on your own infrastructure (MongoDB for storage, Ollama for
inference) and is used from a CLI or the
[Mahlah](https://github.com/gellsmore-svg/Mahlah) chat UI.

## Is it for you?

**Good fit:** you want to ask questions over your own corpus with a local
model, see why each passage was included, and keep provenance as documents
are re-ingested.

**Not a fit:** a hosted service, or anything exposed to an untrusted network.
Tirzah is an early local-first V1; its API and MongoDB are unauthenticated
(see [Know before you rely on it](#know-before-you-rely-on-it)).

## Try it

The supported path is Docker Compose on Linux or WSL:

```bash
docker compose build
docker compose run --rm app tirzah init --docker
docker compose up                       # API and UI on http://127.0.0.1:8765/
```

Then, from the CLI:

```bash
tirzah db-ping                                         # check the database
tirzah ingest-folder /path/to/docs --label my_corpus   # build the graph
tirzah list-docs --limit 5
tirzah ask "What does the corpus say about X?" --retrieval-mode agentic
```

`ask` answers with a local Ollama model (`gemma3:1b` by default; `--model` to
change it), and `--web` adds web research to the corpus. `--adapter mock` gives an offline, deterministic answer, and
`--adapter claude_cli` (or `codex_cli`, `kiro_cli`, `google_cli`,
`grok_cli`) uses a headless coding CLI. Python installs and configuration:
[docs/install.md](docs/install.md). The CLI is `tirzah`; `mnemosyne` remains
as a compatibility alias.

## How it works

1. **Ingest into a graph.** Each document becomes a tree of nodes
   (`source_root` → `source_section` → `source_chunk`), with the source text
   kept verbatim and each node carrying its path, checksum, labels and
   provenance. Re-ingesting creates a new epoch and marks the old tree
   `superseded`; nothing is deleted.
2. **Retrieve with structure.** Search finds the relevant nodes, then
   compiles their context (the node itself, its ancestors, siblings and
   descendants) into a budgeted prompt.
3. **Choose how hard to look.** `direct` retrieves once. `agentic` lets a
   first model call pick read-only tools (`search_nodes`, `compile_context`,
   `list_documents`) before answering. `deep` runs a bounded plan → execute
   → triage → synthesise loop, including meaning-based search that finds
   passages sharing no keywords with the question.
4. **Plan and record.** Requests run under a revisable
   [Deborah](https://github.com/gellsmore-svg/Deborah) plan. Each exchange
   records a restart-state snapshot, so a session can be inspected and
   resumed.

For coding agents connected through Keturah MCP, `search_memory` discovers
candidate nodes; `read_memory(node_id, offset, max_chars)` then returns the stored
source text in bounded pages with document identity and provenance. Search snippets
are discovery hints, not evidence to rely on without reading the source.

## What else it does

| Feature | In short |
|---|---|
| **Human-defined Processes** | Versioned plain-text process templates (Governed, Fluid and Emergency presets) that the planner works within: gates pause for approval, deviations are flagged, overrides need a justification, and every instance is auditable |
| **Interpretive execution** | Plans executed step by step by the Deborah interpreter, revising mid-run, streamed live to Mahlah's process panel, with every model call recorded in [Galeed](https://github.com/gellsmore-svg/galeed) |
| **Milcah specialist** | Optional coherence pressure-tests and counter-framework research via [Milcah](https://github.com/gellsmore-svg/Milcah) |
| **Hoglah routing** | Answers and embeddings submitted through the [Hoglah](https://github.com/gellsmore-svg/hoglah) queue |
| **Real embeddings** | `pip install -e '.[profiles]'` for local embeddings (e.g. `bge-small-en-v1.5`), which switches on hybrid lexical+vector retrieval |
| **LLM-assisted ingestion** | Opt-in, review-gated chunking by a local model or coding CLI; trees stay out of retrieval until `tirzah promote-ingestion` |

Configuration and commands for each are in
[docs/features.md](docs/features.md).

## Know before you rely on it

- **Unauthenticated by design, loopback by default.** The API and MongoDB
  have no authentication. Docker Compose publishes the API on `127.0.0.1`
  only; put an authenticating proxy in front before exposing it anywhere
  else ([SECURITY.md](SECURITY.md)).
- **Defaults favour reproducibility over quality.** Embeddings default to a
  deterministic mock, so retrieval is lexical with local reranking until you
  configure a real embedding adapter. Ingestion defaults to deterministic
  heading and paragraph parsing.
- **Governance records are observational.** Agent identities, policies and
  trust profiles are inspectable but not enforced. Human-defined Processes
  are the part that does pause work for approval.
- The full list is in [docs/v1-known-limitations.md](docs/v1-known-limitations.md).

## Documentation

- [Project brief](docs/project-brief.md): what Tirzah is and why
- [Consolidated requirements and design](docs/consolidated-requirements-and-design.md)
- [Architecture decisions](docs/architecture-decisions.md) · [build roadmap](docs/build-roadmap.md) · [V1 readiness checklist](docs/v1-readiness-checklist.md)
- [Features in depth](docs/features.md) · [install and configuration](docs/install.md)
- The [2026-08-08 review](docs/review-2026-08-08.md) was actioned in 1.16.0.
- [okf/](okf/index.md): a concept map in Open Knowledge Format. The original
  v0.3/v0.1 design documents are kept at the repo root for provenance.

## Issues and contributing

Bugs and questions: [issues](https://github.com/gellsmore-svg/tirzah/issues).
Report security issues privately ([SECURITY.md](SECURITY.md)).
[CONTRIBUTING](CONTRIBUTING.md) · Apache 2.0, see [LICENSE](LICENSE).
