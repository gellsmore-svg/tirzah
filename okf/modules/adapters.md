---
type: Module
title: Adapters
description: The pluggable boundaries to models and inference — answer adapters (mock/ollama_cli/ollama_http/hoglah), embedding adapters (mock/local_command/ollama_http/ollama_powershell/hoglah), the ingestion adapter, and the Hoglah submit/await runtime.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/adapters/answer.py
tags: [tirzah, adapters, ollama, hoglah, embeddings]
timestamp: 2026-06-19T00:00:00Z
---

# Adapters (`adapters/`)

Every model interaction goes through a pluggable adapter, selected in
[config](config.md); a deterministic **mock** default keeps tests/first-runs
offline.

- **`answer.py`** — answer generation: `mock`, `ollama_cli`, `ollama_http`,
  external coding CLIs (**`kiro_cli`**, **`claude_cli`**, **`codex_cli`**,
  **`google_cli`** / `gemini_cli`, **`grok_cli`**), and
  **`hoglah`** (routes generation through a [Hoglah](https://github.com/gellsmore-svg/hoglah)
  queue daemon, with poll/callback delivery).
- **`embedding.py`** — embeddings: `mock`, `local_command` (e.g. bge-small via
  fastembed), `ollama_http`, `ollama_powershell`, and `hoglah`. Produces the node/
  query vectors used by [hybrid & semantic retrieval](../concepts/hybrid-and-semantic.md).
- **`hoglah_runtime.py`** — the shared Hoglah submit/await machinery: a
  `HoglahJobRunner` submits generate/embed jobs and awaits results, over the
  shared SQLite store (poll/callback) or a messaging broker
  (`hoglah_transport: kafka|rabbitmq|redis`, via Hoglah's `MessagingSubmitter`).
- **`ingestion.py`, `mock.py`, `llm_ingestion.py`** — the ingestion adapter
  boundary, the deterministic mock parser (default), and review-gated LLM
  chunking (`ingestion_adapter: llm`) backed by a local Ollama model or an
  external coding CLI (`kiro_cli`, `claude_cli`, `codex_cli`, `google_cli`,
  `grok_cli`).

Routing inference through Hoglah serialises every call through one durable,
restart-safe queue — see Hoglah's [decoupled topology](https://github.com/gellsmore-svg/hoglah/blob/main/okf/concepts/decoupled-topology.md).
