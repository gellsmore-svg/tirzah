---
type: Module
title: Config
description: The runtime configuration — RuntimeConfig (adapters, retrieval mode, hybrid toggle, Ollama/Hoglah settings), RetrievalConfig (budgets, deep-retrieval bounds), and MongoConfig — loaded from config.yaml.
resource: https://github.com/gellsmore-svg/tirzah/blob/main/src/tirzah/config.py
tags: [tirzah, config, runtime]
timestamp: 2026-06-19T00:00:00Z
---

# Config (`config.py`)

Loaded by `load_config(path="config.yaml")`. Three groups:

- **`RuntimeConfig`** — adapter selection (`answer_adapter`, `embedding_adapter`,
  `ingestion_adapter`, `ingestion_model_adapter`, `kiro_*`, `model_adapter`), `retrieval_mode`
- **`RetrievalConfig`** — prompt/context budgets, optional `model_profiles` and
  `tokenizer` (`approx` or `tiktoken`)
  (`direct|agentic|deep`), `hybrid_search_enabled` (**default True**), embedding
  model/dimensions + `profile_command`, Ollama settings (`ollama_base_url`,
  `ollama_model`, …), and the Hoglah settings (`hoglah_transport: store|kafka|
  rabbitmq|redis` + db/output/callback/broker connection fields).
- **`RetrievalConfig`** — context/prompt token budgets, the memory-agent max
  iterations, and the **deep-retrieval bounds** (`deep_max_iterations`,
  `deep_max_candidates`, `deep_shortlist_size`, `deep_page_size`).
- **`MongoConfig`** — `uri` and `database` (default `mnemosyne_dev`).

Config selects the [adapters](adapters.md), tunes the
[retrieval modes](../concepts/retrieval-modes.md) and
[context compilation](../concepts/context-compilation.md) budgets, and points at
the [storage](storage.md). The CLI takes `--config` (global, before the
subcommand).
