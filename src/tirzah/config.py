from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("tirzah.config")


class _StrictModel(BaseModel):
    """Reject unknown keys so typos cannot silently disable controls (review F2/H3)."""

    model_config = ConfigDict(extra="forbid")


class MongoConfig(_StrictModel):
    uri: str = "mongodb://localhost:27017"
    database: str = "tirzah_dev"


class PathConfig(_StrictModel):
    ingest: Path = Path("data/ingest")
    archive: Path = Path("data/archive")
    dead_letter: Path = Path("data/dead_letter")
    staging: Path = Path("data/staging")


class RuntimeConfig(_StrictModel):
    model_adapter: str = "mock"
    # Prefer HTTP: the CLI binary is often missing even when Ollama is up (F6).
    answer_adapter: str = "ollama_http"
    ingestion_adapter: str = "mock"
    # Model used by `ingestion_adapter: llm`. Empty = follow answer_adapter
    # when that is a generator (ollama_*/hoglah/mock or an external CLI),
    # otherwise ollama_http.
    ingestion_model_adapter: str = ""
    ingestion_model: str | None = None
    ingestion_fallback_to_mock: bool = True
    ingestion_max_source_chars: int = Field(default=24000, ge=500, le=200_000)
    embedding_adapter: str = "mock"
    allow_http_ingestion_adapters: bool = False
    embedding_model: str = "nomic-embed-text:latest"
    embedding_dimensions: int = 16
    profile_command: list[str] | str | None = Field(default_factory=list)
    profile_command_mode: str = "single"
    profile_backfill_recommended_batch_limit: int = 25
    profile_backfill_web_max_batches: int = 10
    memory_agent_adapter: str | None = None
    retrieval_mode: str = "direct"
    # Recursive Cairn planning wrapper immediately below the front end.
    recursive_planning_enabled: bool = True
    planning_adapter: str | None = None
    planning_model: str | None = None
    planning_max_revisions: int = Field(default=3, ge=1, le=20)
    planning_max_steps: int = Field(default=12, ge=1, le=30)
    # Local-model confirmation of contradiction candidates before they are
    # queued for review. The lexical disagreement rule is only a pre-filter
    # (about 0.25% precise on a real corpus, measured 2026-09-11). Fails
    # closed: if the model is unavailable or unparseable, nothing is queued.
    contradiction_confirmation_enabled: bool = True
    # Empty = follow answer_adapter and that adapter's configured model.
    contradiction_confirmation_adapter: str | None = None
    contradiction_confirmation_model: str | None = None
    # Walk Cairn plan steps in depends_on order (SPEC §4.6) instead of only
    # wrapping a monolithic ask pipeline.
    plan_interpretive_execution_enabled: bool = False
    # When interpretive mode is on (or framed auto-detect hits), run crystallised
    # substrate/critique plans through Deborah's thin slice instead of Tirzah's
    # rich executor. See tirzah.planning.deborah_bridge.
    plan_framed_execution_enabled: bool = True
    # Validate every plan revision against deborah.validate_plan before execute.
    plan_require_deborah_conformance: bool = False  # soft by default: record errors
    plan_deborah_validate_profile: str = "full"
    # Revise the active plan after each completed step when interpretive mode runs.
    plan_mid_revision_enabled: bool = True
    # Opt-in transient web evidence. Search uses a SearxNG JSON endpoint; results
    # are never promoted into durable graph memory automatically.
    web_research_enabled: bool = False
    web_search_base_url: str = "http://localhost:8080"
    web_timeout_seconds: float = 12.0
    web_max_results: int = 5
    web_max_pages: int = 2
    web_max_content_bytes: int = 500_000
    web_max_content_chars: int = 8_000
    web_allow_private_search_endpoint: bool = False
    # API auth token (header X-Tirzah-Api-Token or Authorization: Bearer …).
    # Empty = no token required. Prefer setting this when binding beyond loopback.
    web_api_token: str = ""
    # Default True: refuse non-loopback API clients (review F1/H1). Docker
    # compose binds the *host* to 127.0.0.1; the container may listen on 0.0.0.0
    # internally while this still blocks off-machine clients via the host map.
    web_localhost_only: bool = True
    web_max_upload_bytes: int = Field(default=1_000_000, ge=1)
    # Blend lexical + query-vector similarity in node search (ADR-020). On by
    # default as of the real-corpus validation; only takes effect with a real
    # (non-mock) embedding adapter and degrades safely to lexical otherwise, so
    # it is harmless under the default mock adapter.
    hybrid_search_enabled: bool = True
    # Optional Atlas/MongoDB vector search index on nodes.embedding.vector.
    # Empty = local cosine scan fallback.
    vector_search_index: str = ""
    hybrid_vector_scan_limit: int = Field(default=500, ge=20, le=10_000)
    near_match_min_score: float = Field(default=0.78, ge=0.5, le=1.0)
    near_match_max_candidates: int = Field(default=8, ge=1, le=32)
    near_match_per_term: int = Field(default=3, ge=1, le=8)
    # Interim stand-in for the REQ-SEM-04 semantic map (not built yet):
    # term -> alternatives for query expansion. None keeps the built-in table.
    query_synonyms: dict[str, list[str]] | None = None
    weak_match_fallback_score: int = Field(default=5, ge=0, le=100)
    # Opt-in: use trust/temporal diagnostics as a bounded secondary ranking signal.
    trust_ranking_enabled: bool = False
    trust_weighting_profile: str | None = None
    trust_ranking_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    trust_ranking_max_boost: int = Field(default=20, ge=0, le=100)
    trust_ranking_hybrid_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    ollama_model: str = "gemma3:1b"
    memory_agent_model: str | None = None
    ollama_format: str | None = None
    memory_agent_ollama_format: str | None = "json"
    ollama_think: bool | str | None = False
    ollama_hide_thinking: bool = True
    ollama_base_url: str = "http://localhost:11434"
    # Resolved from PATH by default (portable); the HTTP path (ollama_base_url) is
    # preferred. Override via config.yaml or the OLLAMA_EXECUTABLE env var for a
    # non-PATH install (e.g. a WSL-mounted ollama.exe).
    ollama_executable: Path = Path("ollama")
    ollama_timeout_seconds: int = 180
    # Optional Kiro CLI answer/ingestion backend (`kiro-cli chat --no-interactive`).
    # Cloud-backed; keep answer_adapter/ingestion_model_adapter on ollama_* for local-only.
    kiro_executable: Path = Path("kiro-cli")
    kiro_timeout_seconds: int = Field(default=180, ge=5, le=3600)
    kiro_model: str | None = None
    kiro_agent: str | None = None
    kiro_effort: str | None = None
    kiro_trust_tools: str = ""
    # Optional Claude Code CLI (`claude -p`). Cloud-backed.
    claude_executable: Path = Path("claude")
    claude_timeout_seconds: int = Field(default=180, ge=5, le=3600)
    claude_model: str | None = None
    claude_max_turns: int = Field(default=1, ge=1, le=20)
    claude_bare: bool = True
    # Optional Codex CLI (`codex exec`). Cloud-backed.
    codex_executable: Path = Path("codex")
    codex_timeout_seconds: int = Field(default=180, ge=5, le=3600)
    codex_model: str | None = None
    codex_sandbox: str = "read-only"
    codex_ephemeral: bool = True
    # Prefix `sudo -n -E --` so Linux sandbox helpers can run elevated.
    # Requires passwordless sudo (`sudo -n`); leave false when a TTY password
    # prompt would hang headless runs.
    codex_sudo: bool = False
    # Optional Google Gemini CLI (`gemini -p`). Cloud-backed. Alias: gemini_cli.
    google_executable: Path = Path("gemini")
    google_timeout_seconds: int = Field(default=180, ge=5, le=3600)
    google_model: str | None = None
    # Optional Grok Build CLI (`grok --prompt-file`). Cloud-backed.
    grok_executable: Path = Path("grok")
    grok_timeout_seconds: int = Field(default=180, ge=5, le=3600)
    grok_model: str | None = None
    grok_max_turns: int = Field(default=3, ge=1, le=20)
    grok_no_auto_update: bool = True
    grok_sandbox: str = ""
    grok_disallowed_tools: str = (
        "run_terminal_cmd,search_replace,web_search,web_fetch,read_file,list_dir,grep"
    )
    # Context window for the HTTP adapter. Must be large enough to hold the
    # conversation history + retrieved context, or Ollama silently truncates the
    # start of the prompt (dropping history). 0 = leave Ollama's default.
    ollama_num_ctx: int = 8192
    # Semantic precision via Mahalath (the Tirzah->Mahalath seam, off by default).
    # When enabled, retrieval resolves key terms to MPL labels/senses from Mahalath's
    # ontology and conditions the answer on them. Fail-soft: an absent/unreachable
    # Mahalath simply yields no labels. See tirzah/semantic.py.
    mahalath_enabled: bool = False
    mahalath_mongo_uri: str = "mongodb://localhost:27017"
    mahalath_mongo_db: str = "mahalath_dev"
    mahalath_language: str = "en"
    # Milcah specialist seam (coherence/research). Off + degrades to no-op when Milcah
    # is absent, mirroring the Mahalath resolver.
    milcah_enabled: bool = False
    milcah_model: str = ""
    # Strict by default: drop fuzzy (partial/text) matches and keep only confident
    # label/exact/alias hits. A loose match attaching a wrong sense is worse than no
    # sense for precision-grade work; flip to false only to accept approximate hints.
    mahalath_strict: bool = True
    hoglah_db_path: Path = Path("data/hoglah/jobs.sqlite3")
    hoglah_ollama_host: str = "http://localhost:11434"
    # Decoupled topology: Tirzah is a pure submitter into the shared queue and a
    # SEPARATE `hoglah run --real` daemon executes jobs. hoglah_use_real is kept
    # for back-compat but is no longer used by the submitter (the daemon owns
    # real execution and its --ollama-host).
    hoglah_use_real: bool = True
    hoglah_wait_timeout_seconds: int | None = None
    # Where the daemon writes terminal results (must match the daemon's
    # HOGLAH_OUTPUT_DIR); Tirzah polls here.
    hoglah_output_dir: Path = Path("data/hoglah/outbox")
    # Result delivery: "poll" the output folder, or "callback" (Tirzah runs a
    # tiny HTTP receiver and hands Hoglah its own URL per job; falls back to the
    # output folder if a push is missed).
    hoglah_delivery: str = "poll"
    hoglah_callback_host: str = "127.0.0.1"
    hoglah_callback_port: int = 0
    # Submission transport: "store" (default — write to the shared SQLite queue and
    # await by poll/callback) or a messaging broker ("kafka" | "rabbitmq" | "redis"),
    # which publishes a job-request message and awaits the result over the same
    # broker. The matching `hoglah {kafka,rabbitmq,redis}-bridge` worker must be
    # running on these topics/queues/streams.
    hoglah_transport: str = "store"
    hoglah_kafka_bootstrap_servers: str = "localhost:9092"
    hoglah_kafka_input_topic: str = "hoglah-jobs"
    hoglah_kafka_results_topic: str = "hoglah-results"
    hoglah_rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    hoglah_rabbitmq_input_queue: str = "hoglah-jobs"
    hoglah_redis_url: str = "redis://localhost:6379/0"
    hoglah_redis_input_stream: str = "hoglah-jobs"
    hoglah_redis_results_stream: str = "hoglah-results"


class QueueConfig(_StrictModel):
    max_attempts: int = Field(default=3, ge=1)


class ModelBudgetProfile(_StrictModel):
    prompt_token_budget: int | None = Field(default=None, ge=64)
    reserved_response_tokens: int | None = Field(default=None, ge=16)
    context_char_budget: int | None = Field(default=None, ge=256)
    # Blank would silently inherit the global tokenizer; omit the key instead.
    tokenizer: str | None = Field(default=None, min_length=1)
    tokenizer_encoding: str | None = None
    chars_per_token: float | None = Field(default=None, gt=0)


class RetrievalConfig(_StrictModel):
    context_char_budget: int = Field(default=4000, ge=256)
    prompt_token_budget: int = Field(default=2000, ge=64)
    reserved_response_tokens: int = Field(default=500, ge=16)
    tokenizer: str = "approx"
    tokenizer_encoding: str | None = None
    chars_per_token: float = Field(default=4.0, gt=0)
    model_profiles: dict[str, ModelBudgetProfile] = Field(default_factory=dict)
    memory_agent_max_iterations: int = Field(default=4, ge=1, le=50)
    # Conversational memory: how many prior turns of the session to thread into
    # the prompt, and how much of each answer to keep.
    conversation_history_turns: int = Field(default=6, ge=0, le=100)
    conversation_history_answer_chars: int = Field(default=600, ge=0)
    # Phase 2: also surface semantically-relevant EARLIER turns (beyond the recent
    # window) by embedding similarity. Off by default — it embeds each turn, which
    # adds latency; enable for long conversations with a real embedder.
    conversation_semantic_recall: bool = False
    conversation_semantic_recall_k: int = Field(default=3, ge=1, le=50)
    # Phase 3: decompose turns into typed semantic chunks (topic/intent/domain/...).
    # Off by default — it runs an extra LLM call per turn (async + durable when on).
    conversation_chunking: bool = False
    # Deep retrieval mode (ADR-020) — bounds for the agent loop + Python pre-rank.
    deep_max_iterations: int = Field(default=4, ge=1, le=50)
    deep_max_candidates: int = Field(default=50, ge=1, le=10_000)
    deep_shortlist_size: int = Field(default=12, ge=1, le=500)
    deep_page_size: int = Field(default=5, ge=1, le=100)
    # Phase 4: recursive Context Sufficiency Score driving the deep loop.
    deep_sufficiency_scoring: bool = True
    deep_sufficiency_stop: float = 9.0  # stop when sufficiency reaches this
    deep_sufficiency_plateau_floor: float = 8.0  # stop at/above this if plateaued
    deep_plateau_passes: int = Field(default=3, ge=1, le=20)
    deep_plateau_epsilon: float = 0.2


class AppConfig(_StrictModel):
    mongo: MongoConfig = Field(default_factory=MongoConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


# Env-var → (section, key) overrides, applied on top of the YAML (and even with
# no config file). Keeps the shared OLLAMA_BASE_URL / MONGO settings in one place
# so the Noa runtime can configure every sibling from a single .env.
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "TIRZAH_MONGO_URI": ("mongo", "uri"),
    "TIRZAH_MONGO_DB": ("mongo", "database"),
    "OLLAMA_BASE_URL": ("runtime", "ollama_base_url"),
    "OLLAMA_EXECUTABLE": ("runtime", "ollama_executable"),
    "KIRO_EXECUTABLE": ("runtime", "kiro_executable"),
    "CLAUDE_EXECUTABLE": ("runtime", "claude_executable"),
    "CODEX_EXECUTABLE": ("runtime", "codex_executable"),
    "GEMINI_EXECUTABLE": ("runtime", "google_executable"),
    "GOOGLE_CLI_EXECUTABLE": ("runtime", "google_executable"),
    "GROK_EXECUTABLE": ("runtime", "grok_executable"),
    "TIRZAH_INGESTION_ADAPTER": ("runtime", "ingestion_adapter"),
    "TIRZAH_INGESTION_MODEL_ADAPTER": ("runtime", "ingestion_model_adapter"),
    "TIRZAH_WEB_RESEARCH_ENABLED": ("runtime", "web_research_enabled"),
    "TIRZAH_RECURSIVE_PLANNING_ENABLED": ("runtime", "recursive_planning_enabled"),
    "TIRZAH_PLAN_INTERPRETIVE_EXECUTION_ENABLED": ("runtime", "plan_interpretive_execution_enabled"),
    "TIRZAH_PLAN_FRAMED_EXECUTION_ENABLED": ("runtime", "plan_framed_execution_enabled"),
    "TIRZAH_PLAN_REQUIRE_DEBORAH_CONFORMANCE": ("runtime", "plan_require_deborah_conformance"),
    "TIRZAH_PLAN_DEBORAH_VALIDATE_PROFILE": ("runtime", "plan_deborah_validate_profile"),
    "TIRZAH_PLAN_MID_REVISION_ENABLED": ("runtime", "plan_mid_revision_enabled"),
    "TIRZAH_WEB_SEARCH_BASE_URL": ("runtime", "web_search_base_url"),
    "TIRZAH_WEB_API_TOKEN": ("runtime", "web_api_token"),
    "TIRZAH_WEB_LOCALHOST_ONLY": ("runtime", "web_localhost_only"),
    "TIRZAH_WEB_MAX_UPLOAD_BYTES": ("runtime", "web_max_upload_bytes"),
    # The Mahalath seam — so `.env` alone enables/points it (no config file needed;
    # a missing config file can no longer silently disable semantic precision).
    "MAHALATH_ENABLED": ("runtime", "mahalath_enabled"),
    "MAHALATH_MONGO_URI": ("runtime", "mahalath_mongo_uri"),
    "MAHALATH_MONGO_DB": ("runtime", "mahalath_mongo_db"),
    "MAHALATH_STRICT": ("runtime", "mahalath_strict"),
    "MILCAH_ENABLED": ("runtime", "milcah_enabled"),
    "MILCAH_MODEL": ("runtime", "milcah_model"),
}


def _apply_env_overrides(data: dict) -> dict:
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            data.setdefault(section, {})[key] = value
    return data


def load_config(path: Path | str = "config.yaml") -> AppConfig:
    # An explicit TIRZAH_CONFIG env wins when the caller didn't pass a non-default
    # path, so the config location need not depend on the current directory.
    if str(path) == "config.yaml" and os.environ.get("TIRZAH_CONFIG"):
        path = os.environ["TIRZAH_CONFIG"]
    config_path = Path(path)
    if not config_path.exists():
        example_path = Path("config.example.yaml")
        config_path = example_path if example_path.exists() else config_path

    data = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        return AppConfig.model_validate(_apply_env_overrides(data))
    except Exception as exc:
        # Surface unknown keys / validation failures clearly (review F2).
        logger.error("Invalid Tirzah config at %s: %s", config_path, exc)
        raise
