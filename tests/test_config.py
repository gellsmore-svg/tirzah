from pathlib import Path

from tirzah.config import load_config


def test_load_config_reads_queue_max_attempts(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
mongo:
  database: custom
queue:
  max_attempts: 5
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.mongo.database == "custom"
    assert config.queue.max_attempts == 5


def test_load_config_reads_retrieval_budget(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
retrieval:
  context_char_budget: 1234
  prompt_token_budget: 200
  reserved_response_tokens: 50
  memory_agent_max_iterations: 7
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.retrieval.context_char_budget == 1234
    assert config.retrieval.prompt_token_budget == 200
    assert config.retrieval.reserved_response_tokens == 50
    assert config.retrieval.memory_agent_max_iterations == 7


def test_load_config_reads_separate_memory_agent_model(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
runtime:
  answer_adapter: ollama_http
  memory_agent_adapter: ollama_http
  ollama_model: final-model
  memory_agent_model: memory-model
  memory_agent_ollama_format: json
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.runtime.answer_adapter == "ollama_http"
    assert config.runtime.memory_agent_adapter == "ollama_http"
    assert config.runtime.ollama_model == "final-model"
    assert config.runtime.memory_agent_model == "memory-model"
    assert config.runtime.memory_agent_ollama_format == "json"


def test_load_config_reads_local_profile_command(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
runtime:
  embedding_adapter: local_command
  embedding_model: local-profile
  profile_command:
    - python
    - tools/profile.py
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.runtime.embedding_adapter == "local_command"
    assert config.runtime.embedding_model == "local-profile"
    assert config.runtime.profile_command == ["python", "tools/profile.py"]


def test_load_config_reads_ingestion_adapter(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
runtime:
  ingestion_adapter: mock
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.runtime.ingestion_adapter == "mock"


def test_load_config_reads_llm_ingestion_and_kiro_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
runtime:
  ingestion_adapter: llm
  ingestion_model_adapter: kiro_cli
  ingestion_model: claude-sonnet
  kiro_executable: /usr/local/bin/kiro-cli
  kiro_timeout_seconds: 90
  kiro_effort: low
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.runtime.ingestion_adapter == "llm"
    assert config.runtime.ingestion_model_adapter == "kiro_cli"
    assert config.runtime.ingestion_model == "claude-sonnet"
    assert str(config.runtime.kiro_executable) == "/usr/local/bin/kiro-cli"
    assert config.runtime.kiro_timeout_seconds == 90
    assert config.runtime.kiro_effort == "low"


def test_load_config_reads_claude_codex_and_google_cli_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
runtime:
  answer_adapter: claude_cli
  ingestion_model_adapter: google_cli
  claude_executable: /usr/bin/claude
  claude_max_turns: 2
  claude_bare: false
  codex_sandbox: workspace-write
  codex_ephemeral: false
  google_executable: /usr/bin/gemini
  google_model: gemini-2.5-flash
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.runtime.answer_adapter == "claude_cli"
    assert str(config.runtime.claude_executable) == "/usr/bin/claude"
    assert config.runtime.claude_max_turns == 2
    assert config.runtime.claude_bare is False
    assert config.runtime.codex_sandbox == "workspace-write"
    assert config.runtime.codex_ephemeral is False
    assert str(config.runtime.google_executable) == "/usr/bin/gemini"
    assert config.runtime.google_model == "gemini-2.5-flash"


def test_load_config_reads_grok_cli_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
runtime:
  answer_adapter: grok_cli
  grok_executable: /home/cello/.local/bin/grok
  grok_model: grok-4.6
  grok_max_turns: 2
  grok_no_auto_update: false
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.runtime.answer_adapter == "grok_cli"
    assert str(config.runtime.grok_executable) == "/home/cello/.local/bin/grok"
    assert config.runtime.grok_model == "grok-4.6"
    assert config.runtime.grok_max_turns == 2
    assert config.runtime.grok_no_auto_update is False


def test_load_config_defaults_ingestion_adapter_to_mock(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("runtime: {}\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.runtime.ingestion_adapter == "mock"


def test_default_mongo_database_is_tirzah_dev(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("runtime: {}\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.mongo.database == "tirzah_dev"


def test_load_config_accepts_unquoted_ollama_think_false(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
runtime:
  ollama_think: false
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.runtime.ollama_think is False


def test_recursive_planning_defaults_are_bounded():
    config = load_config("missing-recursive-planning-config.yaml")
    assert config.runtime.recursive_planning_enabled is True
    assert config.runtime.planning_max_revisions == 3
    assert config.runtime.planning_max_steps == 12
    # Deborah framed handoff (1.15): soft seal + framed slice on by default.
    assert config.runtime.plan_framed_execution_enabled is True
    assert config.runtime.plan_require_deborah_conformance is False
    assert config.runtime.plan_deborah_validate_profile == "full"
    # Security defaults (review 2026-08-08).
    assert config.runtime.web_localhost_only is True
    assert config.runtime.answer_adapter == "ollama_http"


def test_unknown_runtime_key_is_rejected():
    from pydantic import ValidationError
    from tirzah.config import RuntimeConfig

    try:
        RuntimeConfig(web_api_tokn="typo")  # type: ignore[call-arg]
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_retrieval_bounds_reject_zero_iterations():
    from pydantic import ValidationError
    from tirzah.config import RetrievalConfig

    try:
        RetrievalConfig(memory_agent_max_iterations=0)
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass
