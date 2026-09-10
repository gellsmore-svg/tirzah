import subprocess
from pathlib import Path

from tirzah.adapters.answer import answer_adapter
from tirzah.adapters.claude import ClaudeCliAnswerAdapter, claude_cli_command
from tirzah.adapters.codex import CodexCliAnswerAdapter, codex_cli_command
from tirzah.adapters.google import GoogleCliAnswerAdapter, google_cli_command
from tirzah.adapters.grok import GrokCliAnswerAdapter, grok_cli_command, parse_grok_output
from tirzah.config import RuntimeConfig
from tirzah.cli import init_config_payload


def test_answer_adapter_selects_claude_codex_and_google() -> None:
    assert isinstance(answer_adapter(RuntimeConfig(answer_adapter="claude_cli")), ClaudeCliAnswerAdapter)
    assert isinstance(answer_adapter(RuntimeConfig(answer_adapter="codex_cli")), CodexCliAnswerAdapter)
    assert isinstance(answer_adapter(RuntimeConfig(answer_adapter="google_cli")), GoogleCliAnswerAdapter)
    assert isinstance(answer_adapter(RuntimeConfig(answer_adapter="gemini_cli")), GoogleCliAnswerAdapter)
    assert isinstance(answer_adapter(RuntimeConfig(answer_adapter="grok_cli")), GrokCliAnswerAdapter)


def test_claude_cli_command_includes_print_and_optional_flags() -> None:
    cmd = claude_cli_command(
        RuntimeConfig(claude_model="sonnet", claude_max_turns=2, claude_bare=True)
    )
    assert cmd[:4] == ["claude", "--print", "--output-format", "text"]
    assert "--max-turns" in cmd and "2" in cmd
    assert "--bare" in cmd
    assert "--model" in cmd and "sonnet" in cmd


def test_claude_cli_adapter_pipes_prompt(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(cmd, 0, stdout="claude-answer\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    answer = ClaudeCliAnswerAdapter(RuntimeConfig()).answer(
        {"prompt_text": "hello claude", "context_metadata": {"included": []}}
    )
    assert captured["input"] == "hello claude"
    assert "--print" in captured["cmd"]
    assert answer["adapter"] == "claude_cli"
    assert answer["answer"] == "claude-answer"
    assert answer["confidence"] == "model"


def test_claude_cli_retries_without_optional_flags(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--bare" in cmd:
            raise subprocess.CalledProcessError(2, cmd, output="", stderr="unknown flag: --bare")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    answer = ClaudeCliAnswerAdapter(RuntimeConfig(claude_bare=True)).answer(
        {"prompt_text": "ping", "context_metadata": {"included": []}}
    )
    assert answer["answer"] == "ok"
    assert any("--bare" in cmd for cmd in calls)
    assert any("--bare" not in cmd for cmd in calls)


def test_codex_cli_command_reads_stdin() -> None:
    cmd = codex_cli_command(RuntimeConfig(codex_model="gpt-5.2-codex", codex_sandbox="read-only"))
    assert cmd[:2] == ["codex", "exec"]
    assert "--skip-git-repo-check" in cmd
    assert "--sandbox" in cmd and "read-only" in cmd
    assert "--model" in cmd and "gpt-5.2-codex" in cmd
    assert cmd[-1] == "-"


def test_codex_cli_command_can_prefix_sudo() -> None:
    cmd = codex_cli_command(RuntimeConfig(codex_sudo=True))
    assert cmd[:4] == ["sudo", "-n", "-E", "--"]
    assert "codex" in cmd and "exec" in cmd


def test_codex_cli_adapter_pipes_prompt(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout="codex-answer\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    answer = CodexCliAnswerAdapter(RuntimeConfig()).answer(
        {"prompt_text": "hello codex", "context_metadata": {"included": []}}
    )
    assert captured["input"] == "hello codex"
    assert captured["cmd"][:2] == ["codex", "exec"]
    assert captured["env"]["CODEX_QUIET_MODE"] == "1"
    assert answer["adapter"] == "codex_cli"
    assert answer["answer"] == "codex-answer"


def test_google_cli_command_uses_prompt_flag() -> None:
    cmd = google_cli_command(RuntimeConfig(google_model="gemini-2.5-flash"))
    assert cmd[0] == "gemini"
    assert "--prompt" in cmd
    assert "--model" in cmd and "gemini-2.5-flash" in cmd


def test_google_cli_adapter_pipes_prompt(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(cmd, 0, stdout="gemini-answer\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    answer = GoogleCliAnswerAdapter(RuntimeConfig()).answer(
        {"prompt_text": "hello gemini", "context_metadata": {"included": []}}
    )
    assert captured["input"] == "hello gemini"
    assert "--prompt" in captured["cmd"]
    assert answer["adapter"] == "google_cli"
    assert answer["answer"] == "gemini-answer"
    assert answer["confidence"] == "model"


def test_grok_cli_command_uses_prompt_file() -> None:
    cmd = grok_cli_command(
        RuntimeConfig(grok_model="grok-4.6", grok_max_turns=2, grok_no_auto_update=True),
        prompt_file="/tmp/prompt.txt",
    )
    assert cmd[0] == "grok"
    assert "--prompt-file" in cmd and "/tmp/prompt.txt" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--model" in cmd and "grok-4.6" in cmd
    assert "--max-turns" in cmd and "2" in cmd
    assert "--no-auto-update" in cmd


def test_grok_cli_adapter_writes_prompt_file(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        prompt_file = cmd[cmd.index("--prompt-file") + 1]
        captured["prompt"] = Path(prompt_file).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"text":"grok-answer","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}\n',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    answer = GrokCliAnswerAdapter(RuntimeConfig()).answer(
        {"prompt_text": "hello grok", "context_metadata": {"included": []}}
    )
    assert captured["prompt"] == "hello grok"
    assert "--prompt-file" in captured["cmd"]
    assert answer["adapter"] == "grok_cli"
    assert answer["answer"] == "grok-answer"
    assert answer["usage"]["total"] == 5
    assert answer["confidence"] == "model"


def test_parse_grok_output_falls_back_to_plain_text() -> None:
    text, usage = parse_grok_output("plain pong")
    assert text == "plain pong"
    assert usage is None


def test_init_payload_writes_external_cli_runtimes() -> None:
    for name in ("claude_cli", "codex_cli", "google_cli", "grok_cli"):
        payload = init_config_payload(runtime_choice=name)
        assert payload["runtime"]["answer_adapter"] == name
        assert payload["runtime"]["ingestion_model_adapter"] == name
        assert payload["runtime"]["embedding_adapter"] == "mock"
