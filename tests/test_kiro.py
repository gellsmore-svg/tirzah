import subprocess

from tirzah.adapters.answer import answer_adapter
from tirzah.adapters.kiro import KiroCliAnswerAdapter, kiro_cli_command
from tirzah.config import RuntimeConfig


def test_answer_adapter_selects_kiro_cli() -> None:
    adapter = answer_adapter(RuntimeConfig(answer_adapter="kiro_cli"))
    assert isinstance(adapter, KiroCliAnswerAdapter)


def test_kiro_cli_command_includes_optional_flags() -> None:
    config = RuntimeConfig(
        kiro_executable="kiro-cli",
        kiro_agent="ingest",
        kiro_effort="low",
        kiro_model="claude-sonnet",
        kiro_trust_tools="read",
    )
    cmd = kiro_cli_command(config, instruction="Reply with JSON only.")
    assert cmd[:5] == ["kiro-cli", "chat", "--no-interactive", "--wrap", "never"]
    assert "--agent" in cmd and "ingest" in cmd
    assert "--effort" in cmd and "low" in cmd
    assert "--model" in cmd and "claude-sonnet" in cmd
    assert "--trust-tools=read" in cmd
    assert cmd[-1] == "Reply with JSON only."


def test_kiro_cli_adapter_pipes_prompt_on_stdin(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout="kiro-answer\n", stderr="")

    monkeypatch.setenv("KIRO_API_KEY", "secret-key")
    monkeypatch.setattr(subprocess, "run", fake_run)
    answer = KiroCliAnswerAdapter(
        RuntimeConfig(answer_adapter="kiro_cli")
    ).answer({"prompt_text": "hello kiro", "context_metadata": {"included": []}})
    assert captured["input"] == "hello kiro"
    assert captured["cmd"][0] == "kiro-cli"
    assert "--no-interactive" in captured["cmd"]
    assert captured["env"]["KIRO_API_KEY"] == "secret-key"
    assert answer["adapter"] == "kiro_cli"
    assert answer["answer"] == "kiro-answer"
    assert answer["confidence"] == "model"
    assert answer.get("duration_ms") is not None


def test_kiro_cli_retries_without_optional_flags(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--model" in cmd:
            raise subprocess.CalledProcessError(2, cmd, output="", stderr="unknown flag: --model")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    answer = KiroCliAnswerAdapter(
        RuntimeConfig(answer_adapter="kiro_cli", kiro_model="claude-sonnet")
    ).answer({"prompt_text": "ping", "context_metadata": {"included": []}})
    assert answer["answer"] == "ok"
    assert any("--model" in cmd for cmd in calls)
    assert any("--model" not in cmd for cmd in calls)
