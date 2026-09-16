import importlib.util
import json
import subprocess

import pytest

from tirzah.adapters.answer import (
    HoglahAnswerAdapter,
    MockAnswerAdapter,
    OllamaCliAnswerAdapter,
    OllamaHttpAnswerAdapter,
    answer_adapter,
    clean_ollama_output,
    repair_duplicate_wrap_fragments,
    summarize_context_text,
)
from tirzah.config import RuntimeConfig


def test_summarize_context_text_ignores_metadata_lines() -> None:
    summary = summarize_context_text(
        "# Tirzah Context\n"
        "Document: Doc\n"
        "Document ID: doc1\n"
        "## Context Records\n"
        "- Node ID: node1\n"
        "Actual content.\n"
    )

    assert summary == "Actual content."


def test_mock_answer_adapter_returns_used_nodes() -> None:
    answer = MockAnswerAdapter().answer(
        {
            "context_text": "Useful context.",
            "context_metadata": {
                "included": [{"node_id": "node1"}, {"node_id": "node2"}],
            },
        }
    )

    assert answer["adapter"] == "mock_answer"
    assert answer["used_node_ids"] == ["node1", "node2"]
    assert answer.get("duration_ms") is not None
    assert answer.get("usage", {}).get("total", 0) > 0


def test_ollama_http_adapter_captures_usage_and_duration(monkeypatch) -> None:
    from tirzah.adapters import answer as answer_mod

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "response": "hello from ollama",
                "prompt_eval_count": 11,
                "eval_count": 7,
                "total_duration": 250_000_000,  # 250 ms in ns
            }).encode("utf-8")

    monkeypatch.setattr(answer_mod.request, "urlopen", lambda *a, **k: FakeResponse())
    answer = OllamaHttpAnswerAdapter(RuntimeConfig(ollama_model="gemma3:1b")).answer(
        {"prompt_text": "hi", "context_metadata": {"included": []}}
    )
    assert answer["answer"] == "hello from ollama"
    assert answer["usage"]["prompt_tokens"] == 11
    assert answer["usage"]["completion_tokens"] == 7
    assert answer["usage"]["total"] == 18
    assert answer["duration_ms"] == 250


def test_clean_ollama_output_strips_spinner_and_ansi() -> None:
    assert clean_ollama_output("\x1b[?25l⠙ \x1b[Ktirzah-ok\n") == "tirzah-ok"


def test_clean_ollama_output_applies_cursor_rewrites() -> None:
    assert clean_ollama_output("prompte\x1b[7D\x1b[Kprompted\n") == "prompted"


def test_clean_ollama_output_repairs_duplicate_wrap_fragments() -> None:
    raw = (
        "I understand you are asking about the MongoDB instance used when I am prompte\n"
        "prompted to answer your questions. However, the u\n"
        "underlying system includes Mo\n"
        "MongoDB."
    )

    assert clean_ollama_output(raw) == (
        "I understand you are asking about the MongoDB instance used when I am "
        "prompted to answer your questions. However, the underlying system includes MongoDB."
    )


def test_repair_duplicate_wrap_fragments_leaves_intentional_newlines() -> None:
    assert repair_duplicate_wrap_fragments("first line\nsecond line") == "first line\nsecond line"


def test_ollama_cli_adapter_passes_prompt_via_stdin(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(cmd, 0, stdout="answer\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    answer = OllamaCliAnswerAdapter(RuntimeConfig(ollama_model="gemma3:1b")).answer(
        {
            "prompt_text": "prompt body",
            "context_metadata": {"included": []},
        }
    )

    assert captured["cmd"][-4:] == ["--nowordwrap", "--think=false", "--hidethinking", "gemma3:1b"]
    assert captured["input"] == "prompt body"
    assert answer["answer"] == "answer"


def test_ollama_cli_adapter_can_request_json_format(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    answer = OllamaCliAnswerAdapter(
        RuntimeConfig(ollama_model="qwen3.6:latest", ollama_format="json")
    ).answer(
        {
            "prompt_text": "return json",
            "context_metadata": {"included": []},
        }
    )

    assert captured["cmd"] == [
        str(RuntimeConfig().ollama_executable),
        "run",
        "--nowordwrap",
        "--format",
        "json",
        "--think=false",
        "--hidethinking",
        "qwen3.6:latest",
    ]
    assert answer["answer"] == '{"ok":true}'


def test_ollama_cli_adapter_reports_timeout(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=3)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(TimeoutError, match="timed out"):
        OllamaCliAnswerAdapter(RuntimeConfig(ollama_timeout_seconds=3)).answer(
            {
                "prompt_text": "prompt body",
                "context_metadata": {"included": []},
            }
        )


def test_ollama_cli_adapter_retries_without_optional_flags_for_old_ollama(monkeypatch) -> None:
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        if len(captured) == 1:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                stderr="Error: unknown flag: --think",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="answer\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    answer = OllamaCliAnswerAdapter(RuntimeConfig(ollama_model="gemma3:1b")).answer(
        {
            "prompt_text": "prompt body",
            "context_metadata": {"included": []},
        }
    )

    assert "--think=false" in captured[0]
    assert "--hidethinking" in captured[0]
    assert captured[1] == [
        str(RuntimeConfig().ollama_executable),
        "run",
        "--nowordwrap",
        "gemma3:1b",
    ]
    assert answer["answer"] == "answer"


def test_ollama_cli_adapter_wraps_fallback_failure(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        if "--think=false" in cmd:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                stderr="Error: unknown flag: --think",
            )
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=cmd,
            stderr="Error: model failed",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="model failed"):
        OllamaCliAnswerAdapter(RuntimeConfig(ollama_model="gemma3:1b")).answer(
            {
                "prompt_text": "prompt body",
                "context_metadata": {"included": []},
            }
        )


def test_ollama_cli_adapter_can_omit_optional_flags(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="answer\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    OllamaCliAnswerAdapter(
        RuntimeConfig(ollama_think=None, ollama_hide_thinking=False)
    ).answer(
        {
            "prompt_text": "prompt body",
            "context_metadata": {"included": []},
        }
    )

    assert "--think=false" not in captured["cmd"]
    assert "--hidethinking" not in captured["cmd"]


def test_ollama_http_adapter_sends_format_and_think(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"response":"answer"}'

    def fake_urlopen(req, timeout):
        captured["body"] = req.data
        captured["timeout"] = timeout
        return FakeResponse()

    import tirzah.adapters.answer as answer_module

    monkeypatch.setattr(answer_module.request, "urlopen", fake_urlopen)

    answer = OllamaHttpAnswerAdapter(
        RuntimeConfig(ollama_format="json", ollama_think=False)
    ).answer(
        {
            "prompt_text": "prompt body",
            "context_metadata": {"included": []},
        }
    )

    assert b'"format": "json"' in captured["body"]
    assert b'"think": false' in captured["body"]
    assert captured["timeout"] == RuntimeConfig().ollama_timeout_seconds
    assert answer["answer"] == "answer"

def test_ollama_http_sends_sampling_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"response":"answer"}'

    def fake_urlopen(req, timeout):
        captured["body"] = req.data
        return FakeResponse()

    import tirzah.adapters.answer as answer_module

    monkeypatch.setattr(answer_module.request, "urlopen", fake_urlopen)
    prompt = {"prompt_text": "prompt body", "context_metadata": {"included": []}}

    OllamaHttpAnswerAdapter(
        RuntimeConfig(ollama_temperature=0.0, ollama_seed=7, ollama_num_ctx=2048)
    ).answer(prompt)
    assert b'"temperature": 0.0' in captured["body"]
    assert b'"seed": 7' in captured["body"]
    assert b'"num_ctx": 2048' in captured["body"]

    # Unset, the model keeps its own sampling.
    OllamaHttpAnswerAdapter(RuntimeConfig(ollama_num_ctx=0)).answer(prompt)
    assert b'"options"' not in captured["body"]


requires_hoglah = pytest.mark.skipif(
    importlib.util.find_spec("hoglah") is None,
    reason="hoglah optional dependency is not installed",
)


def _hoglah_answer_config(tmp_path, delivery="poll"):
    return RuntimeConfig(
        answer_adapter="hoglah",
        ollama_model="stub-model:1b",
        hoglah_db_path=tmp_path / "hoglah.sqlite3",
        hoglah_output_dir=tmp_path / "outbox",
        hoglah_delivery=delivery,
        hoglah_callback_host="127.0.0.1",
        hoglah_callback_port=0,
        hoglah_wait_timeout_seconds=5,
    )


def _stub_worker(tmp_path):
    import hoglah

    return hoglah.Hoglah(
        config={
            "db_path": str(tmp_path / "hoglah.sqlite3"),
            "output_dir": str(tmp_path / "outbox"),
        },
        start_worker=True,
    )


@requires_hoglah
@pytest.mark.parametrize("delivery", ["poll", "callback"])
def test_hoglah_answer_adapter_via_stub_worker(tmp_path, delivery) -> None:
    """Decoupled topology: submit to the shared queue, a separate worker (here a
    Hoglah StubAdapter worker) executes it, and the result returns by poll or
    callback. Proves submit -> daemon -> deliver without Ollama."""
    worker = _stub_worker(tmp_path)
    adapter = HoglahAnswerAdapter(_hoglah_answer_config(tmp_path, delivery))
    try:
        result = adapter.answer(
            {
                "prompt_text": "prompt body",
                "context_metadata": {"included": [{"node_id": "node1"}]},
            }
        )
    finally:
        adapter.close()
        worker.close()

    assert result["adapter"] == "hoglah"
    assert "[STUB]" in result["answer"]
    assert result["used_node_ids"] == ["node1"]
    assert result["hoglah_job_id"]


def test_hoglah_answer_adapter_reports_missing_optional_dependency(monkeypatch) -> None:
    import tirzah.adapters.hoglah_runtime as runtime_module

    def fake_import_module(_name):
        raise ImportError("missing")

    monkeypatch.setattr(runtime_module, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="tirzah\\[hoglah\\]"):
        answer_adapter(RuntimeConfig(answer_adapter="hoglah")).answer(
            {
                "prompt_text": "prompt body",
                "context_metadata": {"included": []},
            }
        )
