from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any
from urllib import request

from tirzah.adapters.hoglah_runtime import HoglahJobRunner
from tirzah.config import RuntimeConfig


class MockAnswerAdapter:
    name = "mock_answer"

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        clock = time.monotonic()
        context_text = prompt.get("context_text", "")
        answer_lines = [
            "Mock answer based on retrieved context.",
            "",
            summarize_context_text(context_text),
        ]
        answer_text = "\n".join(answer_lines).strip()
        duration_ms = int((time.monotonic() - clock) * 1000)
        # Deterministic mock: no real model — still stamp duration so the spine
        # can measure coverage (Galeed F2 / Tirzah instrumentation backlog).
        return answer_payload(
            self.name,
            "mock",
            prompt,
            answer_text,
            usage=_estimate_usage_from_text(prompt.get("prompt_text") or context_text, answer_text),
            duration_ms=duration_ms,
            confidence="mock",
        )


def summarize_context_text(context_text: str) -> str:
    body_lines = [
        line.strip()
        for line in context_text.splitlines()
        if line.strip()
        and not line.startswith("#")
        and not line.startswith("- Node ID:")
        and not line.startswith("- Labels:")
        and not line.startswith("- Endorsement:")
        and not line.startswith("- Source:")
        and not line.startswith("Document:")
        and not line.startswith("Document ID:")
        and not line.startswith("Focus Node ID:")
        and not line.startswith("##")
    ]
    return " ".join(body_lines[:6]) or "No usable context text was retrieved."


class OllamaCliAnswerAdapter:
    name = "ollama_cli"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        env.setdefault("TERM", "dumb")
        cmd = ollama_cli_command(self.config)
        clock = time.monotonic()
        try:
            completed = subprocess.run(
                cmd,
                input=prompt["prompt_text"],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.ollama_timeout_seconds,
                env=env,
            )
        except subprocess.CalledProcessError as error:
            if not uses_optional_ollama_flags(cmd) or not is_unsupported_flag_error(error):
                raise RuntimeError(ollama_error_message(error)) from error
            completed = self._run_without_optional_flags(prompt, env)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"Ollama CLI timed out after {self.config.ollama_timeout_seconds}s "
                f"while running {self.config.ollama_model}."
            ) from error
        duration_ms = int((time.monotonic() - clock) * 1000)
        answer_text = clean_ollama_output(completed.stdout)
        if not answer_text:
            raise RuntimeError("Ollama CLI returned an empty response.")
        # CLI path has no structured token counts — wall-clock duration only.
        return answer_payload(
            self.name,
            self.config.ollama_model,
            prompt,
            answer_text,
            duration_ms=duration_ms,
        )

    def _run_without_optional_flags(self, prompt: dict[str, Any], env: dict[str, str]):
        try:
            return subprocess.run(
                ollama_cli_command(self.config, include_optional_flags=False),
                input=prompt["prompt_text"],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.ollama_timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"Ollama CLI timed out after {self.config.ollama_timeout_seconds}s "
                f"while running {self.config.ollama_model}."
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(ollama_error_message(error)) from error


class OllamaHttpAnswerAdapter:
    name = "ollama_http"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        request_body = {
            "model": self.config.ollama_model,
            "prompt": prompt["prompt_text"],
            "stream": False,
        }
        if getattr(self.config, "ollama_num_ctx", 0):
            request_body["options"] = {"num_ctx": self.config.ollama_num_ctx}
        if self.config.ollama_format:
            request_body["format"] = self.config.ollama_format
        think_value = ollama_think_http_value(self.config.ollama_think)
        if think_value is not None:
            request_body["think"] = think_value
        payload = json.dumps(request_body).encode("utf-8")
        req = request.Request(
            f"{self.config.ollama_base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        clock = time.monotonic()
        with request.urlopen(req, timeout=self.config.ollama_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        wall_ms = int((time.monotonic() - clock) * 1000)
        if not isinstance(data, dict):
            data = {}
        duration_ms = _duration_ms_from_ollama(data) or wall_ms
        usage = _usage_from_ollama_generate(data)
        return answer_payload(
            self.name,
            self.config.ollama_model,
            prompt,
            data.get("response", ""),
            usage=usage,
            duration_ms=duration_ms,
        )


class HoglahAnswerAdapter:
    """Route answer generation through a Hoglah queue daemon (decoupled topology).

    Tirzah submits a generate job into the shared queue and awaits the terminal
    result by output-folder poll or HTTP callback; a SEPARATE `hoglah run --real`
    daemon executes it. The Hoglah runner is built lazily so merely selecting
    this adapter (e.g. in the factory) does not require the optional hoglah
    package or open a callback port until an answer is actually requested.
    """

    name = "hoglah"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._runner: HoglahJobRunner | None = None

    def _get_runner(self) -> HoglahJobRunner:
        if self._runner is None:
            self._runner = HoglahJobRunner(self.config)
        return self._runner

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        clock = time.monotonic()
        result = self._get_runner().run_generate(
            prompt["prompt_text"],
            model=self.config.ollama_model,
            fmt=self.config.ollama_format,
            tags=["tirzah"],
            metadata={"source": "tirzah"},
        )
        wall_ms = int((time.monotonic() - clock) * 1000)
        status = result.get("status")
        job_id = result.get("job_id")
        if status != "completed":
            raise RuntimeError(result.get("error") or f"Hoglah job {job_id} ended with status {status}.")
        output = result.get("output")
        if not output:
            raise RuntimeError(f"Hoglah job {job_id} completed without output.")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        duration_ms = result.get("duration_ms")
        if duration_ms is None:
            duration_ms = result.get("processing_duration_ms")
        if duration_ms is None:
            duration_ms = wall_ms
        payload = answer_payload(
            self.name,
            self.config.ollama_model,
            prompt,
            output,
            usage=usage if usage else None,
            duration_ms=int(duration_ms) if duration_ms is not None else None,
        )
        payload["hoglah_job_id"] = job_id
        if result.get("truncated"):
            payload["truncated"] = True
            payload["truncation_reason"] = result.get("truncation_reason")
        return payload

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None


def answer_adapter(config: RuntimeConfig):
    if config.answer_adapter == "mock":
        return MockAnswerAdapter()
    if config.answer_adapter == "ollama_cli":
        return OllamaCliAnswerAdapter(config)
    if config.answer_adapter == "ollama_http":
        return OllamaHttpAnswerAdapter(config)
    if config.answer_adapter == "hoglah":
        return HoglahAnswerAdapter(config)
    if config.answer_adapter == "kiro_cli":
        from tirzah.adapters.kiro import KiroCliAnswerAdapter

        return KiroCliAnswerAdapter(config)
    if config.answer_adapter == "claude_cli":
        from tirzah.adapters.claude import ClaudeCliAnswerAdapter

        return ClaudeCliAnswerAdapter(config)
    if config.answer_adapter == "codex_cli":
        from tirzah.adapters.codex import CodexCliAnswerAdapter

        return CodexCliAnswerAdapter(config)
    if config.answer_adapter in {"google_cli", "gemini_cli"}:
        from tirzah.adapters.google import GoogleCliAnswerAdapter

        return GoogleCliAnswerAdapter(config)
    if config.answer_adapter == "grok_cli":
        from tirzah.adapters.grok import GrokCliAnswerAdapter

        return GrokCliAnswerAdapter(config)
    raise ValueError(f"Unknown answer adapter: {config.answer_adapter}")


def generate_text(
    config: RuntimeConfig,
    prompt_text: str,
    *,
    adapter_name: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run a prompt through an answer adapter and return the payload dict."""
    updates: dict[str, Any] = {}
    if adapter_name:
        updates["answer_adapter"] = adapter_name
    if model:
        from tirzah.adapters.cli_runtime import EXTERNAL_MODEL_FIELDS

        chosen = adapter_name or config.answer_adapter
        field = EXTERNAL_MODEL_FIELDS.get(chosen, "ollama_model")
        updates[field] = model
    runtime = config.model_copy(update=updates) if updates else config
    adapter = answer_adapter(runtime)
    try:
        return adapter.answer({"prompt_text": prompt_text, "context_metadata": {"included": []}})
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def ollama_cli_command(config: RuntimeConfig, include_optional_flags: bool = True) -> list[str]:
    cmd = [
        str(config.ollama_executable),
        "run",
        "--nowordwrap",
    ]
    if include_optional_flags:
        if config.ollama_format:
            cmd.extend(["--format", config.ollama_format])
        think_value = ollama_think_cli_value(config.ollama_think)
        if think_value is not None:
            cmd.append(f"--think={think_value}")
        if config.ollama_hide_thinking:
            cmd.append("--hidethinking")
    cmd.append(config.ollama_model)
    return cmd


def ollama_think_cli_value(value: bool | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    return value.strip().lower()


def ollama_think_http_value(value: bool | str | None) -> bool | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return lowered


def uses_optional_ollama_flags(cmd: list[str]) -> bool:
    return any(
        item == "--format" or item == "--hidethinking" or item.startswith("--think=")
        for item in cmd
    )


def is_unsupported_flag_error(error: subprocess.CalledProcessError) -> bool:
    text = "\n".join(str(part or "") for part in [error.stderr, error.stdout, error])
    lowered = text.lower()
    return "unknown flag" in lowered or "flag provided but not defined" in lowered


def ollama_error_message(error: subprocess.CalledProcessError) -> str:
    detail = clean_ollama_output("\n".join(str(part or "") for part in [error.stderr, error.stdout]))
    if detail:
        return f"Ollama CLI failed: {detail}"
    return f"Ollama CLI failed with exit code {error.returncode}."


def answer_payload(
    adapter: str,
    model: str,
    prompt: dict[str, Any],
    answer_text: str,
    *,
    usage: dict[str, int] | None = None,
    duration_ms: int | None = None,
    confidence: str | None = None,
) -> dict[str, Any]:
    included = prompt.get("context_metadata", {}).get("included", [])
    if confidence is None:
        confidence = (
            "model"
            if adapter.startswith("ollama")
            or adapter
            in {"hoglah", "kiro_cli", "claude_cli", "codex_cli", "google_cli", "gemini_cli", "grok_cli"}
            else "mock"
        )
    payload: dict[str, Any] = {
        "adapter": adapter,
        "model": model,
        "answer": answer_text.strip(),
        "used_node_ids": [record["node_id"] for record in included],
        "confidence": confidence,
    }
    if usage:
        payload["usage"] = {
            k: int(v) for k, v in usage.items() if isinstance(v, (int, float))
        }
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    return payload


def instrumentation_from_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Subset of adapter answer fields for process_trace / galeed llm_calls."""
    out: dict[str, Any] = {}
    usage = answer.get("usage")
    if isinstance(usage, dict) and usage:
        out["usage"] = dict(usage)
    if answer.get("duration_ms") is not None:
        out["duration_ms"] = int(answer["duration_ms"])
    return out


def _usage_from_ollama_generate(data: dict[str, Any]) -> dict[str, int]:
    """Ollama /api/generate reports prompt_eval_count / eval_count."""
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)
    if not prompt_tokens and not completion_tokens:
        return {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total": prompt_tokens + completion_tokens,
    }


def _duration_ms_from_ollama(data: dict[str, Any]) -> int | None:
    """Ollama ``total_duration`` is nanoseconds when present."""
    total_ns = data.get("total_duration")
    if total_ns is None:
        return None
    try:
        return max(0, int(int(total_ns) / 1_000_000))
    except (TypeError, ValueError):
        return None


def _estimate_usage_from_text(prompt_text: str, answer_text: str) -> dict[str, int]:
    """Rough token estimate (~4 chars/token) when the model reports none."""
    prompt_tokens = max(1, len(prompt_text or "") // 4) if prompt_text else 0
    completion_tokens = max(1, len(answer_text or "") // 4) if answer_text else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total": prompt_tokens + completion_tokens,
    }


def clean_ollama_output(text: str) -> str:
    rewritten = apply_terminal_rewrites(text)
    without_ansi = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", rewritten)
    without_spinners = re.sub(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]\s*", "", without_ansi)
    without_controls = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", without_spinners)
    repaired_wraps = repair_duplicate_wrap_fragments(without_controls)
    return repaired_wraps.strip()


def repair_duplicate_wrap_fragments(text: str) -> str:
    """Remove stale line-end prefixes left by Ollama terminal wrapping."""
    return re.sub(
        r"(?<![A-Za-z])([A-Za-z]{1,20})\n(?=\1[A-Za-z])",
        "",
        text,
    )


def apply_terminal_rewrites(text: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "\x1b":
            output.append(char)
            index += 1
            continue
        match = re.match(r"\x1b\[([0-9]*)([A-Za-z])", text[index:])
        if not match:
            index += 1
            continue
        amount = int(match.group(1) or "1")
        command = match.group(2)
        if command == "D":
            for _ in range(min(amount, len(output))):
                output.pop()
        elif command == "K":
            while output and output[-1] != "\n":
                output.pop()
        elif command == "G":
            while output and output[-1] != "\n":
                output.pop()
        index += len(match.group(0))
    return "".join(output)
