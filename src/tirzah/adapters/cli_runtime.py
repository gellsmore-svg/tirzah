from __future__ import annotations

import os
import subprocess
from typing import Any

NO_TOOLS_INSTRUCTION = (
    "Follow the instructions in the provided text and reply with only the "
    "requested output. Do not call tools."
)

EXTERNAL_GENERATOR_ADAPTERS = {
    "kiro_cli",
    "claude_cli",
    "codex_cli",
    "google_cli",
    "gemini_cli",
    "grok_cli",
}

GENERATOR_ADAPTERS = {
    "ollama_http",
    "ollama_cli",
    "hoglah",
    "mock",
    *EXTERNAL_GENERATOR_ADAPTERS,
}

EXTERNAL_MODEL_FIELDS = {
    "kiro_cli": "kiro_model",
    "claude_cli": "claude_model",
    "codex_cli": "codex_model",
    "google_cli": "google_model",
    "gemini_cli": "google_model",
    "grok_cli": "grok_model",
}


def cli_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    if extra:
        env.update(extra)
    return env


def is_unsupported_flag_error(error: subprocess.CalledProcessError) -> bool:
    text = "\n".join(str(part or "") for part in [error.stderr, error.stdout, error])
    lowered = text.lower()
    return (
        "unknown flag" in lowered
        or "flag provided but not defined" in lowered
        or "unexpected argument" in lowered
        or "unrecognized arguments" in lowered
        or "no such option" in lowered
    )


def run_prompt_cli(
    *,
    cmd: list[str],
    prompt_text: str,
    timeout: int,
    executable: str,
    name: str,
    install_hint: str,
    extra_env: dict[str, str] | None = None,
    fallback_cmd: list[str] | None = None,
) -> str:
    env = cli_env(extra_env)
    try:
        completed = subprocess.run(
            cmd,
            input=prompt_text,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"{name} executable not found: {executable}. {install_hint}") from error
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"{name} timed out after {timeout}s.") from error
    except subprocess.CalledProcessError as error:
        if fallback_cmd is not None and fallback_cmd != cmd and is_unsupported_flag_error(error):
            return run_prompt_cli(
                cmd=fallback_cmd,
                prompt_text=prompt_text,
                timeout=timeout,
                executable=executable,
                name=name,
                install_hint=install_hint,
                extra_env=extra_env,
            )
        raise RuntimeError(cli_error_message(name, error)) from error
    from tirzah.adapters.answer import clean_ollama_output

    answer_text = clean_ollama_output(completed.stdout)
    if not answer_text:
        raise RuntimeError(f"{name} returned an empty response.")
    return answer_text


def cli_error_message(name: str, error: subprocess.CalledProcessError) -> str:
    from tirzah.adapters.answer import clean_ollama_output

    detail = clean_ollama_output("\n".join(str(part or "") for part in [error.stderr, error.stdout]))
    if detail:
        return f"{name} failed: {detail}"
    return f"{name} failed with exit code {error.returncode}."


def cli_answer_payload(adapter_name: str, model: str, prompt: dict[str, Any], output: str, duration_ms: int):
    from tirzah.adapters.answer import answer_payload

    return answer_payload(adapter_name, model, prompt, output, duration_ms=duration_ms)
