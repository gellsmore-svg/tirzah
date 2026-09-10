from __future__ import annotations

import os
import subprocess
import time
from typing import Any

from tirzah.config import RuntimeConfig

KIRO_INSTALL_HINT = (
    "Install Kiro CLI from https://kiro.dev/docs/cli/installation/ "
    "(curl -fsSL https://cli.kiro.dev/install | bash) and authenticate with "
    "kiro-cli login or KIRO_API_KEY."
)


class KiroCliAnswerAdapter:
    """Headless Kiro CLI answer adapter (`kiro-cli chat --no-interactive`)."""

    name = "kiro_cli"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        from tirzah.adapters.answer import answer_payload

        prompt_text = str(prompt.get("prompt_text") or "")
        if not prompt_text.strip():
            raise RuntimeError("Kiro CLI adapter received an empty prompt.")
        clock = time.monotonic()
        output = run_kiro_cli(self.config, prompt_text)
        duration_ms = int((time.monotonic() - clock) * 1000)
        model = self.config.kiro_model or "kiro"
        return answer_payload(
            self.name,
            model,
            prompt,
            output,
            duration_ms=duration_ms,
        )


def kiro_cli_command(
    config: RuntimeConfig,
    *,
    include_optional_flags: bool = True,
    instruction: str,
) -> list[str]:
    cmd = [
        str(config.kiro_executable),
        "chat",
        "--no-interactive",
        "--wrap",
        "never",
    ]
    if include_optional_flags:
        if config.kiro_agent:
            cmd.extend(["--agent", config.kiro_agent])
        if config.kiro_effort:
            cmd.extend(["--effort", config.kiro_effort])
        if config.kiro_model:
            cmd.extend(["--model", config.kiro_model])
        if config.kiro_trust_tools:
            cmd.append(f"--trust-tools={config.kiro_trust_tools}")
    cmd.append(instruction)
    return cmd


def run_kiro_cli(config: RuntimeConfig, prompt_text: str) -> str:
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("KIRO_LOG_NO_COLOR", "1")
    instruction = (
        "Follow the instructions in the provided text and reply with only the "
        "requested output. Do not call tools."
    )
    cmd = kiro_cli_command(config, instruction=instruction)
    try:
        completed = subprocess.run(
            cmd,
            input=prompt_text,
            check=True,
            capture_output=True,
            text=True,
            timeout=config.kiro_timeout_seconds,
            env=env,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Kiro CLI executable not found: {config.kiro_executable}. {KIRO_INSTALL_HINT}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Kiro CLI timed out after {config.kiro_timeout_seconds}s."
        ) from error
    except subprocess.CalledProcessError as error:
        if uses_optional_kiro_flags(cmd) and is_unsupported_flag_error(error):
            return run_kiro_cli_without_optional_flags(config, prompt_text, env, instruction)
        raise RuntimeError(kiro_error_message(error)) from error
    from tirzah.adapters.answer import clean_ollama_output

    answer_text = clean_ollama_output(completed.stdout)
    if not answer_text:
        raise RuntimeError("Kiro CLI returned an empty response.")
    return answer_text


def run_kiro_cli_without_optional_flags(
    config: RuntimeConfig,
    prompt_text: str,
    env: dict[str, str],
    instruction: str,
) -> str:
    cmd = kiro_cli_command(config, include_optional_flags=False, instruction=instruction)
    try:
        completed = subprocess.run(
            cmd,
            input=prompt_text,
            check=True,
            capture_output=True,
            text=True,
            timeout=config.kiro_timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Kiro CLI timed out after {config.kiro_timeout_seconds}s."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(kiro_error_message(error)) from error
    from tirzah.adapters.answer import clean_ollama_output

    answer_text = clean_ollama_output(completed.stdout)
    if not answer_text:
        raise RuntimeError("Kiro CLI returned an empty response.")
    return answer_text


def uses_optional_kiro_flags(cmd: list[str]) -> bool:
    return any(
        item in {"--agent", "--effort", "--model"} or item.startswith("--trust-tools=")
        for item in cmd
    )


def is_unsupported_flag_error(error: subprocess.CalledProcessError) -> bool:
    text = "\n".join(str(part or "") for part in [error.stderr, error.stdout, error])
    lowered = text.lower()
    return "unknown flag" in lowered or "flag provided but not defined" in lowered or "unexpected argument" in lowered


def kiro_error_message(error: subprocess.CalledProcessError) -> str:
    from tirzah.adapters.answer import clean_ollama_output

    detail = clean_ollama_output("\n".join(str(part or "") for part in [error.stderr, error.stdout]))
    if detail:
        return f"Kiro CLI failed: {detail}"
    return f"Kiro CLI failed with exit code {error.returncode}."
