from __future__ import annotations

import time
from typing import Any

from tirzah.adapters.cli_runtime import (
    NO_TOOLS_INSTRUCTION,
    cli_answer_payload,
    run_prompt_cli,
)
from tirzah.config import RuntimeConfig

GOOGLE_INSTALL_HINT = (
    "Install Gemini CLI from https://geminicli.com/docs/cli/headless/ "
    "(`gemini`) and authenticate with `gemini` login or GEMINI_API_KEY."
)


class GoogleCliAnswerAdapter:
    """Headless Google Gemini CLI adapter (`gemini -p`)."""

    name = "google_cli"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        prompt_text = str(prompt.get("prompt_text") or "")
        if not prompt_text.strip():
            raise RuntimeError("Google CLI adapter received an empty prompt.")
        clock = time.monotonic()
        output = run_google_cli(self.config, prompt_text)
        duration_ms = int((time.monotonic() - clock) * 1000)
        return cli_answer_payload(
            self.name,
            self.config.google_model or "gemini",
            prompt,
            output,
            duration_ms,
        )


def google_cli_command(config: RuntimeConfig, *, include_optional_flags: bool = True) -> list[str]:
    cmd = [str(config.google_executable), "--prompt", NO_TOOLS_INSTRUCTION]
    if include_optional_flags and config.google_model:
        cmd.extend(["--model", config.google_model])
    return cmd


def run_google_cli(config: RuntimeConfig, prompt_text: str) -> str:
    return run_prompt_cli(
        cmd=google_cli_command(config),
        prompt_text=prompt_text,
        timeout=config.google_timeout_seconds,
        executable=str(config.google_executable),
        name="Google CLI",
        install_hint=GOOGLE_INSTALL_HINT,
        fallback_cmd=google_cli_command(config, include_optional_flags=False),
    )
