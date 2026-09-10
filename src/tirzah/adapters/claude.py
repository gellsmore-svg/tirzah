from __future__ import annotations

import time
from typing import Any

from tirzah.adapters.cli_runtime import (
    NO_TOOLS_INSTRUCTION,
    cli_answer_payload,
    run_prompt_cli,
)
from tirzah.config import RuntimeConfig

CLAUDE_INSTALL_HINT = (
    "Install Claude Code from https://docs.claude.com/en/docs/claude-code/headless "
    "and authenticate with `claude` login or ANTHROPIC_API_KEY."
)


class ClaudeCliAnswerAdapter:
    """Headless Claude Code adapter (`claude -p`)."""

    name = "claude_cli"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        prompt_text = str(prompt.get("prompt_text") or "")
        if not prompt_text.strip():
            raise RuntimeError("Claude CLI adapter received an empty prompt.")
        clock = time.monotonic()
        output = run_claude_cli(self.config, prompt_text)
        duration_ms = int((time.monotonic() - clock) * 1000)
        return cli_answer_payload(
            self.name,
            self.config.claude_model or "claude",
            prompt,
            output,
            duration_ms,
        )


def claude_cli_command(config: RuntimeConfig, *, include_optional_flags: bool = True) -> list[str]:
    cmd = [str(config.claude_executable), "--print", "--output-format", "text"]
    if include_optional_flags:
        cmd.extend(["--max-turns", str(config.claude_max_turns)])
        if config.claude_bare:
            cmd.append("--bare")
        if config.claude_model:
            cmd.extend(["--model", config.claude_model])
    cmd.append(NO_TOOLS_INSTRUCTION)
    return cmd


def run_claude_cli(config: RuntimeConfig, prompt_text: str) -> str:
    return run_prompt_cli(
        cmd=claude_cli_command(config),
        prompt_text=prompt_text,
        timeout=config.claude_timeout_seconds,
        executable=str(config.claude_executable),
        name="Claude CLI",
        install_hint=CLAUDE_INSTALL_HINT,
        fallback_cmd=claude_cli_command(config, include_optional_flags=False),
    )
