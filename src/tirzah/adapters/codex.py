from __future__ import annotations

import time
from typing import Any

from tirzah.adapters.cli_runtime import cli_answer_payload, run_prompt_cli
from tirzah.config import RuntimeConfig

CODEX_INSTALL_HINT = (
    "Install Codex CLI (`npm install -g @openai/codex`) and authenticate with "
    "`codex login` or OPENAI_API_KEY."
)


class CodexCliAnswerAdapter:
    """Headless Codex CLI adapter (`codex exec`)."""

    name = "codex_cli"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        prompt_text = str(prompt.get("prompt_text") or "")
        if not prompt_text.strip():
            raise RuntimeError("Codex CLI adapter received an empty prompt.")
        clock = time.monotonic()
        output = run_codex_cli(self.config, prompt_text)
        duration_ms = int((time.monotonic() - clock) * 1000)
        return cli_answer_payload(
            self.name,
            self.config.codex_model or "codex",
            prompt,
            output,
            duration_ms,
        )


def codex_cli_command(
    config: RuntimeConfig,
    *,
    include_optional_flags: bool = True,
    use_sudo: bool | None = None,
) -> list[str]:
    cmd = [str(config.codex_executable), "exec"]
    if include_optional_flags:
        cmd.append("--skip-git-repo-check")
        if config.codex_sandbox:
            cmd.extend(["--sandbox", config.codex_sandbox])
        if config.codex_ephemeral:
            cmd.append("--ephemeral")
        if config.codex_model:
            cmd.extend(["--model", config.codex_model])
    cmd.append("-")
    if use_sudo if use_sudo is not None else config.codex_sudo:
        cmd = ["sudo", "-n", "-E", "--", *cmd]
    return cmd


def run_codex_cli(config: RuntimeConfig, prompt_text: str) -> str:
    extra_env = {"CODEX_QUIET_MODE": "1"}
    try:
        return run_prompt_cli(
            cmd=codex_cli_command(config),
            prompt_text=prompt_text,
            timeout=config.codex_timeout_seconds,
            executable=str(config.codex_executable),
            name="Codex CLI",
            install_hint=CODEX_INSTALL_HINT,
            extra_env=extra_env,
            fallback_cmd=codex_cli_command(config, include_optional_flags=False),
        )
    except RuntimeError as error:
        if config.codex_sudo or not looks_like_codex_privilege_error(error):
            raise
        try:
            return run_prompt_cli(
                cmd=codex_cli_command(config, use_sudo=True),
                prompt_text=prompt_text,
                timeout=config.codex_timeout_seconds,
                executable=str(config.codex_executable),
                name="Codex CLI",
                install_hint=CODEX_INSTALL_HINT,
                extra_env=extra_env,
                fallback_cmd=codex_cli_command(
                    config, include_optional_flags=False, use_sudo=True
                ),
            )
        except RuntimeError as sudo_error:
            if looks_like_sudo_password_error(sudo_error):
                raise RuntimeError(
                    f"{error} Retrying Codex under sudo needs passwordless "
                    "`sudo -n` (this session has no TTY for a password prompt). "
                    "Set runtime.codex_sudo: true after NOPASSWD, or run the "
                    "call from a root shell."
                ) from sudo_error
            raise


def looks_like_codex_privilege_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "operation not permitted",
            "permission denied",
            "eperm",
            "unprivileged user namespace",
            "apparmor",
            "bwrap",
            "landlock",
            "user namespace",
        )
    )


def looks_like_sudo_password_error(error: Exception) -> bool:
    text = str(error).lower()
    return "password is required" in text or "a terminal is required" in text
