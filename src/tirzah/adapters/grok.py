from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from tirzah.adapters.cli_runtime import run_prompt_cli
from tirzah.config import RuntimeConfig

GROK_INSTALL_HINT = (
    "Install Grok CLI from https://x.ai/cli/install.sh (`grok`) and authenticate "
    "with `grok` login or XAI_API_KEY."
)


class GrokCliAnswerAdapter:
    """Headless Grok Build CLI adapter (`grok -p` / `--prompt-file`)."""

    name = "grok_cli"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def answer(self, prompt: dict[str, Any]) -> dict[str, Any]:
        prompt_text = str(prompt.get("prompt_text") or "")
        if not prompt_text.strip():
            raise RuntimeError("Grok CLI adapter received an empty prompt.")
        clock = time.monotonic()
        output, usage = run_grok_cli(self.config, prompt_text)
        duration_ms = int((time.monotonic() - clock) * 1000)
        from tirzah.adapters.answer import answer_payload

        return answer_payload(
            self.name,
            self.config.grok_model or "grok",
            prompt,
            output,
            usage=usage,
            duration_ms=duration_ms,
        )


def grok_cli_command(
    config: RuntimeConfig,
    *,
    prompt_file: str,
    include_optional_flags: bool = True,
) -> list[str]:
    cmd = [str(config.grok_executable), "--prompt-file", prompt_file]
    if include_optional_flags:
        cmd.extend(["--output-format", "json"])
        cmd.append("--no-alt-screen")
        if config.grok_no_auto_update:
            cmd.append("--no-auto-update")
        cmd.extend(["--max-turns", str(config.grok_max_turns)])
        if config.grok_model:
            cmd.extend(["--model", config.grok_model])
        if config.grok_disallowed_tools:
            cmd.extend(["--disallowed-tools", config.grok_disallowed_tools])
        if config.grok_sandbox:
            cmd.extend(["--sandbox", config.grok_sandbox])
        cmd.append("--no-subagents")
        cmd.append("--no-plan")
    return cmd


def run_grok_cli(config: RuntimeConfig, prompt_text: str) -> tuple[str, dict[str, int] | None]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(prompt_text)
        prompt_file = handle.name
    try:
        raw = run_prompt_cli(
            cmd=grok_cli_command(config, prompt_file=prompt_file),
            prompt_text="",
            timeout=config.grok_timeout_seconds,
            executable=str(config.grok_executable),
            name="Grok CLI",
            install_hint=GROK_INSTALL_HINT,
            fallback_cmd=grok_cli_command(
                config, prompt_file=prompt_file, include_optional_flags=False
            ),
        )
        return parse_grok_output(raw)
    finally:
        Path(prompt_file).unlink(missing_ok=True)


def parse_grok_output(raw: str) -> tuple[str, dict[str, int] | None]:
    text = raw.strip()
    usage = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return raw.strip(), None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return raw.strip(), None
    if isinstance(payload, dict):
        answer = str(payload.get("text") or payload.get("result") or "").strip() or raw.strip()
        raw_usage = payload.get("usage")
        if isinstance(raw_usage, dict):
            prompt_tokens = int(raw_usage.get("input_tokens") or 0)
            completion_tokens = int(raw_usage.get("output_tokens") or 0)
            total = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            if prompt_tokens or completion_tokens or total:
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total": total,
                }
        return answer, usage
    return raw.strip(), None
