from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tirzah.config import ModelBudgetProfile, RetrievalConfig


def count_tokens(
    text: str,
    *,
    tokenizer: str = "approx",
    encoding: str | None = None,
    chars_per_token: float = 4.0,
) -> dict[str, Any]:
    if not text:
        return {
            "tokens": 0,
            "tokenizer": tokenizer,
            "tokenizer_encoding": encoding,
            "tokenizer_fallback": False,
        }
    if tokenizer == "tiktoken":
        counted = _tiktoken_count(text, encoding or "cl100k_base")
        if counted is not None:
            return {
                "tokens": counted,
                "tokenizer": "tiktoken",
                "tokenizer_encoding": encoding or "cl100k_base",
                "tokenizer_fallback": False,
            }
        tokenizer = "approx"
        fallback = True
    else:
        fallback = False
    ratio = chars_per_token if chars_per_token > 0 else 4.0
    tokens = max(1, int((len(text) + ratio - 1) // ratio))
    return {
        "tokens": tokens,
        "tokenizer": "approx",
        "tokenizer_encoding": encoding,
        "tokenizer_fallback": fallback or tokenizer == "approx",
    }


def _tiktoken_count(text: str, encoding_name: str) -> int | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        encoder = tiktoken.get_encoding(encoding_name)
        return len(encoder.encode(text))
    except Exception:
        return None


@dataclass(frozen=True)
class BudgetPlan:
    prompt_token_budget: int
    reserved_response_tokens: int
    context_char_budget: int
    tokenizer: str
    tokenizer_encoding: str | None
    chars_per_token: float
    profile_key: str | None
    model: str | None
    adapter: str | None

    def count(self, text: str) -> int:
        return int(
            count_tokens(
                text,
                tokenizer=self.tokenizer,
                encoding=self.tokenizer_encoding,
                chars_per_token=self.chars_per_token,
            )["tokens"]
        )

    def as_envelope_kwargs(self) -> dict[str, Any]:
        return {
            "token_budget": self.prompt_token_budget,
            "reserved_response_tokens": self.reserved_response_tokens,
            "token_counter": self.count,
            "tokenizer": self.tokenizer,
            "tokenizer_encoding": self.tokenizer_encoding,
            "chars_per_token": self.chars_per_token,
            "context_char_budget": self.context_char_budget,
            "profile_key": self.profile_key,
            "budget_model": self.model,
            "budget_adapter": self.adapter,
        }


def resolve_budget_plan(
    retrieval: RetrievalConfig,
    *,
    model: str | None = None,
    adapter: str | None = None,
) -> BudgetPlan:
    profiles = retrieval.model_profiles or {}
    profile: ModelBudgetProfile | None = None
    profile_key = None
    for key in (model, adapter, "default"):
        if key and key in profiles:
            profile = profiles[key]
            profile_key = key
            break
    data = profile.model_dump() if profile is not None else {}
    tokenizer = str(data.get("tokenizer") or retrieval.tokenizer)
    encoding = data.get("tokenizer_encoding") or retrieval.tokenizer_encoding
    chars_per_token = float(data.get("chars_per_token") or retrieval.chars_per_token)
    return BudgetPlan(
        prompt_token_budget=int(data.get("prompt_token_budget") or retrieval.prompt_token_budget),
        reserved_response_tokens=int(
            data.get("reserved_response_tokens") or retrieval.reserved_response_tokens
        ),
        context_char_budget=int(data.get("context_char_budget") or retrieval.context_char_budget),
        tokenizer=tokenizer,
        tokenizer_encoding=encoding,
        chars_per_token=chars_per_token,
        profile_key=profile_key,
        model=model,
        adapter=adapter,
    )


def envelope_budget_args(config: Any, *, model: str | None = None, adapter: str | None = None) -> dict[str, Any]:
    retrieval = getattr(config, "retrieval", None)
    runtime = getattr(config, "runtime", None)
    if retrieval is None:
        return {}
    plan = resolve_budget_plan(
        retrieval,
        model=model or getattr(runtime, "ollama_model", None),
        adapter=adapter or getattr(runtime, "answer_adapter", None),
    )
    return plan.as_envelope_kwargs()


def recount_tokens(text: str, budget: dict[str, Any] | None) -> int:
    meta = budget or {}
    return int(
        count_tokens(
            text,
            tokenizer=str(meta.get("tokenizer") or "approx"),
            encoding=meta.get("tokenizer_encoding"),
            chars_per_token=float(meta.get("chars_per_token") or 4.0),
        )["tokens"]
    )
