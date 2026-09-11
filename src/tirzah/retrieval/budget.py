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
    """Count tokens. ``tokenizer`` is the effective tokenizer; ``tokenizer_fallback``
    is True only when the requested tokenizer was unavailable and approx was used."""
    requested = tokenizer
    if not text:
        return {
            "tokens": 0,
            "tokenizer": tokenizer,
            "tokenizer_requested": requested,
            "tokenizer_encoding": encoding,
            "tokenizer_fallback": False,
        }
    fallback = False
    if tokenizer == "tiktoken":
        counted = _tiktoken_count(text, encoding or "cl100k_base")
        if counted is not None:
            return {
                "tokens": counted,
                "tokenizer": "tiktoken",
                "tokenizer_requested": requested,
                "tokenizer_encoding": encoding or "cl100k_base",
                "tokenizer_fallback": False,
            }
        fallback = True
    ratio = chars_per_token if chars_per_token > 0 else 4.0
    tokens = max(1, int((len(text) + ratio - 1) // ratio))
    return {
        "tokens": tokens,
        "tokenizer": "approx",
        "tokenizer_requested": requested,
        "tokenizer_encoding": encoding,
        "tokenizer_fallback": fallback,
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

    def pick(field: str) -> Any:
        # Only an unset (None) profile field inherits the global value; `or`
        # would also swallow legitimate falsy values.
        value = data.get(field)
        return getattr(retrieval, field) if value is None else value

    return BudgetPlan(
        prompt_token_budget=int(pick("prompt_token_budget")),
        reserved_response_tokens=int(pick("reserved_response_tokens")),
        context_char_budget=int(pick("context_char_budget")),
        tokenizer=str(pick("tokenizer")),
        tokenizer_encoding=pick("tokenizer_encoding"),
        chars_per_token=float(pick("chars_per_token")),
        profile_key=profile_key,
        model=model,
        adapter=adapter,
    )


def request_budget_plan(
    config: Any,
    *,
    runtime: Any = None,
    model: str | None = None,
    adapter: str | None = None,
) -> BudgetPlan | None:
    """Resolve the plan for one request. ``runtime`` is the per-request runtime
    config (model/adapter overrides applied); it defaults to ``config.runtime``."""
    retrieval = getattr(config, "retrieval", None)
    if retrieval is None or not hasattr(retrieval, "prompt_token_budget"):
        return None
    runtime = runtime if runtime is not None else getattr(config, "runtime", None)
    return resolve_budget_plan(
        retrieval,
        model=model or getattr(runtime, "ollama_model", None),
        adapter=adapter or getattr(runtime, "answer_adapter", None),
    )


def envelope_budget_args(
    config: Any,
    *,
    runtime: Any = None,
    model: str | None = None,
    adapter: str | None = None,
) -> dict[str, Any]:
    plan = request_budget_plan(config, runtime=runtime, model=model, adapter=adapter)
    return plan.as_envelope_kwargs() if plan is not None else {}


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
