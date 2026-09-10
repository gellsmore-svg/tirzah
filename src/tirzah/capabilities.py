"""Adapter capability registry + resolved-runtime introspection.

The fresh-install pain (CBO ingested with mock embeddings, Mahalath resolution off
until config was fixed) was invisible: nothing reported what the *active* runtime
could actually do. This module makes the resolved truth explicit — which adapters
are selected, what they're capable of, and which preconditions they imply — so
`tirzah config-status` (and `memory-health`) can surface it, and a regression can
assert it.

Capability flags borrow the shape of a capability-aware routing table: each adapter
declares what it can do and what it needs, and `resolved_runtime` folds the selected
adapters + the Mahalath seam into one queryable snapshot with explicit warnings.
"""

from __future__ import annotations

from typing import Any

# Per-adapter capabilities. `uses_mock` = deterministic, non-semantic output;
# `requires_hoglah` = needs a running `hoglah run` worker to execute jobs.
_ADAPTER_CAPS: dict[str, dict[str, bool]] = {
    "mock": {"uses_live_model": False, "uses_mock": True, "can_answer": True, "can_embed": True, "requires_hoglah": False},
    "ollama_cli": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": True, "requires_hoglah": False, "requires_ollama_binary": True},
    "ollama_http": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": True, "requires_hoglah": False},
    "hoglah": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": True, "requires_hoglah": True},
    "kiro_cli": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": False, "requires_hoglah": False, "requires_kiro_cli": True},
    "claude_cli": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": False, "requires_hoglah": False, "requires_claude_cli": True},
    "codex_cli": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": False, "requires_hoglah": False, "requires_codex_cli": True},
    "google_cli": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": False, "requires_hoglah": False, "requires_google_cli": True},
    "gemini_cli": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": False, "requires_hoglah": False, "requires_google_cli": True},
    "grok_cli": {"uses_live_model": True, "uses_mock": False, "can_answer": True, "can_embed": False, "requires_hoglah": False, "requires_grok_cli": True},
    # Embedding-only adapters (can't answer).
    "local_command": {"uses_live_model": True, "uses_mock": False, "can_answer": False, "can_embed": True, "requires_hoglah": False},
    "ollama_powershell": {"uses_live_model": True, "uses_mock": False, "can_answer": False, "can_embed": True, "requires_hoglah": False, "requires_powershell": True},
}


def adapter_capabilities(name: str) -> dict[str, bool]:
    """Capabilities for one adapter name (unknown adapters flagged, not guessed)."""
    if name not in _ADAPTER_CAPS:
        return {"unknown": True}
    return dict(_ADAPTER_CAPS[name])


def resolved_runtime(runtime: Any) -> dict[str, Any]:
    """Fold the selected adapters + Mahalath seam into one queryable snapshot."""
    model, answer, embed = runtime.model_adapter, runtime.answer_adapter, runtime.embedding_adapter
    c_model = adapter_capabilities(model)
    c_answer = adapter_capabilities(answer)
    c_embed = adapter_capabilities(embed)

    uses_mock = bool(c_model.get("uses_mock") or c_answer.get("uses_mock") or c_embed.get("uses_mock"))
    requires_hoglah = any(c.get("requires_hoglah") for c in (c_model, c_answer, c_embed))
    mahalath_enabled = bool(getattr(runtime, "mahalath_enabled", False))
    embed_is_mock = bool(c_embed.get("uses_mock"))

    capabilities = {
        "can_answer": bool(c_answer.get("can_answer")),
        "can_embed": bool(c_embed.get("can_embed")),
        "can_ingest_semantically": not embed_is_mock,
        "can_resolve_mpl": mahalath_enabled,
        "uses_live_model": bool(c_answer.get("uses_live_model")),
        "uses_mock": uses_mock,
        "requires_hoglah": requires_hoglah,
        "requires_mahalath": mahalath_enabled,
    }

    warnings: list[str] = []
    if embed_is_mock:
        warnings.append("embedding_adapter=mock — embeddings are NOT semantic (deterministic, low-dim).")
    if not mahalath_enabled:
        warnings.append("mahalath_enabled=false — semantic precision (MPL resolution) is OFF.")
    if requires_hoglah:
        warnings.append("adapters route through Hoglah — a `hoglah run` worker must be running.")
    if c_model.get("unknown") or c_answer.get("unknown") or c_embed.get("unknown"):
        warnings.append("an unknown adapter is configured — check model/answer/embedding_adapter names.")

    return {
        "adapters": {"model": model, "answer": answer, "embedding": embed},
        "adapter_capabilities": {"model": c_model, "answer": c_answer, "embedding": c_embed},
        "available_answer_adapters": sorted(
            name for name, caps in _ADAPTER_CAPS.items() if caps.get("can_answer")
        ),
        "available_embedding_adapters": sorted(
            name for name, caps in _ADAPTER_CAPS.items() if caps.get("can_embed")
        ),
        "models": {
            "ollama_model": getattr(runtime, "ollama_model", None),
            "embedding_model": getattr(runtime, "embedding_model", None),
        },
        "embedding_dimensions": getattr(runtime, "embedding_dimensions", None),
        "retrieval_mode": getattr(runtime, "retrieval_mode", None),
        "mahalath": {
            "enabled": mahalath_enabled,
            "strict": getattr(runtime, "mahalath_strict", None),
            "mongo_uri": getattr(runtime, "mahalath_mongo_uri", None),
            "mongo_db": getattr(runtime, "mahalath_mongo_db", None),
        },
        "capabilities": capabilities,
        "warnings": warnings,
    }


def render_runtime_text(snapshot: dict[str, Any]) -> str:
    """Human-readable rendering for `config-status`."""
    a = snapshot["adapters"]
    caps = snapshot["capabilities"]
    lines = [
        "Resolved runtime:",
        f"  adapters: model={a['model']}  answer={a['answer']}  embedding={a['embedding']}",
        f"  models: ollama={snapshot['models']['ollama_model']}  embedding={snapshot['models']['embedding_model']}"
        f"  ({snapshot['embedding_dimensions']}-dim)",
        f"  mahalath: enabled={snapshot['mahalath']['enabled']}  strict={snapshot['mahalath']['strict']}"
        f"  db={snapshot['mahalath']['mongo_db']}",
        "  capabilities: " + ", ".join(f"{k}={v}" for k, v in caps.items()),
    ]
    for w in snapshot["warnings"]:
        lines.append(f"  ! {w}")
    return "\n".join(lines)
