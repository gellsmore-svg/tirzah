"""Tirzah's MCP tool handlers — the *memory in* seam for coding agents.

Keturah ships a generic stdio MCP server (`keturah.mcp.run_stdio_server`); this
module supplies the concrete Tirzah handlers it dispatches to, so an agent
(Codex CLI, Claude Code, …) can query the family's project memory as an MCP
tool. See `Noa/docs/codex-integration-plan.md` (the "memory in" seam).

Kept dependency-light and defensive: importing this never requires a live DB,
and a handler call with no reachable Mongo returns a structured ``{"error": …}``
rather than raising into the agent.
"""

from __future__ import annotations

from typing import Any, Callable

# Lazily-bootstrapped default DB (the MCP server is a standalone process spawned
# by the agent; it connects to the configured Mongo on first use).
_DEFAULT_DB: Any = None
_DEFAULT_DB_TRIED = False


def _default_db() -> Any:
    global _DEFAULT_DB, _DEFAULT_DB_TRIED
    if _DEFAULT_DB_TRIED:
        return _DEFAULT_DB
    _DEFAULT_DB_TRIED = True
    try:
        from tirzah.config import load_config
        from tirzah.db.client import get_database

        _DEFAULT_DB = get_database(load_config().mongo)
    except Exception:  # noqa: BLE001 - no reachable DB → handlers report cleanly
        _DEFAULT_DB = None
    return _DEFAULT_DB


_DEFAULT_CONFIG: Any = None
_DEFAULT_CONFIG_TRIED = False


def _default_config() -> Any:
    global _DEFAULT_CONFIG, _DEFAULT_CONFIG_TRIED
    if _DEFAULT_CONFIG_TRIED:
        return _DEFAULT_CONFIG
    _DEFAULT_CONFIG_TRIED = True
    try:
        from tirzah.config import load_config

        _DEFAULT_CONFIG = load_config()
    except Exception:  # noqa: BLE001
        _DEFAULT_CONFIG = None
    return _DEFAULT_CONFIG


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    """A small, agent-friendly projection of a memory node."""
    text = (
        node.get("content")
        or node.get("text")
        or node.get("text_preview")
        or node.get("summary")
        or node.get("title")
        or ""
    )
    if isinstance(text, str) and len(text) > 400:
        text = text[:400] + "…"
    return {
        "node_id": str(node.get("node_id") or node.get("_id") or ""),
        "title": str(node.get("title") or ""),
        "labels": list(node.get("labels") or []),
        "text": text,
        "document_id": str(node.get("document_id") or ""),
    }


def build_handlers(*, db: Any = None, client: Any = None) -> dict[str, Callable[..., Any]]:
    """Return the Tirzah MCP handlers, keyed by tool name (namespaced + short).

    ``db`` / ``client`` may be injected (tests / an already-configured process);
    otherwise the default DB and Milcah client are bootstrapped lazily on first
    call.
    """

    def search_memory(query: str = "", limit: int = 10, **_kw: Any) -> dict[str, Any]:
        """Search the family's graph memory for nodes matching a query."""
        if not str(query).strip():
            return {"error": "query is required"}
        store = db if db is not None else _default_db()
        if store is None:
            return {"error": "tirzah memory is unavailable (no reachable database)"}
        try:
            from tirzah.retrieval.queries import search_nodes

            nodes = search_nodes(store, query=query, limit=max(1, min(int(limit), 50)))
        except Exception as exc:  # noqa: BLE001 - surface a clean error to the agent
            return {"error": f"search failed: {type(exc).__name__}: {exc}"}
        results = [_compact_node(n) for n in nodes]
        return {"query": query, "count": len(results), "results": results}


    def read_memory(node_id: str = "", offset: int = 0, max_chars: int = 6000, **_kw: Any) -> dict[str, Any]:
        """Read preserved source text for a search hit, with provenance and pagination.

        Search previews are discovery aids, not sufficient evidence. This handler
        reads the original stored node text without re-parsing or summarising it.
        """
        from tirzah.retrieval.queries import parse_object_id

        object_id = parse_object_id(node_id)
        if object_id is None:
            return {"error": "node_id must be a valid MongoDB ObjectId"}
        try:
            start = int(offset)
            page_size = max(1, min(int(max_chars), 16000))
        except (TypeError, ValueError, OverflowError):
            return {"error": "offset and max_chars must be integers"}
        if start < 0:
            return {"error": "offset must be non-negative"}

        db_handle = db if db is not None else _default_db()
        if db_handle is None:
            return {"error": "tirzah memory is unavailable (no reachable database)"}
        try:
            from tirzah.db.memory_store import as_memory_store
            from tirzah.models.ingestion import INACTIVE_RETRIEVAL_STATUSES

            store = as_memory_store(db_handle)
            node = store.get_node(object_id)
            # A search hit should never lead an agent into an unpromoted,
            # rejected or superseded ingestion tree.
            if not node or node.get("status") in INACTIVE_RETRIEVAL_STATUSES:
                return {"error": "node is not available for retrieval"}

            source_text = str(node.get("text") or "")
            start = min(start, len(source_text))
            end = min(start + page_size, len(source_text))
            document = store.get_document(node["document_id"])
            return {
                "node_id": str(node["_id"]),
                "document_id": str(node["document_id"]),
                "document_title": str(document.get("title") or "") if document else "",
                "title": str(node.get("title") or ""),
                "labels": list(node.get("labels") or []),
                "endorsement_label": node.get("endorsement_label"),
                "origin_date": node.get("origin_date"),
                "provenance": node.get("provenance") or {},
                "text": source_text[start:end],
                "offset": start,
                "next_offset": end if end < len(source_text) else None,
                "total_chars": len(source_text),
                "truncated": end < len(source_text),
            }
        except Exception as exc:  # noqa: BLE001 - keep MCP failures structured
            return {"error": f"read failed: {type(exc).__name__}: {exc}"}

    def coherence_check(
        query: str = "", context: str = "", mode: str = "coherence", **_kw: Any
    ) -> dict[str, Any]:
        """Pressure-test a claim/diff/design for coherence via Milcah.

        Matches the manifest's ``coherence_check`` tool so it is both discoverable
        (tools/list) and executable (tools/call).
        """
        if not str(query).strip() and not str(context).strip():
            return {"error": "a query or context to check is required"}
        milcah = client
        if milcah is None:
            config = _default_config()
            try:
                from tirzah.coherence import make_client

                milcah = make_client(getattr(config, "runtime", config)) if config else None
            except Exception:  # noqa: BLE001
                milcah = None
        if milcah is None:
            return {"error": "milcah coherence check is unavailable (not enabled/reachable)"}
        try:
            from tirzah.coherence import SpecialistRequest

            request = SpecialistRequest(
                query=str(query) or "Pressure-test the coherence of the supplied work.",
                mode=str(mode) or "coherence", context=str(context)[:6000], max_iterations=2,
            )
            result = milcah.run(request)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"coherence check failed: {type(exc).__name__}: {exc}"}
        get = (lambda k, d=None: result.get(k, d)) if isinstance(result, dict) else (
            lambda k, d=None: getattr(result, k, d)
        )
        return {
            "objections": list(get("objections", []) or []),
            "claims": list(get("claims", []) or []),
            "evidence": list(get("evidence", []) or []),
            "confidence": float(get("confidence", 0.0) or 0.0),
            "terminal_reason": str(get("terminal_reason", "") or ""),
        }

    def retrieve(query: str = "", limit: int = 10, **kw: Any) -> dict[str, Any]:
        """Deborah-facing retrieve — same search as search_memory, observe-shaped."""
        raw = search_memory(query=query, limit=limit, **kw)
        if "error" in raw:
            return raw
        try:
            from tirzah.deborah import nodes_to_observe

            nodes = [
                {
                    "node_id": r.get("node_id"),
                    "title": r.get("title"),
                    "labels": r.get("labels"),
                    "text": r.get("text"),
                    "document_id": r.get("document_id"),
                }
                for r in (raw.get("results") or [])
            ]
            product = nodes_to_observe(nodes, query=str(query))
            product["results"] = raw.get("results") or []  # keep MCP raw hits too
            return product
        except Exception as exc:  # noqa: BLE001
            return {"error": f"retrieve shape failed: {type(exc).__name__}: {exc}", **raw}

    return {
        "tirzah.search_memory": search_memory,
        "search_memory": search_memory,
        "tirzah.read_memory": read_memory,
        "read_memory": read_memory,
        "tirzah.retrieve": retrieve,
        "retrieve": retrieve,
        "tirzah.coherence_check": coherence_check,
        "coherence_check": coherence_check,
    }
