"""Tirzah MCP handlers — the memory-search seam for coding agents."""

from __future__ import annotations

from tirzah import mcp_handlers


def test_build_handlers_exposes_search_memory() -> None:
    handlers = mcp_handlers.build_handlers(db=object())
    assert "tirzah.search_memory" in handlers
    assert "search_memory" in handlers  # short alias too


def test_search_memory_compacts_nodes(monkeypatch) -> None:
    monkeypatch.setattr(
        "tirzah.retrieval.queries.search_nodes",
        lambda _db, query=None, limit=10: [
            {"node_id": "n1", "labels": ["substrate"], "content": "x" * 500, "document_id": "d1"},
            {"_id": "n2", "text": "short note"},
        ],
    )
    handler = mcp_handlers.build_handlers(db=object())["search_memory"]
    out = handler(query="substrate", limit=5)
    assert out["count"] == 2 and out["query"] == "substrate"
    first = out["results"][0]
    assert first["node_id"] == "n1" and first["labels"] == ["substrate"]
    assert first["text"].endswith("…") and len(first["text"]) == 401  # truncated
    assert out["results"][1]["node_id"] == "n2"


def test_search_memory_uses_serialized_preview_and_title(monkeypatch) -> None:
    monkeypatch.setattr(
        "tirzah.retrieval.queries.search_nodes",
        lambda _db, query=None, limit=10: [
            {"node_id": "n1", "title": "Decision record", "text_preview": "Use Keturah for MCP."},
            {"node_id": "n2", "title": "Title fallback"},
        ],
    )
    handler = mcp_handlers.build_handlers(db=object())["search_memory"]
    out = handler(query="keturah", limit=2)
    assert out["results"][0]["title"] == "Decision record"
    assert out["results"][0]["text"] == "Use Keturah for MCP."
    assert out["results"][1]["text"] == "Title fallback"


def test_search_memory_requires_query() -> None:
    handler = mcp_handlers.build_handlers(db=object())["search_memory"]
    assert "error" in handler(query="   ")


def test_search_memory_reports_missing_db(monkeypatch) -> None:
    # No reachable default DB → clean error, no raise (forced for determinism;
    # this env may otherwise have a live Mongo).
    monkeypatch.setattr(mcp_handlers, "_default_db", lambda: None)
    handler = mcp_handlers.build_handlers()["search_memory"]
    out = handler(query="anything")
    assert "error" in out and "unavailable" in out["error"]


def test_search_memory_surfaces_search_error(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("mongo down")

    monkeypatch.setattr("tirzah.retrieval.queries.search_nodes", boom)
    handler = mcp_handlers.build_handlers(db=object())["search_memory"]
    out = handler(query="x")
    assert "error" in out and "search failed" in out["error"]


def test_coherence_check_requires_input() -> None:
    handler = mcp_handlers.build_handlers(client=object())["coherence_check"]
    assert "error" in handler(query="   ", context="")
    # exposed under the manifest-matching namespaced name too
    assert "tirzah.coherence_check" in mcp_handlers.build_handlers(client=object())


def test_coherence_check_compacts_milcah_verdict() -> None:
    class FakeResult:
        objections = ["contradicts the retrieval contract"]
        claims = ["the diff changes the executor loop"]
        evidence = ["executor.py:512"]
        confidence = 0.42
        terminal_reason = "converged"

    class FakeClient:
        def __init__(self):
            self.seen = None

        def run(self, req):
            self.seen = req
            return FakeResult()

    fake = FakeClient()
    handler = mcp_handlers.build_handlers(client=fake)["coherence_check"]
    out = handler(query="check the QUEUE change", context="a diff")
    assert out["objections"] == ["contradicts the retrieval contract"]
    assert out["confidence"] == 0.42 and out["terminal_reason"] == "converged"
    assert fake.seen.mode == "coherence" and "a diff" in fake.seen.context


def test_coherence_check_reports_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(mcp_handlers, "_default_config", lambda: None)
    handler = mcp_handlers.build_handlers()["coherence_check"]
    assert "unavailable" in handler(query="x")["error"]


def test_read_memory_returns_verbatim_paginated_source_and_provenance(monkeypatch) -> None:
    from bson import ObjectId

    node_id = ObjectId("507f1f77bcf86cd799439011")
    document_id = ObjectId("507f1f77bcf86cd799439012")
    original = "An original source passage with enough text to require multiple pages."

    class FakeStore:
        def get_node(self, requested):
            assert requested == node_id
            return {
                "_id": node_id, "document_id": document_id, "text": original,
                "title": "Source section", "status": "active", "labels": ["source_chunk"],
                "endorsement_label": "explicit_endorsed", "origin_date": "2026-10-20",
                "provenance": {"source_path": "notes/source.md"},
            }

        def get_document(self, requested):
            assert requested == document_id
            return {"title": "Original notes"}

    monkeypatch.setattr("tirzah.db.memory_store.as_memory_store", lambda _db: FakeStore())
    handler = mcp_handlers.build_handlers(db=object())["tirzah.read_memory"]
    first = handler(node_id=str(node_id), max_chars=15)
    assert first["text"] == original[:15]
    assert first["offset"] == 0 and first["next_offset"] == 15
    assert first["total_chars"] == len(original) and first["truncated"] is True
    assert first["document_id"] == str(document_id)
    assert first["document_title"] == "Original notes"
    assert first["provenance"] == {"source_path": "notes/source.md"}
    assert first["endorsement_label"] == "explicit_endorsed"
    assert first["origin_date"] == "2026-10-20"

    chunks = [first["text"]]
    next_offset = first["next_offset"]
    while next_offset is not None:
        page = handler(node_id=str(node_id), offset=next_offset, max_chars=15)
        chunks.append(page["text"])
        next_offset = page["next_offset"]
    assert "".join(chunks) == original


def test_read_memory_binds_output_size_and_rejects_invalid_ids(monkeypatch) -> None:
    from bson import ObjectId

    node_id = ObjectId("507f1f77bcf86cd799439011")
    source = "x" * 20000

    class FakeStore:
        def get_node(self, _node_id):
            return {"_id": node_id, "document_id": "doc", "text": source, "status": "active"}

        def get_document(self, _document_id):
            return None

    monkeypatch.setattr("tirzah.db.memory_store.as_memory_store", lambda _db: FakeStore())
    handler = mcp_handlers.build_handlers(db=object())["read_memory"]
    assert "error" in handler(node_id="not-an-id")
    assert "error" in handler(node_id=str(node_id), offset=-1)
    assert "error" in handler(node_id=str(node_id), offset="nope")
    result = handler(node_id=str(node_id), max_chars=1000000)
    assert len(result["text"]) == 16000 and result["next_offset"] == 16000
    assert result["truncated"] is True


def test_read_memory_does_not_expose_inactive_ingestions(monkeypatch) -> None:
    from bson import ObjectId

    node_id = ObjectId("507f1f77bcf86cd799439011")

    class FakeStore:
        def __init__(self):
            self.status = "pending_review"

        def get_node(self, _node_id):
            return {"_id": node_id, "document_id": "doc", "text": "Private draft", "status": self.status}

        def get_document(self, _document_id):
            raise AssertionError("inactive node must not be read further")

    store = FakeStore()
    monkeypatch.setattr("tirzah.db.memory_store.as_memory_store", lambda _db: store)
    handler = mcp_handlers.build_handlers(db=object())["read_memory"]
    for status in ("pending_review", "rejected", "superseded"):
        store.status = status
        result = handler(node_id=str(node_id))
        assert "error" in result and "text" not in result


def test_read_memory_reports_missing_node_and_database(monkeypatch) -> None:
    from bson import ObjectId

    node_id = str(ObjectId("507f1f77bcf86cd799439011"))
    class FakeStore:
        def get_node(self, _node_id):
            return None

    monkeypatch.setattr("tirzah.db.memory_store.as_memory_store", lambda _db: FakeStore())
    assert "error" in mcp_handlers.build_handlers(db=object())["read_memory"](node_id=node_id)
    monkeypatch.setattr(mcp_handlers, "_default_db", lambda: None)
    assert "unavailable" in mcp_handlers.build_handlers()["read_memory"](node_id=node_id)["error"]
