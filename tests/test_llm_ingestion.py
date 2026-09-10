from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tirzah.adapters.ingestion import ingestion_adapter
from tirzah.adapters.llm_ingestion import (
    LlmIngestionAdapter,
    extract_json_object,
    locate_verbatim,
    result_from_proposal,
)
from tirzah.adapters.mock import MockIngestionAdapter
from tirzah.config import RuntimeConfig
from tirzah.db.repositories import (
    commit_ingestion,
    list_proposed_ingestion_trees,
    promote_ingestion_tree,
    reject_ingestion_tree,
)
from tirzah.models.ingestion import (
    INGESTION_KIND_LLM_PROPOSED,
    TREE_STATUS_ACTIVE,
    TREE_STATUS_PENDING_REVIEW,
    TREE_STATUS_REJECTED,
)
from tirzah.retrieval.queries import search_nodes


SOURCE = """# Memory

## Vortons

A vorton is a closed loop of superconducting cosmic string.

## Charge

Its charge is an integer linking invariant.
"""


def test_ingestion_adapter_selects_llm() -> None:
    adapter = ingestion_adapter(SimpleNamespace(ingestion_adapter="llm"))
    assert isinstance(adapter, LlmIngestionAdapter)


def test_ingestion_adapter_still_defaults_to_mock() -> None:
    assert isinstance(ingestion_adapter(), MockIngestionAdapter)


def test_extract_json_object_from_fenced_text() -> None:
    payload = extract_json_object('```json\n{"title": "T", "sections": []}\n```')
    assert payload == {"title": "T", "sections": []}


def test_locate_verbatim_accepts_whitespace_normalized_excerpt() -> None:
    located = locate_verbatim(SOURCE, "A vorton is a closed loop of\nsuperconducting cosmic string.")
    assert located is not None
    assert "vorton" in located


def test_result_from_proposal_requires_verbatim_chunks(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    with pytest.raises(ValueError, match="no verbatim"):
        result_from_proposal(
            path=path,
            text=SOURCE,
            source_kind="markdown",
            payload={
                "title": "Memory",
                "summary": "notes",
                "sections": [
                    {
                        "title": "Invented",
                        "chunks": [{"title": "nope", "text": "this text is not in the source"}],
                    }
                ],
            },
            extra_labels=["ams"],
            model_adapter="mock",
            model="mock",
        )


def test_result_from_proposal_builds_pending_review_tree(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    result = result_from_proposal(
        path=path,
        text=SOURCE,
        source_kind="markdown",
        payload={
            "title": "Memory",
            "summary": "Vorton notes",
            "sections": [
                {
                    "title": "Vortons",
                    "chunks": [
                        {
                            "title": "definition",
                            "text": "A vorton is a closed loop of superconducting cosmic string.",
                        }
                    ],
                }
            ],
        },
        extra_labels=["ams"],
        model_adapter="ollama_http",
        model="gemma3:1b",
    )
    assert result.adapter == "llm"
    assert result.ingestion_kind == INGESTION_KIND_LLM_PROPOSED
    assert result.tree_status == TREE_STATUS_PENDING_REVIEW
    labels = {tuple(node.labels) for node in result.nodes}
    assert ("source_root", "ams") in labels or any("source_root" in node.labels and "ams" in node.labels for node in result.nodes)
    chunks = [node for node in result.nodes if "source_chunk" in node.labels]
    assert chunks[0].text in SOURCE


def test_llm_adapter_uses_local_mock_generator_and_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    path.write_text(SOURCE, encoding="utf-8")
    result = LlmIngestionAdapter(
        RuntimeConfig(
            ingestion_adapter="llm",
            ingestion_model_adapter="mock",
            ingestion_fallback_to_mock=True,
        )
    ).process(path, SOURCE, "markdown")
    assert result.adapter == "llm"
    assert result.tree_status == TREE_STATUS_ACTIVE
    assert result.nodes[0].metadata["fallback"] == "mock"


def test_llm_adapter_falls_back_to_mock_on_model_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "source.md"
    path.write_text(SOURCE, encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr("tirzah.adapters.llm_ingestion.propose_ingestion_tree", boom)
    result = LlmIngestionAdapter(RuntimeConfig(ingestion_fallback_to_mock=True)).process(
        path, SOURCE, "markdown"
    )
    assert result.tree_status == TREE_STATUS_ACTIVE
    assert result.nodes[0].metadata["fallback"] == "mock"


def test_llm_adapter_raises_when_fallback_disabled(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "source.md"

    def boom(*_args, **_kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr("tirzah.adapters.llm_ingestion.propose_ingestion_tree", boom)
    with pytest.raises(RuntimeError, match="model down"):
        LlmIngestionAdapter(RuntimeConfig(ingestion_fallback_to_mock=False)).process(
            path, SOURCE, "markdown"
        )


def test_commit_and_promote_llm_tree_controls_retrieval() -> None:
    from tests.test_repositories import FakeDb, FakeEmbedder

    path = Path("source.md")
    result = result_from_proposal(
        path=path,
        text=SOURCE,
        source_kind="markdown",
        payload={
            "title": "Memory",
            "summary": "Vorton notes",
            "sections": [
                {
                    "title": "Vortons",
                    "chunks": [
                        {
                            "title": "definition",
                            "text": "A vorton is a closed loop of superconducting cosmic string.",
                        }
                    ],
                }
            ],
        },
        extra_labels=[],
        model_adapter="mock",
        model="mock",
    )
    result.source.checksum_sha256 = "checksum"
    result.created_at = datetime.now(timezone.utc)
    db = FakeDb()
    inserted = commit_ingestion(db, result, embedder=FakeEmbedder())
    assert inserted["tree_status"] == TREE_STATUS_PENDING_REVIEW
    assert db.trees.rows[0]["status"] == TREE_STATUS_PENDING_REVIEW
    assert db.nodes.rows[0]["status"] == TREE_STATUS_PENDING_REVIEW
    assert search_nodes(db) == []
    listed = list_proposed_ingestion_trees(db)
    assert listed[0]["tree_id"] == inserted["tree_id"]
    promoted = promote_ingestion_tree(db, inserted["tree_id"], reviewer="tester")
    assert promoted["ok"] is True
    assert promoted["status"] == TREE_STATUS_ACTIVE
    found = search_nodes(db)
    assert found
    assert all(row["status"] == TREE_STATUS_ACTIVE for row in found)
    assert any("vorton" in (row.get("text_preview") or "") for row in found)


def test_reject_ingestion_tree_keeps_nodes_out_of_retrieval() -> None:
    from tests.test_repositories import FakeDb, FakeEmbedder

    path = Path("source.md")
    result = result_from_proposal(
        path=path,
        text=SOURCE,
        source_kind="markdown",
        payload={
            "title": "Memory",
            "summary": "Vorton notes",
            "sections": [
                {
                    "title": "Vortons",
                    "chunks": [
                        {
                            "title": "definition",
                            "text": "A vorton is a closed loop of superconducting cosmic string.",
                        }
                    ],
                }
            ],
        },
        extra_labels=[],
        model_adapter="mock",
        model="mock",
    )
    result.source.checksum_sha256 = "checksum-2"
    result.created_at = datetime.now(timezone.utc)
    db = FakeDb()
    inserted = commit_ingestion(db, result, embedder=FakeEmbedder())
    rejected = reject_ingestion_tree(db, inserted["document_id"], reviewer="tester", note="bad chunks")
    assert rejected["ok"] is True
    assert rejected["status"] == TREE_STATUS_REJECTED
    assert search_nodes(db) == []
