from datetime import datetime, timezone

from bson import ObjectId

from tirzah.db.repositories import (
    backfill_node_embeddings,
    backfill_structural_graph_edges,
    bounded_graph_group_limit,
    commit_ingestion,
    create_reviewed_semantic_edge,
    document_tree,
    enqueue_contradiction_candidate_batch,
    enqueue_contradiction_candidates,
    enqueue_semantic_edge_candidates,
    enqueue_vector_semantic_edge_candidate_batch,
    enqueue_vector_semantic_edge_candidates,
    graph_edge_status,
    list_semantic_edge_candidates,
    rebuild_document,
    review_semantic_edge_candidate,
    resolved_ingestion_epoch,
    semantic_edge_candidate_exists,
    semantic_edge_candidate_pair_key,
    summarize_node_text,
    update_document_origin_date,
    update_node_summary,
)
from tirzah.models.ingestion import IngestedNode, IngestionResult, SourceRef


class FakeEmbedder:
    name = "fake_embedding"
    model = "fake-model"
    dimensions = 2

    def embed(self, text):
        return {
            "adapter": self.name,
            "model": self.model,
            "dimensions": self.dimensions,
            "vector": [1.0, 0.0],
            "source_text_hash": f"fake:{text}",
        }


class FailingEmbedder:
    name = "failing_embedding"
    model = "failing-model"
    dimensions = None

    def embed(self, _text):
        raise RuntimeError("embedding failed")


class CloseTrackingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_commit_ingestion_rolls_back_partial_insert_on_node_failure() -> None:
    db = FakeDb(fail_nodes=True)
    result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[
            IngestedNode(node_key="root", title="Root", text="Root text"),
        ],
        created_at=datetime.now(timezone.utc),
    )

    try:
        commit_ingestion(db, result)
    except RuntimeError as error:
        assert "node insert failed" in str(error)
    else:
        raise AssertionError("Expected commit_ingestion to propagate node insert failure.")

    assert db.documents.rows == []
    assert db.trees.rows == []
    assert db.nodes.rows == []


def test_resolved_ingestion_epoch_uses_explicit_or_date_default() -> None:
    timestamp = datetime(2026, 5, 30, 12, tzinfo=timezone.utc)
    default_result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[IngestedNode(node_key="root", title="Root", text="Root text")],
        created_at=timestamp,
    )
    explicit_result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[IngestedNode(node_key="root", title="Root", text="Root text")],
        ingestion_epoch="2026-05-30-rs5-rebuild-001",
        created_at=timestamp,
    )

    assert resolved_ingestion_epoch(default_result) == "2026-05-30-default"
    assert resolved_ingestion_epoch(explicit_result) == "2026-05-30-rs5-rebuild-001"


def test_commit_ingestion_stamps_epoch_on_document_tree_nodes_and_provenance() -> None:
    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[IngestedNode(node_key="root", title="Root", text="Root text")],
        ingestion_epoch="2026-05-30-test-001",
        created_at=datetime(2026, 5, 30, 12, tzinfo=timezone.utc),
    )

    inserted = commit_ingestion(db, result)

    assert inserted["ingestion_epoch"] == "2026-05-30-test-001"
    assert db.documents.rows[0]["ingestion_epoch"] == "2026-05-30-test-001"
    assert db.documents.rows[0]["source"]["checksum_sha256"] == "checksum"
    assert db.trees.rows[0]["ingestion_epoch"] == "2026-05-30-test-001"
    assert db.trees.rows[0]["status"] == "active"
    assert db.nodes.rows[0]["ingestion_epoch"] == "2026-05-30-test-001"
    assert db.nodes.rows[0]["status"] == "active"
    assert db.nodes.rows[0]["provenance"]["ingestion_epoch"] == "2026-05-30-test-001"


def test_commit_ingestion_persists_source_origin_date_metadata() -> None:
    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(
            path="source.md",
            kind="markdown",
            checksum_sha256="checksum",
            origin_date="2020-01-01",
            origin_date_source="explicit_content",
            date_candidates=[
                {
                    "source": "explicit_content",
                    "date": "2020-01-01",
                    "raw": "2020",
                    "rationale": "Explicit date marker found in document content.",
                }
            ],
        ),
        title="Source",
        summary="Summary",
        nodes=[IngestedNode(node_key="root", title="Root", text="Root text")],
        created_at=datetime(2026, 5, 30, 12, tzinfo=timezone.utc),
    )

    commit_ingestion(db, result)

    source = db.documents.rows[0]["source"]
    assert source["origin_date"] == "2020-01-01"
    assert source["origin_date_source"] == "explicit_content"
    assert source["date_candidates"][0]["source"] == "explicit_content"
    assert db.nodes.rows[0]["origin_date"] == "2020-01-01"
    assert db.nodes.rows[0]["origin_date_source"] == "explicit_content"


def test_update_document_origin_date_stamps_active_nodes_with_history() -> None:
    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(
            path="source.md",
            kind="markdown",
            checksum_sha256="checksum",
            origin_date="2020-01-01",
            origin_date_source="filename",
            origin_date_confidence=0.7,
        ),
        title="Source",
        summary="Summary",
        nodes=[IngestedNode(node_key="root", title="Root", text="Root text")],
        created_at=datetime(2026, 5, 30, 12, tzinfo=timezone.utc),
    )
    inserted = commit_ingestion(db, result)
    updated = update_document_origin_date(
        db,
        inserted["document_id"],
        "3 February 2021",
        reviewer="tester",
        note="corrected from publication page",
    )
    assert updated["ok"] is True
    assert updated["origin_date"] == "2021-02-03"
    assert updated["origin_date_source"] == "operator"
    assert updated["origin_date_confidence"] == 1.0
    source = db.documents.rows[0]["source"]
    assert source["origin_date"] == "2021-02-03"
    assert source["origin_date_history"][0]["origin_date"] == "2020-01-01"
    assert db.nodes.rows[0]["origin_date"] == "2021-02-03"
    assert db.nodes.rows[0]["origin_date_source"] == "operator"


def test_update_node_summary_records_provenance_history() -> None:
    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[IngestedNode(node_key="root", title="Root", text="Root text", summary="old summary")],
        created_at=datetime(2026, 5, 30, 12, tzinfo=timezone.utc),
    )
    inserted = commit_ingestion(db, result)
    node_id = inserted["node_ids"][0]
    updated = update_node_summary(
        db, node_id, "A vorton is a closed loop.", reviewer="tester", note="hand summary"
    )
    assert updated["ok"] is True
    assert updated["summary"] == "A vorton is a closed loop."
    assert updated["summary_provenance"]["source"] == "operator"
    assert updated["summary_provenance"]["history"][0]["summary"] == "old summary"


def test_commit_ingestion_annotates_nodes_with_embedding_metadata() -> None:
    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[
            IngestedNode(node_key="root", title="Root", text="Root text"),
            IngestedNode(node_key="child", parent_key="root", title="Child", text="Child text"),
        ],
        created_at=datetime.now(timezone.utc),
    )

    inserted = commit_ingestion(db, result)

    assert inserted["embedded_node_count"] == 2
    assert inserted["embedding_adapter"] == "mock_embedding"
    assert inserted["embedding_dimensions"] == 16
    for row in db.nodes.rows:
        embedding = row["embedding"]
        assert embedding["adapter"] == "mock_embedding"
        assert embedding["model"] == "mock-deterministic-v1"
        assert embedding["dimensions"] == 16
        assert len(embedding["vector"]) == 16
        assert embedding["source_text_hash"].startswith("sha256:")
    # Embeddings are deterministic from the node text, not the node identity.
    root = next(row for row in db.nodes.rows if row["node_key"] == "root")
    child = next(row for row in db.nodes.rows if row["node_key"] == "child")
    assert root["embedding"]["vector"] != child["embedding"]["vector"]


def test_commit_ingestion_uses_supplied_embedder() -> None:
    from tirzah.adapters.embedding import MockEmbeddingAdapter

    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[IngestedNode(node_key="root", title="Root", text="Root text")],
        created_at=datetime.now(timezone.utc),
    )

    commit_ingestion(db, result, embedder=MockEmbeddingAdapter(dimensions=8))

    assert len(db.nodes.rows[0]["embedding"]["vector"]) == 8


def test_backfill_node_embeddings_updates_missing_embeddings_only() -> None:
    db = FakeDb()
    document_id = ObjectId()
    missing_id = ObjectId()
    existing_id = ObjectId()
    superseded_id = ObjectId()
    db.nodes.rows.extend(
        [
            {
                "_id": missing_id,
                "document_id": document_id,
                "title": "Missing",
                "text": "Missing embedding text",
                "labels": ["target"],
                "status": "active",
            },
            {
                "_id": existing_id,
                "document_id": document_id,
                "title": "Existing",
                "text": "Existing embedding text",
                "labels": ["target"],
                "status": "active",
                "embedding": {"adapter": "old", "vector": [1]},
            },
            {
                "_id": superseded_id,
                "document_id": document_id,
                "title": "Superseded",
                "text": "Superseded text",
                "labels": ["target"],
                "status": "superseded",
            },
        ]
    )

    result = backfill_node_embeddings(
        db,
        FakeEmbedder(),
        label="target",
        document_id=str(document_id),
        limit=10,
    )

    assert result["ok"] is True
    assert result["scanned_count"] == 1
    assert result["matched_count"] == 1
    assert result["updated_count"] == 1
    assert result["skipped_count"] == 0
    assert result["last_node_id"] == str(missing_id)
    assert result["activity_log"].startswith("Text Similarity Profile Backfill Activity Log")
    assert "1 node(s) given text similarity profiles" in result["activity_log"]
    assert "Interruption behavior" in result["activity_log"]
    assert result["adapter"] == "fake_embedding"
    assert result["model"] == "fake-model"
    assert result["dimensions"] == 2
    missing = next(row for row in db.nodes.rows if row["_id"] == missing_id)
    existing = next(row for row in db.nodes.rows if row["_id"] == existing_id)
    superseded = next(row for row in db.nodes.rows if row["_id"] == superseded_id)
    assert missing["embedding"]["adapter"] == "fake_embedding"
    assert existing["embedding"]["adapter"] == "old"
    assert "embedding" not in superseded


def test_backfill_node_embeddings_force_replaces_existing_embeddings() -> None:
    db = FakeDb()
    node_id = ObjectId()
    db.nodes.rows.append(
        {
            "_id": node_id,
            "document_id": ObjectId(),
            "title": "Existing",
            "text": "Existing embedding text",
            "labels": ["target"],
            "status": "active",
            "embedding": {"adapter": "old", "vector": [1]},
        }
    )

    result = backfill_node_embeddings(db, FakeEmbedder(), label="target", force=True)

    assert result["updated_count"] == 1
    assert db.nodes.rows[0]["embedding"]["adapter"] == "fake_embedding"


def test_backfill_node_embeddings_collects_node_errors() -> None:
    db = FakeDb()
    for title in ("Bad 1", "Bad 2", "Bad 3"):
        db.nodes.rows.append(
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "title": title,
                "text": "bad",
                "status": "active",
            }
        )

    result = backfill_node_embeddings(db, FailingEmbedder(), max_errors=2)

    assert result["ok"] is False
    assert result["reason"] == "all_embedding_updates_failed"
    assert "needs attention" in result["activity_log"]
    assert result["updated_count"] == 0
    assert result["skipped_count"] == 3
    assert result["error_count"] == 3
    assert result["error_sample_limit"] == 2
    assert [error["title"] for error in result["errors"]] == ["Bad 1", "Bad 2"]
    assert result["errors"][0]["error_type"] == "RuntimeError"


def test_backfill_node_embeddings_continues_after_node_id() -> None:
    db = FakeDb()
    first_id = ObjectId()
    second_id = ObjectId()
    db.nodes.rows.extend(
        [
            {"_id": first_id, "title": "First", "text": "first", "status": "active"},
            {"_id": second_id, "title": "Second", "text": "second", "status": "active"},
        ]
    )

    result = backfill_node_embeddings(db, FakeEmbedder(), after_node_id=str(first_id), limit=10)

    assert result["updated_count"] == 1
    assert result["last_node_id"] == str(second_id)
    assert "embedding" not in db.nodes.rows[0]
    assert db.nodes.rows[1]["embedding"]["adapter"] == "fake_embedding"


def test_backfill_node_embeddings_repairs_partial_embedding_without_vector() -> None:
    db = FakeDb()
    node_id = ObjectId()
    db.nodes.rows.append(
        {
            "_id": node_id,
            "title": "Partial",
            "text": "partial",
            "status": "active",
            "embedding": {"adapter": "old"},
        }
    )

    result = backfill_node_embeddings(db, FakeEmbedder())

    assert result["updated_count"] == 1
    assert db.nodes.rows[0]["embedding"]["vector"] == [1.0, 0.0]


def test_backfill_node_embeddings_closes_closeable_embedder() -> None:
    db = FakeDb()
    db.nodes.rows.append(
        {
            "_id": ObjectId(),
            "title": "Closeable",
            "text": "close me",
            "status": "active",
        }
    )
    embedder = CloseTrackingEmbedder()

    result = backfill_node_embeddings(db, embedder)

    assert result["updated_count"] == 1
    assert embedder.closed is True


def test_rebuild_document_restores_previous_records_on_node_failure() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    node_id = ObjectId()
    db = FakeDb(fail_nodes=True)
    db.documents.rows.append(
        {
            "_id": document_id,
            "title": "Old",
            "summary": "Old summary",
            "source": {"path": "old.md", "kind": "markdown"},
        }
    )
    db.trees.rows.append({"_id": tree_id, "document_id": document_id, "label": "source"})
    db.nodes.rows.append(
        {
            "_id": node_id,
            "document_id": document_id,
            "tree_id": tree_id,
            "node_key": "root",
            "title": "Old root",
        }
    )
    result = IngestionResult(
        source=SourceRef(path="new.md", kind="markdown", checksum_sha256="new"),
        title="New",
        summary="New summary",
        nodes=[
            IngestedNode(node_key="root", title="New root", text="New text"),
        ],
        created_at=datetime.now(timezone.utc),
    )

    try:
        rebuild_document(db, str(document_id), result)
    except RuntimeError as error:
        assert "node insert failed" in str(error)
    else:
        raise AssertionError("Expected rebuild_document to propagate node insert failure.")

    assert db.documents.rows == [
        {
            "_id": document_id,
            "title": "Old",
            "summary": "Old summary",
            "source": {"path": "old.md", "kind": "markdown"},
        }
    ]
    assert db.trees.rows == [{"_id": tree_id, "document_id": document_id, "label": "source"}]
    assert db.nodes.rows == [
        {
            "_id": node_id,
            "document_id": document_id,
            "tree_id": tree_id,
            "node_key": "root",
            "title": "Old root",
        }
    ]


def test_rebuild_document_inserts_versioned_tree_and_supersedes_previous_records() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    node_id = ObjectId()
    db = FakeDb()
    db.documents.rows.append(
        {
            "_id": document_id,
            "title": "Old",
            "summary": "Old summary",
            "source": {"path": "old.md", "kind": "markdown"},
            "ingestion_epoch": "legacy",
        }
    )
    db.trees.rows.append(
        {
            "_id": tree_id,
            "document_id": document_id,
            "label": "source",
            "ingestion_epoch": "legacy",
        }
    )
    db.nodes.rows.append(
        {
            "_id": node_id,
            "document_id": document_id,
            "tree_id": tree_id,
            "node_key": "root",
            "title": "Old root",
            "ingestion_epoch": "legacy",
        }
    )
    result = IngestionResult(
        source=SourceRef(path="new.md", kind="markdown", checksum_sha256="new"),
        title="New",
        summary="New summary",
        nodes=[IngestedNode(node_key="root", title="New root", text="New text")],
        ingestion_epoch="2026-05-30-rebuild-001",
        created_at=datetime(2026, 5, 30, 12, tzinfo=timezone.utc),
    )

    inserted = rebuild_document(db, str(document_id), result)

    assert inserted["ingestion_epoch"] == "2026-05-30-rebuild-001"
    assert inserted["versioned"] is True
    assert inserted["superseded_tree_count"] == 1
    assert inserted["superseded_node_count"] == 1
    assert db.documents.rows[0]["ingestion_epoch"] == "2026-05-30-rebuild-001"
    assert len(db.trees.rows) == 2
    assert len(db.nodes.rows) == 2
    old_tree = next(row for row in db.trees.rows if row["_id"] == tree_id)
    old_node = next(row for row in db.nodes.rows if row["_id"] == node_id)
    new_tree = next(row for row in db.trees.rows if row["_id"] != tree_id)
    new_node = next(row for row in db.nodes.rows if row["_id"] != node_id)
    assert old_tree["status"] == "superseded"
    assert old_tree["superseded_by_epoch"] == "2026-05-30-rebuild-001"
    assert old_node["status"] == "superseded"
    assert old_node["superseded_by_epoch"] == "2026-05-30-rebuild-001"
    assert new_tree["ingestion_epoch"] == "2026-05-30-rebuild-001"
    assert new_tree["status"] == "active"
    assert new_node["ingestion_epoch"] == "2026-05-30-rebuild-001"
    assert new_node["status"] == "active"
    assert db.trees.update_many_calls == [
        (
            {"document_id": document_id},
            {"$set": {"status": "superseded", "superseded_by_epoch": "2026-05-30-rebuild-001"}},
        )
    ]
    assert db.nodes.update_many_calls == [
        (
            {"document_id": document_id},
            {"$set": {"status": "superseded", "superseded_by_epoch": "2026-05-30-rebuild-001"}},
        )
    ]


def test_targeted_rebuild_updates_changed_nodes_and_keeps_ids() -> None:
    db = FakeDb()
    original = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Memory",
        summary="Vorton notes",
        nodes=[
            IngestedNode(node_key="root", title="Memory", text="Vorton notes", labels=["source_root"]),
            IngestedNode(
                node_key="section-1",
                parent_key="root",
                title="Vortons",
                text="A vorton is a closed loop.",
                labels=["source_section"],
            ),
            IngestedNode(
                node_key="section-1-paragraph-1",
                parent_key="section-1",
                title="Vortons / paragraph 1",
                text="A vorton is a closed loop.",
                labels=["source_chunk"],
            ),
        ],
        created_at=datetime(2026, 5, 30, 12, tzinfo=timezone.utc),
    )
    inserted = commit_ingestion(db, original, embedder=FakeEmbedder())
    root_id = db.nodes.rows[0]["_id"]
    section_id = db.nodes.rows[1]["_id"]
    chunk_id = db.nodes.rows[2]["_id"]
    tree_id = db.trees.rows[0]["_id"]

    rebuilt = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum-2"),
        title="Memory",
        summary="Vorton notes",
        nodes=[
            IngestedNode(node_key="root", title="Memory", text="Vorton notes", labels=["source_root"]),
            IngestedNode(
                node_key="section-1",
                parent_key="root",
                title="Vortons",
                text="A vorton is a closed loop of superconducting cosmic string.",
                labels=["source_section"],
            ),
            IngestedNode(
                node_key="section-1-paragraph-1",
                parent_key="section-1",
                title="Vortons / paragraph 1",
                text="A vorton is a closed loop of superconducting cosmic string.",
                labels=["source_chunk"],
            ),
        ],
        ingestion_epoch="2026-09-10-diff",
        created_at=datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
    )
    result = rebuild_document(db, inserted["document_id"], rebuilt, embedder=FakeEmbedder(), mode="diff")

    assert result["applied"] is True
    assert result["preserved_node_count"] == 1
    assert result["updated_node_count"] == 2
    assert result["added_node_count"] == 0
    assert result["removed_node_count"] == 0
    assert result["embedded_node_count"] == 2
    assert len(db.trees.rows) == 1
    assert db.trees.rows[0]["_id"] == tree_id
    assert {row["_id"] for row in db.nodes.rows} == {root_id, section_id, chunk_id}
    chunk = next(row for row in db.nodes.rows if row["_id"] == chunk_id)
    assert chunk["text"].startswith("A vorton is a closed loop of superconducting")
    assert chunk["ingestion_epoch"] == "2026-09-10-diff"
    root = next(row for row in db.nodes.rows if row["_id"] == root_id)
    assert root["text"] == "Vorton notes"


def test_compare_rebuild_does_not_mutate_tree() -> None:
    db = FakeDb()
    original = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Memory",
        summary="notes",
        nodes=[IngestedNode(node_key="root", title="Memory", text="notes", labels=["source_root"])],
        created_at=datetime(2026, 5, 30, 12, tzinfo=timezone.utc),
    )
    inserted = commit_ingestion(db, original, embedder=FakeEmbedder())
    snapshot = [dict(row) for row in db.nodes.rows]
    proposed = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Memory",
        summary="notes changed",
        nodes=[IngestedNode(node_key="root", title="Memory", text="notes changed", labels=["source_root"])],
        created_at=datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
    )
    compared = rebuild_document(
        db, inserted["document_id"], proposed, embedder=FakeEmbedder(), compare_only=True
    )
    assert compared["compare_only"] is True
    assert compared["applied"] is False
    assert compared["diff"]["counts"]["changed"] == 1
    assert db.nodes.rows == snapshot


def test_document_tree_returns_only_active_nodes() -> None:
    document_id = ObjectId()
    active_id = ObjectId()
    superseded_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": superseded_id,
                "document_id": document_id,
                "tree_id": ObjectId(),
                "node_key": "old",
                "parent_id": None,
                "order": 0,
                "title": "Old root",
                "labels": ["source_root"],
                "status": "superseded",
            },
            {
                "_id": active_id,
                "document_id": document_id,
                "tree_id": ObjectId(),
                "node_key": "new",
                "parent_id": None,
                "order": 1,
                "title": "New root",
                "labels": ["source_root"],
                "status": "active",
            },
        ]
    )

    tree = document_tree(db, str(document_id))

    assert [node["node_id"] for node in tree] == [str(active_id)]


def test_commit_ingestion_persists_relation_edges() -> None:
    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[
            IngestedNode(node_key="root", title="Root", text="Root text"),
            IngestedNode(
                node_key="child",
                parent_key="root",
                title="Child",
                text="Child text",
                relations=[
                    {
                        "type": "supports",
                        "target_node_key": "root",
                        "weight": "0.75",
                        "confidence": 0.8,
                    },
                    {
                        "type": "supports",
                        "target_node_key": "root",
                    },
                    {
                        "type": "mentions",
                        "target_node_key": "missing",
                    },
                ],
            ),
        ],
        created_at=datetime.now(timezone.utc),
    )

    inserted = commit_ingestion(db, result)

    assert inserted["edge_count"] == 1
    assert inserted["skipped_edge_count"] == 2
    assert len(db.graph_edges.rows) == 1
    edge = db.graph_edges.rows[0]
    assert edge["relation_type"] == "supports"
    assert edge["source_node_key"] == "child"
    assert edge["target_node_key"] == "root"
    assert edge["weight"] == 0.75
    assert edge["confidence"] == 0.8
    assert edge["provenance"]["source"] == "ingestion_node_relation"
    assert edge["source_node_id"] != edge["target_node_id"]


def test_backfill_structural_graph_edges_creates_parent_child_edges() -> None:
    db = FakeDb()
    document_id = ObjectId()
    tree_id = ObjectId()
    parent_id = ObjectId()
    child_id = ObjectId()
    missing_parent_child_id = ObjectId()
    db.nodes.rows.extend(
        [
            {
                "_id": parent_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "node_key": "root",
                "title": "Root",
                "parent_id": None,
            },
            {
                "_id": child_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "node_key": "child",
                "title": "Child",
                "parent_id": parent_id,
            },
            {
                "_id": missing_parent_child_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "node_key": "orphan",
                "title": "Orphan",
                "parent_id": ObjectId(),
            },
        ]
    )

    result = backfill_structural_graph_edges(db)

    assert result == {
        "scanned_node_count": 2,
        "edge_count": 1,
        "skipped_existing_count": 0,
        "skipped_missing_parent_count": 1,
    }
    assert len(db.graph_edges.rows) == 1
    edge = db.graph_edges.rows[0]
    assert edge["source_node_id"] == parent_id
    assert edge["target_node_id"] == child_id
    assert edge["source_node_key"] == "root"
    assert edge["target_node_key"] == "child"
    assert edge["relation_type"] == "contains"
    assert edge["weight"] == 1.0
    assert edge["confidence"] == 1.0
    assert edge["provenance"]["source"] == "node_parent_link"

    second = backfill_structural_graph_edges(db)

    assert second["edge_count"] == 0
    assert second["skipped_existing_count"] == 1


def test_create_reviewed_semantic_edge_persists_user_reviewed_edge() -> None:
    db = FakeDb()
    source_id = ObjectId()
    target_id = ObjectId()
    source_document_id = ObjectId()
    target_document_id = ObjectId()
    db.nodes.rows.extend(
        [
            {
                "_id": source_id,
                "document_id": source_document_id,
                "tree_id": ObjectId(),
                "node_key": "source",
                "labels": ["source_chunk", "taj_mahal", "online_test"],
            },
            {
                "_id": target_id,
                "document_id": target_document_id,
                "tree_id": ObjectId(),
                "node_key": "target",
                "labels": ["source_chunk", "taj_mahal", "memory_reference"],
            },
        ]
    )

    result = create_reviewed_semantic_edge(
        db,
        source_node_id=str(source_id),
        target_node_id=str(target_id),
        relation_type="Related To",
        weight=2,
        confidence="bad",
        reviewer="cello",
        note="Shared Taj Mahal label.",
    )

    assert result["ok"] is True
    assert len(db.graph_edges.rows) == 1
    edge = db.graph_edges.rows[0]
    assert edge["source_node_id"] == source_id
    assert edge["target_node_id"] == target_id
    assert edge["source_document_id"] == source_document_id
    assert edge["target_document_id"] == target_document_id
    assert edge["relation_type"] == "related_to"
    assert edge["weight"] == 1.0
    assert edge["confidence"] == 0.6
    assert edge["provenance"]["source"] == "semantic_candidate_review"
    assert edge["provenance"]["reviewer"] == "cello"
    assert edge["provenance"]["shared_labels"] == ["taj_mahal"]

    duplicate = create_reviewed_semantic_edge(
        db,
        source_node_id=str(source_id),
        target_node_id=str(target_id),
        relation_type="related_to",
    )

    assert duplicate["ok"] is False
    assert duplicate["reason"] == "duplicate_edge"


def test_enqueue_semantic_edge_candidates_stores_pending_review_rows(monkeypatch) -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    document_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.append(
        {
            "_id": source_id,
            "document_id": document_id,
            "tree_id": ObjectId(),
            "node_key": "source",
            "title": "Source",
            "labels": ["taj_mahal"],
        }
    )

    monkeypatch.setattr(
        "tirzah.retrieval.queries.semantic_candidate_nodes",
        lambda _db, node_id, limit=10, include_same_document=False: [
            {
                "node_id": str(target_id),
                "document_id": str(ObjectId()),
                "node_key": "target",
                "title": "Target",
                "shared_labels": ["taj_mahal"],
                "shared_label_count": 1,
            }
        ],
    )

    result = enqueue_semantic_edge_candidates(
        db,
        node_id=str(source_id),
        relation_type="Related To",
        created_by="cello",
    )

    assert result["ok"] is True
    assert result["candidate_count"] == 1
    assert result["enqueued_count"] == 1
    assert len(db.semantic_edge_candidates.rows) == 1
    row = db.semantic_edge_candidates.rows[0]
    assert row["status"] == "pending"
    assert row["source_node_id"] == source_id
    assert row["target_node_id"] == target_id
    assert row["source_document_id"] == document_id
    assert row["relation_type"] == "related_to"
    assert row["pair_key"] == semantic_edge_candidate_pair_key(source_id, target_id, "related_to")
    assert row["shared_labels"] == ["taj_mahal"]
    assert row["created_by"] == "cello"

    duplicate = enqueue_semantic_edge_candidates(db, node_id=str(source_id))

    assert duplicate["enqueued_count"] == 0
    assert duplicate["skipped_existing_count"] == 1


def test_enqueue_vector_semantic_edge_candidates_stores_similarity_review_rows(monkeypatch) -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    document_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.append(
        {
            "_id": source_id,
            "document_id": document_id,
            "tree_id": ObjectId(),
            "node_key": "source",
            "title": "Source",
            "labels": ["taj_mahal"],
        }
    )

    monkeypatch.setattr(
        "tirzah.retrieval.queries.embedding_candidate_nodes",
        lambda _db, node_id, limit=10, include_same_document=False, min_similarity=0.75, **_kwargs: [
            {
                "node_id": str(target_id),
                "document_id": str(ObjectId()),
                "node_key": "target",
                "title": "Target",
                "embedding_similarity": 0.91,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
            }
        ],
    )

    result = enqueue_vector_semantic_edge_candidates(
        db,
        node_id=str(source_id),
        relation_type="Related To",
        created_by="cello",
        min_similarity=0.8,
    )

    assert result["ok"] is True
    assert result["candidate_source"] == "embedding_similarity"
    assert result["min_similarity"] == 0.8
    assert result["candidate_count"] == 1
    assert result["enqueued_count"] == 1
    assert len(db.semantic_edge_candidates.rows) == 1
    row = db.semantic_edge_candidates.rows[0]
    assert row["status"] == "pending"
    assert row["candidate_source"] == "embedding_similarity"
    assert row["source_node_id"] == source_id
    assert row["target_node_id"] == target_id
    assert row["source_document_id"] == document_id
    assert row["relation_type"] == "related_to"
    assert row["pair_key"] == semantic_edge_candidate_pair_key(source_id, target_id, "related_to")
    assert row["embedding_similarity"] == 0.91
    assert row["embedding_model"] == "mock"
    assert row["embedding_dimensions"] == 16
    assert row["selection_context"] == {
        "candidate_source": "embedding_similarity",
        "min_similarity": 0.8,
        "include_same_document": False,
        "requested_limit": 10,
        "embedding_model": "mock",
        "embedding_dimensions": 16,
    }
    assert row["created_by"] == "cello"

    duplicate = enqueue_vector_semantic_edge_candidates(db, node_id=str(source_id))

    assert duplicate["enqueued_count"] == 0
    assert duplicate["skipped_existing_count"] == 1


def test_enqueue_contradiction_candidates_stores_dates_and_provenance(monkeypatch) -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    document_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.append(
        {
            "_id": source_id,
            "document_id": document_id,
            "tree_id": ObjectId(),
            "node_key": "source",
            "title": "Source",
            "labels": ["ams_domain"],
            "origin_date": "2020-01-01",
            "origin_date_source": "filename",
            "origin_date_confidence": 0.9,
            "provenance": {"source_path": "archive/source.md", "adapter": "mock"},
        }
    )

    monkeypatch.setattr(
        "tirzah.retrieval.contradictions.contradiction_candidate_nodes",
        lambda _db, node_id, limit=10, include_same_document=False, min_similarity=0.82, max_similarity=0.97, **_kwargs: [
            {
                "node_id": str(target_id),
                "document_id": str(ObjectId()),
                "node_key": "target",
                "title": "Target",
                "shared_labels": ["ams_domain"],
                "shared_label_count": 1,
                "embedding_similarity": 0.9,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
                "candidate_source": "contradiction_signals",
                "contradiction_signals": {
                    "shared_semantic_labels": True,
                    "origin_date_delta": True,
                    "conflict_lexicon": True,
                    "cue_count": 3,
                    "meets_conservative_bar": True,
                    "shared_labels": ["ams_domain"],
                    "conflict_terms": ["contrary"],
                    "origin_date_delta_days": 1613,
                },
                "source_origin_date": "2020-01-01",
                "source_origin_date_source": "filename",
                "source_origin_date_confidence": 0.9,
                "target_origin_date": "2024-06-01",
                "target_origin_date_source": "header",
                "target_origin_date_confidence": 0.7,
                "origin_date_delta_days": 1613,
                "source_provenance": {"source_path": "archive/source.md", "adapter": "mock"},
                "target_provenance": {"source_path": "archive/target.md", "adapter": "mock"},
            }
        ],
    )

    result = enqueue_contradiction_candidates(
        db,
        node_id=str(source_id),
        created_by="cello",
        min_similarity=0.82,
    )

    assert result["ok"] is True
    assert result["candidate_source"] == "contradiction_signals"
    assert result["relation_type"] == "contradicts"
    assert result["enqueued_count"] == 1
    row = db.semantic_edge_candidates.rows[0]
    assert row["status"] == "pending"
    assert row["relation_type"] == "contradicts"
    assert row["pair_key"] == semantic_edge_candidate_pair_key(source_id, target_id, "contradicts")
    assert row["source_origin_date"] == "2020-01-01"
    assert row["target_origin_date"] == "2024-06-01"
    assert row["origin_date_delta_days"] == 1613
    assert row["source_provenance"]["source_path"] == "archive/source.md"
    assert row["target_provenance"]["source_path"] == "archive/target.md"
    assert row["contradiction_signals"]["cue_count"] == 3
    assert "negation" in row["selection_context"]["admission_rule"]

    duplicate = enqueue_contradiction_candidates(db, node_id=str(source_id))
    assert duplicate["enqueued_count"] == 0
    assert duplicate["skipped_existing_count"] == 1


def test_semantic_edge_candidate_exists_uses_pair_key_for_contradicts_reciprocals() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    db = FakeDb()
    db.semantic_edge_candidates.rows.append(
        {
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "contradicts",
            "pair_key": semantic_edge_candidate_pair_key(source_id, target_id, "contradicts"),
        }
    )

    assert semantic_edge_candidate_exists(db, target_id, source_id, "contradicts") is True
    assert semantic_edge_candidate_exists(db, source_id, target_id, "related_to") is False


def test_enqueue_contradiction_candidate_batch_dry_run_does_not_insert(monkeypatch) -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.append(
        {
            "_id": source_id,
            "node_key": "source-1",
            "title": "Source One",
            "labels": ["ams_domain"],
            "text": "Endorsement is required.",
            "embedding": {"model": "mock", "dimensions": 2},
        }
    )
    monkeypatch.setattr(
        "tirzah.retrieval.contradictions.contradiction_candidate_nodes",
        lambda _db, node_id, **_kwargs: [
            {
                "node_id": str(target_id),
                "document_id": str(ObjectId()),
                "node_key": "target",
                "title": "Target",
                "text_preview": "The later memo is contrary.",
                "embedding_similarity": 0.9,
                "candidate_source": "contradiction_signals",
                "contradiction_signals": {
                    "cue_count": 2,
                    "shared_semantic_labels": True,
                    "origin_date_delta": True,
                    "conflict_lexicon": False,
                },
                "source_origin_date": "2020-01-01",
                "target_origin_date": "2024-01-01",
                "origin_date_delta_days": 1461,
                "source_provenance": {"source_path": "archive/source.md"},
                "target_provenance": {"source_path": "archive/target.md"},
            }
        ],
    )

    result = enqueue_contradiction_candidate_batch(
        db,
        label="ams_domain",
        dry_run=True,
        candidates_per_node=1,
    )

    assert result["ok"] is True
    assert result["would_enqueue_count"] == 1
    assert result["enqueued_count"] == 0
    assert db.semantic_edge_candidates.rows == []
    preview = result["focus_results"][0]["candidate_previews"][0]
    assert preview["target_node_id"] == str(target_id)
    assert preview["origin_date_delta_days"] == 1461
    assert "possible contradiction" in preview["review_hint"]


def test_semantic_edge_candidate_exists_uses_pair_key_for_related_to_reciprocals() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    db = FakeDb()
    db.semantic_edge_candidates.rows.append(
        {
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "related_to",
            "pair_key": semantic_edge_candidate_pair_key(source_id, target_id, "related_to"),
        }
    )

    assert semantic_edge_candidate_exists(db, target_id, source_id, "related_to") is True


def test_semantic_edge_candidate_exists_falls_back_for_legacy_related_to_rows() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    db = FakeDb()
    db.semantic_edge_candidates.rows.append(
        {
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "related_to",
        }
    )

    assert semantic_edge_candidate_exists(db, target_id, source_id, "related_to") is True


def test_semantic_edge_candidate_exists_uses_pair_key_for_directional_relations() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    db = FakeDb()
    db.semantic_edge_candidates.rows.append(
        {
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "supports",
            "pair_key": semantic_edge_candidate_pair_key(source_id, target_id, "supports"),
        }
    )

    assert semantic_edge_candidate_exists(db, source_id, target_id, "supports") is True
    assert semantic_edge_candidate_exists(db, target_id, source_id, "supports") is False


def test_enqueue_vector_semantic_edge_candidate_batch_scopes_focus_nodes(monkeypatch) -> None:
    document_id = ObjectId()
    source_one = ObjectId()
    source_two = ObjectId()
    target_one = ObjectId()
    target_two = ObjectId()
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": source_one,
                "document_id": document_id,
                "node_key": "source-1",
                "title": "Source One",
                "labels": ["ams_domain"],
                "embedding": {"model": "mock", "dimensions": 16, "vector": [1.0]},
            },
            {
                "_id": source_two,
                "document_id": document_id,
                "node_key": "section-1",
                "title": "Source Two",
                "labels": ["ams_domain"],
                "embedding": {"model": "mock", "dimensions": 16, "vector": [1.0]},
            },
        ]
    )

    def fake_embedding_candidates(
        _db,
        node_id,
        limit=10,
        include_same_document=False,
        min_similarity=0.75,
        **_kwargs,
    ):
        target_id = target_one if node_id == str(source_one) else target_two
        return [
            {
                "node_id": str(target_id),
                "document_id": str(ObjectId()),
                "node_key": "target",
                "title": "Target",
                "embedding_similarity": min_similarity,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
            }
        ][:limit]

    monkeypatch.setattr(
        "tirzah.retrieval.queries.embedding_candidate_nodes",
        fake_embedding_candidates,
    )

    result = enqueue_vector_semantic_edge_candidate_batch(
        db,
        label="ams_domain",
        document_id=str(document_id),
        focus_limit=10,
        candidates_per_node=1,
        relation_type="supports",
        created_by="cello",
        min_similarity=0.82,
        exclude_node_keys=["section-1"],
    )

    assert result["ok"] is True
    assert result["candidate_source"] == "embedding_similarity"
    assert result["scope"]["focus_node_count"] == 1
    assert result["scope"]["candidates_per_node"] == 1
    assert result["scope"]["exclude_node_keys"] == ["section-1"]
    assert result["scope"]["relation_type"] == "supports"
    assert result["candidate_count"] == 1
    assert result["enqueued_count"] == 1
    assert len(db.semantic_edge_candidates.rows) == 1
    assert {row["source_node_id"] for row in db.semantic_edge_candidates.rows} == {
        source_one,
    }


def test_enqueue_vector_semantic_edge_candidate_batch_dry_run_does_not_insert(monkeypatch) -> None:
    document_id = ObjectId()
    source_id = ObjectId()
    target_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.append(
        {
            "_id": source_id,
            "document_id": document_id,
            "node_key": "source",
            "title": "Source",
            "text": "Source text for dry run review.",
            "labels": ["ams_domain"],
            "embedding": {"model": "mock", "dimensions": 16, "vector": [1.0]},
        }
    )

    monkeypatch.setattr(
        "tirzah.retrieval.queries.embedding_candidate_nodes",
        lambda _db, node_id, limit=10, include_same_document=False, min_similarity=0.75, **_kwargs: [
            {
                "node_id": str(target_id),
                "document_id": str(ObjectId()),
                "node_key": "target",
                "title": "Target",
                "text_preview": "Target text for dry run review.",
                "embedding_similarity": 0.91,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
            }
        ],
    )

    result = enqueue_vector_semantic_edge_candidate_batch(
        db,
        label="ams_domain",
        focus_limit=5,
        candidates_per_node=1,
        min_similarity=0.8,
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["scope"]["dry_run"] is True
    assert result["candidate_count"] == 1
    assert result["enqueued_count"] == 0
    assert result["would_enqueue_count"] == 1
    assert len(db.semantic_edge_candidates.rows) == 0
    assert result["focus_results"][0]["candidate_previews"][0]["target_title"] == "Target"
    assert (
        result["focus_results"][0]["candidate_previews"][0]["source_text_preview"]
        == "Source text for dry run review."
    )
    assert (
        result["focus_results"][0]["candidate_previews"][0]["target_text_preview"]
        == "Target text for dry run review."
    )
    assert result["focus_results"][0]["candidate_previews"][0]["shared_wording"] == {
        "shared_word_count": 5,
        "source_word_overlap": 0.833,
        "target_word_overlap": 0.833,
        "smaller_text_overlap": 0.833,
        "larger_text_overlap": 0.833,
    }
    assert (
        result["focus_results"][0]["candidate_previews"][0]["review_hint"]
        == "Review hint: high shared wording; check for copied or near-copied source text before accepting."
    )


def test_enqueue_vector_semantic_edge_candidate_batch_dry_run_skips_reciprocal_related_to(
    monkeypatch,
) -> None:
    document_one = ObjectId()
    document_two = ObjectId()
    source_one = ObjectId()
    source_two = ObjectId()
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": source_one,
                "document_id": document_one,
                "node_key": "source-1",
                "title": "Source One",
                "labels": ["ams_domain"],
                "embedding": {"model": "mock", "dimensions": 16, "vector": [1.0]},
            },
            {
                "_id": source_two,
                "document_id": document_two,
                "node_key": "source-2",
                "title": "Source Two",
                "labels": ["ams_domain"],
                "embedding": {"model": "mock", "dimensions": 16, "vector": [1.0]},
            },
        ]
    )

    def fake_embedding_candidates(
        _db,
        node_id,
        limit=10,
        include_same_document=False,
        min_similarity=0.75,
        **_kwargs,
    ):
        if node_id == str(source_one):
            target_id = source_two
            target_document = document_two
            target_title = "Source Two"
        else:
            target_id = source_one
            target_document = document_one
            target_title = "Source One"
        return [
            {
                "node_id": str(target_id),
                "document_id": str(target_document),
                "node_key": "target",
                "title": target_title,
                "embedding_similarity": 0.91,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
            }
        ][:limit]

    monkeypatch.setattr(
        "tirzah.retrieval.queries.embedding_candidate_nodes",
        fake_embedding_candidates,
    )

    result = enqueue_vector_semantic_edge_candidate_batch(
        db,
        label="ams_domain",
        focus_limit=2,
        candidates_per_node=1,
        relation_type="related_to",
        min_similarity=0.8,
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["candidate_count"] == 2
    assert result["would_enqueue_count"] == 1
    assert result["skipped_existing_count"] == 1
    assert result["focus_results"][0]["would_enqueue_count"] == 1
    assert result["focus_results"][1]["skipped_existing_count"] == 1
    assert len(db.semantic_edge_candidates.rows) == 0


def test_enqueue_vector_semantic_edge_candidate_batch_skips_reciprocal_related_to(
    monkeypatch,
) -> None:
    document_one = ObjectId()
    document_two = ObjectId()
    source_one = ObjectId()
    source_two = ObjectId()
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": source_one,
                "document_id": document_one,
                "node_key": "source-1",
                "title": "Source One",
                "labels": ["ams_domain"],
                "embedding": {"model": "mock", "dimensions": 16, "vector": [1.0]},
            },
            {
                "_id": source_two,
                "document_id": document_two,
                "node_key": "source-2",
                "title": "Source Two",
                "labels": ["ams_domain"],
                "embedding": {"model": "mock", "dimensions": 16, "vector": [1.0]},
            },
        ]
    )

    def fake_embedding_candidates(
        _db,
        node_id,
        limit=10,
        include_same_document=False,
        min_similarity=0.75,
        **_kwargs,
    ):
        if node_id == str(source_one):
            target_id = source_two
            target_document = document_two
            target_title = "Source Two"
        else:
            target_id = source_one
            target_document = document_one
            target_title = "Source One"
        return [
            {
                "node_id": str(target_id),
                "document_id": str(target_document),
                "node_key": "target",
                "title": target_title,
                "embedding_similarity": 0.91,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
            }
        ][:limit]

    monkeypatch.setattr(
        "tirzah.retrieval.queries.embedding_candidate_nodes",
        fake_embedding_candidates,
    )

    result = enqueue_vector_semantic_edge_candidate_batch(
        db,
        label="ams_domain",
        focus_limit=2,
        candidates_per_node=1,
        relation_type="related_to",
        min_similarity=0.8,
    )

    assert result["ok"] is True
    assert result["candidate_count"] == 2
    assert result["enqueued_count"] == 1
    assert result["skipped_existing_count"] == 1
    assert len(db.semantic_edge_candidates.rows) == 1
    assert db.semantic_edge_candidates.rows[0]["source_node_id"] == source_one
    assert db.semantic_edge_candidates.rows[0]["target_node_id"] == source_two


def test_list_semantic_edge_candidates_serializes_pending_rows() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    timestamp = datetime(2026, 5, 28, tzinfo=timezone.utc)
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": source_id,
                "title": "Source from node",
                "text": "Source text for candidate review. " * 20,
            },
            {
                "_id": target_id,
                "title": "Target from node",
                "text": "Target text for candidate review. " * 20,
            },
        ]
    )
    db.semantic_edge_candidates.rows.append(
        {
            "_id": ObjectId(),
            "status": "pending",
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "related_to",
            "candidate_source": "embedding_similarity",
            "shared_labels": ["memory_reference"],
            "shared_label_count": 1,
            "embedding_similarity": 0.88,
            "embedding_model": "mock",
            "embedding_dimensions": 16,
            "selection_context": {
                "candidate_source": "embedding_similarity",
                "min_similarity": 0.75,
            },
            "source_title": "Source",
            "target_title": "Target",
            "created_by": "user",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    )

    rows = list_semantic_edge_candidates(db)

    assert rows == [
        {
            "candidate_id": str(db.semantic_edge_candidates.rows[0]["_id"]),
            "status": "pending",
            "source_node_id": str(source_id),
            "target_node_id": str(target_id),
            "source_document_id": None,
            "target_document_id": None,
            "source_node_key": None,
            "target_node_key": None,
            "relation_type": "related_to",
            "candidate_source": "embedding_similarity",
            "shared_labels": ["memory_reference"],
            "shared_label_count": 1,
            "embedding_similarity": 0.88,
            "embedding_model": "mock",
            "embedding_dimensions": 16,
            "selection_context": {
                "candidate_source": "embedding_similarity",
                "min_similarity": 0.75,
            },
            "source_title": "Source",
            "target_title": "Target",
            "source_text_preview": summarize_node_text(
                "Source text for candidate review. " * 20,
                limit=280,
            ),
            "target_text_preview": summarize_node_text(
                "Target text for candidate review. " * 20,
                limit=280,
            ),
            "shared_wording": {
                "shared_word_count": 4,
                "source_word_overlap": 0.8,
                "target_word_overlap": 0.8,
                "smaller_text_overlap": 0.8,
                "larger_text_overlap": 0.8,
            },
            "review_hint": "Review hint: high shared wording; check for copied or near-copied source text before accepting.",
            "source_origin_date": None,
            "source_origin_date_source": None,
            "source_origin_date_confidence": None,
            "target_origin_date": None,
            "target_origin_date_source": None,
            "target_origin_date_confidence": None,
            "origin_date_delta_days": None,
            "source_provenance": {},
            "target_provenance": {},
            "contradiction_signals": {},
            "created_by": "user",
            "reviewer": None,
            "review_note": None,
            "edge_id": None,
            "created_at": "2026-05-28T00:00:00+00:00",
            "updated_at": "2026-05-28T00:00:00+00:00",
            "reviewed_at": None,
        }
    ]


def test_review_semantic_edge_candidate_accepts_and_creates_edge() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    candidate_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": source_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "node_key": "source",
                "title": "Source",
                "text": "Permission rules define who may alter repository records.",
                "labels": ["memory_reference"],
            },
            {
                "_id": target_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "node_key": "target",
                "title": "Target",
                "text": "Access controls describe who may change stored knowledge.",
                "labels": ["memory_reference"],
            },
        ]
    )
    db.semantic_edge_candidates.rows.append(
        {
            "_id": candidate_id,
            "status": "pending",
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "related_to",
            "candidate_source": "embedding_similarity",
            "embedding_similarity": 0.91,
            "embedding_model": "mock",
            "embedding_dimensions": 16,
            "selection_context": {
                "candidate_source": "embedding_similarity",
                "min_similarity": 0.8,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
            },
        }
    )

    result = review_semantic_edge_candidate(
        db,
        candidate_id=str(candidate_id),
        action="accept",
        reviewer="cello",
        note="Looks related.",
        weight=0.8,
        confidence=0.9,
    )

    assert result["ok"] is True
    assert len(db.graph_edges.rows) == 1
    assert result["candidate"]["status"] == "accepted"
    assert result["candidate"]["reviewer"] == "cello"
    assert result["candidate"]["review_note"] == "Looks related."
    assert result["candidate"]["edge_id"] == result["edge"]["edge_id"]
    assert result["candidate"]["source_title"] == "Source"
    assert result["candidate"]["target_title"] == "Target"
    assert result["candidate"]["source_text_preview"] == (
        "Permission rules define who may alter repository records."
    )
    assert result["candidate"]["target_text_preview"] == (
        "Access controls describe who may change stored knowledge."
    )
    assert result["candidate"]["review_hint"] == (
        "Review hint: strong profile match with moderate wording overlap; likely conceptual candidate."
    )
    assert result["edge"]["weight"] == 0.8
    assert result["edge"]["confidence"] == 0.9
    assert result["edge"]["provenance"]["candidate_source"] == "embedding_similarity"
    assert result["edge"]["provenance"]["embedding_similarity"] == 0.91
    assert result["edge"]["provenance"]["embedding_model"] == "mock"
    assert result["edge"]["provenance"]["embedding_dimensions"] == 16
    assert result["edge"]["provenance"]["selection_context"] == {
        "candidate_source": "embedding_similarity",
        "min_similarity": 0.8,
        "embedding_model": "mock",
        "embedding_dimensions": 16,
    }


def test_review_contradiction_candidate_accepts_contradicts_edge() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    candidate_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": source_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "node_key": "source",
                "title": "Source",
                "text": "Endorsement is required before retrieval.",
                "labels": ["ams_domain"],
                "origin_date": "2020-01-01",
                "provenance": {"source_path": "archive/source.md"},
            },
            {
                "_id": target_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "node_key": "target",
                "title": "Target",
                "text": "The later memo is contrary: endorsement is optional.",
                "labels": ["ams_domain"],
                "origin_date": "2024-06-01",
                "provenance": {"source_path": "archive/target.md"},
            },
        ]
    )
    db.semantic_edge_candidates.rows.append(
        {
            "_id": candidate_id,
            "status": "pending",
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "contradicts",
            "candidate_source": "contradiction_signals",
            "embedding_similarity": 0.9,
            "source_origin_date": "2020-01-01",
            "target_origin_date": "2024-06-01",
            "origin_date_delta_days": 1613,
            "source_provenance": {"source_path": "archive/source.md"},
            "target_provenance": {"source_path": "archive/target.md"},
            "contradiction_signals": {
                "shared_semantic_labels": True,
                "origin_date_delta": True,
                "conflict_lexicon": True,
                "cue_count": 3,
            },
        }
    )

    result = review_semantic_edge_candidate(
        db,
        candidate_id=str(candidate_id),
        action="accept",
        reviewer="cello",
        note="These claims conflict.",
    )

    assert result["ok"] is True
    assert result["edge"]["relation_type"] == "contradicts"
    assert result["candidate"]["status"] == "accepted"
    assert "possible contradiction" in result["candidate"]["review_hint"]
    assert result["candidate"]["source_origin_date"] == "2020-01-01"
    assert result["candidate"]["target_origin_date"] == "2024-06-01"
    assert result["edge"]["provenance"]["candidate_source"] == "contradiction_signals"
    assert result["edge"]["provenance"]["contradiction_signals"]["cue_count"] == 3
    assert result["edge"]["provenance"]["source_origin_date"] == "2020-01-01"
    assert result["edge"]["provenance"]["target_provenance"]["source_path"] == "archive/target.md"


def test_review_semantic_edge_candidate_rejects_pending_candidate() -> None:
    source_id = ObjectId()
    target_id = ObjectId()
    candidate_id = ObjectId()
    db = FakeDb()
    db.nodes.rows.extend(
        [
            {
                "_id": source_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "node_key": "source",
                "title": "Rejected Source",
                "text": "Identical template wording can make weak candidates look important.",
                "labels": ["memory_reference"],
            },
            {
                "_id": target_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "node_key": "target",
                "title": "Rejected Target",
                "text": "Identical template wording can make weak candidates look important.",
                "labels": ["memory_reference"],
            },
        ]
    )
    db.semantic_edge_candidates.rows.append(
        {
            "_id": candidate_id,
            "status": "pending",
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": "related_to",
        }
    )

    result = review_semantic_edge_candidate(
        db,
        candidate_id=str(candidate_id),
        action="reject",
        reviewer="cello",
        note="Too broad.",
    )

    assert result["ok"] is True
    assert result["candidate"]["status"] == "rejected"
    assert result["candidate"]["reviewer"] == "cello"
    assert result["candidate"]["review_note"] == "Too broad."
    assert result["candidate"]["source_title"] == "Rejected Source"
    assert result["candidate"]["target_title"] == "Rejected Target"
    assert result["candidate"]["source_text_preview"] == (
        "Identical template wording can make weak candidates look important."
    )
    assert result["candidate"]["target_text_preview"] == (
        "Identical template wording can make weak candidates look important."
    )
    assert result["candidate"]["review_hint"] == (
        "Review hint: high shared wording; check for copied or near-copied source text before accepting."
    )
    assert db.graph_edges.rows == []


def test_graph_edge_status_counts_relations_and_provenance_sources() -> None:
    db = FakeDb()
    db.graph_edges.rows.extend(
        [
            {
                "relation_type": "contains",
                "provenance": {"source": "node_parent_link"},
            },
            {
                "relation_type": "contains",
                "provenance": {"source": "node_parent_link"},
            },
            {
                "relation_type": "supports",
                "provenance": {"source": "ingestion_node_relation"},
            },
        ]
    )

    assert graph_edge_status(db) == {
        "edge_count": 3,
        "relation_types": [
            {"value": "contains", "count": 2},
            {"value": "supports", "count": 1},
        ],
        "provenance_sources": [
            {"value": "node_parent_link", "count": 2},
            {"value": "ingestion_node_relation", "count": 1},
        ],
    }


def test_graph_edge_status_handles_missing_graph_edges_collection() -> None:
    db = object()

    assert graph_edge_status(db) == {
        "edge_count": 0,
        "relation_types": [],
        "provenance_sources": [],
    }


def test_graph_edge_status_reports_null_buckets() -> None:
    db = FakeDb()
    db.graph_edges.rows.append({})

    assert graph_edge_status(db) == {
        "edge_count": 1,
        "relation_types": [{"value": None, "count": 1}],
        "provenance_sources": [{"value": None, "count": 1}],
    }


def test_graph_edge_status_applies_limit_to_group_buckets() -> None:
    db = FakeDb()
    db.graph_edges.rows.extend(
        [
            {"relation_type": "alpha", "provenance": {"source": "a"}},
            {"relation_type": "beta", "provenance": {"source": "b"}},
            {"relation_type": "gamma", "provenance": {"source": "c"}},
        ]
    )

    result = graph_edge_status(db, limit=2)

    assert result["relation_types"] == [
        {"value": "alpha", "count": 1},
        {"value": "beta", "count": 1},
    ]
    assert result["provenance_sources"] == [
        {"value": "a", "count": 1},
        {"value": "b", "count": 1},
    ]


def test_bounded_graph_group_limit_clamps_explicit_limits() -> None:
    assert bounded_graph_group_limit(0) == 1
    assert bounded_graph_group_limit(-5) == 1
    assert bounded_graph_group_limit(999) == 50
    assert bounded_graph_group_limit("bad") == 10


def test_commit_ingestion_rolls_back_edges_on_insert_failure() -> None:
    db = FakeDb(fail_edges=True)
    result = IngestionResult(
        source=SourceRef(path="source.md", kind="markdown", checksum_sha256="checksum"),
        title="Source",
        summary="Summary",
        nodes=[
            IngestedNode(node_key="root", title="Root", text="Root text"),
            IngestedNode(
                node_key="child",
                title="Child",
                text="Child text",
                relations=[{"type": "supports", "target_node_key": "root"}],
            ),
        ],
        created_at=datetime.now(timezone.utc),
    )

    try:
        commit_ingestion(db, result)
    except RuntimeError as error:
        assert "edge insert failed" in str(error)
    else:
        raise AssertionError("Expected commit_ingestion to propagate edge insert failure.")

    assert db.documents.rows == []
    assert db.trees.rows == []
    assert db.nodes.rows == []
    assert db.graph_edges.rows == []


class FakeInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class FakeCursor(list):
    def sort(self, field, direction=1):
        super().sort(key=lambda row: mongo_sort_value(nested_get(row, field)), reverse=direction < 0)
        return self

    def limit(self, value):
        return FakeCursor(self[:value])


class FakeCollection:
    def __init__(self, fail_insert=False, fail_insert_many=False):
        self.rows = []
        self.fail_insert = fail_insert
        self.fail_insert_many = fail_insert_many
        self.update_many_calls = []

    def find_one(self, query):
        row = next((row for row in self.rows if matches(row, query)), None)
        return dict(row) if row else None

    def find(self, query, _projection=None):
        return FakeCursor([dict(row) for row in self.rows if matches(row, query)])

    def count_documents(self, query):
        return len([row for row in self.rows if matches(row, query)])

    def aggregate(self, pipeline):
        group_field = pipeline[0]["$group"]["_id"].removeprefix("$")
        sort_spec = pipeline[1].get("$sort", {})
        counts = {}
        for row in self.rows:
            value = nested_get(row, group_field)
            counts[value] = counts.get(value, 0) + 1
        rows = [{"_id": key, "count": value} for key, value in counts.items()]
        for field, direction in reversed(list(sort_spec.items())):
            rows.sort(
                key=lambda item, sort_field=field: mongo_sort_value(item.get(sort_field)),
                reverse=direction < 0,
            )
        limit = pipeline[-1].get("$limit", len(rows))
        return rows[:limit]

    def insert_one(self, row):
        if self.fail_insert:
            raise RuntimeError("node insert failed")
        row = dict(row)
        row["_id"] = ObjectId()
        self.rows.append(row)
        return FakeInsertResult(row["_id"])

    def insert_many(self, rows):
        if self.fail_insert_many:
            raise RuntimeError("edge insert failed")
        for row in rows:
            row = dict(row)
            row.setdefault("_id", ObjectId())
            self.rows.append(row)
        return None

    def replace_one(self, query, replacement, upsert=False):
        for index, row in enumerate(self.rows):
            if matches(row, query):
                self.rows[index] = dict(replacement)
                break
        else:
            if upsert:
                self.rows.append(dict(replacement))
        return None

    def update_one(self, query, update):
        row = next((item for item in self.rows if matches(item, query)), None)
        if row:
            row.update(update.get("$set", {}))
        return None

    def update_many(self, query, update):
        self.update_many_calls.append((dict(query), scrub_bulk_update(update)))
        for row in self.rows:
            if matches(row, query):
                row.update(update.get("$set", {}))
        return None

    def delete_many(self, query):
        self.rows = [row for row in self.rows if not matches(row, query)]
        return None

    def delete_one(self, query):
        for index, row in enumerate(self.rows):
            if matches(row, query):
                del self.rows[index]
                break
        return None


class FakeDb:
    def __init__(self, fail_nodes=False, fail_edges=False):
        self.documents = FakeCollection()
        self.trees = FakeCollection()
        self.nodes = FakeCollection(fail_insert=fail_nodes)
        self.graph_edges = FakeCollection(fail_insert_many=fail_edges)
        self.semantic_edge_candidates = FakeCollection()


def matches(row, query):
    for key, expected in query.items():
        actual = nested_get(row, key)
        if isinstance(expected, dict):
            if "$exists" in expected and (actual is not None) is not expected["$exists"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$nin" in expected and actual in expected["$nin"]:
                return False
            if "$gte" in expected and not (actual is not None and actual >= expected["$gte"]):
                return False
            if "$lte" in expected and not (actual is not None and actual <= expected["$lte"]):
                return False
            if "$gt" in expected and not (actual is not None and actual > expected["$gt"]):
                return False
            continue
        if isinstance(actual, list):
            if expected not in actual:
                return False
            continue
        if actual != expected:
            return False
    return True


def scrub_bulk_update(update):
    scrubbed = {"$set": dict(update.get("$set", {}))}
    scrubbed["$set"].pop("updated_at", None)
    return scrubbed


def nested_get(row, dotted_key):
    value = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def mongo_sort_value(value):
    if value is None:
        return (0, "")
    return (1, value)


# --- 2026-09-10 review: rebuild integrity (#32 #33 #34 #35 #47 #53 #55) -------


def _sectioned_result(sections, *, checksum="c1", origin=None, created=None):
    nodes = [
        IngestedNode(
            node_key="root",
            title="Doc",
            text=" ".join(text for _title, text in sections),
            labels=["source_root"],
        )
    ]
    for index, (title, text) in enumerate(sections, start=1):
        key = f"section-{index}"
        nodes.append(
            IngestedNode(node_key=key, parent_key="root", title=title, text=text, labels=["source_section"])
        )
        nodes.append(
            IngestedNode(
                node_key=f"{key}-paragraph-1",
                parent_key=key,
                title=f"{title} / paragraph 1",
                text=text,
                labels=["source_chunk"],
            )
        )
    origin_date, origin_source = origin or (None, None)
    return IngestionResult(
        source=SourceRef(
            path="doc.md",
            kind="markdown",
            checksum_sha256=checksum,
            origin_date=origin_date,
            origin_date_source=origin_source,
        ),
        title="Doc",
        summary="s",
        nodes=nodes,
        created_at=created or datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
    )


def _active(db, title, label="source_section"):
    return next(
        row
        for row in db.nodes.rows
        if row["title"] == title and label in row["labels"] and row.get("status") == "active"
    )


def _reviewed_edges(db):
    return [row for row in db.graph_edges.rows if (row.get("provenance") or {}).get("adapter") == "user_review"]


ALPHA_BETA = [("Alpha", "Alpha text."), ("Beta", "Beta text.")]


def _ingest_with_reviewed_edge(db):
    inserted = commit_ingestion(db, _sectioned_result(ALPHA_BETA), embedder=FakeEmbedder())
    alpha, beta = _active(db, "Alpha"), _active(db, "Beta")
    assert create_reviewed_semantic_edge(db, str(alpha["_id"]), str(beta["_id"]))["ok"] is True
    return inserted["document_id"], alpha, beta


def test_targeted_noop_rebuild_keeps_reviewed_edges() -> None:
    # #32: a byte-identical diff rebuild used to hard-delete every reviewed edge.
    db = FakeDb()
    document_id, alpha, beta = _ingest_with_reviewed_edge(db)

    result = rebuild_document(db, document_id, _sectioned_result(ALPHA_BETA), embedder=FakeEmbedder(), mode="diff")

    assert result["diff"]["counts"]["changed"] == 0
    [edge] = _reviewed_edges(db)
    assert (edge["source_node_id"], edge["target_node_id"]) == (alpha["_id"], beta["_id"])
    assert result["reviewed_edge_count"] == 1
    assert result["flagged_reviewed_edge_count"] == 0
    assert "needs_review" not in edge


def test_targeted_rebuild_flags_reviewed_edge_whose_endpoint_text_changed() -> None:
    db = FakeDb()
    document_id, _alpha, _beta = _ingest_with_reviewed_edge(db)

    changed = [("Alpha", "Alpha text."), ("Beta", "Beta text, revised.")]
    result = rebuild_document(db, document_id, _sectioned_result(changed), embedder=FakeEmbedder(), mode="diff")

    [edge] = _reviewed_edges(db)
    assert edge["needs_review"] is True
    assert edge["needs_review_reason"] == "endpoint_content_changed"
    assert result["flagged_reviewed_edge_count"] == 1


def test_full_rebuild_carries_reviewed_edges_onto_new_node_ids() -> None:
    # #32 second half: full rebuild used to leave reviewed edges on superseded ids.
    db = FakeDb()
    document_id, alpha, beta = _ingest_with_reviewed_edge(db)

    result = rebuild_document(db, document_id, _sectioned_result(ALPHA_BETA), embedder=FakeEmbedder())

    new_alpha, new_beta = _active(db, "Alpha"), _active(db, "Beta")
    assert new_alpha["_id"] != alpha["_id"]
    [edge] = _reviewed_edges(db)
    assert (edge["source_node_id"], edge["target_node_id"]) == (new_alpha["_id"], new_beta["_id"])
    assert edge["tree_id"] == new_alpha["tree_id"]
    assert result["remapped_reviewed_edge_count"] == 1
    assert result["orphaned_reviewed_edge_count"] == 0


def test_targeted_rebuild_keeps_endorsement_with_its_text_when_sections_shift() -> None:
    # #33: inserting a section at the top used to rebind Alpha's id (and its
    # endorsement and usage) onto the new section's text.
    db = FakeDb()
    inserted = commit_ingestion(db, _sectioned_result(ALPHA_BETA), embedder=FakeEmbedder())
    alpha = _active(db, "Alpha")
    db.nodes.update_one(
        {"_id": alpha["_id"]}, {"$set": {"endorsement_label": "explicit_endorsed", "usage_score": 42}}
    )

    shifted = [("New", "NNN new text."), *ALPHA_BETA]
    rebuild_document(db, inserted["document_id"], _sectioned_result(shifted), embedder=FakeEmbedder(), mode="diff")

    kept = next(row for row in db.nodes.rows if row["_id"] == alpha["_id"])
    assert (kept["title"], kept["text"]) == ("Alpha", "Alpha text.")
    assert kept["endorsement_label"] == "explicit_endorsed"
    assert kept["usage_score"] == 42
    new = _active(db, "New")
    assert new["_id"] != alpha["_id"]
    assert new["endorsement_label"] == "unreviewed"


def test_targeted_rebuild_clears_judgements_when_node_content_changes() -> None:
    db = FakeDb()
    inserted = commit_ingestion(db, _sectioned_result(ALPHA_BETA), embedder=FakeEmbedder())
    alpha = _active(db, "Alpha")
    db.nodes.update_one(
        {"_id": alpha["_id"]},
        {"$set": {"endorsement_label": "explicit_endorsed", "usage_score": 42, "continuity_critical": True}},
    )

    revised = [("Alpha", "Alpha text, rewritten."), ("Beta", "Beta text.")]
    result = rebuild_document(
        db, inserted["document_id"], _sectioned_result(revised), embedder=FakeEmbedder(), mode="diff"
    )

    row = next(row for row in db.nodes.rows if row["_id"] == alpha["_id"])
    assert row["text"] == "Alpha text, rewritten."
    assert row["endorsement_label"] == "unreviewed"
    assert row["usage_score"] == 0
    assert row["continuity_critical"] is False
    assert row["content_change_history"][-1]["previous_endorsement_label"] == "explicit_endorsed"
    assert result["cleared_endorsement_count"] == 1


def test_rebuild_keeps_operator_origin_date_history_and_homogeneous_nodes() -> None:
    # #34: rebuild used to revert the operator's date, drop its history, and
    # leave the document's nodes split across two origin dates.
    db = FakeDb()
    inserted = commit_ingestion(
        db, _sectioned_result(ALPHA_BETA, origin=("2026-01-01", "file_created")), embedder=FakeEmbedder()
    )
    document_id = inserted["document_id"]
    assert update_document_origin_date(db, document_id, "1998-04-05", reviewer="alice", note="letterhead")["ok"]

    revised = [("Alpha", "Alpha text."), ("Beta", "Beta text, revised.")]
    result = rebuild_document(
        db,
        document_id,
        _sectioned_result(revised, origin=("2026-09-10", "file_created")),
        embedder=FakeEmbedder(),
        mode="diff",
    )

    source = db.documents.rows[0]["source"]
    assert (source["origin_date"], source["origin_date_source"]) == ("1998-04-05", "operator")
    assert len(source["origin_date_history"]) == 1
    assert any(candidate.get("source") == "operator" for candidate in source["date_candidates"])
    active = [row for row in db.nodes.rows if row.get("status") == "active"]
    assert {row["origin_date"] for row in active} == {"1998-04-05"}
    assert result["source_changes"]["origin_date_preserved"]["recomputed_origin_date"] == "2026-09-10"


def test_rebuild_records_previous_checksum() -> None:
    # #35: the stored checksum must name the current content; the old one is kept.
    db = FakeDb()
    inserted = commit_ingestion(db, _sectioned_result(ALPHA_BETA, checksum="c1"), embedder=FakeEmbedder())

    result = rebuild_document(
        db, inserted["document_id"], _sectioned_result(ALPHA_BETA, checksum="c2"), embedder=FakeEmbedder(), mode="diff"
    )

    source = db.documents.rows[0]["source"]
    assert source["checksum_sha256"] == "c2"
    assert [entry["checksum_sha256"] for entry in source["previous_checksums"]] == ["c1"]
    assert result["source_changes"]["checksum_changed"] == {"from": "c1", "to": "c2"}


def test_rebuild_moves_operator_summary_to_history_when_regenerating() -> None:
    # #47: a regenerated summary used to keep the operator's provenance.
    from tirzah.retrieval.queries import skip_summary_for_record

    db = FakeDb()
    inserted = commit_ingestion(db, _sectioned_result(ALPHA_BETA), embedder=FakeEmbedder())
    chunk = _active(db, "Beta / paragraph 1", label="source_chunk")
    assert update_node_summary(db, str(chunk["_id"]), "Operator gloss of beta.", reviewer="alice")["ok"]

    revised = [("Alpha", "Alpha text."), ("Beta", "Beta text, revised.")]
    result = rebuild_document(
        db, inserted["document_id"], _sectioned_result(revised), embedder=FakeEmbedder(), mode="diff"
    )

    row = next(row for row in db.nodes.rows if row["_id"] == chunk["_id"])
    assert row["summary_provenance"]["source"] == "derived_extractive"
    assert row["summary_provenance"]["history"][-1]["summary"] == "Operator gloss of beta."
    assert row["summary_provenance"]["history"][-1]["source"] == "operator"
    assert skip_summary_for_record(row)[1] == "derived_extractive"
    assert result["summary_provenance_reset_count"] == 1


def test_skip_summary_does_not_attribute_rewritten_text_to_its_old_provenance() -> None:
    from tirzah.ingestion.diff import node_content_sha256
    from tirzah.retrieval.queries import skip_summary_for_record

    stamp = {"source": "operator", "summary_sha256": node_content_sha256("Operator gloss.")}
    assert skip_summary_for_record({"summary": "Operator gloss.", "summary_provenance": stamp})[1] == "operator"
    assert skip_summary_for_record({"summary": "Machine extract.", "summary_provenance": stamp})[1] == "stored"


def test_targeted_rebuild_rollback_never_mass_deletes_nodes() -> None:
    # #53: rollback used to delete_many the whole tree, then insert_many it back.
    db = FakeDb(fail_edges=True)
    inserted = commit_ingestion(db, _sectioned_result(ALPHA_BETA), embedder=FakeEmbedder())
    snapshot = {row["_id"]: dict(row) for row in db.nodes.rows}
    rebuilt = _sectioned_result([("Alpha", "Alpha text."), ("Beta", "Beta revised."), ("Gamma", "Gamma.")])
    rebuilt.nodes[1].relations = [{"target_node_key": "root", "relation_type": "part_of"}]
    bulk_deletes = []
    real_delete_many = db.nodes.delete_many
    db.nodes.delete_many = lambda query: bulk_deletes.append(query) or real_delete_many(query)

    try:
        rebuild_document(db, inserted["document_id"], rebuilt, embedder=FakeEmbedder(), mode="diff")
    except RuntimeError as error:
        assert "edge insert failed" in str(error)
    else:
        raise AssertionError("Expected the edge insert failure to propagate.")

    assert bulk_deletes == []
    assert {row["_id"]: row for row in db.nodes.rows} == snapshot


def test_ingestion_links_children_emitted_before_their_parent() -> None:
    # #55: a child before its parent used to be stored with parent_id=None.
    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(path="x.md", kind="markdown"),
        title="X",
        summary="s",
        nodes=[
            IngestedNode(node_key="child", parent_key="root", title="Child", text="child"),
            IngestedNode(node_key="root", title="Root", text="root"),
        ],
    )

    commit_ingestion(db, result, embedder=FakeEmbedder())

    by_key = {row["node_key"]: row for row in db.nodes.rows}
    assert by_key["child"]["parent_id"] == by_key["root"]["_id"]


def test_ingestion_rejects_unknown_parent_key_before_writing() -> None:
    from tirzah.db.repositories import IngestionStructureError

    db = FakeDb()
    result = IngestionResult(
        source=SourceRef(path="x.md", kind="markdown"),
        title="X",
        summary="s",
        nodes=[
            IngestedNode(node_key="root", title="Root", text="root"),
            IngestedNode(node_key="orphan", parent_key="missing", title="Orphan", text="orphan"),
        ],
    )

    try:
        commit_ingestion(db, result, embedder=FakeEmbedder())
    except IngestionStructureError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected IngestionStructureError for an unresolved parent_key.")
    assert db.documents.rows == []
    assert db.nodes.rows == []


def test_contradiction_batch_replaces_excluded_focus_nodes_and_pages(monkeypatch) -> None:
    # #65: exclusions used to shrink the batch (limit applied before them),
    # and with no cursor a sweep could never reach past its first page.
    db = FakeDb()
    ids = []
    for index in range(1, 7):
        node_id = ObjectId()
        ids.append(node_id)
        db.nodes.rows.append(
            {
                "_id": node_id,
                "node_key": f"k{index}",
                "title": f"Node {index}",
                "labels": ["source_chunk"],
                "text": f"text {index}",
                "embedding": {"model": "mock", "dimensions": 2},
            }
        )
    monkeypatch.setattr(
        "tirzah.retrieval.contradictions.contradiction_candidate_nodes", lambda _db, node_id, **_kwargs: []
    )

    first = enqueue_contradiction_candidate_batch(db, focus_limit=3, exclude_node_keys=["k1", "k2"], dry_run=True)

    assert [row["node_id"] for row in first["focus_results"]] == [str(node_id) for node_id in ids[2:5]]
    assert first["scope"]["focus_node_count"] == 3
    assert first["scope"]["excluded_focus_counts"]["excluded_node_key"] == 2
    assert first["scope"]["exhausted"] is False
    assert first["scope"]["next_after_node_id"] == str(ids[4])

    second = enqueue_contradiction_candidate_batch(
        db, focus_limit=3, dry_run=True, after_node_id=first["scope"]["next_after_node_id"]
    )
    assert [row["node_id"] for row in second["focus_results"]] == [str(ids[5])]
    assert second["scope"]["exhausted"] is True
    assert second["scope"]["next_after_node_id"] is None

    bad = enqueue_contradiction_candidate_batch(db, dry_run=True, after_node_id="not-an-id")
    assert bad == {"ok": False, "reason": "invalid_after_node_id", "after_node_id": "not-an-id"}
