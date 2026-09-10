from datetime import datetime, timezone

from bson import ObjectId

from tirzah.retrieval.queries import (
    attach_query_similarity,
    build_prompt_envelope,
    build_prompt_envelope_without_context,
    context_record,
    default_no_context_system_instruction,
    embedding_candidate_nodes,
    embedding_candidate_report,
    estimate_tokens,
    expand_graph_paths,
    expand_proximity,
    graph_edges_for_node,
    nearby_siblings,
    node_search_score,
    node_search_sort_key,
    parse_iso_datetime,
    prioritize_records,
    query_embedding_candidate_nodes,
    render_context_document,
    render_record,
    search_nodes,
    semantic_candidate_nodes,
    semantic_labels,
    serialize_document,
    serialize_node,
    text_query_filters,
    text_query_terms,
)
from tirzah.db.memory_store import MemoryStore


def test_serialize_document_handles_source_and_dates() -> None:
    document = {
        "_id": "doc1",
        "schema_version": 1,
        "title": "Title",
        "summary": "Summary",
        "source": {"path": "source.md"},
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
    }

    serialized = serialize_document(document)

    assert serialized["document_id"] == "doc1"
    assert serialized["source"]["path"] == "source.md"
    assert serialized["created_at"] == "2026-01-01T00:00:00+00:00"


def test_serialize_node_returns_preview_and_ids() -> None:
    last_used_at = datetime(2026, 1, 3, tzinfo=timezone.utc)
    node = {
        "_id": "node1",
        "document_id": "doc1",
        "tree_id": "tree1",
        "parent_id": None,
        "node_key": "root",
        "parent_key": None,
        "title": "Title",
        "text": "x" * 400,
        "labels": ["source_root"],
        "endorsement_label": "unreviewed",
        "usage_score": 7,
        "last_used_at": last_used_at,
        "provenance": {"adapter": "mock"},
        "created_at": None,
    }

    serialized = serialize_node(node)

    assert serialized["node_id"] == "node1"
    assert serialized["document_id"] == "doc1"
    assert serialized["text_preview"] == "x" * 300
    assert serialized["usage_score"] == 7
    assert serialized["usage_score_bonus"] == 7
    assert serialized["last_used_at"] == "2026-01-03T00:00:00+00:00"


def test_serialize_node_preserves_raw_usage_score_and_reports_bounded_bonus() -> None:
    node = {
        "_id": "node1",
        "document_id": "doc1",
        "tree_id": "tree1",
        "title": "Title",
        "text": "text",
        "usage_score": 50,
    }

    serialized = serialize_node(node)

    assert serialized["usage_score"] == 50
    assert serialized["usage_score_bonus"] == 10


def test_serialize_node_defaults_malformed_usage_score() -> None:
    node = {
        "_id": "node1",
        "document_id": "doc1",
        "tree_id": "tree1",
        "title": "Title",
        "text": "text",
        "usage_score": "many",
    }

    serialized = serialize_node(node)

    assert serialized["usage_score"] == 0
    assert serialized["usage_score_bonus"] == 0


def test_parse_iso_datetime_accepts_z_suffix() -> None:
    parsed = parse_iso_datetime("2026-01-01T00:00:00Z")

    assert parsed.isoformat() == "2026-01-01T00:00:00+00:00"


def test_text_query_terms_extracts_natural_query_content_terms() -> None:
    assert text_query_terms(
        "Who commissioned the Taj Mahal, and who was it built to commemorate?"
    ) == ["commissioned", "Taj", "Mahal", "built", "commemorate"]


def test_text_query_filters_combines_exact_query_and_content_terms() -> None:
    filters = text_query_filters("Who commissioned the Taj Mahal?")

    assert len(filters) == 12
    assert [set(query_filter) for query_filter in filters] == [
        {"title"},
        {"text"},
        {"labels"},
        {"title"},
        {"text"},
        {"labels"},
        {"title"},
        {"text"},
        {"labels"},
        {"title"},
        {"text"},
        {"labels"},
    ]
    assert filters[0]["title"].pattern == "Who\\ commissioned\\ the\\ Taj\\ Mahal\\?"
    assert filters[3]["title"].pattern == "\\bcommissioned\\b"
    assert filters[6]["title"].pattern == "\\bTaj\\b"
    assert filters[9]["title"].pattern == "\\bMahal\\b"


def test_text_query_filters_deduplicates_single_term_query() -> None:
    assert len(text_query_filters("Mahal")) == 3


def test_node_search_score_uses_whole_terms_for_query_tokens() -> None:
    exact_term = {
        "title": "Construction note",
        "text": "The tomb was built as an object.",
        "labels": ["source_chunk"],
    }
    substring_only = {
        "title": "Construction note",
        "text": "The tomb was rebuilt as an object.",
        "labels": ["source_chunk"],
    }

    assert node_search_score(exact_term, "built object") > node_search_score(
        substring_only,
        "built object",
    )


def test_context_record_adds_role_and_distance() -> None:
    node = {
        "_id": "node1",
        "document_id": "doc1",
        "tree_id": "tree1",
        "parent_id": None,
        "title": "Title",
        "text": "Body",
    }

    record = context_record("focus", node, 0)

    assert record["role"] == "focus"
    assert record["distance"] == 0
    assert record["node_id"] == "node1"


def test_semantic_labels_excludes_structural_source_labels() -> None:
    assert semantic_labels(
        ["source_chunk", "source_custom", "taj_mahal", "online_test", ""]
    ) == ["online_test", "taj_mahal"]


def test_search_nodes_applies_identity_label_and_document_exclusions() -> None:
    allowed_document_id = ObjectId()
    excluded_document_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": ObjectId(),
                "document_id": allowed_document_id,
                "tree_id": ObjectId(),
                "title": "Allowed memory",
                "text": "memory",
                "labels": ["source_chunk", "public"],
            },
            {
                "_id": ObjectId(),
                "document_id": allowed_document_id,
                "tree_id": ObjectId(),
                "title": "Restricted memory",
                "text": "memory",
                "labels": ["source_chunk", "restricted"],
            },
            {
                "_id": ObjectId(),
                "document_id": excluded_document_id,
                "tree_id": ObjectId(),
                "title": "Hidden document memory",
                "text": "memory",
                "labels": ["source_chunk", "public"],
            },
        ]
    )

    results = search_nodes(
        db,
        identity={
            "excluded_labels": ["restricted"],
            "excluded_document_ids": [str(excluded_document_id)],
        },
    )

    assert [result["title"] for result in results] == ["Allowed memory"]


def test_search_nodes_expands_candidate_window_when_identity_filters() -> None:
    allowed_document_id = ObjectId()
    excluded_document_id = ObjectId()
    excluded_nodes = [
        {
            "_id": ObjectId(),
            "document_id": excluded_document_id,
            "tree_id": ObjectId(),
            "title": f"Hidden memory {index}",
            "text": "memory",
            "labels": ["source_chunk"],
        }
        for index in range(5)
    ]
    db = FakeDb(
        [
            *excluded_nodes,
            {
                "_id": ObjectId(),
                "document_id": allowed_document_id,
                "tree_id": ObjectId(),
                "title": "Allowed later memory",
                "text": "memory",
                "labels": ["source_chunk"],
            },
        ]
    )

    results = search_nodes(
        db,
        limit=1,
        identity={"excluded_document_ids": [str(excluded_document_id)]},
    )

    assert [result["title"] for result in results] == ["Allowed later memory"]


def test_search_nodes_ignores_superseded_nodes() -> None:
    document_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": ObjectId(),
                "document_id": document_id,
                "tree_id": ObjectId(),
                "title": "Old memory",
                "text": "memory",
                "labels": ["source_chunk"],
                "status": "superseded",
            },
            {
                "_id": ObjectId(),
                "document_id": document_id,
                "tree_id": ObjectId(),
                "title": "Active memory",
                "text": "memory",
                "labels": ["source_chunk"],
                "status": "active",
            },
        ]
    )

    results = search_nodes(db)

    assert [result["title"] for result in results] == ["Active memory"]


def test_search_nodes_ignores_pending_review_nodes() -> None:
    document_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": ObjectId(),
                "document_id": document_id,
                "tree_id": ObjectId(),
                "title": "Proposed memory",
                "text": "memory",
                "labels": ["source_chunk"],
                "status": "pending_review",
            },
            {
                "_id": ObjectId(),
                "document_id": document_id,
                "tree_id": ObjectId(),
                "title": "Active memory",
                "text": "memory",
                "labels": ["source_chunk"],
                "status": "active",
            },
        ]
    )

    results = search_nodes(db)

    assert [result["title"] for result in results] == ["Active memory"]


def test_search_nodes_filters_by_origin_date_range() -> None:
    document_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": ObjectId(),
                "document_id": document_id,
                "tree_id": ObjectId(),
                "title": "Old paper",
                "text": "memory",
                "labels": ["source_chunk"],
                "status": "active",
                "origin_date": "2010-01-01",
            },
            {
                "_id": ObjectId(),
                "document_id": document_id,
                "tree_id": ObjectId(),
                "title": "Recent paper",
                "text": "memory",
                "labels": ["source_chunk"],
                "status": "active",
                "origin_date": "2024-06-01",
            },
        ]
    )

    results = search_nodes(db, origin_after="2020-01-01")
    assert [row["title"] for row in results] == ["Recent paper"]


def test_node_search_sort_prefers_newer_origin_date_as_secondary_signal() -> None:
    older = {
        "title": "memory",
        "text": "memory",
        "labels": ["source_chunk"],
        "origin_date": "2010-01-01",
    }
    newer = {
        "title": "memory",
        "text": "memory",
        "labels": ["source_chunk"],
        "origin_date": "2024-01-01",
    }
    assert node_search_sort_key(newer, "memory") > node_search_sort_key(older, "memory")


class FakeCursor(list):
    def sort(self, *_args):
        return self

    def limit(self, limit):
        return FakeCursor(self[:limit])


class FakeNodes:
    def __init__(self, nodes):
        self.nodes = nodes

    def find_one(self, query):
        return next((node for node in self.nodes if matches(node, query)), None)

    def find(self, query):
        return FakeCursor(
            [node for node in self.nodes if matches(node, query)]
        )


class FakeEdges:
    def __init__(self, edges):
        self.edges = edges

    def find(self, query):
        return FakeCursor([edge for edge in self.edges if matches(edge, query)])


class FakeDb:
    def __init__(self, nodes, edges=None):
        self.nodes = FakeNodes(nodes)
        self.graph_edges = FakeEdges(edges or [])


def matches(row, query):
    for key, expected in query.items():
        if key == "$or":
            if not any(matches(row, option) for option in expected):
                return False
            continue
        if isinstance(expected, dict):
            value = nested_get(row, key)
            if "$exists" in expected and (value is not None) is not expected["$exists"]:
                return False
            if "$ne" in expected and value == expected["$ne"]:
                return False
            if "$nin" in expected and value in expected["$nin"]:
                return False
            if "$gte" in expected and not (value is not None and value >= expected["$gte"]):
                return False
            if "$lte" in expected and not (value is not None and value <= expected["$lte"]):
                return False
            if "$in" in expected:
                if isinstance(value, list):
                    if not set(value) & set(expected["$in"]):
                        return False
                elif value not in expected["$in"]:
                    return False
            continue
        if nested_get(row, key) != expected:
            return False
    return True


def nested_get(row, dotted_key):
    value = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def test_semantic_candidate_nodes_ranks_shared_label_matches() -> None:
    document_id = ObjectId()
    other_document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    strong_id = ObjectId()
    weak_id = ObjectId()
    same_doc_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "labels": ["source_chunk", "taj_mahal", "online_test"],
            },
            {
                "_id": weak_id,
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Weak",
                "text": "Weak text",
                "labels": ["source_chunk", "taj_mahal"],
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Root Candidate",
                "text": "Root text",
                "labels": ["source_root", "taj_mahal", "online_test"],
            },
            {
                "_id": strong_id,
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Strong",
                "text": "Strong text",
                "labels": ["source_section", "taj_mahal", "online_test"],
            },
            {
                "_id": same_doc_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Same Doc",
                "text": "Same doc text",
                "labels": ["taj_mahal", "online_test"],
            },
        ]
    )

    candidates = semantic_candidate_nodes(db, str(focus_id))

    assert [candidate["title"] for candidate in candidates] == ["Strong", "Weak"]
    assert candidates[0]["shared_labels"] == ["online_test", "taj_mahal"]
    assert candidates[0]["shared_label_count"] == 2


def test_semantic_candidate_nodes_uses_natural_title_order_for_ties() -> None:
    document_id = ObjectId()
    other_document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "labels": ["topic"],
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Paragraph 10",
                "text": "Tenth text",
                "labels": ["topic"],
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Paragraph 2",
                "text": "Second text",
                "labels": ["topic"],
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Paragraph 1",
                "text": "First text",
                "labels": ["topic"],
            },
        ]
    )

    candidates = semantic_candidate_nodes(db, str(focus_id))

    assert [candidate["title"] for candidate in candidates] == [
        "Paragraph 1",
        "Paragraph 2",
        "Paragraph 10",
    ]


def test_embedding_candidate_nodes_ranks_comparable_vector_matches() -> None:
    document_id = ObjectId()
    other_document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    strong_id = ObjectId()
    weak_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "labels": ["source_chunk"],
                "embedding": {
                    "model": "mock",
                    "dimensions": 3,
                    "vector": [1.0, 0.0, 0.0],
                },
            },
            {
                "_id": weak_id,
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Weak",
                "text": "Weak text",
                "labels": ["source_chunk"],
                "embedding": {
                    "model": "mock",
                    "dimensions": 3,
                    "vector": [0.8, 0.6, 0.0],
                },
            },
            {
                "_id": strong_id,
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Strong",
                "text": "Strong text",
                "labels": ["source_section"],
                "embedding": {
                    "model": "mock",
                    "dimensions": 3,
                    "vector": [0.99, 0.01, 0.0],
                },
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Wrong Model",
                "text": "Wrong model text",
                "labels": ["source_chunk"],
                "embedding": {
                    "model": "other",
                    "dimensions": 3,
                    "vector": [1.0, 0.0, 0.0],
                },
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Root Candidate",
                "text": "Root text",
                "labels": ["source_root"],
                "embedding": {
                    "model": "mock",
                    "dimensions": 3,
                    "vector": [1.0, 0.0, 0.0],
                },
            },
            {
                "_id": ObjectId(),
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Same Doc",
                "text": "Same doc text",
                "labels": ["source_chunk"],
                "embedding": {
                    "model": "mock",
                    "dimensions": 3,
                    "vector": [1.0, 0.0, 0.0],
                },
            },
        ]
    )

    candidates = embedding_candidate_nodes(db, str(focus_id), min_similarity=0.75)

    assert [candidate["title"] for candidate in candidates] == ["Strong", "Weak"]
    assert candidates[0]["node_id"] == str(strong_id)
    assert candidates[0]["embedding_model"] == "mock"
    assert candidates[0]["embedding_dimensions"] == 3
    assert candidates[0]["embedding_similarity"] > candidates[1]["embedding_similarity"]


def test_embedding_candidate_nodes_can_include_same_document_when_requested() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    same_doc_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
            {
                "_id": same_doc_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Same Doc",
                "text": "Same doc text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
        ]
    )

    excluded = embedding_candidate_nodes(db, str(focus_id))
    included = embedding_candidate_nodes(db, str(focus_id), include_same_document=True)

    assert excluded == []
    assert [candidate["node_id"] for candidate in included] == [str(same_doc_id)]


def test_embedding_candidate_nodes_filters_unembedded_rows_before_candidate_limit() -> None:
    document_id = ObjectId()
    other_document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    target_id = ObjectId()
    unembedded = [
        {
            "_id": ObjectId(),
            "document_id": other_document_id,
            "tree_id": tree_id,
            "title": f"Unembedded {index}",
            "text": "Old node without embedding.",
        }
        for index in range(120)
    ]
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
            *unembedded,
            {
                "_id": target_id,
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Late Embedded Target",
                "text": "Late target text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [0.9, 0.1]},
            },
        ]
    )

    candidates = embedding_candidate_nodes(db, str(focus_id), min_similarity=0.5, limit=1)

    assert [candidate["node_id"] for candidate in candidates] == [str(target_id)]


def test_embedding_candidate_nodes_requires_matching_model_metadata() -> None:
    document_id = ObjectId()
    other_document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Missing Model",
                "text": "Missing model text",
                "embedding": {"dimensions": 2, "vector": [1.0, 0.0]},
            },
        ]
    )

    assert embedding_candidate_nodes(db, str(focus_id), min_similarity=0.5) == []


def test_embedding_candidate_report_explains_scan_and_exclusions() -> None:
    document_id = ObjectId()
    other_document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "embedding": {
                    "adapter": "mock",
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                },
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Strong",
                "text": "Strong text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [0.99, 0.01]},
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Below threshold",
                "text": "Weak text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [0.2, 0.8]},
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "Wrong model",
                "text": "Wrong model text",
                "embedding": {"model": "other", "dimensions": 2, "vector": [1.0, 0.0]},
            },
            {
                "_id": ObjectId(),
                "document_id": other_document_id,
                "tree_id": tree_id,
                "title": "No vector",
                "text": "No vector text",
                "embedding": {"model": "mock", "dimensions": 2},
            },
        ]
    )

    report = embedding_candidate_report(db, str(focus_id), min_similarity=0.75, limit=3)

    assert report["ok"] is True
    assert [node["title"] for node in report["nodes"]] == ["Strong"]
    assert report["diagnostics"]["focus"] == {
        "node_id": str(focus_id),
        "title": "Focus",
        "adapter": "mock",
        "model": "mock",
        "dimensions": 2,
    }
    assert report["diagnostics"]["scanned_count"] == 4
    assert report["diagnostics"]["scan_truncated"] is False
    assert report["diagnostics"]["returned_count"] == 1
    assert report["diagnostics"]["exclusions"]["below_threshold"] == 1
    assert report["diagnostics"]["exclusions"]["incompatible_embedding"] == 1
    assert report["diagnostics"]["exclusions"]["invalid_embedding"] == 1
    assert report["diagnostics"]["candidate_scan_limit"] == 1000
    assert report["diagnostics"]["returned_source_documents"] == [
        {
            "document_id": str(other_document_id),
            "source_path": None,
            "candidate_count": 1,
            "best_similarity": report["nodes"][0]["embedding_similarity"],
            "best_rank_score": report["nodes"][0]["embedding_rank_score"],
        }
    ]


def test_embedding_candidate_report_honors_candidate_scan_limit() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    hidden_match_id = ObjectId()
    filler = [
        {
            "_id": ObjectId(),
            "document_id": ObjectId(),
            "tree_id": tree_id,
            "title": f"Filler {index}",
            "text": "Filler text",
            "embedding": {"model": "mock", "dimensions": 2, "vector": [0.0, 1.0]},
        }
        for index in range(120)
    ]
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
            *filler,
            {
                "_id": hidden_match_id,
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Late strong match",
                "text": "Late strong text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
        ]
    )

    short_scan = embedding_candidate_report(
        db,
        str(focus_id),
        min_similarity=0.75,
        limit=5,
        candidate_scan_limit=100,
    )
    full_scan = embedding_candidate_report(
        db,
        str(focus_id),
        min_similarity=0.75,
        limit=5,
        candidate_scan_limit=200,
    )

    assert [node["title"] for node in short_scan["nodes"]] == []
    assert [node["title"] for node in full_scan["nodes"]] == ["Late strong match"]
    assert short_scan["diagnostics"]["candidate_scan_limit"] == 100
    assert short_scan["diagnostics"]["scanned_count"] == 100
    assert short_scan["diagnostics"]["scan_truncated"] is True
    assert full_scan["diagnostics"]["candidate_scan_limit"] == 200
    assert full_scan["diagnostics"]["scan_truncated"] is False


def test_embedding_candidate_report_deduplicates_candidate_text_hashes() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Section duplicate",
                "text": "Same text",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "same",
                },
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Paragraph duplicate",
                "text": "Same text",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "same",
                },
            },
        ]
    )

    report = embedding_candidate_report(
        db,
        str(focus_id),
        min_similarity=0.75,
        limit=5,
        candidate_scan_limit=10,
    )

    assert [node["title"] for node in report["nodes"]] == ["Section duplicate"]
    assert report["diagnostics"]["exclusions"]["duplicate_text"] == 1


def test_embedding_candidate_report_excludes_candidate_with_focus_text_hash() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Copied policy text",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "copied",
                },
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Copied policy paragraph",
                "text": "Copied policy text",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "copied",
                },
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Real related passage",
                "text": "Different but strongly related policy text",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [0.9, 0.1],
                    "source_text_hash": "related",
                },
            },
        ]
    )

    report = embedding_candidate_report(
        db,
        str(focus_id),
        min_similarity=0.75,
        limit=5,
        candidate_scan_limit=10,
    )

    assert [node["title"] for node in report["nodes"]] == ["Real related passage"]
    assert report["diagnostics"]["exclusions"]["duplicate_text"] == 1


def test_embedding_candidate_report_excludes_case_variant_focus_text() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "1. canonical policy\n2. domain policy",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "lowercase",
                },
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Copied policy paragraph",
                "text": "1. Canonical policy\n2. Domain policy",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "titlecase",
                },
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Real related passage",
                "text": "Different but strongly related policy text",
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [0.9, 0.1],
                    "source_text_hash": "related",
                },
            },
        ]
    )

    report = embedding_candidate_report(
        db,
        str(focus_id),
        min_similarity=0.75,
        limit=5,
        candidate_scan_limit=10,
    )

    assert [node["title"] for node in report["nodes"]] == ["Real related passage"]
    assert report["diagnostics"]["exclusions"]["duplicate_text"] == 1


def test_embedding_candidate_report_excludes_near_copied_focus_text() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": (
                    "When documents appear to conflict, use this order: canonical policy, "
                    "domain policy, project-local workflow, advisory guidance, and archival "
                    "or historical material."
                ),
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "source",
                },
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Near copied policy",
                "text": (
                    "When two documents appear to conflict, use this order: canonical policy, "
                    "domain policy, project-local workflow, advisory guidance, and archival "
                    "or historical material."
                ),
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [1.0, 0.0],
                    "source_text_hash": "near-copy",
                },
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Related but different",
                "text": (
                    "A governance process may compare authority, recency, document purpose, "
                    "and operator intent before deciding which source controls a task."
                ),
                "embedding": {
                    "model": "mock",
                    "dimensions": 2,
                    "vector": [0.9, 0.1],
                    "source_text_hash": "related",
                },
            },
        ]
    )

    report = embedding_candidate_report(
        db,
        str(focus_id),
        min_similarity=0.75,
        limit=5,
        candidate_scan_limit=10,
    )

    assert [node["title"] for node in report["nodes"]] == ["Related but different"]
    assert report["diagnostics"]["exclusions"]["duplicate_text"] == 1


def test_embedding_candidate_report_ranks_prose_above_link_index_when_close() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
                "embedding": {"model": "mock", "dimensions": 2, "vector": [1.0, 0.0]},
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Link index",
                "text": "\n".join(
                    [
                        "- [One](/home/cello/domains/AMS/one.md)",
                        "- [Two](/home/cello/domains/AMS/two.md)",
                        "- [Three](/home/cello/domains/AMS/three.md)",
                        "- [Four](/home/cello/domains/AMS/four.md)",
                        "- [Five](/home/cello/domains/AMS/five.md)",
                    ]
                ),
                "embedding": {"model": "mock", "dimensions": 2, "vector": [0.99, 0.01]},
            },
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "tree_id": tree_id,
                "title": "Prose candidate",
                "text": (
                    "This candidate explains the same policy area in prose, with enough "
                    "context to support a human review decision."
                ),
                "embedding": {"model": "mock", "dimensions": 2, "vector": [0.97, 0.243]},
            },
        ]
    )

    report = embedding_candidate_report(
        db,
        str(focus_id),
        min_similarity=0.75,
        limit=2,
        candidate_scan_limit=10,
    )

    assert [node["title"] for node in report["nodes"]] == ["Prose candidate", "Link index"]
    assert report["nodes"][0]["link_index_penalty"] == 0.0
    assert report["nodes"][1]["link_index_penalty"] == 0.05
    assert report["nodes"][1]["embedding_similarity"] > report["nodes"][0]["embedding_similarity"]
    assert report["nodes"][0]["embedding_rank_score"] > report["nodes"][1]["embedding_rank_score"]


def test_embedding_candidate_report_reports_missing_focus_embedding() -> None:
    focus_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "title": "Focus",
                "text": "Focus text",
            },
        ]
    )

    report = embedding_candidate_report(db, str(focus_id))

    assert report["ok"] is False
    assert report["reason"] == "focus_embedding_unavailable"
    assert report["nodes"] == []
    assert report["diagnostics"]["focus"]["has_embedding"] is False


def test_nearby_siblings_uses_sibling_position_not_order_delta() -> None:
    nodes = [
        {"_id": "a", "parent_id": "root", "order": 1},
        {"_id": "b", "parent_id": "root", "order": 4},
        {"_id": "c", "parent_id": "root", "order": 9},
    ]

    siblings = nearby_siblings(FakeDb(nodes), nodes[1], window=1)

    assert [sibling["_id"] for sibling in siblings] == ["a", "c"]


def test_graph_edges_for_node_serializes_adjacent_nodes() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    source_id = ObjectId()
    target_id = ObjectId()
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db = FakeDb(
        [
            {
                "_id": source_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Source",
                "text": "Source text",
            },
            {
                "_id": target_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Target",
                "text": "Target text",
            },
        ],
        [
            {
                "_id": ObjectId(),
                "schema_version": 1,
                "document_id": document_id,
                "tree_id": tree_id,
                "source_node_id": source_id,
                "target_node_id": target_id,
                "source_node_key": "source",
                "target_node_key": "target",
                "relation_type": "supports",
                "weight": 0.75,
                "confidence": 0.8,
                "direction": "directed",
                "provenance": {"source": "test"},
                "created_at": created_at,
                "updated_at": created_at,
            }
        ],
    )

    edges = graph_edges_for_node(db, str(source_id), direction="outgoing")

    assert len(edges) == 1
    assert edges[0]["relation_type"] == "supports"
    assert edges[0]["source_node_id"] == str(source_id)
    assert edges[0]["target_node_id"] == str(target_id)
    assert edges[0]["source_node"]["title"] == "Source"
    assert edges[0]["target_node"]["title"] == "Target"
    assert edges[0]["created_at"] == "2026-01-01T00:00:00+00:00"


def test_graph_edges_for_node_filters_relation_type_and_direction() -> None:
    node_id = ObjectId()
    other_id = ObjectId()
    db = FakeDb(
        [],
        [
            {
                "_id": ObjectId(),
                "source_node_id": node_id,
                "target_node_id": other_id,
                "relation_type": "supports",
            },
            {
                "_id": ObjectId(),
                "source_node_id": other_id,
                "target_node_id": node_id,
                "relation_type": "contradicts",
            },
        ],
    )

    edges = graph_edges_for_node(
        db,
        str(node_id),
        direction="incoming",
        relation_type="contradicts",
    )

    assert len(edges) == 1
    assert edges[0]["relation_type"] == "contradicts"


def test_expand_proximity_scores_adjacent_nodes() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    strong_id = ObjectId()
    weak_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
            },
            {
                "_id": strong_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Strong",
                "text": "Strong text",
            },
            {
                "_id": weak_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Weak",
                "text": "Weak text",
            },
        ],
        [
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": weak_id,
                "relation_type": "supports",
                "weight": 0.5,
                "confidence": 0.5,
            },
            {
                "_id": ObjectId(),
                "source_node_id": strong_id,
                "target_node_id": focus_id,
                "relation_type": "supports",
                "weight": 0.9,
                "confidence": 0.8,
                "provenance": {
                    "source": "semantic_candidate_review",
                    "reviewer": "tester",
                    "shared_label_count": 2,
                    "candidate_source": "embedding_similarity",
                    "embedding_similarity": 0.91,
                    "embedding_model": "mock",
                    "embedding_dimensions": 16,
                    "selection_context": {"min_similarity": 0.8},
                },
            },
        ],
    )

    nodes = expand_proximity(db, str(focus_id), relation_type="supports")

    assert [node["title"] for node in nodes] == ["Strong", "Weak"]
    assert [node["proximity_score"] for node in nodes] == [0.72, 0.25]
    assert nodes[0]["edge"]["source_node_id"] == str(strong_id)
    assert nodes[0]["edge"]["provenance_source"] == "semantic_candidate_review"
    assert nodes[0]["edge"]["reviewer"] == "tester"
    assert nodes[0]["edge"]["shared_label_count"] == 2
    assert nodes[0]["edge"]["candidate_source"] == "embedding_similarity"
    assert nodes[0]["edge"]["embedding_similarity"] == 0.91
    assert nodes[0]["edge"]["embedding_model"] == "mock"
    assert nodes[0]["edge"]["embedding_dimensions"] == 16
    assert nodes[0]["edge"]["selection_min_similarity"] == 0.8


def test_expand_proximity_prioritizes_reviewed_semantic_edges_over_contains() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    child_id = ObjectId()
    related_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
            },
            {
                "_id": child_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Structural Child",
                "text": "Child text",
            },
            {
                "_id": related_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "title": "Reviewed Meaning",
                "text": "Related text",
            },
        ],
        [
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": child_id,
                "relation_type": "contains",
                "weight": 1.0,
                "confidence": 1.0,
                "provenance": {"source": "node_parent_link"},
            },
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": related_id,
                "relation_type": "related_to",
                "weight": 0.6,
                "confidence": 0.7,
                "provenance": {
                    "source": "semantic_candidate_review",
                    "reviewer": "tester",
                },
            },
        ],
    )

    nodes = expand_proximity(db, str(focus_id), limit=1)

    assert [node["title"] for node in nodes] == ["Reviewed Meaning"]
    assert nodes[0]["proximity_score"] == 0.42
    assert nodes[0]["edge"]["relation_type"] == "related_to"
    assert nodes[0]["edge"]["provenance_source"] == "semantic_candidate_review"


def test_expand_proximity_returns_empty_for_bad_node_id() -> None:
    assert expand_proximity(FakeDb([]), "bad") == []


def test_expand_graph_paths_scores_bounded_multi_hop_paths() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    mid_id = ObjectId()
    strong_deep_id = ObjectId()
    weak_direct_id = ObjectId()
    cycle_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Focus",
                "text": "Focus text",
            },
            {
                "_id": mid_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Middle",
                "text": "Middle text",
            },
            {
                "_id": strong_deep_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Strong Deep",
                "text": "Deep text",
            },
            {
                "_id": weak_direct_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Weak Direct",
                "text": "Weak text",
            },
            {
                "_id": cycle_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "title": "Cycle",
                "text": "Cycle text",
            },
        ],
        [
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": mid_id,
                "relation_type": "supports",
                "weight": 0.9,
                "confidence": 0.9,
            },
            {
                "_id": ObjectId(),
                "source_node_id": mid_id,
                "target_node_id": strong_deep_id,
                "relation_type": "supports",
                "weight": 0.8,
                "confidence": 0.8,
                "provenance": {
                    "source": "semantic_candidate_review",
                    "reviewer": "tester",
                    "shared_label_count": 1,
                },
            },
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": weak_direct_id,
                "relation_type": "supports",
                "weight": 0.4,
                "confidence": 0.4,
            },
            {
                "_id": ObjectId(),
                "source_node_id": mid_id,
                "target_node_id": focus_id,
                "relation_type": "supports",
                "weight": 1.0,
                "confidence": 1.0,
            },
        ],
    )

    paths = expand_graph_paths(db, str(focus_id), direction="outgoing", max_depth=2)

    assert [path["title"] for path in paths] == ["Middle", "Strong Deep", "Weak Direct"]
    assert paths[0]["path_score"] == 0.81
    assert paths[1]["path_score"] == 0.5184
    assert paths[1]["path_depth"] == 2
    assert [edge["relation_type"] for edge in paths[1]["path_edges"]] == ["supports", "supports"]
    assert paths[1]["path_edges"][1]["provenance_source"] == "semantic_candidate_review"
    assert paths[1]["path_edges"][1]["reviewer"] == "tester"
    assert paths[1]["path_edges"][1]["shared_label_count"] == 1
    assert str(focus_id) not in [path["node_id"] for path in paths]


def test_expand_graph_paths_uses_natural_order_for_equal_structural_scores() -> None:
    document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    first_id = ObjectId()
    second_id = ObjectId()
    tenth_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": focus_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "node_key": "section-1",
                "title": "Section",
                "text": "Section text",
            },
            {
                "_id": tenth_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "node_key": "section-1-paragraph-10",
                "title": "Paragraph 10",
                "text": "Tenth text",
            },
            {
                "_id": second_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "node_key": "section-1-paragraph-2",
                "title": "Paragraph 2",
                "text": "Second text",
            },
            {
                "_id": first_id,
                "document_id": document_id,
                "tree_id": tree_id,
                "node_key": "section-1-paragraph-1",
                "title": "Paragraph 1",
                "text": "First text",
            },
        ],
        [
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": tenth_id,
                "relation_type": "contains",
                "weight": 1.0,
                "confidence": 1.0,
            },
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": second_id,
                "relation_type": "contains",
                "weight": 1.0,
                "confidence": 1.0,
            },
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": first_id,
                "relation_type": "contains",
                "weight": 1.0,
                "confidence": 1.0,
            },
        ],
    )

    paths = expand_graph_paths(db, str(focus_id), direction="outgoing")

    assert [path["title"] for path in paths] == [
        "Paragraph 1",
        "Paragraph 2",
        "Paragraph 10",
    ]


def test_expand_graph_paths_returns_empty_for_bad_node_id() -> None:
    assert expand_graph_paths(FakeDb([]), "bad") == []


def test_render_record_includes_role_title_and_source() -> None:
    block = render_record(
        {
            "role": "focus",
            "distance": 0,
            "title": "Title",
            "node_id": "node1",
            "labels": ["source_chunk"],
            "endorsement_label": "unreviewed",
            "provenance": {"archive_path": "archive.md"},
            "text_preview": "Body",
        }
    )

    assert "### focus d=0: Title" in block
    assert "archive.md" in block
    assert "Body" in block


def test_render_context_document_enforces_char_budget() -> None:
    context = {
        "document": {"title": "Doc", "document_id": "doc1"},
        "focus_node_id": "node1",
        "records": [
            {
                "role": "focus",
                "distance": 0,
                "title": "A",
                "node_id": "node1",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text_preview": "short",
            },
            {
                "role": "descendant",
                "distance": 1,
                "title": "B",
                "node_id": "node2",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text_preview": "x" * 1000,
            },
        ],
    }

    rendered = render_context_document(context, char_budget=350)

    assert len(rendered["included"]) == 1
    assert rendered["skipped"][0]["node_id"] == "node2"


def test_render_context_document_includes_skip_summaries() -> None:
    context = {
        "document": {"title": "Doc", "document_id": "doc1"},
        "focus_node_id": "node1",
        "records": [
            {
                "role": "focus",
                "distance": 0,
                "title": "A",
                "node_id": "node1",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text_preview": "short",
            },
            {
                "role": "descendant",
                "distance": 1,
                "title": "Stored section",
                "node_id": "node2",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text": "x" * 1000,
                "summary": "A vorton is a closed loop.",
                "summary_provenance": {"source": "operator"},
            },
            {
                "role": "descendant",
                "distance": 1,
                "title": "Derived section",
                "node_id": "node3",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text": "Its charge is an integer linking invariant. " * 40,
            },
        ],
    }

    rendered = render_context_document(context, char_budget=900)

    assert "Skipped under budget" in rendered["text"]
    assert "A vorton is a closed loop." in rendered["text"]
    by_id = {row["node_id"]: row for row in rendered["skipped"]}
    assert by_id["node2"]["included_as"] == "summary"
    assert by_id["node2"]["summary_source"] == "operator"
    assert by_id["node3"]["included_as"] == "summary"
    assert by_id["node3"]["summary_source"] == "derived_extractive"
    assert "integer linking invariant" in (by_id["node3"]["summary"] or "")


def test_prioritize_records_keeps_focus_before_descendants_and_ancestors() -> None:
    records = [
        {"role": "ancestor", "distance": 1, "title": "A"},
        {"role": "descendant", "distance": 1, "title": "B"},
        {"role": "focus", "distance": 0, "title": "C"},
    ]

    ordered = prioritize_records(records)

    assert [record["role"] for record in ordered] == ["focus", "descendant", "ancestor"]


def test_estimate_tokens_uses_four_char_approximation() -> None:
    assert estimate_tokens("12345") == 2


def test_build_prompt_envelope_includes_query_budget_and_context() -> None:
    context = {
        "document": {"title": "Doc", "document_id": "doc1"},
        "focus_node_id": "node1",
        "records": [
            {
                "role": "focus",
                "distance": 0,
                "title": "A",
                "node_id": "node1",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text_preview": "context body",
            },
        ],
    }

    envelope = build_prompt_envelope(
        context,
        query="What matters?",
        token_budget=400,
        reserved_response_tokens=25,
    )

    assert envelope["query"] == "What matters?"
    assert "context body" in envelope["prompt_text"]
    assert envelope["budget"]["available_context_tokens"] > 0
    assert envelope["budget"]["estimated_total_with_reserved_response_tokens"] <= 400


def _semantic_context():
    return {
        "document": {"title": "Doc", "document_id": "doc1"},
        "focus_node_id": "node1",
        "records": [{
            "role": "focus", "distance": 0, "title": "A", "node_id": "node1",
            "labels": [], "endorsement_label": "unreviewed", "provenance": {},
            "text_preview": "the form of the law",
        }],
    }


def test_build_prompt_envelope_default_has_no_semantic_block() -> None:
    env = build_prompt_envelope(_semantic_context(), query="what is form?", token_budget=200)
    assert env["semantic"] == [] and env["semantic_summary"] == ""
    assert "Semantic Precision" not in env["prompt_text"]  # default off = unchanged prompt


def test_build_prompt_envelope_with_resolver_injects_block_and_keys() -> None:
    from tirzah.semantic import SemanticLabel

    class Resolver:
        def resolve(self, terms):
            return [SemanticLabel(term="form", mpl_label="MPL-004",
                                  canonical_term="form (structure)", senses=["structural"],
                                  match_kind="exact")]

    env = build_prompt_envelope(_semantic_context(), query="what is form?",
                                token_budget=400, reserved_response_tokens=25, resolver=Resolver())
    assert "## Semantic Precision (MPL)" in env["prompt_text"]
    assert "[MPL-004]" in env["prompt_text"]
    assert env["semantic"][0]["mpl_label"] == "MPL-004"
    assert env["semantic_summary"] == "interpreted as: form→MPL-004 (structural)"
    assert "the form of the law" in env["context_text"]  # context still present


def test_build_prompt_envelope_surfaces_semantic_unavailable() -> None:
    class Resolver:
        last_error = None
        last_error_type = None

        def resolve(self, terms):
            self.last_error = "mahalath down"
            self.last_error_type = "ServerSelectionTimeoutError"
            return []

    env = build_prompt_envelope(
        _semantic_context(),
        query="what is form?",
        token_budget=400,
        reserved_response_tokens=25,
        resolver=Resolver(),
    )
    assert env["semantic"] == []
    assert env["semantic_status"] == "unavailable"
    assert env["semantic_diagnostic"]["reason"] == "mahalath_unavailable"


def test_render_context_document_uses_full_context_text() -> None:
    long_text = "A" * 350 + " full-tail"
    rendered = render_context_document(
        {
            "document": {"title": "Doc", "document_id": "doc1"},
            "focus_node_id": "node1",
            "records": [
                {
                    "role": "focus",
                    "distance": 0,
                    "title": "A",
                    "node_id": "node1",
                    "labels": [],
                    "endorsement_label": "unreviewed",
                    "provenance": {},
                    "text_preview": long_text[:300],
                    "text": long_text,
                },
            ],
        },
        char_budget=1000,
    )

    assert "full-tail" in rendered["text"]


def test_build_prompt_envelope_without_context_uses_no_context_instruction() -> None:
    envelope = build_prompt_envelope_without_context("What is stored?")

    assert envelope["system_instruction"] == default_no_context_system_instruction()
    # the instruction is clean + conversational and forbids process narration
    instruction = envelope["system_instruction"]
    assert "conversational assistant" in instruction
    assert "do not describe your internal process" in instruction.lower()
    # the prompt no longer dumps runtime-facts / process scaffolding into the model
    assert "Runtime Facts" not in envelope["prompt_text"]
    assert "Mongo lookup ran" not in envelope["prompt_text"]
    assert "behind-the-scenes" not in envelope["prompt_text"]
    assert "No stored memory matched this request" in envelope["prompt_text"]
    assert envelope["context_metadata"]["included"] == []


def test_node_search_score_demotes_empty_title_only_chunks() -> None:
    empty_chunk = {
        "title": "AMS Substrate Coherence Schema v1 / paragraph 1",
        "text": "",
        "labels": ["source_chunk"],
    }
    useful_section = {
        "title": "AMS Substrate Coherence Schema v1",
        "text": "Purpose and shared fields for the substrate coherence programme.",
        "labels": ["source_section"],
    }

    assert node_search_score(useful_section, "substrate coherence schema") > node_search_score(
        empty_chunk,
        "substrate coherence schema",
    )


def test_node_search_score_prefers_human_endorsed_nodes() -> None:
    unreviewed = {
        "title": "Memory note",
        "text": "memory context",
        "labels": ["source_chunk"],
        "endorsement_label": "unreviewed",
    }
    endorsed = {
        **unreviewed,
        "endorsement_label": "explicit_endorsed",
    }

    assert node_search_score(endorsed, "memory") > node_search_score(unreviewed, "memory")


def test_node_search_score_penalizes_rejected_nodes() -> None:
    unreviewed = {
        "title": "Memory note",
        "text": "memory context",
        "labels": ["source_chunk"],
        "endorsement_label": "unreviewed",
    }
    rejected = {
        **unreviewed,
        "endorsement_label": "rejected",
    }

    assert node_search_score(unreviewed, "memory") > node_search_score(rejected, "memory")


def test_node_search_score_penalizes_unreviewed_generated_output() -> None:
    source = {
        "title": "Memory note",
        "text": "memory context",
        "labels": ["source_chunk"],
        "endorsement_label": "unreviewed",
    }
    generated = {
        **source,
        "labels": ["source_chunk", "generated_output"],
    }

    assert node_search_score(source, "memory") > node_search_score(generated, "memory")


def test_node_search_score_adds_bounded_usage_signal() -> None:
    unused = {
        "title": "Memory note",
        "text": "memory context",
        "labels": ["source_chunk"],
        "endorsement_label": "unreviewed",
        "usage_score": 0,
    }
    heavily_used = {
        **unused,
        "usage_score": 50,
    }

    assert node_search_score(heavily_used, "memory") == node_search_score(unused, "memory") + 10


def test_usage_signal_does_not_outweigh_rejection_penalty() -> None:
    unreviewed = {
        "title": "Memory note",
        "text": "memory context",
        "labels": ["source_chunk"],
        "endorsement_label": "unreviewed",
    }
    rejected_used = {
        **unreviewed,
        "endorsement_label": "rejected",
        "usage_score": 100,
    }

    assert node_search_score(unreviewed, "memory") > node_search_score(rejected_used, "memory")


def test_node_search_sort_key_uses_last_used_at_after_score_ties() -> None:
    older = {
        "title": "Memory note",
        "text": "memory context",
        "labels": ["source_chunk"],
        "last_used_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    newer = {
        **older,
        "last_used_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
    }

    assert node_search_sort_key(newer, "memory") > node_search_sort_key(older, "memory")


def test_node_search_score_prefers_specific_chunk_over_large_container_section() -> None:
    query = "Taj Mahal commissioned Shah Jahan Mumtaz"
    large_section = {
        "title": "Taj Mahal - Wikipedia Extract",
        "text": "Taj Mahal commissioned Shah Jahan Mumtaz " + ("background " * 400),
        "labels": ["source_section"],
    }
    specific_chunk = {
        "title": "Taj Mahal - Wikipedia Extract / paragraph 6",
        "text": "The Taj Mahal was commissioned by Shah Jahan for Mumtaz Mahal.",
        "labels": ["source_chunk"],
    }

    assert node_search_score(specific_chunk, query) > node_search_score(
        large_section,
        query,
    )


def test_node_search_sort_key_breaks_score_ties_toward_compact_nodes() -> None:
    concise = {
        "title": "Taj Mahal - Wikipedia Extract / paragraph 6",
        "text": "The Taj Mahal was commissioned by Shah Jahan for Mumtaz Mahal.",
        "labels": ["source_chunk"],
    }
    broad = {
        **concise,
        "text": concise["text"] + (" Construction details." * 100),
    }

    assert node_search_score(concise, "Taj Mahal commissioned Mumtaz") == node_search_score(
        broad,
        "Taj Mahal commissioned Mumtaz",
    )
    assert node_search_sort_key(concise, "Taj Mahal commissioned Mumtaz") > node_search_sort_key(
        broad,
        "Taj Mahal commissioned Mumtaz",
    )


def _embedding(vector):
    return {"vector": vector, "dimensions": len(vector), "model": "test-embed"}


def test_attach_query_similarity_sets_cosine_for_comparable_nodes() -> None:
    query_embedding = _embedding([1.0, 0.0])
    nodes = [
        {"node_id": "match", "embedding": _embedding([1.0, 0.0])},
        {"node_id": "orthogonal", "embedding": _embedding([0.0, 1.0])},
        {"node_id": "wrong_model", "embedding": {"vector": [1.0, 0.0], "dimensions": 2, "model": "other"}},
        {"node_id": "no_embedding"},
    ]
    attach_query_similarity(nodes, query_embedding)
    by_id = {node["node_id"]: node for node in nodes}
    assert by_id["match"]["embedding_similarity"] == 1.0
    assert by_id["orthogonal"]["embedding_similarity"] == 0.0
    # incomparable model / missing embedding get no vector signal.
    assert "embedding_similarity" not in by_id["wrong_model"]
    assert "embedding_similarity" not in by_id["no_embedding"]


class _StubStore(MemoryStore):
    """A MemoryStore that returns fixed nodes with light filter evaluation."""

    def __init__(self, nodes):
        self._stub_nodes = nodes

    def find_nodes(self, filters, sort=None, limit=None):
        nodes = [dict(node) for node in self._stub_nodes if _stub_matches(node, filters or {})]
        return nodes[:limit] if limit else nodes


def _stub_matches(node, filters):
    for key, expected in filters.items():
        if key == "$or":
            if not any(_stub_option_matches(node, option) for option in expected):
                return False
            continue
        if key == "embedding.model" and isinstance(expected, dict) and "$exists" in expected:
            has_model = bool((node.get("embedding") or {}).get("model"))
            if has_model is not bool(expected["$exists"]):
                return False
            continue
        if key == "embedding.dimensions" and isinstance(expected, dict) and "$exists" in expected:
            has_dims = (node.get("embedding") or {}).get("dimensions") is not None
            if has_dims is not bool(expected["$exists"]):
                return False
            continue
        if isinstance(expected, dict) and "$nin" in expected:
            if node.get(key) in expected["$nin"]:
                return False
            continue
        if isinstance(expected, dict) and "$ne" in expected:
            if node.get(key) == expected["$ne"]:
                return False
            continue
        if hasattr(expected, "search"):
            if not expected.search(str(node.get(key) or "")):
                return False
            continue
        if node.get(key) != expected:
            return False
    return True


def _stub_option_matches(node, option):
    for field, expected in option.items():
        value = str(node.get(field) or "")
        if hasattr(expected, "search"):
            if expected.search(value):
                return True
        elif value == str(expected):
            return True
    return False


def test_search_nodes_hybrid_reranks_by_query_embedding() -> None:
    doc_id = ObjectId()
    store = _StubStore(
        [
            {
                "_id": ObjectId(),
                "document_id": doc_id,
                "tree_id": ObjectId(),
                "title": "Memory Systems Overview",
                "text": "memory",
                "labels": ["source_chunk"],
            },
            {
                "_id": ObjectId(),
                "document_id": doc_id,
                "tree_id": ObjectId(),
                "title": "Storage Internals",
                "text": "memory note",
                "labels": ["source_chunk"],
                "embedding": _embedding([1.0, 0.0]),
            },
        ]
    )

    lexical = [row["title"] for row in search_nodes(store, query="memory")]
    assert lexical == ["Memory Systems Overview", "Storage Internals"]

    hybrid = [
        row["title"]
        for row in search_nodes(store, query="memory", query_embedding=_embedding([1.0, 0.0]))
    ]
    # the vector-similar (but lexically weaker) node is lifted above the
    # lexically-stronger one once the query embedding is supplied.
    assert hybrid == ["Storage Internals", "Memory Systems Overview"]
    assert [row.get("match_source") for row in search_nodes(
        store, query="memory", query_embedding=_embedding([1.0, 0.0])
    )][0] in {"hybrid", "vector", "lexical"}


def test_search_nodes_includes_vector_only_matches() -> None:
    doc_id = ObjectId()
    store = _StubStore(
        [
            {
                "_id": ObjectId(),
                "document_id": doc_id,
                "tree_id": ObjectId(),
                "title": "Memory Systems Overview",
                "text": "memory",
                "labels": ["source_chunk"],
            },
            {
                "_id": ObjectId(),
                "document_id": doc_id,
                "tree_id": ObjectId(),
                "title": "Closed Loop",
                "text": "a superconducting cosmic string",
                "labels": ["source_chunk"],
                "embedding": _embedding([1.0, 0.0]),
            },
        ]
    )
    lexical = [row["title"] for row in search_nodes(store, query="memory")]
    assert lexical == ["Memory Systems Overview"]
    hybrid = search_nodes(store, query="memory", query_embedding=_embedding([1.0, 0.0]))
    titles = [row["title"] for row in hybrid]
    assert "Closed Loop" in titles
    vector_row = next(row for row in hybrid if row["title"] == "Closed Loop")
    assert vector_row["match_source"] == "vector"
    assert vector_row["hybrid_score"] is not None


def test_atlas_vector_search_falls_back_to_local_scan() -> None:
    doc_id = ObjectId()
    node = {
        "_id": ObjectId(),
        "document_id": doc_id,
        "tree_id": ObjectId(),
        "title": "Closed Loop",
        "text": "a superconducting cosmic string",
        "labels": ["source_chunk"],
        "embedding": _embedding([1.0, 0.0]),
    }
    store = _StubStore([node])

    class _Nodes:
        def aggregate(self, _pipeline):
            raise RuntimeError("no atlas")

    store.db = type("DB", (), {"nodes": _Nodes()})()
    rows = search_nodes(
        store,
        query="memory",
        query_embedding=_embedding([1.0, 0.0]),
        vector_search_index="node_embeddings",
    )
    assert [row["title"] for row in rows] == ["Closed Loop"]
    assert rows[0]["match_source"] == "vector"


def test_query_embedding_candidate_nodes_ranks_by_meaning() -> None:
    doc_id = ObjectId()

    def _node(title, vector=None, embedding=None, labels=None):
        node = {
            "_id": ObjectId(),
            "document_id": doc_id,
            "tree_id": ObjectId(),
            "title": title,
            "text": title,
            "labels": labels or ["source_chunk"],
        }
        if embedding is not None:
            node["embedding"] = embedding
        elif vector is not None:
            node["embedding"] = _embedding(vector)
        return node

    store = _StubStore(
        [
            _node("Closest", vector=[1.0, 0.0]),
            _node("Orthogonal", vector=[0.0, 1.0]),
            _node("Near", vector=[0.9, 0.1]),
            _node("No embedding"),  # excluded — no vector
            # incomparable model -> excluded
            _node("Wrong model", embedding={"vector": [1.0, 0.0], "dimensions": 2, "model": "other"}),
        ]
    )

    results = query_embedding_candidate_nodes(store, _embedding([1.0, 0.0]), limit=10)
    titles = [r["title"] for r in results]
    # ranked purely by cosine to the query; embedding-less / incomparable dropped.
    assert titles == ["Closest", "Near", "Orthogonal"]
    assert results[0]["embedding_similarity"] == 1.0
    assert results[0]["embedding_model"] == "test-embed"

    # a missing/invalid query embedding yields nothing (planner degrades to lexical).
    assert query_embedding_candidate_nodes(store, None) == []
    assert query_embedding_candidate_nodes(store, {"vector": []}) == []
