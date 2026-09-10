from bson import ObjectId

from tirzah.retrieval.contradictions import (
    conflict_lexicon_matches,
    contradiction_candidate_nodes,
    contradiction_signals_for_pair,
    origin_date_delta_days,
    parse_origin_date,
)


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
        return FakeCursor([node for node in self.nodes if matches(node, query)])


class FakeDb:
    def __init__(self, nodes):
        self.nodes = FakeNodes(nodes)


def matches(row, query):
    for key, expected in query.items():
        if isinstance(expected, dict):
            value = nested_get(row, key)
            if "$exists" in expected and (value is not None) is not expected["$exists"]:
                return False
            if "$ne" in expected and value == expected["$ne"]:
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


def embedding(vector, model="mock"):
    return {"model": model, "dimensions": len(vector), "vector": vector}


def node(
    *,
    node_id=None,
    title="Node",
    text="Node body text for contradiction scoring.",
    labels=None,
    origin_date=None,
    document_id=None,
    tree_id=None,
    vector=None,
    provenance=None,
    node_key="node",
):
    return {
        "_id": node_id or ObjectId(),
        "document_id": document_id or ObjectId(),
        "tree_id": tree_id or ObjectId(),
        "node_key": node_key,
        "title": title,
        "text": text,
        "labels": labels or ["source_chunk"],
        "origin_date": origin_date,
        "origin_date_source": "filename" if origin_date else None,
        "origin_date_confidence": 0.8 if origin_date else None,
        "provenance": provenance or {},
        "embedding": embedding(vector or [1.0, 0.0, 0.0]),
    }


def test_conflict_lexicon_matches_whole_words() -> None:
    assert conflict_lexicon_matches("This claim is false, unlike the prior note.") == [
        "false",
        "unlike",
    ]
    assert conflict_lexicon_matches("notable notification") == []
    assert conflict_lexicon_matches("A vs B, however the later source is incorrect") == [
        "however",
        "incorrect",
        "vs",
    ]


def test_origin_date_delta_requires_both_dates() -> None:
    assert origin_date_delta_days("2020-01-01", "2024-01-02") == 1462
    assert origin_date_delta_days("2024-06-01", None) is None
    assert origin_date_delta_days(None, None) is None
    assert parse_origin_date("2024-06-01T12:00:00Z").isoformat() == "2024-06-01"


def test_contradiction_signals_require_two_cues() -> None:
    source = node(
        title="Claim",
        text="The protocol requires endorsement before retrieval.",
        labels=["source_chunk", "ams_domain"],
        origin_date="2020-01-01",
    )
    labeled_dated = node(
        title="Later claim",
        text="The protocol requires endorsement before retrieval in production.",
        labels=["source_chunk", "ams_domain"],
        origin_date="2024-01-01",
    )
    labeled_conflict = node(
        title="Denied claim",
        text="This is false: endorsement is not required before retrieval.",
        labels=["source_chunk", "ams_domain"],
        origin_date="2020-01-01",
    )
    dated_conflict = node(
        title="Refit",
        text="The later memo is contrary: endorsement is optional.",
        labels=["source_chunk"],
        origin_date="2024-06-01",
    )
    labels_only = node(
        title="Same era",
        text="The protocol still requires endorsement before retrieval.",
        labels=["source_chunk", "ams_domain"],
        origin_date="2020-01-01",
    )

    labeled_dated_signals = contradiction_signals_for_pair(
        source, labeled_dated, embedding_similarity=0.9
    )
    assert labeled_dated_signals["cue_count"] == 2
    assert labeled_dated_signals["meets_conservative_bar"] is True
    assert labeled_dated_signals["conflict_lexicon"] is False

    labeled_conflict_signals = contradiction_signals_for_pair(
        source, labeled_conflict, embedding_similarity=0.9
    )
    assert labeled_conflict_signals["shared_semantic_labels"] is True
    assert labeled_conflict_signals["conflict_lexicon"] is True
    assert labeled_conflict_signals["meets_conservative_bar"] is True

    dated_conflict_signals = contradiction_signals_for_pair(
        source, dated_conflict, embedding_similarity=0.88
    )
    assert dated_conflict_signals["shared_semantic_labels"] is False
    assert dated_conflict_signals["origin_date_delta"] is True
    assert dated_conflict_signals["conflict_lexicon"] is True
    assert dated_conflict_signals["meets_conservative_bar"] is True

    labels_only_signals = contradiction_signals_for_pair(
        source, labels_only, embedding_similarity=0.9
    )
    assert labels_only_signals["cue_count"] == 1
    assert labels_only_signals["meets_conservative_bar"] is False


def test_contradiction_signals_reject_near_duplicate_similarity() -> None:
    source = node(
        labels=["source_chunk", "ams_domain"],
        origin_date="2020-01-01",
        text="Endorsement is required. This claim is false otherwise.",
    )
    target = node(
        labels=["source_chunk", "ams_domain"],
        origin_date="2024-01-01",
        text="Endorsement is required. This later claim is false otherwise.",
    )
    signals = contradiction_signals_for_pair(source, target, embedding_similarity=0.99)
    assert signals["cue_count"] >= 2
    assert signals["similarity_in_band"] is False
    assert signals["meets_conservative_bar"] is False


def test_contradiction_candidate_nodes_keeps_high_bar_matches() -> None:
    document_id = ObjectId()
    other_document_id = ObjectId()
    tree_id = ObjectId()
    focus_id = ObjectId()
    accepted_id = ObjectId()
    near_copy_id = ObjectId()
    weak_id = ObjectId()
    one_cue_id = ObjectId()
    db = FakeDb(
        [
            node(
                node_id=focus_id,
                document_id=document_id,
                tree_id=tree_id,
                title="Focus claim",
                text="Endorsement is required before retrieval use.",
                labels=["source_chunk", "ams_domain"],
                origin_date="2020-01-01",
                provenance={"source_path": "archive/focus.md", "adapter": "mock"},
                vector=[1.0, 0.0, 0.0],
                node_key="focus",
            ),
            node(
                node_id=accepted_id,
                document_id=other_document_id,
                tree_id=tree_id,
                title="Later denial",
                text="The later memo is contrary: endorsement is not required before retrieval use.",
                labels=["source_chunk", "ams_domain"],
                origin_date="2024-06-01",
                provenance={"source_path": "archive/later.md", "adapter": "mock"},
                vector=[0.9, 0.4358898943540673, 0.0],
                node_key="accepted",
            ),
            node(
                node_id=near_copy_id,
                document_id=other_document_id,
                tree_id=tree_id,
                title="Near copy",
                text="Operators however restated the contrary rule: skip endorsement entirely.",
                labels=["source_chunk", "ams_domain"],
                origin_date="2024-06-01",
                vector=[0.99, 0.14106735979665886, 0.0],
                node_key="near",
            ),
            node(
                node_id=weak_id,
                document_id=other_document_id,
                tree_id=tree_id,
                title="Weak neighbor",
                text="The later memo is contrary: endorsement is not required before retrieval use at all.",
                labels=["source_chunk", "ams_domain"],
                origin_date="2024-06-01",
                vector=[0.7, 0.714142842854285, 0.0],
                node_key="weak",
            ),
            node(
                node_id=one_cue_id,
                document_id=other_document_id,
                tree_id=tree_id,
                title="Same-era overlap",
                text="Endorsement remains required before retrieval use in this handbook.",
                labels=["source_chunk", "ams_domain"],
                origin_date="2020-01-01",
                vector=[0.9, 0.4358898943540673, 0.0],
                node_key="one-cue",
            ),
        ]
    )

    candidates = contradiction_candidate_nodes(db, str(focus_id), min_similarity=0.82)

    assert [candidate["title"] for candidate in candidates] == ["Later denial"]
    candidate = candidates[0]
    assert candidate["node_id"] == str(accepted_id)
    assert candidate["relation_type"] == "contradicts"
    assert candidate["candidate_source"] == "contradiction_signals"
    assert candidate["source_origin_date"] == "2020-01-01"
    assert candidate["target_origin_date"] == "2024-06-01"
    assert candidate["origin_date_delta_days"] == 1613
    assert candidate["source_provenance"]["source_path"] == "archive/focus.md"
    assert candidate["target_provenance"]["source_path"] == "archive/later.md"
    signals = candidate["contradiction_signals"]
    assert signals["cue_count"] == 3
    assert signals["shared_semantic_labels"] is True
    assert signals["origin_date_delta"] is True
    assert signals["conflict_lexicon"] is True
    assert "contrary" in signals["conflict_terms"]
    assert 0.82 <= candidate["embedding_similarity"] < 0.97
