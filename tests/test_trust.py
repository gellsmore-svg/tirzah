from datetime import datetime, timezone

from bson import ObjectId

from tirzah.retrieval.trust import (
    temporal_recency_component,
    trust_temporal_diagnostic,
    trust_temporal_diagnostic_for_node,
    trust_temporal_diagnostics_for_nodes,
)


def test_trust_temporal_diagnostic_reports_components_without_reranking() -> None:
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)
    node = {
        "_id": ObjectId(),
        "document_id": ObjectId(),
        "tree_id": ObjectId(),
        "parent_id": None,
        "title": "Governance note",
        "text": "Memory process note.",
        "labels": ["source_chunk"],
        "endorsement_label": "unreviewed",
        "trust_score": 0.7,
        "usage_score": 5,
        "verification_required": True,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    profile = {
        "weighting_profile_id": "diagnostic",
        "recency_importance": 0.3,
        "frequency_importance": 0.2,
        "stability_importance": 0.5,
        "verification_importance": 0.5,
        "default_decay_half_life_days": 10,
    }

    diagnostic = trust_temporal_diagnostic(node, profile=profile, now=now)

    assert diagnostic["components"] == {
        "trust": 0.7,
        "temporal": 0.75,
        "recency": 0.5,
        "frequency": 0.5,
        "verification": 0.0,
    }
    assert diagnostic["score"] == 0.5125
    assert diagnostic["signals"]["endorsement_label"] == "unreviewed"
    assert diagnostic["signals"]["origin_date"] is None


def test_trust_temporal_diagnostic_uses_origin_date_for_recency() -> None:
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)
    node = {
        "endorsement_label": "unreviewed",
        "created_at": datetime(2026, 1, 10, tzinfo=timezone.utc),
        "origin_date": "2026-01-01",
        "usage_score": 0,
    }
    profile = {"recency_importance": 1.0, "default_decay_half_life_days": 10}
    diagnostic = trust_temporal_diagnostic(node, profile=profile, now=now)
    assert diagnostic["signals"]["origin_date"] == "2026-01-01"
    assert diagnostic["components"]["recency"] == 0.5


def test_trust_temporal_diagnostic_for_node_fetches_profile() -> None:
    node_id = ObjectId()
    db = FakeDb(
        nodes=[
            {
                "_id": node_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "parent_id": None,
                "title": "Node",
                "text": "text",
                "labels": ["source_chunk"],
                "endorsement_label": "explicit_endorsed",
                "temporal_profile_id": "default_balanced",
                "usage_score": 0,
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ],
        profiles=[
            {
                "weighting_profile_id": "default_balanced",
                "recency_importance": 0.3,
                "frequency_importance": 0.2,
                "stability_importance": 0.5,
                "verification_importance": 0.5,
                "default_decay_half_life_days": None,
            }
        ],
    )

    result = trust_temporal_diagnostic_for_node(db, str(node_id))

    assert result is not None
    assert result["node"]["node_id"] == str(node_id)
    assert result["weighting_profile"]["weighting_profile_id"] == "default_balanced"
    assert result["diagnostic"]["components"]["trust"] == 1.0


def test_trust_temporal_diagnostic_for_node_rejects_invalid_id() -> None:
    assert trust_temporal_diagnostic_for_node(FakeDb(nodes=[], profiles=[]), "bad-id") is None


def test_trust_temporal_diagnostics_for_nodes_batches_node_lookup() -> None:
    node_id = ObjectId()
    other_id = ObjectId()
    db = FakeDb(
        nodes=[
            {
                "_id": node_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "parent_id": None,
                "title": "Node",
                "text": "text",
                "labels": ["source_chunk"],
                "endorsement_label": "explicit_endorsed",
                "usage_score": 0,
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            },
            {
                "_id": other_id,
                "document_id": ObjectId(),
                "tree_id": ObjectId(),
                "parent_id": None,
                "title": "Other",
                "text": "text",
                "labels": ["source_chunk"],
                "endorsement_label": "rejected",
                "usage_score": 0,
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            },
        ],
        profiles=[],
    )

    results = trust_temporal_diagnostics_for_nodes(db, [str(node_id), "bad-id", str(other_id)])

    assert set(results) == {str(node_id), str(other_id)}
    assert results[str(node_id)]["diagnostic"]["components"]["trust"] == 1.0
    assert results[str(other_id)]["diagnostic"]["components"]["trust"] == 0.0
    assert db.nodes.find_calls == 1


def test_temporal_recency_component_defaults_to_fresh_without_half_life() -> None:
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)

    assert temporal_recency_component(datetime(2026, 1, 1, tzinfo=timezone.utc), None, now) == 1.0


def test_temporal_recency_component_handles_naive_mongo_datetimes() -> None:
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)

    assert temporal_recency_component(datetime(2026, 1, 1), 10, now) == 0.5


class FakeCursor(list):
    def sort(self, *_args):
        return self

    def limit(self, value):
        return FakeCursor(self[:value])


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.find_calls = 0

    def find_one(self, query, projection=None):
        row = next((item for item in self.rows if matches(item, query)), None)
        if row is None:
            return None
        if projection and projection.get("_id") == 0:
            return {key: value for key, value in row.items() if key != "_id"}
        return dict(row)

    def find(self, query, projection=None):
        self.find_calls += 1
        rows = []
        for row in self.rows:
            if not matches(row, query):
                continue
            if projection and projection.get("_id") == 0:
                rows.append({key: value for key, value in row.items() if key != "_id"})
            else:
                rows.append(dict(row))
        return FakeCursor(rows)


class FakeDb:
    def __init__(self, nodes, profiles):
        self.nodes = FakeCollection(nodes)
        self.trust_weighting_profiles = FakeCollection(profiles)


def matches(row, query):
    for key, value in query.items():
        if isinstance(value, dict) and "$in" in value:
            if row.get(key) not in value["$in"]:
                return False
            continue
        if row.get(key) != value:
            return False
    return True
