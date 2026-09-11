from datetime import datetime, timezone

from bson import ObjectId

from tirzah.db.memory_store import MemoryStore
from tirzah.retrieval.trust import (
    NEUTRAL_RECENCY,
    apply_trust_ranking,
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


def test_apply_trust_ranking_is_noop_when_disabled() -> None:
    rows = [
        {"node_id": "a", "hybrid_score": 0.5, "endorsement_label": "unreviewed"},
        {"node_id": "b", "hybrid_score": 0.5, "endorsement_label": "explicit_endorsed"},
    ]
    assert apply_trust_ranking(rows, enabled=False) == rows


def test_apply_trust_ranking_reorders_and_records_before_after() -> None:
    rows = [
        {"node_id": "a", "hybrid_score": 0.5, "endorsement_label": "unreviewed"},
        {"node_id": "b", "hybrid_score": 0.5, "endorsement_label": "explicit_endorsed"},
    ]
    ranked = apply_trust_ranking(rows, enabled=True, hybrid_weight=0.15)
    assert [row["node_id"] for row in ranked] == ["b", "a"]
    assert ranked[0]["rank_before"] == 1
    assert ranked[0]["rank_after"] == 0
    assert ranked[0]["trust_ranking"]["position_delta"] == 1
    assert ranked[0]["trust_ranking"]["ranking_score_after"] > ranked[0]["trust_ranking"]["ranking_score_before"]
    assert ranked[1]["trust_ranking"]["position_delta"] == -1


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


def test_temporal_recency_component_parses_iso_strings_and_neutralises_undated() -> None:
    # Issue #40: serialized rows carry ISO strings, and undated nodes used to score 1.0.
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)
    assert temporal_recency_component("2026-01-01T00:00:00+00:00", 10, now) == 0.5
    assert temporal_recency_component(None, 10, now) == NEUTRAL_RECENCY
    assert temporal_recency_component("not a date", 10, now) == NEUTRAL_RECENCY
    assert temporal_recency_component(None, None, now) == 1.0


def _stored_node(title: str, text: str, **extra) -> dict:
    return {
        "_id": ObjectId(),
        "document_id": ObjectId(),
        "tree_id": ObjectId(),
        "parent_id": None,
        "title": title,
        "text": text,
        "labels": ["source_chunk"],
        "endorsement_label": "unreviewed",
        "usage_score": 0,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        **extra,
    }


def test_search_trust_ranking_reads_trust_signals_from_stored_nodes() -> None:
    # Issue #40: serialize_node omits trust_score/verification flags, so the
    # ranker used to see every row as neutral and apply a zero boost.
    from tirzah.db.memory_store import as_memory_store
    from tirzah.retrieval.queries import apply_search_trust_ranking, serialize_node

    distrusted = _stored_node("A", "text", trust_score=0.05, verification_required=True)
    neutral = _stored_node("B", "text")
    rows = [{**serialize_node(node), "lexical_score": 10} for node in (distrusted, neutral)]
    assert "trust_score" not in rows[0]

    ranked = apply_search_trust_ranking(
        as_memory_store(FakeDb(nodes=[distrusted, neutral], profiles=[])),
        rows,
        enabled=True,
        profile_id=None,
        weight=1.0,
        max_boost=20,
        hybrid_weight=0.15,
    )
    by_id = {row["node_id"]: row["trust_ranking"] for row in ranked}
    assert by_id[str(distrusted["_id"])]["components"]["trust"] == 0.05
    assert by_id[str(distrusted["_id"])]["trust_boost"] < 0
    assert [row["node_id"] for row in ranked] == [str(neutral["_id"]), str(distrusted["_id"])]


class _PoolStore(MemoryStore):
    """Returns its whole node list for any filter (honours `_id $in` and limit)."""

    def __init__(self, nodes):
        super().__init__(db=None)
        self.pool = nodes

    def find_nodes(self, filters, *, sort=None, limit=None):
        wanted = filters.get("_id", {}).get("$in") if isinstance(filters.get("_id"), dict) else None
        rows = [node for node in self.pool if wanted is None or node["_id"] in wanted]
        return rows if limit is None else rows[:limit]


def test_search_nodes_trust_ranking_can_promote_from_beyond_limit() -> None:
    # Issue #42: re-ranking after truncation could never bring a trusted node in.
    from tirzah.retrieval.queries import search_nodes

    decoys = [
        _stored_node(f"Vorton vorton {index}", "vorton vorton vorton", trust_score=0.0)
        for index in range(10)
    ]
    trusted = _stored_node("Note", "a note that mentions vorton once", trust_score=1.0)
    store = _PoolStore([*decoys, trusted])

    # Lexical: decoys 80, trusted 22 — a boost wide enough to close that gap
    # isolates the truncation question from boost calibration.
    off = search_nodes(store, query="vorton", limit=5, trust_ranking_max_boost=60)
    on = search_nodes(
        store, query="vorton", limit=5, trust_ranking_enabled=True, trust_ranking_max_boost=60
    )

    assert str(trusted["_id"]) not in [row["node_id"] for row in off]
    assert len(on) == 5
    assert on[0]["node_id"] == str(trusted["_id"])


def test_seeded_default_profile_makes_recency_participate() -> None:
    # Issue #41: the only seeded profile had no half-life, so temporal was constant.
    from tirzah.db.indexes import DEFAULT_TRUST_WEIGHTING_PROFILES

    profile = DEFAULT_TRUST_WEIGHTING_PROFILES[0]
    now = datetime(2026, 9, 11, tzinfo=timezone.utc)
    old = trust_temporal_diagnostic({"origin_date": "1990-01-01"}, profile=profile, now=now)
    new = trust_temporal_diagnostic({"origin_date": "2026-09-01"}, profile=profile, now=now)
    assert new["components"]["recency"] > old["components"]["recency"]
    assert new["score"] > old["score"]
    ranked = apply_trust_ranking([{"node_id": "a", "lexical_score": 1}], enabled=True, profile=profile)
    assert ranked[0]["trust_ranking"]["temporal_decay_active"] is True


def test_apply_trust_ranking_normalises_mixed_hybrid_and_lexical_pools() -> None:
    # Issue #56: a lexical row (score 20) must not outrank a hybrid row (0.95)
    # just because the two scales were sorted together.
    rows = [
        {"node_id": "h", "hybrid_score": 0.95, "lexical_score": 40, "endorsement_label": "unreviewed"},
        {"node_id": "l", "lexical_score": 20, "endorsement_label": "unreviewed"},
    ]
    ranked = apply_trust_ranking(rows, enabled=True)
    assert [row["node_id"] for row in ranked] == ["h", "l"]
    scales = {row["node_id"]: row["trust_ranking"]["score_scale"] for row in ranked}
    assert scales == {"h": "hybrid", "l": "lexical_normalized"}
    assert ranked[1]["trust_ranking"]["ranking_score_after"] < 1.2


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
