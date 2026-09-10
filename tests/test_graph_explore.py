from bson import ObjectId

from tirzah.retrieval.graph_explore import (
    graph_neighborhood,
    mermaid_id,
    render_graph_explore_mermaid,
    render_graph_explore_text,
)


class FakeCursor(list):
    def sort(self, *_args, **_kwargs):
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


def _node(node_id, title, **fields):
    document_id = fields.pop("document_id", ObjectId())
    tree_id = fields.pop("tree_id", ObjectId())
    return {
        "_id": node_id,
        "document_id": document_id,
        "tree_id": tree_id,
        "title": title,
        "text": fields.pop("text", f"{title} text"),
        "labels": fields.pop("labels", ["source_chunk"]),
        **fields,
    }


def test_graph_neighborhood_includes_one_and_two_hop_edges() -> None:
    focus_id = ObjectId()
    mid_id = ObjectId()
    deep_id = ObjectId()
    document_id = ObjectId()
    tree_id = ObjectId()
    db = FakeDb(
        [
            _node(
                focus_id,
                "Focus",
                document_id=document_id,
                tree_id=tree_id,
                origin_date="2020-01-01",
                endorsement_label="implicit_endorsed",
                provenance={"source_path": "archive/focus.md"},
            ),
            _node(
                mid_id,
                "Middle",
                document_id=document_id,
                tree_id=tree_id,
                origin_date="2021-01-01",
                endorsement_label="explicit_endorsed",
                provenance={"source_path": "archive/mid.md"},
            ),
            _node(
                deep_id,
                "Deep",
                document_id=document_id,
                tree_id=tree_id,
                origin_date="2024-01-01",
                endorsement_label="implicit_endorsed",
                provenance={"source_path": "archive/deep.md"},
            ),
        ],
        [
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": mid_id,
                "relation_type": "related_to",
                "weight": 0.8,
                "confidence": 0.7,
                "provenance": {"source": "semantic_candidate_review", "reviewer": "cello"},
            },
            {
                "_id": ObjectId(),
                "source_node_id": mid_id,
                "target_node_id": deep_id,
                "relation_type": "contradicts",
                "weight": 0.6,
                "confidence": 0.6,
                "provenance": {"source": "semantic_candidate_review", "reviewer": "cello"},
            },
        ],
    )

    report = graph_neighborhood(db, str(focus_id), max_depth=2, direction="outgoing")

    assert report["ok"] is True
    assert report["focus"]["title"] == "Focus"
    assert report["focus"]["origin_date"] == "2020-01-01"
    assert report["focus"]["provenance"]["source_path"] == "archive/focus.md"
    titles = [node["title"] for node in report["nodes"]]
    assert titles == ["Focus", "Middle", "Deep"]
    assert [edge["relation_type"] for edge in report["edges"]] == ["related_to", "contradicts"]
    assert report["edges"][0]["hop"] == 1
    assert report["edges"][1]["hop"] == 2
    assert report["edges"][0]["reviewer"] == "cello"

    text = render_graph_explore_text(report)
    assert "Graph neighborhood: Focus" in text
    assert "Middle | related_to | hop 1" in text
    assert "Deep | contradicts | hop 2" in text
    assert "provenance: archive/mid.md" in text
    assert "reviewed by: cello" in text

    mermaid = render_graph_explore_mermaid(report)
    assert mermaid.startswith("flowchart LR")
    assert f"{mermaid_id(str(focus_id))} -- related_to --> {mermaid_id(str(mid_id))}" in mermaid
    assert "contradicts" in mermaid
    assert f"style {mermaid_id(str(focus_id))}" in mermaid


def test_graph_neighborhood_skips_superseded_and_identity_and_endorsement() -> None:
    focus_id = ObjectId()
    kept_id = ObjectId()
    superseded_id = ObjectId()
    excluded_id = ObjectId()
    unendorsed_id = ObjectId()
    db = FakeDb(
        [
            _node(focus_id, "Focus", labels=["source_chunk", "ams_domain"]),
            _node(
                kept_id,
                "Kept",
                labels=["source_chunk", "ams_domain"],
                endorsement_label="explicit_endorsed",
            ),
            _node(
                superseded_id,
                "Old",
                labels=["source_chunk", "ams_domain"],
                status="superseded",
                endorsement_label="explicit_endorsed",
            ),
            _node(
                excluded_id,
                "Secret",
                labels=["source_chunk", "hidden"],
                endorsement_label="explicit_endorsed",
            ),
            _node(
                unendorsed_id,
                "Draft",
                labels=["source_chunk", "ams_domain"],
                endorsement_label="unreviewed",
            ),
        ],
        [
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": kept_id,
                "relation_type": "related_to",
            },
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": superseded_id,
                "relation_type": "related_to",
            },
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": excluded_id,
                "relation_type": "related_to",
            },
            {
                "_id": ObjectId(),
                "source_node_id": focus_id,
                "target_node_id": unendorsed_id,
                "relation_type": "related_to",
            },
        ],
    )

    report = graph_neighborhood(
        db,
        str(focus_id),
        max_depth=1,
        direction="outgoing",
        endorsement_label="explicit_endorsed",
        identity={"excluded_labels": ["hidden"]},
    )

    assert [node["title"] for node in report["nodes"]] == ["Focus", "Kept"]
    assert report["diagnostics"]["exclusions"]["superseded"] == 1
    assert report["diagnostics"]["exclusions"]["identity"] == 1
    assert report["diagnostics"]["exclusions"]["endorsement"] == 1


def test_graph_neighborhood_invalid_node() -> None:
    report = graph_neighborhood(FakeDb([]), "bad")
    assert report["ok"] is False
    assert report["reason"] == "invalid_node_id"
    assert "invalid node id" in render_graph_explore_text(report)
    assert render_graph_explore_mermaid(report).startswith("%%")


def test_mermaid_escapes_quotes_in_titles() -> None:
    report = {
        "ok": True,
        "focus": {"node_id": "abc123", "title": 'Say "hello"', "hop": 0},
        "nodes": [{"node_id": "abc123", "title": 'Say "hello"', "hop": 0}],
        "edges": [],
    }
    mermaid = render_graph_explore_mermaid(report)
    assert '["Say \'hello\'"]' in mermaid
