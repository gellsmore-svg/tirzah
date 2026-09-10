from tirzah.ingestion.diff import diff_ingestion_trees, node_content_sha256
from tirzah.models.ingestion import IngestedNode


def _node(key, text, *, parent=None, title=None, label="source_chunk"):
    return IngestedNode(
        node_key=key,
        parent_key=parent,
        title=title or key,
        text=text,
        labels=[label],
    )


def test_diff_detects_changed_unchanged_added_and_removed() -> None:
    existing = [
        {
            "_id": "root",
            "node_key": "root",
            "parent_key": None,
            "title": "Doc",
            "text": "summary",
            "labels": ["source_root"],
            "status": "active",
            "content_sha256": node_content_sha256("summary"),
        },
        {
            "_id": "s1",
            "node_key": "section-1",
            "parent_key": "root",
            "title": "One",
            "text": "alpha",
            "labels": ["source_section"],
            "status": "active",
            "content_sha256": node_content_sha256("alpha"),
        },
        {
            "_id": "gone",
            "node_key": "section-2",
            "parent_key": "root",
            "title": "Two",
            "text": "beta",
            "labels": ["source_section"],
            "status": "active",
            "content_sha256": node_content_sha256("beta"),
        },
    ]
    proposed = [
        _node("root", "summary", title="Doc", label="source_root"),
        _node("section-1", "alpha changed", parent="root", title="One", label="source_section"),
        _node("section-3", "gamma", parent="root", title="Three", label="source_section"),
    ]
    diff = diff_ingestion_trees(existing, proposed)
    assert diff["counts"]["unchanged"] == 1
    assert diff["counts"]["changed"] == 1
    assert diff["counts"]["added"] == 1
    assert diff["counts"]["removed"] == 1
    assert diff["changed"][0]["node_id"] == "s1"
    assert diff["added"][0]["node_key"] == "section-3"
    assert diff["removed"][0]["node_key"] == "section-2"


def test_diff_matches_moved_chunk_by_content_hash() -> None:
    text = "A vorton is a closed loop."
    existing = [
        {
            "_id": "chunk",
            "node_key": "section-1-paragraph-1",
            "parent_key": "section-1",
            "title": "One / paragraph 1",
            "text": text,
            "labels": ["source_chunk"],
            "status": "active",
            "content_sha256": node_content_sha256(text),
        }
    ]
    proposed = [
        _node("section-2-paragraph-1", text, parent="section-2", title="Two / paragraph 1"),
    ]
    diff = diff_ingestion_trees(existing, proposed)
    assert diff["counts"]["moved"] == 1
    assert diff["moved"][0]["node_id"] == "chunk"
    assert diff["moved"][0]["node_key"] == "section-2-paragraph-1"
