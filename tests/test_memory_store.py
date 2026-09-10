from datetime import datetime, timezone

from bson import ObjectId

from tirzah.db.memory_store import MemoryStore, as_memory_store


def test_as_memory_store_returns_existing_store() -> None:
    db = FakeDb()
    store = MemoryStore(db)

    assert as_memory_store(store) is store


def test_memory_store_wraps_common_memory_reads() -> None:
    document_id = ObjectId()
    node_id = ObjectId()
    child_id = ObjectId()
    db = FakeDb(
        documents=[
            {
                "_id": document_id,
                "title": "Document",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ],
        trees=[{"_id": ObjectId(), "document_id": document_id, "status": "active"}],
        nodes=[
            {"_id": node_id, "document_id": document_id, "parent_id": None, "order": 0},
            {"_id": child_id, "document_id": document_id, "parent_id": node_id, "order": 1},
        ],
    )
    store = MemoryStore(db)

    assert [row["_id"] for row in store.find_documents_by_ids([document_id])] == [document_id]
    assert store.get_document(document_id)["title"] == "Document"
    assert store.active_tree_count(document_id) == 1
    assert store.active_node_count(document_id) == 2
    assert store.get_node(node_id)["_id"] == node_id
    assert [node["_id"] for node in store.child_nodes(node_id)] == [child_id]
    assert store.update_nodes({"_id": {"$in": [child_id]}}, {"$set": {"usage_score": 1}}) == 1
    assert store.get_node(child_id)["usage_score"] == 1


def test_memory_store_records_retrieval_trace() -> None:
    db = FakeDb()
    trace_id = MemoryStore(db).record_retrieval_trace(
        {"request_id": "req1", "strategy": "direct"}
    )

    assert trace_id
    assert db.retrieval_traces.rows[0]["request_id"] == "req1"
    assert db.retrieval_traces.rows[0]["created_at"]


class FakeInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class FakeUpdateResult:
    def __init__(self, modified_count):
        self.modified_count = modified_count


class FakeCursor(list):
    def sort(self, key, direction):
        reverse = direction < 0
        return FakeCursor(sorted(self, key=lambda row: row.get(key), reverse=reverse))

    def limit(self, limit):
        return FakeCursor(self[:limit])


class FakeCollection:
    def __init__(self, rows=None):
        self.rows = rows or []

    def find(self, query):
        return FakeCursor([row for row in self.rows if matches(row, query)])

    def find_one(self, query):
        return next((row for row in self.rows if matches(row, query)), None)

    def count_documents(self, query):
        return len(self.find(query))

    def insert_one(self, row):
        inserted = {**row, "_id": row.get("_id") or ObjectId()}
        self.rows.append(inserted)
        return FakeInsertResult(inserted["_id"])

    def update_one(self, query, update):
        row = self.find_one(query)
        if not row:
            return FakeUpdateResult(0)
        row.update(update.get("$set") or {})
        return FakeUpdateResult(1)

    def update_many(self, query, update):
        modified_count = 0
        for row in self.find(query):
            row.update(update.get("$set") or {})
            modified_count += 1
        return FakeUpdateResult(modified_count)


class FakeDb:
    def __init__(self, documents=None, trees=None, nodes=None, graph_edges=None):
        self.documents = FakeCollection(documents)
        self.trees = FakeCollection(trees)
        self.nodes = FakeCollection(nodes)
        self.graph_edges = FakeCollection(graph_edges)
        self.retrieval_traces = FakeCollection()


def matches(row, query):
    for key, expected in query.items():
        value = row.get(key)
        if isinstance(expected, dict):
            if "$ne" in expected and value == expected["$ne"]:
                return False
            if "$nin" in expected and value in expected["$nin"]:
                return False
            if "$in" in expected and value not in expected["$in"]:
                return False
            continue
        if value != expected:
            return False
    return True
