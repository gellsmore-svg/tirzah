from datetime import datetime, timezone

import pytest
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from tirzah.sessions.exchanges import save_exchange
from tirzah.sessions.output_ingestion import (
    answer_output_text,
    list_output_ingestion_jobs,
    output_content_hash,
    output_job_to_ingestion_result,
    process_next_output_ingestion,
    queue_exchange_output,
)


def test_answer_output_text_strips_answer_body() -> None:
    assert answer_output_text({"answer": "  useful memory  "}) == "useful memory"
    assert answer_output_text({}) == ""


def test_output_content_hash_is_stable() -> None:
    assert output_content_hash("s1", "question", "answer") == output_content_hash(
        "s1",
        "question",
        "answer",
    )


def test_queue_exchange_output_stores_pending_job() -> None:
    db = FakeDb()

    job_id = queue_exchange_output(
        db,
        exchange_id="exchange1",
        session_id="session1",
        query="What changed?",
        answer={"answer": "A new memory.", "adapter": "mock", "model": "test"},
        used_node_ids=["node1"],
        active_document_ids=["doc1"],
    )

    assert job_id == str(db.output_ingestion_queue.rows[0]["_id"])
    row = db.output_ingestion_queue.rows[0]
    assert row["status"] == "pending"
    assert row["source_type"] == "llm_answer"
    assert row["exchange_id"] == "exchange1"
    assert row["answer_text"] == "A new memory."
    assert row["used_node_ids"] == ["node1"]
    assert row["active_document_ids"] == ["doc1"]


def test_queue_exchange_output_skips_empty_answer() -> None:
    db = FakeDb()

    assert queue_exchange_output(db, "exchange1", "session1", "q", {"answer": " "}, [], []) is None
    assert db.output_ingestion_queue.rows == []


def test_queue_exchange_output_returns_existing_job_on_duplicate_exchange() -> None:
    existing_id = ObjectId()
    db = FakeDb()
    db.output_ingestion_queue = DuplicateExchangeCollection(
        [
            {
                "_id": existing_id,
                "exchange_id": "exchange1",
                "status": "pending",
            }
        ]
    )

    job_id = queue_exchange_output(
        db,
        exchange_id="exchange1",
        session_id="session1",
        query="What changed?",
        answer={"answer": "A new memory."},
        used_node_ids=[],
        active_document_ids=[],
    )

    assert job_id == str(existing_id)
    assert len(db.output_ingestion_queue.rows) == 1


def test_save_exchange_links_output_ingestion_job() -> None:
    db = FakeDb()
    node_id = str(ObjectId())
    db.nodes.rows.append(
        {
            "_id": ObjectId(node_id),
            "endorsement_label": "unreviewed",
            "usage_score": 4,
        }
    )

    exchange_id = save_exchange(
        db,
        query="What should be remembered?",
        answer={"answer": "Remember this.", "used_node_ids": [node_id]},
        prompt={"budget": {}, "context_metadata": {}},
        focus_node_id=node_id,
        session_id="session1",
    )

    exchange = db.exchanges.rows[0]
    job = db.output_ingestion_queue.rows[0]
    assert exchange_id == str(exchange["_id"])
    assert exchange["output_ingestion_job_id"] == str(job["_id"])
    assert job["exchange_id"] == exchange_id
    assert job["answer_text"] == "Remember this."
    assert job["used_node_ids"] == [node_id]
    assert exchange["scored_node_count"] == 1
    assert exchange["project_domain_id"] == "tirzah"
    assert exchange["conversation_domain_id"] == "session1"
    assert db.project_domains.rows[0]["domain_id"] == "tirzah"
    assert db.conversation_domains.rows[0]["domain_id"] == "session1"
    assert db.conversation_domains.rows[0]["project_domain_id"] == "tirzah"
    assert db.nodes.rows[0]["usage_score"] == 5


def test_save_exchange_does_not_score_nodes_when_exchange_insert_fails() -> None:
    db = FakeDb()
    node_id = str(ObjectId())
    db.nodes.rows.append(
        {
            "_id": ObjectId(node_id),
            "endorsement_label": "unreviewed",
            "usage_score": 4,
        }
    )
    db.exchanges = FailingInsertCollection()

    with pytest.raises(RuntimeError):
        save_exchange(
            db,
            query="What should be remembered?",
            answer={"answer": "Remember this.", "used_node_ids": [node_id]},
            prompt={"budget": {}, "context_metadata": {}},
            focus_node_id=node_id,
            session_id="session1",
        )

    assert db.nodes.rows[0]["usage_score"] == 4


def test_save_exchange_records_score_count_before_output_queue_failure(monkeypatch) -> None:
    import tirzah.sessions.exchanges as exchanges

    db = FakeDb()
    node_id = str(ObjectId())
    db.nodes.rows.append(
        {
            "_id": ObjectId(node_id),
            "endorsement_label": "unreviewed",
            "usage_score": 4,
        }
    )

    def fail_queue(*_args, **_kwargs):
        raise RuntimeError("queue failed")

    monkeypatch.setattr(exchanges, "queue_exchange_output", fail_queue)

    with pytest.raises(RuntimeError):
        save_exchange(
            db,
            query="What should be remembered?",
            answer={"answer": "Remember this.", "used_node_ids": [node_id]},
            prompt={"budget": {}, "context_metadata": {}},
            focus_node_id=node_id,
            session_id="session1",
        )

    assert db.nodes.rows[0]["usage_score"] == 5
    assert db.exchanges.rows[0]["scored_node_count"] == 1
    assert "output_ingestion_job_id" not in db.exchanges.rows[0]


def test_save_exchange_continues_when_prompt_iteration_write_fails() -> None:
    db = FakeDb()
    db.session_continuity = FailingUpdateCollection()

    exchange_id = save_exchange(
        db,
        query="What should be remembered?",
        answer={"answer": "Remember this.", "used_node_ids": []},
        prompt={"budget": {}, "context_metadata": {}},
        focus_node_id=None,
        session_id="session1",
    )

    assert exchange_id == str(db.exchanges.rows[0]["_id"])
    assert db.exchanges.rows[0]["prompt_iteration_error"] == {
        "type": "RuntimeError",
        "message": "continuity failed",
    }


def test_list_output_ingestion_jobs_filters_and_serializes() -> None:
    db = FakeDb()
    now = datetime.now(timezone.utc)
    db.output_ingestion_queue.rows = [
        {
            "_id": ObjectId(),
            "schema_version": 1,
            "status": "pending",
            "source_type": "llm_answer",
            "exchange_id": "exchange1",
            "session_id": "s1",
            "query": "q",
            "answer_text": "A" * 600,
            "used_node_ids": [],
            "active_document_ids": [],
            "content_hash_sha256": "hash",
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": ObjectId(),
            "schema_version": 1,
            "status": "completed",
            "source_type": "llm_answer",
            "exchange_id": "exchange2",
            "session_id": "s2",
            "query": "q",
            "answer_text": "other",
            "created_at": now,
            "updated_at": now,
        },
    ]

    jobs = list_output_ingestion_jobs(db, status="pending", session_id="s1")

    assert len(jobs) == 1
    assert jobs[0]["exchange_id"] == "exchange1"
    assert len(jobs[0]["answer_preview"]) == 500


def test_output_job_to_ingestion_result_preserves_provenance() -> None:
    job = {
        "exchange_id": "exchange1",
        "session_id": "session1",
        "query": "What changed?",
        "answer_text": "A new memory.",
        "content_hash_sha256": "hash",
        "answer_adapter": "mock",
        "answer_model": "model",
        "used_node_ids": ["node1"],
        "active_document_ids": ["doc1"],
    }

    result = output_job_to_ingestion_result(job)

    assert result.title == "LLM Answer: What changed?"
    assert result.source.path == "tirzah://exchange/exchange1/answer"
    assert result.source.kind == "llm_answer"
    assert result.source.checksum_sha256 == "hash"
    assert result.tree_label == "llm_output"
    assert result.nodes[1].labels == ["generated_output", "llm_answer", "source_chunk"]
    assert result.nodes[1].metadata["exchange_id"] == "exchange1"
    assert result.nodes[1].metadata["used_node_ids"] == ["node1"]


def test_process_next_output_ingestion_commits_graph_records() -> None:
    db = FakeDb()
    exchange_id = ObjectId()
    db.exchanges.rows.append({"_id": exchange_id})
    queue_exchange_output(
        db,
        exchange_id=str(exchange_id),
        session_id="session1",
        query="What changed?",
        answer={"answer": "A new memory.", "adapter": "mock", "model": "test"},
        used_node_ids=[],
        active_document_ids=[],
    )

    result = process_next_output_ingestion(db)

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert len(db.documents.rows) == 1
    assert len(db.trees.rows) == 1
    assert len(db.nodes.rows) == 2
    assert db.documents.rows[0]["source"]["kind"] == "llm_answer"
    assert db.nodes.rows[1]["labels"] == ["generated_output", "llm_answer", "source_chunk"]
    assert db.output_ingestion_queue.rows[0]["status"] == "completed"
    assert db.exchanges.rows[0]["output_document_id"] == result["document_id"]
    assert db.active_documents.rows[0]["session_id"] == "session1"


def test_process_next_output_ingestion_can_target_session() -> None:
    db = FakeDb()
    older_exchange_id = ObjectId()
    target_exchange_id = ObjectId()
    db.exchanges.rows.append({"_id": older_exchange_id})
    db.exchanges.rows.append({"_id": target_exchange_id})
    queue_exchange_output(
        db,
        exchange_id=str(older_exchange_id),
        session_id="older",
        query="Older query?",
        answer={"answer": "Older answer.", "adapter": "mock", "model": "test"},
        used_node_ids=[],
        active_document_ids=[],
    )
    queue_exchange_output(
        db,
        exchange_id=str(target_exchange_id),
        session_id="target",
        query="Target query?",
        answer={"answer": "Target answer.", "adapter": "mock", "model": "test"},
        used_node_ids=[],
        active_document_ids=[],
    )

    result = process_next_output_ingestion(db, session_id="target")

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert db.output_ingestion_queue.rows[0]["status"] == "pending"
    assert db.output_ingestion_queue.rows[1]["status"] == "completed"
    assert db.active_documents.rows[0]["session_id"] == "target"


def test_process_next_output_ingestion_can_target_job_id() -> None:
    db = FakeDb()
    older_exchange_id = ObjectId()
    target_exchange_id = ObjectId()
    db.exchanges.rows.append({"_id": older_exchange_id})
    db.exchanges.rows.append({"_id": target_exchange_id})
    queue_exchange_output(
        db,
        exchange_id=str(older_exchange_id),
        session_id="older",
        query="Older query?",
        answer={"answer": "Older answer.", "adapter": "mock", "model": "test"},
        used_node_ids=[],
        active_document_ids=[],
    )
    target_job_id = queue_exchange_output(
        db,
        exchange_id=str(target_exchange_id),
        session_id="target",
        query="Target query?",
        answer={"answer": "Target answer.", "adapter": "mock", "model": "test"},
        used_node_ids=[],
        active_document_ids=[],
    )

    result = process_next_output_ingestion(db, job_id=target_job_id)

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["job_id"] == target_job_id
    assert db.output_ingestion_queue.rows[0]["status"] == "pending"
    assert db.output_ingestion_queue.rows[1]["status"] == "completed"


def test_process_next_output_ingestion_returns_idle_without_pending_jobs() -> None:
    assert process_next_output_ingestion(FakeDb()) == {"ok": True, "status": "idle"}


def test_process_next_output_ingestion_idle_includes_target_filter() -> None:
    assert process_next_output_ingestion(FakeDb(), session_id="missing") == {
        "ok": True,
        "status": "idle",
        "session_id": "missing",
    }


class FakeInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class FakeCursor(list):
    def sort(self, field, direction):
        reverse = direction < 0
        self[:] = sorted(self, key=lambda row: row.get(field), reverse=reverse)
        return self

    def limit(self, limit):
        return FakeCursor(self[:limit])


class FakeCollection:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def insert_one(self, row):
        row = dict(row)
        row["_id"] = ObjectId()
        self.rows.append(row)
        return FakeInsertResult(row["_id"])

    def find(self, query=None, projection=None):
        rows = [row for row in self.rows if matches(row, query or {})]
        return FakeCursor([project(row, projection) for row in rows])

    def find_one(self, query=None, projection=None):
        rows = self.find(query or {}, projection)
        return rows[0] if rows else None

    def find_one_and_update(self, filter_query, update, sort=None, return_document=None):
        rows = [row for row in self.rows if matches(row, filter_query)]
        if not rows:
            return None
        if sort:
            for field, direction in reversed(sort):
                rows.sort(key=lambda row: row.get(field), reverse=direction < 0)
        row = rows[0]
        apply_update(row, update)
        return dict(row)

    def update_one(self, filter_query, update, upsert=False):
        row = next((item for item in self.rows if matches(item, filter_query)), None)
        if row is None:
            if not upsert:
                return None
            row = dict(filter_query)
            row.update(update.get("$setOnInsert", {}))
            self.rows.append(row)
        apply_update(row, update)
        return None

    def update_many(self, filter_query, update):
        modified_count = 0
        for row in self.rows:
            if not matches(row, filter_query):
                continue
            apply_update(row, update)
            modified_count += 1
        return FakeUpdateResult(modified_count)

    def delete_many(self, query):
        self.rows = [row for row in self.rows if not matches(row, query)]
        return None

    def delete_one(self, query):
        for index, row in enumerate(self.rows):
            if matches(row, query):
                del self.rows[index]
                break
        return None


class DuplicateExchangeCollection(FakeCollection):
    def insert_one(self, row):
        if any(existing.get("exchange_id") == row.get("exchange_id") for existing in self.rows):
            raise DuplicateKeyError("duplicate exchange_id")
        return super().insert_one(row)


class FailingInsertCollection(FakeCollection):
    def insert_one(self, row):
        raise RuntimeError("insert failed")


class FailingUpdateCollection(FakeCollection):
    def update_many(self, filter_query, update):
        raise RuntimeError("continuity failed")


class FakeUpdateResult:
    def __init__(self, modified_count):
        self.modified_count = modified_count


class FakeDb:
    def __init__(self):
        self.sessions = FakeCollection()
        self.exchanges = FakeCollection()
        self.output_ingestion_queue = FakeCollection()
        self.documents = FakeCollection()
        self.trees = FakeCollection()
        self.nodes = FakeCollection()
        self.active_documents = FakeCollection()
        self.project_domains = FakeCollection()
        self.conversation_domains = FakeCollection()


def matches(row, query):
    for key, expected in query.items():
        actual = row.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif isinstance(expected, dict) and "$ne" in expected:
            if actual == expected["$ne"]:
                return False
        elif isinstance(expected, dict) and "$nin" in expected:
            if actual in expected["$nin"]:
                return False
        elif actual != expected:
            return False
    return True


def project(row, projection):
    if not projection:
        return dict(row)
    projected = {key: row[key] for key in projection if key in row}
    if "_id" in row and projection.get("_id", 1):
        projected["_id"] = row["_id"]
    return projected


def apply_update(row, update):
    row.update(update.get("$set", {}))
    for field, value in update.get("$inc", {}).items():
        row[field] = row.get(field, 0) + value
    for field, value in update.get("$addToSet", {}).items():
        row.setdefault(field, [])
        for item in value.get("$each", []):
            if item not in row[field]:
                row[field].append(item)
