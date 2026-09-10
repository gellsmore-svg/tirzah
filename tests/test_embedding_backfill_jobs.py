from datetime import datetime, timezone

from bson import ObjectId

import tirzah.ingestion.embedding_backfill as embedding_backfill
from tirzah.ingestion.embedding_backfill import (
    create_embedding_backfill_job,
    list_embedding_backfill_jobs,
    process_embedding_backfill_batches,
    process_next_embedding_backfill_job,
    requeue_processing_embedding_backfill_job,
)


def test_create_embedding_backfill_job_persists_scope() -> None:
    db = FakeDb(nodes=[{"status": "active", "labels": ["ams_domain"]}])

    job = create_embedding_backfill_job(
        db,
        batch_limit=25,
        label="ams_domain",
        document_id="doc1",
        force=True,
        created_by="test",
    )

    assert job["status"] == "pending"
    assert job["batch_limit"] == 25
    assert job["label"] == "ams_domain"
    assert job["document_id"] == "doc1"
    assert job["force"] is True
    assert job["scope_total"] is None
    assert job["remaining_estimate"] is None
    assert job["progress_percent"] is None
    assert job["batches_remaining_estimate"] is None
    assert job["created_by"] == "test"


def test_create_embedding_backfill_job_counts_scope_total() -> None:
    db = FakeDb(
        nodes=[
            {"status": "active", "labels": ["target"]},
            {"status": "active", "labels": ["target"], "embedding": {"vector": [1.0]}},
            {"status": "active", "labels": ["other"]},
            {"status": "superseded", "labels": ["target"]},
        ]
    )

    job = create_embedding_backfill_job(db, batch_limit=25, label="target")

    assert job["scope_total"] == 1
    assert job["remaining_estimate"] == 1
    assert job["progress_percent"] == 0.0
    assert job["batches_remaining_estimate"] == 1


def test_embedding_backfill_job_progress_estimates_remaining_batches() -> None:
    db = FakeDb(
        nodes=[
            {"status": "active", "labels": ["target"]},
            {"status": "active", "labels": ["target"]},
            {"status": "active", "labels": ["target"]},
            {"status": "active", "labels": ["target"]},
            {"status": "active", "labels": ["target"]},
        ]
    )
    job = create_embedding_backfill_job(db, batch_limit=2, label="target")
    db.embedding_backfill_jobs.rows[0]["updated_count"] = 1

    serialized = list_embedding_backfill_jobs(db)[0]

    assert job["scope_total"] == 5
    assert serialized["remaining_estimate"] == 4
    assert serialized["progress_percent"] == 20.0
    assert serialized["batches_remaining_estimate"] == 2


def test_embedding_backfill_job_zero_scope_has_no_progress_percent() -> None:
    db = FakeDb(nodes=[{"status": "superseded", "labels": ["target"]}])

    job = create_embedding_backfill_job(db, batch_limit=2, label="target")

    assert job["scope_total"] == 0
    assert job["remaining_estimate"] == 0
    assert job["progress_percent"] is None
    assert job["batches_remaining_estimate"] == 0


def test_embedding_backfill_job_force_recovery_notes_rebuild_risk() -> None:
    db = FakeDb(nodes=[{"status": "active", "labels": ["target"], "embedding": {"vector": [1.0]}}])

    job = create_embedding_backfill_job(db, batch_limit=2, label="target", force=True)

    assert "force replace is enabled" in job["interruption_recovery"]


def test_process_next_embedding_backfill_job_keeps_full_batch_pending(monkeypatch) -> None:
    db = FakeDb()
    create_embedding_backfill_job(db, batch_limit=2)

    monkeypatch.setattr(
        embedding_backfill,
        "backfill_node_embeddings",
        lambda *_args, **_kwargs: {
            "ok": True,
            "matched_count": 2,
            "updated_count": 2,
            "skipped_count": 0,
            "error_count": 0,
            "last_node_id": "node2",
            "activity_log": "Text Similarity Profile Backfill Activity Log\n- Status: batch completed.",
        },
    )

    result = process_next_embedding_backfill_job(db, embedder="embedder")

    assert result["status"] == "pending"
    row = db.embedding_backfill_jobs.rows[0]
    assert row["status"] == "pending"
    assert row["batch_count"] == 1
    assert row["updated_count"] == 2
    assert serialize_first_job(db)["progress_percent"] is None
    assert row["last_node_id"] == "node2"
    assert row["last_result"]["activity_log"].startswith("Text Similarity Profile Backfill Activity Log")
    serialized = serialize_first_job(db)
    assert serialized["resume_after_node_id"] == "node2"
    assert serialized["transaction_boundary"] == "batch_cursor_after_completed_batch"
    assert serialized["node_write_boundary"] == "individual_node_update"
    assert "nodes already given profiles are skipped" in serialized["interruption_recovery"]


def test_process_next_embedding_backfill_job_completes_partial_batch(monkeypatch) -> None:
    db = FakeDb()
    create_embedding_backfill_job(db, batch_limit=5)

    monkeypatch.setattr(
        embedding_backfill,
        "backfill_node_embeddings",
        lambda *_args, **_kwargs: {
            "ok": True,
            "matched_count": 1,
            "updated_count": 1,
            "skipped_count": 0,
            "error_count": 0,
        },
    )

    result = process_next_embedding_backfill_job(db, embedder="embedder")

    assert result["status"] == "completed"
    assert db.embedding_backfill_jobs.rows[0]["status"] == "completed"


def test_process_next_embedding_backfill_job_keeps_forced_full_batch_pending(monkeypatch) -> None:
    db = FakeDb()
    create_embedding_backfill_job(db, batch_limit=2, force=True)

    monkeypatch.setattr(
        embedding_backfill,
        "backfill_node_embeddings",
        lambda *_args, **_kwargs: {
            "ok": True,
            "matched_count": 2,
            "updated_count": 2,
            "skipped_count": 0,
            "error_count": 0,
            "last_node_id": "node2",
        },
    )

    result = process_next_embedding_backfill_job(db, embedder="embedder")

    assert result["status"] == "pending"
    assert db.embedding_backfill_jobs.rows[0]["status"] == "pending"
    assert db.embedding_backfill_jobs.rows[0]["last_node_id"] == "node2"


def test_process_next_embedding_backfill_job_blocks_on_failure(monkeypatch) -> None:
    db = FakeDb()
    create_embedding_backfill_job(db, batch_limit=5)

    monkeypatch.setattr(
        embedding_backfill,
        "backfill_node_embeddings",
        lambda *_args, **_kwargs: {
            "ok": False,
            "reason": "all_embedding_updates_failed",
            "matched_count": 2,
            "updated_count": 0,
            "skipped_count": 2,
            "error_count": 2,
        },
    )

    result = process_next_embedding_backfill_job(db, embedder="embedder")

    assert result["status"] == "blocked"
    row = db.embedding_backfill_jobs.rows[0]
    assert row["status"] == "blocked"
    assert row["reason"] == "all_embedding_updates_failed"
    assert row["error_count"] == 2


def test_process_next_embedding_backfill_job_blocks_on_exception(monkeypatch) -> None:
    db = FakeDb()
    create_embedding_backfill_job(db, batch_limit=5)

    def raise_error(*_args, **_kwargs):
        raise RuntimeError("adapter offline")

    monkeypatch.setattr(embedding_backfill, "backfill_node_embeddings", raise_error)

    result = process_next_embedding_backfill_job(db, embedder="embedder")

    assert result["ok"] is False
    assert result["status"] == "blocked"
    row = db.embedding_backfill_jobs.rows[0]
    assert row["status"] == "blocked"
    assert row["reason"] == "embedding_backfill_exception"
    assert row["error_count"] == 1
    assert row["last_result"]["activity_log"].startswith("Text Similarity Profile Backfill Activity Log")


def test_process_embedding_backfill_batches_runs_until_requested_limit(monkeypatch) -> None:
    calls = []

    def fake_process(_db, _embedder):
        calls.append(len(calls))
        return {
            "ok": True,
            "status": "pending",
            "job_id": "job1",
            "result": {"updated_count": 2, "skipped_count": 1, "error_count": 0},
        }

    monkeypatch.setattr(embedding_backfill, "process_next_embedding_backfill_job", fake_process)

    result = process_embedding_backfill_batches(FakeDb(), embedder="embedder", max_batches=3)

    assert result["status"] == "pending"
    assert result["processed_batches"] == 3
    assert result["updated_count"] == 6
    assert result["skipped_count"] == 3


def test_process_embedding_backfill_batches_stops_on_completed(monkeypatch) -> None:
    results = [
        {"ok": True, "status": "pending", "result": {"updated_count": 1}},
        {"ok": True, "status": "completed", "result": {"updated_count": 1}},
        {"ok": True, "status": "pending", "result": {"updated_count": 1}},
    ]

    monkeypatch.setattr(
        embedding_backfill,
        "process_next_embedding_backfill_job",
        lambda _db, _embedder: results.pop(0),
    )

    result = process_embedding_backfill_batches(FakeDb(), embedder="embedder", max_batches=3)

    assert result["status"] == "completed"
    assert result["processed_batches"] == 2
    assert result["updated_count"] == 2


def test_process_embedding_backfill_batches_stops_on_blocked(monkeypatch) -> None:
    results = [
        {"ok": True, "status": "pending", "result": {"updated_count": 1}},
        {"ok": False, "status": "blocked", "result": {"error_count": 2, "reason": "adapter_offline"}},
        {"ok": True, "status": "pending", "result": {"updated_count": 1}},
    ]

    monkeypatch.setattr(
        embedding_backfill,
        "process_next_embedding_backfill_job",
        lambda _db, _embedder: results.pop(0),
    )

    result = process_embedding_backfill_batches(FakeDb(), embedder="embedder", max_batches=3)

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["processed_batches"] == 2
    assert result["updated_count"] == 1
    assert result["error_count"] == 2


def test_process_embedding_backfill_batches_reports_idle_queue(monkeypatch) -> None:
    monkeypatch.setattr(
        embedding_backfill,
        "process_next_embedding_backfill_job",
        lambda _db, _embedder: {"ok": True, "status": "idle"},
    )

    result = process_embedding_backfill_batches(FakeDb(), embedder="embedder", max_batches=5)

    assert result["ok"] is True
    assert result["status"] == "idle"
    assert result["processed_batches"] == 1
    assert result["updated_count"] == 0


def test_list_embedding_backfill_jobs_filters_and_serializes() -> None:
    db = FakeDb()
    db.embedding_backfill_jobs.rows = [
        {"_id": ObjectId(), "status": "completed", "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        {
            "_id": ObjectId(),
            "status": "pending",
            "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "last_result": {"activity_log": "Text Similarity Profile Backfill Activity Log\n- Status: pending."},
        },
    ]

    jobs = list_embedding_backfill_jobs(db, status="pending")

    assert len(jobs) == 1
    assert jobs[0]["status"] == "pending"
    assert isinstance(jobs[0]["job_id"], str)
    assert jobs[0]["scope_total"] is None
    assert jobs[0]["remaining_estimate"] is None
    assert jobs[0]["progress_percent"] is None
    assert jobs[0]["batches_remaining_estimate"] is None
    assert jobs[0]["last_result"]["activity_log"].startswith("Text Similarity Profile Backfill Activity Log")
    assert jobs[0]["resume_after_node_id"] is None
    assert "nodes already given profiles are skipped" in jobs[0]["interruption_recovery"]


def test_requeue_processing_embedding_backfill_job_marks_pending() -> None:
    db = FakeDb()
    job_id = ObjectId()
    db.embedding_backfill_jobs.rows = [
        {
            "_id": job_id,
            "status": "processing",
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "updated_count": 10,
            "scope_total": 20,
        }
    ]

    result = requeue_processing_embedding_backfill_job(
        db,
        str(job_id),
        reason="worker_restart",
        actor="tester",
    )

    assert result["ok"] is True
    assert result["job"]["status"] == "pending"
    assert result["job"]["reason"] == "worker_restart"
    assert result["job"]["requeued_by"] == "tester"
    assert result["job"]["progress_percent"] == 50.0
    assert result["job"]["resume_after_node_id"] is None
    assert db.embedding_backfill_jobs.rows[0]["status"] == "pending"


def test_requeue_processing_embedding_backfill_job_rejects_non_processing() -> None:
    db = FakeDb()
    job_id = ObjectId()
    db.embedding_backfill_jobs.rows = [
        {"_id": job_id, "status": "pending", "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    ]

    result = requeue_processing_embedding_backfill_job(db, str(job_id))

    assert result == {
        "ok": False,
        "status": "not_requeued",
        "reason": "job_not_processing_or_not_found",
        "job_id": str(job_id),
    }
    assert db.embedding_backfill_jobs.rows[0]["status"] == "pending"


def test_requeue_processing_embedding_backfill_job_rejects_invalid_id() -> None:
    result = requeue_processing_embedding_backfill_job(FakeDb(), "not-an-object-id")

    assert result == {
        "ok": False,
        "status": "invalid_job_id",
        "reason": "invalid_job_id",
        "job_id": "not-an-object-id",
    }


class FakeInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class FakeCursor(list):
    def sort(self, field, direction=-1):
        super().sort(key=lambda row: row.get(field), reverse=direction < 0)
        return self

    def limit(self, limit):
        return FakeCursor(self[:limit])


class FakeCollection:
    def __init__(self):
        self.rows = []

    def insert_one(self, row):
        row = dict(row)
        row["_id"] = ObjectId()
        self.rows.append(row)
        return FakeInsertResult(row["_id"])

    def find(self, query=None):
        return FakeCursor([dict(row) for row in self.rows if matches(row, query or {})])

    def find_one_and_update(self, filter_query, update, sort=None, return_document=None):
        rows = [row for row in self.rows if matches(row, filter_query)]
        if not rows:
            return None
        if sort:
            for field, direction in reversed(sort):
                rows.sort(key=lambda row: row.get(field), reverse=direction < 0)
        apply_update(rows[0], update)
        return dict(rows[0])

    def update_one(self, filter_query, update):
        row = next((row for row in self.rows if matches(row, filter_query)), None)
        if row:
            apply_update(row, update)
        return None


class FakeNodeCollection(FakeCollection):
    def count_documents(self, query):
        return len([row for row in self.rows if matches(row, query)])


class FakeDb:
    def __init__(self, nodes=None):
        self.embedding_backfill_jobs = FakeCollection()
        self.nodes = FakeNodeCollection()
        self.nodes.rows = list(nodes or [])


def matches(row, query):
    for key, value in query.items():
        row_value = dotted_value(row, key)
        if isinstance(value, dict):
            if "$ne" in value and row_value == value["$ne"]:
                return False
            if "$nin" in value and row_value in value["$nin"]:
                return False
            if "$exists" in value and (row_value is not None) != bool(value["$exists"]):
                return False
            if "$gt" in value and not row_value > value["$gt"]:
                return False
            continue
        if isinstance(row_value, list):
            if value not in row_value:
                return False
        elif row_value != value:
            return False
    return True


def dotted_value(row, key):
    value = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def serialize_first_job(db):
    return list_embedding_backfill_jobs(db)[0]


def apply_update(row, update):
    row.update(update.get("$set", {}))
    for field, value in update.get("$inc", {}).items():
        row[field] = row.get(field, 0) + value
