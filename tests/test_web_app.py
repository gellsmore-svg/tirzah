from datetime import datetime, timezone
from pathlib import Path

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from tirzah.config import AppConfig, PathConfig, RuntimeConfig, load_config
from tirzah.db.client import get_database
from tirzah.web.app import (
    annotate_embedding_coverage,
    app,
    embedding_backfill_batch_failure_reason,
    embedding_backfill_batch_step_ids,
    embedding_backfill_status,
    embedding_coverage,
    list_ingestion_epochs,
    parse_ollama_model_list,
    parse_ollama_model_rows,
    profile_adapter_status,
    process_inbox_activity_log,
    recommended_embedding_backfill_job,
)
from tirzah.db.serializers import serialize_queue_job, serialize_queue_summary


class SimpleEpochDb:
    def __init__(self, documents: list[dict]):
        self.documents = SimpleEpochCollection(documents)


class SimpleEpochCollection:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def aggregate(self, pipeline):
        limit = pipeline[-1]["$limit"]
        grouped = {}
        for row in self.rows:
            epoch = row.get("ingestion_epoch")
            bucket = grouped.setdefault(
                epoch,
                {
                    "_id": epoch,
                    "document_count": 0,
                    "dated_document_count": 0,
                    "created_values": [],
                    "updated_values": [],
                    "origin_dates": [],
                },
            )
            bucket["document_count"] += 1
            bucket["created_values"].append(row.get("created_at"))
            bucket["updated_values"].append(row.get("updated_at"))
            origin_date = (row.get("source") or {}).get("origin_date")
            if origin_date is not None:
                bucket["dated_document_count"] += 1
                bucket["origin_dates"].append(origin_date)
        rows = []
        for bucket in grouped.values():
            origin_dates = bucket["origin_dates"]
            rows.append(
                {
                    "_id": bucket["_id"],
                    "document_count": bucket["document_count"],
                    "dated_document_count": bucket["dated_document_count"],
                    "first_created_at": min(bucket["created_values"]),
                    "last_updated_at": max(bucket["updated_values"]),
                    "earliest_origin_date": min(origin_dates) if origin_dates else "9999-12-31",
                    "latest_origin_date": max(origin_dates) if origin_dates else None,
                }
            )
        rows.sort(key=lambda row: row["last_updated_at"], reverse=True)
        return rows[:limit]


class SimpleEmbeddingDb:
    def __init__(self, nodes: list[dict]):
        self.nodes = SimpleEmbeddingCollection(nodes)


class SimpleEmbeddingCollection:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def count_documents(self, query):
        return len([row for row in self.rows if simple_match(row, query)])

    def aggregate(self, pipeline):
        match_query = pipeline[0]["$match"]
        grouped = {}
        for row in self.rows:
            if not simple_match(row, match_query):
                continue
            embedding = row.get("embedding") or {}
            key = (
                embedding.get("adapter"),
                embedding.get("model"),
                embedding.get("dimensions"),
            )
            grouped[key] = grouped.get(key, 0) + 1
        rows = [
            {
                "_id": {"adapter": key[0], "model": key[1], "dimensions": key[2]},
                "count": count,
            }
            for key, count in grouped.items()
        ]
        rows.sort(key=lambda row: row["count"], reverse=True)
        return rows[: pipeline[-1]["$limit"]]


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True




def test_ask_endpoint_uses_recursive_planning_wrapper(monkeypatch) -> None:
    client = TestClient(app)
    captured = {}

    def wrapped(_db, _config, **kwargs):
        captured.update(kwargs)
        return {"ok": True, "answer": "planned", "request_plan": {"revision": 2}}

    monkeypatch.setattr("tirzah.sessions.run.process_frontend_request", wrapped)
    response = client.post("/api/ask", json={
        "query": "Research X", "session_id": "s1", "retrieval_mode": "agentic",
        "recursive_planning": True, "web_research": True,
    })
    assert response.status_code == 200
    assert response.json()["request_plan"]["revision"] == 2
    assert captured["query"] == "Research X"
    assert captured["executor"] is not None
    assert captured["planning_enabled"] is True
    assert captured["web_research"] is True




def test_ask_endpoint_surfaces_plan_execution_fields(monkeypatch) -> None:
    client = TestClient(app)

    def wrapped(_db, _config, **kwargs):
        return {
            "ok": True,
            "answer": "planned answer",
            "session_id": "s1",
            "plan_execution": {"plan_id": "plan_test", "status": "completed", "step_count": 4},
            "context_bundle_summary": {"tools": ["search_nodes"], "tool_count": 1},
            "process_trace": [],
        }

    monkeypatch.setattr("tirzah.sessions.run.process_frontend_request", wrapped)
    response = client.post(
        "/api/ask",
        json={"query": "Q", "session_id": "s1", "recursive_planning": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["plan_execution"]["plan_id"] == "plan_test"
    assert payload["context_bundle_summary"]["tools"] == ["search_nodes"]


def test_ask_endpoint_returns_three_channel_contract(monkeypatch) -> None:
    client = TestClient(app)

    def wrapped(_db, _config, **kwargs):
        return {
            "ok": True,
            "answer": "A clean conversational answer.",
            "session_id": "s1",
            "answer_adapter": "mock",
            "answer_model": "gemma",
            "process_trace": [
                {"step": "user_prompt", "input": {"query": "Q"}, "output": {}},
                # output.status collides with emit()'s `status` kwarg — must be namespaced
                {"step": "retrieval_context", "input": {}, "output": {"node_count": 3, "status": "stable"}},
                {"step": "sufficiency", "input": {}, "output": {"context_sufficiency_score": 8.4, "recursion": 2}},
                {"step": "specialist_coherence", "input": {"mode": "coherence"},
                 "output": {"claims": 1, "objections": 2, "confidence": 0.6, "terminal_reason": "converged"}},
                {"step": "answer_adapter", "input": {"adapter": "mock", "model": "gemma"}, "output": {"ok": True}},
            ],
        }

    monkeypatch.setattr("tirzah.sessions.run.process_frontend_request", wrapped)
    response = client.post("/api/ask", json={"query": "Q", "session_id": "s1"})
    assert response.status_code == 200
    body = response.json()

    # answer channel stays a clean string; ids present
    assert body["answer"] == "A clean conversational answer."
    assert body["traceId"].startswith("trace_")
    assert body["sessionId"] == "s1"
    assert body["messageId"].startswith("msg_")
    assert body["requestId"].startswith("req_")

    # process channel: structured events, separate from the answer
    types = [e["type"] for e in body["processEvents"]]
    assert "message.user.submitted" in types  # bookend emitted at the boundary
    assert "process.started" in types
    assert "context.selected" in types  # translated from retrieval_context
    assert "model.response.completed" in types  # translated from answer_adapter
    assert "answer.finalized" in types and "process.completed" in types
    # the user_prompt step is not duplicated (boundary emits message.user.submitted)
    assert types.count("message.user.submitted") == 1
    # the final answer is logged on exactly the answer.finalized event
    finalized = next(e for e in body["processEvents"] if e["type"] == "answer.finalized")
    assert finalized["metadata"]["answer"] == "A clean conversational answer."
    # a step's own status is namespaced so it doesn't collide with the event status
    context_event = next(e for e in body["processEvents"] if e["type"] == "context.selected")
    assert context_event["status"] == "ok"
    assert context_event["metadata"]["step_status"] == "stable"
    # Phase 4 visibility: the sufficiency score surfaces as its own event
    sufficiency_event = next(e for e in body["processEvents"] if e["type"] == "context.sufficiency")
    assert sufficiency_event["metadata"]["context_sufficiency_score"] == 8.4
    assert sufficiency_event["summary"] == "Score Context Sufficiency"
    # specialist (Milcah) call surfaces as its own event
    specialist_event = next(e for e in body["processEvents"] if e["type"] == "specialist.completed")
    assert specialist_event["metadata"]["confidence"] == 0.6
    assert specialist_event["metadata"]["terminal_reason"] == "converged"
    assert specialist_event["summary"] == "Specialist Coherence Call"


def test_ask_events_conform_to_galeed_trace_contract(monkeypatch) -> None:
    # Executable seam contract: every trace event Tirzah emits must satisfy Galeed's
    # contract — a registered event type, the current schema_version, and the
    # correlation ids that let a trace be stitched across repos.
    import galeed

    client = TestClient(app)

    def wrapped(_db, _config, **kwargs):
        return {
            "ok": True,
            "answer": "ans",
            "session_id": "sX",
            "answer_adapter": "mock",
            "answer_model": "gemma",
            "process_trace": [
                {"step": "retrieval_context", "input": {}, "output": {"node_count": 2, "status": "stable"}},
                {"step": "sufficiency", "input": {}, "output": {"context_sufficiency_score": 7.0}},
                {"step": "answer_adapter", "input": {"adapter": "mock"}, "output": {"ok": True}},
            ],
        }

    monkeypatch.setattr("tirzah.sessions.run.process_frontend_request", wrapped)
    body = client.post("/api/ask", json={"query": "Q", "session_id": "sX"}).json()
    events = body["processEvents"]
    assert events, "expected emitted trace events"

    for event in events:
        # 1. no ad-hoc event strings — every type is in Galeed's vocabulary
        assert event["type"] in galeed.KNOWN_EVENT_TYPES, f"unregistered event type: {event['type']}"
        # 2. every event is stamped with the current schema version
        assert event["schema_version"] == galeed.SCHEMA_VERSION
        # 3. correlation ids tie each event back to the response envelope
        ids = galeed.correlation_ids(event)
        assert ids["trace_id"] == body["traceId"]
        assert ids["session_id"] == body["sessionId"]
    # events carry the same trace id and are monotonically sequenced
    assert {e["trace_id"] for e in body["processEvents"]} == {body["traceId"]}
    assert [e["seq"] for e in body["processEvents"]] == sorted(e["seq"] for e in body["processEvents"])


def test_feedback_endpoint_captures_against_trace() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/feedback",
        json={
            "text": "the answer leaked process scaffolding into the chat",
            "session_id": "s1",
            "trace_id": "trace_abc",
            "kind": "bug",
            "source": "claude",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["feedbackId"].startswith("fb_")
    assert body["traceId"] == "trace_abc"  # tied to the current trace
    assert body["feedback"]["text"].startswith("the answer leaked")
    assert body["feedback"]["kind"] == "bug"
    assert body["feedback"]["source"] == "claude"
    assert body["feedback"]["status"] == "open"


def test_feedback_list_endpoint(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "tirzah.web.app.list_feedback",
        lambda _db, **kwargs: [{"feedback_id": "fb_1", "text": "x", "session_id": kwargs.get("session_id")}],
    )
    response = client.get("/api/feedback?session_id=s1")
    assert response.status_code == 200
    assert response.json()["feedback"][0]["feedback_id"] == "fb_1"


def test_trace_sessions_endpoint(monkeypatch) -> None:
    client = TestClient(app)
    sample = [{"session_id": "s1", "event_count": 8, "sources": ["tirzah"], "trace_count": 2}]
    monkeypatch.setattr("tirzah.web.app.list_trace_sessions", lambda _db, **kwargs: sample)
    response = client.get("/api/trace/sessions")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["sessions"] == sample


def test_trace_events_endpoint_replays(monkeypatch) -> None:
    client = TestClient(app)
    sample = [{"type": "process.started", "trace_id": "t1", "seq": 1}]
    monkeypatch.setattr("tirzah.web.app.list_trace_events", lambda _db, **kwargs: sample)
    response = client.get("/api/trace/events?trace_id=t1")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["traceId"] == "t1"
    assert body["events"] == sample


def test_trace_stream_route_and_frame_format() -> None:
    # Avoid consuming the live (blocking) stream in-process; verify the route is
    # wired and the SSE frame format is correct. Live delivery is covered by the
    # TraceBus publish/subscribe unit tests.
    from tirzah.web.app import _sse_frame

    frame = _sse_frame({"type": "process.started", "trace_id": "t1", "seq": 1})
    # data-only frame: a single onmessage handler catches every event; type is in JSON
    assert frame.startswith("data: ")
    assert "event:" not in frame
    assert '"type": "process.started"' in frame
    assert '"trace_id": "t1"' in frame
    assert frame.endswith("\n\n")
    assert "/api/trace/stream" in {route.path for route in app.routes}


def test_plan_execution_endpoints(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "tirzah.web.app.list_plan_executions",
        lambda _db, session_id, status=None, limit=20: [
            {
                "plan_id": "plan1",
                "revision": 1,
                "session_id": session_id,
                "status": status or "running",
                "artifacts": {"retrieval_package": {"query": "q"}},
                "steps": [{"id": "1", "status": "completed"}, {"id": "2", "status": "pending"}],
                "completed_step_ids": ["1"],
            }
        ],
    )
    response = client.get("/api/plan-executions", params={"session_id": "s1", "status": "running", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["executions"][0]["plan_id"] == "plan1"
    assert body["executions"][0]["artifact_keys"] == ["retrieval_package"]
    assert body["executions"][0]["completed_count"] == 1
    assert "artifacts" not in body["executions"][0]

    monkeypatch.setattr(
        "tirzah.web.app.get_plan_execution",
        lambda _db, plan_id, revision, session_id: {
            "plan_id": plan_id,
            "revision": revision,
            "session_id": session_id,
            "status": "completed",
            "artifacts": {"answer": {"text": "done"}},
        },
    )
    response = client.get(
        "/api/plan-executions/plan1",
        params={"revision": 1, "session_id": "s1"},
    )
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == "completed"

    monkeypatch.setattr("tirzah.web.app.get_plan_execution", lambda *_args, **_kwargs: None)
    response = client.get(
        "/api/plan-executions/missing",
        params={"revision": 1, "session_id": "s1"},
    )
    assert response.status_code == 404


def test_plan_revision_endpoints(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "tirzah.web.app.list_plan_revisions",
        lambda _db, plan_id, limit=20: [{"plan_id": plan_id, "revision": 1, "limit": limit}],
    )
    response = client.get("/api/plans/plan1", params={"limit": 5})
    assert response.status_code == 200
    assert response.json()["latest"]["revision"] == 1

    class Revised:
        def to_dict(self):
            return {"plan_id": "plan1", "revision": 2, "status": "stable"}
    monkeypatch.setattr("tirzah.web.app.revise_saved_plan", lambda _db, _config, **kwargs: Revised())
    response = client.post(
        "/api/plans/plan1/revise",
        json={"new_information": {"fact": "new evidence"}, "session_id": "s1"},
    )
    assert response.status_code == 200
    assert response.json()["plan"]["revision"] == 2


def test_memory_health_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.memory_health_payload",
        lambda _db: {"ok": True, "totals": {"documents": 2}},
    )

    response = client.get("/api/memory-health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "totals": {"documents": 2}}


def test_session_continuity_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.session_continuity",
        lambda _db, *, session_id, limit: {
            "session_id": session_id,
            "limit": limit,
            "latest": {"exchange_id": "ex1"},
            "recent": [],
        },
    )

    response = client.get("/api/session-continuity", params={"session_id": "s1", "limit": 3})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "session_id": "s1",
        "limit": 3,
        "latest": {"exchange_id": "ex1"},
        "recent": [],
    }


def test_index_serves_single_ui() -> None:
    # The backend serves one UI (built Mahlah, or the not-built fallback) — the old
    # hand-rolled static UI is retired.
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Tirzah" in response.text
    assert '<div id="root">' in response.text or "has not been built yet" in response.text


def test_process_inbox_activity_log_prefers_human_summary() -> None:
    log = process_inbox_activity_log(
        [{"status": "pending"}, {"status": "rejected"}],
        [{"status": "completed", "activity_log": "Ingestion Activity Log\n- Status: completed."}],
    )

    assert log.startswith("Inbox Processing Activity Log")
    assert "Queue intake: 1 accepted, 1 rejected." in log
    assert "Run 1" in log
    assert "Status: completed." in log


def test_ingestion_status_endpoint_reports_epochs_and_runs(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_ingestion_epochs",
        lambda _db, limit=8: [
            {
                "ingestion_epoch": "epoch1",
                "document_count": 2,
                "dated_document_count": 1,
                "limit": limit,
            }
        ],
    )
    monkeypatch.setattr(
        "tirzah.web.app.list_process_runs",
        lambda _db, session_id=None, status=None, limit=20: [
            {"run_id": "run1", "session_id": session_id, "status": status, "limit": limit}
        ],
    )
    monkeypatch.setattr(
        "tirzah.web.app.embedding_coverage",
        lambda _db, label=None: {
            "total_active_nodes": 4,
            "embedded_active_nodes": 3,
            "missing_active_embeddings": 1,
            "embedded_percent": 75.0,
            "profiles": [],
            "label": label,
            "status": "incomplete",
            "summary": "3 of 4 active node(s) have text similarity profiles; 1 still need profiles.",
            "recommended_action": "Continue processing profile backfill job batches before relying on profile-based candidate review.",
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        "tirzah.web.app.embedding_backfill_status",
        lambda _db, coverage, **_kwargs: {
            "status": "pending",
            "summary": "1 recent profile backfill job(s) are queued.",
            "recommended_action": "Process the next bounded backfill batches and refresh ingestion status.",
            "recommended_job": {
                "batch_limit": 1,
                "force": False,
                "missing_embedding_only": True,
                "requires_real_adapter": False,
                "estimated_total_batches": 1,
                "recommended_web_batches": 1,
                "estimated_nodes_per_web_run": 1,
                "summary": "Queue a missing-profile job with batch limit 1; process up to 1 batch(es) per web run. Current coverage needs about 1 total batch(es).",
            },
            "recent_status_counts": {"pending": 1},
            "recent_jobs_checked": 1,
            "next_job": {"job_id": "job1", "status": "pending"},
        },
    )

    response = client.get("/api/ingestion/status", params={"limit": 4})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "epochs": [
            {
                "ingestion_epoch": "epoch1",
                "document_count": 2,
                "dated_document_count": 1,
                "limit": 4,
            }
        ],
        "embedding": {
            "total_active_nodes": 4,
            "embedded_active_nodes": 3,
            "missing_active_embeddings": 1,
            "embedded_percent": 75.0,
            "profiles": [],
            "label": None,
            "status": "incomplete",
            "summary": "3 of 4 active node(s) have text similarity profiles; 1 still need profiles.",
            "recommended_action": "Continue processing profile backfill job batches before relying on profile-based candidate review.",
            "warnings": [],
        },
        "embedding_backfill": {
            "status": "pending",
            "summary": "1 recent profile backfill job(s) are queued.",
            "recommended_action": "Process the next bounded backfill batches and refresh ingestion status.",
            "recommended_job": {
                "batch_limit": 1,
                "force": False,
                "missing_embedding_only": True,
                "requires_real_adapter": False,
                "estimated_total_batches": 1,
                "recommended_web_batches": 1,
                "estimated_nodes_per_web_run": 1,
                "summary": "Queue a missing-profile job with batch limit 1; process up to 1 batch(es) per web run. Current coverage needs about 1 total batch(es).",
            },
            "recent_status_counts": {"pending": 1},
            "recent_jobs_checked": 1,
            "next_job": {"job_id": "job1", "status": "pending"},
        },
        "runs": [{"run_id": "run1", "session_id": "ingestion", "status": None, "limit": 4}],
    }


def test_backfill_embeddings_endpoint_uses_configured_adapter(monkeypatch) -> None:
    client = TestClient(app)
    embedder = {"name": "fake"}
    updates = []

    monkeypatch.setattr("tirzah.web.app.embedding_adapter", lambda _runtime: embedder)
    monkeypatch.setattr(
        "tirzah.web.app.create_process_run",
        lambda _db, **kwargs: {"run_id": "run1", **kwargs},
    )
    monkeypatch.setattr(
        "tirzah.web.app.update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )
    monkeypatch.setattr(
        "tirzah.web.app.backfill_node_embeddings",
        lambda _db, used_embedder, **kwargs: {
            "ok": True,
            "adapter": used_embedder["name"],
            **kwargs,
        },
    )

    response = client.post(
        "/api/backfill-embeddings",
        json={
            "limit": 9,
            "label": "target",
            "document_id": "doc1",
            "force": True,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "adapter": "fake",
        "limit": 9,
        "label": "target",
        "document_id": "doc1",
        "force": True,
        "process_run_id": "run1",
        "process_status": "completed",
    }
    assert updates == [
        {
            "run_id": "run1",
            "status": "completed",
            "current_step_id": "embedding_backfill_completed",
            "completed_step_id": "embedding_backfill_batch",
            "exception": None,
        }
    ]


def test_backfill_embeddings_endpoint_marks_process_blocked_on_batch_failure(monkeypatch) -> None:
    client = TestClient(app)
    updates = []

    monkeypatch.setattr("tirzah.web.app.embedding_adapter", lambda _runtime: "embedder")
    monkeypatch.setattr(
        "tirzah.web.app.create_process_run",
        lambda _db, **kwargs: {"run_id": "run2", **kwargs},
    )
    monkeypatch.setattr(
        "tirzah.web.app.update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )
    monkeypatch.setattr(
        "tirzah.web.app.backfill_node_embeddings",
        lambda *_args, **_kwargs: {
            "ok": False,
            "reason": "all_embedding_updates_failed",
            "error_count": 2,
            "errors": [{"title": "Bad", "error": "failed"}],
        },
    )

    response = client.post("/api/backfill-embeddings", json={"limit": 2})

    assert response.status_code == 200
    assert response.json()["process_run_id"] == "run2"
    assert response.json()["process_status"] == "blocked"
    assert updates[0]["status"] == "blocked"
    assert updates[0]["current_step_id"] == "embedding_backfill_blocked"
    assert updates[0]["exception"]["reason"] == "all_embedding_updates_failed"
    assert updates[0]["exception"]["details"]["error_count"] == 2


def test_embedding_backfill_job_endpoints(monkeypatch) -> None:
    client = TestClient(app)
    updates = []

    monkeypatch.setattr(
        "tirzah.web.app.create_embedding_backfill_job",
        lambda _db, **kwargs: {"job_id": "job1", **kwargs},
    )
    monkeypatch.setattr(
        "tirzah.web.app.list_embedding_backfill_jobs",
        lambda _db, status=None, limit=10: [{"job_id": "job1", "status": status, "limit": limit}],
    )
    monkeypatch.setattr("tirzah.web.app.embedding_adapter", lambda _runtime: "embedder")
    monkeypatch.setattr(
        "tirzah.web.app.create_process_run",
        lambda _db, **kwargs: {"run_id": "run1", **kwargs},
    )
    monkeypatch.setattr(
        "tirzah.web.app.update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )
    monkeypatch.setattr(
        "tirzah.web.app.process_embedding_backfill_batches",
        lambda _db, embedder, max_batches=1: {
            "ok": True,
            "status": "completed",
            "requested_batches": max_batches,
            "processed_batches": 2,
            "updated_count": 2,
            "skipped_count": 0,
            "error_count": 0,
            "results": [
                {"ok": True, "status": "pending", "embedder": embedder},
                {"ok": True, "status": "completed", "embedder": embedder},
            ],
        },
    )
    monkeypatch.setattr(
        "tirzah.web.app.requeue_processing_embedding_backfill_job",
        lambda _db, job_id, reason, actor: {
            "ok": True,
            "status": "pending",
            "job": {
                "job_id": job_id,
                "status": "pending",
                "reason": reason,
                "requeued_by": actor,
            },
        },
    )

    created = client.post(
        "/api/embedding-backfill-jobs",
        json={"limit": 12, "label": "target", "document_id": "doc1", "force": True},
    )
    listed = client.get("/api/embedding-backfill-jobs", params={"status": "pending", "limit": 3})
    requeued = client.post(
        "/api/embedding-backfill-jobs/job1/requeue",
        json={"reason": "worker_restart", "actor": "tester"},
    )
    processed = client.post("/api/process-embedding-backfill-job", params={"max_batches": 4})

    assert created.status_code == 200
    assert created.json()["job"] == {
        "job_id": "job1",
        "batch_limit": 12,
        "label": "target",
        "document_id": "doc1",
        "force": True,
        "created_by": "web",
    }
    assert listed.json()["jobs"] == [{"job_id": "job1", "status": "pending", "limit": 3}]
    assert requeued.json() == {
        "ok": True,
        "status": "pending",
        "job": {
            "job_id": "job1",
            "status": "pending",
            "reason": "worker_restart",
            "requeued_by": "tester",
        },
    }
    assert processed.json() == {
        "ok": True,
        "status": "completed",
        "requested_batches": 4,
        "processed_batches": 2,
        "updated_count": 2,
        "skipped_count": 0,
        "error_count": 0,
        "results": [
            {"ok": True, "status": "pending", "embedder": "embedder"},
            {"ok": True, "status": "completed", "embedder": "embedder"},
        ],
        "process_run_id": "run1",
        "process_status": "completed",
    }
    assert [update["completed_step_id"] for update in updates] == [
        "embedding_backfill_job_batch_1_pending",
        "embedding_backfill_job_batch_2_completed",
        "embedding_backfill_job_run",
    ]
    assert updates[-1]["status"] == "completed"
    assert updates[-1]["current_step_id"] == "embedding_backfill_job_batch_processed"


def test_embedding_backfill_requeue_endpoint_rejects_non_processing_job(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.requeue_processing_embedding_backfill_job",
        lambda _db, job_id, reason, actor: {
            "ok": False,
            "status": "not_requeued",
            "reason": "job_not_processing_or_not_found",
            "job_id": job_id,
        },
    )

    response = client.post("/api/embedding-backfill-jobs/job1/requeue", json={})

    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "job_not_processing_or_not_found"


def test_embedding_backfill_batch_failure_reason_uses_batch_reason() -> None:
    result = {
        "results": [
            {"result": {"reason": "earlier_partial_issue"}},
            {"result": {"reason": "terminal_adapter_offline"}},
        ]
    }

    assert embedding_backfill_batch_failure_reason(result) == "terminal_adapter_offline"


def test_embedding_backfill_batch_step_ids_include_status_and_position() -> None:
    assert embedding_backfill_batch_step_ids(
        {
            "results": [
                {"status": "pending"},
                {"status": "blocked"},
            ]
        }
    ) == [
        "embedding_backfill_job_batch_1_pending",
        "embedding_backfill_job_batch_2_blocked",
    ]


def test_process_embedding_backfill_job_clamps_web_batch_count(monkeypatch) -> None:
    client = TestClient(app)
    calls = []

    monkeypatch.setattr("tirzah.web.app.embedding_adapter", lambda _runtime: "embedder")
    monkeypatch.setattr(
        "tirzah.web.app.create_process_run",
        lambda _db, **kwargs: {"run_id": "run1", **kwargs},
    )
    monkeypatch.setattr("tirzah.web.app.update_process_run", lambda _db, run_id, **kwargs: None)

    def fake_batches(_db, _embedder, max_batches=1):
        calls.append(max_batches)
        return {
            "ok": True,
            "status": "pending",
            "requested_batches": max_batches,
            "processed_batches": 1,
            "updated_count": 1,
            "skipped_count": 0,
            "error_count": 0,
            "results": [],
        }

    monkeypatch.setattr("tirzah.web.app.process_embedding_backfill_batches", fake_batches)

    response = client.post("/api/process-embedding-backfill-job", params={"max_batches": 50})

    assert response.status_code == 200
    assert calls == [10]
    assert response.json()["requested_batches"] == 10


def test_backfill_embeddings_endpoint_blocks_disallowed_embedding_adapter(monkeypatch) -> None:
    client = TestClient(app)
    updates = []

    monkeypatch.setattr(
        "tirzah.web.app.create_process_run",
        lambda _db, **kwargs: {"run_id": "run1", **kwargs},
    )
    monkeypatch.setattr(
        "tirzah.web.app.update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )

    def disallowed_adapter(_runtime):
        raise ValueError("HTTP-backed embedding adapter is not allowed.")

    monkeypatch.setattr("tirzah.web.app.embedding_adapter", disallowed_adapter)

    response = client.post("/api/backfill-embeddings", json={"limit": 2})

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["reason"] == "embedding_adapter_not_allowed"
    assert response.json()["process_status"] == "blocked"
    assert updates[0]["status"] == "blocked"
    assert updates[0]["exception"]["reason"] == "embedding_adapter_not_allowed"


def test_process_embedding_backfill_job_blocks_disallowed_embedding_adapter(monkeypatch) -> None:
    client = TestClient(app)
    updates = []

    monkeypatch.setattr(
        "tirzah.web.app.create_process_run",
        lambda _db, **kwargs: {"run_id": "run1", **kwargs},
    )
    monkeypatch.setattr(
        "tirzah.web.app.update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )

    def disallowed_adapter(_runtime):
        raise ValueError("HTTP-backed embedding adapter is not allowed.")

    monkeypatch.setattr("tirzah.web.app.embedding_adapter", disallowed_adapter)

    response = client.post("/api/process-embedding-backfill-job", params={"max_batches": 2})

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["status"] == "blocked"
    assert response.json()["reason"] == "embedding_adapter_not_allowed"
    assert response.json()["requested_batches"] == 2
    assert response.json()["processed_batches"] == 0
    assert updates[0]["status"] == "blocked"
    assert updates[0]["exception"]["reason"] == "embedding_adapter_not_allowed"


def test_list_ingestion_epochs_reports_dated_document_coverage() -> None:
    db = SimpleEpochDb(
        [
            {
                "ingestion_epoch": "epoch-new",
                "source": {"origin_date": "2020-01-01"},
                "created_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 6, 2, tzinfo=timezone.utc),
            },
            {
                "ingestion_epoch": "epoch-new",
                "source": {},
                "created_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 6, 3, tzinfo=timezone.utc),
            },
            {
                "ingestion_epoch": "epoch-old",
                "source": {},
                "created_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 5, 2, tzinfo=timezone.utc),
            },
        ]
    )

    epochs = list_ingestion_epochs(db, limit=5)

    assert epochs[0]["ingestion_epoch"] == "epoch-new"
    assert epochs[0]["document_count"] == 2
    assert epochs[0]["dated_document_count"] == 1
    assert epochs[0]["earliest_origin_date"] == "2020-01-01"
    assert epochs[0]["latest_origin_date"] == "2020-01-01"
    assert epochs[1]["ingestion_epoch"] == "epoch-old"
    assert epochs[1]["dated_document_count"] == 0
    assert epochs[1]["earliest_origin_date"] is None
    assert epochs[1]["latest_origin_date"] is None


def test_embedding_coverage_reports_active_embedding_readiness() -> None:
    db = SimpleEmbeddingDb(
        [
            {"status": "active", "labels": ["target"], "embedding": {"vector": [1.0]}},
            {"status": "active", "labels": ["target"]},
            {
                "status": "active",
                "labels": ["other"],
                "embedding": {
                    "adapter": "ollama_powershell_embedding",
                    "model": "nomic-embed-text:latest",
                    "dimensions": 768,
                    "vector": [1.0],
                },
            },
            {"status": "superseded", "labels": ["target"], "embedding": {"vector": [1.0]}},
        ]
    )

    coverage = embedding_coverage(db)
    target_coverage = embedding_coverage(db, label="target")

    assert coverage == {
        "total_active_nodes": 3,
        "embedded_active_nodes": 2,
        "missing_active_embeddings": 1,
        "embedded_percent": 66.7,
        "profiles": [
            {
                "adapter": None,
                "model": None,
                "dimensions": None,
                "count": 1,
                "is_mock": False,
            },
            {
                "adapter": "ollama_powershell_embedding",
                "model": "nomic-embed-text:latest",
                "dimensions": 768,
                "count": 1,
                "is_mock": False,
            },
        ],
        "label": None,
        "status": "incomplete",
        "summary": "2 of 3 active node(s) have text similarity profiles; 1 still need profiles.",
        "recommended_action": "Continue processing profile backfill job batches before relying on profile-based candidate review.",
        "warnings": [
            "2 profile representations are present; compare models and dimensions before broad profile-based review."
        ],
    }
    assert target_coverage == {
        "total_active_nodes": 2,
        "embedded_active_nodes": 1,
        "missing_active_embeddings": 1,
        "embedded_percent": 50.0,
        "profiles": [
            {
                "adapter": None,
                "model": None,
                "dimensions": None,
                "count": 1,
                "is_mock": False,
            }
        ],
        "label": "target",
        "status": "incomplete",
        "summary": "1 of 2 active node(s) for label `target` have text similarity profiles; 1 still need profiles.",
        "recommended_action": "Continue processing profile backfill job batches before relying on profile-based candidate review.",
        "warnings": [],
    }


def test_annotate_embedding_coverage_reports_operator_action_states() -> None:
    empty = annotate_embedding_coverage(
        {
            "total_active_nodes": 0,
            "embedded_active_nodes": 0,
            "missing_active_embeddings": 0,
            "embedded_percent": 0.0,
            "profiles": [],
            "label": None,
        }
    )
    not_started = annotate_embedding_coverage(
        {
            "total_active_nodes": 3,
            "embedded_active_nodes": 0,
            "missing_active_embeddings": 3,
            "embedded_percent": 0.0,
            "profiles": [],
            "label": None,
        }
    )
    mock_ready = annotate_embedding_coverage(
        {
            "total_active_nodes": 2,
            "embedded_active_nodes": 2,
            "missing_active_embeddings": 0,
            "embedded_percent": 100.0,
            "profiles": [{"adapter": "mock_embedding", "count": 2, "is_mock": True}],
            "label": None,
        }
    )
    ready = annotate_embedding_coverage(
        {
            "total_active_nodes": 2,
            "embedded_active_nodes": 2,
            "missing_active_embeddings": 0,
            "embedded_percent": 100.0,
            "profiles": [{"adapter": "ollama_powershell_embedding", "count": 2, "is_mock": False}],
            "label": None,
        }
    )

    assert empty["status"] == "empty"
    assert "Ingest source documents" in empty["recommended_action"]
    assert not_started["status"] == "not_started"
    assert "Queue a scoped profile backfill job" in not_started["recommended_action"]
    assert mock_ready["status"] == "mock_only"
    assert "forced profile backfill" in mock_ready["recommended_action"]
    assert ready["status"] == "ready"
    assert "Preview profile matches" in ready["recommended_action"]


def test_embedding_backfill_status_reports_pending_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        "tirzah.ingestion.status.list_embedding_backfill_jobs",
        lambda _db, limit=20: [
            {"job_id": "job1", "status": "pending", "batch_limit": 10},
            {"job_id": "job2", "status": "completed", "batch_limit": 10},
        ],
    )

    status = embedding_backfill_status(None, {"missing_active_embeddings": 5})

    assert status["status"] == "pending"
    assert status["recent_status_counts"] == {"pending": 1, "completed": 1}
    assert status["next_job"]["job_id"] == "job1"
    assert "Process the next bounded backfill batches" in status["recommended_action"]


def test_embedding_backfill_status_reports_blocked_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        "tirzah.ingestion.status.list_embedding_backfill_jobs",
        lambda _db, limit=20: [{"job_id": "job1", "status": "blocked", "reason": "failed"}],
    )

    status = embedding_backfill_status(None, {"missing_active_embeddings": 5})

    assert status["status"] == "blocked"
    assert status["next_job"]["job_id"] == "job1"
    assert "Open the latest job log" in status["recommended_action"]


def test_embedding_backfill_status_reports_needed_when_no_job_exists(monkeypatch) -> None:
    monkeypatch.setattr("tirzah.ingestion.status.list_embedding_backfill_jobs", lambda _db, limit=20: [])

    status = embedding_backfill_status(None, {"missing_active_embeddings": 5})

    assert status["status"] == "needed"
    assert status["next_job"] is None
    assert status["recommended_job"]["batch_limit"] == 5
    assert status["recommended_job"]["recommended_web_batches"] == 1
    assert "Queue a profile backfill job" in status["recommended_action"]


def test_embedding_backfill_status_blocks_recommendation_for_disallowed_adapter(monkeypatch) -> None:
    monkeypatch.setattr("tirzah.ingestion.status.list_embedding_backfill_jobs", lambda _db, limit=20: [])

    status = embedding_backfill_status(
        None,
        {"missing_active_embeddings": 5},
        embedding_adapter_allowed=False,
        configured_embedding_adapter="ollama_powershell",
    )

    assert status["status"] == "embedding_adapter_blocked"
    assert status["recommended_job"] is None
    assert "local non-HTTP profile adapter" in status["recommended_action"]
    assert "ollama_powershell" in status["summary"]


def test_embedding_backfill_status_blocks_missing_local_profile_command(monkeypatch) -> None:
    monkeypatch.setattr("tirzah.ingestion.status.list_embedding_backfill_jobs", lambda _db, limit=20: [])

    status = embedding_backfill_status(
        None,
        {"missing_active_embeddings": 5},
        embedding_adapter_allowed=True,
        configured_embedding_adapter="local_command",
        profile_adapter_status={"status": "missing_profile_command", "ready": False},
    )

    assert status["status"] == "profile_command_missing"
    assert status["recommended_job"] is None
    assert "runtime.profile_command" in status["recommended_action"]


def test_embedding_backfill_status_reports_not_needed_when_coverage_complete(monkeypatch) -> None:
    monkeypatch.setattr("tirzah.ingestion.status.list_embedding_backfill_jobs", lambda _db, limit=20: [])

    status = embedding_backfill_status(None, {"missing_active_embeddings": 0})

    assert status["status"] == "not_needed"
    assert status["recommended_job"] is None
    assert "profile-match preview" in status["recommended_action"]


def test_embedding_backfill_status_reports_real_backfill_needed_for_mock_coverage(monkeypatch) -> None:
    monkeypatch.setattr("tirzah.ingestion.status.list_embedding_backfill_jobs", lambda _db, limit=20: [])

    status = embedding_backfill_status(
        None,
        {
            "status": "mock_only",
            "total_active_nodes": 25,
            "embedded_active_nodes": 25,
            "missing_active_embeddings": 0,
        },
    )

    assert status["status"] == "real_backfill_needed"
    assert status["recommended_job"]["force"] is True
    assert status["recommended_job"]["missing_embedding_only"] is False
    assert status["recommended_job"]["requires_real_adapter"] is True
    assert "model-backed profile adapter" in status["recommended_action"]


def test_recommended_embedding_backfill_job_estimates_large_corpus() -> None:
    recommendation = recommended_embedding_backfill_job({"missing_active_embeddings": 176428})

    assert recommendation == {
        "batch_limit": 25,
        "force": False,
        "missing_embedding_only": True,
        "requires_real_adapter": False,
        "estimated_total_batches": 7058,
        "recommended_web_batches": 10,
        "estimated_nodes_per_web_run": 250,
        "summary": "Queue a missing-profile job with batch limit 25; process up to 10 batch(es) per web run. Current coverage needs about 7058 total batch(es).",
    }


def test_recommended_embedding_backfill_job_uses_configured_limits() -> None:
    recommendation = recommended_embedding_backfill_job(
        {"missing_active_embeddings": 90},
        recommended_batch_limit=30,
        web_max_batches=2,
    )

    assert recommendation["batch_limit"] == 30
    assert recommendation["recommended_web_batches"] == 2
    assert recommendation["estimated_nodes_per_web_run"] == 60
    assert recommendation["estimated_total_batches"] == 3


def test_recommended_embedding_backfill_job_skips_complete_coverage() -> None:
    assert recommended_embedding_backfill_job({"missing_active_embeddings": 0}) is None


def test_recommended_embedding_backfill_job_handles_mock_only_coverage() -> None:
    recommendation = recommended_embedding_backfill_job(
        {
            "status": "mock_only",
            "total_active_nodes": 2500,
            "embedded_active_nodes": 2500,
            "missing_active_embeddings": 0,
        }
    )

    assert recommendation == {
        "batch_limit": 25,
        "force": True,
        "missing_embedding_only": False,
        "requires_real_adapter": True,
        "estimated_total_batches": 100,
        "recommended_web_batches": 10,
        "estimated_nodes_per_web_run": 250,
        "summary": "After configuring a local model-backed profile adapter, queue a forced backfill with batch limit 25; process up to 10 batch(es) per web run. Current coverage needs about 100 total forced batch(es).",
    }


def test_index_serves_html() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Tirzah" in response.text


def test_queue_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/queue")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_serialize_queue_summary_does_not_mutate_source_row() -> None:
    job_id = ObjectId()
    timestamp = datetime(2026, 6, 14, tzinfo=timezone.utc)
    oldest_pending = {
        "_id": job_id,
        "path": "data/ingest/source.md",
        "created_at": timestamp,
        "attempts": 0,
    }
    summary = {
        "total": 1,
        "statuses": {"pending": 1},
        "oldest_pending": oldest_pending,
    }

    serialized = serialize_queue_summary(summary)

    assert serialized["oldest_pending"]["_id"] == str(job_id)
    assert serialized["oldest_pending"]["created_at"] == "2026-06-14T00:00:00+00:00"
    assert summary["oldest_pending"]["_id"] == job_id
    assert summary["oldest_pending"]["created_at"] == timestamp


def test_queue_endpoint_serializes_without_mutating_summary(monkeypatch) -> None:
    job_id = ObjectId()
    timestamp = datetime(2026, 6, 14, tzinfo=timezone.utc)
    summary = {
        "total": 1,
        "statuses": {"pending": 1},
        "oldest_pending": {
            "_id": job_id,
            "path": "data/ingest/source.md",
            "created_at": timestamp,
            "attempts": 0,
        },
    }

    monkeypatch.setattr("tirzah.web.app.queue_summary", lambda _db: summary)

    response = TestClient(app).get("/api/queue")

    assert response.status_code == 200
    assert response.json()["oldest_pending"]["_id"] == str(job_id)
    assert response.json()["oldest_pending"]["created_at"] == "2026-06-14T00:00:00+00:00"
    assert summary["oldest_pending"]["_id"] == job_id
    assert summary["oldest_pending"]["created_at"] == timestamp


def test_runtime_endpoint_lists_llm_controls() -> None:
    client = TestClient(app)

    response = client.get("/api/runtime")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "ollama_cli" in data["available_adapters"]
    assert "hoglah" in data["available_adapters"]
    assert data["resolved_runtime"]["adapters"]["answer"]
    assert data["available_embedding_adapters"] == ["mock", "local_command"]
    assert "ollama_http" in data["non_compliant_embedding_adapters"]
    assert "ollama_powershell" in data["non_compliant_embedding_adapters"]
    assert data["embedding_adapter_policy"] == "ingestion_and_retrieval_no_http"
    assert data["profile_adapter_status"]["status"] in {
        "stub_profile_adapter",
        "http_adapter_blocked",
        "missing_profile_command",
        "ready",
    }
    assert data["memory_agent_adapter_policy"] == "local_only_no_http"
    assert data["default_model"]
    assert data["default_embedding_adapter"]
    assert data["default_embedding_model"]
    assert data["profile_backfill_recommended_batch_limit"] == 25
    assert data["profile_backfill_web_max_batches"] == 10
    assert data["memory_agent_model"]
    assert "gemma4:latest" in data["known_models"]
    assert data["model_options"]


def test_profile_adapter_status_reports_ready_and_blocked_states() -> None:
    assert profile_adapter_status(RuntimeConfig(embedding_adapter="mock"))["status"] == "stub_profile_adapter"
    assert (
        profile_adapter_status(RuntimeConfig(embedding_adapter="ollama_powershell"))["status"]
        == "http_adapter_blocked"
    )
    assert (
        profile_adapter_status(RuntimeConfig(embedding_adapter="local_command"))["status"]
        == "missing_profile_command"
    )
    ready = profile_adapter_status(
        RuntimeConfig(
            embedding_adapter="local_command",
            profile_command=["profile-tool"],
        )
    )
    assert ready["status"] == "ready"
    assert ready["command_configured"] is True


def test_parse_ollama_model_list_returns_model_names() -> None:
    output = """NAME                    ID              SIZE      MODIFIED
gemma4:26b              5571076f3d70    17 GB     16 hours ago
qwen3.5:35b             3460ffeede54    23 GB     17 hours ago
mistral-small:latest    8039dd90c113    14 GB     17 hours ago
"""

    assert parse_ollama_model_list(output) == [
        "qwen3.5:35b",
        "gemma4:26b",
        "mistral-small:latest",
    ]


def test_parse_ollama_model_rows_sorts_largest_first_and_labels_size() -> None:
    output = """NAME                    ID              SIZE      MODIFIED
small:latest            aaa             815 MB    1 day ago
large:latest            bbb             23 GB     1 day ago
medium:latest           ccc             7.2 GB    1 day ago
"""

    rows = parse_ollama_model_rows(output)

    assert [row["name"] for row in rows] == [
        "large:latest",
        "medium:latest",
        "small:latest",
    ]
    assert [row["size_category"] for row in rows] == ["large", "medium", "small"]


def test_session_endpoints() -> None:
    client = TestClient(app)

    created = client.post(
        "/api/sessions",
        json={"title": "Web Test Session", "session_id": "web-test-session"},
    )
    listed = client.get("/api/sessions")

    assert created.status_code == 200
    assert created.json()["session"]["session_id"] == "web-test-session"
    assert listed.status_code == 200
    assert any(
        session["session_id"] == "web-test-session"
        for session in listed.json()["sessions"]
    )


@pytest.mark.real_mongo
def test_active_documents_endpoint_filters_by_session() -> None:
    client = TestClient(app)
    db = get_database(load_config().mongo)
    session_id = "web-active-documents-test"
    db.active_documents.delete_many({"session_id": session_id})
    now = datetime.now(timezone.utc)
    db.active_documents.insert_one(
        {
            "schema_version": 1,
            "session_id": session_id,
            "document_id": str(ObjectId()),
            "title": "Active Doc",
            "source": {"path": "active.md"},
            "labels": ["source_section"],
            "node_ids": [str(ObjectId())],
            "reference_count": 1,
            "first_referenced_at": now,
            "last_referenced_at": now,
        }
    )

    try:
        response = client.get(
            "/api/active-documents",
            params={"session_id": session_id, "limit": 5},
        )
    finally:
        db.active_documents.delete_many({"session_id": session_id})

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["session_id"] == session_id
    assert data["documents"][0]["title"] == "Active Doc"


def test_governance_agent_identities_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_agent_identities",
        lambda _db, limit=20: [{"identity_id": "tirzah_shared", "limit": limit}],
    )

    response = client.get("/api/governance/agent-identities", params={"limit": 2})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "identities": [{"identity_id": "tirzah_shared", "limit": 2}],
    }


def test_governance_agent_identity_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.get_agent_identity",
        lambda _db, identity_id: {"identity_id": identity_id},
    )

    response = client.get("/api/governance/agent-identities/tirzah_shared")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "identity": {"identity_id": "tirzah_shared"},
    }


def test_governance_trust_weighting_profiles_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_trust_weighting_profiles",
        lambda _db, limit=20: [{"weighting_profile_id": "default_balanced", "limit": limit}],
    )

    response = client.get("/api/governance/trust-weighting-profiles", params={"limit": 3})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "profiles": [{"weighting_profile_id": "default_balanced", "limit": 3}],
    }


def test_governance_trust_weighting_profile_endpoint_reports_missing(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.get_trust_weighting_profile",
        lambda _db, weighting_profile_id: None,
    )

    response = client.get("/api/governance/trust-weighting-profiles/missing")

    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "profile": None,
    }


def test_governance_trust_diagnostic_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.trust_temporal_diagnostic_for_node",
        lambda _db, node_id, weighting_profile_id=None: {
            "node_id": node_id,
            "profile_id": weighting_profile_id,
        },
    )

    response = client.get(
        "/api/governance/trust-diagnostics/nodes/node1",
        params={"profile_id": "default_balanced"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "result": {"node_id": "node1", "profile_id": "default_balanced"},
    }


def test_governance_policies_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_governance_policies",
        lambda _db, limit=20: [{"policy_id": "read_only", "limit": limit}],
    )

    response = client.get("/api/governance/policies", params={"limit": 4})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "policies": [{"policy_id": "read_only", "limit": 4}],
    }


def test_governance_policy_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.get_governance_policy",
        lambda _db, policy_id: {"policy_id": policy_id},
    )

    response = client.get("/api/governance/policies/read_only")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "policy": {"policy_id": "read_only"},
    }


def test_governance_process_objects_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_process_objects",
        lambda _db, limit=20: [{"process_id": "review_before_write", "limit": limit}],
    )

    response = client.get("/api/governance/process-objects", params={"limit": 5})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "processes": [{"process_id": "review_before_write", "limit": 5}],
    }


def test_governance_process_object_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.get_process_object",
        lambda _db, process_id: {"process_id": process_id},
    )

    response = client.get("/api/governance/process-objects/review_before_write")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "process": {"process_id": "review_before_write"},
    }


def test_governance_process_runs_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_process_runs",
        lambda _db, session_id=None, status=None, limit=20: [
            {"run_id": "run1", "session_id": session_id, "status": status, "limit": limit}
        ],
    )

    response = client.get(
        "/api/governance/process-runs",
        params={"session_id": "s1", "status": "active", "limit": 6},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "runs": [{"run_id": "run1", "session_id": "s1", "status": "active", "limit": 6}],
    }


def test_governance_process_run_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.get_process_run",
        lambda _db, run_id: {"run_id": run_id},
    )

    response = client.get("/api/governance/process-runs/run1")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "run": {"run_id": "run1"},
    }


def test_governance_create_process_run_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.create_process_run",
        lambda _db, **kwargs: {"run_id": "run1", **kwargs},
    )

    response = client.post(
        "/api/governance/process-runs",
        json={
            "process_id": "restart_continuity",
            "session_id": "s1",
            "identity_id": "tirzah_shared",
            "current_step_id": "inspect_state",
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["run"]["process_id"] == "restart_continuity"
    assert response.json()["run"]["current_step_id"] == "inspect_state"


def test_governance_create_process_run_endpoint_reports_invalid_status() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/governance/process-runs",
        json={"process_id": "restart_continuity", "status": "invalid"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "error": "unsupported_process_run_status",
        "run": None,
    }


def test_governance_update_process_run_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.update_process_run",
        lambda _db, run_id, **kwargs: {"run_id": run_id, **kwargs},
    )

    response = client.patch(
        "/api/governance/process-runs/run1",
        json={"status": "completed", "completed_step_id": "write_restart"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["run"]["run_id"] == "run1"
    assert response.json()["run"]["status"] == "completed"


@pytest.mark.real_mongo
def test_output_ingestion_endpoint_filters_by_session() -> None:
    client = TestClient(app)
    db = get_database(load_config().mongo)
    session_id = "web-output-ingestion-test"
    db.output_ingestion_queue.delete_many({"session_id": session_id})
    now = datetime.now(timezone.utc)
    db.output_ingestion_queue.insert_one(
        {
            "schema_version": 1,
            "status": "pending",
            "source_type": "llm_answer",
            "exchange_id": str(ObjectId()),
            "session_id": session_id,
            "query": "What changed?",
            "answer_text": "Captured output.",
            "used_node_ids": [],
            "active_document_ids": [],
            "content_hash_sha256": "hash",
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
        }
    )

    try:
        response = client.get(
            "/api/output-ingestion",
            params={"session_id": session_id, "status": "pending", "limit": 5},
        )
    finally:
        db.output_ingestion_queue.delete_many({"session_id": session_id})

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["jobs"][0]["source_type"] == "llm_answer"
    assert data["jobs"][0]["answer_preview"] == "Captured output."


def test_process_output_ingestion_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.process_next_output_ingestion",
        lambda _db: {"ok": True, "status": "idle"},
    )

    response = client.post("/api/process-output-ingestion")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "idle"}


def test_generated_output_review_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_generated_output_nodes",
        lambda _db, limit=20, endorsement_label=None: [
            {"node_id": "node1", "endorsement_label": endorsement_label, "limit": limit}
        ],
    )

    response = client.get(
        "/api/review/generated-output",
        params={"endorsement": "unreviewed", "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["nodes"] == [
        {"node_id": "node1", "endorsement_label": "unreviewed", "limit": 1}
    ]


def test_generated_output_review_endpoint_reports_invalid_filter(monkeypatch) -> None:
    client = TestClient(app)

    def fake_list(*_args, **_kwargs):
        raise ValueError("Unsupported endorsement label: trusted")

    monkeypatch.setattr("tirzah.web.app.list_generated_output_nodes", fake_list)

    response = client.get(
        "/api/review/generated-output",
        params={"endorsement": "trusted"},
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "invalid_endorsement_label"


def test_endorse_node_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.update_node_endorsement",
        lambda _db, node_id, endorsement_label, reviewer="user", note=None: {
            "ok": True,
            "node_id": node_id,
            "endorsement_label": endorsement_label,
            "reviewer": reviewer,
            "note": note,
        },
    )

    response = client.post(
        "/api/review/endorse-node",
        json={
            "node_id": "node1",
            "endorsement": "rejected",
            "reviewer": "tester",
            "note": "bad answer",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "node_id": "node1",
        "endorsement_label": "rejected",
        "reviewer": "tester",
        "note": "bad answer",
    }


def test_search_accepts_origin_date_filters(monkeypatch) -> None:
    client = TestClient(app)
    captured = {}

    def fake_search(_db, query=None, label=None, origin_after=None, origin_before=None, limit=10, **kwargs):
        captured.update(
            {
                "query": query,
                "origin_after": origin_after,
                "origin_before": origin_before,
                "limit": limit,
            }
        )
        return [{"title": "Recent"}]

    monkeypatch.setattr("tirzah.web.app.search_nodes", fake_search)
    response = client.get(
        "/api/search",
        params={"query": "vorton", "origin_after": "2020-01-01", "origin_before": "2024-12-31"},
    )
    assert response.status_code == 200
    assert captured["origin_after"] == "2020-01-01"
    assert captured["origin_before"] == "2024-12-31"


def test_set_origin_date_endpoint(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "tirzah.web.app.update_document_origin_date",
        lambda _db, document_id, origin_date, reviewer="user", note=None: {
            "ok": True,
            "document_id": document_id,
            "origin_date": origin_date,
            "reviewer": reviewer,
            "note": note,
        },
    )
    response = client.post(
        "/api/documents/doc1/origin-date",
        json={"origin_date": "2021-02-03", "reviewer": "tester", "note": "fixed"},
    )
    assert response.status_code == 200
    assert response.json()["origin_date"] == "2021-02-03"
    assert response.json()["note"] == "fixed"


def test_set_node_summary_endpoint(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "tirzah.web.app.update_node_summary",
        lambda _db, node_id, summary, reviewer="user", note=None: {
            "ok": True,
            "node_id": node_id,
            "summary": summary,
            "reviewer": reviewer,
            "note": note,
        },
    )
    response = client.post(
        "/api/nodes/node1/summary",
        json={"summary": "A vorton is a closed loop.", "reviewer": "tester"},
    )
    assert response.status_code == 200
    assert response.json()["summary"] == "A vorton is a closed loop."


def test_rebuild_diff_and_rebuild_document_endpoints(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "tirzah.cli.rebuild_document_from_existing_source",
        lambda _db, document_id, source=None, ingestion_epoch=None, runtime_config=None, mode="full", compare_only=False: {
            "ok": True,
            "document_id": document_id,
            "mode": "diff" if compare_only or mode == "diff" else mode,
            "compare_only": compare_only,
            "source": source,
            "ingestion_epoch": ingestion_epoch,
        },
    )

    preview = client.get("/api/documents/doc1/rebuild-diff")
    assert preview.status_code == 200
    assert preview.json()["compare_only"] is True
    assert preview.json()["document_id"] == "doc1"

    applied = client.post(
        "/api/rebuild-document",
        json={"document_id": "doc1", "mode": "diff", "source": "/tmp/source.md"},
    )
    assert applied.status_code == 200
    assert applied.json()["mode"] == "diff"
    assert applied.json()["source"] == "/tmp/source.md"
    assert applied.json()["compare_only"] is False


def test_proposed_ingestion_review_endpoints(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "tirzah.web.app.list_proposed_ingestion_trees",
        lambda _db, limit=20: [{"tree_id": "tree1", "limit": limit}],
    )
    monkeypatch.setattr(
        "tirzah.web.app.promote_ingestion_tree",
        lambda _db, identifier, reviewer="user", note=None, endorsement_label=None: {
            "ok": True,
            "tree_id": identifier,
            "reviewer": reviewer,
            "note": note,
            "endorsement_label": endorsement_label,
            "status": "active",
        },
    )
    monkeypatch.setattr(
        "tirzah.web.app.reject_ingestion_tree",
        lambda _db, identifier, reviewer="user", note=None: {
            "ok": True,
            "tree_id": identifier,
            "reviewer": reviewer,
            "note": note,
            "status": "rejected",
        },
    )

    listed = client.get("/api/review/proposed-ingestions", params={"limit": 3})
    assert listed.status_code == 200
    assert listed.json()["trees"] == [{"tree_id": "tree1", "limit": 3}]

    promoted = client.post(
        "/api/review/promote-ingestion",
        json={"identifier": "tree1", "reviewer": "tester", "endorsement": "explicit_endorsed"},
    )
    assert promoted.status_code == 200
    assert promoted.json()["status"] == "active"
    assert promoted.json()["endorsement_label"] == "explicit_endorsed"

    rejected = client.post(
        "/api/review/reject-ingestion",
        json={"identifier": "tree1", "note": "bad chunks"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"


def test_semantic_edge_candidates_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.list_semantic_edge_candidates",
        lambda _db, status="pending", limit=20, relation_type=None: [
            {
                "candidate_id": "candidate1",
                "status": status,
                "limit": limit,
                "relation_type": relation_type,
            }
        ],
    )

    response = client.get(
        "/api/review/semantic-edge-candidates",
        params={"status": "pending", "limit": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "candidates": [
            {
                "candidate_id": "candidate1",
                "status": "pending",
                "limit": 2,
                "relation_type": None,
            }
        ],
    }


def test_enqueue_label_semantic_edge_candidates_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.enqueue_semantic_edge_candidates",
        lambda _db, **kwargs: {"ok": True, "source": "label", **kwargs},
    )

    response = client.post(
        "/api/review/enqueue-semantic-edge-candidates",
        json={
            "node_id": "node1",
            "candidate_source": "label_overlap",
            "include_same_document": True,
            "relation_type": "supports",
            "created_by": "tester",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "source": "label",
        "node_id": "node1",
        "limit": 3,
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "tester",
    }


def test_enqueue_vector_semantic_edge_candidates_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.enqueue_vector_semantic_edge_candidates",
        lambda _db, **kwargs: {"ok": True, "source": "vector", **kwargs},
    )

    response = client.post(
        "/api/review/enqueue-semantic-edge-candidates",
        json={
            "node_id": "node1",
            "candidate_source": "embedding_similarity",
            "include_same_document": True,
            "relation_type": "supports",
            "created_by": "tester",
            "min_similarity": 0.82,
            "limit": 3,
            "candidate_scan_limit": 500,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "source": "vector",
        "node_id": "node1",
        "limit": 3,
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "tester",
        "min_similarity": 0.82,
        "candidate_scan_limit": 500,
    }


def test_vector_semantic_candidates_preview_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.embedding_candidate_report",
        lambda _db, **kwargs: {
            "ok": True,
            "nodes": [{"node_id": kwargs["node_id"], "title": "Target"}],
            "diagnostics": {"returned_count": 1, **kwargs},
        },
    )

    response = client.get(
        "/api/review/vector-semantic-candidates",
        params={
            "node_id": "node1",
            "include_same_document": True,
            "min_similarity": 0.82,
            "limit": 3,
            "candidate_scan_limit": 500,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "nodes": [{"node_id": "node1", "title": "Target"}],
        "diagnostics": {
            "returned_count": 1,
            "node_id": "node1",
            "include_same_document": True,
            "min_similarity": 0.82,
            "limit": 3,
            "candidate_scan_limit": 500,
        },
    }


def test_graph_edges_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.graph_edges_for_node",
        lambda _db, **kwargs: [{"edge_id": "edge1", **kwargs}],
    )

    response = client.get(
        "/api/graph/edges/node1",
        params={"direction": "outgoing", "relation_type": "supports", "limit": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "node_id": "node1",
        "edges": [
            {
                "edge_id": "edge1",
                "node_id": "node1",
                "direction": "outgoing",
                "relation_type": "supports",
                "limit": 3,
            }
        ],
    }


def test_graph_proximity_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.expand_proximity",
        lambda _db, **kwargs: [{"node_id": "near1", **kwargs}],
    )

    response = client.get(
        "/api/graph/proximity/node1",
        params={"direction": "incoming", "relation_type": "related_to", "limit": 4},
    )

    assert response.status_code == 200
    assert response.json()["nodes"][0] == {
        "node_id": "node1",
        "direction": "incoming",
        "relation_type": "related_to",
        "limit": 4,
    }


def test_graph_paths_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.expand_graph_paths",
        lambda _db, **kwargs: [{"target": "path1", **kwargs}],
    )

    response = client.get(
        "/api/graph/paths/node1",
        params={
            "direction": "both",
            "relation_type": "supports",
            "max_depth": 3,
            "limit": 5,
            "branch_limit": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["paths"][0] == {
        "target": "path1",
        "node_id": "node1",
        "direction": "both",
        "relation_type": "supports",
        "max_depth": 3,
        "limit": 5,
        "branch_limit": 2,
    }


def test_enqueue_vector_semantic_batch_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.enqueue_vector_semantic_edge_candidate_batch",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    response = client.post(
        "/api/review/enqueue-vector-semantic-batch",
        json={
            "label": "source_section",
            "document_id": "doc1",
            "focus_limit": 5,
            "candidates_per_node": 1,
            "include_same_document": True,
            "relation_type": "supports",
            "created_by": "tester",
            "min_similarity": 0.82,
            "candidate_scan_limit": 500,
            "exclude_node_keys": ["section-1"],
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "label": "source_section",
        "document_id": "doc1",
        "focus_limit": 5,
        "candidates_per_node": 1,
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "tester",
        "min_similarity": 0.82,
        "candidate_scan_limit": 500,
        "exclude_node_keys": ["section-1"],
        "dry_run": True,
    }


def test_enqueue_contradiction_candidates_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.enqueue_contradiction_candidates",
        lambda _db, **kwargs: {"ok": True, "source": "contradiction", **kwargs},
    )

    response = client.post(
        "/api/review/enqueue-semantic-edge-candidates",
        json={
            "node_id": "node1",
            "candidate_source": "contradiction_signals",
            "include_same_document": True,
            "created_by": "tester",
            "min_similarity": 0.84,
            "max_similarity": 0.96,
            "limit": 3,
            "candidate_scan_limit": 400,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "source": "contradiction",
        "node_id": "node1",
        "limit": 3,
        "include_same_document": True,
        "created_by": "tester",
        "min_similarity": 0.84,
        "max_similarity": 0.96,
        "candidate_scan_limit": 400,
    }


def test_contradiction_candidates_preview_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.contradiction_candidate_report",
        lambda _db, **kwargs: {
            "ok": True,
            "nodes": [{"node_id": kwargs["node_id"], "title": "Target"}],
            "diagnostics": {"returned_count": 1, **kwargs},
        },
    )

    response = client.get(
        "/api/review/contradiction-candidates",
        params={
            "node_id": "node1",
            "include_same_document": True,
            "min_similarity": 0.84,
            "max_similarity": 0.96,
            "limit": 3,
            "candidate_scan_limit": 400,
        },
    )

    assert response.status_code == 200
    assert response.json()["diagnostics"]["min_similarity"] == 0.84
    assert response.json()["diagnostics"]["max_similarity"] == 0.96
    assert response.json()["nodes"][0]["title"] == "Target"


def test_enqueue_contradiction_batch_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.enqueue_contradiction_candidate_batch",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    response = client.post(
        "/api/review/enqueue-contradiction-batch",
        json={
            "label": "source_section",
            "document_id": "doc1",
            "focus_limit": 5,
            "candidates_per_node": 1,
            "include_same_document": True,
            "created_by": "tester",
            "min_similarity": 0.84,
            "max_similarity": 0.96,
            "candidate_scan_limit": 400,
            "exclude_node_keys": ["section-1"],
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "label": "source_section",
        "document_id": "doc1",
        "focus_limit": 5,
        "candidates_per_node": 1,
        "include_same_document": True,
        "created_by": "tester",
        "min_similarity": 0.84,
        "max_similarity": 0.96,
        "candidate_scan_limit": 400,
        "exclude_node_keys": ["section-1"],
        "dry_run": True,
    }


def test_enqueue_semantic_edge_candidates_endpoint_rejects_unknown_source() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/review/enqueue-semantic-edge-candidates",
        json={"node_id": "node1", "candidate_source": "unknown"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "reason": "invalid_candidate_source",
        "candidate_source": "unknown",
    }


def test_review_semantic_edge_candidate_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "tirzah.web.app.review_semantic_edge_candidate",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    response = client.post(
        "/api/review/semantic-edge-candidate",
        json={
            "candidate_id": "candidate1",
            "action": "accept",
            "reviewer": "tester",
            "note": "looks useful",
            "weight": 0.8,
            "confidence": 0.9,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "candidate_id": "candidate1",
        "action": "accept",
        "reviewer": "tester",
        "note": "looks useful",
        "weight": 0.8,
        "confidence": 0.9,
    }


@pytest.mark.real_mongo
def test_history_endpoint_filters_seeded_rows() -> None:
    client = TestClient(app)
    db = get_database(load_config().mongo)
    session_id = "web-filter-test"
    db.exchanges.delete_many({"session_id": session_id})
    now = datetime.now(timezone.utc)
    db.exchanges.insert_many(
        [
            {
                "schema_version": 1,
                "session_id": session_id,
                "focus_node_id": None,
                "query": "Tirzah design purpose",
                "answer": {
                    "adapter": "ollama_cli",
                    "model": "qwen3.6:latest",
                    "answer": "graph memory layer",
                    "used_node_ids": [],
                },
                "created_at": now,
            },
            {
                "schema_version": 1,
                "session_id": session_id,
                "focus_node_id": None,
                "query": "Other prompt",
                "answer": {
                    "adapter": "mock_answer",
                    "model": None,
                    "answer": "unrelated",
                    "used_node_ids": [],
                },
                "created_at": now,
            },
        ]
    )

    try:
        response = client.get(
            "/api/history",
            params={
                "limit": 3,
                "session_id": session_id,
                "q": "Tirzah",
                "adapter": "ollama_cli",
                "model": "qwen3.6:latest",
            },
        )
    finally:
        db.exchanges.delete_many({"session_id": session_id})

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert [row["query"] for row in data["exchanges"]] == ["Tirzah design purpose"]


@pytest.mark.real_mongo
def test_jobs_endpoint_filters_seeded_rows() -> None:
    client = TestClient(app)
    db = get_database(load_config().mongo)
    marker = ObjectId().binary.hex()
    db.queue.delete_many({"path": {"$regex": marker}})
    now = datetime.now(timezone.utc)
    db.queue.insert_many(
        [
            {
                "path": f"data/ingest/{marker}-missing.md",
                "checksum_sha256": f"{marker}aaa",
                "status": "failed",
                "reason": "source_missing",
                "attempts": 1,
                "created_at": now,
                "updated_at": now,
            },
            {
                "path": f"data/ingest/{marker}-duplicate.md",
                "checksum_sha256": f"{marker}bbb",
                "status": "rejected",
                "reason": "duplicate_checksum",
                "attempts": 0,
                "created_at": now,
                "updated_at": now,
            },
        ]
    )

    try:
        response = client.get(
            "/api/jobs",
            params={
                "limit": 3,
                "status": "failed",
                "q": marker,
                "reason": "source_missing",
            },
        )
    finally:
        db.queue.delete_many({"path": {"$regex": marker}})

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert [row["reason"] for row in data["jobs"]] == ["source_missing"]


def test_serialize_queue_job_does_not_mutate_source_row() -> None:
    job_id = ObjectId()
    document_id = ObjectId()
    existing_queue_id = ObjectId()
    timestamp = datetime(2026, 6, 14, tzinfo=timezone.utc)
    source = {
        "_id": job_id,
        "path": "data/ingest/source.md",
        "existing_queue_id": existing_queue_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "result": {"document_id": document_id, "node_ids": ["n1"]},
    }

    serialized = serialize_queue_job(source)

    assert serialized["_id"] == str(job_id)
    assert serialized["existing_queue_id"] == str(existing_queue_id)
    assert serialized["created_at"] == "2026-06-14T00:00:00+00:00"
    assert serialized["updated_at"] == "2026-06-14T00:00:00+00:00"
    assert serialized["result"]["document_id"] == str(document_id)
    assert source["_id"] == job_id
    assert source["existing_queue_id"] == existing_queue_id
    assert source["created_at"] == timestamp
    assert source["result"]["document_id"] == document_id


def test_jobs_endpoint_serializes_without_mutating_recent_jobs(monkeypatch) -> None:
    job_id = ObjectId()
    document_id = ObjectId()
    timestamp = datetime(2026, 6, 14, tzinfo=timezone.utc)
    jobs = [
        {
            "_id": job_id,
            "path": "data/ingest/source.md",
            "status": "completed",
            "created_at": timestamp,
            "updated_at": timestamp,
            "result": {"document_id": document_id},
        }
    ]

    monkeypatch.setattr("tirzah.web.app.recent_jobs", lambda *_args, **_kwargs: jobs)

    response = TestClient(app).get("/api/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"][0]["_id"] == str(job_id)
    assert response.json()["jobs"][0]["result"]["document_id"] == str(document_id)
    assert jobs[0]["_id"] == job_id
    assert jobs[0]["result"]["document_id"] == document_id


def test_upload_source_stages_supported_file() -> None:
    client = TestClient(app)
    config = load_config()
    filename = f"web-upload-{ObjectId()}.md"
    path = config.paths.ingest / filename
    path.unlink(missing_ok=True)

    try:
        response = client.post(
            "/api/upload-source",
            json={"filename": f"../{filename}", "content": "# Uploaded\n\nSource text."},
        )
        data = response.json()
    finally:
        path.unlink(missing_ok=True)

    assert response.status_code == 200
    assert data["ok"] is True
    assert data["status"] == "staged"
    assert data["filename"] == filename
    assert Path(data["path"]).name == filename


def test_upload_source_rejects_unsupported_suffix() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/upload-source",
        json={"filename": "source.pdf", "content": "%PDF"},
    )

    assert response.status_code == 400
    assert "Unsupported source type" in response.json()["detail"]


def test_upload_source_rejects_oversized_content(monkeypatch, tmp_path) -> None:
    from tirzah.web.app import create_app

    config = AppConfig(
        paths=PathConfig(ingest=tmp_path / "ingest"),
        runtime=RuntimeConfig(web_max_upload_bytes=4),
    )
    monkeypatch.setattr("tirzah.web.app.load_config", lambda: config)
    monkeypatch.setattr("tirzah.web.app.get_database", lambda _mongo: object())
    monkeypatch.setattr("tirzah.web.app.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.process.templates.seed_presets", lambda _db: None)
    client = TestClient(create_app())

    response = client.post(
        "/api/upload-source",
        json={"filename": "source.md", "content": "12345"},
    )

    assert response.status_code == 413


def test_api_token_protects_api_routes(monkeypatch) -> None:
    from tirzah.web.app import create_app

    config = AppConfig(runtime=RuntimeConfig(web_api_token="secret"))
    monkeypatch.setattr("tirzah.web.app.load_config", lambda: config)
    monkeypatch.setattr("tirzah.web.app.get_database", lambda _mongo: object())
    monkeypatch.setattr("tirzah.web.app.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.process.templates.seed_presets", lambda _db: None)
    client = TestClient(create_app())

    assert client.get("/api/documents").status_code == 401


def test_admin_cli_equivalent_endpoints(monkeypatch) -> None:
    from tirzah.web.app import create_app

    class Collection:
        def find(self, query=None, projection=None):
            return []

    class Db:
        def __getitem__(self, name):
            return Collection()

    monkeypatch.setattr("tirzah.web.app.load_config", lambda: AppConfig())
    monkeypatch.setattr("tirzah.web.app.get_database", lambda _mongo: Db())
    monkeypatch.setattr("tirzah.web.app.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.process.templates.seed_presets", lambda _db: None)
    client = TestClient(create_app())

    assert client.get("/api/config-status").json()["runtime"]["config_source"]["mongo_db"] == "tirzah_dev"
    assert client.get("/api/db-ping").json()["ok"] is True
    migrate = client.post("/api/migrate", params={"status": True}).json()
    assert migrate["ok"] is True
    assert migrate["report"]["target_version"] >= 2


def test_ingest_folder_lists_supported_files() -> None:
    client = TestClient(app)
    config = load_config()
    filename = f"web-folder-{ObjectId()}.txt"
    supported_path = config.paths.ingest / filename
    unsupported_path = config.paths.ingest / f"{filename}.pdf"
    config.paths.ingest.mkdir(parents=True, exist_ok=True)
    supported_path.write_text("Date: 2020\n\nfolder source", encoding="utf-8")
    unsupported_path.write_text("ignored", encoding="utf-8")

    try:
        response = client.get("/api/ingest-folder")
        data = response.json()
    finally:
        supported_path.unlink(missing_ok=True)
        unsupported_path.unlink(missing_ok=True)

    assert response.status_code == 200
    assert data["ok"] is True
    assert data["ordering"] == "origin_date_then_path"
    names = [row["name"] for row in data["files"]]
    assert filename in names
    assert f"{filename}.pdf" not in names
    row = next(row for row in data["files"] if row["name"] == filename)
    assert row["origin_date"] == "2020-01-01"
    assert row["origin_date_source"] == "explicit_content"
    assert row["date_candidate_count"] >= 1


def test_ingest_folder_lists_unreadable_file_without_failing() -> None:
    client = TestClient(app)
    config = load_config()
    filename = f"web-folder-unreadable-{ObjectId()}.txt"
    unreadable_path = config.paths.ingest / filename
    config.paths.ingest.mkdir(parents=True, exist_ok=True)
    unreadable_path.write_bytes(b"\xff\xfe\x00\x00")

    try:
        response = client.get("/api/ingest-folder")
        data = response.json()
    finally:
        unreadable_path.unlink(missing_ok=True)

    assert response.status_code == 200
    row = next(row for row in data["files"] if row["name"] == filename)
    assert row["status"] == "unreadable"
    assert row["error"] == "UnicodeDecodeError"
    assert row["origin_date"] is None


def simple_match(row, query):
    for key, expected in query.items():
        actual = simple_nested_get(row, key)
        if isinstance(expected, dict):
            if "$exists" in expected and (actual is not None) is not expected["$exists"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$nin" in expected and actual in expected["$nin"]:
                return False
            continue
        if isinstance(actual, list):
            if expected not in actual:
                return False
            continue
        if actual != expected:
            return False
    return True


def simple_nested_get(row, dotted_key):
    value = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
