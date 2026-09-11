import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bson import ObjectId

from tirzah.cli import (
    chronological_folder_source_plan,
    discover_folder_sources,
    document_ids_for_label,
    destructive_rebuild_refusal,
    existing_document_extra_labels,
    ingest_source_path,
    init_config_payload,
    main,
    rebuild_document_from_existing_source,
    serve_app,
    write_initial_config,
)
from tirzah.db.health import memory_health_payload, render_memory_health_text
from tirzah.db.repositories import DuplicateSourceError
from tirzah.models.ingestion import IngestedNode, IngestionResult, SourceRef


def test_discover_folder_sources_finds_markdown_and_text(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.md").write_text("a", encoding="utf-8")
    (root / "a2.markdown").write_text("a2", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "c.json").write_text("{}", encoding="utf-8")
    (root / "d.pdf").write_text("nope", encoding="utf-8")

    assert [path.name for path in discover_folder_sources(root)] == [
        "a.md",
        "a2.markdown",
        "b.txt",
        "c.json",
    ]


def test_discover_folder_sources_skips_git_directory(tmp_path: Path) -> None:
    root = tmp_path / "source"
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "ignored.md").write_text("ignored", encoding="utf-8")
    (root / "included.md").write_text("included", encoding="utf-8")

    assert [path.name for path in discover_folder_sources(root)] == ["included.md"]


def test_chronological_folder_source_plan_orders_by_origin_date(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    late_filename = root / "source-2026.md"
    explicit_earlier = root / "source-2024.md"
    undated = root / "undated.md"
    late_filename.write_text("No explicit date.", encoding="utf-8")
    explicit_earlier.write_text("Date: 2020\n\nEarlier content.", encoding="utf-8")
    undated.write_text("No date marker.", encoding="utf-8")

    plan = chronological_folder_source_plan(root)

    assert [item["path"].name for item in plan] == [
        "source-2024.md",
        "source-2026.md",
        "undated.md",
    ]
    assert plan[0]["origin_date"] == "2020-01-01"
    assert plan[0]["origin_date_source"] == "explicit_content"


def test_chronological_folder_source_plan_keeps_unreadable_file_as_error(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    readable = root / "readable.md"
    unreadable = root / "unreadable.md"
    readable.write_text("Date: 2020\n\nReadable.", encoding="utf-8")
    unreadable.write_bytes(b"\xff\xfe\x00\x00")

    plan = chronological_folder_source_plan(root)

    assert [item["path"].name for item in plan] == ["readable.md", "unreadable.md"]
    assert plan[1]["error"] == "UnicodeDecodeError"
    assert plan[1]["origin_date"] is None


def test_existing_document_extra_labels_excludes_structural_labels() -> None:
    document_id = ObjectId()
    db = FakeDb(
        [
            {"document_id": document_id, "labels": ["source_root", "memory_reference"]},
            {"document_id": document_id, "labels": ["source_chunk", "external_corpus"]},
        ]
    )

    assert existing_document_extra_labels(db, str(document_id)) == [
        "external_corpus",
        "memory_reference",
    ]


def test_cli_process_output_ingestion_passes_target_filters(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "process-output-ingestion",
            "--session-id",
            "session1",
            "--job-id",
            "job1",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)

    def fake_process(_db, **kwargs):
        return {"ok": True, **kwargs}

    monkeypatch.setattr("tirzah.cli.process_next_output_ingestion", fake_process)

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {"ok": True, "session_id": "session1", "job_id": "job1"}


def test_cli_session_continuity_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "session-continuity",
            "--session-id",
            "s1",
            "--limit",
            "2",
            "--format",
            "json",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.session_continuity",
        lambda _db, *, session_id, limit: {
            "session_id": session_id,
            "limit": limit,
            "latest": {"exchange_id": "ex1"},
            "recent": [],
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "session_id": "s1",
        "limit": 2,
        "latest": {"exchange_id": "ex1"},
        "recent": [],
    }


def test_cli_restart_render_command_writes_file(tmp_path: Path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "restart.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "restart-render", "--session-id", "s1", "--output", str(output_path)],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.session_continuity",
        lambda _db, *, session_id, limit: {
            "session_id": session_id,
            "latest": {"exchange_id": "ex1", "query": "Resume?"},
            "recent": [],
        },
    )

    main()

    assert "Wrote restart state" in capsys.readouterr().out
    rendered = output_path.read_text(encoding="utf-8")
    assert rendered.startswith("# Restart State")
    assert "Resume?" in rendered


def test_init_config_payload_uses_docker_mongo_and_mock_defaults() -> None:
    payload = init_config_payload(docker=True, runtime_choice="mock")

    assert payload["mongo"]["uri"] == "mongodb://mongo:27017"
    assert payload["mongo"]["database"] == "tirzah"
    assert payload["runtime"]["answer_adapter"] == "mock"
    assert payload["runtime"]["memory_agent_adapter"] == "mock"
    assert payload["runtime"]["embedding_adapter"] == "mock"


def test_init_config_payload_local_command_uses_packaged_helper() -> None:
    payload = init_config_payload(docker=False, runtime_choice="local_command")

    assert payload["runtime"]["embedding_adapter"] == "local_command"
    assert payload["runtime"]["profile_command"] == ["tirzah-profile-helper", "--worker"]
    assert payload["runtime"]["profile_command_mode"] == "worker"


def test_init_config_payload_hoglah_uses_optional_answer_queue() -> None:
    payload = init_config_payload(docker=True, runtime_choice="hoglah")

    assert payload["runtime"]["answer_adapter"] == "hoglah"
    assert payload["runtime"]["memory_agent_adapter"] is None
    assert payload["runtime"]["embedding_adapter"] == "mock"
    assert payload["runtime"]["hoglah_ollama_host"] == "http://host.docker.internal:11434"
    assert payload["runtime"]["hoglah_db_path"] == "data/hoglah/jobs.sqlite3"


def test_init_config_payload_kiro_cli_uses_headless_chat() -> None:
    payload = init_config_payload(docker=False, runtime_choice="kiro_cli")

    assert payload["runtime"]["answer_adapter"] == "kiro_cli"
    assert payload["runtime"]["ingestion_model_adapter"] == "kiro_cli"
    assert payload["runtime"]["embedding_adapter"] == "mock"


def test_write_initial_config_creates_config_and_data_dirs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"

    result = write_initial_config(config_path, docker=True, non_interactive=True)

    assert result["ok"] is True
    assert config_path.exists()
    assert (tmp_path / "data" / "ingest").is_dir()
    assert (tmp_path / "data" / "archive").is_dir()
    assert "mongodb://mongo:27017" in config_path.read_text(encoding="utf-8")


def test_write_initial_config_refuses_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("existing: true\n", encoding="utf-8")

    result = write_initial_config(config_path, non_interactive=True)

    assert result["ok"] is False
    assert result["reason"] == "config_exists"
    assert config_path.read_text(encoding="utf-8") == "existing: true\n"


def test_cli_config_status_does_not_connect_to_mongo(monkeypatch, capsys) -> None:
    from tirzah.config import AppConfig

    monkeypatch.setattr(sys, "argv", ["tirzah", "config-status", "--format", "json"])
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: AppConfig())

    def fail_get_database(_config):
        raise AssertionError("config-status should not connect to Mongo")

    monkeypatch.setattr("tirzah.cli.get_database", fail_get_database)

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["config_source"]["mongo_db"] == "tirzah_dev"
    assert "capabilities" in output


def test_cli_init_does_not_connect_to_mongo(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "init", "--docker", "--non-interactive"],
    )

    def fail_get_database(_config):
        raise AssertionError("init should not connect to Mongo")

    monkeypatch.setattr("tirzah.cli.get_database", fail_get_database)

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert (tmp_path / "config.yaml").exists()


def test_memory_health_payload_counts_corpus_and_attention() -> None:
    db = SimpleNamespace(
        documents=HealthCollection([{"title": "Doc"}]),
        trees=HealthCollection([{"label": "active"}]),
        nodes=HealthCollection(
            [
                {
                    "labels": ["source_chunk"],
                    "embedding": {"vector": [1.0]},
                    "endorsement_label": "explicit_endorsed",
                },
                {
                    "labels": ["source_chunk"],
                    "endorsement_label": "unreviewed",
                },
                {
                    "labels": ["generated_output"],
                    "embedding": {"vector": [0.5]},
                    "endorsement_label": "unreviewed",
                },
            ]
        ),
        graph_edges=HealthCollection([{"relation_type": "related_to"}]),
        semantic_edge_candidates=HealthCollection(
            [{"status": "pending"}, {"status": "accepted"}]
        ),
        queue=HealthCollection([{"status": "failed"}, {"status": "pending"}]),
        embedding_backfill_jobs=HealthCollection([{"status": "pending"}]),
        output_ingestion_queue=HealthCollection(
            [{"status": "failed"}, {"status": "completed"}, {"status": "pending"}]
        ),
    )

    report = memory_health_payload(db)

    assert report["totals"]["documents"] == 1
    assert report["totals"]["nodes"] == 3
    assert report["totals"]["source_chunks"] == 2
    assert report["totals"]["generated_outputs"] == 1
    assert report["profile_coverage"] == {
        "profiled_source_chunks": 1,
        "eligible_source_chunks": 2,
        "percent": 50.0,
    }
    assert report["endorsements"] == {"explicit_endorsed": 1, "unreviewed": 2}
    assert report["queues"]["semantic_edge_candidates"]["pending"] == 1
    assert [item["kind"] for item in report["attention"]] == [
        "profile_coverage",
        "pending_profile_backfill",
        "pending_semantic_review",
        "pending_output_ingestion",
        "failed_jobs",
    ]


def test_render_memory_health_text_summarizes_attention() -> None:
    report = {
        "totals": {
            "documents": 1,
            "nodes": 3,
            "source_chunks": 2,
            "graph_edges": 1,
            "semantic_edge_candidates": 2,
        },
        "profile_coverage": {
            "profiled_source_chunks": 1,
            "eligible_source_chunks": 2,
            "percent": 50.0,
        },
        "endorsements": {"unreviewed": 2},
        "queues": {
            "ingestion": {"failed": 1},
            "profile_backfill": {"pending": 1},
            "output_ingestion": {},
            "semantic_edge_candidates": {"pending": 1},
        },
        "attention": [{"message": "Run a backfill."}],
    }

    text = render_memory_health_text(report)

    assert "Memory health" in text
    assert "Corpus: 1 document(s), 3 node(s), 2 source chunk(s)" in text
    assert "Profiles: 1 / 2 source chunk(s) (50.0%)" in text
    assert "- Run a backfill." in text


def test_cli_memory_health_json_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tirzah", "memory-health", "--format", "json"])
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.memory_health_payload",
        lambda _db: {"ok": True, "totals": {"documents": 1}},
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "totals": {"documents": 1},
    }


def test_serve_app_uses_uvicorn_target(monkeypatch) -> None:
    called = {}

    def fake_run(app, **kwargs):
        called["app"] = app
        called.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    serve_app("0.0.0.0", 8765, reload=True)

    assert called == {
        "app": "tirzah.web.app:app",
        "host": "0.0.0.0",
        "port": 8765,
        "reload": True,
    }


def test_document_ids_for_label_returns_sorted_strings() -> None:
    first = ObjectId()
    second = ObjectId()
    db = FakeDb(
        [
            {"document_id": second, "labels": ["ams_domain"]},
            {"document_id": first, "labels": ["ams_domain"]},
            {"document_id": ObjectId(), "labels": ["other"]},
        ]
    )

    assert document_ids_for_label(db, "ams_domain") == sorted([str(first), str(second)])


def test_destructive_rebuild_refusal_explains_force_replace() -> None:
    refusal = destructive_rebuild_refusal("rebuild-document")

    assert refusal["ok"] is False
    assert refusal["reason"] == "destructive_rebuild_requires_force_replace"
    assert "--force-replace" in refusal["message"]


def test_rebuild_document_uses_original_source_path_for_adapter_title(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "abc123.txt"
    archive.write_text("plain text", encoding="utf-8")
    document_id = ObjectId()
    db = FakeDb(
        [],
        document={
            "document_id": str(document_id),
            "source": {
                "path": "data/ingest/original-name.txt",
                "archive_path": str(archive),
                "checksum_sha256": "abc123",
            },
        },
    )
    captured = {}

    def fake_rebuild(_db, _document_id, result, **_kwargs):
        captured["title"] = result.title
        return {"document_id": str(document_id), "replaced": True}

    monkeypatch.setattr("tirzah.cli.get_document", lambda _db, _document_id: db.document)
    monkeypatch.setattr("tirzah.cli.rebuild_document", fake_rebuild)

    result = rebuild_document_from_existing_source(db, str(document_id))

    assert result["ok"] is True
    assert captured["title"] == "original-name"


def test_rebuild_document_uses_runtime_ingestion_adapter(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "abc123.txt"
    archive.write_text("plain text", encoding="utf-8")
    document_id = ObjectId()
    db = FakeDb(
        [],
        document={
            "document_id": str(document_id),
            "source": {
                "path": "data/ingest/original-name.txt",
                "archive_path": str(archive),
                "checksum_sha256": "abc123",
            },
        },
    )
    runtime = SimpleNamespace(ingestion_adapter="fake_ingestion")
    captured = {}

    class FakeAdapter:
        def process(self, path, text, source_kind, extra_labels=None):
            captured["path"] = path
            captured["text"] = text
            captured["source_kind"] = source_kind
            captured["extra_labels"] = extra_labels
            return IngestionResult(
                source=SourceRef(path=str(path), kind=source_kind),
                title="fake rebuild",
                summary="fake summary",
                nodes=[
                    IngestedNode(
                        node_key="root",
                        title="fake rebuild",
                        text="fake summary",
                        labels=["source_root"],
                    )
                ],
                adapter="fake_ingestion",
            )

    monkeypatch.setattr("tirzah.cli.get_document", lambda _db, _document_id: db.document)
    monkeypatch.setattr("tirzah.cli.existing_document_extra_labels", lambda _db, _document_id: ["domain_label"])
    monkeypatch.setattr("tirzah.cli.ingestion_adapter", lambda runtime_config: FakeAdapter())
    monkeypatch.setattr(
        "tirzah.cli.rebuild_document",
        lambda _db, _document_id, result, **_kwargs: {
            "document_id": str(document_id),
            "adapter": result.adapter,
            "title": result.title,
            "mode": _kwargs.get("mode"),
            "compare_only": _kwargs.get("compare_only"),
        },
    )

    result = rebuild_document_from_existing_source(db, str(document_id), runtime_config=runtime)

    assert result["ok"] is True
    assert result["adapter"] == "fake_ingestion"
    assert result["title"] == "fake rebuild"
    assert captured == {
        "path": Path("data/ingest/original-name.txt"),
        "text": "plain text",
        "source_kind": "txt",
        "extra_labels": ["domain_label"],
    }
    assert result["mode"] == "full"
    assert result["compare_only"] is False


def test_rebuild_document_from_existing_source_passes_diff_flags(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "abc123.txt"
    archive.write_text("plain text", encoding="utf-8")
    document_id = ObjectId()
    db = FakeDb(
        [],
        document={
            "document_id": str(document_id),
            "source": {"archive_path": str(archive), "checksum_sha256": "abc123"},
        },
    )
    captured = {}

    monkeypatch.setattr("tirzah.cli.get_document", lambda _db, _document_id: db.document)
    monkeypatch.setattr("tirzah.cli.existing_document_extra_labels", lambda _db, _document_id: [])
    monkeypatch.setattr(
        "tirzah.cli.ingestion_adapter",
        lambda runtime_config: type(
            "Adapter",
            (),
            {
                "process": staticmethod(
                    lambda path, text, source_kind, extra_labels=None: IngestionResult(
                        source=SourceRef(path=str(path), kind=source_kind),
                        title="t",
                        summary="s",
                        nodes=[IngestedNode(node_key="root", title="t", text="s")],
                    )
                )
            },
        )(),
    )
    monkeypatch.setattr(
        "tirzah.cli.rebuild_document",
        lambda _db, _document_id, result, **kwargs: captured.update(kwargs) or {"document_id": str(document_id)},
    )

    result = rebuild_document_from_existing_source(
        db,
        str(document_id),
        mode="diff",
        compare_only=True,
    )
    assert result["ok"] is True
    assert captured == {"mode": "diff", "compare_only": True}


def test_ingest_source_path_uses_configured_ingestion_adapter(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("# Source\n\nText.", encoding="utf-8")
    config = SimpleNamespace(
        runtime=SimpleNamespace(ingestion_adapter="fake_ingestion", embedding_adapter="mock"),
        paths=SimpleNamespace(archive=tmp_path / "archive"),
    )
    captured = {}

    class FakeAdapter:
        def process(self, path, text, source_kind, extra_labels=None):
            captured["path"] = path
            captured["text"] = text
            captured["source_kind"] = source_kind
            captured["extra_labels"] = extra_labels
            return IngestionResult(
                source=SourceRef(path=str(path), kind=source_kind),
                title="fake title",
                summary="fake summary",
                nodes=[
                    IngestedNode(
                        node_key="root",
                        title="fake title",
                        text="fake summary",
                        labels=["source_root"],
                    )
                ],
                adapter="fake_ingestion",
            )

    monkeypatch.setattr("tirzah.cli.find_duplicate_by_checksum", lambda _db, _checksum: None)
    monkeypatch.setattr("tirzah.cli.ingestion_adapter", lambda runtime: FakeAdapter())
    monkeypatch.setattr("tirzah.cli.archive_source", lambda path, archive_dir, checksum: tmp_path / "archive.md")
    monkeypatch.setattr("tirzah.cli.embedding_adapter", lambda _runtime: "embedder")
    monkeypatch.setattr(
        "tirzah.cli.commit_ingestion",
        lambda _db, result, embedder=None: {"document_id": "doc1", "tree_id": "tree1", "node_ids": ["node1"]},
    )

    result = ingest_source_path("db", config, source, ["custom label"])

    assert result["ok"] is True
    assert result["activity_report"]["semantic_processing"]["adapter"] == "fake_ingestion"
    assert captured == {
        "path": source,
        "text": "# Source\n\nText.",
        "source_kind": "md",
        "extra_labels": ["custom label"],
    }


def test_cli_graph_edges_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tirzah", "graph-edges", "node1", "--limit", "2"])
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.graph_edges_for_node",
        lambda _db, node_id, direction="both", relation_type=None, limit=10: [
            {"node_id": node_id, "direction": direction, "limit": limit}
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "edges": [{"node_id": "node1", "direction": "both", "limit": 2}],
    }


def test_cli_backfill_structural_graph_edges_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "backfill-structural-graph-edges", "--limit", "7"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.backfill_structural_graph_edges",
        lambda _db, limit=None: {
            "scanned_node_count": limit,
            "edge_count": 3,
            "skipped_existing_count": 4,
            "skipped_missing_parent_count": 0,
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "scanned_node_count": 7,
        "edge_count": 3,
        "skipped_existing_count": 4,
        "skipped_missing_parent_count": 0,
    }


def test_cli_backfill_embeddings_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "backfill-embeddings",
            "--limit",
            "7",
            "--label",
            "target",
            "--document-id",
            "doc1",
            "--force",
        ],
    )
    config = SimpleNamespace(mongo=SimpleNamespace(), runtime=SimpleNamespace())
    embedder = SimpleNamespace(name="fake_embedding")
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: config)
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.cli.embedding_adapter", lambda _runtime: embedder)
    monkeypatch.setattr(
        "tirzah.cli.backfill_node_embeddings",
        lambda _db, used_embedder, **kwargs: {
            "ok": True,
            "embedder": used_embedder.name,
            **kwargs,
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "embedder": "fake_embedding",
        "limit": 7,
        "label": "target",
        "document_id": "doc1",
        "force": True,
    }


def test_cli_embedding_backfill_jobs_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "embedding-backfill-jobs", "--status", "pending", "--limit", "3"],
    )
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: SimpleNamespace(mongo=SimpleNamespace()))
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_embedding_backfill_jobs",
        lambda _db, status=None, limit=20: [{"status": status, "limit": limit}],
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "jobs": [{"status": "pending", "limit": 3}],
    }


def test_cli_backfill_profiles_alias(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "backfill-profiles",
            "--limit",
            "7",
            "--label",
            "target",
            "--document-id",
            "doc1",
            "--force",
        ],
    )
    config = SimpleNamespace(mongo=SimpleNamespace(), runtime=SimpleNamespace())
    embedder = SimpleNamespace(name="fake_profile")
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: config)
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.cli.embedding_adapter", lambda _runtime: embedder)
    monkeypatch.setattr(
        "tirzah.cli.backfill_node_embeddings",
        lambda _db, used_embedder, **kwargs: {
            "ok": True,
            "embedder": used_embedder.name,
            **kwargs,
        },
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "embedder": "fake_profile",
        "limit": 7,
        "label": "target",
        "document_id": "doc1",
        "force": True,
    }


def test_cli_profile_backfill_jobs_alias(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "profile-backfill-jobs", "--status", "pending", "--limit", "3"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_embedding_backfill_jobs",
        lambda _db, status=None, limit=20: [{"status": status, "limit": limit}],
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "jobs": [{"status": "pending", "limit": 3}],
    }


def test_cli_queue_embedding_backfill_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "queue-embedding-backfill",
            "--limit",
            "9",
            "--label",
            "target",
            "--document-id",
            "doc1",
            "--force",
            "--created-by",
            "tester",
        ],
    )
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: SimpleNamespace(mongo=SimpleNamespace()))
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.create_embedding_backfill_job",
        lambda _db, **kwargs: {"job_id": "job1", **kwargs},
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "job": {
            "job_id": "job1",
            "batch_limit": 9,
            "label": "target",
            "document_id": "doc1",
            "force": True,
            "created_by": "tester",
        },
    }


def test_cli_process_embedding_backfill_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tirzah", "process-embedding-backfill", "--max-batches", "4"])
    config = SimpleNamespace(mongo=SimpleNamespace(), runtime=SimpleNamespace())
    embedder = SimpleNamespace(name="fake_embedding")
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: config)
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.cli.embedding_adapter", lambda _runtime: embedder)
    monkeypatch.setattr(
        "tirzah.cli.process_embedding_backfill_batches",
        lambda _db, used_embedder, max_batches=1: {
            "ok": True,
            "embedder": used_embedder.name,
            "max_batches": max_batches,
        },
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "embedder": "fake_embedding",
        "max_batches": 4,
    }


def test_cli_queue_status_serializes_without_mutating_summary(monkeypatch, capsys) -> None:
    job_id = ObjectId()
    timestamp = datetime(2026, 6, 15, tzinfo=timezone.utc)
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
    monkeypatch.setattr(sys, "argv", ["tirzah", "queue-status"])
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: SimpleNamespace(mongo=SimpleNamespace()))
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.cli.queue_summary", lambda _db: summary)

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["oldest_pending"]["_id"] == str(job_id)
    assert output["oldest_pending"]["created_at"] == "2026-06-15T00:00:00+00:00"
    assert summary["oldest_pending"]["_id"] == job_id
    assert summary["oldest_pending"]["created_at"] == timestamp


def test_cli_queue_recent_serializes_without_mutating_jobs(monkeypatch, capsys) -> None:
    job_id = ObjectId()
    document_id = ObjectId()
    timestamp = datetime(2026, 6, 15, tzinfo=timezone.utc)
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
    monkeypatch.setattr(sys, "argv", ["tirzah", "queue-recent"])
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: SimpleNamespace(mongo=SimpleNamespace()))
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.cli.recent_jobs", lambda *_args, **_kwargs: jobs)

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["jobs"][0]["_id"] == str(job_id)
    assert output["jobs"][0]["result"]["document_id"] == str(document_id)
    assert jobs[0]["_id"] == job_id
    assert jobs[0]["result"]["document_id"] == document_id


def test_ingest_source_path_success_returns_activity_fields(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    source_path.write_text("# Source\n\nBody.", encoding="utf-8")
    archive_path = tmp_path / "archive" / "source.md"
    config = SimpleNamespace(
        paths=SimpleNamespace(archive=tmp_path / "archive"),
        runtime=SimpleNamespace(),
    )

    monkeypatch.setattr("tirzah.cli.sha256_file", lambda _path: "checksum")
    monkeypatch.setattr("tirzah.cli.find_duplicate_by_checksum", lambda _db, _checksum: None)
    monkeypatch.setattr(
        "tirzah.cli.archive_source",
        lambda _path, _archive_root, _checksum: archive_path,
    )
    monkeypatch.setattr("tirzah.cli.embedding_adapter", lambda _runtime: "embedder")
    monkeypatch.setattr(
        "tirzah.cli.commit_ingestion",
        lambda _db, result, embedder=None: {
            "document_id": "doc1",
            "tree_id": "tree1",
            "node_ids": ["node1"],
            "ingestion_epoch": result.ingestion_epoch,
            "embedded_node_count": 1,
            "embedding_model": "mock",
            "embedding_dimensions": 16,
        },
    )

    result = ingest_source_path(
        "db",
        config,
        source_path,
        labels=["source_root"],
        ingestion_epoch="2026-06-15-test",
    )

    assert result["ok"] is True
    assert result["document_id"] == "doc1"
    assert result["activity_report"]["kind"] == "ingestion_activity_report"
    assert result["activity_report"]["repository_actions"]["document_id"] == "doc1"
    assert "Repository write: document doc1" in result["activity_log"]


def test_ingest_source_path_duplicate_precheck_returns_activity_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.md"
    source_path.write_text("# Source\n\nBody.", encoding="utf-8")
    existing_document_id = ObjectId()
    config = SimpleNamespace(
        paths=SimpleNamespace(archive=tmp_path / "archive"),
        runtime=SimpleNamespace(),
    )

    monkeypatch.setattr("tirzah.cli.sha256_file", lambda _path: "checksum")
    monkeypatch.setattr(
        "tirzah.cli.find_duplicate_by_checksum",
        lambda _db, _checksum: {"_id": existing_document_id},
    )

    result = ingest_source_path("db", config, source_path, labels=[])

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["reason"] == "duplicate_checksum"
    assert result["checksum_sha256"] == "checksum"
    assert result["existing_document_id"] == str(existing_document_id)
    assert result["activity_report"]["status"] == "rejected"
    assert result["activity_report"]["outcome"]["reason"] == "duplicate_checksum"
    assert result["activity_report"]["outcome"]["details"] == {
        "existing_document_id": str(existing_document_id)
    }
    assert "Status: rejected." in result["activity_log"]
    assert "Outcome reason: duplicate_checksum." in result["activity_log"]
    assert "Existing document:" in result["activity_log"]


def test_ingest_source_path_commit_duplicate_returns_activity_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.md"
    source_path.write_text("# Source\n\nBody.", encoding="utf-8")
    archive_path = tmp_path / "archive" / "source.md"
    existing_document_id = ObjectId()
    config = SimpleNamespace(
        paths=SimpleNamespace(archive=tmp_path / "archive"),
        runtime=SimpleNamespace(),
    )

    def raise_duplicate(_db, _result, embedder=None):
        raise DuplicateSourceError("commit-checksum", existing_document_id)

    monkeypatch.setattr("tirzah.cli.sha256_file", lambda _path: "checksum")
    monkeypatch.setattr("tirzah.cli.find_duplicate_by_checksum", lambda _db, _checksum: None)
    monkeypatch.setattr(
        "tirzah.cli.archive_source",
        lambda _path, _archive_root, _checksum: archive_path,
    )
    monkeypatch.setattr("tirzah.cli.embedding_adapter", lambda _runtime: "embedder")
    monkeypatch.setattr("tirzah.cli.commit_ingestion", raise_duplicate)

    result = ingest_source_path("db", config, source_path, labels=["source_root"])

    assert result["ok"] is False
    assert result["status"] == "rejected"
    assert result["reason"] == "duplicate_checksum"
    assert result["checksum_sha256"] == "commit-checksum"
    assert result["existing_document_id"] == str(existing_document_id)
    assert result["activity_report"]["status"] == "rejected"
    assert result["activity_report"]["outcome"]["details"] == {
        "existing_document_id": str(existing_document_id)
    }
    assert result["activity_report"]["semantic_processing"]["title"] == "Source"
    assert "Semantic processing: mock generated" in result["activity_log"]
    assert "Existing document:" in result["activity_log"]


def test_cli_queue_profile_backfill_alias(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "queue-profile-backfill",
            "--limit",
            "9",
            "--label",
            "target",
            "--document-id",
            "doc1",
            "--force",
            "--created-by",
            "tester",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.create_embedding_backfill_job",
        lambda _db, **kwargs: {"job_id": "job1", **kwargs},
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "job": {
            "job_id": "job1",
            "batch_limit": 9,
            "label": "target",
            "document_id": "doc1",
            "force": True,
            "created_by": "tester",
        },
    }


def test_cli_process_profile_backfill_alias(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "process-profile-backfill", "--max-batches", "4"],
    )
    config = SimpleNamespace(mongo=SimpleNamespace(), runtime=SimpleNamespace())
    embedder = SimpleNamespace(name="fake_profile")
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: config)
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.cli.embedding_adapter", lambda _runtime: embedder)
    monkeypatch.setattr(
        "tirzah.cli.process_embedding_backfill_batches",
        lambda _db, used_embedder, max_batches=1: {
            "ok": True,
            "embedder": used_embedder.name,
            "max_batches": max_batches,
        },
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "embedder": "fake_profile",
        "max_batches": 4,
    }


def test_cli_graph_status_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tirzah", "graph-status", "--limit", "3"])
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.graph_edge_status",
        lambda _db, limit=10: {
            "edge_count": 12,
            "relation_types": [{"value": "contains", "count": 12}],
            "provenance_sources": [{"value": "node_parent_link", "count": 12}],
            "limit": limit,
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "edge_count": 12,
        "relation_types": [{"value": "contains", "count": 12}],
        "provenance_sources": [{"value": "node_parent_link", "count": 12}],
        "limit": 3,
    }


def test_cli_graph_status_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "graph-status", "--limit", "3", "--format", "text"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.graph_edge_status",
        lambda _db, limit=10: {
            "edge_count": 20,
            "relation_types": [
                {"value": "contains", "count": 16},
                {"value": "related_to", "count": 4},
            ],
            "provenance_sources": [
                {"value": "node_parent_link", "count": 16},
                {"value": "semantic_candidate_review", "count": 4},
            ],
            "limit": limit,
        },
    )

    main()

    output = capsys.readouterr().out
    assert "Graph status: 20 live edge(s)" in output
    assert "Relation types:" in output
    assert "- contains: 16" in output
    assert "- related_to: 4" in output
    assert "Provenance sources:" in output
    assert "- semantic_candidate_review: 4" in output


def test_cli_list_docs_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tirzah", "list-docs", "--limit", "2"])
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_documents",
        lambda _db, limit=20: [{"document_id": "doc1", "title": "Doc", "limit": limit}],
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "documents": [{"document_id": "doc1", "title": "Doc", "limit": 2}],
    }


def test_cli_list_docs_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "list-docs", "--limit", "2", "--format", "text"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_documents",
        lambda _db, limit=20: [
            {
                "document_id": "doc1",
                "title": "Readable Doc",
                "summary": "A useful source document for review.",
                "source": {"path": "/tmp/readable.md"},
                "origin_date": "2026-03-10",
                "origin_date_source": "explicit_content",
                "ingestion_epoch": "epoch-1",
                "limit": limit,
            }
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "Documents: 1 shown" in output
    assert "1. Readable Doc" in output
    assert "id: doc1" in output
    assert "source: /tmp/readable.md" in output
    assert "origin date: 2026-03-10 (explicit_content)" in output
    assert "ingestion epoch: epoch-1" in output
    assert "summary: A useful source document for review." in output


def test_cli_expand_proximity_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "expand-proximity", "node1", "--direction", "incoming"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.expand_proximity",
        lambda _db, node_id, direction="both", relation_type=None, limit=10: [
            {"node_id": node_id, "direction": direction, "limit": limit}
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "nodes": [{"node_id": "node1", "direction": "incoming", "limit": 10}],
    }


def test_cli_expand_proximity_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "expand-proximity",
            "node1",
            "--limit",
            "2",
            "--format",
            "text",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.expand_proximity",
        lambda _db, node_id, direction="both", relation_type=None, limit=10: [
            {
                "node_id": "related1",
                "title": "Related Concept",
                "text_preview": "Related concept preview.",
                "proximity_score": 0.72,
                "edge": {
                    "edge_id": "edge1",
                    "relation_type": "related_to",
                    "provenance_source": "semantic_candidate_review",
                    "reviewer": "tester",
                    "candidate_source": "embedding_similarity",
                    "embedding_similarity": 0.91,
                    "selection_min_similarity": 0.8,
                    "embedding_model": "local-profile",
                },
            }
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "Proximity expansion: 1 match(es) shown" in output
    assert "Related Concept | related_to | score 0.72" in output
    assert "edge: edge1 | semantic_candidate_review" in output
    assert "reviewed by: tester" in output
    assert "profile evidence: similarity 0.91 | threshold 0.8 | model local-profile" in output
    assert "text: Related concept preview." in output


def test_cli_expand_graph_paths_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "expand-graph-paths",
            "node1",
            "--direction",
            "outgoing",
            "--max-depth",
            "3",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)

    def fake_expand_graph_paths(
        _db,
        node_id,
        direction="both",
        relation_type=None,
        max_depth=2,
        branch_limit=5,
        limit=10,
    ):
        return [
            {
                "node_id": node_id,
                "direction": direction,
                "max_depth": max_depth,
                "branch_limit": branch_limit,
                "limit": limit,
            }
        ]

    monkeypatch.setattr("tirzah.cli.expand_graph_paths", fake_expand_graph_paths)

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "paths": [
            {
                "node_id": "node1",
                "direction": "outgoing",
                "max_depth": 3,
                "branch_limit": 5,
                "limit": 10,
            }
        ],
    }


def test_cli_graph_explore_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "graph-explore",
            "node1",
            "--direction",
            "outgoing",
            "--relation-type",
            "related_to",
            "--max-depth",
            "2",
            "--endorsement",
            "explicit_endorsed",
            "--format",
            "text",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.graph_neighborhood",
        lambda _db, node_id, **kwargs: {
            "ok": True,
            "focus": {
                "node_id": node_id,
                "title": "Focus",
                "endorsement_label": kwargs.get("endorsement_label"),
                "origin_date": "2020-01-01",
                "provenance": {"source_path": "archive/focus.md"},
            },
            "nodes": [
                {"node_id": node_id, "title": "Focus", "hop": 0},
                {"node_id": "node2", "title": "Neighbor", "hop": 1, "provenance": {"source_path": "archive/n.md"}},
            ],
            "edges": [
                {
                    "edge_id": "edge1",
                    "source_node_id": node_id,
                    "target_node_id": "node2",
                    "relation_type": kwargs.get("relation_type"),
                    "hop": 1,
                    "provenance_source": "semantic_candidate_review",
                    "reviewer": "cello",
                }
            ],
            "diagnostics": {"max_depth": kwargs.get("max_depth"), "direction": kwargs.get("direction")},
        },
    )

    main()

    output = capsys.readouterr().out
    assert "Graph neighborhood: Focus" in output
    assert "Neighbor | related_to | hop 1" in output
    assert "reviewed by: cello" in output
    assert "provenance: archive/n.md" in output


def test_cli_graph_explore_mermaid_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "graph-explore", "abc123", "--format", "mermaid"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.graph_neighborhood",
        lambda _db, node_id, **_kwargs: {
            "ok": True,
            "focus": {"node_id": node_id, "title": "Focus", "hop": 0},
            "nodes": [
                {"node_id": node_id, "title": "Focus", "hop": 0},
                {"node_id": "def456", "title": "Neighbor", "hop": 1},
            ],
            "edges": [
                {
                    "source_node_id": node_id,
                    "target_node_id": "def456",
                    "relation_type": "supports",
                    "hop": 1,
                }
            ],
            "diagnostics": {},
        },
    )

    main()

    output = capsys.readouterr().out
    assert "flowchart LR" in output
    assert "supports" in output


def test_cli_semantic_candidates_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "semantic-candidates", "node1", "--include-same-document"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.semantic_candidate_nodes",
        lambda _db, node_id, limit=10, include_same_document=False: [
            {
                "node_id": node_id,
                "limit": limit,
                "include_same_document": include_same_document,
            }
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "nodes": [
            {
                "node_id": "node1",
                "limit": 10,
                "include_same_document": True,
            }
        ],
    }


def test_cli_create_semantic_edge_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "create-semantic-edge",
            "source1",
            "target1",
            "--relation-type",
            "supports",
            "--weight",
            "0.8",
            "--confidence",
            "0.9",
            "--reviewer",
            "cello",
            "--note",
            "Reviewed candidate.",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.create_reviewed_semantic_edge",
        lambda _db, **kwargs: {"ok": True, "edge": kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "edge": {
            "source_node_id": "source1",
            "target_node_id": "target1",
            "relation_type": "supports",
            "weight": 0.8,
            "confidence": 0.9,
            "reviewer": "cello",
            "note": "Reviewed candidate.",
        },
    }


def test_cli_enqueue_semantic_candidates_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-semantic-candidates",
            "node1",
            "--include-same-document",
            "--relation-type",
            "supports",
            "--created-by",
            "cello",
            "--limit",
            "3",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_semantic_edge_candidates",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "node_id": "node1",
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "cello",
        "limit": 3,
    }


def test_cli_embedding_smoke_command(monkeypatch, capsys) -> None:
    config = SimpleNamespace(
        mongo=SimpleNamespace(),
        runtime=SimpleNamespace(embedding_adapter="mock", embedding_model="mock-model"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "embedding-smoke",
            "Taj Mahal",
            "--adapter",
            "ollama_http",
            "--model",
            "embed-model",
            "--allow-http-diagnostic",
        ],
    )
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: config)
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr(
        "tirzah.cli.embedding_smoke_payload",
        lambda cfg, text: {
            "ok": True,
            "text": text,
            "adapter": cfg.runtime.embedding_adapter,
            "model": cfg.runtime.embedding_model,
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "text": "Taj Mahal",
        "adapter": "ollama_http",
        "model": "embed-model",
    }


def test_embedding_smoke_payload_returns_structured_adapter_errors() -> None:
    from tirzah.cli import embedding_smoke_payload

    class FailingAdapter:
        name = "failing_embedding"
        model = "broken-model"

        def embed(self, _text):
            raise RuntimeError("model unavailable")

    config = SimpleNamespace(runtime=SimpleNamespace())

    import tirzah.cli as cli

    original_factory = cli.embedding_adapter
    cli.embedding_adapter = lambda _runtime: FailingAdapter()
    try:
        output = embedding_smoke_payload(config, "text")
    finally:
        cli.embedding_adapter = original_factory

    assert output == {
        "ok": False,
        "adapter": "failing_embedding",
        "model": "broken-model",
        "error": "model unavailable",
        "error_type": "RuntimeError",
    }


def test_embedding_smoke_payload_reports_disallowed_http_adapter() -> None:
    from tirzah.cli import embedding_smoke_payload

    config = SimpleNamespace(
        runtime=SimpleNamespace(
            embedding_adapter="ollama_http",
            embedding_model="nomic-embed-text:latest",
            allow_http_ingestion_adapters=False,
        )
    )

    output = embedding_smoke_payload(config, "text")

    assert output["ok"] is False
    assert output["adapter"] == "ollama_http"
    assert output["model"] == "nomic-embed-text:latest"
    assert output["error_type"] == "EmbeddingAdapterPolicyError"
    assert "HTTP-backed" in output["error"]


def test_embedding_smoke_payload_reports_missing_local_profile_command() -> None:
    from tirzah.cli import embedding_smoke_payload

    config = SimpleNamespace(
        runtime=SimpleNamespace(
            embedding_adapter="local_command",
            embedding_model="local-profile",
            profile_command=[],
            allow_http_ingestion_adapters=False,
        )
    )

    output = embedding_smoke_payload(config, "text")

    assert output["ok"] is False
    assert output["adapter"] == "local_command"
    assert output["model"] == "local-profile"
    assert output["error_type"] == "ValueError"
    assert "runtime.profile_command" in output["error"]


def test_cli_vector_semantic_candidates_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "vector-semantic-candidates",
            "node1",
            "--include-same-document",
            "--min-similarity",
            "0.82",
            "--limit",
            "3",
            "--candidate-scan-limit",
            "500",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.embedding_candidate_report",
        lambda _db, node_id, **kwargs: {
            "ok": True,
            "nodes": [{"node_id": node_id, **kwargs}],
            "diagnostics": {"returned_count": 1, **kwargs},
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "nodes": [
            {
                "node_id": "node1",
                "include_same_document": True,
                "min_similarity": 0.82,
                "limit": 3,
                "candidate_scan_limit": 500,
            }
        ],
        "diagnostics": {
            "returned_count": 1,
            "include_same_document": True,
            "min_similarity": 0.82,
            "limit": 3,
            "candidate_scan_limit": 500,
        },
    }


def test_cli_vector_semantic_candidates_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "vector-semantic-candidates",
            "node1",
            "--min-similarity",
            "0.82",
            "--limit",
            "3",
            "--candidate-scan-limit",
            "500",
            "--format",
            "text",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.embedding_candidate_report",
        lambda _db, node_id, **_kwargs: {
            "ok": True,
            "nodes": [
                {
                    "node_id": "target1",
                    "title": "Target",
                    "embedding_similarity": 0.91,
                    "embedding_rank_score": 0.9,
                    "text_preview": "Target preview.",
                    "provenance": {"source_path": "/tmp/source.md"},
                }
            ],
            "diagnostics": {
                "focus": {
                    "node_id": node_id,
                    "title": "Focus",
                    "model": "profile-model",
                },
                "min_similarity": 0.82,
                "scanned_count": 20,
                "candidate_scan_limit": 500,
                "returned_count": 1,
                "candidate_count_before_limit": 1,
                "exclusions": {
                    "duplicate_text": 2,
                    "below_threshold": 17,
                    "invalid_embedding": 0,
                    "incompatible_embedding": 0,
                },
            },
        },
    )

    main()

    output = capsys.readouterr().out
    assert "Profile candidate preview: ready" in output
    assert "focus: Focus | profile-model" in output
    assert "threshold: 0.82" in output
    assert "scan: 20 scanned of 500 candidate limit" in output
    assert "returned: 1 shown from 1 above threshold" in output
    assert "excluded: 2 duplicate text, 17 below threshold" in output
    assert "1. Target | profile similarity 0.91" in output
    assert "rank score: 0.9" in output
    assert "source: /tmp/source.md" in output
    assert "text: Target preview." in output


def test_cli_profile_semantic_candidates_alias(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "profile-semantic-candidates",
            "node1",
            "--include-same-document",
            "--min-similarity",
            "0.82",
            "--limit",
            "3",
            "--candidate-scan-limit",
            "500",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.embedding_candidate_report",
        lambda _db, node_id, **kwargs: {
            "ok": True,
            "nodes": [{"node_id": node_id, **kwargs}],
            "diagnostics": {"returned_count": 1, **kwargs},
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["nodes"] == [
        {
            "node_id": "node1",
            "include_same_document": True,
            "min_similarity": 0.82,
            "limit": 3,
            "candidate_scan_limit": 500,
        }
    ]


def test_cli_enqueue_vector_semantic_candidates_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-vector-semantic-candidates",
            "node1",
            "--include-same-document",
            "--relation-type",
            "supports",
            "--created-by",
            "cello",
            "--min-similarity",
            "0.82",
            "--limit",
            "3",
            "--candidate-scan-limit",
            "500",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_vector_semantic_edge_candidates",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "node_id": "node1",
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "cello",
        "min_similarity": 0.82,
        "limit": 3,
        "candidate_scan_limit": 500,
    }


def test_cli_enqueue_profile_semantic_candidates_alias(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-profile-semantic-candidates",
            "node1",
            "--include-same-document",
            "--relation-type",
            "supports",
            "--created-by",
            "cello",
            "--min-similarity",
            "0.82",
            "--limit",
            "3",
            "--candidate-scan-limit",
            "500",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_vector_semantic_edge_candidates",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "node_id": "node1",
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "cello",
        "min_similarity": 0.82,
        "limit": 3,
        "candidate_scan_limit": 500,
    }


def test_cli_enqueue_vector_semantic_batch_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-vector-semantic-batch",
            "--label",
            "ams_domain",
            "--document-id",
            "doc1",
            "--focus-limit",
            "12",
            "--candidates-per-node",
            "2",
            "--include-same-document",
            "--relation-type",
            "supports",
            "--created-by",
            "cello",
            "--min-similarity",
            "0.82",
            "--candidate-scan-limit",
            "500",
            "--exclude-node-key",
            "section-1",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_vector_semantic_edge_candidate_batch",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "label": "ams_domain",
        "document_id": "doc1",
        "focus_limit": 12,
        "candidates_per_node": 2,
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "cello",
        "min_similarity": 0.82,
        "candidate_scan_limit": 500,
        "exclude_node_keys": ["section-1"],
        "dry_run": True,
    }


def test_cli_enqueue_profile_semantic_batch_alias(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-profile-semantic-batch",
            "--label",
            "ams_domain",
            "--document-id",
            "doc1",
            "--focus-limit",
            "12",
            "--candidates-per-node",
            "2",
            "--include-same-document",
            "--relation-type",
            "supports",
            "--created-by",
            "cello",
            "--min-similarity",
            "0.82",
            "--candidate-scan-limit",
            "500",
            "--exclude-node-key",
            "section-1",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_vector_semantic_edge_candidate_batch",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "label": "ams_domain",
        "document_id": "doc1",
        "focus_limit": 12,
        "candidates_per_node": 2,
        "include_same_document": True,
        "relation_type": "supports",
        "created_by": "cello",
        "min_similarity": 0.82,
        "candidate_scan_limit": 500,
        "exclude_node_keys": ["section-1"],
        "dry_run": True,
    }


def test_cli_enqueue_vector_semantic_batch_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-vector-semantic-batch",
            "--label",
            "ams_domain",
            "--focus-limit",
            "5",
            "--candidates-per-node",
            "1",
            "--dry-run",
            "--format",
            "text",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_vector_semantic_edge_candidate_batch",
        lambda _db, **_kwargs: {
            "ok": True,
            "candidate_count": 1,
            "would_enqueue_count": 1,
            "enqueued_count": 0,
            "skipped_existing_count": 0,
            "skipped_invalid_count": 0,
            "scope": {
                "label": "ams_domain",
                "document_id": None,
                "focus_limit": 5,
                "focus_node_count": 1,
                "candidates_per_node": 1,
                "min_similarity": 0.75,
                "dry_run": True,
            },
            "focus_results": [
                {
                    "node_id": "source1",
                    "title": "Source",
                    "would_enqueue_count": 1,
                    "skipped_existing_count": 0,
                    "candidate_previews": [
                        {
                            "target_node_id": "target1",
                            "target_title": "Target",
                            "embedding_similarity": 0.91,
                            "review_hint": "Review hint: likely conceptual candidate.",
                            "shared_wording": {
                                "source_word_overlap": 0.49,
                                "target_word_overlap": 0.51,
                            },
                            "source_text_preview": "Source preview.",
                            "target_text_preview": "Target preview.",
                        }
                    ],
                }
            ],
        },
    )

    main()

    output = capsys.readouterr().out
    assert "Profile candidate batch: ready" in output
    assert "mode: dry run" in output
    assert "scope: label ams_domain, document any" in output
    assert "candidates found: 1 | would queue: 1" in output
    assert "- Source | would queue 1 | existing 0" in output
    assert "-> Target | profile similarity 0.91 | Review hint: likely conceptual candidate." in output
    assert "shared wording: source 49%, target 51%" in output
    assert "source text: Source preview." in output
    assert "target text: Target preview." in output


def test_cli_contradiction_candidates_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "contradiction-candidates",
            "node1",
            "--include-same-document",
            "--min-similarity",
            "0.84",
            "--max-similarity",
            "0.96",
            "--limit",
            "3",
            "--candidate-scan-limit",
            "400",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.contradiction_candidate_report",
        lambda _db, node_id, **kwargs: {"ok": True, "node_id": node_id, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "node_id": "node1",
        "include_same_document": True,
        "min_similarity": 0.84,
        "max_similarity": 0.96,
        "limit": 3,
        "candidate_scan_limit": 400,
    }


def test_cli_enqueue_contradiction_candidates_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-contradiction-candidates",
            "node1",
            "--include-same-document",
            "--created-by",
            "cello",
            "--min-similarity",
            "0.84",
            "--max-similarity",
            "0.96",
            "--limit",
            "3",
            "--candidate-scan-limit",
            "400",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_contradiction_candidates",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "node_id": "node1",
        "include_same_document": True,
        "created_by": "cello",
        "min_similarity": 0.84,
        "max_similarity": 0.96,
        "limit": 3,
        "candidate_scan_limit": 400,
    }


def test_cli_enqueue_contradiction_batch_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-contradiction-batch",
            "--label",
            "ams_domain",
            "--document-id",
            "doc1",
            "--focus-limit",
            "12",
            "--candidates-per-node",
            "2",
            "--include-same-document",
            "--created-by",
            "cello",
            "--min-similarity",
            "0.84",
            "--max-similarity",
            "0.96",
            "--candidate-scan-limit",
            "400",
            "--exclude-node-key",
            "section-1",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_contradiction_candidate_batch",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "label": "ams_domain",
        "document_id": "doc1",
        "focus_limit": 12,
        "candidates_per_node": 2,
        "include_same_document": True,
        "created_by": "cello",
        "min_similarity": 0.84,
        "max_similarity": 0.96,
        "candidate_scan_limit": 400,
        "exclude_node_keys": ["section-1"],
        "dry_run": True,
    }


def test_cli_enqueue_contradiction_batch_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "enqueue-contradiction-batch",
            "--label",
            "ams_domain",
            "--dry-run",
            "--format",
            "text",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.enqueue_contradiction_candidate_batch",
        lambda _db, **_kwargs: {
            "ok": True,
            "candidate_count": 1,
            "would_enqueue_count": 1,
            "enqueued_count": 0,
            "skipped_existing_count": 0,
            "skipped_invalid_count": 0,
            "scope": {
                "label": "ams_domain",
                "document_id": None,
                "focus_limit": 25,
                "focus_node_count": 1,
                "candidates_per_node": 2,
                "min_similarity": 0.82,
                "max_similarity": 0.97,
                "dry_run": True,
            },
            "focus_results": [
                {
                    "node_id": "source1",
                    "title": "Source",
                    "would_enqueue_count": 1,
                    "skipped_existing_count": 0,
                    "candidate_previews": [
                        {
                            "target_node_id": "target1",
                            "target_title": "Target",
                            "embedding_similarity": 0.9,
                            "contradiction_signals": {"cue_count": 2},
                            "review_hint": "Review hint: possible contradiction (shared labels, date delta).",
                            "source_origin_date": "2020-01-01",
                            "target_origin_date": "2024-06-01",
                            "origin_date_delta_days": 1613,
                            "source_text_preview": "Source preview.",
                            "target_text_preview": "Target preview.",
                        }
                    ],
                }
            ],
        },
    )

    main()

    output = capsys.readouterr().out
    assert "Contradiction candidate batch: ready" in output
    assert "mode: dry run" in output
    assert "origin dates: 2020-01-01 -> 2024-06-01 (delta 1613d)" in output
    assert "possible contradiction" in output


def test_cli_semantic_edge_candidates_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "semantic-edge-candidates",
            "--status",
            "pending",
            "--relation-type",
            "contradicts",
            "--limit",
            "4",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_semantic_edge_candidates",
        lambda _db, status="pending", limit=20, relation_type=None: [
            {"status": status, "limit": limit, "relation_type": relation_type}
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "candidates": [{"status": "pending", "limit": 4, "relation_type": "contradicts"}],
    }


def test_cli_semantic_edge_candidates_text_format(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "semantic-edge-candidates",
            "--status",
            "pending",
            "--limit",
            "4",
            "--format",
            "text",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_semantic_edge_candidates",
        lambda _db, status="pending", limit=20, relation_type=None: [
            {
                "candidate_id": "candidate1",
                "relation_type": "related_to",
                "source_title": "Source",
                "target_title": "Target",
                "candidate_source": "embedding_similarity",
                "embedding_similarity": 0.91,
                "shared_wording": {
                    "source_word_overlap": 0.49,
                    "target_word_overlap": 0.51,
                },
                "review_hint": "Review hint: likely conceptual candidate.",
                "source_text_preview": "Source preview.",
                "target_text_preview": "Target preview.",
            }
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "Semantic edge candidates: 1 shown" in output
    assert "1. related_to | Source -> Target" in output
    assert "id: candidate1" in output
    assert "profile similarity 0.91" in output
    assert "shared wording: source 49%, target 51%" in output
    assert "Review hint: likely conceptual candidate." in output
    assert "source text: Source preview." in output
    assert "target text: Target preview." in output


def test_cli_review_semantic_edge_candidate_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "review-semantic-edge-candidate",
            "candidate1",
            "--action",
            "accept",
            "--reviewer",
            "cello",
            "--note",
            "Reviewed candidate.",
            "--weight",
            "0.8",
            "--confidence",
            "0.9",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.review_semantic_edge_candidate",
        lambda _db, **kwargs: {"ok": True, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "candidate_id": "candidate1",
        "action": "accept",
        "reviewer": "cello",
        "note": "Reviewed candidate.",
        "weight": 0.8,
        "confidence": 0.9,
    }


def test_cli_agent_identities_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tirzah", "agent-identities", "--limit", "3"])
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_agent_identities",
        lambda _db, limit=20: [{"identity_id": "tirzah_shared", "limit": limit}],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "identities": [{"identity_id": "tirzah_shared", "limit": 3}],
    }


def test_cli_agent_identity_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tirzah", "agent-identity", "tirzah_shared"])
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.get_agent_identity",
        lambda _db, identity_id: {"identity_id": identity_id},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "identity": {"identity_id": "tirzah_shared"},
    }


def test_cli_trust_weighting_profiles_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "trust-weighting-profiles", "--limit", "2"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_trust_weighting_profiles",
        lambda _db, limit=20: [{"weighting_profile_id": "default_balanced", "limit": limit}],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "profiles": [{"weighting_profile_id": "default_balanced", "limit": 2}],
    }


def test_cli_trust_weighting_profile_command_reports_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "trust-weighting-profile", "missing"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.get_trust_weighting_profile",
        lambda _db, weighting_profile_id: None,
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": False,
        "profile": None,
    }


def test_cli_trust_diagnostic_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "trust-diagnostic", "node1", "--profile-id", "default_balanced"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.trust_temporal_diagnostic_for_node",
        lambda _db, node_id, weighting_profile_id=None: {
            "node_id": node_id,
            "profile_id": weighting_profile_id,
        },
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "result": {"node_id": "node1", "profile_id": "default_balanced"},
    }


def test_cli_process_runs_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tirzah", "process-runs", "--session-id", "s1", "--status", "active", "--limit", "2"],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.list_process_runs",
        lambda _db, session_id=None, status=None, limit=20: [
            {"run_id": "run1", "session_id": session_id, "status": status, "limit": limit}
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True,
        "runs": [{"run_id": "run1", "session_id": "s1", "status": "active", "limit": 2}],
    }


def test_cli_start_process_run_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "start-process-run",
            "restart_continuity",
            "--session-id",
            "s1",
            "--identity-id",
            "tirzah_shared",
            "--current-step-id",
            "inspect_state",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.create_process_run",
        lambda _db, **kwargs: {"run_id": "run1", **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["run"]["process_id"] == "restart_continuity"
    assert output["run"]["current_step_id"] == "inspect_state"


def test_cli_update_process_run_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tirzah",
            "update-process-run",
            "run1",
            "--status",
            "exception_requested",
            "--completed-step-id",
            "inspect_state",
            "--exception-reason",
            "better path",
            "--exception-proposal",
            "skip duplicate step",
        ],
    )
    monkeypatch.setattr(
        "tirzah.cli.load_config",
        lambda _path: SimpleNamespace(mongo=SimpleNamespace()),
    )
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr(
        "tirzah.cli.update_process_run",
        lambda _db, run_id, **kwargs: {"run_id": run_id, **kwargs},
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["run"]["run_id"] == "run1"
    assert output["run"]["status"] == "exception_requested"
    assert output["run"]["exception"]["reason"] == "better path"


class FakeCollection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def find(self, query: dict, _projection: dict) -> list[dict]:
        return [row for row in self.rows if row["document_id"] == query["document_id"]]

    def distinct(self, field: str, query: dict) -> list:
        values = []
        for row in self.rows:
            if query["labels"] not in row.get("labels", []):
                continue
            value = row[field]
            if value not in values:
                values.append(value)
        return values


class HealthCollection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows


class FakeDb:
    def __init__(self, nodes: list[dict], document: dict | None = None) -> None:
        self.nodes = FakeCollection(nodes)
        self.document = document
