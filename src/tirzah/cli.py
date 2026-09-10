from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from bson import ObjectId

from tirzah.adapters.embedding import embedding_adapter
from tirzah.adapters.ingestion import ingestion_adapter
from tirzah.config import AppConfig, RuntimeConfig, load_config
from tirzah.db.client import get_database
from tirzah.db.governance import (
    PROCESS_RUN_STATUSES,
    create_process_run,
    get_agent_identity,
    get_governance_policy,
    get_process_object,
    get_process_run,
    get_trust_weighting_profile,
    list_agent_identities,
    list_governance_policies,
    list_process_objects,
    list_process_runs,
    list_trust_weighting_profiles,
    update_process_run,
)
from tirzah.db.health import memory_health_payload, render_memory_health_text
from tirzah.db.indexes import ensure_indexes
from tirzah.db.serializers import serialize_queue_job, serialize_queue_summary
from tirzah.db.repositories import (
    DuplicateSourceError,
    backfill_node_embeddings,
    backfill_schema_metadata,
    backfill_structural_graph_edges,
    commit_ingestion,
    create_reviewed_semantic_edge,
    enqueue_semantic_edge_candidates,
    enqueue_vector_semantic_edge_candidate_batch,
    enqueue_vector_semantic_edge_candidates,
    find_duplicate_by_checksum,
    document_tree,
    graph_edge_status,
    label_definitions,
    list_proposed_ingestion_trees,
    list_semantic_edge_candidates,
    promote_ingestion_tree,
    rebuild_document,
    reject_ingestion_tree,
    review_semantic_edge_candidate,
    update_document_origin_date,
    update_node_summary,
)
from tirzah.db.queue import enqueue_source, queue_summary, recent_jobs
from tirzah.ingestion.activity import (
    attach_ingestion_activity,
    ingestion_activity_fields,
    ingestion_activity_report,
)
from tirzah.ingestion.dates import analyze_source_dates, annotate_source_dates
from tirzah.ingestion.embedding_backfill import (
    create_embedding_backfill_job,
    list_embedding_backfill_jobs,
    process_embedding_backfill_batches,
)
from tirzah.ingestion.files import archive_source, move_request_file, sha256_file
from tirzah.ingestion.parser import SUPPORTED_SUFFIXES, read_text_source
from tirzah.ingestion.worker import discover_sources, process_next
from tirzah.retrieval.queries import (
    build_prompt_envelope,
    compile_context,
    expand_graph_paths,
    expand_proximity,
    get_document,
    graph_edges_for_node,
    list_documents,
    node_context,
    parse_iso_date,
    parse_iso_datetime,
    render_context_document,
    search_nodes,
    embedding_candidate_report,
    semantic_candidate_nodes,
)
from tirzah.retrieval.trust import trust_temporal_diagnostic_for_node
from tirzah.sessions.active_documents import list_active_documents
from tirzah.sessions.continuity import (
    render_restart_markdown,
    render_session_continuity_text,
    session_continuity,
)
from tirzah.sessions.exchanges import recent_exchanges
from tirzah.sessions.endorsements import (
    ENDORSEMENT_LABELS,
    list_generated_output_nodes,
    update_node_endorsement,
)
from tirzah.sessions.interaction import answer_query, build_query_embedding
from tirzah.sessions.run import run_traced_interaction
from tirzah.sessions.output_ingestion import (
    list_output_ingestion_jobs,
    process_next_output_ingestion,
)
from tirzah.sessions.registry import create_session, list_sessions


STRUCTURAL_NODE_LABELS = {"source_root", "source_section", "source_chunk"}

INIT_RUNTIME_CHOICES = {
    "1": "mock",
    "2": "ollama_cli",
    "3": "ollama_http",
    "4": "local_command",
    "5": "hoglah",
    "6": "kiro_cli",
    "7": "claude_cli",
    "8": "codex_cli",
    "9": "google_cli",
    "10": "grok_cli",
}
EXTERNAL_CLI_RUNTIMES = {"kiro_cli", "claude_cli", "codex_cli", "google_cli", "grok_cli"}


def discover_folder_sources(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and ".git" not in path.parts
    )


def chronological_folder_source_plan(root: Path) -> list[dict]:
    plan = []
    for path in discover_folder_sources(root):
        try:
            text, _source_kind = read_text_source(path)
            date_analysis = analyze_source_dates(path, text)
        except Exception as error:
            plan.append(
                {
                    "path": path,
                    "origin_date": None,
                    "origin_date_source": None,
                    "date_candidates": [],
                    "error": error.__class__.__name__,
                    "message": str(error),
                }
            )
            continue
        plan.append(
            {
                "path": path,
                "origin_date": date_analysis.get("origin_date"),
                "origin_date_source": date_analysis.get("origin_date_source"),
                "date_candidates": date_analysis.get("date_candidates") or [],
            }
        )
    return sorted(plan, key=chronological_source_sort_key)


def chronological_source_sort_key(item: dict) -> tuple[str, str]:
    origin_date = item.get("origin_date") or "9999-12-31"
    return (origin_date, str(item.get("path") or ""))


def ingest_source_path(
    db,
    config,
    path: Path,
    labels: list[str],
    ingestion_epoch: str | None = None,
) -> dict:
    checksum = sha256_file(path)
    duplicate = find_duplicate_by_checksum(db, checksum)
    if duplicate:
        rejected = {
            "ok": False,
            "path": str(path),
            "status": "rejected",
            "reason": "duplicate_checksum",
            "checksum_sha256": checksum,
            "existing_document_id": str(duplicate["_id"]),
            "message": "File rejected because identical content has already been ingested.",
        }
        report = ingestion_activity_report(
            path=path,
            status="rejected",
            checksum_sha256=checksum,
            reason="duplicate_checksum",
            message=rejected["message"],
            details={"existing_document_id": rejected["existing_document_id"]},
        )
        return attach_ingestion_activity(rejected, report)

    text, source_kind = read_text_source(path)
    result = ingestion_adapter(config.runtime).process(
        path,
        text,
        source_kind,
        extra_labels=labels,
    )
    annotate_source_dates(result, path, text)
    result.ingestion_epoch = ingestion_epoch
    archived_path = archive_source(path, config.paths.archive, checksum)
    result.source.checksum_sha256 = checksum
    result.source.archive_path = str(archived_path)
    try:
        inserted = commit_ingestion(db, result, embedder=embedding_adapter(config.runtime))
    except DuplicateSourceError as error:
        rejected = {
            "ok": False,
            "path": str(path),
            "status": "rejected",
            "reason": "duplicate_checksum",
            "checksum_sha256": error.checksum,
            "existing_document_id": str(error.existing_document_id),
            "message": "File rejected because identical content has already been ingested.",
        }
        report = ingestion_activity_report(
            path=path,
            status="rejected",
            checksum_sha256=error.checksum,
            result=result,
            reason="duplicate_checksum",
            message=rejected["message"],
            details={"existing_document_id": rejected["existing_document_id"]},
        )
        return attach_ingestion_activity(rejected, report)
    inserted["ok"] = True
    inserted["path"] = str(path)
    inserted["archive_path"] = str(archived_path)
    inserted["checksum_sha256"] = checksum
    report = ingestion_activity_report(
        path=path,
        status="completed",
        checksum_sha256=checksum,
        result=result,
        inserted=inserted,
    )
    return {**inserted, **ingestion_activity_fields(report)}


def embedding_smoke_payload(config, text: str) -> dict:
    try:
        adapter = embedding_adapter(config.runtime)
    except Exception as error:
        return {
            "ok": False,
            "adapter": getattr(config.runtime, "embedding_adapter", None),
            "model": getattr(config.runtime, "embedding_model", None),
            "error": str(error),
            "error_type": error.__class__.__name__,
        }
    try:
        embedding = adapter.embed(text)
    except Exception as error:
        return {
            "ok": False,
            "adapter": getattr(adapter, "name", None),
            "model": getattr(adapter, "model", None),
            "error": str(error),
            "error_type": error.__class__.__name__,
        }
    vector = embedding.get("vector") or []
    return {
        "ok": True,
        "adapter": embedding.get("adapter"),
        "model": embedding.get("model"),
        "dimensions": embedding.get("dimensions"),
        "source_text_hash": embedding.get("source_text_hash"),
        "vector_preview": vector[:5],
    }


def rejection_reason_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        reason = result.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def existing_document_extra_labels(db, document_id: str) -> list[str]:
    labels = set()
    for row in db.nodes.find({"document_id": ObjectId(document_id)}, {"labels": 1}):
        labels.update(row.get("labels", []))
    return sorted(label for label in labels if label not in STRUCTURAL_NODE_LABELS)


def document_ids_for_label(db, label: str) -> list[str]:
    document_ids = db.nodes.distinct("document_id", {"labels": label})
    return sorted(str(document_id) for document_id in document_ids)


def destructive_rebuild_refusal(command: str) -> dict:
    return {
        "ok": False,
        "reason": "destructive_rebuild_requires_force_replace",
        "command": command,
        "message": (
            "This command deletes and recreates existing trees/nodes. Requirements call for "
            "versioned replacement, not silent deletion. Re-run with --force-replace only for "
            "explicit maintenance work."
        ),
    }


def rebuild_document_from_existing_source(
    db,
    document_id: str,
    source_override: str | None = None,
    ingestion_epoch: str | None = None,
    runtime_config: RuntimeConfig | None = None,
    *,
    mode: str = "full",
    compare_only: bool = False,
) -> dict:
    document = get_document(db, document_id)
    if not document:
        return {
            "ok": False,
            "reason": "document_not_found",
            "document_id": document_id,
        }
    source = document.get("source", {})
    source_path = Path(
        source_override
        or source.get("archive_path")
        or source.get("path")
        or ""
    )
    if not source_path.exists():
        return {
            "ok": False,
            "reason": "source_missing",
            "document_id": document_id,
            "path": str(source_path),
        }
    text, source_kind = read_text_source(source_path)
    adapter_path = Path(source.get("path") or source_path)
    result = ingestion_adapter(runtime_config).process(
        adapter_path,
        text,
        source_kind,
        extra_labels=existing_document_extra_labels(db, document_id),
    )
    annotate_source_dates(result, adapter_path, text)
    result.source.path = source.get("path") or str(source_path)
    result.source.checksum_sha256 = source.get("checksum_sha256") or sha256_file(source_path)
    result.source.archive_path = source.get("archive_path") or str(source_path)
    result.ingestion_epoch = ingestion_epoch
    inserted = rebuild_document(
        db,
        document_id,
        result,
        mode=mode,
        compare_only=compare_only,
    )
    inserted["ok"] = True
    inserted["source_path"] = str(source_path)
    inserted["checksum_sha256"] = result.source.checksum_sha256
    return inserted


def render_semantic_edge_candidates_text(candidates: list[dict]) -> str:
    if not candidates:
        return "No semantic edge candidates found."
    lines = [f"Semantic edge candidates: {len(candidates)} shown"]
    for index, candidate in enumerate(candidates, start=1):
        lines.append("")
        lines.append(
            f"{index}. {candidate.get('relation_type') or 'relation'} | "
            f"{candidate.get('source_title') or candidate.get('source_node_id')} -> "
            f"{candidate.get('target_title') or candidate.get('target_node_id')}"
        )
        lines.append(f"   id: {candidate.get('candidate_id')}")
        evidence = (
            f"profile similarity {candidate.get('embedding_similarity')}"
            if candidate.get("candidate_source") == "embedding_similarity"
            else f"labels {', '.join(candidate.get('shared_labels') or [])}"
        )
        lines.append(
            f"   source: {candidate.get('candidate_source') or 'label_overlap'} | {evidence}"
        )
        shared_wording = candidate.get("shared_wording") or {}
        if shared_wording:
            lines.append(
                "   shared wording: "
                f"source {percent(shared_wording.get('source_word_overlap'))}, "
                f"target {percent(shared_wording.get('target_word_overlap'))}"
            )
        if candidate.get("review_hint"):
            lines.append(f"   {candidate['review_hint']}")
        if candidate.get("source_text_preview"):
            lines.append(f"   source text: {candidate['source_text_preview']}")
        if candidate.get("target_text_preview"):
            lines.append(f"   target text: {candidate['target_text_preview']}")
    return "\n".join(lines)


def render_proximity_text(nodes: list[dict]) -> str:
    if not nodes:
        return "No proximity matches found."
    lines = [f"Proximity expansion: {len(nodes)} match(es) shown"]
    for index, node in enumerate(nodes, start=1):
        edge = node.get("edge") or {}
        lines.append("")
        lines.append(
            f"{index}. {node.get('title') or node.get('node_id')} | "
            f"{edge.get('relation_type') or 'relation'} | score {node.get('proximity_score')}"
        )
        lines.append(f"   node: {node.get('node_id')}")
        lines.append(
            f"   edge: {edge.get('edge_id')} | "
            f"{edge.get('provenance_source') or 'unknown source'}"
        )
        if edge.get("reviewer"):
            lines.append(f"   reviewed by: {edge.get('reviewer')}")
        if edge.get("candidate_source") == "embedding_similarity":
            lines.append(
                "   profile evidence: "
                f"similarity {edge.get('embedding_similarity')} | "
                f"threshold {edge.get('selection_min_similarity')} | "
                f"model {edge.get('embedding_model')}"
            )
        if node.get("text_preview"):
            lines.append(f"   text: {node['text_preview']}")
    return "\n".join(lines)


def render_documents_text(documents: list[dict]) -> str:
    if not documents:
        return "No documents found."
    lines = [f"Documents: {len(documents)} shown"]
    for index, document in enumerate(documents, start=1):
        source = document.get("source") or {}
        lines.append("")
        lines.append(f"{index}. {document.get('title') or document.get('document_id')}")
        lines.append(f"   id: {document.get('document_id')}")
        if source.get("path"):
            lines.append(f"   source: {source.get('path')}")
        if document.get("origin_date"):
            lines.append(
                f"   origin date: {document.get('origin_date')} "
                f"({document.get('origin_date_source') or 'unknown source'})"
            )
        if document.get("ingestion_epoch"):
            lines.append(f"   ingestion epoch: {document.get('ingestion_epoch')}")
        summary = str(document.get("summary") or "").strip()
        if summary:
            lines.append(f"   summary: {summary[:280]}")
    return "\n".join(lines)


def render_graph_status_text(status: dict) -> str:
    lines = [f"Graph status: {status.get('edge_count', 0)} live edge(s)"]
    relation_types = status.get("relation_types") or []
    if relation_types:
        lines.append("")
        lines.append("Relation types:")
        for item in relation_types:
            lines.append(f"- {item.get('value') or '(none)'}: {item.get('count', 0)}")
    provenance_sources = status.get("provenance_sources") or []
    if provenance_sources:
        lines.append("")
        lines.append("Provenance sources:")
        for item in provenance_sources:
            lines.append(f"- {item.get('value') or '(none)'}: {item.get('count', 0)}")
    return "\n".join(lines)


def render_vector_semantic_candidates_text(report: dict) -> str:
    diagnostics = report.get("diagnostics") or {}
    focus = diagnostics.get("focus") or {}
    exclusions = diagnostics.get("exclusions") or {}
    lines = [
        f"Profile candidate preview: {'ready' if report.get('ok') else 'needs attention'}",
    ]
    if report.get("reason"):
        lines.append(f"reason: {report['reason']}")
    if focus:
        lines.append(
            f"focus: {focus.get('title') or focus.get('node_id')} | "
            f"{focus.get('model') or 'unknown model'}"
        )
    lines.extend(
        [
            f"threshold: {diagnostics.get('min_similarity')}",
            (
                f"scan: {diagnostics.get('scanned_count', 0)} scanned of "
                f"{diagnostics.get('candidate_scan_limit')} candidate limit"
            ),
            (
                "scan status: truncated; raise the candidate scan limit or choose a narrower scope"
                if diagnostics.get("scan_truncated")
                else "scan status: complete within the candidate scan limit"
            ),
            (
                f"returned: {diagnostics.get('returned_count', 0)} shown from "
                f"{diagnostics.get('candidate_count_before_limit', 0)} above threshold"
            ),
            (
                "excluded: "
                f"{exclusions.get('duplicate_text', 0)} duplicate text, "
                f"{exclusions.get('below_threshold', 0)} below threshold, "
                f"{exclusions.get('invalid_embedding', 0)} invalid profile, "
                f"{exclusions.get('incompatible_embedding', 0)} incompatible profile"
            ),
        ]
    )
    nodes = report.get("nodes") or []
    for index, node in enumerate(nodes, start=1):
        lines.append("")
        lines.append(
            f"{index}. {node.get('title') or node.get('node_id')} | "
            f"profile similarity {node.get('embedding_similarity')}"
        )
        if node.get("embedding_rank_score") is not None:
            lines.append(f"   rank score: {node.get('embedding_rank_score')}")
        provenance = node.get("provenance") if isinstance(node.get("provenance"), dict) else {}
        if provenance.get("source_path"):
            lines.append(f"   source: {provenance['source_path']}")
        if node.get("text_preview"):
            lines.append(f"   text: {node['text_preview']}")
    return "\n".join(lines)


def render_vector_semantic_batch_text(result: dict) -> str:
    scope = result.get("scope") or {}
    lines = [
        f"Profile candidate batch: {'ready' if result.get('ok') else 'needs attention'}",
        f"mode: {'dry run' if scope.get('dry_run') else 'queue pending review rows'}",
        f"scope: label {scope.get('label') or 'any'}, document {scope.get('document_id') or 'any'}",
        (
            f"focus nodes: {scope.get('focus_node_count', 0)} of limit {scope.get('focus_limit')} | "
            f"per focus: {scope.get('candidates_per_node')} | threshold {scope.get('min_similarity')}"
        ),
        (
            f"candidates found: {result.get('candidate_count', 0)} | "
            f"would queue: {result.get('would_enqueue_count', 0)} | "
            f"queued: {result.get('enqueued_count', 0)} | "
            f"existing: {result.get('skipped_existing_count', 0)} | "
            f"invalid: {result.get('skipped_invalid_count', 0)}"
        ),
    ]
    if result.get("reason"):
        lines.append(f"reason: {result['reason']}")
    focus_results = result.get("focus_results") or []
    shown = 0
    for focus in focus_results:
        previews = focus.get("candidate_previews") or []
        if not previews:
            continue
        lines.append("")
        lines.append(
            f"- {focus.get('title') or focus.get('node_id')} | "
            f"would queue {focus.get('would_enqueue_count', 0)} | "
            f"existing {focus.get('skipped_existing_count', 0)}"
        )
        for candidate in previews[:3]:
            lines.append(
                f"  -> {candidate.get('target_title') or candidate.get('target_node_id')} | "
                f"profile similarity {candidate.get('embedding_similarity')} | "
                f"{candidate.get('review_hint') or ''}".rstrip()
            )
            shared_wording = candidate.get("shared_wording") or {}
            if shared_wording:
                lines.append(
                    "     shared wording: "
                    f"source {percent(shared_wording.get('source_word_overlap'))}, "
                    f"target {percent(shared_wording.get('target_word_overlap'))}"
                )
            if candidate.get("source_text_preview"):
                lines.append(f"     source text: {candidate['source_text_preview']}")
            if candidate.get("target_text_preview"):
                lines.append(f"     target text: {candidate['target_text_preview']}")
        shown += 1
        if shown >= 8:
            break
    return "\n".join(lines)


def percent(value: object) -> str:
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "n/a"


def add_profile_backfill_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--limit", type=int, default=100)
    command.add_argument("--label", default=None)
    command.add_argument("--document-id", default=None)
    command.add_argument("--force", action="store_true")


def add_profile_candidate_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("node_id")
    command.add_argument("--include-same-document", action="store_true")
    command.add_argument("--min-similarity", type=float, default=0.75)
    command.add_argument("--limit", type=int, default=10)
    command.add_argument("--candidate-scan-limit", type=int, default=None)
    command.add_argument("--format", choices=["json", "text"], default="json")


def add_enqueue_profile_candidate_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("node_id")
    command.add_argument("--include-same-document", action="store_true")
    command.add_argument("--relation-type", default="related_to")
    command.add_argument("--created-by", default="user")
    command.add_argument("--min-similarity", type=float, default=0.75)
    command.add_argument("--limit", type=int, default=10)
    command.add_argument("--candidate-scan-limit", type=int, default=None)


def add_enqueue_profile_batch_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--label", default=None)
    command.add_argument("--document-id", default=None)
    command.add_argument("--focus-limit", type=int, default=25)
    command.add_argument("--candidates-per-node", type=int, default=2)
    command.add_argument("--include-same-document", action="store_true")
    command.add_argument("--relation-type", default="related_to")
    command.add_argument("--created-by", default="user")
    command.add_argument("--min-similarity", type=float, default=0.75)
    command.add_argument("--candidate-scan-limit", type=int, default=None)
    command.add_argument("--exclude-node-key", action="append", default=[])
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--format", choices=["json", "text"], default="json")


def init_config_payload(
    *,
    docker: bool = False,
    runtime_choice: str = "mock",
    database: str = "tirzah",
) -> dict:
    config = AppConfig()
    config.mongo.database = database
    if docker:
        config.mongo.uri = "mongodb://mongo:27017"
    if runtime_choice == "mock":
        config.runtime.answer_adapter = "mock"
        config.runtime.memory_agent_adapter = "mock"
        config.runtime.embedding_adapter = "mock"
    elif runtime_choice == "ollama_cli":
        config.runtime.answer_adapter = "ollama_cli"
        config.runtime.memory_agent_adapter = None
        config.runtime.embedding_adapter = "mock"
    elif runtime_choice == "ollama_http":
        config.runtime.answer_adapter = "ollama_http"
        config.runtime.memory_agent_adapter = "ollama_cli"
        config.runtime.embedding_adapter = "mock"
        config.runtime.ollama_base_url = "http://host.docker.internal:11434" if docker else "http://localhost:11434"
    elif runtime_choice == "hoglah":
        config.runtime.answer_adapter = "hoglah"
        config.runtime.memory_agent_adapter = None
        config.runtime.embedding_adapter = "mock"
        config.runtime.hoglah_ollama_host = "http://host.docker.internal:11434" if docker else "http://localhost:11434"
    elif runtime_choice in EXTERNAL_CLI_RUNTIMES:
        config.runtime.answer_adapter = runtime_choice
        config.runtime.memory_agent_adapter = None
        config.runtime.embedding_adapter = "mock"
        config.runtime.ingestion_model_adapter = runtime_choice
    elif runtime_choice == "local_command":
        config.runtime.answer_adapter = "mock"
        config.runtime.memory_agent_adapter = "mock"
        config.runtime.embedding_adapter = "local_command"
        config.runtime.embedding_model = "BAAI/bge-small-en-v1.5"
        config.runtime.embedding_dimensions = 384
        config.runtime.profile_command = [
            "tirzah-profile-helper",
            "--worker",
        ]
        config.runtime.profile_command_mode = "worker"
    else:
        raise ValueError(f"Unknown runtime choice: {runtime_choice}")
    return json.loads(config.model_dump_json())


def interactive_runtime_choice(default: str = "mock") -> str:
    print("Choose Tirzah runtime defaults:")
    print("  1. mock - deterministic local diagnostics")
    print("  2. ollama_cli - call a local Ollama executable")
    print("  3. ollama_http - call an existing Ollama HTTP server")
    print("  4. local_command - mock answers plus local profile helper")
    print("  5. hoglah - queue answers through the optional Hoglah package")
    print("  6. kiro_cli - call headless Kiro CLI (optional; not local-only)")
    print("  7. claude_cli - call headless Claude Code (`claude -p`)")
    print("  8. codex_cli - call headless Codex CLI (`codex exec`)")
    print("  9. google_cli - call headless Gemini CLI (`gemini -p`)")
    print(" 10. grok_cli - call headless Grok CLI (`grok --prompt-file`)")
    answer = input(f"Runtime [1-10, default {runtime_choice_label(default)}]: ").strip()
    if not answer:
        return default
    return INIT_RUNTIME_CHOICES.get(answer, default)


def runtime_choice_label(choice: str) -> str:
    for key, value in INIT_RUNTIME_CHOICES.items():
        if value == choice:
            return key
    return "1"


def write_initial_config(
    config_path: Path,
    *,
    docker: bool = False,
    force: bool = False,
    non_interactive: bool = False,
    runtime_choice: str | None = None,
) -> dict:
    if config_path.exists() and not force:
        return {
            "ok": False,
            "reason": "config_exists",
            "path": str(config_path),
            "message": "Config already exists. Re-run with --force to replace it.",
        }
    selected_runtime = runtime_choice or ("mock" if non_interactive else interactive_runtime_choice())
    payload = init_config_payload(docker=docker, runtime_choice=selected_runtime)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    data_paths = [Path(value) for value in payload["paths"].values()]
    for path in data_paths:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "path": str(config_path),
        "docker": docker,
        "runtime": selected_runtime,
        "data_paths": [str(path) for path in data_paths],
    }


def serve_app(host: str, port: int, reload: bool = False) -> None:
    try:
        import uvicorn  # noqa: F401  (also gates fastapi, imported by the app)
        import fastapi  # noqa: F401
    except ImportError:
        raise SystemExit(
            "tirzah serve needs the web extra. Install it:  pip install 'tirzah[web]'"
        )
    import uvicorn

    uvicorn.run("tirzah.web.app:app", host=host, port=port, reload=reload)


def _add_cmd(sub, name: str, **kwargs):
    """add_parser with a default help= so `tirzah --help` is usable (review F5)."""
    kwargs.setdefault("help", name.replace("-", " "))
    return sub.add_parser(name, **kwargs)


class _SubcommandParser(argparse.ArgumentParser):
    """Print this subcommand's usage on error, not the top-level command list."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tirzah")
    parser.add_argument("--config", default="config.yaml")

    subcommands = parser.add_subparsers(
        dest="command", required=True, parser_class=_SubcommandParser
    )
    init = _add_cmd(subcommands, "init")
    init.add_argument("--docker", action="store_true")
    init.add_argument("--force", action="store_true")
    init.add_argument("--non-interactive", action="store_true")
    init.add_argument(
        "--runtime",
        choices=[
            "mock",
            "ollama_cli",
            "ollama_http",
            "local_command",
            "hoglah",
            "kiro_cli",
            "claude_cli",
            "codex_cli",
            "google_cli",
            "grok_cli",
        ],
        default=None,
        help="Runtime defaults to write. Interactive mode prompts when omitted.",
    )
    serve = _add_cmd(subcommands, "serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--reload", action="store_true")
    _add_cmd(subcommands, "db-ping")
    migrate_parser = _add_cmd(subcommands, 
        "migrate",
        help="Run all pending schema migrations in order (idempotent); the "
        "consolidated entry point for the backfill-* one-shots.",
    )
    migrate_parser.add_argument("--status", action="store_true",
                                help="Show applied + pending migrations without running anything.")
    migrate_parser.add_argument("--dry-run", action="store_true",
                                help="List the migrations that would run, without applying them.")
    _add_cmd(subcommands, "backfill-source-metadata")
    _add_cmd(subcommands, "backfill-schema-metadata")
    graph_status = _add_cmd(subcommands, "graph-status")
    graph_status.add_argument("--limit", type=int, default=10)
    graph_status.add_argument("--format", choices=["json", "text"], default="json")
    structural_edges = _add_cmd(subcommands, "backfill-structural-graph-edges")
    structural_edges.add_argument("--limit", type=int, default=None)
    embedding_backfill = _add_cmd(subcommands, "backfill-embeddings")
    add_profile_backfill_arguments(embedding_backfill)
    turn_embedding_backfill = _add_cmd(subcommands, "backfill-turn-embeddings")
    turn_embedding_backfill.add_argument("--limit", type=int, default=1000)
    chunk_backfill = _add_cmd(subcommands, "backfill-chunks")
    chunk_backfill.add_argument("--limit", type=int, default=1000)
    profile_backfill = _add_cmd(subcommands, "backfill-profiles")
    add_profile_backfill_arguments(profile_backfill)
    embedding_jobs = _add_cmd(subcommands, "embedding-backfill-jobs")
    embedding_jobs.add_argument("--status", default=None)
    embedding_jobs.add_argument("--limit", type=int, default=20)
    profile_jobs = _add_cmd(subcommands, "profile-backfill-jobs")
    profile_jobs.add_argument("--status", default=None)
    profile_jobs.add_argument("--limit", type=int, default=20)
    queue_embedding_job = _add_cmd(subcommands, "queue-embedding-backfill")
    queue_embedding_job.add_argument("--limit", type=int, default=100)
    queue_embedding_job.add_argument("--label", default=None)
    queue_embedding_job.add_argument("--document-id", default=None)
    queue_embedding_job.add_argument("--force", action="store_true")
    queue_embedding_job.add_argument("--created-by", default="cli")
    queue_profile_job = _add_cmd(subcommands, "queue-profile-backfill")
    queue_profile_job.add_argument("--limit", type=int, default=100)
    queue_profile_job.add_argument("--label", default=None)
    queue_profile_job.add_argument("--document-id", default=None)
    queue_profile_job.add_argument("--force", action="store_true")
    queue_profile_job.add_argument("--created-by", default="cli")
    process_embedding_job = _add_cmd(subcommands, "process-embedding-backfill")
    process_embedding_job.add_argument("--max-batches", type=int, default=1)
    process_profile_job = _add_cmd(subcommands, "process-profile-backfill")
    process_profile_job.add_argument("--max-batches", type=int, default=1)
    _add_cmd(subcommands, "enqueue-inbox")
    _add_cmd(subcommands, "process-next")
    _add_cmd(subcommands, "process-inbox")
    _add_cmd(subcommands, "queue-status")
    memory_health = _add_cmd(subcommands, "memory-health")
    memory_health.add_argument("--format", choices=["json", "text"], default="text")
    config_status = _add_cmd(subcommands, 
        "config-status",
        help="Show the resolved runtime: active config file, adapters + their "
        "capabilities, models, embedding dims, and the Mahalath seam state.",
    )
    config_status.add_argument("--format", choices=["json", "text"], default="text")
    _add_cmd(subcommands, "labels")
    _add_cmd(subcommands, "sessions")
    embedding_smoke = _add_cmd(subcommands, "embedding-smoke")
    embedding_smoke.add_argument("text")
    embedding_smoke.add_argument(
        "--adapter",
        choices=["mock", "local_command", "ollama_http", "ollama_powershell", "hoglah"],
        default=None,
    )
    embedding_smoke.add_argument("--model", default=None)
    embedding_smoke.add_argument("--allow-http-diagnostic", action="store_true")

    agent_identities = _add_cmd(subcommands, "agent-identities")
    agent_identities.add_argument("--limit", type=int, default=20)
    agent_identity = _add_cmd(subcommands, "agent-identity")
    agent_identity.add_argument("identity_id")

    trust_profiles = _add_cmd(subcommands, "trust-weighting-profiles")
    trust_profiles.add_argument("--limit", type=int, default=20)
    trust_profile = _add_cmd(subcommands, "trust-weighting-profile")
    trust_profile.add_argument("weighting_profile_id")
    trust_diagnostic = _add_cmd(subcommands, "trust-diagnostic")
    trust_diagnostic.add_argument("node_id")
    trust_diagnostic.add_argument("--profile-id", default=None)

    governance_policies = _add_cmd(subcommands, "governance-policies")
    governance_policies.add_argument("--limit", type=int, default=20)
    governance_policy = _add_cmd(subcommands, "governance-policy")
    governance_policy.add_argument("policy_id")

    process_objects = _add_cmd(subcommands, "process-objects")
    process_objects.add_argument("--limit", type=int, default=20)
    process_object = _add_cmd(subcommands, "process-object")
    process_object.add_argument("process_id")

    process_runs = _add_cmd(subcommands, "process-runs")
    process_runs.add_argument("--session-id", default=None)
    process_runs.add_argument("--status", default=None, choices=sorted(PROCESS_RUN_STATUSES))
    process_runs.add_argument("--limit", type=int, default=20)
    process_run = _add_cmd(subcommands, "process-run")
    process_run.add_argument("run_id")
    start_process_run = _add_cmd(subcommands, "start-process-run")
    start_process_run.add_argument("process_id")
    start_process_run.add_argument("--session-id", default="default")
    start_process_run.add_argument("--identity-id", default=None)
    start_process_run.add_argument("--current-step-id", default=None)
    start_process_run.add_argument("--status", default="active", choices=sorted(PROCESS_RUN_STATUSES))
    update_process_run_parser = _add_cmd(subcommands, "update-process-run")
    update_process_run_parser.add_argument("run_id")
    update_process_run_parser.add_argument("--status", default=None, choices=sorted(PROCESS_RUN_STATUSES))
    update_process_run_parser.add_argument("--current-step-id", default=None)
    update_process_run_parser.add_argument("--completed-step-id", default=None)
    update_process_run_parser.add_argument("--exchange-id", default=None)
    update_process_run_parser.add_argument("--exception-reason", default=None)
    update_process_run_parser.add_argument("--exception-proposal", default=None)
    update_process_run_parser.add_argument("--exception-reviewer", default=None)
    update_process_run_parser.add_argument("--exception-note", default=None)

    active_documents = _add_cmd(subcommands, "active-documents")
    active_documents.add_argument("--session-id", default="default")
    active_documents.add_argument("--limit", type=int, default=20)
    continuity = _add_cmd(subcommands, "session-continuity")
    continuity.add_argument("--session-id", default="default")
    continuity.add_argument("--limit", type=int, default=5)
    continuity.add_argument("--format", choices=["json", "text"], default="text")
    restart_render = _add_cmd(subcommands, "restart-render")
    restart_render.add_argument("--session-id", default="default")
    restart_render.add_argument("--limit", type=int, default=5)
    restart_render.add_argument(
        "--output", default=None, help="Write the rendered markdown to this path instead of stdout."
    )

    output_jobs = _add_cmd(subcommands, "output-ingestion")
    output_jobs.add_argument("--session-id", default=None)
    output_jobs.add_argument("--status", default=None)
    output_jobs.add_argument("--limit", type=int, default=20)
    process_output_jobs = _add_cmd(subcommands, "process-output-ingestion")
    process_output_jobs.add_argument("--session-id", default=None)
    process_output_jobs.add_argument("--job-id", default=None)

    review_outputs = _add_cmd(subcommands, "review-generated-output")
    review_outputs.add_argument("--endorsement", default=None, choices=sorted(ENDORSEMENT_LABELS))
    review_outputs.add_argument("--limit", type=int, default=20)

    endorse_node = _add_cmd(subcommands, "endorse-node")
    endorse_node.add_argument("node_id")
    endorse_node.add_argument("--endorsement", required=True, choices=sorted(ENDORSEMENT_LABELS))
    endorse_node.add_argument("--reviewer", default="user")
    endorse_node.add_argument("--note", default=None)

    list_proposed = _add_cmd(
        subcommands,
        "list-proposed-ingestions",
        help="List LLM-proposed ingestion trees waiting for review.",
    )
    list_proposed.add_argument("--limit", type=int, default=20)
    promote_ingestion = _add_cmd(
        subcommands,
        "promote-ingestion",
        help="Promote a pending LLM-proposed tree so retrieval can use it.",
    )
    promote_ingestion.add_argument("identifier", help="Tree id or document id.")
    promote_ingestion.add_argument("--reviewer", default="user")
    promote_ingestion.add_argument("--note", default=None)
    promote_ingestion.add_argument(
        "--endorsement",
        default=None,
        choices=sorted(ENDORSEMENT_LABELS),
        help="Optional endorsement to stamp on the promoted nodes.",
    )
    reject_ingestion = _add_cmd(
        subcommands,
        "reject-ingestion",
        help="Reject a pending LLM-proposed tree (excluded from retrieval).",
    )
    reject_ingestion.add_argument("identifier", help="Tree id or document id.")
    reject_ingestion.add_argument("--reviewer", default="user")
    reject_ingestion.add_argument("--note", default=None)

    create_session_cmd = _add_cmd(subcommands, "create-session")
    create_session_cmd.add_argument("--title", default=None)
    create_session_cmd.add_argument("--session-id", default=None)

    list_docs = _add_cmd(subcommands, "list-docs")
    list_docs.add_argument("--limit", type=int, default=20)
    list_docs.add_argument("--format", choices=["json", "text"], default="json")

    show_doc = _add_cmd(subcommands, "show-doc")
    show_doc.add_argument("document_id")

    search = _add_cmd(subcommands, "search-nodes")
    search.add_argument("--query", default=None)
    search.add_argument("--label", default=None)
    search.add_argument("--endorsement", default=None)
    search.add_argument("--document-id", default=None)
    search.add_argument("--created-after", default=None)
    search.add_argument("--created-before", default=None)
    search.add_argument(
        "--origin-after",
        default=None,
        help="Keep nodes whose origin_date is on or after this date (YYYY-MM-DD).",
    )
    search.add_argument(
        "--origin-before",
        default=None,
        help="Keep nodes whose origin_date is on or before this date (YYYY-MM-DD).",
    )
    search.add_argument("--limit", type=int, default=20)
    search.add_argument(
        "--trust-ranking",
        action="store_true",
        help="Apply opt-in trust/temporal ranking after lexical/hybrid order.",
    )
    search.add_argument(
        "--trust-profile",
        default=None,
        help="Trust weighting profile id (default: runtime.trust_weighting_profile or identity).",
    )

    set_origin = _add_cmd(
        subcommands,
        "set-origin-date",
        help="Correct a document origin date and stamp active nodes with provenance.",
    )
    set_origin.add_argument("document_id")
    set_origin.add_argument("--date", required=True, help="Origin date (YYYY-MM-DD or a parseable date).")
    set_origin.add_argument("--reviewer", default="user")
    set_origin.add_argument("--note", default=None)

    set_summary = _add_cmd(
        subcommands,
        "set-node-summary",
        help="Set a provenance-tagged summary used when context budget skips a node.",
    )
    set_summary.add_argument("node_id")
    set_summary.add_argument("--summary", required=True)
    set_summary.add_argument("--reviewer", default="user")
    set_summary.add_argument("--note", default=None)

    context = _add_cmd(subcommands, "node-context")
    context.add_argument("node_id")
    context.add_argument("--child-limit", type=int, default=20)

    graph_edges = _add_cmd(subcommands, "graph-edges")
    graph_edges.add_argument("node_id")
    graph_edges.add_argument("--direction", choices=["incoming", "outgoing", "both"], default="both")
    graph_edges.add_argument("--relation-type", default=None)
    graph_edges.add_argument("--limit", type=int, default=10)

    proximity = _add_cmd(subcommands, "expand-proximity")
    proximity.add_argument("node_id")
    proximity.add_argument("--direction", choices=["incoming", "outgoing", "both"], default="both")
    proximity.add_argument("--relation-type", default=None)
    proximity.add_argument("--limit", type=int, default=10)
    proximity.add_argument("--format", choices=["json", "text"], default="json")

    graph_paths = _add_cmd(subcommands, "expand-graph-paths")
    graph_paths.add_argument("node_id")
    graph_paths.add_argument("--direction", choices=["incoming", "outgoing", "both"], default="both")
    graph_paths.add_argument("--relation-type", default=None)
    graph_paths.add_argument("--max-depth", type=int, default=2)
    graph_paths.add_argument("--branch-limit", type=int, default=5)
    graph_paths.add_argument("--limit", type=int, default=10)

    semantic_candidates = _add_cmd(subcommands, "semantic-candidates")
    semantic_candidates.add_argument("node_id")
    semantic_candidates.add_argument("--include-same-document", action="store_true")
    semantic_candidates.add_argument("--limit", type=int, default=10)

    vector_candidates = _add_cmd(subcommands, "vector-semantic-candidates")
    add_profile_candidate_arguments(vector_candidates)
    profile_candidates = _add_cmd(subcommands, "profile-semantic-candidates")
    add_profile_candidate_arguments(profile_candidates)

    enqueue_semantic = _add_cmd(subcommands, "enqueue-semantic-candidates")
    enqueue_semantic.add_argument("node_id")
    enqueue_semantic.add_argument("--include-same-document", action="store_true")
    enqueue_semantic.add_argument("--relation-type", default="related_to")
    enqueue_semantic.add_argument("--created-by", default="user")
    enqueue_semantic.add_argument("--limit", type=int, default=10)

    enqueue_vector_semantic = _add_cmd(subcommands, "enqueue-vector-semantic-candidates")
    add_enqueue_profile_candidate_arguments(enqueue_vector_semantic)
    enqueue_profile_semantic = _add_cmd(subcommands, "enqueue-profile-semantic-candidates")
    add_enqueue_profile_candidate_arguments(enqueue_profile_semantic)

    enqueue_vector_semantic_batch = _add_cmd(subcommands, 
        "enqueue-vector-semantic-batch"
    )
    add_enqueue_profile_batch_arguments(enqueue_vector_semantic_batch)
    enqueue_profile_semantic_batch = _add_cmd(subcommands, 
        "enqueue-profile-semantic-batch"
    )
    add_enqueue_profile_batch_arguments(enqueue_profile_semantic_batch)

    semantic_queue = _add_cmd(subcommands, "semantic-edge-candidates")
    semantic_queue.add_argument("--status", default="pending")
    semantic_queue.add_argument("--limit", type=int, default=20)
    semantic_queue.add_argument("--format", choices=["json", "text"], default="json")

    semantic_candidate_review = _add_cmd(subcommands, "review-semantic-edge-candidate")
    semantic_candidate_review.add_argument("candidate_id")
    semantic_candidate_review.add_argument("--action", required=True, choices=["accept", "reject"])
    semantic_candidate_review.add_argument("--reviewer", default="user")
    semantic_candidate_review.add_argument("--note", default=None)
    semantic_candidate_review.add_argument("--weight", type=float, default=0.7)
    semantic_candidate_review.add_argument("--confidence", type=float, default=0.6)

    semantic_edge = _add_cmd(subcommands, "create-semantic-edge")
    semantic_edge.add_argument("source_node_id")
    semantic_edge.add_argument("target_node_id")
    semantic_edge.add_argument("--relation-type", default="related_to")
    semantic_edge.add_argument("--weight", type=float, default=0.7)
    semantic_edge.add_argument("--confidence", type=float, default=0.6)
    semantic_edge.add_argument("--reviewer", default="user")
    semantic_edge.add_argument("--note", default=None)

    compile_ctx = _add_cmd(subcommands, "compile-context")
    compile_ctx.add_argument("node_id")
    compile_ctx.add_argument("--ancestor-depth", type=int, default=3)
    compile_ctx.add_argument("--sibling-window", type=int, default=1)
    compile_ctx.add_argument("--child-depth", type=int, default=1)
    compile_ctx.add_argument("--child-limit", type=int, default=20)

    render_ctx = _add_cmd(subcommands, "render-context")
    render_ctx.add_argument("node_id")
    render_ctx.add_argument("--ancestor-depth", type=int, default=3)
    render_ctx.add_argument("--sibling-window", type=int, default=1)
    render_ctx.add_argument("--child-depth", type=int, default=1)
    render_ctx.add_argument("--child-limit", type=int, default=20)
    render_ctx.add_argument("--char-budget", type=int, default=None)
    render_ctx.add_argument("--json", action="store_true")

    prompt = _add_cmd(subcommands, "build-prompt")
    prompt.add_argument("node_id")
    prompt.add_argument("--query", required=True)
    prompt.add_argument("--ancestor-depth", type=int, default=3)
    prompt.add_argument("--sibling-window", type=int, default=1)
    prompt.add_argument("--child-depth", type=int, default=1)
    prompt.add_argument("--child-limit", type=int, default=20)
    prompt.add_argument("--token-budget", type=int, default=None)
    prompt.add_argument("--reserved-response-tokens", type=int, default=None)
    prompt.add_argument("--system-instruction", default=None)
    prompt.add_argument("--text", action="store_true")

    ask = _add_cmd(subcommands, "ask")
    ask.add_argument("query")
    ask.add_argument("--node-id", default=None)
    ask.add_argument("--session-id", default="default")
    ask.add_argument("--adapter", default=None)
    ask.add_argument("--model", default=None)
    ask.add_argument("--retrieval-mode", choices=["direct", "agentic", "deep"], default=None)
    ask.add_argument("--web", action="store_true", help="allow bounded web search/fetch; non-agentic mode is promoted to agentic")
    ask.add_argument(
        "--recursive-planning",
        action="store_true",
        help="wrap the request in Cairn plan propose/execute/revise (uses runtime planning flags)",
    )
    ask.add_argument("--json", action="store_true")

    chat = _add_cmd(subcommands, "chat")
    chat.add_argument("--node-id", default=None)
    chat.add_argument("--session-id", default="default")
    chat.add_argument("--adapter", default=None)
    chat.add_argument("--model", default=None)
    chat.add_argument("--retrieval-mode", choices=["direct", "agentic", "deep"], default=None)
    chat.add_argument("--web", action="store_true", help="allow bounded web search/fetch")

    history = _add_cmd(subcommands, "history")
    history.add_argument("--session-id", default=None)
    history.add_argument("--query", default=None)
    history.add_argument("--adapter", default=None)
    history.add_argument("--model", default=None)
    history.add_argument("--limit", type=int, default=10)

    plan_executions = _add_cmd(subcommands, "plan-executions")
    plan_exec_sub = plan_executions.add_subparsers(dest="plan_exec_command", required=True)
    plan_exec_list = plan_exec_sub.add_parser("list")
    plan_exec_list.add_argument("--session-id", default="default")
    plan_exec_list.add_argument("--status", default=None, choices=["running", "completed", "blocked"])
    plan_exec_list.add_argument("--limit", type=int, default=20)
    plan_exec_show = plan_exec_sub.add_parser("show")
    plan_exec_show.add_argument("plan_id")
    plan_exec_show.add_argument("--revision", type=int, required=True)
    plan_exec_show.add_argument("--session-id", default="default")

    process_p = _add_cmd(subcommands, "process", help="Human-defined process templates + instances.")
    process_sub = process_p.add_subparsers(dest="process_command", required=True)
    process_sub.add_parser("seed-presets")
    p_templates = process_sub.add_parser("templates")
    p_templates.add_argument("--include-presets", action="store_true", default=True)
    p_new = process_sub.add_parser("new-template")
    p_new.add_argument("name")
    p_new.add_argument("--body", required=True, help="Process text, or '-' to read stdin.")
    p_new.add_argument("--description", default="")
    p_new.add_argument("--category", default=None)
    p_new.add_argument("--risk", default=None, choices=["low", "medium", "high"])
    p_new.add_argument("--scope", default=None)
    p_start = process_sub.add_parser("start")
    p_start.add_argument("template_id")
    p_start.add_argument("--task", required=True)
    p_start.add_argument("--session-id", default=None)
    p_start.add_argument("--version", type=int, default=None)
    p_inst = process_sub.add_parser("instances")
    p_inst.add_argument("--session-id", default=None)
    p_inst.add_argument("--status", default=None)
    p_inst.add_argument("--limit", type=int, default=20)
    p_gate = process_sub.add_parser("gate")
    p_gate.add_argument("instance_id")
    p_gate.add_argument("--step-id", required=True)
    p_gate.add_argument("--approve", dest="approved", action="store_true")
    p_gate.add_argument("--reject", dest="approved", action="store_false")
    p_gate.set_defaults(approved=True)
    p_gate.add_argument("--note", default=None)
    p_override = process_sub.add_parser("override")
    p_override.add_argument("instance_id")
    p_override.add_argument("--justification", required=True)
    p_complete = process_sub.add_parser("complete")
    p_complete.add_argument("instance_id")
    p_complete.add_argument("--outcome", default="completed")
    p_retro = process_sub.add_parser("retrospective")
    p_retro.add_argument("instance_id")
    p_metrics = process_sub.add_parser("metrics")
    p_metrics.add_argument("--template-id", default=None)
    p_history = process_sub.add_parser("history")
    p_history.add_argument("--task", required=True)
    p_history.add_argument("--limit", type=int, default=10)
    p_review = process_sub.add_parser("review", help="Tirzah reviews a draft process body.")
    p_review.add_argument("--body", required=True, help="Process text, or '-' for stdin.")
    p_review.add_argument("--intended-use", default="")
    p_review.add_argument("--no-model", action="store_true", help="Structural review only.")
    p_trial = process_sub.add_parser("trial", help="Dry-run a process against a sample task.")
    p_trial.add_argument("--body", required=True, help="Process text, or '-' for stdin.")
    p_trial.add_argument("--sample-task", required=True)
    p_suggest = process_sub.add_parser("suggest", help="Suggest a process for a task.")
    p_suggest.add_argument("--task", required=True)
    p_suggest.add_argument("--risk", default=None, choices=["low", "medium", "high"])
    p_suggest.add_argument("--scope", default=None)
    p_suggest.add_argument("--use-model", action="store_true")
    p_evolve = process_sub.add_parser("evolve", help="Propose (or apply) an evolved template from usage.")
    p_evolve.add_argument("template_id")
    p_evolve.add_argument("--use-model", action="store_true", help="LLM rewrite for the proposal.")
    p_evolve.add_argument("--apply", action="store_true", help="Apply the proposed body as a new version.")
    p_outcomes = process_sub.add_parser("outcomes", help="Author/inspect a template's outcomes loop.")
    outcomes_sub = p_outcomes.add_subparsers(dest="outcomes_command", required=True)
    o_show = outcomes_sub.add_parser("show", help="Show a template's outcomes + loop config.")
    o_show.add_argument("template_id")
    o_set = outcomes_sub.add_parser("set", help="Set outcomes + loop on a template (new version).")
    o_set.add_argument("template_id")
    o_set.add_argument("--from", dest="from_file", required=True,
                       help="JSON file with {outcomes, outcomes_loop}, or '-' for stdin.")
    o_validate = outcomes_sub.add_parser("validate", help="Validate work against an instance's outcomes.")
    o_validate.add_argument("instance_id")
    o_validate.add_argument("--answer", default=None, help="Work text, or '-' for stdin.")

    queue_recent = _add_cmd(subcommands, "queue-recent")
    queue_recent.add_argument("--limit", type=int, default=10)
    queue_recent.add_argument("--status", default=None)
    queue_recent.add_argument("--query", default=None)
    queue_recent.add_argument("--reason", default=None)

    show_tree = _add_cmd(subcommands, "show-tree")
    show_tree.add_argument("document_id")
    show_tree.add_argument(
        "--compare",
        action="store_true",
        help="Diff the active tree against a fresh parse of the archived source.",
    )

    ingest_one = _add_cmd(subcommands, "ingest-one")
    ingest_one.add_argument("path")
    ingest_one.add_argument(
        "--label",
        action="append",
        default=[],
        help="Additional node label to apply to every ingested node. May be repeated.",
    )
    ingest_one.add_argument(
        "--ingestion-epoch",
        default=None,
        help="Optional epoch identifier to stamp on the document, tree, and nodes.",
    )

    ingest_folder = _add_cmd(subcommands, "ingest-folder")
    ingest_folder.add_argument("path")
    ingest_folder.add_argument(
        "--label",
        action="append",
        default=[],
        help="Additional node label to apply to every ingested node. May be repeated.",
    )
    ingest_folder.add_argument(
        "--ingestion-epoch",
        default=None,
        help="Optional epoch identifier to stamp on every ingested document, tree, and node.",
    )
    ingest_folder.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to ingest from the folder.",
    )
    ingest_folder.add_argument(
        "--include-results",
        action="store_true",
        help="Include every per-file result in the JSON output. By default only a summary is printed.",
    )

    rebuild_doc = _add_cmd(subcommands, "rebuild-document")
    rebuild_doc.add_argument("document_id")
    rebuild_doc.add_argument(
        "--source",
        default=None,
        help="Optional source path override. Defaults to the document archive path, then original source path.",
    )
    rebuild_doc.add_argument(
        "--force-replace",
        action="store_true",
        help="Deprecated compatibility flag. Rebuilds are versioned and non-destructive.",
    )
    rebuild_doc.add_argument(
        "--ingestion-epoch",
        default=None,
        help="Optional epoch identifier to stamp on replacement trees/nodes.",
    )
    rebuild_doc.add_argument(
        "--diff-only",
        action="store_true",
        help="Update only changed sections/chunks in the active tree; keep stable node ids.",
    )
    rebuild_doc.add_argument(
        "--compare",
        action="store_true",
        help="Print the structural diff without applying a rebuild.",
    )

    rebuild_by_label = _add_cmd(subcommands, "rebuild-by-label")
    rebuild_by_label.add_argument("--label", required=True)
    rebuild_by_label.add_argument("--limit", type=int, default=None)
    rebuild_by_label.add_argument(
        "--force-replace",
        action="store_true",
        help="Deprecated compatibility flag. Rebuilds are versioned and non-destructive.",
    )
    rebuild_by_label.add_argument(
        "--diff-only",
        action="store_true",
        help="Update only changed sections/chunks in each active tree; keep stable node ids.",
    )
    rebuild_by_label.add_argument(
        "--compare",
        action="store_true",
        help="Print structural diffs without applying rebuilds.",
    )
    rebuild_by_label.add_argument(
        "--ingestion-epoch",
        default=None,
        help="Optional epoch identifier to stamp on each replacement tree/node set.",
    )

    args = parser.parse_args()

    if args.command == "init":
        result = write_initial_config(
            Path(args.config),
            docker=args.docker,
            force=args.force,
            non_interactive=args.non_interactive,
            runtime_choice=args.runtime,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "serve":
        serve_app(host=args.host, port=args.port, reload=args.reload)
        return

    config = load_config(args.config)

    if args.command == "config-status":
        from tirzah.capabilities import render_runtime_text, resolved_runtime

        snapshot = resolved_runtime(config.runtime)
        snapshot["config_source"] = {
            "config_path": os.environ.get("TIRZAH_CONFIG") or args.config,
            "mongo_uri": config.mongo.uri,
            "mongo_db": config.mongo.database,
        }
        if args.format == "json":
            print(json.dumps(snapshot, indent=2))
            return
        print(f"config: {snapshot['config_source']['config_path']}  "
              f"(mongo {snapshot['config_source']['mongo_db']})")
        print(render_runtime_text(snapshot))
        return

    db = get_database(config.mongo)

    if args.command == "db-ping":
        ensure_indexes(db)
        print(json.dumps({"ok": True, "database": config.mongo.database}))
        return

    if args.command == "backfill-turn-embeddings":
        from tirzah.sessions.interaction import backfill_turn_embeddings

        embedded = backfill_turn_embeddings(db, config, config.runtime, limit=args.limit)
        print(json.dumps({"ok": True, "embedded": embedded}))
        return

    if args.command == "backfill-chunks":
        from tirzah.sessions.interaction import backfill_chunks

        chunked = backfill_chunks(db, config, config.runtime, limit=args.limit)
        print(json.dumps({"ok": True, "chunked": chunked}))
        return

    if args.command == "backfill-source-metadata":
        ensure_indexes(db)
        from tirzah.migrations import backfill_source_metadata

        print(json.dumps({"ok": True, **backfill_source_metadata(db, archive_dir=config.paths.archive)}, indent=2))
        return

    if args.command == "migrate":
        from tirzah import migrations

        ensure_indexes(db, force=True)
        report = migrations.status(db) if args.status else migrations.migrate(db, dry_run=args.dry_run)
        print(json.dumps(report, indent=2, default=str))
        return

    if args.command == "backfill-schema-metadata":
        ensure_indexes(db)
        print(json.dumps({"ok": True, **backfill_schema_metadata(db)}, indent=2))
        return

    if args.command == "graph-status":
        ensure_indexes(db)
        status = {"ok": True, **graph_edge_status(db, limit=args.limit)}
        if args.format == "text":
            print(render_graph_status_text(status))
            return
        print(json.dumps(status, indent=2))
        return

    if args.command == "backfill-structural-graph-edges":
        ensure_indexes(db)
        print(
            json.dumps(
                {"ok": True, **backfill_structural_graph_edges(db, limit=args.limit)},
                indent=2,
            )
        )
        return

    if args.command in {"backfill-embeddings", "backfill-profiles"}:
        ensure_indexes(db)
        print(
            json.dumps(
                backfill_node_embeddings(
                    db,
                    embedding_adapter(config.runtime),
                    limit=args.limit,
                    label=args.label,
                    document_id=args.document_id,
                    force=args.force,
                ),
                indent=2,
            )
        )
        return

    if args.command in {"embedding-backfill-jobs", "profile-backfill-jobs"}:
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "jobs": list_embedding_backfill_jobs(
                        db,
                        status=args.status,
                        limit=args.limit,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command in {"queue-embedding-backfill", "queue-profile-backfill"}:
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "job": create_embedding_backfill_job(
                        db,
                        batch_limit=args.limit,
                        label=args.label,
                        document_id=args.document_id,
                        force=args.force,
                        created_by=args.created_by,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command in {"process-embedding-backfill", "process-profile-backfill"}:
        ensure_indexes(db)
        print(
            json.dumps(
                process_embedding_backfill_batches(
                    db,
                    embedding_adapter(config.runtime),
                    max_batches=args.max_batches,
                ),
                indent=2,
            )
        )
        return

    if args.command == "enqueue-inbox":
        ensure_indexes(db)
        jobs = []
        for path in discover_sources(config.paths.ingest):
            checksum = sha256_file(path)
            job = enqueue_source(db, path, checksum)
            dead_letter_path = None
            if job["status"] == "rejected" and path.exists():
                dead_letter_path = move_request_file(path, config.paths.dead_letter / "duplicate", checksum)
                db.queue.update_one(
                    {"_id": job["_id"]},
                    {"$set": {"details.dead_letter_path": str(dead_letter_path)}},
                )
            jobs.append(
                {
                    "job_id": str(job["_id"]),
                    "path": job["path"],
                    "checksum_sha256": job["checksum_sha256"],
                    "status": job["status"],
                    "reason": job.get("reason"),
                    "dead_letter_path": str(dead_letter_path) if dead_letter_path else None,
                    "existing_document_id": str(job["existing_document_id"])
                    if job.get("existing_document_id")
                    else None,
                }
            )
        print(json.dumps({"ok": True, "enqueued": jobs}, indent=2))
        return

    if args.command == "process-next":
        ensure_indexes(db)
        print(json.dumps(process_next(db, config), indent=2))
        return

    if args.command == "queue-status":
        ensure_indexes(db)
        print(json.dumps({"ok": True, **serialize_queue_summary(queue_summary(db))}, indent=2))
        return

    if args.command == "labels":
        ensure_indexes(db)
        print(json.dumps({"ok": True, "labels": label_definitions(db)}, indent=2))
        return

    if args.command == "memory-health":
        ensure_indexes(db)
        report = memory_health_payload(db)
        if getattr(config, "runtime", None) is not None:
            from tirzah.capabilities import resolved_runtime

            report["runtime"] = resolved_runtime(config.runtime)  # adapter capability report
        if args.format == "json":
            print(json.dumps(report, indent=2))
            return
        print(render_memory_health_text(report))
        return

    if args.command == "session-continuity":
        ensure_indexes(db)
        payload = {"ok": True, **session_continuity(db, session_id=args.session_id, limit=args.limit)}
        if args.format == "json":
            print(json.dumps(payload, indent=2))
            return
        print(render_session_continuity_text(payload))
        return
    if args.command == "restart-render":
        ensure_indexes(db)
        payload = session_continuity(db, session_id=args.session_id, limit=args.limit)
        markdown = render_restart_markdown(payload)
        if args.output:
            Path(args.output).write_text(markdown, encoding="utf-8")
            print(f"Wrote restart state for session '{args.session_id}' to {args.output}")
            return
        print(markdown, end="")
        return

    if args.command == "agent-identities":
        ensure_indexes(db)
        print(
            json.dumps(
                {"ok": True, "identities": list_agent_identities(db, limit=args.limit)},
                indent=2,
            )
        )
        return

    if args.command == "agent-identity":
        ensure_indexes(db)
        identity = get_agent_identity(db, args.identity_id)
        print(json.dumps({"ok": identity is not None, "identity": identity}, indent=2))
        return

    if args.command == "trust-weighting-profiles":
        ensure_indexes(db)
        print(
            json.dumps(
                {"ok": True, "profiles": list_trust_weighting_profiles(db, limit=args.limit)},
                indent=2,
            )
        )
        return

    if args.command == "trust-weighting-profile":
        ensure_indexes(db)
        profile = get_trust_weighting_profile(db, args.weighting_profile_id)
        print(json.dumps({"ok": profile is not None, "profile": profile}, indent=2))
        return

    if args.command == "trust-diagnostic":
        ensure_indexes(db)
        diagnostic = trust_temporal_diagnostic_for_node(
            db,
            args.node_id,
            weighting_profile_id=args.profile_id,
        )
        print(json.dumps({"ok": diagnostic is not None, "result": diagnostic}, indent=2))
        return

    if args.command == "governance-policies":
        ensure_indexes(db)
        print(
            json.dumps(
                {"ok": True, "policies": list_governance_policies(db, limit=args.limit)},
                indent=2,
            )
        )
        return

    if args.command == "governance-policy":
        ensure_indexes(db)
        policy = get_governance_policy(db, args.policy_id)
        print(json.dumps({"ok": policy is not None, "policy": policy}, indent=2))
        return

    if args.command == "process-objects":
        ensure_indexes(db)
        print(
            json.dumps(
                {"ok": True, "processes": list_process_objects(db, limit=args.limit)},
                indent=2,
            )
        )
        return

    if args.command == "process-object":
        ensure_indexes(db)
        process = get_process_object(db, args.process_id)
        print(json.dumps({"ok": process is not None, "process": process}, indent=2))
        return

    if args.command == "process-runs":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "runs": list_process_runs(
                        db,
                        session_id=args.session_id,
                        status=args.status,
                        limit=args.limit,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "process-run":
        ensure_indexes(db)
        run = get_process_run(db, args.run_id)
        print(json.dumps({"ok": run is not None, "run": run}, indent=2))
        return

    if args.command == "start-process-run":
        ensure_indexes(db)
        run = create_process_run(
            db,
            process_id=args.process_id,
            session_id=args.session_id,
            identity_id=args.identity_id,
            current_step_id=args.current_step_id,
            status=args.status,
        )
        print(json.dumps({"ok": True, "run": run}, indent=2))
        return

    if args.command == "update-process-run":
        ensure_indexes(db)
        exception = None
        if args.exception_reason or args.exception_proposal:
            exception = {
                "reason": args.exception_reason,
                "proposal": args.exception_proposal,
                "reviewer": args.exception_reviewer,
                "note": args.exception_note,
            }
        run = update_process_run(
            db,
            args.run_id,
            status=args.status,
            current_step_id=args.current_step_id,
            completed_step_id=args.completed_step_id,
            exchange_id=args.exchange_id,
            exception=exception,
        )
        print(json.dumps({"ok": run is not None, "run": run}, indent=2))
        return

    if args.command == "sessions":
        ensure_indexes(db)
        print(json.dumps({"ok": True, "sessions": list_sessions(db)}, indent=2))
        return

    if args.command == "embedding-smoke":
        if args.adapter:
            config.runtime.embedding_adapter = args.adapter
        if args.allow_http_diagnostic:
            config.runtime.allow_http_ingestion_adapters = True
        if args.model:
            config.runtime.embedding_model = args.model
        print(json.dumps(embedding_smoke_payload(config, args.text), indent=2))
        return

    if args.command == "active-documents":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "session_id": args.session_id,
                    "documents": list_active_documents(
                        db,
                        session_id=args.session_id,
                        limit=args.limit,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "output-ingestion":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "jobs": list_output_ingestion_jobs(
                        db,
                        limit=args.limit,
                        status=args.status,
                        session_id=args.session_id,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "process-output-ingestion":
        ensure_indexes(db)
        print(
            json.dumps(
                process_next_output_ingestion(
                    db,
                    session_id=args.session_id,
                    job_id=args.job_id,
                ),
                indent=2,
            )
        )
        return

    if args.command == "review-generated-output":
        ensure_indexes(db)
        try:
            nodes = list_generated_output_nodes(
                db,
                limit=args.limit,
                endorsement_label=args.endorsement,
            )
        except ValueError as error:
            print(
                json.dumps(
                    {"ok": False, "reason": "invalid_endorsement_label", "error": str(error)},
                    indent=2,
                )
            )
            return
        print(
            json.dumps(
                {
                    "ok": True,
                    "nodes": nodes,
                },
                indent=2,
            )
        )
        return

    if args.command == "list-proposed-ingestions":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "trees": list_proposed_ingestion_trees(db, limit=args.limit),
                },
                indent=2,
            )
        )
        return

    if args.command == "promote-ingestion":
        ensure_indexes(db)
        print(
            json.dumps(
                promote_ingestion_tree(
                    db,
                    args.identifier,
                    reviewer=args.reviewer,
                    note=args.note,
                    endorsement_label=args.endorsement,
                ),
                indent=2,
            )
        )
        return

    if args.command == "reject-ingestion":
        ensure_indexes(db)
        print(
            json.dumps(
                reject_ingestion_tree(
                    db,
                    args.identifier,
                    reviewer=args.reviewer,
                    note=args.note,
                ),
                indent=2,
            )
        )
        return

    if args.command == "endorse-node":
        ensure_indexes(db)
        print(
            json.dumps(
                update_node_endorsement(
                    db,
                    node_id=args.node_id,
                    endorsement_label=args.endorsement,
                    reviewer=args.reviewer,
                    note=args.note,
                ),
                indent=2,
            )
        )
        return

    if args.command == "create-session":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "session": create_session(
                        db,
                        title=args.title,
                        session_id=args.session_id,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "list-docs":
        ensure_indexes(db)
        documents = list_documents(db, args.limit)
        if args.format == "text":
            print(render_documents_text(documents))
            return
        print(json.dumps({"ok": True, "documents": documents}, indent=2))
        return

    if args.command == "show-doc":
        ensure_indexes(db)
        document = get_document(db, args.document_id)
        print(json.dumps({"ok": document is not None, "document": document}, indent=2))
        return

    if args.command == "set-origin-date":
        ensure_indexes(db)
        print(
            json.dumps(
                update_document_origin_date(
                    db,
                    args.document_id,
                    args.date,
                    reviewer=args.reviewer,
                    note=args.note,
                ),
                indent=2,
            )
        )
        return

    if args.command == "set-node-summary":
        ensure_indexes(db)
        print(
            json.dumps(
                update_node_summary(
                    db,
                    args.node_id,
                    args.summary,
                    reviewer=args.reviewer,
                    note=args.note,
                ),
                indent=2,
            )
        )
        return

    if args.command == "search-nodes":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "nodes": search_nodes(
                        db,
                        query=args.query,
                        label=args.label,
                        endorsement_label=args.endorsement,
                        document_id=args.document_id,
                        created_after=parse_iso_datetime(args.created_after),
                        created_before=parse_iso_datetime(args.created_before),
                        origin_after=parse_iso_date(args.origin_after),
                        origin_before=parse_iso_date(args.origin_before),
                        limit=args.limit,
                        query_embedding=build_query_embedding(config.runtime, args.query),
                        vector_search_index=config.runtime.vector_search_index or None,
                        vector_scan_limit=config.runtime.hybrid_vector_scan_limit,
                        trust_ranking_enabled=args.trust_ranking or config.runtime.trust_ranking_enabled,
                        trust_weighting_profile=args.trust_profile or config.runtime.trust_weighting_profile,
                        trust_ranking_weight=config.runtime.trust_ranking_weight,
                        trust_ranking_max_boost=config.runtime.trust_ranking_max_boost,
                        trust_ranking_hybrid_weight=config.runtime.trust_ranking_hybrid_weight,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "node-context":
        ensure_indexes(db)
        context_result = node_context(db, args.node_id, child_limit=args.child_limit)
        print(json.dumps({"ok": context_result is not None, "context": context_result}, indent=2))
        return

    if args.command == "graph-edges":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "edges": graph_edges_for_node(
                        db,
                        args.node_id,
                        direction=args.direction,
                        relation_type=args.relation_type,
                        limit=args.limit,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "expand-proximity":
        ensure_indexes(db)
        nodes = expand_proximity(
            db,
            args.node_id,
            direction=args.direction,
            relation_type=args.relation_type,
            limit=args.limit,
        )
        if args.format == "text":
            print(render_proximity_text(nodes))
            return
        print(
            json.dumps(
                {
                    "ok": True,
                    "nodes": nodes,
                },
                indent=2,
            )
        )
        return

    if args.command == "expand-graph-paths":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "paths": expand_graph_paths(
                        db,
                        args.node_id,
                        direction=args.direction,
                        relation_type=args.relation_type,
                        max_depth=args.max_depth,
                        branch_limit=args.branch_limit,
                        limit=args.limit,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "semantic-candidates":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "nodes": semantic_candidate_nodes(
                        db,
                        args.node_id,
                        include_same_document=args.include_same_document,
                        limit=args.limit,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command in {"vector-semantic-candidates", "profile-semantic-candidates"}:
        ensure_indexes(db)
        report = embedding_candidate_report(
            db,
            args.node_id,
            include_same_document=args.include_same_document,
            min_similarity=args.min_similarity,
            limit=args.limit,
            candidate_scan_limit=args.candidate_scan_limit,
        )
        if args.format == "text":
            print(render_vector_semantic_candidates_text(report))
        else:
            print(json.dumps(report, indent=2))
        return

    if args.command == "enqueue-semantic-candidates":
        ensure_indexes(db)
        print(
            json.dumps(
                enqueue_semantic_edge_candidates(
                    db,
                    node_id=args.node_id,
                    include_same_document=args.include_same_document,
                    relation_type=args.relation_type,
                    created_by=args.created_by,
                    limit=args.limit,
                ),
                indent=2,
            )
        )
        return

    if args.command in {
        "enqueue-vector-semantic-candidates",
        "enqueue-profile-semantic-candidates",
    }:
        ensure_indexes(db)
        print(
            json.dumps(
                enqueue_vector_semantic_edge_candidates(
                    db,
                    node_id=args.node_id,
                    include_same_document=args.include_same_document,
                    relation_type=args.relation_type,
                    created_by=args.created_by,
                    min_similarity=args.min_similarity,
                    limit=args.limit,
                    candidate_scan_limit=args.candidate_scan_limit,
                ),
                indent=2,
            )
        )
        return

    if args.command in {
        "enqueue-vector-semantic-batch",
        "enqueue-profile-semantic-batch",
    }:
        ensure_indexes(db)
        result = enqueue_vector_semantic_edge_candidate_batch(
            db,
            label=args.label,
            document_id=args.document_id,
            focus_limit=args.focus_limit,
            candidates_per_node=args.candidates_per_node,
            include_same_document=args.include_same_document,
            relation_type=args.relation_type,
            created_by=args.created_by,
            min_similarity=args.min_similarity,
            candidate_scan_limit=args.candidate_scan_limit,
            exclude_node_keys=args.exclude_node_key,
            dry_run=args.dry_run,
        )
        if args.format == "text":
            print(render_vector_semantic_batch_text(result))
        else:
            print(json.dumps(result, indent=2))
        return

    if args.command == "semantic-edge-candidates":
        ensure_indexes(db)
        candidates = list_semantic_edge_candidates(
            db,
            status=args.status,
            limit=args.limit,
        )
        if args.format == "text":
            print(render_semantic_edge_candidates_text(candidates))
        else:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "candidates": candidates,
                    },
                    indent=2,
                )
            )
        return

    if args.command == "review-semantic-edge-candidate":
        ensure_indexes(db)
        print(
            json.dumps(
                review_semantic_edge_candidate(
                    db,
                    candidate_id=args.candidate_id,
                    action=args.action,
                    reviewer=args.reviewer,
                    note=args.note,
                    weight=args.weight,
                    confidence=args.confidence,
                ),
                indent=2,
            )
        )
        return

    if args.command == "create-semantic-edge":
        ensure_indexes(db)
        print(
            json.dumps(
                create_reviewed_semantic_edge(
                    db,
                    source_node_id=args.source_node_id,
                    target_node_id=args.target_node_id,
                    relation_type=args.relation_type,
                    weight=args.weight,
                    confidence=args.confidence,
                    reviewer=args.reviewer,
                    note=args.note,
                ),
                indent=2,
            )
        )
        return

    if args.command == "compile-context":
        ensure_indexes(db)
        context_result = compile_context(
            db,
            args.node_id,
            ancestor_depth=args.ancestor_depth,
            sibling_window=args.sibling_window,
            child_depth=args.child_depth,
            child_limit=args.child_limit,
        )
        print(json.dumps({"ok": context_result is not None, "context": context_result}, indent=2))
        return

    if args.command == "render-context":
        ensure_indexes(db)
        context_result = compile_context(
            db,
            args.node_id,
            ancestor_depth=args.ancestor_depth,
            sibling_window=args.sibling_window,
            child_depth=args.child_depth,
            child_limit=args.child_limit,
        )
        if not context_result:
            print(json.dumps({"ok": False, "context": None}, indent=2))
            return
        rendered = render_context_document(
            context_result,
            char_budget=args.char_budget or config.retrieval.context_char_budget,
        )
        if args.json:
            print(json.dumps({"ok": True, "rendered": rendered}, indent=2))
        else:
            print(rendered["text"])
        return

    if args.command == "build-prompt":
        ensure_indexes(db)
        context_result = compile_context(
            db,
            args.node_id,
            ancestor_depth=args.ancestor_depth,
            sibling_window=args.sibling_window,
            child_depth=args.child_depth,
            child_limit=args.child_limit,
        )
        if not context_result:
            print(json.dumps({"ok": False, "prompt": None}, indent=2))
            return
        from tirzah.semantic import make_resolver

        from tirzah.retrieval.budget import envelope_budget_args

        budget_args = envelope_budget_args(config)
        if args.token_budget:
            budget_args["token_budget"] = args.token_budget
        if args.reserved_response_tokens:
            budget_args["reserved_response_tokens"] = args.reserved_response_tokens
        envelope = build_prompt_envelope(
            context_result,
            query=args.query,
            system_instruction=args.system_instruction,
            resolver=make_resolver(config.runtime),
            semantic_strict=config.runtime.mahalath_strict,
            **budget_args,
        )
        if envelope.get("semantic_summary"):
            print(f"# {envelope['semantic_summary']}", file=sys.stderr)
        if args.text:
            print(envelope["prompt_text"])
        else:
            print(json.dumps({"ok": True, "prompt": envelope}, indent=2))
        return

    if args.command == "process":
        ensure_indexes(db)
        from tirzah.process import enforcement as proc_enf
        from tirzah.process import instances as proc_inst
        from tirzah.process import retrospective as proc_retro
        from tirzah.process import templates as proc_tmpl

        def _emit(payload: dict) -> None:
            print(json.dumps(payload, indent=2, default=str))

        cmd = args.process_command
        if cmd == "seed-presets":
            _emit({"ok": True, "created": [p["name"] for p in proc_tmpl.seed_presets(db)]})
            return
        if cmd == "templates":
            _emit({"ok": True, "templates": proc_tmpl.list_templates(db, include_presets=args.include_presets)})
            return
        if cmd == "new-template":
            body = sys.stdin.read() if args.body == "-" else args.body
            try:
                template = proc_tmpl.create_template(
                    db, name=args.name, body=body, description=args.description,
                    category=args.category, risk_level=args.risk, scope=args.scope,
                )
            except ValueError as error:
                _emit({"ok": False, "error": str(error)})
                return
            _emit({"ok": True, "template": template})
            return
        if cmd == "start":
            try:
                instance = proc_inst.start_instance(
                    db, template_id=args.template_id, task=args.task,
                    session_id=args.session_id, version=args.version,
                )
            except ValueError as error:
                _emit({"ok": False, "error": str(error)})
                return
            _emit({"ok": True, "instance": instance})
            return
        if cmd == "instances":
            _emit({"ok": True, "instances": proc_inst.list_instances(
                db, session_id=args.session_id, status=args.status, limit=args.limit)})
            return
        if cmd == "gate":
            _emit({"ok": True, "instance": proc_enf.resolve_gate(
                db, args.instance_id, step_id=args.step_id, approved=args.approved, note=args.note)})
            return
        if cmd == "override":
            try:
                instance = proc_enf.record_override(db, args.instance_id, justification=args.justification)
            except ValueError as error:
                _emit({"ok": False, "error": str(error)})
                return
            _emit({"ok": True, "instance": instance})
            return
        if cmd == "complete":
            _emit({"ok": True, "instance": proc_inst.complete_instance(db, args.instance_id, outcome=args.outcome)})
            return
        if cmd == "retrospective":
            retro = proc_retro.build_retrospective(db, args.instance_id)
            _emit({"ok": retro is not None, "retrospective": retro})
            return
        if cmd == "metrics":
            _emit({"ok": True, "metrics": proc_retro.usage_metrics(db, template_id=args.template_id)})
            return
        if cmd == "history":
            _emit({"ok": True, "history": proc_retro.similar_task_history(db, task=args.task, limit=args.limit)})
            return
        if cmd == "review":
            from tirzah.process import refinement as proc_ref

            body = sys.stdin.read() if args.body == "-" else args.body
            ask = None if args.no_model else proc_ref.default_ask(config)
            _emit({"ok": True, "review": proc_ref.review_process(
                body, ask=ask, intended_use=args.intended_use)})
            return
        if cmd == "trial":
            from tirzah.process import refinement as proc_ref

            body = sys.stdin.read() if args.body == "-" else args.body
            _emit({"ok": True, "trial": proc_ref.trial_process(
                db, config, body=body, sample_task=args.sample_task)})
            return
        if cmd == "suggest":
            from tirzah.process import refinement as proc_ref
            from tirzah.process import selection as proc_sel

            ask = proc_ref.default_ask(config) if args.use_model else None
            _emit({"ok": True, "suggestion": proc_sel.suggest_process(
                db, task=args.task, risk_level=args.risk, scope=args.scope, ask=ask)})
            return
        if cmd == "evolve":
            from tirzah.process import evolution as proc_evo
            from tirzah.process import refinement as proc_ref

            ask = proc_ref.default_ask(config) if args.use_model else None
            proposal = proc_evo.propose_evolution(db, args.template_id, ask=ask)
            if not args.apply or not proposal.get("ready"):
                _emit({"ok": True, "proposal": proposal})
                return
            applied = proc_evo.apply_evolution(
                db, args.template_id, body=proposal["proposed_body"],
                rationale=proposal["rationale"], based_on_instances=proposal["instance_count"],
            )
            _emit({"ok": True, "applied": applied})
            return
        if cmd == "outcomes":
            ocmd = args.outcomes_command
            if ocmd == "show":
                template = proc_tmpl.get_template(db, args.template_id)
                if template is None:
                    _emit({"ok": False, "error": f"unknown template: {args.template_id}"})
                    return
                _emit({"ok": True, "outcomes": template.get("outcomes", []),
                       "outcomes_loop": template.get("outcomes_loop")})
                return
            if ocmd == "set":
                import json as _json

                raw = sys.stdin.read() if args.from_file == "-" else Path(args.from_file).read_text()
                spec = _json.loads(raw)
                try:
                    template = proc_tmpl.revise_template(
                        db, args.template_id,
                        outcomes=spec.get("outcomes", []),
                        outcomes_loop=spec.get("outcomes_loop"),
                    )
                except ValueError as error:
                    _emit({"ok": False, "error": str(error)})
                    return
                _emit({"ok": True, "template": template})
                return
            if ocmd == "validate":
                from tirzah.process import outcomes as proc_oc

                instance = proc_inst.get_instance(db, args.instance_id)
                if instance is None:
                    _emit({"ok": False, "error": f"unknown instance: {args.instance_id}"})
                    return
                answer = sys.stdin.read() if args.answer == "-" else args.answer
                validation = proc_oc.validate_outcomes(instance, {"answer": answer or ""})
                _emit({"ok": True, "validation": validation})
                return
            return
        return

    if args.command == "plan-executions":
        ensure_indexes(db)
        from tirzah.planning.execution_store import (
            compact_execution_summary,
            get_plan_execution,
            list_plan_executions,
        )

        if args.plan_exec_command == "list":
            rows = list_plan_executions(db, args.session_id, status=args.status, limit=args.limit)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "session_id": args.session_id,
                        "executions": [compact_execution_summary(row) for row in rows],
                    },
                    indent=2,
                    default=str,
                )
            )
            return
        row = get_plan_execution(db, args.plan_id, args.revision, args.session_id)
        if not row:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "reason": "unknown_plan_execution",
                        "plan_id": args.plan_id,
                        "revision": args.revision,
                        "session_id": args.session_id,
                    },
                    indent=2,
                )
            )
            sys.exit(1)
        print(json.dumps({"ok": True, "execution": row}, indent=2, default=str))
        return

    if args.command == "ask":
        ensure_indexes(db)
        result = run_traced_interaction(
            db,
            config,
            query=args.query,
            session_id=args.session_id,
            executor=answer_query,
            planning_enabled=args.recursive_planning or None,
            source="tirzah-cli",
            focus_node_id=args.node_id,
            answer_adapter_name=args.adapter,
            ollama_model=args.model,
            retrieval_mode=args.retrieval_mode,
            web_research=args.web or None,
        )
        if args.json or not result.get("ok"):
            print(json.dumps(result, indent=2))
        else:
            print(result["answer"])
            if result.get("semantic_summary"):
                print(f"\n[{result['semantic_summary']}]")
            print(f"\nexchange_id: {result['exchange_id']}")
        if not result.get("ok"):
            sys.exit(1)
        return

    if args.command == "chat":
        ensure_indexes(db)
        print("Tirzah chat. Type /exit to quit.")
        while True:
            try:
                query = input("> ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query in {"/exit", "/quit"}:
                break
            result = run_traced_interaction(
                db,
                config,
                query=query,
                session_id=args.session_id,
                executor=answer_query,
                planning_enabled=None,
                source="tirzah-cli",
                focus_node_id=args.node_id,
                answer_adapter_name=args.adapter,
                ollama_model=args.model,
                retrieval_mode=args.retrieval_mode,
                web_research=args.web or None,
            )
            if not result.get("ok"):
                print(json.dumps(result, indent=2))
            else:
                print(result["answer"])
                print(f"[exchange_id: {result['exchange_id']}]\n")
        return

    if args.command == "history":
        ensure_indexes(db)
        print(
            json.dumps(
                {
                    "ok": True,
                    "exchanges": recent_exchanges(
                        db,
                        limit=args.limit,
                        session_id=args.session_id,
                        query_text=args.query,
                        adapter=args.adapter,
                        model=args.model,
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "show-tree":
        ensure_indexes(db)
        payload = {"ok": True, "nodes": document_tree(db, args.document_id)}
        if args.compare:
            compared = rebuild_document_from_existing_source(
                db,
                args.document_id,
                runtime_config=config.runtime,
                compare_only=True,
            )
            payload["compare"] = compared.get("diff")
            payload["compare_ok"] = compared.get("ok")
            if compared.get("reason"):
                payload["compare_reason"] = compared["reason"]
        print(json.dumps(payload, indent=2))
        return

    if args.command == "queue-recent":
        ensure_indexes(db)
        jobs = [
            serialize_queue_job(job)
            for job in recent_jobs(
                db,
                limit=args.limit,
                status=args.status,
                query_text=args.query,
                reason=args.reason,
            )
        ]
        print(json.dumps({"ok": True, "jobs": jobs}, indent=2))
        return

    if args.command == "process-inbox":
        ensure_indexes(db)
        enqueued = []
        for path in discover_sources(config.paths.ingest):
            checksum = sha256_file(path)
            job = enqueue_source(db, path, checksum)
            dead_letter_path = None
            if job["status"] == "rejected" and path.exists():
                dead_letter_path = move_request_file(path, config.paths.dead_letter / "duplicate", checksum)
                db.queue.update_one(
                    {"_id": job["_id"]},
                    {"$set": {"details.dead_letter_path": str(dead_letter_path)}},
                )
            enqueued.append(
                {
                    "job_id": str(job["_id"]),
                    "path": job["path"],
                    "status": job["status"],
                    "reason": job.get("reason"),
                    "dead_letter_path": str(dead_letter_path) if dead_letter_path else None,
                }
            )
        processed = []
        while True:
            result = process_next(db, config)
            if result["status"] == "idle":
                break
            processed.append(result)
        print(json.dumps({"ok": True, "enqueued": enqueued, "processed": processed}, indent=2))
        return

    if args.command == "ingest-one":
        ensure_indexes(db)
        print(
            json.dumps(
                ingest_source_path(
                    db,
                    config,
                    Path(args.path),
                    args.label,
                    ingestion_epoch=args.ingestion_epoch,
                ),
                indent=2,
            )
        )
        return

    if args.command == "ingest-folder":
        ensure_indexes(db)
        root = Path(args.path)
        results = []
        source_plan = chronological_folder_source_plan(root)
        if args.limit is not None:
            source_plan = source_plan[: args.limit]
        for item in source_plan:
            path = item["path"]
            if item.get("error"):
                results.append(
                    {
                        "ok": False,
                        "path": str(path),
                        "status": "rejected",
                        "reason": "source_unreadable",
                        "error": item["error"],
                        "message": item.get("message"),
                    }
                )
                continue
            results.append(
                ingest_source_path(
                    db,
                    config,
                    path,
                    args.label,
                    ingestion_epoch=args.ingestion_epoch,
                )
            )
        rejected = [result for result in results if not result.get("ok")]
        output = {
            "ok": True,
            "root": str(root),
            "ingestion_epoch": args.ingestion_epoch,
            "ordering": "origin_date_then_path",
            "file_count": len(source_plan),
            "inserted": sum(1 for result in results if result.get("ok")),
            "rejected": len(rejected),
            "rejection_reasons": rejection_reason_counts(rejected),
            "source_order": [
                {
                    "path": str(item["path"]),
                    "origin_date": item.get("origin_date"),
                    "origin_date_source": item.get("origin_date_source"),
                    "error": item.get("error"),
                }
                for item in source_plan[:20]
            ],
        }
        if args.include_results:
            output["results"] = results
        print(
            json.dumps(output, indent=2)
        )
        return

    if args.command == "rebuild-document":
        ensure_indexes(db)
        print(
            json.dumps(
                rebuild_document_from_existing_source(
                    db,
                    args.document_id,
                    args.source,
                    ingestion_epoch=args.ingestion_epoch,
                    runtime_config=config.runtime,
                    mode="diff" if args.diff_only else "full",
                    compare_only=args.compare,
                ),
                indent=2,
            )
        )
        return

    if args.command == "rebuild-by-label":
        ensure_indexes(db)
        document_ids = document_ids_for_label(db, args.label)
        if args.limit is not None:
            document_ids = document_ids[: args.limit]
        results = [
            rebuild_document_from_existing_source(
                db,
                document_id,
                ingestion_epoch=args.ingestion_epoch,
                runtime_config=config.runtime,
                mode="diff" if args.diff_only else "full",
                compare_only=args.compare,
            )
            for document_id in document_ids
        ]
        failures = [result for result in results if not result.get("ok")]
        print(
            json.dumps(
                {
                    "ok": not failures,
                    "label": args.label,
                    "document_count": len(document_ids),
                    "rebuilt": sum(1 for result in results if result.get("ok")),
                    "failed": len(failures),
                    "failures": failures,
                },
                indent=2,
            )
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
