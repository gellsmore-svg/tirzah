from __future__ import annotations

from pathlib import Path
from typing import Any

from tirzah.ingestion.formats import canonical_kind
from tirzah.models.ingestion import IngestionResult


def ingestion_activity_report(
    *,
    path: Path | str,
    status: str,
    checksum_sha256: str | None = None,
    result: IngestionResult | None = None,
    inserted: dict[str, Any] | None = None,
    job_id: str | None = None,
    process_run_id: str | None = None,
    reason: str | None = None,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = result.source if result else None
    nodes = result.nodes if result else []
    relation_count = sum(len(node.relations) for node in nodes)
    labels = sorted({label for node in nodes for label in node.labels})
    inserted = inserted or {}
    details = details or {}
    stored_node_count = inserted.get("node_count")
    if stored_node_count is None and inserted.get("node_ids") is not None:
        stored_node_count = len(inserted["node_ids"])
    # What the parser did to the source on the way into the tree (kind,
    # strategy, dropped/normalised content). The archived source is untouched.
    analysis = dict(getattr(result, "source_analysis", None) or {}) if result else {}
    if source and not analysis.get("canonical_kind"):
        analysis["canonical_kind"] = canonical_kind(source.kind)
    return {
        "kind": "ingestion_activity_report",
        "status": status,
        "source": {
            "path": str(path),
            "kind": source.kind if source else details.get("source_kind"),
            "checksum_sha256": checksum_sha256 or (source.checksum_sha256 if source else None),
            "archive_path": inserted.get("archive_path") or (source.archive_path if source else None),
            "processed_path": inserted.get("processed_path"),
        },
        "queue": {
            "job_id": job_id,
            "process_run_id": process_run_id,
        },
        "source_dates": {
            "origin_date": source.origin_date if source else None,
            "origin_date_source": source.origin_date_source if source else None,
            "candidate_count": len(source.date_candidates) if source else 0,
            "candidates": source.date_candidates if source else [],
        },
        "source_analysis": analysis,
        "semantic_processing": {
            "adapter": result.adapter if result else None,
            "title": result.title if result else None,
            "summary": result.summary if result else None,
            "node_count": len(nodes),
            "relation_hint_count": relation_count,
            "labels": labels,
            "sample_nodes": [
                {
                    "title": node.title,
                    "labels": node.labels,
                    "relation_hint_count": len(node.relations),
                }
                for node in nodes[:5]
            ],
        },
        "repository_actions": {
            "document_id": inserted.get("document_id"),
            "tree_id": inserted.get("tree_id"),
            "node_count": stored_node_count,
            "edge_count": inserted.get("edge_count"),
            "ingestion_epoch": inserted.get("ingestion_epoch") or (result.ingestion_epoch if result else None),
            "embedded_node_count": inserted.get("embedded_node_count"),
            "embedding_model": inserted.get("embedding_model"),
            "embedding_dimensions": inserted.get("embedding_dimensions"),
        },
        "outcome": {
            "reason": reason,
            "message": message,
            "details": details,
        },
    }


def ingestion_activity_log(report: dict[str, Any]) -> str:
    source = report.get("source") or {}
    dates = report.get("source_dates") or {}
    semantic = report.get("semantic_processing") or {}
    repo = report.get("repository_actions") or {}
    outcome = report.get("outcome") or {}
    queue = report.get("queue") or {}

    lines = [
        "Ingestion Activity Log",
        f"- Status: {report.get('status') or 'unknown'}.",
        f"- Source: {source.get('path') or 'unknown'}"
        f" ({source.get('kind') or 'type unknown'}).",
    ]
    if source.get("checksum_sha256"):
        lines.append(f"- Checksum: {source['checksum_sha256']}.")
    if queue.get("job_id"):
        lines.append(f"- Queue job: {queue['job_id']}.")
    if queue.get("process_run_id"):
        lines.append(f"- Process run: {queue['process_run_id']}.")

    if outcome.get("reason"):
        lines.append(f"- Outcome reason: {outcome['reason']}.")
    if outcome.get("message"):
        lines.append(f"- Operator note: {outcome['message']}")

    if dates.get("origin_date"):
        lines.append(
            "- Source dating: selected "
            f"{dates['origin_date']} from {human_date_source(dates.get('origin_date_source'))}; "
            f"{dates.get('candidate_count', 0)} candidate(s) inspected."
        )
    elif dates.get("candidate_count"):
        lines.append(f"- Source dating: {dates['candidate_count']} candidate(s) inspected; no date selected.")

    analysis = report.get("source_analysis") or {}
    if analysis.get("chunk_strategy"):
        lines.append(
            f"- Source analysis: parsed as {analysis.get('canonical_kind') or 'unknown'} "
            f"with the {analysis['chunk_strategy']} chunk strategy."
        )
    lines.extend(parser_loss_lines(analysis))

    if semantic.get("adapter"):
        lines.append(f"- Semantic processing: {semantic['adapter']} generated {semantic.get('node_count', 0)} node(s).")
    if semantic.get("relation_hint_count"):
        lines.append(f"- Relationship hints: {semantic['relation_hint_count']} proposed by ingestion.")
    if semantic.get("labels"):
        lines.append(f"- Labels applied: {', '.join(semantic['labels'])}.")

    if repo.get("document_id"):
        lines.append(
            "- Repository write: document "
            f"{repo['document_id']} with {repo.get('node_count', 0)} node(s)"
            f" in epoch {repo.get('ingestion_epoch') or 'unknown'}."
        )
    if repo.get("embedded_node_count"):
        lines.append(
            "- Text similarity profiles: "
            f"{repo['embedded_node_count']} node(s) profiled with "
            f"{repo.get('embedding_model') or 'unknown model'} "
            f"({repo.get('embedding_dimensions') or 0} dims)."
        )
    if source.get("archive_path"):
        lines.append(f"- Archive: {source['archive_path']}.")
    if source.get("processed_path"):
        lines.append(f"- Processed source moved to: {source['processed_path']}.")

    details = outcome.get("details") or {}
    if details.get("dead_letter_path"):
        lines.append(f"- Dead letter: {details['dead_letter_path']}.")
    if details.get("existing_document_id"):
        lines.append(f"- Existing document: {details['existing_document_id']}.")
    if details.get("error"):
        lines.append(f"- Error: {details['error']}.")

    return "\n".join(lines)


def parser_loss_lines(analysis: dict[str, Any]) -> list[str]:
    """Log lines for content the parser dropped or reshaped on the way into the
    parsed tree, so no transformation of the source goes unrecorded."""
    lines = []
    if analysis.get("parse_fallback"):
        lines.append(f"- Parser fallback: structured parse failed; stored {analysis['parse_fallback']}.")
    if analysis.get("dropped_chrome_elements"):
        lines.append(
            f"- Omitted from the parsed tree: {analysis['dropped_chrome_elements']} nav/footer element(s) "
            f"({analysis.get('dropped_chrome_chars', 0)} chars)."
        )
    if analysis.get("skipped_script_style_chars"):
        lines.append(
            f"- Omitted from the parsed tree: {analysis['skipped_script_style_chars']} chars of "
            "script/style content."
        )
    if analysis.get("preserved_pre_blocks"):
        lines.append(f"- Preformatted blocks kept verbatim: {analysis['preserved_pre_blocks']}.")
    if analysis.get("front_matter_stripped_chars"):
        lines.append(
            f"- Front matter stripped from the parsed tree: {analysis['front_matter_stripped_chars']} chars."
        )
    if analysis.get("csv_ragged_row_count"):
        lines.append(
            f"- CSV ragged rows: {analysis['csv_ragged_row_count']} "
            f"({analysis.get('csv_surplus_cell_count', 0)} surplus cell(s) kept under column_N keys, "
            f"{analysis.get('csv_short_row_count', 0)} short row(s))."
        )
    if (analysis.get("csv_section_count") or 0) > 1:
        lines.append(
            f"- CSV chunking: {analysis.get('csv_row_count', 0)} row(s) in "
            f"{analysis['csv_section_count']} section(s)."
        )
    if lines:
        lines.append("- The archived source file is unchanged.")
    return lines


def attach_ingestion_activity(result: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    result.update(ingestion_activity_fields(report))
    return result


def ingestion_activity_fields(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "activity_report": report,
        "activity_log": ingestion_activity_log(report),
    }


def human_date_source(source: str | None) -> str:
    labels = {
        "explicit_content": "an explicit date in the document",
        "filename": "the filename",
        "file_created": "the filesystem creation/change time",
        "file_modified": "the filesystem modification time",
    }
    return labels.get(source or "", source or "unknown evidence")
