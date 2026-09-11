from __future__ import annotations

import re
from typing import Any

from pymongo.database import Database

from tirzah.db.memory_store import MemoryStore, as_memory_store
from tirzah.retrieval.queries import (
    adjacent_node_for_edge,
    bounded_int,
    edge_summary,
    graph_edges_for_node,
    is_superseded_node,
    node_visible_to_identity,
    parse_object_id,
    serialize_node,
)


def graph_neighborhood(
    db: Database | MemoryStore,
    node_id: str,
    *,
    max_depth: int = 2,
    direction: str = "both",
    relation_type: str | None = None,
    limit: int = 20,
    branch_limit: int = 8,
    endorsement_label: str | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One- or two-hop neighborhood for CLI/web graph exploration.

    Reuses `graph_edges_for_node`. Skips superseded nodes. Optional identity
    and endorsement filters apply to neighbors, not the focus node.
    """
    store = as_memory_store(db)
    depth = bounded_int(max_depth, default=2, minimum=1, maximum=2)
    bounded_limit = bounded_int(limit, default=20, minimum=1, maximum=50)
    bounded_branch = bounded_int(branch_limit, default=8, minimum=1, maximum=20)
    diagnostics: dict[str, Any] = {
        "node_id": node_id,
        "max_depth": depth,
        "direction": direction,
        "relation_type": relation_type,
        "limit": bounded_limit,
        "branch_limit": bounded_branch,
        "endorsement_label": endorsement_label,
        "identity_id": (identity or {}).get("identity_id"),
        "exclusions": {
            "superseded": 0,
            "identity": 0,
            "endorsement": 0,
            "duplicate_edge": 0,
            # Dropped by the caps, so "20 shown" is never read as "20 exist".
            "node_limit": 0,
            "branch_limit": 0,
        },
        # Nodes whose edge fetch hit its own limit (more edges may exist).
        "edge_fetch_truncated": 0,
    }
    object_id = parse_object_id(node_id)
    if not object_id:
        return {"ok": False, "reason": "invalid_node_id", "nodes": [], "edges": [], "diagnostics": diagnostics}
    focus = store.get_node(object_id)
    if not focus:
        return {"ok": False, "reason": "node_not_found", "nodes": [], "edges": [], "diagnostics": diagnostics}

    focus_payload = compact_explore_node(serialize_node(focus), hop=0)
    nodes_by_id = {focus_payload["node_id"]: focus_payload}
    edges: list[dict[str, Any]] = []
    seen_edge_ids: set[str] = set()
    frontier = [focus_payload["node_id"]]
    exclusions = diagnostics["exclusions"]

    fetch_limit = max(bounded_branch * 4, 20)
    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for current_id in frontier:
            current_edges = graph_edges_for_node(
                db,
                node_id=current_id,
                direction=direction,
                relation_type=relation_type,
                limit=fetch_limit,
            )
            if len(current_edges) >= fetch_limit:
                diagnostics["edge_fetch_truncated"] += 1
            # branch_limit is per node: every frontier node gets its own
            # allowance instead of the first one filling the whole hop.
            branch_count = 0
            for index, edge in enumerate(current_edges):
                if branch_count >= bounded_branch:
                    exclusions["branch_limit"] += len(current_edges) - index
                    break
                edge_id = str(edge.get("edge_id") or "")
                if edge_id and edge_id in seen_edge_ids:
                    exclusions["duplicate_edge"] += 1
                    continue
                adjacent = adjacent_node_for_edge(edge, current_id)
                if not adjacent:
                    continue
                if is_superseded_node(adjacent):
                    exclusions["superseded"] += 1
                    continue
                if identity and not node_visible_to_identity(adjacent, identity):
                    exclusions["identity"] += 1
                    continue
                if endorsement_label and adjacent.get("endorsement_label") != endorsement_label:
                    exclusions["endorsement"] += 1
                    continue
                if edge_id:
                    seen_edge_ids.add(edge_id)
                payload = compact_explore_node(adjacent, hop=hop)
                adjacent_id = payload["node_id"]
                if not adjacent_id:
                    continue
                if hop > 1 and adjacent_id == focus_payload["node_id"]:
                    continue
                if adjacent_id not in nodes_by_id:
                    if len(nodes_by_id) >= bounded_limit + 1:
                        exclusions["node_limit"] += 1
                        continue
                    nodes_by_id[adjacent_id] = payload
                    next_frontier.append(adjacent_id)
                edges.append(compact_explore_edge(edge, hop=hop))
                branch_count += 1
        frontier = next_frontier
        if not frontier:
            break

    ordered_nodes = [nodes_by_id[focus_payload["node_id"]]] + [
        node for node_id_key, node in nodes_by_id.items() if node_id_key != focus_payload["node_id"]
    ]
    return {
        "ok": True,
        "reason": None,
        "focus": focus_payload,
        "nodes": ordered_nodes,
        "edges": edges,
        "diagnostics": diagnostics,
    }


def compact_explore_node(node: dict[str, Any], hop: int) -> dict[str, Any]:
    provenance = node.get("provenance") if isinstance(node.get("provenance"), dict) else {}
    compact_prov = {
        key: provenance[key]
        for key in ("source_path", "adapter", "source", "source_checksum_sha256")
        if provenance.get(key)
    }
    node_id = node.get("node_id") or node.get("_id")
    document_id = node.get("document_id")
    text_preview = node.get("text_preview")
    if not text_preview:
        text_preview = str(node.get("text") or "")[:300]
    return {
        "node_id": str(node_id) if node_id else None,
        "title": node.get("title"),
        "node_key": node.get("node_key"),
        "document_id": str(document_id) if document_id else None,
        "labels": node.get("labels") or [],
        "endorsement_label": node.get("endorsement_label"),
        "origin_date": node.get("origin_date"),
        "text_preview": text_preview,
        "status": node.get("status"),
        "hop": hop,
        "provenance": compact_prov,
    }


def compact_explore_edge(edge: dict[str, Any], hop: int) -> dict[str, Any]:
    summary = edge_summary(edge)
    summary["hop"] = hop
    return summary


def render_graph_explore_text(report: dict[str, Any]) -> str:
    if not report.get("ok"):
        reason = report.get("reason") or "unavailable"
        return f"Graph neighborhood: {reason.replace('_', ' ')}."
    focus = report.get("focus") or {}
    diagnostics = report.get("diagnostics") or {}
    nodes = {node.get("node_id"): node for node in report.get("nodes") or [] if node.get("node_id")}
    lines = [
        f"Graph neighborhood: {focus.get('title') or focus.get('node_id') or 'unknown'}",
        (
            f"focus: {focus.get('node_id')} | "
            f"endorsement {focus.get('endorsement_label') or 'none'} | "
            f"origin {focus.get('origin_date') or 'unknown'}"
        ),
        (
            f"hops: {diagnostics.get('max_depth')} | "
            f"nodes {len(report.get('nodes') or [])} | "
            f"edges {len(report.get('edges') or [])}"
        ),
    ]
    if focus.get("provenance", {}).get("source_path"):
        lines.append(f"provenance: {focus['provenance']['source_path']}")
    hop1_edges = [edge for edge in report.get("edges") or [] if edge.get("hop") == 1]
    if not hop1_edges:
        lines.append("")
        lines.append("No neighboring edges in this neighborhood.")
        return "\n".join(lines)
    for index, edge in enumerate(hop1_edges, start=1):
        neighbor_id = _other_node_id(edge, focus.get("node_id"))
        neighbor = nodes.get(neighbor_id) or {}
        lines.append("")
        lines.append(_edge_line(index, neighbor, edge, indent=""))
        lines.extend(_edge_detail_lines(edge, neighbor, indent="   "))
        child_edges = [
            child
            for child in report.get("edges") or []
            if child.get("hop") == 2 and _edge_touches(child, neighbor_id)
        ]
        for child_index, child in enumerate(child_edges, start=1):
            grandchild_id = _other_node_id(child, neighbor_id)
            grandchild = nodes.get(grandchild_id) or {}
            marker = f"{index}.{child_index}"
            lines.append(_edge_line(marker, grandchild, child, indent="   "))
            lines.extend(_edge_detail_lines(child, grandchild, indent="      "))
    return "\n".join(lines)


def render_graph_explore_mermaid(report: dict[str, Any]) -> str:
    if not report.get("ok"):
        reason = report.get("reason") or "unavailable"
        return f"%% graph neighborhood unavailable: {reason}"
    focus = report.get("focus") or {}
    lines = ["flowchart LR"]
    seen_nodes: set[str] = set()
    for node in report.get("nodes") or []:
        node_id = node.get("node_id")
        if not node_id or node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        lines.append(f'  {mermaid_id(node_id)}["{mermaid_label(node)}"]')
    for edge in report.get("edges") or []:
        source = edge.get("source_node_id")
        target = edge.get("target_node_id")
        if not source or not target:
            continue
        relation = mermaid_safe(str(edge.get("relation_type") or "edge"))
        lines.append(f"  {mermaid_id(source)} -- {relation} --> {mermaid_id(target)}")
    if focus.get("node_id"):
        lines.append(f"  style {mermaid_id(focus['node_id'])} fill:#1b3,stroke:#0a2,color:#fff")
    return "\n".join(lines)


def mermaid_id(node_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(node_id))
    return f"n{cleaned[:24] or 'node'}"


def mermaid_label(node: dict[str, Any]) -> str:
    title = mermaid_safe(str(node.get("title") or node.get("node_id") or "node"))
    if len(title) > 42:
        title = title[:41] + "…"
    hop = node.get("hop")
    if hop:
        return f"{title} (h{hop})"
    return title


def mermaid_safe(value: str) -> str:
    return (
        value.replace("\\", "/")
        .replace('"', "'")
        .replace("<", "(")
        .replace(">", ")")
        .replace("\n", " ")
        .replace("[", "(")
        .replace("]", ")")
    )


def _other_node_id(edge: dict[str, Any], node_id: str | None) -> str | None:
    if edge.get("source_node_id") == node_id:
        return edge.get("target_node_id")
    if edge.get("target_node_id") == node_id:
        return edge.get("source_node_id")
    return edge.get("target_node_id") or edge.get("source_node_id")


def _edge_touches(edge: dict[str, Any], node_id: str | None) -> bool:
    return node_id in {edge.get("source_node_id"), edge.get("target_node_id")}


def _edge_line(index: Any, node: dict[str, Any], edge: dict[str, Any], indent: str) -> str:
    return (
        f"{indent}{index}. {node.get('title') or node.get('node_id') or 'node'} | "
        f"{edge.get('relation_type') or 'relation'} | hop {edge.get('hop')}"
    )


def _edge_detail_lines(edge: dict[str, Any], node: dict[str, Any], indent: str) -> list[str]:
    lines = [
        f"{indent}node: {node.get('node_id')}",
        (
            f"{indent}edge: {edge.get('edge_id') or 'unknown'} | "
            f"{edge.get('provenance_source') or 'unknown source'}"
        ),
    ]
    if edge.get("reviewer"):
        lines.append(f"{indent}reviewed by: {edge['reviewer']}")
    if node.get("endorsement_label"):
        lines.append(f"{indent}endorsement: {node['endorsement_label']}")
    if node.get("origin_date"):
        lines.append(f"{indent}origin date: {node['origin_date']}")
    provenance = node.get("provenance") or {}
    if provenance.get("source_path"):
        lines.append(f"{indent}provenance: {provenance['source_path']}")
    return lines
