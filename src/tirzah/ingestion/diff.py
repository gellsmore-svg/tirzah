from __future__ import annotations

import hashlib
from typing import Any

from tirzah.models.ingestion import INACTIVE_RETRIEVAL_STATUSES, IngestedNode

STRUCTURAL_LABELS = ("source_root", "source_section", "source_chunk")


def node_content_sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def existing_content_sha256(node: dict[str, Any]) -> str:
    stored = node.get("content_sha256") or (node.get("metadata") or {}).get("content_sha256")
    if stored:
        return str(stored)
    return node_content_sha256(node.get("text"))


def structural_label(labels: list[str] | None) -> str:
    for label in STRUCTURAL_LABELS:
        if label in (labels or []):
            return label
    return (labels or ["node"])[0]


def diff_ingestion_trees(
    existing_nodes: list[dict[str, Any]],
    proposed_nodes: list[IngestedNode] | list[dict[str, Any]],
) -> dict[str, Any]:
    existing = [node for node in existing_nodes if node.get("status") not in INACTIVE_RETRIEVAL_STATUSES]
    proposed = [as_proposed(node) for node in proposed_nodes]
    unmatched_existing = list(existing)
    unmatched_proposed = list(proposed)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for matcher in (_match_by_keys, _match_by_hash, _match_by_title):
        still_unmatched: list[dict[str, Any]] = []
        for proposed_node in unmatched_proposed:
            match = matcher(proposed_node, unmatched_existing)
            if match is None:
                still_unmatched.append(proposed_node)
                continue
            unmatched_existing.remove(match)
            pairs.append((match, proposed_node))
        unmatched_proposed = still_unmatched

    unchanged: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    for existing_node, proposed_node in pairs:
        entry = {
            "node_id": str(existing_node.get("_id")) if existing_node.get("_id") is not None else None,
            "node_key": proposed_node["node_key"],
            "parent_key": proposed_node["parent_key"],
            "previous_node_key": existing_node.get("node_key"),
            "previous_parent_key": existing_node.get("parent_key"),
            "title": proposed_node["title"],
            "previous_title": existing_node.get("title"),
            "content_sha256": proposed_node["content_sha256"],
            "previous_content_sha256": existing_content_sha256(existing_node),
            "existing": existing_node,
            "proposed": proposed_node,
        }
        same_content = entry["content_sha256"] == entry["previous_content_sha256"]
        same_title = (proposed_node["title"] or "") == (existing_node.get("title") or "")
        same_place = (
            proposed_node["node_key"] == existing_node.get("node_key")
            and proposed_node["parent_key"] == existing_node.get("parent_key")
        )
        if same_content and same_place and same_title:
            unchanged.append(entry)
        elif same_content and not same_place:
            moved.append(entry)
        else:
            if not same_place:
                entry["moved"] = True
            changed.append(entry)

    added = [
        {
            "node_id": None,
            "node_key": node["node_key"],
            "parent_key": node["parent_key"],
            "title": node["title"],
            "content_sha256": node["content_sha256"],
            "proposed": node,
        }
        for node in unmatched_proposed
    ]
    removed = [
        {
            "node_id": str(node.get("_id")) if node.get("_id") is not None else None,
            "node_key": node.get("node_key"),
            "parent_key": node.get("parent_key"),
            "title": node.get("title"),
            "content_sha256": existing_content_sha256(node),
            "existing": node,
        }
        for node in unmatched_existing
    ]
    return {
        "unchanged": unchanged,
        "changed": changed,
        "moved": moved,
        "added": added,
        "removed": removed,
        "counts": {
            "unchanged": len(unchanged),
            "changed": len(changed),
            "moved": len(moved),
            "added": len(added),
            "removed": len(removed),
            "existing": len(existing),
            "proposed": len([as_proposed(node) for node in proposed_nodes]),
        },
    }


def serialize_rebuild_diff(diff: dict[str, Any]) -> dict[str, Any]:
    def slim(entries: list[dict[str, Any]], *, include_proposed: bool = False) -> list[dict[str, Any]]:
        rows = []
        for entry in entries:
            row = {
                "node_id": entry.get("node_id"),
                "node_key": entry.get("node_key"),
                "parent_key": entry.get("parent_key"),
                "title": entry.get("title"),
                "content_sha256": entry.get("content_sha256"),
            }
            if entry.get("previous_node_key") and entry.get("previous_node_key") != entry.get("node_key"):
                row["previous_node_key"] = entry.get("previous_node_key")
            if entry.get("previous_parent_key") and entry.get("previous_parent_key") != entry.get("parent_key"):
                row["previous_parent_key"] = entry.get("previous_parent_key")
            if entry.get("previous_title") and entry.get("previous_title") != entry.get("title"):
                row["previous_title"] = entry.get("previous_title")
            if include_proposed:
                proposed = entry.get("proposed") or {}
                row["text_preview"] = str(proposed.get("text") or "")[:240]
            rows.append(row)
        return rows

    return {
        "counts": diff["counts"],
        "unchanged": slim(diff["unchanged"]),
        "changed": slim(diff["changed"], include_proposed=True),
        "moved": slim(diff["moved"]),
        "added": slim(diff["added"], include_proposed=True),
        "removed": slim(diff["removed"]),
    }


def as_proposed(node: IngestedNode | dict[str, Any]) -> dict[str, Any]:
    if isinstance(node, IngestedNode):
        payload = node.model_dump()
    else:
        payload = dict(node)
    payload["content_sha256"] = node_content_sha256(payload.get("text"))
    payload["structural_label"] = structural_label(payload.get("labels"))
    payload.setdefault("parent_key", None)
    return payload


def _match_by_keys(proposed: dict[str, Any], existing_nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [
        node
        for node in existing_nodes
        if node.get("node_key") == proposed["node_key"]
        and node.get("parent_key") == proposed["parent_key"]
    ]
    return matches[0] if len(matches) == 1 else None


def _match_by_hash(proposed: dict[str, Any], existing_nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [
        node
        for node in existing_nodes
        if existing_content_sha256(node) == proposed["content_sha256"]
        and structural_label(node.get("labels")) == proposed["structural_label"]
    ]
    return matches[0] if len(matches) == 1 else None


def _match_by_title(proposed: dict[str, Any], existing_nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    title = (proposed.get("title") or "").strip().lower()
    if not title:
        return None
    matches = [
        node
        for node in existing_nodes
        if (node.get("title") or "").strip().lower() == title
        and structural_label(node.get("labels")) == proposed["structural_label"]
        and node.get("parent_key") == proposed["parent_key"]
    ]
    return matches[0] if len(matches) == 1 else None


