from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from pymongo.database import Database

from tirzah.db.memory_store import MemoryStore, as_memory_store
from tirzah.retrieval.queries import (
    STRUCTURAL_LABELS,
    embedding_candidate_nodes,
    parse_object_id,
    serialize_node,
)


CONTRADICTION_RELATION_TYPE = "contradicts"
CONTRADICTION_CANDIDATE_SOURCE = "contradiction_signals"
DEFAULT_CONTRADICTION_MIN_SIMILARITY = 0.82
DEFAULT_CONTRADICTION_MAX_SIMILARITY = 0.97
MIN_CONTRADICTION_CUES = 2
MIN_ORIGIN_DATE_DELTA_DAYS = 1

# Whole-word disagreement cues. Weak terms like "not" still need a second
# independent cue (shared labels or a date delta) before a candidate is queued.
CONFLICT_LEXICON = (
    "not",
    "never",
    "unlike",
    "however",
    "instead",
    "contrary",
    "false",
    "incorrect",
    "wrong",
    "vs",
    "versus",
    "contradict",
    "contradiction",
    "deny",
    "refute",
)
CONFLICT_LEXICON_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in CONFLICT_LEXICON) + r")\b",
    re.IGNORECASE,
)


def contradiction_candidate_nodes(
    db: Database | MemoryStore,
    node_id: str,
    limit: int = 10,
    include_same_document: bool = False,
    min_similarity: float = DEFAULT_CONTRADICTION_MIN_SIMILARITY,
    max_similarity: float = DEFAULT_CONTRADICTION_MAX_SIMILARITY,
    candidate_scan_limit: int | None = None,
) -> list[dict[str, Any]]:
    return contradiction_candidate_report(
        db,
        node_id=node_id,
        limit=limit,
        include_same_document=include_same_document,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        candidate_scan_limit=candidate_scan_limit,
    )["nodes"]


def contradiction_candidate_report(
    db: Database | MemoryStore,
    node_id: str,
    limit: int = 10,
    include_same_document: bool = False,
    min_similarity: float = DEFAULT_CONTRADICTION_MIN_SIMILARITY,
    max_similarity: float = DEFAULT_CONTRADICTION_MAX_SIMILARITY,
    candidate_scan_limit: int | None = None,
) -> dict[str, Any]:
    store = as_memory_store(db)
    bounded_limit = bounded_contradiction_limit(limit)
    min_threshold = bounded_similarity(
        min_similarity, default=DEFAULT_CONTRADICTION_MIN_SIMILARITY
    )
    max_threshold = bounded_similarity(
        max_similarity, default=DEFAULT_CONTRADICTION_MAX_SIMILARITY
    )
    if max_threshold < min_threshold:
        max_threshold = min_threshold
    diagnostics: dict[str, Any] = {
        "node_id": node_id,
        "limit": bounded_limit,
        "include_same_document": include_same_document,
        "min_similarity": min_threshold,
        "max_similarity": max_threshold,
        "min_cues": MIN_CONTRADICTION_CUES,
        "candidate_scan_limit": candidate_scan_limit,
        "candidate_source": CONTRADICTION_CANDIDATE_SOURCE,
        "relation_type": CONTRADICTION_RELATION_TYPE,
        "considered_count": 0,
        "returned_count": 0,
        "exclusions": {
            "above_max_similarity": 0,
            "insufficient_cues": 0,
            "missing_target": 0,
        },
    }
    node_object_id = parse_object_id(node_id)
    if not node_object_id:
        return {
            "ok": False,
            "reason": "invalid_node_id",
            "nodes": [],
            "diagnostics": diagnostics,
        }
    focus = store.get_node(node_object_id)
    if not focus:
        return {
            "ok": False,
            "reason": "node_not_found",
            "nodes": [],
            "diagnostics": diagnostics,
        }

    considered_limit = max(bounded_limit * 5, bounded_limit)
    neighbors = embedding_candidate_nodes(
        db,
        node_id=node_id,
        limit=considered_limit,
        include_same_document=include_same_document,
        min_similarity=min_threshold,
        candidate_scan_limit=candidate_scan_limit,
    )
    diagnostics["considered_count"] = len(neighbors)
    nodes: list[dict[str, Any]] = []
    for neighbor in neighbors:
        target_id = parse_object_id(neighbor.get("node_id"))
        if target_id is None:
            diagnostics["exclusions"]["missing_target"] += 1
            continue
        target = store.get_node(target_id)
        if not target:
            diagnostics["exclusions"]["missing_target"] += 1
            continue
        similarity = safe_float(neighbor.get("embedding_similarity"))
        if similarity >= max_threshold:
            diagnostics["exclusions"]["above_max_similarity"] += 1
            continue
        signals = contradiction_signals_for_pair(
            focus,
            target,
            embedding_similarity=similarity,
            min_similarity=min_threshold,
            max_similarity=max_threshold,
        )
        if not signals.get("meets_conservative_bar"):
            diagnostics["exclusions"]["insufficient_cues"] += 1
            continue
        nodes.append(serialize_contradiction_candidate(focus, target, neighbor, signals))
        if len(nodes) >= bounded_limit:
            break
    diagnostics["returned_count"] = len(nodes)
    return {
        "ok": True,
        "reason": None,
        "nodes": nodes,
        "diagnostics": diagnostics,
    }


def contradiction_signals_for_pair(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    embedding_similarity: float | None = None,
    min_similarity: float = DEFAULT_CONTRADICTION_MIN_SIMILARITY,
    max_similarity: float = DEFAULT_CONTRADICTION_MAX_SIMILARITY,
) -> dict[str, Any]:
    shared = shared_semantic_label_values(source, target)
    delta_days = origin_date_delta_days(source.get("origin_date"), target.get("origin_date"))
    conflict_terms = sorted(
        set(conflict_lexicon_matches(node_conflict_text(source)))
        | set(conflict_lexicon_matches(node_conflict_text(target)))
    )
    cues = {
        "shared_semantic_labels": bool(shared),
        "origin_date_delta": delta_days is not None
        and delta_days >= MIN_ORIGIN_DATE_DELTA_DAYS,
        "conflict_lexicon": bool(conflict_terms),
    }
    cue_count = sum(1 for matched in cues.values() if matched)
    similarity = safe_float(embedding_similarity)
    similarity_in_band = min_similarity <= similarity < max_similarity
    return {
        **cues,
        "cue_count": cue_count,
        "meets_conservative_bar": cue_count >= MIN_CONTRADICTION_CUES and similarity_in_band,
        "shared_labels": shared,
        "conflict_terms": conflict_terms,
        "origin_date_delta_days": delta_days,
        "embedding_similarity": round(similarity, 6) if embedding_similarity is not None else None,
        "similarity_in_band": similarity_in_band,
        "min_cues": MIN_CONTRADICTION_CUES,
    }


def serialize_contradiction_candidate(
    source: dict[str, Any],
    target: dict[str, Any],
    neighbor: dict[str, Any],
    signals: dict[str, Any],
) -> dict[str, Any]:
    serialized = serialize_node(target)
    serialized.update(
        {
            "embedding_similarity": neighbor.get("embedding_similarity"),
            "embedding_rank_score": neighbor.get("embedding_rank_score"),
            "link_index_penalty": neighbor.get("link_index_penalty"),
            "embedding_model": neighbor.get("embedding_model"),
            "embedding_dimensions": neighbor.get("embedding_dimensions"),
            "shared_labels": signals.get("shared_labels") or [],
            "shared_label_count": len(signals.get("shared_labels") or []),
            "relation_type": CONTRADICTION_RELATION_TYPE,
            "candidate_source": CONTRADICTION_CANDIDATE_SOURCE,
            "contradiction_signals": signals,
            "source_origin_date": source.get("origin_date"),
            "source_origin_date_source": source.get("origin_date_source"),
            "source_origin_date_confidence": source.get("origin_date_confidence"),
            "target_origin_date": target.get("origin_date"),
            "target_origin_date_source": target.get("origin_date_source"),
            "target_origin_date_confidence": target.get("origin_date_confidence"),
            "origin_date_delta_days": signals.get("origin_date_delta_days"),
            "source_provenance": compact_node_provenance(source),
            "target_provenance": compact_node_provenance(target),
        }
    )
    return serialized


def compact_node_provenance(node: dict[str, Any] | None) -> dict[str, Any]:
    provenance = dict((node or {}).get("provenance") or {})
    compact: dict[str, Any] = {}
    for key in (
        "source_path",
        "source_checksum_sha256",
        "checksum_sha256",
        "archive_path",
        "adapter",
        "ingestion_epoch",
        "source",
    ):
        value = provenance.get(key)
        if value:
            compact[key] = value
    return compact


def shared_semantic_label_values(source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    return sorted(
        label
        for label in set(source.get("labels") or []) & set(target.get("labels") or [])
        if label and label not in STRUCTURAL_LABELS and not str(label).startswith("source_")
    )


def node_conflict_text(node: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(node.get("title") or ""),
            str(node.get("text") or ""),
            str(node.get("text_preview") or ""),
            str(node.get("summary") or ""),
        )
        if part
    )


def conflict_lexicon_matches(text: str) -> list[str]:
    if not text:
        return []
    return sorted({match.group(1).casefold() for match in CONFLICT_LEXICON_PATTERN.finditer(text)})


def origin_date_delta_days(source_date: Any, target_date: Any) -> int | None:
    left = parse_origin_date(source_date)
    right = parse_origin_date(target_date)
    if left is None or right is None:
        return None
    return abs((left - right).days)


def parse_origin_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def bounded_contradiction_limit(value: Any, maximum: int = 50) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 10
    return max(1, min(parsed, maximum))


def bounded_similarity(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(-1.0, min(parsed, 1.0))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
