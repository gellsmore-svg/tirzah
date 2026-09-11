from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from pymongo.database import Database

from tirzah.db.memory_store import MemoryStore, as_memory_store
from tirzah.ingestion.formats import FORMAT_LABELS
from tirzah.retrieval.queries import (
    STRUCTURAL_LABELS,
    embedding_candidate_report,
    parse_object_id,
    serialize_node,
)
from tirzah.retrieval.reformulate import light_stems


CONTRADICTION_RELATION_TYPE = "contradicts"
CONTRADICTION_CANDIDATE_SOURCE = "contradiction_signals"
# Similarity only bounds the neighbour scan. Denial lowers embedding similarity
# relative to paraphrase, so a high floor filters out exactly the pairs sought
# (the genuine pairs in #64 measured 0.77-0.82 with nomic-embed-text, while
# paraphrases sat above 0.82). The ceiling still drops near-duplicates.
# Re-measure when changing embedding model.
DEFAULT_CONTRADICTION_MIN_SIMILARITY = 0.6
DEFAULT_CONTRADICTION_MAX_SIMILARITY = 0.97
MIN_ORIGIN_DATE_DELTA_DAYS = 1
# Share of the smaller node's claim words the pair must have in common for a
# one-sided negation to read as a denial of the same claim.
SHARED_CLAIM_MIN_OVERLAP = 0.4
CONTRADICTION_ADMISSION_RULE = (
    "a negation/refutation term on one side only, over shared claim wording "
    f"(overlap >= {SHARED_CLAIM_MIN_OVERLAP}), with similarity in [min, max); "
    "shared labels and origin-date delta only break ties"
)

# Whole-word disagreement vocabulary, shown to reviewers as conflict terms.
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
# Terms that negate or refute a claim, unlike discourse markers ("however",
# "vs", "unlike", "instead") that turn up just as often in agreeing prose.
NEGATION_TERMS = frozenset(
    {"not", "never", "false", "incorrect", "wrong", "contrary", "contradict", "contradiction", "deny", "refute"}
)
CLAIM_STOPWORDS = frozenset(
    "a an the is are was were be been being of to in on at by for with from and or but if as "
    "it its this that these those there their they them we you which who what when where how "
    "than then so such can could may might must shall should will would do does did has have had".split()
)
_CLAIM_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
# Ingestion format labels say how a file was parsed, not what it is about.
FORMAT_LABEL_VALUES = frozenset(label for labels in FORMAT_LABELS.values() for label in labels)


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
        "admission_rule": CONTRADICTION_ADMISSION_RULE,
        "candidate_scan_limit": candidate_scan_limit,
        "candidate_source": CONTRADICTION_CANDIDATE_SOURCE,
        "relation_type": CONTRADICTION_RELATION_TYPE,
        "considered_count": 0,
        "returned_count": 0,
        "exclusions": {
            "below_min_similarity": 0,
            "above_max_similarity": 0,
            "no_disagreement_evidence": 0,
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
    neighbor_report = embedding_candidate_report(
        db,
        node_id=node_id,
        limit=considered_limit,
        include_same_document=include_same_document,
        min_similarity=min_threshold,
        candidate_scan_limit=candidate_scan_limit,
    )
    neighbors = neighbor_report.get("nodes") or []
    # Surface the scan's own drops so "0 considered" is explainable.
    scan = neighbor_report.get("diagnostics") or {}
    scan_exclusions = dict(scan.get("exclusions") or {})
    diagnostics["embedding_scan"] = {
        "ok": neighbor_report.get("ok"),
        "reason": neighbor_report.get("reason"),
        "scanned_count": scan.get("scanned_count", 0),
        "scan_truncated": bool(scan.get("scan_truncated")),
        "exclusions": scan_exclusions,
    }
    diagnostics["exclusions"]["below_min_similarity"] = int(scan_exclusions.get("below_threshold") or 0)
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
            diagnostics["exclusions"]["no_disagreement_evidence"] += 1
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
    """Disagreement evidence for a pair. Admission (``meets_conservative_bar``)
    needs a negation/refutation term on one side only over shared claim
    wording - "X is Y" against "X is not Y". Shared labels and an origin-date
    delta carry no disagreement information, so they are reported (and
    counted in ``cue_count``) only as tie-breakers."""
    shared = shared_semantic_label_values(source, target)
    delta_days = origin_date_delta_days(source.get("origin_date"), target.get("origin_date"))
    source_terms = set(conflict_lexicon_matches(node_conflict_text(source)))
    target_terms = set(conflict_lexicon_matches(node_conflict_text(target)))
    conflict_terms = sorted(source_terms | target_terms)
    negation_asymmetry = bool((source_terms ^ target_terms) & NEGATION_TERMS)
    overlap = claim_overlap(source, target)
    negated_shared_claim = negation_asymmetry and overlap >= SHARED_CLAIM_MIN_OVERLAP
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
        "negation_asymmetry": negation_asymmetry,
        "claim_overlap": round(overlap, 4),
        "negated_shared_claim": negated_shared_claim,
        "meets_conservative_bar": negated_shared_claim and similarity_in_band,
        "shared_labels": shared,
        "conflict_terms": conflict_terms,
        "origin_date_delta_days": delta_days,
        "embedding_similarity": round(similarity, 6) if embedding_similarity is not None else None,
        "similarity_in_band": similarity_in_band,
        "admission_rule": CONTRADICTION_ADMISSION_RULE,
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
        if label
        and label not in STRUCTURAL_LABELS
        and label not in FORMAT_LABEL_VALUES
        and not str(label).startswith("source_")
    )


def claim_tokens(node: dict[str, Any]) -> set[str]:
    """Lightly stemmed content words of a node's title and text, excluding
    stopwords and the conflict vocabulary itself."""
    text = f"{node.get('title') or ''} {node.get('text') or node.get('text_preview') or ''}".lower()
    tokens = set()
    for token in _CLAIM_TOKEN_PATTERN.findall(text):
        if len(token) < 3 or token in CLAIM_STOPWORDS or token in CONFLICT_LEXICON:
            continue
        stems = light_stems(token)
        tokens.add(stems[0] if stems else token)
    return tokens


def claim_overlap(source: dict[str, Any], target: dict[str, Any]) -> float:
    """Shared claim words as a share of the smaller node's claim words."""
    left, right = claim_tokens(source), claim_tokens(target)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


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
