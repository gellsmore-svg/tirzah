from __future__ import annotations

from datetime import datetime, timezone
from math import pow
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.database import Database

from tirzah.db.memory_store import MemoryStore, as_memory_store
from tirzah.db.governance import get_trust_weighting_profile
from tirzah.retrieval.queries import parsed_usage_score, serialize_node


ENDORSEMENT_TRUST = {
    "explicit_endorsed": 1.0,
    "implicit_endorsed": 0.8,
    "unreviewed": 0.55,
    "rejected": 0.0,
}


def trust_temporal_diagnostic_for_node(
    db: Database | MemoryStore,
    node_id: str,
    *,
    weighting_profile_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    try:
        object_id = ObjectId(node_id)
    except (InvalidId, TypeError):
        return None
    store = as_memory_store(db)
    node = store.get_node(object_id)
    if not node:
        return None
    profile_id = weighting_profile_id or node.get("temporal_profile_id")
    profile = get_trust_weighting_profile(store.db, profile_id) if profile_id else None
    diagnostic = trust_temporal_diagnostic(node, profile=profile, now=now)
    return {
        "node": serialize_node(node),
        "weighting_profile": profile,
        "diagnostic": diagnostic,
    }


def trust_temporal_diagnostics_for_nodes(
    db: Database | MemoryStore,
    node_ids: list[str],
    *,
    weighting_profile_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    object_ids_by_text: dict[str, ObjectId] = {}
    for node_id in node_ids:
        try:
            object_ids_by_text[str(node_id)] = ObjectId(str(node_id))
        except (InvalidId, TypeError):
            continue
    if not object_ids_by_text:
        return {}
    store = as_memory_store(db)
    nodes = store.find_nodes({"_id": {"$in": list(object_ids_by_text.values())}})
    nodes_by_id = {str(node["_id"]): node for node in nodes}
    shared_profile = (
        get_trust_weighting_profile(store.db, weighting_profile_id)
        if weighting_profile_id
        else None
    )
    profile_cache: dict[str, dict[str, Any] | None] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for node_id in object_ids_by_text:
        node = nodes_by_id.get(node_id)
        if not node:
            continue
        profile = shared_profile
        if profile is None:
            profile_id = node.get("temporal_profile_id")
            if profile_id:
                if profile_id not in profile_cache:
                    profile_cache[profile_id] = get_trust_weighting_profile(store.db, profile_id)
                profile = profile_cache[profile_id]
        diagnostics[node_id] = {
            "node": serialize_node(node),
            "weighting_profile": profile,
            "diagnostic": trust_temporal_diagnostic(node, profile=profile, now=now),
        }
    return diagnostics


def trust_temporal_diagnostic(
    node: dict[str, Any],
    *,
    profile: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    profile = profile or {}
    trust_score = explicit_or_endorsement_trust(node)
    recency_component = temporal_recency_component(
        origin_or_created_timestamp(node) or node.get("last_verified_at") or node.get("created_at"),
        profile.get("default_decay_half_life_days"),
        current_time,
    )
    stability = bounded_float(profile.get("stability_importance"), default=0.0)
    temporal_component = bounded_float(stability + ((1.0 - stability) * recency_component))
    frequency_component = min(parsed_usage_score(node.get("usage_score")) / 10.0, 1.0)
    verification_component = verification_score(node)
    weights = {
        "trust": 1.0,
        "temporal": bounded_float(profile.get("recency_importance"), default=0.0),
        "frequency": bounded_float(profile.get("frequency_importance"), default=0.0),
        "verification": bounded_float(profile.get("verification_importance"), default=0.0),
    }
    weighted_score = weighted_average(
        {
            "trust": trust_score,
            "temporal": temporal_component,
            "frequency": frequency_component,
            "verification": verification_component,
        },
        weights,
    )
    return {
        "score": round(weighted_score, 4),
        "components": {
            "trust": round(trust_score, 4),
            "temporal": round(temporal_component, 4),
            "recency": round(recency_component, 4),
            "frequency": round(frequency_component, 4),
            "verification": round(verification_component, 4),
        },
        "weights": weights,
        "signals": {
            "endorsement_label": node.get("endorsement_label"),
            "explicit_trust_score": node.get("trust_score"),
            "usage_score": parsed_usage_score(node.get("usage_score")),
            "created_at": iso(node.get("created_at")),
            "origin_date": node.get("origin_date"),
            "origin_date_source": node.get("origin_date_source"),
            "origin_date_confidence": node.get("origin_date_confidence"),
            "last_used_at": iso(node.get("last_used_at")),
            "last_verified_at": iso(node.get("last_verified_at")),
            "verification_required": bool(node.get("verification_required")),
        },
    }


def trust_ranking_boost(score: float, *, max_boost: int = 20, weight: float = 1.0) -> int:
    delta = (bounded_float(score, default=0.5) - 0.5) * 2
    return int(max(-max_boost, min(max_boost, round(delta * max_boost * weight))))


def trust_ranking_unit_boost(score: float, *, weight: float = 0.15) -> float:
    delta = (bounded_float(score, default=0.5) - 0.5) * 2
    bound = abs(weight)
    return round(max(-bound, min(bound, delta * bound)), 6)


def apply_trust_ranking(
    rows: list[dict[str, Any]],
    *,
    enabled: bool = False,
    profile: dict[str, Any] | None = None,
    weight: float = 1.0,
    max_boost: int = 20,
    hybrid_weight: float = 0.15,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    if not enabled or not rows:
        return rows
    ranked = []
    hybrid = any(row.get("hybrid_score") is not None for row in rows)
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["rank_before"] = index
        diagnostic = trust_temporal_diagnostic(updated, profile=profile, now=now)
        if hybrid and updated.get("hybrid_score") is not None:
            primary = float(updated["hybrid_score"])
            boost = trust_ranking_unit_boost(diagnostic["score"], weight=hybrid_weight)
            after = round(primary + boost, 6)
        else:
            primary = float(updated.get("lexical_score") or 0)
            boost = trust_ranking_boost(diagnostic["score"], max_boost=max_boost, weight=weight)
            after = primary + boost
        updated["trust_ranking"] = {
            "primary_score": primary,
            "trust_score": diagnostic["score"],
            "trust_boost": boost,
            "ranking_score_before": primary,
            "ranking_score_after": after,
            "profile_id": (profile or {}).get("weighting_profile_id"),
            "components": diagnostic.get("components"),
        }
        ranked.append(updated)
    ranked.sort(
        key=lambda row: (
            float((row.get("trust_ranking") or {}).get("ranking_score_after") or 0),
            -int(row.get("rank_before") or 0),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked):
        row["rank_after"] = index
        ranking = row.get("trust_ranking") or {}
        ranking["position_delta"] = int(row.get("rank_before") or 0) - index
        row["trust_ranking"] = ranking
    return ranked


def origin_or_created_timestamp(node: dict[str, Any]) -> datetime | None:
    origin = node.get("origin_date")
    if isinstance(origin, str) and origin:
        try:
            year, month, day = (int(part) for part in origin[:10].split("-"))
            return datetime(year, month, day, tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None
    return None


def explicit_or_endorsement_trust(node: dict[str, Any]) -> float:
    if node.get("trust_score") is not None:
        return bounded_float(node.get("trust_score"))
    return ENDORSEMENT_TRUST.get(node.get("endorsement_label"), 0.5)


def verification_score(node: dict[str, Any]) -> float:
    if not node.get("verification_required"):
        return 1.0
    if node.get("last_verified_at"):
        return 1.0
    return 0.0


def temporal_recency_component(
    timestamp: Any,
    half_life_days: Any,
    now: datetime,
) -> float:
    half_life = positive_float_or_none(half_life_days)
    if not timestamp or not half_life:
        return 1.0
    if not isinstance(timestamp, datetime):
        return 1.0
    age_seconds = max(0.0, (aware_utc(now) - aware_utc(timestamp)).total_seconds())
    age_days = age_seconds / 86400.0
    return bounded_float(pow(0.5, age_days / half_life))


def weighted_average(values: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(values[key] * weights[key] for key in weights) / total_weight


def bounded_float(value: Any, *, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(parsed, 1.0))


def positive_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso(value: Any) -> str | None:
    if not value:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return None
