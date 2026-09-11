from __future__ import annotations

from datetime import datetime, timezone

from pymongo.database import Database

from tirzah.db.schema import ensure_required_collections

# Bump when index set changes so ensure_indexes re-runs (review M4).
# 3: nodes.origin_date and (document_id, origin_date) for origin-date filters.
_INDEX_SCHEMA_VERSION = 3


LABEL_DEFINITIONS = [
    {
        "key": "unreviewed",
        "scope": "endorsement",
        "description": "Default state for ingested chunks whose content has not been explicitly or implicitly endorsed.",
    },
    {
        "key": "implicit_endorsed",
        "scope": "endorsement",
        "description": "Content inferred to be endorsed by trusted placement or workflow context.",
    },
    {
        "key": "explicit_endorsed",
        "scope": "endorsement",
        "description": "Content explicitly endorsed by a user action or direct instruction.",
    },
    {
        "key": "rejected",
        "scope": "endorsement",
        "description": "Content marked as not trusted for retrieval or reasoning.",
    },
    {
        "key": "source_chunk",
        "scope": "node",
        "description": "A node created directly from source document content during ingestion.",
    },
    {
        "key": "source_root",
        "scope": "node",
        "description": "The root node for a source document tree.",
    },
    {
        "key": "source_section",
        "scope": "node",
        "description": "A section-level node derived from markdown headings or deterministic section parsing.",
    },
    {
        "key": "external_corpus",
        "scope": "node",
        "description": "Imported working-data corpus content from an external source, not canonical project memory.",
    },
    {
        "key": "public_domain",
        "scope": "node",
        "description": "Source material identified as public domain for the current working jurisdiction.",
    },
    {
        "key": "memory_reference",
        "scope": "node",
        "description": "Reference material about memory, recall, cognition, or related retrieval concepts.",
    },
    {
        "key": "ams_domain",
        "scope": "node",
        "description": "Imported source material from the AMS domain repository.",
    },
    {
        "key": "imported_domain",
        "scope": "node",
        "description": "Material imported from another local domain into Tirzah for retrieval experiments.",
    },
    {
        "key": "research_corpus",
        "scope": "node",
        "description": "Research corpus material used as working retrieval data.",
    },
    {
        "key": "generated_output",
        "scope": "node",
        "description": "Content produced by a model or Tirzah workflow and captured for later review.",
    },
    {
        "key": "llm_answer",
        "scope": "node",
        "description": "A node created from a persisted LLM answer output.",
    },
]

DEFAULT_AGENT_IDENTITIES = [
    {
        "identity_id": "tirzah_shared",
        "title": "Tirzah Shared Identity",
        "kind": "shared",
        "description": "Default shared identity for project-wide retrieval and governance scaffolding.",
        "trusted_labels": [],
        "excluded_labels": [],
        "trusted_document_ids": [],
        "excluded_document_ids": [],
        "trusted_tree_ids": [],
        "excluded_tree_ids": [],
        "allowed_relation_types": [],
        "excluded_relation_types": [],
        "weighting_profile_id": "default_balanced",
        "required_process_ids": [],
        "governance_policy_ids": [],
    }
]

DEFAULT_TRUST_WEIGHTING_PROFILES = [
    {
        "weighting_profile_id": "default_balanced",
        "title": "Default Balanced Weighting",
        "recency_importance": 0.3,
        "frequency_importance": 0.2,
        "stability_importance": 0.5,
        "context_sensitivity": 0.4,
        "verification_importance": 0.5,
        # Non-null so recency actually participates in trust ranking; long,
        # because corpora here are historical. seed_governance_defaults $sets
        # this on every ensure_indexes, so existing installs pick it up.
        "default_decay_half_life_days": 1825,
    }
]


def ensure_indexes(db: Database, *, force: bool = False) -> None:
    """Create indexes if the schema version marker is missing/stale (review M4).

    Governance label/identity seeds still run every call (cheap upserts).
    Pass ``force=True`` from ``tirzah migrate`` to rebuild the index set.
    """
    ensure_required_collections(db)
    meta = db.tirzah_meta
    marker = meta.find_one({"_id": "indexes"})
    if (
        not force
        and marker
        and int(marker.get("version") or 0) == _INDEX_SCHEMA_VERSION
    ):
        seed_governance_defaults(db)
        return

    for index in db.documents.list_indexes():
        if index.get("key") == {"source.path": 1} and index.get("unique"):
            db.documents.drop_index(index["name"])
    db.documents.create_index("source.path")
    db.documents.create_index("source.checksum_sha256", unique=True, sparse=True)
    db.documents.create_index("title")
    db.documents.create_index("created_at")
    db.documents.create_index("ingestion_epoch")
    db.trees.create_index([("document_id", 1), ("label", 1)])
    db.trees.create_index([("document_id", 1), ("ingestion_epoch", 1)])
    db.nodes.create_index([("tree_id", 1), ("parent_id", 1)])
    db.nodes.create_index([("tree_id", 1), ("node_key", 1)], unique=True)
    db.nodes.create_index("labels")
    db.nodes.create_index("endorsement_label")
    db.nodes.create_index("provenance.source_checksum_sha256")
    db.nodes.create_index("title")
    db.nodes.create_index("text")
    db.nodes.create_index([("document_id", 1), ("created_at", -1)])
    db.nodes.create_index([("document_id", 1), ("ingestion_epoch", 1)])
    db.nodes.create_index([("embedding.model", 1), ("embedding.dimensions", 1), ("document_id", 1)])
    db.nodes.create_index("ingestion_epoch")
    db.nodes.create_index("created_at")
    db.nodes.create_index("origin_date")
    db.nodes.create_index([("document_id", 1), ("origin_date", 1)])
    db.nodes.create_index("usage_score")
    db.nodes.create_index("last_used_at")
    db.graph_edges.create_index([("source_node_id", 1), ("relation_type", 1)])
    db.graph_edges.create_index([("target_node_id", 1), ("relation_type", 1)])
    db.graph_edges.create_index([("document_id", 1), ("relation_type", 1)])
    db.graph_edges.create_index(
        [("source_node_id", 1), ("target_node_id", 1), ("relation_type", 1)],
        unique=True,
    )
    db.semantic_edge_candidates.create_index([("status", 1), ("created_at", -1)])
    db.semantic_edge_candidates.create_index("pair_key")
    db.semantic_edge_candidates.create_index(
        [("source_node_id", 1), ("target_node_id", 1), ("relation_type", 1)],
        unique=True,
    )
    db.agent_identities.create_index("identity_id", unique=True)
    db.agent_identities.create_index("kind")
    db.trust_weighting_profiles.create_index("weighting_profile_id", unique=True)
    db.governance_policies.create_index("policy_id", unique=True)
    db.governance_policies.create_index([("authority_layer", 1), ("priority", 1)])
    db.process_objects.create_index("process_id", unique=True)
    db.process_runs.create_index("run_id", unique=True)
    db.process_runs.create_index([("session_id", 1), ("status", 1)])
    db.recursive_plans.create_index([("plan_id", 1), ("revision", 1)], unique=True)
    db.recursive_plans.create_index([("session_id", 1), ("created_at", -1)])
    db.plan_executions.create_index(
        [("plan_id", 1), ("revision", 1), ("session_id", 1)],
        unique=True,
    )
    db.plan_executions.create_index([("session_id", 1), ("status", 1), ("updated_at", -1)])
    db.queue.create_index([("status", 1), ("created_at", 1)])
    db.queue.create_index("checksum_sha256")
    db.queue.create_index("path")
    db.exchanges.create_index([("session_id", 1), ("created_at", -1)])
    db.exchanges.create_index("focus_node_id")
    db.exchanges.create_index("output_ingestion_job_id")
    db.output_ingestion_queue.create_index([("status", 1), ("created_at", 1)])
    db.output_ingestion_queue.create_index([("session_id", 1), ("created_at", -1)])
    db.output_ingestion_queue.create_index("exchange_id", unique=True)
    db.output_ingestion_queue.create_index("content_hash_sha256")
    db.retrieval_traces.create_index("created_at")
    db.retrieval_traces.create_index("session_id")
    db.retrieval_traces.create_index("exchange_id")
    db.embedding_backfill_jobs.create_index([("status", 1), ("created_at", 1)])
    db.embedding_backfill_jobs.create_index([("updated_at", -1)])
    db.sessions.create_index("session_id", unique=True)
    db.sessions.create_index("updated_at")
    db.active_documents.create_index([("session_id", 1), ("last_referenced_at", -1)])
    db.active_documents.create_index([("session_id", 1), ("document_id", 1)], unique=True)
    db.session_continuity.create_index([("session_id", 1), ("is_latest", 1)])
    db.session_continuity.create_index([("session_id", 1), ("created_at", -1)])
    db.session_continuity.create_index("exchange_id")
    db.project_domains.create_index("domain_id", unique=True)
    db.project_domains.create_index("updated_at")
    db.conversation_domains.create_index("domain_id", unique=True)
    db.conversation_domains.create_index([("project_domain_id", 1), ("updated_at", -1)])
    db.conversation_domains.create_index([("session_id", 1), ("updated_at", -1)])
    db.label_definitions.create_index("key", unique=True)
    for definition in LABEL_DEFINITIONS:
        db.label_definitions.update_one(
            {"key": definition["key"]},
            {"$set": definition},
            upsert=True,
        )
    meta.update_one(
        {"_id": "indexes"},
        {
            "$set": {
                "version": _INDEX_SCHEMA_VERSION,
                "applied_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
    seed_governance_defaults(db)


def seed_governance_defaults(db: Database) -> None:
    now = datetime.now(timezone.utc)
    for identity in DEFAULT_AGENT_IDENTITIES:
        db.agent_identities.update_one(
            {"identity_id": identity["identity_id"]},
            {
                "$set": {**identity, "updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
    for profile in DEFAULT_TRUST_WEIGHTING_PROFILES:
        db.trust_weighting_profiles.update_one(
            {"weighting_profile_id": profile["weighting_profile_id"]},
            {
                "$set": {**profile, "updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
