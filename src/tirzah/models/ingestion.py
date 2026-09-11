from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


SCHEMA_VERSION = 1
DEFAULT_ENDORSEMENT_LABEL = "unreviewed"
INGESTION_KIND_DETERMINISTIC = "deterministic"
INGESTION_KIND_LLM_PROPOSED = "llm_proposed"
TREE_STATUS_ACTIVE = "active"
TREE_STATUS_PENDING_REVIEW = "pending_review"
TREE_STATUS_REJECTED = "rejected"
TREE_STATUS_SUPERSEDED = "superseded"
INACTIVE_RETRIEVAL_STATUSES = (
    TREE_STATUS_SUPERSEDED,
    TREE_STATUS_PENDING_REVIEW,
    TREE_STATUS_REJECTED,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceRef(BaseModel):
    path: str
    kind: str
    checksum_sha256: str | None = None
    archive_path: str | None = None
    origin_date: str | None = None
    origin_date_source: str | None = None
    origin_date_confidence: float | None = None
    date_candidates: list[dict[str, Any]] = Field(default_factory=list)
    origin_date_history: list[dict[str, Any]] = Field(default_factory=list)


class Provenance(BaseModel):
    source_path: str
    source_checksum_sha256: str | None = None
    archive_path: str | None = None
    ingestion_epoch: str | None = None
    endorsement_label: str = DEFAULT_ENDORSEMENT_LABEL
    adapter: str = "mock"


class IngestedNode(BaseModel):
    node_key: str
    parent_key: str | None = None
    title: str
    text: str
    summary: str | None = None
    labels: list[str] = Field(default_factory=list)
    endorsement_label: str = DEFAULT_ENDORSEMENT_LABEL
    relations: list[dict[str, Any]] = Field(default_factory=list)
    proximity: dict[str, Any] = Field(default_factory=dict)
    usage_score: int = 0
    continuity_critical: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    source: SourceRef
    title: str
    summary: str
    tree_label: str = "source"
    nodes: list[IngestedNode]
    adapter: str = "mock"
    ingestion_kind: str = INGESTION_KIND_DETERMINISTIC
    tree_status: str = TREE_STATUS_ACTIVE
    ingestion_epoch: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    # Parser report: canonical kind, chunk strategy, dropped/normalised content.
    source_analysis: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    title: str
    summary: str
    source: SourceRef
    ingestion_epoch: str
    created_at: datetime
    updated_at: datetime


class TreeRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    document_id: Any
    label: str
    kind: str = "source_document"
    ingestion_epoch: str
    status: str = TREE_STATUS_ACTIVE
    ingestion_kind: str = INGESTION_KIND_DETERMINISTIC
    adapter: str = "mock"
    created_at: datetime
    updated_at: datetime


class NodeRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    document_id: Any
    tree_id: Any
    parent_id: Any | None = None
    node_key: str
    parent_key: str | None = None
    order: int
    title: str
    text: str
    summary: str = ""
    labels: list[str] = Field(default_factory=list)
    endorsement_label: str = DEFAULT_ENDORSEMENT_LABEL
    relations: list[dict[str, Any]] = Field(default_factory=list)
    proximity: dict[str, Any] = Field(default_factory=dict)
    usage_score: int = 0
    continuity_critical: bool = False
    ingestion_epoch: str
    status: str = "active"
    content_sha256: str = ""
    origin_date: str | None = None
    origin_date_source: str | None = None
    origin_date_confidence: float | None = None
    provenance: Provenance
    embedding: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
