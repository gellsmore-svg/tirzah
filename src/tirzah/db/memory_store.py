from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo.database import Database

from tirzah.db.schema import collection_available
from tirzah.models.ingestion import INACTIVE_RETRIEVAL_STATUSES


class MemoryStore:
    """Mongo-backed persistence facade for Tirzah memory operations."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def list_documents(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return list(self.db.documents.find({}).sort("created_at", -1).limit(limit))

    def get_document(self, document_id: object) -> dict[str, Any] | None:
        return self.db.documents.find_one({"_id": document_id})

    def find_documents(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self.db.documents.find(filters))

    def find_documents_by_ids(self, document_ids: list[object]) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        return self.find_documents({"_id": {"$in": document_ids}})

    def active_tree_count(self, document_id: object) -> int:
        return self.db.trees.count_documents(
            {"document_id": document_id, "status": {"$nin": list(INACTIVE_RETRIEVAL_STATUSES)}}
        )

    def active_node_count(self, document_id: object) -> int:
        return self.db.nodes.count_documents(
            {"document_id": document_id, "status": {"$nin": list(INACTIVE_RETRIEVAL_STATUSES)}}
        )

    def get_node(self, node_id: object) -> dict[str, Any] | None:
        return self.db.nodes.find_one({"_id": node_id})

    def find_nodes(
        self,
        filters: dict[str, Any],
        *,
        sort: tuple[str, int] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        cursor = self.db.nodes.find(filters)
        if sort:
            cursor = cursor.sort(*sort)
        if limit is not None:
            cursor = cursor.limit(limit)
        return list(cursor)

    def child_nodes(self, parent_id: object, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.find_nodes({"parent_id": parent_id}, sort=("order", 1), limit=limit)

    def update_node(self, node_id: object, update: dict[str, Any]) -> int:
        result = self.db.nodes.update_one({"_id": node_id}, update)
        return int(getattr(result, "modified_count", 0) or 0)

    def update_nodes(self, filters: dict[str, Any], update: dict[str, Any]) -> int:
        if not hasattr(self.db.nodes, "update_many"):
            return 0
        result = self.db.nodes.update_many(filters, update)
        return int(getattr(result, "modified_count", 0) or 0)

    def graph_edges(
        self,
        filters: dict[str, Any],
        *,
        limit: int = 10,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        if not collection_available(self.db, "graph_edges"):
            return []
        sort_order = -1 if newest_first else 1
        return list(self.db.graph_edges.find(filters).sort("created_at", sort_order).limit(limit))

    def update_chunk_metadata(
        self,
        node_id: object,
        metadata: dict[str, Any],
    ) -> int:
        return self.update_node(
            node_id,
            {"$set": {"metadata": metadata, "updated_at": datetime.now(timezone.utc)}},
        )

    def record_retrieval_trace(self, trace: dict[str, Any]) -> str:
        row = {
            **trace,
            "created_at": trace.get("created_at") or datetime.now(timezone.utc),
        }
        result = self.db.retrieval_traces.insert_one(row)
        return str(result.inserted_id)


def as_memory_store(value: Database | MemoryStore) -> MemoryStore:
    if isinstance(value, MemoryStore):
        return value
    return MemoryStore(value)
