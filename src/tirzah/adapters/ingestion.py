from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tirzah.adapters.llm_ingestion import LlmIngestionAdapter
from tirzah.adapters.mock import MockIngestionAdapter
from tirzah.config import RuntimeConfig
from tirzah.models.ingestion import IngestionResult


class IngestionAdapter(Protocol):
    def process(
        self,
        path: Path,
        text: str,
        source_kind: str,
        extra_labels: list[str] | None = None,
    ) -> IngestionResult:
        ...


def ingestion_adapter(config: RuntimeConfig | None = None) -> IngestionAdapter:
    name = getattr(config, "ingestion_adapter", "mock") if config else "mock"
    if name == "mock":
        return MockIngestionAdapter()
    if name in {"llm", "llm_assisted"}:
        if config is None:
            raise ValueError("LLM ingestion adapter requires runtime config.")
        return LlmIngestionAdapter(config)
    raise ValueError(f"Unknown ingestion adapter: {name}")
