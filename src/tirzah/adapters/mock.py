from __future__ import annotations

from pathlib import Path

from tirzah.ingestion.formats import (
    canonical_kind,
    chunk_strategy_for,
    format_labels_for,
    parse_structure,
    strip_front_matter,
)
from tirzah.models.ingestion import IngestedNode, IngestionResult, SourceRef


class MockIngestionAdapter:
    def process(
        self,
        path: Path,
        text: str,
        source_kind: str,
        extra_labels: list[str] | None = None,
    ) -> IngestionResult:
        kind = canonical_kind(source_kind)
        title = first_heading(strip_front_matter(text)) or path.stem
        analysis: dict = {}
        sections = parse_structure(text, kind, title, analysis=analysis)
        if kind == "html" and sections:
            title = sections[0]["title"] or title
        labels = normalized_extra_labels([*(extra_labels or []), *format_labels_for(kind)])
        strategy = chunk_strategy_for(kind)
        outline_text = "\n\n".join(item["text"] for item in sections if item.get("text")) or text
        nodes = [with_extra_labels(root_node(title, outline_text, len(sections)), labels)]
        for section_index, section in enumerate(sections, start=1):
            section_key = f"section-{section_index}"
            nodes.append(
                with_extra_labels(
                    IngestedNode(
                        node_key=section_key,
                        parent_key="root",
                        title=section["title"],
                        text=section["text"],
                        labels=["source_section"],
                        metadata={
                            "adapter": "mock",
                            "chunk_strategy": strategy,
                            "source_kind": kind,
                            "section_index": section_index,
                        },
                    ),
                    labels,
                )
            )
            for paragraph_index, paragraph in enumerate(section["paragraphs"], start=1):
                nodes.append(
                    with_extra_labels(
                        IngestedNode(
                            node_key=f"{section_key}-paragraph-{paragraph_index}",
                            parent_key=section_key,
                            title=f"{section['title']} / paragraph {paragraph_index}",
                            text=paragraph,
                            labels=["source_chunk"],
                            metadata={
                                "adapter": "mock",
                                "chunk_strategy": f"{strategy}_chunks",
                                "source_kind": kind,
                                "section_index": section_index,
                                "paragraph_index": paragraph_index,
                            },
                        ),
                        labels,
                    )
                )
        return IngestionResult(
            source=SourceRef(path=str(path), kind=source_kind),
            title=title,
            summary=summarize(text),
            nodes=nodes,
            adapter="mock",
            source_analysis=analysis,
        )


def first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def root_node(title: str, text: str, section_count: int) -> IngestedNode:
    return IngestedNode(
        node_key="root",
        title=title,
        text=summarize(text),
        labels=["source_root"],
        metadata={
            "adapter": "mock",
            "chunk_strategy": "document_root",
            "section_count": section_count,
        },
    )


def normalized_extra_labels(labels: list[str] | None) -> list[str]:
    normalized = []
    for label in labels or []:
        cleaned = "_".join(label.strip().lower().split())
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def with_extra_labels(node: IngestedNode, extra_labels: list[str]) -> IngestedNode:
    for label in extra_labels:
        if label not in node.labels:
            node.labels.append(label)
    return node


def parse_sections(text: str, fallback_title: str) -> list[dict]:
    return parse_structure(text, "markdown", fallback_title)


def summarize(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    return compact[:limit]
