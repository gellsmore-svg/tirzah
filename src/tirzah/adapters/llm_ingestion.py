from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tirzah.adapters.mock import MockIngestionAdapter, normalized_extra_labels, with_extra_labels
from tirzah.config import RuntimeConfig
from tirzah.models.ingestion import (
    INGESTION_KIND_DETERMINISTIC,
    INGESTION_KIND_LLM_PROPOSED,
    TREE_STATUS_ACTIVE,
    TREE_STATUS_PENDING_REVIEW,
    IngestedNode,
    IngestionResult,
    SourceRef,
)

MAX_SECTIONS = 40
MAX_CHUNKS_PER_SECTION = 30
MIN_VALID_CHUNKS = 1


class LlmIngestionAdapter:
    """Review-gated LLM chunking. Mock stays the default factory selection."""

    name = "llm"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def process(
        self,
        path: Path,
        text: str,
        source_kind: str,
        extra_labels: list[str] | None = None,
    ) -> IngestionResult:
        labels = normalized_extra_labels(extra_labels)
        mock_result = MockIngestionAdapter().process(path, text, source_kind, extra_labels)
        model_adapter = resolved_ingestion_model_adapter(self.config)
        model = resolved_ingestion_model(self.config, model_adapter)
        try:
            payload = propose_ingestion_tree(self.config, text, path=path)
            result = result_from_proposal(
                path=path,
                text=text,
                source_kind=source_kind,
                payload=payload,
                extra_labels=labels,
                model_adapter=model_adapter,
                model=model,
            )
        except Exception as error:
            if not self.config.ingestion_fallback_to_mock:
                raise
            result = mock_fallback_result(
                mock_result,
                extra_labels=labels,
                model_adapter=model_adapter,
                model=model,
                error=error,
            )
        return result


def resolved_ingestion_model_adapter(config: RuntimeConfig) -> str:
    name = (config.ingestion_model_adapter or "").strip()
    if name:
        return name
    from tirzah.adapters.cli_runtime import GENERATOR_ADAPTERS

    if config.answer_adapter in GENERATOR_ADAPTERS:
        return config.answer_adapter
    return "ollama_http"


def resolved_ingestion_model(config: RuntimeConfig, model_adapter: str) -> str:
    from tirzah.adapters.cli_runtime import EXTERNAL_MODEL_FIELDS

    if config.ingestion_model:
        return config.ingestion_model
    field = EXTERNAL_MODEL_FIELDS.get(model_adapter)
    if field:
        return getattr(config, field, None) or model_adapter.replace("_cli", "")
    return config.ollama_model


def propose_ingestion_tree(config: RuntimeConfig, text: str, *, path: Path) -> dict[str, Any]:
    from tirzah.adapters.answer import generate_text

    excerpt = text
    truncated = False
    limit = config.ingestion_max_source_chars
    if limit and len(text) > limit:
        excerpt = text[:limit]
        truncated = True
    prompt = build_ingestion_prompt(excerpt, path=path, truncated=truncated)
    model_adapter = resolved_ingestion_model_adapter(config)
    answer = generate_text(
        config,
        prompt,
        adapter_name=model_adapter,
        model=config.ingestion_model or None,
    )
    raw = answer.get("answer") if isinstance(answer, dict) else str(answer)
    payload = extract_json_object(raw or "")
    if not isinstance(payload, dict):
        raise ValueError("LLM ingestion adapter did not return a JSON object.")
    return payload


def build_ingestion_prompt(text: str, *, path: Path, truncated: bool) -> str:
    truncation_note = (
        " The source was truncated for the prompt; only propose chunks from this excerpt."
        if truncated
        else ""
    )
    return (
        "Propose a hierarchical memory tree for this source document. "
        "Copy chunk text VERBATIM from the source; never paraphrase. "
        "Use headings and natural section boundaries. "
        'Return ONLY JSON of the form '
        '{"title": "...", "summary": "...", "sections": ['
        '{"title": "...", "summary": "...", "chunks": [{"title": "...", "text": "..."}]}'
        "]}."
        f"{truncation_note}\n\n"
        f"Source path: {path}\n\n"
        f"{text}\n"
    )


def result_from_proposal(
    *,
    path: Path,
    text: str,
    source_kind: str,
    payload: dict[str, Any],
    extra_labels: list[str],
    model_adapter: str,
    model: str,
) -> IngestionResult:
    title = str(payload.get("title") or path.stem).strip() or path.stem
    summary = str(payload.get("summary") or "").strip() or summarize_text(text)
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("LLM ingestion proposal has no sections.")
    nodes = [
        with_extra_labels(
            IngestedNode(
                node_key="root",
                title=title,
                text=summary,
                summary=summary,
                labels=["source_root"],
                metadata={
                    "adapter": "llm",
                    "ingestion_kind": INGESTION_KIND_LLM_PROPOSED,
                    "chunk_strategy": "llm_proposed",
                    "model_adapter": model_adapter,
                    "model": model,
                    "section_count": 0,
                },
            ),
            extra_labels,
        )
    ]
    valid_chunks = 0
    section_count = 0
    for section_index, section in enumerate(sections[:MAX_SECTIONS], start=1):
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or f"Section {section_index}").strip()
        raw_chunks = section.get("chunks")
        if not isinstance(raw_chunks, list):
            raw_chunks = []
        verified: list[dict[str, str]] = []
        for item in raw_chunks[:MAX_CHUNKS_PER_SECTION]:
            if not isinstance(item, dict):
                continue
            excerpt = str(item.get("text") or "").strip()
            located = locate_verbatim(text, excerpt)
            if not located:
                continue
            chunk_title = str(item.get("title") or "").strip() or f"{section_title} / chunk {len(verified) + 1}"
            verified.append({"title": chunk_title, "text": located})
        if not verified:
            continue
        section_count += 1
        section_key = f"section-{section_count}"
        section_text = "\n\n".join(chunk["text"] for chunk in verified)
        section_summary = str(section.get("summary") or "").strip() or summarize_text(section_text)
        nodes.append(
            with_extra_labels(
                IngestedNode(
                    node_key=section_key,
                    parent_key="root",
                    title=section_title,
                    text=section_text,
                    summary=section_summary,
                    labels=["source_section"],
                    metadata={
                        "adapter": "llm",
                        "ingestion_kind": INGESTION_KIND_LLM_PROPOSED,
                        "chunk_strategy": "llm_sections",
                        "model_adapter": model_adapter,
                        "model": model,
                        "section_index": section_count,
                    },
                ),
                extra_labels,
            )
        )
        for paragraph_index, chunk in enumerate(verified, start=1):
            valid_chunks += 1
            nodes.append(
                with_extra_labels(
                    IngestedNode(
                        node_key=f"{section_key}-chunk-{paragraph_index}",
                        parent_key=section_key,
                        title=chunk["title"],
                        text=chunk["text"],
                        labels=["source_chunk"],
                        metadata={
                            "adapter": "llm",
                            "ingestion_kind": INGESTION_KIND_LLM_PROPOSED,
                            "chunk_strategy": "llm_verbatim",
                            "model_adapter": model_adapter,
                            "model": model,
                            "section_index": section_count,
                            "paragraph_index": paragraph_index,
                        },
                    ),
                    extra_labels,
                )
            )
    if valid_chunks < MIN_VALID_CHUNKS:
        raise ValueError("LLM ingestion proposal had no verbatim source chunks.")
    nodes[0].metadata["section_count"] = section_count
    return IngestionResult(
        source=SourceRef(path=str(path), kind=source_kind),
        title=title,
        summary=summary,
        nodes=nodes,
        adapter="llm",
        ingestion_kind=INGESTION_KIND_LLM_PROPOSED,
        tree_status=TREE_STATUS_PENDING_REVIEW,
    )


def mock_fallback_result(
    mock_result: IngestionResult,
    *,
    extra_labels: list[str],
    model_adapter: str,
    model: str,
    error: Exception,
) -> IngestionResult:
    for node in mock_result.nodes:
        with_extra_labels(node, extra_labels)
        node.metadata = {
            **(node.metadata or {}),
            "adapter": "llm",
            "ingestion_kind": INGESTION_KIND_DETERMINISTIC,
            "fallback": "mock",
            "fallback_reason": error.__class__.__name__,
            "model_adapter": model_adapter,
            "model": model,
        }
    mock_result.adapter = "llm"
    mock_result.ingestion_kind = INGESTION_KIND_DETERMINISTIC
    mock_result.tree_status = TREE_STATUS_ACTIVE
    return mock_result


def extract_json_object(text: str) -> Any:
    if not isinstance(text, str):
        return None
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = fenced.split("\n", 1)[-1]
        if fenced.endswith("```"):
            fenced = fenced[: -3]
        text = fenced
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def locate_verbatim(source: str, excerpt: str) -> str | None:
    if not excerpt or not excerpt.strip():
        return None
    if excerpt in source:
        return excerpt
    compact_excerpt = " ".join(excerpt.split())
    if not compact_excerpt:
        return None
    compact_source = " ".join(source.split())
    index = compact_source.find(compact_excerpt)
    if index == -1:
        return None
    return recover_original_span(source, compact_excerpt)


def recover_original_span(source: str, compact_excerpt: str) -> str | None:
    source_tokens = source.split()
    excerpt_tokens = compact_excerpt.split()
    if not excerpt_tokens or len(excerpt_tokens) > len(source_tokens):
        return compact_excerpt
    needle_len = len(excerpt_tokens)
    for start in range(0, len(source_tokens) - needle_len + 1):
        window = source_tokens[start : start + needle_len]
        if window == excerpt_tokens:
            return " ".join(window)
    return compact_excerpt


def summarize_text(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    return compact[:limit]
