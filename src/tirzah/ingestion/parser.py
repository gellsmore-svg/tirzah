from __future__ import annotations

from pathlib import Path

from tirzah.ingestion.formats import KIND_BY_SUFFIX

SUPPORTED_SUFFIXES = set(KIND_BY_SUFFIX)


def read_text_source(path: Path | str) -> tuple[str, str]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported source type: {suffix or '<none>'}")
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read(), suffix.lstrip(".")
