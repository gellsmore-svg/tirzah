from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tirzah.models.ingestion import IngestionResult

ORIGIN_DATE_CONFIDENCE = {
    "operator": 1.0,
    "explicit_content": 0.9,
    "filename": 0.7,
    "file_created": 0.4,
    "file_modified": 0.3,
}


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

EXPLICIT_DATE_PATTERNS = [
    re.compile(
        r"(?im)^\s*(?:publication\s+date|published|created|written|date)\s*[:\-]\s*"
        r"(?P<date>\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4})\b"
    ),
    re.compile(
        r"(?i)\b(?:publication\s+date|published|created|written|date)\s*[:\-]\s*"
        r"(?P<date>\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4})\b"
    ),
]

FILENAME_DATE_PATTERNS = [
    re.compile(r"(?P<year>\d{4})[-_](?P<month>\d{1,2})[-_](?P<day>\d{1,2})"),
    re.compile(r"(?P<year>\d{4})[-_](?P<month>\d{1,2})\b"),
    re.compile(r"\b(?P<year>\d{4})\b"),
]


def annotate_source_dates(result: IngestionResult, path: Path, text: str) -> IngestionResult:
    analysis = analyze_source_dates(path, text)
    result.source.origin_date = analysis.get("origin_date")
    result.source.origin_date_source = analysis.get("origin_date_source")
    result.source.origin_date_confidence = analysis.get("origin_date_confidence")
    result.source.date_candidates = analysis.get("date_candidates") or []
    return result


def analyze_source_dates(path: Path, text: str) -> dict[str, Any]:
    candidates = [
        *explicit_date_candidates(text),
        *filename_date_candidates(path),
        *filesystem_date_candidates(path),
    ]
    selected = selected_origin_date(candidates)
    origin_source = selected.get("source") if selected else None
    confidence = origin_date_confidence_for(origin_source, raw=selected.get("raw") if selected else None)
    return {
        "origin_date": selected.get("date") if selected else None,
        "origin_date_source": origin_source,
        "origin_date_confidence": confidence,
        "date_candidates": candidates,
    }


def origin_date_confidence_for(source: str | None, *, raw: str | None = None) -> float | None:
    if not source:
        return None
    confidence = ORIGIN_DATE_CONFIDENCE.get(source)
    if confidence is None:
        return None
    if raw and raw.strip().isdigit() and len(raw.strip()) == 4:
        return round(min(confidence, 0.6), 3)
    return confidence


def selected_origin_date(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for source in ("explicit_content", "filename", "file_created", "file_modified"):
        matching = [candidate for candidate in candidates if candidate.get("source") == source]
        if matching:
            return min(matching, key=lambda candidate: candidate["date"])
    return None


def explicit_date_candidates(text: str) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    for pattern in EXPLICIT_DATE_PATTERNS:
        for match in pattern.finditer(text[:20000]):
            parsed = parse_human_date(match.group("date"))
            if not parsed:
                continue
            key = ("explicit_content", parsed.isoformat(), match.group("date"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "source": "explicit_content",
                    "date": parsed.isoformat(),
                    "raw": match.group("date"),
                    "rationale": "Explicit date marker found in document content.",
                }
            )
    return candidates


def filename_date_candidates(path: Path) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    name = path.name
    for pattern in FILENAME_DATE_PATTERNS:
        for match in pattern.finditer(name):
            year = int(match.group("year"))
            month = int(match.groupdict().get("month") or 1)
            day = int(match.groupdict().get("day") or 1)
            parsed = safe_date(year, month, day)
            if not parsed:
                continue
            key = parsed.isoformat()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "source": "filename",
                    "date": parsed.isoformat(),
                    "raw": match.group(0),
                    "rationale": "Date-like value found in filename.",
                }
            )
    return candidates


def filesystem_date_candidates(path: Path) -> list[dict[str, Any]]:
    try:
        stat = path.stat()
    except OSError:
        return []
    return [
        {
            "source": "file_created",
            "date": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).date().isoformat(),
            "raw": str(stat.st_ctime),
            "rationale": "Filesystem creation/change timestamp.",
        },
        {
            "source": "file_modified",
            "date": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).date().isoformat(),
            "raw": str(stat.st_mtime),
            "rationale": "Filesystem modification timestamp.",
        },
    ]


def parse_human_date(value: str) -> date | None:
    cleaned = value.strip().replace(",", "")
    iso_match = re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})", cleaned)
    if iso_match:
        return safe_date(
            int(iso_match.group("year")),
            int(iso_match.group("month")),
            int(iso_match.group("day")),
        )
    day_month_year = re.fullmatch(
        r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})",
        cleaned,
    )
    if day_month_year:
        return safe_date(
            int(day_month_year.group("year")),
            MONTHS.get(day_month_year.group("month").lower(), 0),
            int(day_month_year.group("day")),
        )
    month_day_year = re.fullmatch(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})",
        cleaned,
    )
    if month_day_year:
        return safe_date(
            int(month_day_year.group("year")),
            MONTHS.get(month_day_year.group("month").lower(), 0),
            int(month_day_year.group("day")),
        )
    year_only = re.fullmatch(r"(?P<year>\d{4})", cleaned)
    if year_only:
        return safe_date(int(year_only.group("year")), 1, 1)
    return None


def safe_date(year: int, month: int, day: int) -> date | None:
    if year < 1000 or year > 3000:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None
