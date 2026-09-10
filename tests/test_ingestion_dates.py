from __future__ import annotations

import os
from datetime import datetime, timezone

from tirzah.ingestion.dates import (
    analyze_source_dates,
    explicit_date_candidates,
    filename_date_candidates,
    origin_date_confidence_for,
    parse_human_date,
)


def test_origin_date_confidence_scales_by_source_and_year_only() -> None:
    assert origin_date_confidence_for("operator") == 1.0
    assert origin_date_confidence_for("explicit_content", raw="2020-05-17") == 0.9
    assert origin_date_confidence_for("explicit_content", raw="2020") == 0.6
    assert origin_date_confidence_for("filename") == 0.7


def test_parse_human_date_accepts_supported_formats() -> None:
    assert parse_human_date("2020-02-03").isoformat() == "2020-02-03"
    assert parse_human_date("3 February 2020").isoformat() == "2020-02-03"
    assert parse_human_date("February 3, 2020").isoformat() == "2020-02-03"
    assert parse_human_date("2020").isoformat() == "2020-01-01"


def test_explicit_date_candidates_find_marked_content_dates() -> None:
    candidates = explicit_date_candidates(
        "# Source\n\nPublication date: 2020-05-17\n\nFilename says 2026."
    )

    assert candidates[0]["source"] == "explicit_content"
    assert candidates[0]["date"] == "2020-05-17"
    assert candidates[0]["raw"] == "2020-05-17"


def test_filename_date_candidates_support_year_month_day_and_year_only(tmp_path) -> None:
    dated = tmp_path / "rs5-clause-2024-03-09.md"
    year_only = tmp_path / "notes-1999.txt"

    assert filename_date_candidates(dated)[0]["date"] == "2024-03-09"
    assert filename_date_candidates(year_only)[0]["date"] == "1999-01-01"


def test_analyze_source_dates_prioritizes_explicit_content_over_filename(tmp_path) -> None:
    path = tmp_path / "source-2026-01-01.md"
    path.write_text("Date: 2020\n\nContent.", encoding="utf-8")

    analysis = analyze_source_dates(path, path.read_text(encoding="utf-8"))

    assert analysis["origin_date"] == "2020-01-01"
    assert analysis["origin_date_source"] == "explicit_content"
    assert analysis["origin_date_confidence"] == 0.6
    assert [candidate["source"] for candidate in analysis["date_candidates"][:2]] == [
        "explicit_content",
        "filename",
    ]


def test_analyze_source_dates_uses_filesystem_modified_date_when_needed(tmp_path) -> None:
    path = tmp_path / "undated-notes.md"
    path.write_text("No date markers here.", encoding="utf-8")
    timestamp = datetime(2021, 4, 5, tzinfo=timezone.utc).timestamp()
    os.utime(path, (timestamp, timestamp))

    analysis = analyze_source_dates(path, path.read_text(encoding="utf-8"))

    assert analysis["origin_date_source"] in {"file_created", "file_modified"}
    assert any(
        candidate["source"] == "file_modified" and candidate["date"] == "2021-04-05"
        for candidate in analysis["date_candidates"]
    )
