"""Regression tests for the 2026-09-10 review findings (GitHub #32-#66)."""

from types import SimpleNamespace

from bson import ObjectId

from tirzah.db.memory_store import MemoryStore
from tirzah.retrieval.queries import (
    ATLAS_MAX_NUM_CANDIDATES,
    atlas_vector_search_nodes,
    collect_vector_candidate_nodes,
    normalize_origin_bound,
    origin_filter_bounds,
)

EMBEDDING = {"model": "real", "dimensions": 2, "vector": [1.0, 0.0]}


# --- #46: origin-date bounds -------------------------------------------------


def test_normalize_origin_bound_widens_partial_dates() -> None:
    assert normalize_origin_bound("2026", bound="after") == "2026-01-01"
    assert normalize_origin_bound("2026", bound="before") == "2026-12-31"
    assert normalize_origin_bound("2024-02", bound="before") == "2024-02-29"
    assert normalize_origin_bound("2024-02", bound="after") == "2024-02-01"
    assert normalize_origin_bound("2026-03-05", bound="before") == "2026-03-05"
    assert normalize_origin_bound("2026-13", bound="after") is None
    assert normalize_origin_bound("bogus", bound="after") is None
    assert normalize_origin_bound("", bound="after") is None


def test_origin_filter_bounds_reports_ignored_bounds() -> None:
    after, before, ignored = origin_filter_bounds("bogus", "2026-01")
    assert after is None
    assert before == "2026-01-31"
    assert [(row["field"], row["value"], row["reason"]) for row in ignored] == [
        ("origin_after", "bogus", "unparseable_date")
    ]
    assert origin_filter_bounds(None, None) == (None, None, [])


def test_planner_search_nodes_rejects_malformed_origin_date(monkeypatch) -> None:
    import tirzah.sessions.interaction as ix

    seen: dict = {}

    def fake_search_tool(db, **kwargs):
        seen.update(kwargs)
        return {"matches": []}, {}

    monkeypatch.setattr(ix, "execute_search_nodes_tool", fake_search_tool)

    bad = ix.execute_tool_calls(
        None, [{"tool": "search_nodes", "arguments": {"query": "x", "origin_after": "bogus"}}]
    )
    assert bad[0]["ok"] is False
    assert "origin_after" in bad[0]["error"]
    assert "repair_instruction" in bad[0]
    assert seen == {}

    good = ix.execute_tool_calls(
        None, [{"tool": "search_nodes", "arguments": {"query": "x", "origin_before": "2026"}}]
    )
    assert good[0]["ok"] is True
    assert seen["origin_before"] == "2026-12-31"
    assert seen["origin_after"] is None


# --- #43: Atlas $vectorSearch scoping ----------------------------------------


class _AtlasStore(MemoryStore):
    """Atlas returns every row; Mongo `find` only matches `in_scope` ids."""

    def __init__(self, rows, in_scope, *, reject_filter: bool = False):
        super().__init__(db=SimpleNamespace(nodes=SimpleNamespace(aggregate=self._aggregate)))
        self.rows = rows
        self.in_scope = set(in_scope)
        self.reject_filter = reject_filter
        self.stages: list[dict] = []
        self.find_filters: list[dict] = []

    def _aggregate(self, pipeline):
        stage = pipeline[0]["$vectorSearch"]
        self.stages.append(stage)
        if self.reject_filter and "filter" in stage:
            raise RuntimeError("Path 'document_id' needs to be indexed as filter")
        return list(self.rows)

    def find_nodes(self, filters, *, sort=None, limit=None):
        self.find_filters.append(filters)
        wanted = filters["_id"]["$in"]
        return [row for row in self.rows if row["_id"] in wanted and row["_id"] in self.in_scope]


def _vector_node(text: str) -> dict:
    return {"_id": ObjectId(), "text": text, "labels": ["source_chunk"], "embedding": dict(EMBEDDING)}


def test_atlas_candidates_are_rescoped_to_query_filters() -> None:
    in_scope = _vector_node("in scope")
    out_of_scope = _vector_node("OUT OF SCOPE 1999 other-doc")
    store = _AtlasStore([in_scope, out_of_scope], in_scope=[in_scope["_id"]])
    document_id = ObjectId()
    filters = {"document_id": document_id, "origin_date": {"$gte": "2026-01-01"}}

    pool = collect_vector_candidate_nodes(store, EMBEDDING, filters=filters, limit=10, index_name="vec")

    assert [node["text"] for node in pool] == ["in scope"]
    assert store.stages[0]["filter"] == filters
    assert store.find_filters[0]["document_id"] == document_id
    assert store.find_filters[0]["origin_date"] == {"$gte": "2026-01-01"}


def test_atlas_retries_unfiltered_when_filter_is_rejected() -> None:
    node = _vector_node("in scope")
    store = _AtlasStore([node], in_scope=[node["_id"]], reject_filter=True)

    pool = collect_vector_candidate_nodes(
        store, EMBEDDING, filters={"document_id": ObjectId()}, limit=10, index_name="vec"
    )

    assert ["filter" in stage for stage in store.stages] == [True, False]
    assert [row["_id"] for row in pool] == [node["_id"]]


def test_atlas_num_candidates_is_clamped() -> None:
    store = _AtlasStore([], in_scope=[])
    atlas_vector_search_nodes(store, EMBEDDING, index_name="vec", limit=10_000)
    stage = store.stages[0]
    assert stage["numCandidates"] == ATLAS_MAX_NUM_CANDIDATES
    assert stage["limit"] <= stage["numCandidates"]


# --- #66: web limits ---------------------------------------------------------


def test_web_bounded_limit_clamps_both_ends() -> None:
    from tirzah.web.app import _bounded_limit

    assert _bounded_limit(1_000_000, maximum=100) == 100
    assert _bounded_limit(0, maximum=100) == 1
    assert _bounded_limit(-1, maximum=100) == 1
    assert _bounded_limit(10, maximum=100) == 10


# --- #36 / #37 / #48: ingestion parsers and activity report -------------------


def test_html_pre_whitespace_is_preserved() -> None:
    from tirzah.ingestion.formats import parse_html_sections

    code = "def main():\n    x = 1\n\n    return x"
    [section] = parse_html_sections(f"<h2>Code</h2><pre>{code}</pre>", "fb")
    assert section["text"] == code
    assert section["paragraphs"] == [code]


def test_html_heading_keeps_marked_up_text() -> None:
    from tirzah.ingestion.formats import parse_html_sections

    [section] = parse_html_sections("<h1>Install <em>Tirzah</em> now</h1><p>Body para.</p>", "fb")
    assert section["title"] == "Install Tirzah now"
    assert section["text"] == "Body para."


def test_html_inline_markup_and_containers_do_not_merge_or_split_words() -> None:
    from tirzah.ingestion.formats import parse_html_sections

    [section] = parse_html_sections(
        "<h1>T</h1><p>foo<b>bar</b> baz</p><div>hello</div><div>world</div>", "fb"
    )
    assert section["paragraphs"] == ["foobar baz", "hello", "world"]


def test_html_dropped_chrome_is_counted() -> None:
    from tirzah.ingestion.formats import parse_structure

    analysis: dict = {}
    parse_structure(
        "<nav>Home | Docs</nav><h1>T</h1><p>Body</p><footer>(c) 2026</footer><script>x()</script>",
        "html",
        "fb",
        analysis=analysis,
    )
    assert analysis["canonical_kind"] == "html"
    assert analysis["chunk_strategy"] == "html_headings"
    assert analysis["dropped_chrome_elements"] == 2
    assert analysis["dropped_chrome_chars"] == len("Home | Docs") + len("(c) 2026")
    assert analysis["skipped_script_style_chars"] == len("x()")


def test_csv_keeps_cells_past_the_header_count() -> None:
    from tirzah.ingestion.formats import parse_structure

    analysis: dict = {}
    [section] = parse_structure("a,b\n1,2,3,EXTRA\n4\n", "csv", "fb", analysis=analysis)
    assert section["paragraphs"] == ["a: 1; b: 2; column_3: 3; column_4: EXTRA", "a: 4"]
    assert analysis["csv_ragged_row_count"] == 2
    assert analysis["csv_surplus_cell_count"] == 2
    assert analysis["csv_short_row_count"] == 1


def test_csv_quotes_separator_bearing_cells_and_chunks_large_files() -> None:
    from tirzah.ingestion.formats import CSV_ROWS_PER_SECTION, parse_csv_sections

    [section] = parse_csv_sections('a,b\n"x; y",z\n', "fb")
    assert section["paragraphs"] == ['a: "x; y"; b: z']

    rows = "\n".join(f"{index},v" for index in range(CSV_ROWS_PER_SECTION * 2 + 1))
    sections = parse_csv_sections(f"n,v\n{rows}\n", "fb")
    assert len(sections) == 3
    assert sections[0]["title"] == f"fb rows 1-{CSV_ROWS_PER_SECTION}"
    assert sum(len(item["paragraphs"]) for item in sections) == CSV_ROWS_PER_SECTION * 2 + 1


def test_ingestion_activity_report_carries_source_analysis(tmp_path) -> None:
    from tirzah.adapters.mock import MockIngestionAdapter
    from tirzah.ingestion.activity import ingestion_activity_log, ingestion_activity_report

    path = tmp_path / "page.html"
    text = "<nav>menu</nav><h1>T</h1><pre>a\n  b</pre>"
    result = MockIngestionAdapter().process(path, text, "html")
    report = ingestion_activity_report(path=path, status="committed", result=result)

    assert report["source_analysis"]["chunk_strategy"] == "html_headings"
    assert report["source_analysis"]["dropped_chrome_elements"] == 1
    log = ingestion_activity_log(report)
    assert "parsed as html with the html_headings chunk strategy" in log
    assert "1 nav/footer element(s)" in log
    assert "Preformatted blocks kept verbatim: 1." in log
