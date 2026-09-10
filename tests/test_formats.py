from pathlib import Path

from tirzah.adapters.mock import MockIngestionAdapter
from tirzah.ingestion.formats import parse_structure
from tirzah.ingestion.parser import SUPPORTED_SUFFIXES, read_text_source


def test_supported_suffixes_include_html_code_and_structured_formats() -> None:
    assert {".html", ".py", ".js", ".json", ".yaml", ".csv"}.issubset(SUPPORTED_SUFFIXES)


def test_read_text_source_preserves_html_bytes(tmp_path: Path) -> None:
    source = tmp_path / "note.html"
    raw = "<html><body><h1>Hi</h1><p>There</p></body></html>"
    source.write_text(raw, encoding="utf-8")
    text, kind = read_text_source(source)
    assert kind == "html"
    assert text == raw


def test_html_parser_uses_headings_and_strips_chrome() -> None:
    html = """
    <html><head><title>Export</title></head>
    <body>
      <nav>Skip this chrome</nav>
      <h1>Vortons</h1>
      <p>A vorton is a closed loop.</p>
      <script>alert('nope')</script>
      <h2>Charge</h2>
      <p>Its charge is an integer.</p>
    </body></html>
    """
    result = MockIngestionAdapter().process(Path("page.html"), html, "html")
    titles = [node.title for node in result.nodes if "source_section" in node.labels]
    texts = " ".join(node.text for node in result.nodes)
    assert "Vortons" in titles
    assert "Charge" in titles
    assert "html_export" in result.nodes[0].labels
    assert "Skip this chrome" not in texts
    assert "alert" not in texts
    assert "closed loop" in texts


def test_python_parser_splits_on_functions_and_classes() -> None:
    source = '''"""Module."""
import os

def helper():
    return os.name

class Thing:
    def run(self):
        return helper()
'''
    result = MockIngestionAdapter().process(Path("mod.py"), source, "py")
    section_titles = [node.title for node in result.nodes if "source_section" in node.labels]
    assert "code_python" in result.nodes[0].labels
    assert any("helper" in title for title in section_titles)
    assert any("Thing" in title for title in section_titles)
    chunks = [node.text for node in result.nodes if "source_chunk" in node.labels]
    assert any("def helper" in chunk for chunk in chunks)


def test_json_parser_uses_top_level_keys() -> None:
    text = '{"title": "Memory", "body": "A vorton is a closed loop."}'
    sections = parse_structure(text, "json", "doc")
    assert [item["title"] for item in sections] == ["title", "body"]
    result = MockIngestionAdapter().process(Path("note.json"), text, "json")
    assert "structured_json" in result.nodes[0].labels


def test_csv_parser_emits_row_chunks() -> None:
    text = "name,role\nAda,analyst\nGrace,operator\n"
    result = MockIngestionAdapter().process(Path("people.csv"), text, "csv")
    chunks = [node.text for node in result.nodes if "source_chunk" in node.labels]
    assert any("Ada" in chunk and "analyst" in chunk for chunk in chunks)
    assert "structured_csv" in result.nodes[0].labels


def test_markdown_front_matter_is_stripped_from_structure() -> None:
    text = "---\ntitle: hidden\n---\n# Visible\n\nBody paragraph.\n"
    result = MockIngestionAdapter().process(Path("note.md"), text, "md")
    assert result.title == "Visible"
    assert "hidden" not in result.nodes[1].text
    assert result.nodes[2].text == "Body paragraph."
