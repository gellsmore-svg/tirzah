from __future__ import annotations

import ast
import csv
import json
import re
from html.parser import HTMLParser
from io import StringIO
from typing import Any

KIND_BY_SUFFIX = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".html": "html",
    ".htm": "html",
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".csv": "csv",
}

FORMAT_LABELS = {
    "html": ["html_export"],
    "python": ["code_python"],
    "javascript": ["code_javascript"],
    "typescript": ["code_typescript"],
    "rust": ["code_rust"],
    "go": ["code_go"],
    "java": ["code_java"],
    "ruby": ["code_ruby"],
    "json": ["structured_json"],
    "yaml": ["structured_yaml"],
    "csv": ["structured_csv"],
}

CHUNK_STRATEGY = {
    "markdown": "markdown_sections",
    "text": "markdown_sections",
    "html": "html_headings",
    "python": "python_ast",
    "javascript": "code_declarations",
    "typescript": "code_declarations",
    "rust": "code_declarations",
    "go": "code_declarations",
    "java": "code_declarations",
    "ruby": "code_declarations",
    "json": "json_keys",
    "yaml": "yaml_keys",
    "csv": "csv_rows",
}

_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def canonical_kind(source_kind: str | None) -> str:
    kind = (source_kind or "").lower().lstrip(".")
    mapped = KIND_BY_SUFFIX.get(f".{kind}")
    return mapped or kind or "text"


def format_labels_for(kind: str) -> list[str]:
    return list(FORMAT_LABELS.get(kind, []))


def chunk_strategy_for(kind: str) -> str:
    return CHUNK_STRATEGY.get(kind, "markdown_sections")


def parse_structure(
    text: str,
    source_kind: str,
    fallback_title: str,
    analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Parse ``text`` into sections. When ``analysis`` is given it is filled with
    the canonical kind, chunk strategy and parser, plus what the parser dropped
    or normalised, for the ingestion activity report."""
    kind = canonical_kind(source_kind)
    parsers = {
        "markdown": parse_markdown_sections,
        "text": parse_markdown_sections,
        "html": parse_html_sections,
        "python": parse_python_sections,
        "javascript": parse_code_sections,
        "typescript": parse_code_sections,
        "rust": parse_code_sections,
        "go": parse_code_sections,
        "java": parse_code_sections,
        "ruby": parse_code_sections,
        "json": parse_json_sections,
        "yaml": parse_yaml_sections,
        "csv": parse_csv_sections,
    }
    parser = parsers.get(kind, parse_markdown_sections)
    stats: dict[str, Any] = {}
    if parser in (parse_markdown_sections, parse_html_sections, parse_csv_sections):
        sections = parser(text, fallback_title, stats=stats)
    else:
        sections = parser(text, fallback_title)
    if analysis is not None:
        analysis.update(
            {
                "canonical_kind": kind,
                "chunk_strategy": chunk_strategy_for(kind),
                "parser": parser.__name__,
                **stats,
            }
        )
    return sections or [section(fallback_title, text)]


def parse_markdown_sections(
    text: str, fallback_title: str, stats: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    body = strip_front_matter(text)
    if stats is not None and len(body) != len(text):
        stats["front_matter_stripped_chars"] = len(text) - len(body)
    sections: list[dict[str, Any]] = []
    current_title = fallback_title
    current_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append(section(current_title, "\n".join(current_lines)))
            current_title = stripped.lstrip("#").strip() or fallback_title
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append(section(current_title, "\n".join(current_lines)))
    if not sections:
        sections.append(section(fallback_title, body))
    return sections


def strip_front_matter(text: str) -> str:
    return _FRONT_MATTER.sub("", text, count=1)


def parse_html_sections(
    text: str, fallback_title: str, stats: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    parser = HtmlOutlineParser(fallback_title)
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        if stats is not None:
            stats["parse_fallback"] = "html_tags_stripped"
        return [section(fallback_title, strip_tags(text))]
    sections = parser.sections()
    if stats is not None:
        stats.update(parser.stats)
    return sections or [section(parser.document_title or fallback_title, parser.visible_text() or strip_tags(text))]


class HtmlOutlineParser(HTMLParser):
    """Outline an HTML page into heading-led sections of paragraph blocks.

    Text is accumulated raw per block and whitespace-collapsed only when the
    section is flushed, except inside ``<pre>``, which is kept verbatim.
    Headings gather all their inline text until the closing tag. Skipped
    script/style and nav/footer content is counted in ``stats``.
    """

    SKIP = {"script", "style", "noscript"}
    CHROME = {"nav", "footer"}
    HEADINGS = {f"h{index}" for index in range(1, 7)}
    BLOCKS = {"p", "li", "pre", "blockquote", "dt", "dd"}
    # Structural containers: they end the current block so adjacent text in
    # sibling containers is not run together, but open no block themselves.
    BOUNDARIES = {
        "div", "section", "article", "main", "header", "aside", "body",
        "table", "tr", "td", "th", "ul", "ol", "dl", "figure", "figcaption", "hr",
    }

    def __init__(self, fallback_title: str) -> None:
        super().__init__(convert_charrefs=True)
        self.fallback_title = fallback_title
        self.document_title = ""
        self._skip_depth = 0
        self._chrome_depth = 0
        self._pre_depth = 0
        self._in_title = False
        self._heading_parts: list[str] | None = None
        self._current_title = fallback_title
        self._blocks: list[str] = []
        self._block_is_pre: list[bool] = []
        self._block_open = False
        self._sections: list[dict[str, Any]] = []
        self.stats: dict[str, int] = {
            "dropped_chrome_elements": 0,
            "dropped_chrome_chars": 0,
            "skipped_script_style_chars": 0,
            "preserved_pre_blocks": 0,
        }

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag in self.CHROME:
            if not self._chrome_depth:
                self.stats["dropped_chrome_elements"] += 1
            self._chrome_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if self._skip_depth or self._chrome_depth:
            return
        if tag in self.HEADINGS:
            self._flush_section()
            self._heading_parts = []
            return
        if tag == "br":
            self._append("\n" if self._pre_depth else " ")
            return
        if tag in self.BLOCKS:
            self._close_heading()
            if tag == "pre":
                self._pre_depth += 1
                if self._pre_depth > 1:
                    return
            self._start_block(is_pre=self._pre_depth > 0)
            return
        if tag in self.BOUNDARIES and not self._pre_depth:
            self._block_open = False

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.CHROME and self._chrome_depth:
            self._chrome_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif self._skip_depth or self._chrome_depth:
            return
        elif tag in self.HEADINGS:
            self._close_heading()
        elif tag in self.BLOCKS:
            if tag == "pre" and self._pre_depth:
                self._pre_depth -= 1
            if not self._pre_depth:
                self._block_open = False
        elif tag in self.BOUNDARIES and not self._pre_depth:
            self._block_open = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            self.stats["skipped_script_style_chars"] += len(data.strip())
            return
        if self._chrome_depth:
            self.stats["dropped_chrome_chars"] += len(data.strip())
            return
        if self._in_title:
            text = " ".join(data.split())
            if text and not self.document_title:
                self.document_title = text
            return
        if self._heading_parts is not None:
            self._heading_parts.append(data)
            return
        if not self._pre_depth and not data.strip():
            if self._block_open:
                self._append(" ")
            return
        if not self._block_open:
            self._start_block(is_pre=self._pre_depth > 0)
        self._append(data)

    def _start_block(self, *, is_pre: bool) -> None:
        self._blocks.append("")
        self._block_is_pre.append(is_pre)
        self._block_open = True

    def _append(self, data: str) -> None:
        if not self._blocks:
            self._start_block(is_pre=self._pre_depth > 0)
        self._blocks[-1] += data

    def _close_heading(self) -> None:
        if self._heading_parts is None:
            return
        title = " ".join("".join(self._heading_parts).split())
        if title:
            self._current_title = title
        self._heading_parts = None

    def _flush_section(self) -> None:
        self._close_heading()
        paragraphs = []
        for block, is_pre in zip(self._blocks, self._block_is_pre):
            if is_pre:
                cleaned = block.strip("\n")
                if cleaned.strip():
                    paragraphs.append(cleaned)
                    self.stats["preserved_pre_blocks"] += 1
            else:
                cleaned = " ".join(block.split())
                if cleaned:
                    paragraphs.append(cleaned)
        if paragraphs:
            self._sections.append(
                section(self._current_title, "\n\n".join(paragraphs), paragraphs=paragraphs)
            )
        self._blocks = []
        self._block_is_pre = []
        self._block_open = False

    def sections(self) -> list[dict[str, Any]]:
        self._flush_section()
        if self.document_title and self._sections and self._sections[0]["title"] == self.fallback_title:
            self._sections[0]["title"] = self.document_title
        return self._sections

    def visible_text(self) -> str:
        return "\n\n".join(item["text"] for item in self._sections)


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def parse_python_sections(text: str, fallback_title: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return parse_code_sections(text, fallback_title)
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    preamble_end = min((getattr(node, "lineno", 1) - 1 for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), default=len(lines))
    preamble = "\n".join(lines[:preamble_end]).strip()
    if preamble:
        sections.append(section("module preamble", preamble))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        snippet = ast.get_source_segment(text, node) or node.name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        sections.append(section(f"{kind} {node.name}", snippet))
    return sections or [section(fallback_title, text)]


_CODE_PATTERNS = [
    re.compile(r"(?m)^(export\s+)?(default\s+)?(async\s+)?function\s+(?P<name>[\w$]+)"),
    re.compile(r"(?m)^(export\s+)?(default\s+)?class\s+(?P<name>[\w$]+)"),
    re.compile(r"(?m)^(pub\s+)?(async\s+)?fn\s+(?P<name>\w+)"),
    re.compile(r"(?m)^impl(?:\s+[^\{]+)?\s+(?P<name>\w+)"),
    re.compile(r"(?m)^func\s+(?:\([^)]+\)\s+)?(?P<name>\w+)"),
    re.compile(r"(?m)^type\s+(?P<name>\w+)"),
    re.compile(r"(?m)^(public|private|protected)?\s*(static\s+)?(class|interface|enum)\s+(?P<name>\w+)"),
    re.compile(r"(?m)^(module|class)\s+(?P<name>\w+)"),
    re.compile(r"(?m)^def\s+(?P<name>\w+)"),
]


def parse_code_sections(text: str, fallback_title: str) -> list[dict[str, Any]]:
    starts: list[tuple[int, str]] = []
    for pattern in _CODE_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group("name")
            starts.append((match.start(), name))
    starts.sort()
    unique: list[tuple[int, str]] = []
    seen = set()
    for offset, name in starts:
        if offset in seen:
            continue
        seen.add(offset)
        unique.append((offset, name))
    if not unique:
        return [section(fallback_title, text)]
    sections: list[dict[str, Any]] = []
    if unique[0][0] > 0:
        preamble = text[: unique[0][0]].strip()
        if preamble:
            sections.append(section("module preamble", preamble))
    for index, (offset, name) in enumerate(unique):
        end = unique[index + 1][0] if index + 1 < len(unique) else len(text)
        snippet = text[offset:end].strip()
        sections.append(section(name, snippet))
    return sections


def parse_json_sections(text: str, fallback_title: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [section(fallback_title, text)]
    return _mapping_sections(payload, fallback_title)


def parse_yaml_sections(text: str, fallback_title: str) -> list[dict[str, Any]]:
    try:
        import yaml

        payload = yaml.safe_load(text)
    except Exception:
        return [section(fallback_title, text)]
    return _mapping_sections(payload, fallback_title)


def _mapping_sections(payload: Any, fallback_title: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        sections = []
        for key, value in payload.items():
            rendered = json.dumps(value, indent=2, ensure_ascii=False) if not isinstance(value, str) else value
            sections.append(section(str(key), rendered))
        return sections or [section(fallback_title, json.dumps(payload, indent=2))]
    if isinstance(payload, list):
        paragraphs = [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in payload
        ]
        return [section(fallback_title, "\n\n".join(paragraphs) if paragraphs else "[]")]
    return [section(fallback_title, json.dumps(payload, indent=2, ensure_ascii=False) if payload is not None else "")]


CSV_ROWS_PER_SECTION = 50


def parse_csv_sections(
    text: str, fallback_title: str, stats: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """One paragraph per data row as ``header: value`` pairs, in sections of
    at most :data:`CSV_ROWS_PER_SECTION` rows. Cells past the header count are
    kept under synthetic ``column_N`` keys, never dropped."""
    try:
        rows = [row for row in csv.reader(StringIO(text)) if row]
    except csv.Error:
        if stats is not None:
            stats["parse_fallback"] = "csv_raw_text"
        return [section(fallback_title, text)]
    if not rows:
        return [section(fallback_title, text)]
    if len(rows) == 1:
        return [section(fallback_title, _csv_line(rows[0]))]
    headers = [cell.strip() for cell in rows[0]]
    ragged = surplus = short = 0
    paragraphs = []
    for row in rows[1:]:
        if len(row) != len(headers):
            ragged += 1
            if len(row) > len(headers):
                surplus += len(row) - len(headers)
            else:
                short += 1
        keys = [
            headers[index] if index < len(headers) and headers[index] else f"column_{index + 1}"
            for index in range(len(row))
        ]
        paragraphs.append("; ".join(f"{key}: {_csv_value(value)}" for key, value in zip(keys, row)))
    sections = []
    for start in range(0, len(paragraphs), CSV_ROWS_PER_SECTION):
        group = paragraphs[start : start + CSV_ROWS_PER_SECTION]
        title = (
            f"{fallback_title} rows"
            if len(paragraphs) <= CSV_ROWS_PER_SECTION
            else f"{fallback_title} rows {start + 1}-{start + len(group)}"
        )
        sections.append(section(title, "\n\n".join(group), paragraphs=group))
    if stats is not None:
        stats.update(
            {
                "csv_row_count": len(paragraphs),
                "csv_ragged_row_count": ragged,
                "csv_surplus_cell_count": surplus,
                "csv_short_row_count": short,
                "csv_section_count": len(sections),
            }
        )
    return sections


def _csv_value(value: str) -> str:
    # Quote values that would otherwise read as a pair separator or break a paragraph.
    return json.dumps(value, ensure_ascii=False) if any(mark in value for mark in (";", "\n", "\r")) else value


def _csv_line(row: list[str]) -> str:
    buffer = StringIO()
    csv.writer(buffer).writerow(row)
    return buffer.getvalue().rstrip("\r\n")


def section(title: str, text: str, paragraphs: list[str] | None = None) -> dict[str, Any]:
    """A section dict. Pass ``paragraphs`` when the caller has already split
    the text (e.g. preformatted blocks whose blank lines are not boundaries)."""
    if paragraphs is not None:
        kept = [paragraph for paragraph in paragraphs if paragraph.strip()]
        section_text = text.strip("\n")
        return {"title": title, "text": section_text, "paragraphs": kept or [section_text]}
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    section_text = text.strip()
    return {"title": title, "text": section_text, "paragraphs": paragraphs or [section_text]}
