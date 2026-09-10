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


def parse_structure(text: str, source_kind: str, fallback_title: str) -> list[dict[str, Any]]:
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
    sections = parser(text, fallback_title)
    return sections or [section(fallback_title, text)]


def parse_markdown_sections(text: str, fallback_title: str) -> list[dict[str, Any]]:
    body = strip_front_matter(text)
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


def parse_html_sections(text: str, fallback_title: str) -> list[dict[str, Any]]:
    parser = HtmlOutlineParser(fallback_title)
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return [section(fallback_title, strip_tags(text))]
    return parser.sections() or [section(parser.document_title or fallback_title, parser.visible_text() or strip_tags(text))]


class HtmlOutlineParser(HTMLParser):
    SKIP = {"script", "style", "noscript"}
    CHROME = {"nav", "footer"}
    HEADINGS = {f"h{index}" for index in range(1, 7)}
    BLOCKS = {"p", "li", "pre", "blockquote", "dt", "dd"}

    def __init__(self, fallback_title: str) -> None:
        super().__init__(convert_charrefs=True)
        self.fallback_title = fallback_title
        self.document_title = ""
        self._skip_depth = 0
        self._chrome_depth = 0
        self._in_title = False
        self._heading_open = False
        self._current_title = fallback_title
        self._blocks: list[str] = []
        self._sections: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag in self.CHROME:
            self._chrome_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if self._skip_depth or self._chrome_depth:
            return
        if tag in self.HEADINGS:
            self._flush_section()
            self._heading_open = True
            return
        if tag in self.BLOCKS:
            self._heading_open = False
            self._blocks.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.CHROME and self._chrome_depth:
            self._chrome_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.HEADINGS:
            self._heading_open = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._chrome_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title and not self.document_title:
            self.document_title = text
            return
        if self._heading_open:
            self._current_title = text
            self._heading_open = False
            return
        if self._blocks:
            self._blocks[-1] = f"{self._blocks[-1]} {text}".strip() if self._blocks[-1] else text
        else:
            self._blocks.append(text)

    def _flush_section(self) -> None:
        paragraphs = [block.strip() for block in self._blocks if block and block.strip()]
        if paragraphs:
            self._sections.append(section(self._current_title, "\n\n".join(paragraphs)))
        self._blocks = []
        self._heading_open = False

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


def parse_csv_sections(text: str, fallback_title: str) -> list[dict[str, Any]]:
    try:
        rows = list(csv.reader(StringIO(text)))
    except csv.Error:
        return [section(fallback_title, text)]
    if not rows:
        return [section(fallback_title, text)]
    headers = [cell.strip() for cell in rows[0]]
    paragraphs = []
    for row in rows[1:] or rows:
        if headers and len(rows) > 1:
            pairs = [f"{headers[index]}: {value}" for index, value in enumerate(row) if index < len(headers)]
            paragraphs.append("; ".join(pairs) if pairs else ",".join(row))
        else:
            paragraphs.append(",".join(row))
    title = fallback_title if len(rows) == 1 else f"{fallback_title} rows"
    return [section(title, "\n\n".join(paragraphs))]


def section(title: str, text: str) -> dict[str, Any]:
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    section_text = text.strip()
    return {"title": title, "text": section_text, "paragraphs": paragraphs or [section_text]}
