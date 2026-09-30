import json
from email.message import Message

import pytest

from tirzah.config import RuntimeConfig
from tirzah.sessions.interaction import (
    allowed_tool_specs,
    build_memory_agent_prompt,
    execute_tool_calls,
    build_agentic_answer_envelope,
)
from tirzah.web_research import (
    WebResearchClient,
    WebResearchConfig,
    _PinnedHTTPConnection,
    _public_url,
)


class FakeDb:
    pass


@pytest.mark.parametrize("address", ["224.0.0.1", "100.64.0.1", "64:ff9b::7f00:1", "10.1.1.1"])
def test_non_public_addresses_are_blocked(monkeypatch, address):
    monkeypatch.setattr(
        "tirzah.web_research.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", (address, 80))],
    )
    with pytest.raises(ValueError, match="Private or non-global"):
        _public_url("http://example.test/x", allow_private_hosts=False)


def test_connection_uses_the_checked_address(monkeypatch):
    monkeypatch.setattr(
        "tirzah.web_research.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 80))],
    )
    seen = {}

    def fake_create(address, timeout, source_address):
        seen["address"] = address
        raise OSError("stop")

    conn = _PinnedHTTPConnection("example.test", 80, timeout=1, allow_private_hosts=False)
    monkeypatch.setattr(conn, "_create_connection", fake_create)
    with pytest.raises(OSError, match="stop"):
        conn.connect()
    assert seen["address"] == ("93.184.216.34", 80)


def test_outcome_fields_are_not_interpolated_into_attributes():
    from tirzah.web.outcomes_ui import OUTCOMES_COMPOSER_HTML

    assert "value='\"+" not in OUTCOMES_COMPOSER_HTML
    assert "id.value" in OUTCOMES_COMPOSER_HTML


def test_private_hosts_are_blocked(monkeypatch):
    monkeypatch.setattr(
        "tirzah.web_research.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(ValueError, match="Private"):
        _public_url("http://localhost:8080/search", allow_private_hosts=False)


def test_web_tools_only_appear_when_enabled():
    assert "web_search" not in {s["tool"] for s in allowed_tool_specs()}
    assert "web_search" in {s["tool"] for s in allowed_tool_specs(web_enabled=True)}
    prompt = build_memory_agent_prompt(
        "current evidence", None, "s", [], [], web_enabled=True
    )
    assert "web_search" in prompt and "untrusted evidence" in prompt


def test_disabled_web_tool_returns_repair_guidance():
    runtime = RuntimeConfig(web_research_enabled=False)
    result = execute_tool_calls(
        FakeDb(),
        [{"tool": "web_search", "arguments": {"query": "x"}}],
        runtime_config=runtime,
    )[0]
    assert result["ok"] is False
    assert "disabled" in result["error"].lower()
    assert "--web" in result["usage"]


def test_search_and_fetch_preserve_provenance(monkeypatch):
    monkeypatch.setattr(
        "tirzah.web_research.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )

    class Response:
        def __init__(self, body, content_type="application/json"):
            self.body = body
            self.headers = Message()
            self.headers["Content-Type"] = content_type

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self, n=-1):
            return self.body[:n]

    client = WebResearchClient(
        WebResearchConfig(enabled=True, allow_private_search_endpoint=True, max_pages=1)
    )
    responses = iter(
        [
            Response(
                json.dumps(
                    {
                        "results": [
                            {
                                "title": "Source",
                                "url": "https://example.test/a",
                                "content": "summary",
                            }
                        ]
                    }
                ).encode()
            ),
            Response(
                b"<html><body><script>ignore()</script><p>Evidence text</p></body></html>",
                "text/html",
            ),
        ]
    )
    monkeypatch.setattr(client, "_open", lambda url, **kwargs: next(responses))
    sources = client.research("test query")
    assert sources[0].url == "https://example.test/a"
    assert sources[0].snippet == "summary"
    assert "Evidence text" in sources[0].content and "ignore" not in sources[0].content


def test_web_context_has_distinct_status_and_no_fake_node_ids():
    envelope = build_agentic_answer_envelope(
        query="current evidence",
        tool_results=[
            {
                "index": 0,
                "tool": "web_search",
                "arguments": {"query": "current evidence"},
                "ok": True,
                "output": {
                    "query": "current evidence",
                    "sources": [
                        {
                            "title": "Source",
                            "url": "https://example.test",
                            "snippet": "summary",
                            "content": "evidence",
                            "retrieved_at": "2026-06-24T00:00:00+00:00",
                        }
                    ],
                },
            }
        ],
        token_budget=2000,
        reserved_response_tokens=500,
    )
    metadata = envelope["context_metadata"]
    assert metadata["retrieval_status"] == "agentic_web_context"
    assert metadata["included"] == []
    assert metadata["evidence_summary"]["web_source_count"] == 1
    assert "https://example.test" in envelope["prompt_text"]
