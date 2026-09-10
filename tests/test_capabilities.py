from tirzah.capabilities import adapter_capabilities, render_runtime_text, resolved_runtime
from tirzah.config import RuntimeConfig


def test_adapter_capabilities_known_and_unknown():
    assert adapter_capabilities("mock")["uses_mock"] is True
    assert adapter_capabilities("hoglah")["requires_hoglah"] is True
    assert adapter_capabilities("ollama_http")["uses_live_model"] is True
    assert adapter_capabilities("kiro_cli")["can_answer"] is True
    assert adapter_capabilities("kiro_cli")["requires_kiro_cli"] is True
    assert adapter_capabilities("claude_cli")["requires_claude_cli"] is True
    assert adapter_capabilities("codex_cli")["requires_codex_cli"] is True
    assert adapter_capabilities("google_cli")["requires_google_cli"] is True
    assert adapter_capabilities("gemini_cli")["can_answer"] is True
    assert adapter_capabilities("grok_cli")["requires_grok_cli"] is True
    assert adapter_capabilities("nope") == {"unknown": True}


def test_resolved_runtime_flags_mock_and_disabled_mahalath():
    # The exact CBO situation: mock embeddings + Mahalath off.
    rt = RuntimeConfig(model_adapter="mock", answer_adapter="mock", embedding_adapter="mock",
                       mahalath_enabled=False)
    snap = resolved_runtime(rt)
    caps = snap["capabilities"]
    assert caps["uses_mock"] is True
    assert caps["can_ingest_semantically"] is False
    assert caps["can_resolve_mpl"] is False
    assert caps["requires_hoglah"] is False
    msgs = " ".join(snap["warnings"])
    assert "embedding_adapter=mock" in msgs and "mahalath_enabled=false" in msgs


def test_resolved_runtime_hoglah_real_stack():
    rt = RuntimeConfig(model_adapter="hoglah", answer_adapter="hoglah", embedding_adapter="hoglah",
                       embedding_model="bge-m3:latest", embedding_dimensions=1024,
                       mahalath_enabled=True, mahalath_strict=True)
    snap = resolved_runtime(rt)
    caps = snap["capabilities"]
    assert caps["uses_mock"] is False
    assert caps["can_ingest_semantically"] is True
    assert caps["can_resolve_mpl"] is True and caps["requires_hoglah"] is True
    assert snap["embedding_dimensions"] == 1024
    assert any("Hoglah" in w for w in snap["warnings"])  # worker-required reminder


def test_render_runtime_text_smoke():
    snap = resolved_runtime(RuntimeConfig())
    text = render_runtime_text(snap)
    assert "Resolved runtime:" in text and "adapters:" in text and "capabilities:" in text
