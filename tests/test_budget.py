import pytest
from pydantic import ValidationError

from tirzah.config import AppConfig, ModelBudgetProfile, RetrievalConfig, RuntimeConfig
from tirzah.retrieval.budget import count_tokens, envelope_budget_args, resolve_budget_plan
from tirzah.retrieval.queries import build_prompt_envelope, estimate_tokens


def test_count_tokens_approx_uses_four_chars() -> None:
    result = count_tokens("12345", tokenizer="approx", chars_per_token=4)
    assert result["tokens"] == 2
    assert result["tokenizer"] == "approx"


def test_count_tokens_tiktoken_falls_back_when_missing() -> None:
    result = count_tokens("hello world", tokenizer="tiktoken", encoding="cl100k_base")
    assert result["tokens"] >= 1
    assert result["tokenizer"] in {"tiktoken", "approx"}


def test_resolve_budget_plan_uses_model_profile() -> None:
    retrieval = RetrievalConfig(
        prompt_token_budget=2000,
        reserved_response_tokens=500,
        model_profiles={
            "gemma3:1b": ModelBudgetProfile(prompt_token_budget=1200, reserved_response_tokens=200)
        },
    )
    plan = resolve_budget_plan(retrieval, model="gemma3:1b", adapter="ollama_http")
    assert plan.profile_key == "gemma3:1b"
    assert plan.prompt_token_budget == 1200
    assert plan.reserved_response_tokens == 200


def test_envelope_budget_args_follow_runtime_model() -> None:
    config = AppConfig(
        runtime=RuntimeConfig(ollama_model="gemma3:1b", answer_adapter="ollama_http"),
        retrieval=RetrievalConfig(
            prompt_token_budget=2000,
            model_profiles={"gemma3:1b": ModelBudgetProfile(prompt_token_budget=900)},
        ),
    )
    args = envelope_budget_args(config)
    assert args["token_budget"] == 900
    assert args["budget_model"] == "gemma3:1b"


def test_build_prompt_envelope_reports_tokenizer_and_truncation() -> None:
    context = {
        "document": {"title": "Doc", "document_id": "doc1"},
        "focus_node_id": "node1",
        "records": [
            {
                "role": "focus",
                "distance": 0,
                "title": "A",
                "node_id": "node1",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text_preview": "short",
            },
            {
                "role": "descendant",
                "distance": 1,
                "title": "B",
                "node_id": "node2",
                "labels": [],
                "endorsement_label": "unreviewed",
                "provenance": {},
                "text_preview": "x" * 2000,
            },
        ],
    }
    envelope = build_prompt_envelope(
        context,
        query="what is this",
        token_budget=200,
        reserved_response_tokens=50,
        tokenizer="approx",
        chars_per_token=4.0,
    )
    assert envelope["budget"]["tokenizer"] == "approx"
    assert envelope["budget"]["token_budget"] == 200
    assert envelope["context_metadata"]["skipped_count"] >= 1
    assert estimate_tokens("abcd") == 1


def test_count_tokens_approx_is_not_reported_as_fallback() -> None:
    # Issue #52: a requested approx count is not a degraded one.
    result = count_tokens("hello world", tokenizer="approx")
    assert result["tokenizer_fallback"] is False
    assert result["tokenizer_requested"] == "approx"
    assert count_tokens("", tokenizer="approx")["tokenizer_fallback"] is False


def test_count_tokens_reports_real_tiktoken_fallback(monkeypatch) -> None:
    import tirzah.retrieval.budget as budget

    monkeypatch.setattr(budget, "_tiktoken_count", lambda text, encoding: None)
    result = count_tokens("hello world", tokenizer="tiktoken")
    assert result["tokenizer"] == "approx"
    assert result["tokenizer_requested"] == "tiktoken"
    assert result["tokenizer_fallback"] is True


def test_model_profile_rejects_blank_tokenizer() -> None:
    # Issue #54: "" used to silently inherit the global tokenizer.
    with pytest.raises(ValidationError):
        ModelBudgetProfile(tokenizer="")


def test_resolve_budget_plan_profile_tokenizer_overrides_global() -> None:
    retrieval = RetrievalConfig(
        tokenizer="tiktoken",
        model_profiles={"m": ModelBudgetProfile(tokenizer="approx")},
    )
    assert resolve_budget_plan(retrieval, model="m").tokenizer == "approx"
    assert resolve_budget_plan(retrieval, model="other").tokenizer == "tiktoken"


def _two_model_config() -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(ollama_model="gemma3:4b", answer_adapter="ollama_cli"),
        retrieval=RetrievalConfig(
            model_profiles={
                "gemma3:4b": ModelBudgetProfile(prompt_token_budget=2048, context_char_budget=4000),
                "llama3.1:70b": ModelBudgetProfile(
                    prompt_token_budget=32000, context_char_budget=90000
                ),
            }
        ),
    )


def test_envelope_budget_args_follow_request_runtime_override() -> None:
    # Issue #45: the model selected for the request, not the default, sets the budget.
    config = _two_model_config()
    selected = config.runtime.model_copy()
    selected.ollama_model = "llama3.1:70b"
    args = envelope_budget_args(config, runtime=selected)
    assert args["profile_key"] == "llama3.1:70b"
    assert args["token_budget"] == 32000
    assert args["context_char_budget"] == 90000
    assert envelope_budget_args(config)["profile_key"] == "gemma3:4b"


def test_direct_retrieval_passes_request_runtime_to_prompt(monkeypatch) -> None:
    import tirzah.sessions.interaction as ix
    from tirzah.sessions.answer_phases import retrieve_for_answer

    captured: dict = {}

    def fake_prepare(db, **kwargs):
        captured.update(kwargs)
        return {
            "selected_node_id": None,
            "selected_node_source": None,
            "prompt": {"context_metadata": {}},
            "retrieval_status": "no_focus_node",
            "retrieval_output": {},
        }

    monkeypatch.setattr(ix, "start_answer_process_run", lambda db, **kwargs: None)
    monkeypatch.setattr(ix, "prepare_direct_answer_prompt", fake_prepare)
    result = retrieve_for_answer(
        object(), _two_model_config(), "what is x", ollama_model="llama3.1:70b", retrieval_mode="direct"
    )
    assert result["ok"] is True
    assert captured["runtime_config"].ollama_model == "llama3.1:70b"


def test_pack_synthesis_chunks_bounds_deep_prompt() -> None:
    # Issue #63: 120 chunks at defaults used to overflow the budget 18x.
    from tirzah.retrieval.deep import build_synthesis_prompt, pack_synthesis_chunks

    plan = resolve_budget_plan(RetrievalConfig())
    chunks = [{"node_id": f"n{i}", "title": f"T{i}", "text": "x" * 1200} for i in range(120)]
    packed = pack_synthesis_chunks("q", chunks, plan, history_block="earlier turn")
    prompt = build_synthesis_prompt("q", packed["kept"], "earlier turn")
    assert plan.count(prompt) <= plan.prompt_token_budget - plan.reserved_response_tokens
    assert packed["budget"]["used_chars"] <= plan.context_char_budget
    assert len(packed["kept"]) + len(packed["skipped"]) == 120
    assert packed["budget"]["skipped_count"] == len(packed["skipped"]) > 0
    assert packed["skipped"][0]["reason"] == "prompt_budget"


def test_pack_synthesis_chunks_truncates_oversized_first_chunk() -> None:
    from tirzah.retrieval.deep import pack_synthesis_chunks

    plan = resolve_budget_plan(RetrievalConfig())
    packed = pack_synthesis_chunks("q", [{"node_id": "big", "text": "y" * 50000}], plan)
    assert [chunk["node_id"] for chunk in packed["kept"]] == ["big"]
    assert packed["kept"][0]["truncated_from_chars"] == 50000
    assert len(packed["kept"][0]["text"]) < plan.context_char_budget
    assert packed["budget"]["truncated_count"] == 1
    assert pack_synthesis_chunks("q", [{"node_id": "a"}], None)["budget"] == {}
