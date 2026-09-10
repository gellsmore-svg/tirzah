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
