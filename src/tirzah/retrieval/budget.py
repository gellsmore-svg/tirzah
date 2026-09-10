# src/tirzah/retrieval/budget.py

def resolve_budget_plan(data: dict | None, retrieval: RetrievalConfig) -> BudgetPlan:
    if data is None:
        data = {}
    
    prompt_token_budget = (
        data.get("prompt_token_budget")
        if data.get("prompt_token_budget") is not None
        else retrieval.prompt_token_budget
    )
    reserved_response_tokens = (
        data.get("reserved_response_tokens")
        if data.get("reserved_response_tokens") is not None
        else retrieval.reserved_response_tokens
    )
    context_char_budget = (
        data.get("context_char_budget")
        if data.get("context_char_budget") is not None
        else retrieval.context_char_budget
    )
    chars_per_token = (
        data.get("chars_per_token")
        if data.get("chars_per_token") is not None
        else retrieval.chars_per_token
    )
    tokenizer = (
        data.get("tokenizer")
        if data.get("tokenizer") is not None
        else retrieval.tokenizer
    )

    return BudgetPlan(
        prompt_token_budget=prompt_token_budget,
        reserved_response_tokens=reserved_response_tokens,
        context_char_budget=context_char_budget,
        chars_per_token=chars_per_token,
        tokenizer=tokenizer,
    )