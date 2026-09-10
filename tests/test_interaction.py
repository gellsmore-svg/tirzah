from datetime import datetime, timezone

from bson import ObjectId

from tirzah.config import AppConfig, RuntimeConfig
from tirzah.sessions.activity_reports import evidence_summary_lines
from tirzah.sessions.interaction import (
    active_document_default_node_id,
    active_document_reference_query,
    active_document_vocabulary_values,
    agentic_evidence_summary,
    answer_query,
    build_active_document_source_fallback_envelope,
    build_agentic_answer_envelope,
    build_memory_agent_prompt,
    build_query_assembly,
    combined_query_text,
    compact_fallback_query_details,
    direct_context_controller_decision,
    direct_retrieval_decision,
    execute_search_nodes_tool,
    fallback_queries,
    memory_agent_runtime_config,
    near_match_vocabulary,
    near_match_terms,
    parse_memory_agent_decision,
    parse_tool_calls,
    prepare_tool_results_for_answer,
    ranked_focus_matches,
    render_tool_results,
    run_memory_agent_loop,
    included_nodes_from_tool_results,
    score_node_match,
    select_active_document_focus_node,
    select_focus_node,
    summarize_tool_results_for_memory_agent,
    validate_controller_decision,
    vocabulary_terms,
)


class FakeDb:
    pass


class FakeCursor(list):
    def limit(self, limit):
        return FakeCursor(self[:limit])


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def find(self, *_args):
        return FakeCursor(self.rows)

    def distinct(self, key):
        values = []
        for row in self.rows:
            value = row.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif value is not None:
                values.append(value)
        return values


class FakeVocabularyDb:
    def __init__(self):
        self.label_definitions = FakeCollection(
            [
                {
                    "key": "memory_reference",
                    "description": "Reference material about memory.",
                }
            ]
        )
        self.documents = FakeCollection(
            [
                {
                    "title": "Technical Design",
                    "source": {"path": "/tmp/design.md"},
                }
            ]
        )
        self.nodes = FakeCollection([{"labels": ["taj_mahal"]}])


class FakeNodeCollection:
    def __init__(self, rows):
        self.rows = rows

    def find_one(self, query, sort=None):
        rows = [row for row in self.rows if mongo_matches(row, query)]
        if sort:
            for field, direction in reversed(sort):
                rows.sort(key=lambda row: row.get(field, 0), reverse=direction < 0)
        return rows[0] if rows else None


class FakeNodeDb:
    def __init__(self, nodes):
        self.nodes = FakeNodeCollection(nodes)


class EndToEndFakeInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class EndToEndFakeUpdateResult:
    def __init__(self, modified_count):
        self.modified_count = modified_count


class EndToEndFakeCursor(list):
    def sort(self, field, direction=None):
        if isinstance(field, list):
            sort_fields = field
        else:
            sort_fields = [(field, direction)]
        rows = list(self)
        for sort_field, sort_direction in reversed(sort_fields):
            rows.sort(
                key=lambda row: row.get(sort_field) or datetime.min.replace(tzinfo=timezone.utc),
                reverse=sort_direction < 0,
            )
        return EndToEndFakeCursor(rows)

    def limit(self, limit):
        return EndToEndFakeCursor(self[:limit])


class EndToEndFakeCollection:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def insert_one(self, row):
        inserted = dict(row)
        inserted["_id"] = inserted.get("_id") or ObjectId()
        self.rows.append(inserted)
        return EndToEndFakeInsertResult(inserted["_id"])

    def find(self, query=None, projection=None):
        rows = [row for row in self.rows if end_to_end_matches(row, query or {})]
        return EndToEndFakeCursor([end_to_end_project(row, projection) for row in rows])

    def find_one(self, query=None, projection=None, sort=None):
        rows = self.find(query or {}, projection)
        if sort:
            rows = rows.sort(sort)
        return rows[0] if rows else None

    def distinct(self, key):
        values = []
        for row in self.rows:
            value = row.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif value is not None:
                values.append(value)
        return values

    def update_one(self, filter_query, update, upsert=False):
        row = next((item for item in self.rows if end_to_end_matches(item, filter_query)), None)
        if row is None:
            if not upsert:
                return EndToEndFakeUpdateResult(0)
            row = dict(filter_query)
            row.update(update.get("$setOnInsert", {}))
            self.rows.append(row)
        end_to_end_apply_update(row, update)
        return EndToEndFakeUpdateResult(1)

    def update_many(self, filter_query, update):
        modified_count = 0
        for row in self.rows:
            if not end_to_end_matches(row, filter_query):
                continue
            end_to_end_apply_update(row, update)
            modified_count += 1
        return EndToEndFakeUpdateResult(modified_count)


class EndToEndAnswerDb:
    def __init__(self, *, document_id, node_id):
        self.sessions = EndToEndFakeCollection()
        self.exchanges = EndToEndFakeCollection()
        self.output_ingestion_queue = EndToEndFakeCollection()
        self.active_documents = EndToEndFakeCollection()
        self.project_domains = EndToEndFakeCollection()
        self.conversation_domains = EndToEndFakeCollection()
        self.process_runs = EndToEndFakeCollection()
        self.documents = EndToEndFakeCollection(
            [
                {
                    "_id": document_id,
                    "title": "Active Document",
                    "source": {"path": "active.md"},
                }
            ]
        )
        self.nodes = EndToEndFakeCollection(
            [
                {
                    "_id": node_id,
                    "document_id": document_id,
                    "tree_id": ObjectId(),
                    "title": "Active Evidence",
                    "text": "Active document evidence.",
                    "labels": ["source_chunk", "active_test"],
                    "endorsement_label": "unreviewed",
                    "usage_score": 0,
                    "created_at": datetime.now(timezone.utc),
                }
            ]
        )


def mongo_matches(row, query):
    for key, expected in query.items():
        actual = row.get(key)
        if isinstance(actual, list):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    return True


def end_to_end_matches(row, query):
    for key, expected in query.items():
        actual = end_to_end_nested_get(row, key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$nin" in expected and actual in expected["$nin"]:
                return False
            continue
        if isinstance(actual, list):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    return True


def end_to_end_nested_get(row, dotted_key):
    value = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def end_to_end_project(row, projection):
    if not projection:
        return dict(row)
    projected = {
        key: end_to_end_nested_get(row, key)
        for key in projection
        if end_to_end_nested_get(row, key) is not None
    }
    if "_id" in row and projection.get("_id", 1):
        projected["_id"] = row["_id"]
    return projected


def end_to_end_apply_update(row, update):
    row.update(update.get("$setOnInsert", {}))
    for field, value in update.get("$set", {}).items():
        end_to_end_set_nested(row, field, value)
    for field, value in update.get("$inc", {}).items():
        row[field] = row.get(field, 0) + value
    for field, value in update.get("$addToSet", {}).items():
        row.setdefault(field, [])
        for item in value.get("$each", []):
            if item not in row[field]:
                row[field].append(item)
    for field, value in update.get("$push", {}).items():
        row.setdefault(field, []).append(value)


def end_to_end_set_nested(row, dotted_key, value):
    target = row
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def test_select_focus_node_returns_none_without_matches(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    monkeypatch.setattr(interaction, "search_nodes", lambda *args, **kwargs: [])

    assert select_focus_node(FakeDb(), "missing") is None


def test_select_focus_node_falls_back_to_ranked_terms(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        if query == "Tirzah" and label == "source_chunk":
            return [
                {
                    "node_id": "generic",
                    "title": "Worker Smoke",
                    "text_preview": "Tirzah ingestion worker",
                },
                {
                    "node_id": "technical",
                    "title": "Tirzah Technical Design Document",
                    "text_preview": "Architecture notes",
                },
            ]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)

    assert select_focus_node(FakeDb(), "What does the Tirzah technical design say?") == "technical"


def test_direct_retrieval_decision_skips_low_intent_and_generic_prompts() -> None:
    greeting = direct_retrieval_decision("hello")
    assert greeting["category"] == "low_intent"
    assert greeting["should_search_corpus"] is False

    generic = direct_retrieval_decision("tell me more")
    assert generic["category"] == "generic_prompt"
    assert generic["should_search_corpus"] is False

    repository = direct_retrieval_decision("What does the Tirzah technical design say?")
    assert repository["category"] == "repository_query"
    assert repository["should_search_corpus"] is True
    assert repository["minimum_match_score"] >= 1


def test_direct_context_controller_decision_marks_scaffold_owner() -> None:
    retrieval = direct_retrieval_decision("What does the Tirzah technical design say?")
    decision = direct_context_controller_decision(
        retrieval_decision=retrieval,
        selected_node_id="node1",
        selected_node_source="corpus",
        retrieval_status="matched_context",
    )

    assert decision["mode"] == "direct_scaffold"
    assert decision["current_owner"] == "deterministic_guardrail"
    assert decision["target_owner"] == "memory_agent_controller"
    assert decision["action"] == "use_repository_context"
    assert decision["validation_issues"] == []


def test_validate_controller_decision_reports_schema_issues() -> None:
    issues = validate_controller_decision(
        {
            "action": "invented_action",
            "reason": "",
            "confidence": 1.5,
        }
    )

    assert {issue["field"] for issue in issues} == {"action", "reason", "confidence"}


def test_select_focus_node_rejects_weak_best_match(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        if label == "source_chunk":
            return [
                {
                    "node_id": "weak",
                    "title": "Unrelated Header",
                    "text_preview": "memory",
                    "labels": ["source_chunk"],
                }
            ]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)

    assert select_focus_node(FakeDb(), "memory") is None


def test_active_document_reference_query_detects_session_references() -> None:
    assert active_document_reference_query("What does this document say about memory?")
    assert active_document_reference_query("Summarize the previous source.")
    assert not active_document_reference_query("What does the Taj Mahal article say?")


def test_select_active_document_focus_node_scopes_search_to_active_document(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []
    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=5: [
            {"document_id": "doc1", "title": "Active Source"}
        ],
    )

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        calls.append({"query": query, "label": label, "document_id": document_id})
        if document_id == "doc1" and label == "source_chunk":
            return [
                {
                    "node_id": "active-node",
                    "title": "Active Source",
                    "text_preview": "memory capture",
                    "labels": ["source_chunk"],
                }
            ]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)

    assert select_active_document_focus_node(FakeDb(), "What does this document say?", "s1") == "active-node"
    assert calls[0]["document_id"] == "doc1"
    assert calls[0]["label"] == "source_chunk"


def test_select_active_document_focus_node_falls_back_to_document_root(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    document_id = ObjectId()
    root_id = ObjectId()
    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=5: [
            {"document_id": str(document_id), "title": "Active Source", "node_ids": []}
        ],
    )
    monkeypatch.setattr(interaction, "search_nodes", lambda *args, **kwargs: [])
    db = FakeNodeDb(
        [
            {
                "_id": root_id,
                "document_id": document_id,
                "labels": ["source_root"],
                "order": 0,
            }
        ]
    )

    assert select_active_document_focus_node(db, "What does this document say?", "s1") == str(root_id)


def test_select_active_document_focus_node_checks_later_documents_before_default(
    monkeypatch,
) -> None:
    import tirzah.sessions.interaction as interaction

    first_document_id = ObjectId()
    second_document_id = ObjectId()
    first_root_id = ObjectId()
    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=5: [
            {"document_id": str(first_document_id), "title": "First Source", "node_ids": []},
            {"document_id": str(second_document_id), "title": "Second Source", "node_ids": []},
        ],
    )

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        if document_id == str(second_document_id) and label == "source_chunk":
            return [
                {
                    "node_id": "second-topical-node",
                    "title": "Second Source",
                    "text_preview": "topical match",
                    "labels": ["source_chunk"],
                }
            ]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    db = FakeNodeDb(
        [
            {
                "_id": first_root_id,
                "document_id": first_document_id,
                "labels": ["source_root"],
                "order": 0,
            }
        ]
    )

    assert (
        select_active_document_focus_node(db, "What does the previous source say?", "s1")
        == "second-topical-node"
    )


def test_active_document_default_node_prefers_root_then_active_node() -> None:
    document_id = ObjectId()
    active_node_id = ObjectId()
    root_id = ObjectId()
    db = FakeNodeDb(
        [
            {
                "_id": active_node_id,
                "document_id": document_id,
                "labels": ["source_chunk"],
                "order": 1,
            },
            {
                "_id": root_id,
                "document_id": document_id,
                "labels": ["source_root"],
                "order": 0,
            },
        ]
    )

    assert active_document_default_node_id(db, str(document_id), [str(active_node_id)]) == str(root_id)


def test_active_document_default_node_uses_active_node_without_root() -> None:
    document_id = ObjectId()
    active_node_id = ObjectId()
    db = FakeNodeDb(
        [
            {
                "_id": active_node_id,
                "document_id": document_id,
                "labels": ["source_chunk"],
                "order": 1,
            }
        ]
    )

    assert active_document_default_node_id(db, str(document_id), [str(active_node_id)]) == str(active_node_id)


def test_render_conversation_history_chronological() -> None:
    from tirzah.sessions.interaction import render_conversation_history

    # recent_exchanges returns newest-first; rendering must be oldest-first
    exchanges = [
        {"query": "second question", "answer": "second answer"},
        {"query": "first question", "answer": "first answer"},
    ]
    block = render_conversation_history(exchanges)
    assert "## Conversation So Far" in block
    assert block.index("first question") < block.index("second question")
    assert "User: first question" in block and "Assistant: first answer" in block


def test_render_conversation_history_empty() -> None:
    from tirzah.sessions.interaction import render_conversation_history

    assert render_conversation_history([]) == ""


def test_inject_history_into_prompt() -> None:
    from tirzah.sessions.interaction import inject_history_into_prompt

    prompt = {"prompt_text": "INSTRUCTION\n\n## User Query\nhello\n", "budget": {"reserved_response_tokens": 100}}
    out = inject_history_into_prompt(prompt, "## Conversation So Far\nUser: hi\nAssistant: hey")
    text = out["prompt_text"]
    assert "## Conversation So Far" in text
    assert text.index("Conversation So Far") < text.index("## User Query")  # before the query
    # empty history is a no-op
    assert inject_history_into_prompt(prompt, "")["prompt_text"] == prompt["prompt_text"]


def test_relevant_exchanges_ranks_by_cosine() -> None:
    from datetime import datetime, timezone

    from bson import ObjectId

    from tirzah.sessions.exchanges import relevant_exchanges

    class Coll:
        def __init__(self, rows):
            self.rows = rows

        def find(self, query=None):
            query = query or {}
            return [r for r in self.rows if all(r.get(k) == v for k, v in query.items())]

    class Db:
        def __init__(self, rows):
            self.exchanges = Coll(rows)

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"_id": ObjectId(), "session_id": "s", "query": "about cats", "answer": {"answer": "cat"}, "turn_embedding": [1.0, 0.0], "created_at": now},
        {"_id": ObjectId(), "session_id": "s", "query": "about dogs", "answer": {"answer": "dog"}, "turn_embedding": [0.0, 1.0], "created_at": now},
        {"_id": ObjectId(), "session_id": "s", "query": "no embedding", "answer": {"answer": "x"}, "turn_embedding": None, "created_at": now},
    ]
    db = Db(rows)
    out = relevant_exchanges(db, session_id="s", query_vector=[0.9, 0.1], limit=2)
    assert out[0]["query"] == "about cats"  # closest to [0.9, 0.1]
    assert all(o["query"] != "no embedding" for o in out)  # un-embedded turns skipped
    excluded = out[0]["exchange_id"]
    out2 = relevant_exchanges(db, session_id="s", query_vector=[0.9, 0.1], limit=2, exclude_exchange_ids={excluded})
    assert all(o["exchange_id"] != excluded for o in out2)
    assert relevant_exchanges(db, session_id="s", query_vector=None) == []  # no vector -> empty


def test_background_scheduling_uses_priority_queue(monkeypatch) -> None:
    import tirzah.sessions.background as background
    import tirzah.sessions.interaction as interaction
    from tirzah.config import AppConfig

    captured: list[dict] = []

    class FakeQueue:
        def submit(self, fn, *args, priority=None, key=None, **kwargs):
            captured.append({"fn": fn.__name__, "priority": priority, "key": key})

    monkeypatch.setattr(background, "get_background_queue", lambda: FakeQueue())
    config = AppConfig()
    config.retrieval.conversation_semantic_recall = True
    config.retrieval.conversation_chunking = True

    interaction.schedule_turn_embedding(None, config, config.runtime, "exid", "sess1", "q", "a")
    interaction.schedule_chunking(None, config, config.runtime, "exid", "sess1", "q", "a")

    assert captured[0] == {"fn": "_embed_and_backfill", "priority": 2, "key": "sess1"}  # memory completion
    assert captured[1] == {"fn": "_chunk_and_store", "priority": 3, "key": "sess1"}  # chunking, same session key


def test_backfill_turn_embeddings(monkeypatch) -> None:
    from bson import ObjectId

    import tirzah.sessions.interaction as interaction
    from tirzah.config import AppConfig

    updated: list = []

    class Cursor:
        def __init__(self, rows):
            self._rows = rows

        def sort(self, *args, **kwargs):
            return self

        def limit(self, n):
            return self._rows[:n]

    class Coll:
        def __init__(self, rows):
            self.rows = rows

        def find(self, query=None):
            query = query or {}
            return Cursor([r for r in self.rows if all(r.get(k) == v for k, v in query.items())])

        def update_one(self, query, update):
            updated.append((query, update))

    class Db:
        def __init__(self, rows):
            self.exchanges = Coll(rows)

    rows = [
        {"_id": ObjectId(), "query": "q1", "answer": {"answer": "a1"}, "turn_embedding": None, "created_at": None},
        {"_id": ObjectId(), "query": "q2", "answer": {"answer": "a2"}, "turn_embedding": [1.0], "created_at": None},
    ]
    db = Db(rows)
    monkeypatch.setattr(interaction, "build_query_embedding", lambda _rc, _text: {"vector": [0.1, 0.2]})

    config = AppConfig()
    config.retrieval.conversation_semantic_recall = True
    embedded = interaction.backfill_turn_embeddings(db, config, config.runtime, limit=10)
    assert embedded == 1  # only the un-embedded turn
    assert len(updated) == 1
    assert updated[0][1] == {"$set": {"turn_embedding": [0.1, 0.2]}}

    config.retrieval.conversation_semantic_recall = False
    assert interaction.backfill_turn_embeddings(db, config, config.runtime) == 0  # no-op when disabled


def test_answer_query_uses_prompt_without_focus_node(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    captured = {}

    class FakeAnswerAdapter:
        def answer(self, prompt):
            captured["prompt"] = prompt
            return {
                "adapter": "fake",
                "answer": "direct answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")

    result = answer_query(FakeDb(), AppConfig(), "plain prompt")

    assert result["ok"] is True
    assert result["focus_node_id"] is None
    assert result["retrieval_status"] == "no_focus_node"
    assert result["used_node_ids"] == []
    assert result["activity_report"]["kind"] == "answer_activity_report"
    assert result["activity_report"]["context_construction"]["status"] == "no_focus_node"
    assert result["activity_report"]["context_construction"]["evidence_summary"] == {
        "mode": "direct",
        "retrieval_status": "no_focus_node",
        "selected_node_id": None,
        "selected_node_source": None,
        "included_node_count": 0,
        "included_node_ids": [],
        "skipped_count": 0,
        "source_fallback_used": False,
        "source_documents": [],
    }
    assert (
        result["activity_report"]["context_construction"]["controller_decision"]["target_owner"]
        == "memory_agent_controller"
    )
    assert result["activity_report"]["llm_activity"]["call_count"] == 1
    assert result["activity_log"].startswith("Answer Activity Log")
    assert "Context construction:" in result["activity_log"]
    assert "Evidence summary: 0 repository node(s) included." not in result["activity_log"]
    # the query is in the prompt's User Query section, not echoed into the context
    # block (the no-context "Submitted Prompt" scaffolding was removed)
    assert "plain prompt" in captured["prompt"]["prompt_text"]
    assert "No stored memory matched this request" in captured["prompt"]["context_text"]
    assert "## Controller Decision" in captured["prompt"]["prompt_text"]
    assert "- Action: skip_weak_or_missing_repository_context" in captured["prompt"]["prompt_text"]
    assert captured["prompt"]["context_metadata"]["evidence_summary"] == {
        "mode": "direct",
        "retrieval_status": "no_focus_node",
        "selected_node_id": None,
        "selected_node_source": None,
        "included_node_count": 0,
        "included_node_ids": [],
        "skipped_count": 0,
        "source_fallback_used": False,
        "source_documents": [],
    }
    assert [step["step"] for step in result["process_trace"]] == [
        "user_prompt",
        "retrieval_context",
        "answer_adapter",
    ]
    assert "No stored memory matched this request" in result["process_trace"][1]["output"]["context_text"]
    assert result["process_trace"][1]["output"]["controller_decision"]["mode"] == "direct_scaffold"


def test_answer_query_surfaces_semantic_summary(monkeypatch) -> None:
    """The Mahalath seam's resolved senses reach the ask result (semantic_summary)."""
    import tirzah.semantic as semantic
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {"adapter": "fake", "answer": "an answer", "used_node_ids": ["corpus-node"]}

    class FakeResolver:
        def resolve(self, terms):
            if any(t.casefold() == "form" for t in terms):
                return [semantic.SemanticLabel(term="form", mpl_label="MPL-901",
                                               canonical_term="form", senses=["logical"],
                                               match_kind="exact")]
            return []

    monkeypatch.setattr(interaction, "select_focus_node", lambda *_a, **_k: "corpus-node")
    monkeypatch.setattr(interaction, "compile_context", lambda _db, node_id: {
        "document": {"title": "Doc", "document_id": "doc1"},
        "focus_node_id": node_id,
        "records": [{"role": "focus", "distance": 0, "title": "A", "node_id": node_id,
                     "labels": [], "endorsement_label": "unreviewed", "provenance": {},
                     "text": "the form of the law", "text_preview": "the form of the law"}],
    })
    monkeypatch.setattr(semantic, "make_resolver", lambda _runtime: FakeResolver())
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *a, **k: "exchange1")

    result = answer_query(
        FakeDb(),
        AppConfig(runtime=RuntimeConfig(answer_adapter="fake", mahalath_enabled=True)),
        "what is form?",
    )

    assert result["ok"] is True
    assert result["semantic_summary"] == "interpreted as: form→MPL-901 (logical)"
    assert result["semantic"][0]["mpl_label"] == "MPL-901"


def test_answer_query_does_not_retrieve_corpus_for_greeting(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "Hello.",
                "used_node_ids": [],
            }

    def fake_select_focus_node(_db, query):
        calls.append(query)
        return "should-not-run"

    monkeypatch.setattr(interaction, "select_focus_node", fake_select_focus_node)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")

    result = answer_query(FakeDb(), AppConfig(), "hello")

    assert calls == []
    assert result["ok"] is True
    assert result["focus_node_id"] is None
    assert result["retrieval_status"] == "no_focus_node"
    assert result["used_node_ids"] == []


def test_answer_query_handles_empty_prompt_without_retrieval_crash(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "No prompt was supplied.",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")

    result = answer_query(FakeDb(), AppConfig(), "   ")

    assert result["ok"] is True
    assert result["retrieval_status"] == "no_focus_node"
    assert result["focus_node_id"] is None


def test_answer_query_forces_no_context_direct_path_for_agentic_greeting(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "Hello.",
                "used_node_ids": [],
            }

    def fail_agentic(**_kwargs):
        raise AssertionError("agentic retrieval should not run for low-intent greetings")

    monkeypatch.setattr(interaction, "answer_query_agentic", fail_agentic)
    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: "should-not-run")
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")

    result = answer_query(FakeDb(), AppConfig(), "hello", retrieval_mode="agentic")

    assert result["ok"] is True
    assert result["retrieval_status"] == "no_focus_node"
    assert result["focus_node_id"] is None
    assert result["process_trace"][0]["output"]["retrieval_mode_override"] == "direct_no_context_low_intent"


def test_answer_query_records_process_run(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    created = []
    updates = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "direct answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: created.append(kwargs) or {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )

    result = answer_query(FakeDb(), AppConfig(), "plain prompt", session_id="s1")

    assert result["process_run_id"] == "run1"
    assert created == [
        {
            "process_id": "answer_query",
            "session_id": "s1",
            "current_step_id": "direct_retrieval",
            "status": "active",
        }
    ]
    assert updates == [
        {
            "run_id": "run1",
            "status": "completed",
            "current_step_id": "answer_saved",
            "completed_step_id": "answer_adapter",
            "exchange_id": "exchange1",
            "exception": None,
        }
    ]


def test_answer_query_continues_when_process_run_create_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "direct answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: (_ for _ in ()).throw(RuntimeError("process DB down")),
    )

    result = answer_query(FakeDb(), AppConfig(), "plain prompt", session_id="s1")

    assert result["ok"] is True
    assert result["exchange_id"] == "exchange1"
    assert result["process_run_id"] is None


def test_answer_query_continues_when_process_run_update_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "direct answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: (_ for _ in ()).throw(RuntimeError("process update failed")),
    )

    result = answer_query(FakeDb(), AppConfig(), "plain prompt", session_id="s1")

    assert result["ok"] is True
    assert result["exchange_id"] == "exchange1"
    assert result["process_run_id"] == "run1"


def test_answer_query_marks_process_run_blocked_when_adapter_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    updates = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            raise RuntimeError("adapter offline")

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )

    result = answer_query(FakeDb(), AppConfig(), "plain prompt", session_id="s1")

    assert result["ok"] is False
    assert result["process_run_id"] == "run1"
    assert result["activity_report"]["status"] == "blocked"
    assert "Status: blocked" in result["activity_log"]
    assert updates[0]["status"] == "blocked"
    assert updates[0]["current_step_id"] == "answer_adapter_failed"
    assert updates[0]["exception"]["reason"] == "answer_adapter_failed"
    assert updates[0]["exception"]["type"] == "RuntimeError"
    assert "adapter offline" in updates[0]["exception"]["note"]


def test_answer_query_marks_process_run_blocked_when_exchange_save_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    updates = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "answer text",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(
        interaction,
        "save_exchange",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )

    result = answer_query(FakeDb(), AppConfig(), "plain prompt", session_id="s1")

    assert result["ok"] is False
    assert result["reason"] == "answer_save_failed"
    assert result["process_run_id"] == "run1"
    assert result["process_trace"][-1]["step"] == "save_exchange"
    assert result["process_trace"][-1]["output"]["ok"] is False
    assert updates[0]["status"] == "blocked"
    assert updates[0]["current_step_id"] == "answer_save_failed"
    assert updates[0]["exception"]["reason"] == "answer_save_failed"
    assert updates[0]["exception"]["type"] == "RuntimeError"
    assert "write failed" in updates[0]["exception"]["note"]


def test_answer_query_marks_process_run_blocked_when_retrieval_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    updates = []

    monkeypatch.setattr(
        interaction,
        "select_focus_node",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("retrieval broke")),
    )
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )

    result = answer_query(FakeDb(), AppConfig(), "plain prompt", session_id="s1")

    assert result["ok"] is False
    assert result["reason"] == "retrieval_failed"
    assert result["process_run_id"] == "run1"
    assert updates[0]["status"] == "blocked"
    assert updates[0]["current_step_id"] == "retrieval_failed"
    assert updates[0]["exception"]["reason"] == "retrieval_failed"
    assert updates[0]["exception"]["type"] == "RuntimeError"
    assert "retrieval broke" in updates[0]["exception"]["note"]


def test_answer_query_uses_active_document_source_fallback(monkeypatch, tmp_path) -> None:
    import tirzah.sessions.interaction as interaction

    source_path = tmp_path / "active-source.md"
    source_path.write_text(
        "# Active Source\n\nThe source fallback contains the selected test evidence.",
        encoding="utf-8",
    )
    captured = {}

    class FakeAnswerAdapter:
        def answer(self, prompt):
            captured["prompt"] = prompt
            return {
                "adapter": "fake",
                "answer": "source fallback answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_active_document_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=5: [
            {
                "document_id": "doc1",
                "title": "Active Source",
                "source": {"path": str(source_path)},
            }
        ],
    )
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")

    result = answer_query(
        FakeDb(),
        AppConfig(runtime=RuntimeConfig(answer_adapter="fake")),
        "What does this document say about selected test evidence?",
        session_id="s1",
    )

    assert result["ok"] is True
    assert result["focus_node_id"] is None
    assert result["used_node_ids"] == []
    assert result["retrieval_status"] == "active_document_source_fallback"
    assert "source fallback contains the selected test evidence" in captured["prompt"]["context_text"]
    assert captured["prompt"]["context_metadata"]["included"] == []
    assert captured["prompt"]["context_metadata"]["source_fallback"]["document_id"] == "doc1"
    evidence = captured["prompt"]["context_metadata"]["evidence_summary"]
    assert evidence["source_fallback_used"] is True
    assert evidence["source_documents"][0]["document_id"] == "doc1"
    assert evidence["source_documents"][0]["truncated"] is False
    assert result["process_trace"][1]["output"]["retrieval_status"] == (
        "active_document_source_fallback"
    )


def test_answer_query_uses_active_document_source_fallback_after_no_focus_match(
    monkeypatch,
    tmp_path,
) -> None:
    import tirzah.sessions.interaction as interaction

    source_path = tmp_path / "active-source.md"
    source_path.write_text(
        "# Active Source\n\nA local source fallback can answer broader follow-up questions.",
        encoding="utf-8",
    )
    captured = {}

    class FakeAnswerAdapter:
        def answer(self, prompt):
            captured["prompt"] = prompt
            return {
                "adapter": "fake",
                "answer": "source fallback answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "select_focus_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=5: [
            {
                "document_id": "doc1",
                "title": "Active Source",
                "source": {"path": str(source_path)},
            }
        ],
    )
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")

    result = answer_query(
        FakeDb(),
        AppConfig(runtime=RuntimeConfig(answer_adapter="fake")),
        "What source evidence is available?",
        session_id="s1",
    )

    assert result["ok"] is True
    assert result["retrieval_status"] == "active_document_source_fallback"
    assert "local source fallback" in captured["prompt"]["context_text"]
    assert captured["prompt"]["context_metadata"]["source_fallback"]["document_id"] == "doc1"


def test_active_document_source_fallback_skips_unreadable_first_document(tmp_path) -> None:
    second_source = tmp_path / "second.md"
    second_source.write_text("Second source evidence survives fallback.", encoding="utf-8")

    envelope = build_active_document_source_fallback_envelope(
        active_documents=[
            {
                "document_id": "first",
                "title": "First Source",
                "source": {"path": str(tmp_path / "missing.md")},
            },
            {
                "document_id": "second",
                "title": "Second Source",
                "source": {"path": str(second_source)},
            },
        ],
        query="What does this document say?",
    )

    assert envelope is not None
    assert "Second source evidence survives fallback" in envelope["context_text"]
    assert envelope["context_metadata"]["source_fallback"]["document_id"] == "second"


def test_active_document_source_fallback_rejects_binary_source(tmp_path) -> None:
    binary_source = tmp_path / "source.pdf"
    binary_source.write_bytes(b"%PDF-1.7\x00\xff\x00\xff")

    envelope = build_active_document_source_fallback_envelope(
        active_documents=[
            {
                "document_id": "binary",
                "title": "Binary Source",
                "source": {"path": str(binary_source)},
            }
        ],
        query="What does this document say?",
    )

    assert envelope is None


def test_active_document_source_fallback_respects_small_prompt_budget(tmp_path) -> None:
    source_path = tmp_path / "small-budget.md"
    source_path.write_text("x" * 1000, encoding="utf-8")

    envelope = build_active_document_source_fallback_envelope(
        active_documents=[
            {
                "document_id": "doc1",
                "title": "Budget Source",
                "source": {"path": str(source_path)},
            }
        ],
        query="What does this document say?",
        token_budget=320,
        reserved_response_tokens=10,
    )

    assert envelope is not None
    assert envelope["context_metadata"]["source_fallback"]["used_chars"] == 40


def test_answer_query_uses_active_document_reference_before_corpus(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": "active answer",
                "used_node_ids": ["active-node"],
            }

    monkeypatch.setattr(
        interaction,
        "select_active_document_focus_node",
        lambda _db, _query, session_id, **_kwargs: "active-node" if session_id == "s1" else None,
    )
    monkeypatch.setattr(interaction, "select_focus_node", lambda *_args, **_kwargs: "corpus-node")
    monkeypatch.setattr(
        interaction,
        "compile_context",
        lambda _db, node_id: {
            "document": {"title": "Active Doc", "document_id": "doc1"},
            "focus_node_id": node_id,
            "records": [
                {
                    "role": "focus",
                    "distance": 0,
                    "title": "Active",
                    "node_id": node_id,
                    "labels": ["source_chunk"],
                    "endorsement_label": "unreviewed",
                    "provenance": {},
                    "text": "Active document text.",
                    "text_preview": "Active document text.",
                }
            ],
        },
    )
    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")

    result = answer_query(
        FakeDb(),
        AppConfig(runtime=RuntimeConfig(answer_adapter="fake")),
        "What does this document say?",
        session_id="s1",
    )

    assert result["focus_node_id"] == "active-node"
    assert result["process_trace"][1]["input"]["selected_node_source"] == "active_document"


def test_answer_query_records_active_document_for_followup_reference(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    document_id = ObjectId()
    node_id = ObjectId()
    db = EndToEndAnswerDb(document_id=document_id, node_id=node_id)
    prompts = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            prompts.append(prompt)
            return {
                "adapter": "fake",
                "answer": "active answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())

    first = answer_query(
        db,
        AppConfig(runtime=RuntimeConfig(answer_adapter="fake")),
        "What does the active evidence say?",
        focus_node_id=str(node_id),
        session_id="v1-smoke",
    )
    followup = answer_query(
        db,
        AppConfig(runtime=RuntimeConfig(answer_adapter="fake")),
        "What does this document say?",
        session_id="v1-smoke",
    )

    assert first["ok"] is True
    assert followup["ok"] is True
    assert first["focus_node_id"] == str(node_id)
    assert followup["focus_node_id"] == str(node_id)
    assert followup["process_trace"][1]["input"]["selected_node_source"] == "active_document"
    assert "Active document evidence." in prompts[1]["context_text"]
    assert db.active_documents.rows[0]["session_id"] == "v1-smoke"
    assert db.active_documents.rows[0]["document_id"] == str(document_id)
    assert db.active_documents.rows[0]["node_ids"] == [str(node_id)]
    assert db.exchanges.rows[0]["active_document_ids"] == [str(document_id)]
    assert db.exchanges.rows[1]["active_document_ids"] == [str(document_id)]


def test_parse_tool_calls_extracts_json() -> None:
    calls = parse_tool_calls(
        '```json\n{"tool_calls":[{"tool":"search_nodes","arguments":{"query":"memory","limit":2}}]}\n```'
    )

    assert calls == [
        {
            "tool": "search_nodes",
            "arguments": {"query": "memory", "limit": 2},
        }
    ]


def test_parse_tool_calls_allows_model_newlines_inside_strings() -> None:
    calls = parse_tool_calls(
        '{"tool_calls":[{"tool":"search_nodes","arguments":{"query":"technical\n design"}}]}'
    )

    assert calls[0]["arguments"]["query"] == "technical\n design"


def test_parse_tool_calls_skips_thinking_json_fragments() -> None:
    calls = parse_tool_calls(
        'Thinking about {"draft": true}.\n'
        '{"tool_calls":[{"tool":"search_nodes","arguments":{"query":"memory"}}]}'
    )

    assert calls == [
        {
            "tool": "search_nodes",
            "arguments": {"query": "memory"},
        }
    ]


def test_parse_tool_calls_scans_invalid_whole_object_response() -> None:
    calls = parse_tool_calls(
        '{"draft": true}\n'
        '{"tool_calls":[{"tool":"search_nodes","arguments":{"query":"memory"}}]}'
    )

    assert calls[0]["arguments"]["query"] == "memory"


def test_agentic_answer_query_runs_planner_tools_then_answer(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    prompts = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            prompts.append(prompt["prompt_text"])
            if len(prompts) == 1:
                return {
                    "adapter": "fake",
                    "answer": '{"status":"continue","tool_calls":[{"tool":"search_nodes","arguments":{"query":"memory"}}]}',
                    "used_node_ids": [],
                }
            if len(prompts) == 2:
                return {
                    "adapter": "fake",
                    "answer": '{"status":"done","tool_calls":[],"compiled_context_notes":"enough","controller_decision":{"action":"use_repository_context","reason":"memory evidence is sufficient","confidence":0.8}}',
                    "used_node_ids": [],
                }
            return {
                "adapter": "fake",
                "answer": "final answer",
                "used_node_ids": ["node1"],
            }

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(
        interaction,
        "execute_tool_calls",
        lambda _db, calls, original_query=None, session_id="default", runtime_config=None: [
            {
                "index": 0,
                "tool": calls[0]["tool"],
                "arguments": calls[0]["arguments"],
                "ok": True,
                "output": [{"node_id": "node1", "title": "Memory"}],
            }
        ],
    )
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="agentic", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory")

    assert result["ok"] is True
    assert result["answer"] == "final answer"
    assert [step["step"] for step in result["process_trace"]] == [
        "user_prompt",
        "memory_agent_iteration",
        "memory_agent_iteration",
        "answer_adapter",
    ]
    assert result["process_trace"][1]["output"]["tool_results"][0]["tool"] == "search_nodes"
    assert result["process_trace"][2]["output"]["stopped"] is True
    controller_decision = result["process_trace"][3]["input"]["controller_decision"]
    assert controller_decision["mode"] == "agentic"
    assert controller_decision["current_owner"] == "memory_agent_controller"
    assert controller_decision["proposed_by"] == "memory_agent"
    assert controller_decision["reason"] == "memory evidence is sufficient"
    assert controller_decision["confidence"] == 0.8
    assert result["activity_report"]["context_construction"]["controller_decision"] == controller_decision
    evidence_summary = result["process_trace"][3]["input"]["evidence_summary"]
    assert evidence_summary == {
        "tool_result_count": 1,
        "successful_tool_count": 1,
        "failed_tool_count": 0,
        "match_count": 1,
        "context_count": 0,
        "record_count": 0,
        "included_node_count": 0,
        "included_node_ids": [],
        "source_documents": [],
    }
    assert result["activity_report"]["context_construction"]["evidence_summary"] == evidence_summary
    assert "Evidence summary: 1 memory-tool result(s), 1 successful and 0 failed." in result["activity_log"]
    assert "Tirzah Tool Results" in prompts[2]


def test_agentic_answer_query_marks_process_run_blocked_when_planning_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    updates = []

    monkeypatch.setattr(
        interaction,
        "run_memory_agent_loop",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("planner broke")),
    )
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="agentic", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory", session_id="s1")

    assert result["ok"] is False
    assert result["reason"] == "agentic_retrieval_failed"
    assert result["process_run_id"] == "run1"
    assert updates[0]["status"] == "blocked"
    assert updates[0]["current_step_id"] == "agentic_retrieval_failed"
    assert updates[0]["exception"]["reason"] == "agentic_retrieval_failed"
    assert updates[0]["exception"]["type"] == "RuntimeError"
    assert "planner broke" in updates[0]["exception"]["note"]


def test_agentic_answer_query_marks_process_run_blocked_when_adapter_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    updates = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            if "You are the Tirzah memory-agent." in prompt["prompt_text"]:
                return {
                    "adapter": "fake",
                    "answer": '{"status":"done","tool_calls":[]}',
                    "used_node_ids": [],
                }
            raise RuntimeError("agentic adapter offline")

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="agentic", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory", session_id="s1")

    assert result["ok"] is False
    assert result["reason"] == "answer_adapter_failed"
    assert result["process_run_id"] == "run1"
    assert updates[0]["status"] == "blocked"
    assert updates[0]["current_step_id"] == "answer_adapter_failed"
    assert updates[0]["exception"]["reason"] == "answer_adapter_failed"
    assert updates[0]["exception"]["type"] == "RuntimeError"
    assert "agentic adapter offline" in updates[0]["exception"]["note"]


def test_agentic_answer_query_marks_process_run_blocked_when_exchange_save_fails(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    updates = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            if "You are the Tirzah memory-agent." in prompt["prompt_text"]:
                return {
                    "adapter": "fake",
                    "answer": '{"status":"done","tool_calls":[]}',
                    "used_node_ids": [],
                }
            return {
                "adapter": "fake",
                "answer": "final answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(
        interaction,
        "save_exchange",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("agentic write failed")),
    )
    monkeypatch.setattr(
        interaction,
        "create_process_run",
        lambda _db, **kwargs: {"run_id": "run1"},
    )
    monkeypatch.setattr(
        interaction,
        "update_process_run",
        lambda _db, run_id, **kwargs: updates.append({"run_id": run_id, **kwargs}),
    )
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="agentic", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory", session_id="s1")

    assert result["ok"] is False
    assert result["reason"] == "answer_save_failed"
    assert result["process_run_id"] == "run1"
    assert result["process_trace"][-1]["step"] == "save_exchange"
    assert updates[0]["status"] == "blocked"
    assert updates[0]["current_step_id"] == "answer_save_failed"
    assert updates[0]["exception"]["reason"] == "answer_save_failed"
    assert updates[0]["exception"]["type"] == "RuntimeError"
    assert "agentic write failed" in updates[0]["exception"]["note"]


def test_agentic_answer_query_falls_back_when_planner_stops_without_tools(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    prompts = []
    executed = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            prompts.append(prompt["prompt_text"])
            if "You are the Tirzah memory-agent." in prompt["prompt_text"]:
                return {
                    "adapter": "fake",
                    "answer": '{"status":"done","tool_calls":[],"compiled_context_notes":"none"}',
                    "used_node_ids": [],
                }
            return {
                "adapter": "fake",
                "answer": "final answer",
                "used_node_ids": ["node1"],
            }

    def fake_execute_tool_calls(_db, calls, original_query=None, session_id="default", runtime_config=None):
        executed.extend(calls)
        assert session_id == "default"
        return [
            {
                "index": 0,
                "tool": calls[0]["tool"],
                "arguments": calls[0]["arguments"],
                "ok": True,
                "output": {
                    "matches": [{"node_id": "node1", "title": "Memory"}],
                    "compiled_contexts": [{"focus_node_id": "node1", "records": []}],
                },
            }
        ]

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "execute_tool_calls", fake_execute_tool_calls)
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    config = AppConfig(
        runtime=RuntimeConfig(
            retrieval_mode="agentic",
            answer_adapter="fake",
            memory_agent_ollama_format=None,
        )
    )

    result = answer_query(FakeDb(), config, "find memory")

    assert result["ok"] is True
    assert executed == [{"tool": "search_nodes", "arguments": {"query": "find memory", "limit": 5}}]
    assert result["process_trace"][1]["output"]["decision"]["fallback_reason"] == (
        "memory_agent_returned_no_initial_tool_calls"
    )
    assert "stop_reason" not in result["process_trace"][1]["output"]
    assert result["process_trace"][2]["output"]["stopped"] is True
    assert [step["step"] for step in result["process_trace"]] == [
        "user_prompt",
        "memory_agent_iteration",
        "memory_agent_iteration",
        "answer_adapter",
    ]
    assert result["answer"] == "final answer"
    assert "Tirzah Tool Results" in prompts[-1]


def test_agentic_answer_query_stops_after_parse_failure_fallback_context(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    prompts = []
    executed = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            prompts.append(prompt["prompt_text"])
            if "You are the Tirzah memory-agent." in prompt["prompt_text"]:
                return {
                    "adapter": "fake",
                    "answer": "not json",
                    "used_node_ids": [],
                }
            return {
                "adapter": "fake",
                "answer": "final answer",
                "used_node_ids": ["node1"],
            }

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    def fake_execute_tool_calls(_db, calls, original_query=None, session_id="default", runtime_config=None):
        executed.extend(calls)
        return [
            {
                "index": 0,
                "tool": calls[0]["tool"],
                "arguments": calls[0]["arguments"],
                "ok": True,
                "output": {
                    "matches": [{"node_id": "node1", "title": "Memory"}],
                    "compiled_contexts": [{"focus_node_id": "node1", "records": []}],
                },
            }
        ]

    monkeypatch.setattr(interaction, "execute_tool_calls", fake_execute_tool_calls)
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="agentic", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory")

    assert result["ok"] is True
    assert result["answer"] == "final answer"
    assert executed == [{"tool": "search_nodes", "arguments": {"query": "find memory", "limit": 5}}]
    assert len(prompts) == 2
    assert [step["step"] for step in result["process_trace"]] == [
        "user_prompt",
        "memory_agent_iteration",
        "answer_adapter",
    ]
    assert result["process_trace"][1]["output"]["ok"] is True
    assert result["process_trace"][1]["output"]["decision"]["fallback_reason"] == (
        "memory_agent_decision_failed"
    )
    assert result["process_trace"][1]["output"]["stop_reason"] == "fallback_context_gathered"
    assert "Tirzah Tool Results" in prompts[-1]


def test_agentic_answer_query_marks_unavailable_fallback_context(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            if "You are the Tirzah memory-agent." in prompt["prompt_text"]:
                return {
                    "adapter": "fake",
                    "answer": "not json",
                    "used_node_ids": [],
                }
            return {
                "adapter": "fake",
                "answer": "final answer",
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(interaction, "execute_tool_calls", lambda *args, **kwargs: [])
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="agentic", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory")

    assert result["ok"] is True
    assert result["retrieval_status"] == "agentic_no_tool_context"
    assert result["process_trace"][1]["output"]["ok"] is False
    assert result["process_trace"][1]["output"]["stop_reason"] == "fallback_context_unavailable"


def test_agentic_answer_query_stops_when_planner_fails_after_context(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    prompts = []

    class FakeAnswerAdapter:
        def answer(self, prompt):
            prompts.append(prompt["prompt_text"])
            if len(prompts) == 1:
                return {
                    "adapter": "fake",
                    "answer": '{"status":"continue","tool_calls":[{"tool":"search_nodes","arguments":{"query":"memory"}}]}',
                    "used_node_ids": [],
                }
            if "You are the Tirzah memory-agent." in prompt["prompt_text"]:
                return {
                    "adapter": "fake",
                    "answer": "not json",
                    "used_node_ids": [],
                }
            return {
                "adapter": "fake",
                "answer": "final answer",
                "used_node_ids": ["node1"],
            }

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    monkeypatch.setattr(
        interaction,
        "execute_tool_calls",
        lambda _db, calls, original_query=None, session_id="default", runtime_config=None: [
            {
                "index": 0,
                "tool": calls[0]["tool"],
                "arguments": calls[0]["arguments"],
                "ok": True,
                "output": {
                    "matches": [{"node_id": "node1", "title": "Memory"}],
                    "compiled_contexts": [{"focus_node_id": "node1", "records": []}],
                },
            }
        ],
    )
    monkeypatch.setattr(interaction, "save_exchange", lambda *args, **kwargs: "exchange1")
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="agentic", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory")

    assert result["ok"] is True
    assert result["answer"] == "final answer"
    assert [step["step"] for step in result["process_trace"]] == [
        "user_prompt",
        "memory_agent_iteration",
        "memory_agent_iteration",
        "answer_adapter",
    ]
    assert result["process_trace"][2]["output"]["ok"] is False
    assert result["process_trace"][2]["output"]["decision"]["fallback_reason"] == (
        "memory_agent_failed_after_tool_context"
    )
    assert result["process_trace"][2]["output"]["stop_reason"] == (
        "memory_agent_failed_after_tool_context"
    )
    assert "Tirzah Tool Results" in prompts[-1]


def test_parse_memory_agent_decision_accepts_done_status() -> None:
    decision = parse_memory_agent_decision(
        '{"status":"done","tool_calls":[],"compiled_context_notes":"sufficient"}'
    )

    assert decision == {
        "status": "done",
        "tool_calls": [],
        "compiled_context_notes": "sufficient",
        "controller_decision": None,
        "context_proposal": None,
    }


def test_parse_memory_agent_decision_accepts_context_proposal() -> None:
    decision = parse_memory_agent_decision(
        '{"status":"done","tool_calls":[],"context_proposal":{"selected_node_ids":["node2"],"rationale":"best evidence","organization":["definition first"]}}'
    )

    assert decision["context_proposal"] == {
        "selected_node_ids": ["node2"],
        "rationale": "best evidence",
        "organization": ["definition first"],
    }


def test_parse_memory_agent_decision_accepts_controller_decision() -> None:
    decision = parse_memory_agent_decision(
        '{"status":"done","tool_calls":[],"controller_decision":{"action":"use_repository_context","reason":"good evidence","confidence":1.7}}'
    )

    assert decision["controller_decision"] == {
        "action": "use_repository_context",
        "reason": "good evidence",
        "confidence": 1.0,
    }


def test_parse_memory_agent_decision_accepts_active_document_tool() -> None:
    decision = parse_memory_agent_decision(
        '{"status":"continue","tool_calls":[{"tool":"list_active_documents","arguments":{"limit":3}}]}'
    )

    assert decision["tool_calls"] == [
        {"tool": "list_active_documents", "arguments": {"limit": 3}}
    ]


def test_parse_memory_agent_decision_accepts_exact_lookup_tools() -> None:
    decision = parse_memory_agent_decision(
        '{"status":"continue","tool_calls":['
        '{"tool":"get_document","arguments":{"document_id":"doc1"}},'
        '{"tool":"get_document_tree","arguments":{"document_id":"doc1"}},'
        '{"tool":"get_node_context","arguments":{"node_id":"node1"}}'
        "]}"
    )

    assert [call["tool"] for call in decision["tool_calls"]] == [
        "get_document",
        "get_document_tree",
        "get_node_context",
    ]


def test_parse_memory_agent_decision_accepts_graph_edge_tool() -> None:
    decision = parse_memory_agent_decision(
        '{"status":"continue","tool_calls":['
        '{"tool":"expand_proximity","arguments":{"node_id":"node1","direction":"outgoing"}},'
        '{"tool":"expand_graph_paths","arguments":{"node_id":"node1","max_depth":2}},'
        '{"tool":"semantic_candidates","arguments":{"node_id":"node1"}}'
        "]}"
    )

    assert decision["tool_calls"] == [
        {
            "tool": "expand_proximity",
            "arguments": {"node_id": "node1", "direction": "outgoing"},
        },
        {
            "tool": "expand_graph_paths",
            "arguments": {"node_id": "node1", "max_depth": 2},
        },
        {
            "tool": "semantic_candidates",
            "arguments": {"node_id": "node1"},
        },
    ]


def test_memory_agent_runtime_can_differ_from_answer_runtime() -> None:
    runtime = RuntimeConfig(
        answer_adapter="ollama_cli",
        ollama_model="final",
        memory_agent_adapter="ollama_cli",
        memory_agent_model="memory",
        memory_agent_ollama_format="json",
    )

    memory_runtime = memory_agent_runtime_config(runtime)

    assert memory_runtime.answer_adapter == "ollama_cli"
    assert memory_runtime.ollama_model == "memory"
    assert memory_runtime.ollama_format == "json"
    assert runtime.answer_adapter == "ollama_cli"
    assert runtime.ollama_model == "final"
    assert runtime.ollama_format is None


def test_memory_agent_runtime_rejects_explicit_http_adapter() -> None:
    runtime = RuntimeConfig(
        answer_adapter="ollama_cli",
        memory_agent_adapter="ollama_http",
    )

    try:
        memory_agent_runtime_config(runtime)
    except ValueError as error:
        assert "HTTP model adapters are not allowed" in str(error)
    else:
        raise AssertionError("Expected HTTP memory-agent adapter to be rejected.")


def test_memory_agent_runtime_does_not_inherit_http_answer_adapter() -> None:
    runtime = RuntimeConfig(
        answer_adapter="ollama_http",
        memory_agent_adapter=None,
        ollama_model="final",
        memory_agent_model="memory",
    )

    memory_runtime = memory_agent_runtime_config(runtime)

    assert memory_runtime.answer_adapter == "ollama_cli"
    assert memory_runtime.ollama_model == "memory"


def test_memory_agent_trace_records_runtime_json_controls(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    class FakeAnswerAdapter:
        def answer(self, prompt):
            return {
                "adapter": "fake",
                "answer": '{"status":"done","tool_calls":[]}',
                "used_node_ids": [],
            }

    monkeypatch.setattr(interaction, "answer_adapter", lambda _config: FakeAnswerAdapter())
    process_trace = []
    runtime = RuntimeConfig(
        answer_adapter="fake",
        memory_agent_adapter="fake",
        memory_agent_model="planner",
        memory_agent_ollama_format="json",
        ollama_think=False,
        ollama_hide_thinking=True,
    )

    run_memory_agent_loop(
        FakeDb(),
        runtime,
        query="find memory",
        focus_node_id=None,
        session_id="s1",
        max_iterations=1,
        process_trace=process_trace,
    )

    step_input = process_trace[0]["input"]
    assert step_input["adapter"] == "fake"
    assert step_input["model"] == "planner"
    assert step_input["format"] == "json"
    assert step_input["think"] is False
    assert step_input["hide_thinking"] is True


def test_build_memory_agent_prompt_includes_query_assembly_guidance() -> None:
    prompt = build_memory_agent_prompt(
        query="What does the Tirzah technical design say the system is for?",
        focus_node_id=None,
        session_id="design-session",
        active_documents=[
            {
                "document_id": "doc1",
                "title": "Tirzah Technical Design",
            }
        ],
        history=[],
    )

    assert "Query assembly:" in prompt
    assert "- Lexical terms: Tirzah, technical, design, system" in prompt
    assert "- Exact phrases: Tirzah technical, technical design, design system" in prompt
    assert "- Named anchors: Tirzah" in prompt
    assert "- Near-match terms: none" in prompt
    assert "- Suggested fallback searches: Tirzah technical" in prompt
    assert "session_id: design-session" in prompt
    assert "Tirzah Technical Design" in prompt


def test_build_memory_agent_prompt_includes_active_identity_summary() -> None:
    prompt = build_memory_agent_prompt(
        query="Find memory governance context",
        focus_node_id=None,
        session_id="governance-session",
        active_documents=[],
        active_identities=[
            {
                "identity_id": "tirzah_shared",
                "title": "Tirzah Shared Identity",
                "kind": "shared",
                "description": "Project-wide retrieval and governance scaffold.",
                "trusted_labels": ["memory_reference"],
                "excluded_labels": ["restricted"],
                "allowed_relation_types": ["related_to"],
                "excluded_relation_types": ["hidden_from_testing_agent"],
                "weighting_profile_id": "default_balanced",
                "required_process_ids": ["review_before_write"],
                "governance_policy_ids": ["read_only_memory_agent"],
                "ignored_large_field": "not included",
            }
        ],
        history=[],
    )

    assert "Active identities:" in prompt
    assert "tirzah_shared" in prompt
    assert "default_balanced" in prompt
    assert "read_only_memory_agent" in prompt
    assert "ignored_large_field" not in prompt


def test_memory_agent_tool_summary_includes_search_diagnostics() -> None:
    summary = summarize_tool_results_for_memory_agent(
        [
            {
                "tool": "search_nodes",
                "arguments": {"query": "technical design"},
                "ok": True,
                "output": {
                    "matches": [
                        {
                            "node_id": "node1",
                            "title": "System Name",
                            "labels": ["source_section"],
                            "text_preview": "Tirzah is a memory layer.",
                        }
                    ]
                },
                "details": {
                    "query_assembly": {
                        "lexical_terms": ["technical", "design"],
                        "exact_phrases": ["technical design"],
                        "anchor_terms": ["Tirzah"],
                        "near_match_terms": [],
                    },
                    "fallback_queries": [
                        {"query": "technical design", "result_count": 3},
                        {"query": "design", "result_count": 10},
                    ],
                },
            }
        ]
    )

    assert summary[0]["query_assembly"] == {
        "lexical_terms": ["technical", "design"],
        "exact_phrases": ["technical design"],
        "anchor_terms": ["Tirzah"],
        "near_match_terms": [],
    }
    assert summary[0]["fallback_queries"] == [
        {"query": "technical design", "result_count": 3},
        {"query": "design", "result_count": 10},
    ]


def test_memory_agent_tool_summary_includes_trust_diagnostic() -> None:
    diagnostic = {
        "score": 0.68,
        "components": {"trust": 0.55, "temporal": 1.0},
        "weighting_profile_id": "default_balanced",
        "signals": {"endorsement_label": "unreviewed"},
    }

    summary = summarize_tool_results_for_memory_agent(
        [
            {
                "tool": "search_nodes",
                "arguments": {"query": "memory"},
                "ok": True,
                "output": {
                    "matches": [
                        {
                            "node_id": "node1",
                            "title": "System Name",
                            "labels": ["source_section"],
                            "text_preview": "Tirzah is a memory layer.",
                            "trust_diagnostic": diagnostic,
                        }
                    ]
                },
                "details": {},
            }
        ]
    )

    assert summary[0]["top_matches"][0]["trust_diagnostic"] == diagnostic


def test_memory_agent_tool_summary_includes_proximity_matches() -> None:
    summary = summarize_tool_results_for_memory_agent(
        [
            {
                "tool": "expand_proximity",
                "arguments": {"node_id": "node1", "direction": "outgoing"},
                "ok": True,
                "output": {
                    "matches": [
                        {
                            "node_id": "near1",
                            "title": "Nearby Node",
                            "labels": ["source_section"],
                            "text_preview": "A nearby graph node.",
                            "proximity_score": 0.72,
                            "edge": {
                                "relation_type": "supports",
                                "weight": 0.9,
                                "confidence": 0.8,
                                "source_node_id": "node1",
                                "target_node_id": "near1",
                                "provenance_source": "semantic_candidate_review",
                                "reviewer": "tester",
                                "shared_label_count": 2,
                                "candidate_source": "embedding_similarity",
                                "embedding_similarity": 0.91,
                                "embedding_model": "mock",
                                "embedding_dimensions": 16,
                                "selection_min_similarity": 0.8,
                            },
                        }
                    ],
                    "compiled_contexts": [{"focus_node_id": "near1", "records": []}],
                },
            }
        ]
    )

    assert summary[0]["match_count"] == 1
    assert summary[0]["top_matches"] == [
        {
            "node_id": "near1",
            "title": "Nearby Node",
            "labels": ["source_section"],
            "text_preview": "A nearby graph node.",
            "proximity_score": 0.72,
            "edge": {
                "relation_type": "supports",
                "weight": 0.9,
                "confidence": 0.8,
                "provenance_source": "semantic_candidate_review",
                "reviewer": "tester",
                "shared_label_count": 2,
                "candidate_source": "embedding_similarity",
                "embedding_similarity": 0.91,
                "embedding_model": "mock",
                "embedding_dimensions": 16,
                "selection_min_similarity": 0.8,
            },
        }
    ]


def test_memory_agent_tool_summary_includes_graph_path_matches() -> None:
    summary = summarize_tool_results_for_memory_agent(
        [
            {
                "tool": "expand_graph_paths",
                "arguments": {"node_id": "node1", "max_depth": 2},
                "ok": True,
                "output": {
                    "matches": [
                        {
                            "node_id": "near2",
                            "title": "Two Hop Node",
                            "labels": ["source_section"],
                            "text_preview": "A two-hop graph node.",
                            "path_score": 0.5184,
                            "path_depth": 2,
                            "path_edges": [
                                {
                                    "relation_type": "supports",
                                    "weight": 0.9,
                                    "confidence": 0.9,
                                    "provenance_source": "semantic_candidate_review",
                                    "reviewer": "tester",
                                    "shared_label_count": 1,
                                },
                                {"relation_type": "extends", "weight": 0.8, "confidence": 0.8},
                            ],
                        }
                    ],
                    "compiled_contexts": [{"focus_node_id": "near2", "records": []}],
                },
            }
        ]
    )

    assert summary[0]["match_count"] == 1
    assert summary[0]["top_matches"] == [
        {
            "node_id": "near2",
            "title": "Two Hop Node",
            "labels": ["source_section"],
            "text_preview": "A two-hop graph node.",
            "path_score": 0.5184,
            "path_depth": 2,
            "path_edges": [
                {
                    "relation_type": "supports",
                    "weight": 0.9,
                    "confidence": 0.9,
                    "provenance_source": "semantic_candidate_review",
                    "reviewer": "tester",
                    "shared_label_count": 1,
                },
                {"relation_type": "extends", "weight": 0.8, "confidence": 0.8},
            ],
        }
    ]


def test_memory_agent_tool_summary_includes_semantic_candidates() -> None:
    summary = summarize_tool_results_for_memory_agent(
        [
            {
                "tool": "semantic_candidates",
                "arguments": {"node_id": "node1"},
                "ok": True,
                "output": {
                    "matches": [
                        {
                            "node_id": "near3",
                            "title": "Label Match",
                            "labels": ["source_section", "taj_mahal"],
                            "text_preview": "A label-overlap candidate.",
                            "shared_labels": ["taj_mahal"],
                            "shared_label_count": 1,
                        }
                    ],
                    "compiled_contexts": [{"focus_node_id": "near3", "records": []}],
                },
            }
        ]
    )

    assert summary[0]["match_count"] == 1
    assert summary[0]["top_matches"] == [
        {
            "node_id": "near3",
            "title": "Label Match",
            "labels": ["source_section", "taj_mahal"],
            "text_preview": "A label-overlap candidate.",
            "shared_labels": ["taj_mahal"],
            "shared_label_count": 1,
        }
    ]


def test_memory_agent_tool_summary_omits_missing_search_diagnostics() -> None:
    summary = summarize_tool_results_for_memory_agent(
        [
            {
                "tool": "search_nodes",
                "arguments": {"query": "memory"},
                "ok": True,
                "output": {"matches": []},
            }
        ]
    )

    assert "query_assembly" not in summary[0]
    assert "fallback_queries" not in summary[0]


def test_memory_agent_tool_summary_omits_empty_query_assembly() -> None:
    summary = summarize_tool_results_for_memory_agent(
        [
            {
                "tool": "search_nodes",
                "arguments": {"query": "what does the"},
                "ok": True,
                "output": {"matches": []},
                "details": {
                    "query_assembly": {
                        "lexical_terms": [],
                        "exact_phrases": [],
                        "anchor_terms": [],
                    }
                },
            }
        ]
    )

    assert "query_assembly" not in summary[0]


def test_compact_fallback_query_details_skips_missing_queries_and_caps() -> None:
    items = [{"result_count": 99}]
    items.extend({"query": f"q{index}", "result_count": index} for index in range(7))

    assert compact_fallback_query_details(items) == [
        {"query": "q0", "result_count": 0},
        {"query": "q1", "result_count": 1},
        {"query": "q2", "result_count": 2},
        {"query": "q3", "result_count": 3},
        {"query": "q4", "result_count": 4},
    ]


def test_execute_search_nodes_tool_falls_back_to_terms(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        calls.append(query)
        if query == "Tirzah":
            return [{"node_id": "node1", "title": "Tirzah"}]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)

    output, details = execute_search_nodes_tool(
        FakeDb(),
        query="technical desig\ndesign Tirzah",
    )

    assert output["matches"] == [{"node_id": "node1", "title": "Tirzah"}]
    assert output["compiled_contexts"] == []
    assert calls[0] == "technical desig design Tirzah"
    assert "desig design" in details["query_assembly"]["exact_phrases"]
    assert any(item["query"] == "Tirzah" for item in details["fallback_queries"])


def test_execute_search_nodes_tool_passes_active_identity(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    identities = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, identity=None, **kwargs):
        identities.append(identity)
        if identity:
            return [{"node_id": "node1", "title": "Allowed"}]
        return [
            {"node_id": "node1", "title": "Allowed", "labels": ["public"]},
            {"node_id": "node2", "title": "Restricted", "labels": ["restricted"]},
        ]

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)
    monkeypatch.setattr(
        interaction,
        "first_active_agent_identity",
        lambda _db: {"identity_id": "restricted_agent", "excluded_labels": ["restricted"]},
    )

    output, details = execute_search_nodes_tool(FakeDb(), query="memory", session_id="s1")

    assert output["matches"] == [{"node_id": "node1", "title": "Allowed"}]
    assert identities[0] is None
    assert identities[1]["identity_id"] == "restricted_agent"
    assert details["active_identity_id"] == "restricted_agent"
    assert details["identity_excluded_count"] == 1
    assert details["identity_exclusion_sample_size"] == 2


def test_execute_search_nodes_tool_adds_trust_diagnostics(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, identity=None, **kwargs):
        return [{"node_id": "node1", "title": "Memory", "labels": ["source_chunk"]}]

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)
    monkeypatch.setattr(
        interaction,
        "trust_temporal_diagnostics_for_nodes",
        lambda _db, node_ids, weighting_profile_id=None: {
            node_ids[0]: {
                "weighting_profile": {"weighting_profile_id": weighting_profile_id},
                "diagnostic": {
                    "score": 0.68,
                    "components": {"trust": 0.55, "temporal": 1.0},
                    "signals": {"endorsement_label": "unreviewed", "usage_score": 2},
                },
            },
        },
    )

    output, details = execute_search_nodes_tool(FakeDb(), query="memory")

    assert output["matches"][0]["trust_diagnostic"]["score"] == 0.68
    assert details["trust_diagnostics"][0]["node_id"] == "node1"
    assert details["trust_diagnostics"][0]["components"]["trust"] == 0.55


def test_execute_search_nodes_tool_batches_trust_diagnostics(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, identity=None, **kwargs):
        return [
            {"node_id": "node1", "title": "Memory 1", "labels": ["source_chunk"]},
            {"node_id": "node2", "title": "Memory 2", "labels": ["source_chunk"]},
        ]

    def fake_trust_diagnostics(_db, node_ids, weighting_profile_id=None):
        calls.append((list(node_ids), weighting_profile_id))
        return {
            node_id: {
                "weighting_profile": {"weighting_profile_id": weighting_profile_id},
                "diagnostic": {
                    "score": 0.68,
                    "components": {"trust": 0.55},
                    "signals": {"endorsement_label": "unreviewed"},
                },
            }
            for node_id in node_ids
        }

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)
    monkeypatch.setattr(interaction, "trust_temporal_diagnostics_for_nodes", fake_trust_diagnostics)

    output, details = execute_search_nodes_tool(FakeDb(), query="memory")

    assert calls == [(["node1", "node2"], None)]
    assert len(output["matches"]) == 2
    assert [item["node_id"] for item in details["trust_diagnostics"]] == ["node1", "node2"]


def test_execute_tool_calls_can_list_active_documents(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=5: [
            {
                "session_id": session_id,
                "document_id": "doc1",
                "title": "Active Doc",
            }
        ][:limit],
    )

    results = interaction.execute_tool_calls(
        FakeDb(),
        [{"tool": "list_active_documents", "arguments": {"limit": 1}}],
        session_id="design",
    )

    assert results[0]["ok"] is True
    assert results[0]["output"] == [
        {
            "session_id": "design",
            "document_id": "doc1",
            "title": "Active Doc",
        }
    ]


def test_execute_tool_calls_can_run_exact_lookup_tools(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    monkeypatch.setattr(
        interaction,
        "get_document",
        lambda _db, document_id: {"document_id": document_id, "title": "Design"},
    )
    monkeypatch.setattr(
        interaction,
        "document_tree",
        lambda _db, document_id: [{"document_id": document_id, "node_id": "node1"}],
    )
    monkeypatch.setattr(
        interaction,
        "node_context",
        lambda _db, node_id, child_limit=5: {
            "focus_node_id": node_id,
            "node": {"node_id": node_id, "title": "Focus"},
            "children": [{"node_id": "child1"}],
            "child_limit": child_limit,
        },
    )
    monkeypatch.setattr(
        interaction,
        "graph_edges_for_node",
        lambda _db, node_id, direction="both", relation_type=None, limit=5: [
            {
                "source_node_id": node_id,
                "target_node_id": "node2",
                "relation_type": relation_type or "supports",
                "direction": direction,
                "limit": limit,
            }
        ],
    )
    monkeypatch.setattr(
        interaction,
        "expand_proximity",
        lambda _db, node_id, direction="both", relation_type=None, limit=5: [
            {
                "node_id": "node2",
                "title": "Adjacent",
                "proximity_score": 0.9,
                "direction": direction,
                "limit": limit,
            }
        ],
    )
    def fake_expand_graph_paths(
        _db,
        node_id,
        direction="both",
        relation_type=None,
        max_depth=2,
        branch_limit=5,
        limit=5,
    ):
        return [
            {
                "node_id": "node3",
                "title": "Path Adjacent",
                "path_score": 0.81,
                "path_depth": max_depth,
                "direction": direction,
                "branch_limit": branch_limit,
                "limit": limit,
            }
        ]

    monkeypatch.setattr(interaction, "expand_graph_paths", fake_expand_graph_paths)
    monkeypatch.setattr(
        interaction,
        "semantic_candidate_nodes",
        lambda _db, node_id, include_same_document=False, limit=5: [
            {
                "node_id": "node4",
                "title": "Semantic Adjacent",
                "shared_labels": ["topic"],
                "shared_label_count": 1,
                "include_same_document": include_same_document,
                "limit": limit,
            }
        ],
    )
    monkeypatch.setattr(
        interaction,
        "compile_context",
        lambda _db, node_id: {"focus_node_id": node_id, "records": []},
    )

    results = interaction.execute_tool_calls(
        FakeDb(),
        [
            {"tool": "get_document", "arguments": {"document_id": "doc1"}},
            {"tool": "get_document_tree", "arguments": {"document_id": "doc1"}},
            {"tool": "get_node_context", "arguments": {"node_id": "node1", "child_limit": 99}},
            {
                "tool": "get_graph_edges",
                "arguments": {
                    "node_id": "node1",
                    "direction": "sideways",
                    "relation_type": "supports",
                    "limit": 99,
                },
            },
            {
                "tool": "expand_proximity",
                "arguments": {
                    "node_id": "node1",
                    "direction": "outgoing",
                    "limit": 99,
                },
            },
            {
                "tool": "expand_graph_paths",
                "arguments": {
                    "node_id": "node1",
                    "direction": "outgoing",
                    "max_depth": 99,
                    "branch_limit": 99,
                    "limit": 99,
                },
            },
            {
                "tool": "semantic_candidates",
                "arguments": {
                    "node_id": "node1",
                    "include_same_document": True,
                    "limit": 99,
                },
            },
        ],
    )

    assert [result["ok"] for result in results] == [True, True, True, True, True, True, True]
    assert results[0]["output"]["title"] == "Design"
    assert results[1]["output"] == [{"document_id": "doc1", "node_id": "node1"}]
    assert results[2]["output"]["child_limit"] == 10
    assert results[3]["output"] == [
        {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "relation_type": "supports",
            "direction": "both",
            "limit": 10,
        }
    ]
    assert results[4]["output"] == {
        "matches": [
            {
                "node_id": "node2",
                "title": "Adjacent",
                "proximity_score": 0.9,
                "direction": "outgoing",
                "limit": 10,
            }
        ],
        "compiled_contexts": [{"focus_node_id": "node2", "records": []}],
    }
    assert results[5]["output"] == {
        "matches": [
            {
                "node_id": "node3",
                "title": "Path Adjacent",
                "path_score": 0.81,
                "path_depth": 3,
                "direction": "outgoing",
                "branch_limit": 10,
                "limit": 10,
            }
        ],
        "compiled_contexts": [{"focus_node_id": "node3", "records": []}],
    }
    assert results[6]["output"] == {
        "matches": [
            {
                "node_id": "node4",
                "title": "Semantic Adjacent",
                "shared_labels": ["topic"],
                "shared_label_count": 1,
                "include_same_document": True,
                "limit": 10,
            }
        ],
        "compiled_contexts": [{"focus_node_id": "node4", "records": []}],
    }


def test_execute_tool_calls_returns_instructional_tool_errors() -> None:
    import tirzah.sessions.interaction as interaction

    results = interaction.execute_tool_calls(
        FakeDb(),
        [
            {"tool": "compile_context", "arguments": {}},
            {"tool": "not_a_tool", "arguments": {}},
        ],
    )

    assert results[0]["ok"] is False
    assert results[0]["error"] == "compile_context requires node_id."
    assert "node_id" in results[0]["usage"]
    assert "Do not repeat" in results[0]["repair_instruction"]
    assert results[1]["ok"] is False
    assert "Unsupported tool" in results[1]["error"]
    assert "search_nodes" in results[1]["usage"]


def test_memory_agent_history_keeps_tool_repair_guidance() -> None:
    import tirzah.sessions.interaction as interaction

    results = interaction.execute_tool_calls(
        FakeDb(),
        [{"tool": "compile_context", "arguments": {}}],
    )

    summary = summarize_tool_results_for_memory_agent(results)
    assert summary[0]["error"] == "compile_context requires node_id."
    assert "node_id" in summary[0]["usage"]
    assert "Do not repeat" in summary[0]["repair_instruction"]

    prompt = build_memory_agent_prompt(
        query="find memory",
        focus_node_id=None,
        session_id="web",
        active_documents=[],
        history=[{"iteration": 1, "tool_results": summary}],
    )

    assert "Tool repair guidance from prior iterations:" in prompt
    assert "compile_context requires node_id" in prompt
    assert "Revise the next tool call" in prompt


def test_answer_activity_report_summarizes_tool_repair_guidance() -> None:
    from tirzah.sessions.activity_reports import answer_activity_log, answer_activity_report

    result = {
        "ok": True,
        "query": "find memory",
        "adapter": "mock",
        "model": "mock",
        "answer": "answer",
        "process_trace": [
            {
                "step": "user_prompt",
                "input": {"query": "find memory", "session_id": "web"},
                "output": {},
            },
            {
                "step": "memory_agent_iteration",
                "input": {"iteration": 1, "adapter": "mock", "model": "mock"},
                "output": {
                    "ok": True,
                    "tool_results": [
                        {
                            "tool": "compile_context",
                            "arguments": {},
                            "ok": False,
                            "error": "compile_context requires node_id.",
                            "usage": "Call compile_context with node_id.",
                            "repair_instruction": "Revise the next tool call.",
                        }
                    ],
                },
            },
            {
                "step": "answer_adapter",
                "input": {"adapter": "mock", "model": "mock"},
                "output": {"ok": True, "answer": "answer", "used_node_ids": []},
            },
        ],
    }

    report = answer_activity_report(result)
    assert report["tool_repair_guidance"][0]["tool"] == "compile_context"
    assert "node_id" in report["tool_repair_guidance"][0]["usage"]
    log = answer_activity_log(report)
    assert "Tool-call repair guidance:" in log
    assert "compile_context requires node_id" in log


def test_evidence_summary_lines_render_source_documents_concisely() -> None:
    lines = evidence_summary_lines(
        {
            "included_node_count": 2,
            "source_fallback_used": True,
            "tool_result_count": 4,
            "successful_tool_count": 3,
            "failed_tool_count": 1,
            "vector_edge_evidence_count": 2,
            "source_documents": [
                {"title": "Alpha"},
                {"document_id": "doc-beta"},
                {},
                {"title": "Delta"},
            ],
        }
    )

    assert lines == [
        "- Evidence summary: 2 repository node(s) included.",
        "- Evidence summary: source-file fallback was used.",
        "- Evidence summary: 4 memory-tool result(s), 3 successful and 1 failed.",
        "- Evidence summary: 2 vector-derived reviewed edge(s) influenced retrieved context.",
        "- Evidence sources: Alpha, doc-beta, untitled source, plus 1 more.",
    ]


def test_evidence_summary_lines_can_suppress_direct_duplicates() -> None:
    lines = evidence_summary_lines(
        {
            "included_node_count": 0,
            "source_fallback_used": True,
            "source_documents": [{"title": "Active Source"}],
        },
        include_included_nodes=False,
        include_source_fallback=False,
        include_source_documents=False,
    )

    assert lines == []


def test_agentic_evidence_summary_counts_vector_edge_evidence() -> None:
    summary = agentic_evidence_summary(
        [
            {
                "tool": "expand_proximity",
                "ok": True,
                "output": {
                    "match_count": 1,
                    "matches": [
                        {
                            "node_id": "near1",
                            "edge": {
                                "candidate_source": "embedding_similarity",
                                "embedding_similarity": 0.91,
                            },
                        }
                    ],
                    "top_contexts": [],
                },
            },
            {
                "tool": "expand_graph_paths",
                "ok": True,
                "output": {
                    "match_count": 1,
                    "matches": [
                        {
                            "node_id": "near2",
                            "path_edges": [
                                {"candidate_source": "label_overlap"},
                                {"candidate_source": "embedding_similarity"},
                            ],
                        }
                    ],
                    "top_contexts": [],
                },
            },
        ]
    )

    assert summary["vector_edge_evidence_count"] == 2


def test_document_tree_lookup_does_not_mark_tree_nodes_as_used() -> None:
    included = included_nodes_from_tool_results(
        [
            {
                "tool": "get_document_tree",
                "ok": True,
                "output": [{"node_id": "node1", "title": "Tree Node"}],
            },
            {
                "tool": "get_node_context",
                "ok": True,
                "output": {"focus_node_id": "node2", "node": {"node_id": "node2"}},
            },
        ]
    )

    assert [record["node_id"] for record in included] == ["node2"]


def test_active_document_vocabulary_values_extracts_titles_sources_and_labels() -> None:
    assert active_document_vocabulary_values(
        [
            {
                "title": "Technical Design",
                "source": {"path": "/tmp/Tirzah_Technical_Design.md"},
                "labels": ["tirzah_design"],
            }
        ]
    ) == [
        "Technical Design",
        "/tmp/Tirzah_Technical_Design.md",
        "tirzah_design",
    ]


def test_memory_agent_prompt_uses_active_documents_for_near_match_guidance() -> None:
    prompt = build_memory_agent_prompt(
        query="tecnical desgin",
        focus_node_id=None,
        session_id="s1",
        active_documents=[
            {
                "title": "Technical Design",
                "source": {"path": "/tmp/design.md"},
                "labels": [],
            }
        ],
        history=[],
    )

    assert "Near-match terms: tecnical->Technical" in prompt
    assert "desgin->Design" in prompt


def test_execute_search_nodes_tool_uses_session_active_documents_for_near_match_terms(
    monkeypatch,
) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        calls.append(query)
        if query == "Technical":
            return [{"node_id": "node1", "title": "Technical Design"}]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)
    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=20: [
            {
                "session_id": session_id,
                "title": "Technical Design",
                "source": {"path": "/tmp/design.md"},
                "labels": [],
            }
        ],
    )

    output, details = execute_search_nodes_tool(FakeVocabularyDb(), query="tecnical", session_id="s1")

    assert output["matches"] == [{"node_id": "node1", "title": "Technical Design"}]
    assert calls[:2] == ["tecnical", "Technical"]
    assert details["query_assembly"]["near_match_terms"][0]["candidate_term"] == "Technical"


def test_near_match_vocabulary_can_use_active_documents_without_document_collection(
    monkeypatch,
) -> None:
    import tirzah.sessions.interaction as interaction

    monkeypatch.setattr(
        interaction,
        "list_active_documents",
        lambda _db, session_id, limit=20: [
            {
                "session_id": session_id,
                "title": "Session Technical Notes",
                "source": {},
                "labels": [],
            }
        ],
    )

    assert near_match_vocabulary(FakeDb(), session_id="s1")[:3] == [
        "Session",
        "Technical",
        "Notes",
    ]


def test_execute_search_nodes_tool_uses_near_match_terms_after_empty_search(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        calls.append(query)
        if query == "technical":
            return [{"node_id": "node1", "title": "Technical Design"}]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)
    monkeypatch.setattr(interaction, "near_match_vocabulary", lambda _db: ["technical"])

    output, details = execute_search_nodes_tool(FakeDb(), query="tecnical")

    assert output["matches"] == [{"node_id": "node1", "title": "Technical Design"}]
    assert {"source_term": "tecnical", "candidate_term": "technical", "score": 0.94, "reason": "near_token_match"} in details[
        "query_assembly"
    ]["near_match_terms"]
    assert "technical" in calls


def test_execute_search_nodes_tool_uses_near_match_terms_after_weak_match(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, identity=None, **kwargs):
        calls.append(query)
        if query == "tecnical":
            return [{"node_id": "weak", "title": "Reference", "text_preview": "Generic note."}]
        if query == "technical":
            return [{"node_id": "strong", "title": "Technical Design"}]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)
    monkeypatch.setattr(interaction, "near_match_vocabulary", lambda _db: ["technical"])

    output, details = execute_search_nodes_tool(FakeDb(), query="tecnical")

    assert output["matches"][0]["node_id"] == "strong"
    assert details["fallback_trigger"] == "weak_matches"
    assert details["weak_match_score_threshold"] == 5
    assert "technical" in calls


def test_execute_search_nodes_tool_uses_original_query_for_intent_terms(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        if query == "system":
            return [
                {
                    "node_id": "generic",
                    "title": "Tirzah Technical Design Document",
                    "text_preview": "Header",
                    "labels": ["source_root"],
                },
                {
                    "node_id": "concept",
                    "title": "1. System Name and Concept",
                    "text_preview": "Tirzah is a locally operated memory layer.",
                    "labels": ["source_section"],
                },
            ]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "compile_context", lambda _db, _node_id: None)

    output, details = execute_search_nodes_tool(
        FakeDb(),
        query="Tirzah technical design",
        original_query="What does the Tirzah technical design say the system is for?",
    )

    assert output["matches"][0]["node_id"] == "concept"
    assert details["ranking_query"] == (
        "Tirzah technical design What does the say system is for"
    )
    assert details["query_assembly"]["lexical_terms"] == [
        "Tirzah",
        "technical",
        "design",
        "system",
    ]
    assert any(item["query"] == "system" for item in details["fallback_queries"])


def test_ranked_focus_matches_uses_near_match_terms_after_empty_search(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        calls.append(query)
        if query == "technical":
            return [
                {
                    "node_id": "node1",
                    "title": "Technical Design",
                    "text_preview": "Technical design note.",
                    "labels": ["source_section"],
                }
            ]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "near_match_vocabulary", lambda _db: ["technical"])

    matches = ranked_focus_matches(FakeDb(), query="tecnical", label=None, limit=3)

    assert matches[0]["node_id"] == "node1"
    assert "technical" in calls


def test_ranked_focus_matches_uses_near_match_terms_after_weak_match(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    calls = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        calls.append(query)
        if query == "tecnical":
            return [{"node_id": "weak", "title": "Reference", "text_preview": "Generic note."}]
        if query == "technical":
            return [
                {
                    "node_id": "strong",
                    "title": "Technical Design",
                    "text_preview": "Technical design note.",
                    "labels": ["source_section"],
                }
            ]
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "near_match_vocabulary", lambda _db: ["technical"])

    matches = ranked_focus_matches(FakeDb(), query="tecnical", label=None, limit=3)

    assert matches[0]["node_id"] == "strong"
    assert "technical" in calls


def test_execute_search_nodes_tool_compiles_top_match(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    monkeypatch.setattr(
        interaction,
        "search_nodes",
        lambda *args, **kwargs: [{"node_id": "node1", "title": "Tirzah"}],
    )
    monkeypatch.setattr(
        interaction,
        "compile_context",
        lambda _db, node_id: {"focus_node_id": node_id, "records": []},
    )

    output, _details = execute_search_nodes_tool(FakeDb(), query="Tirzah")

    assert output["compiled_contexts"] == [{"focus_node_id": "node1", "records": []}]


def test_score_node_match_prefers_specific_title_terms() -> None:
    technical = {
        "title": "Tirzah Technical Design Document",
        "text_preview": "Architecture notes",
    }
    generic = {
        "title": "Worker Smoke",
        "text_preview": "Tirzah ingestion worker",
    }

    assert score_node_match(technical, "Tirzah technical design") > score_node_match(
        generic,
        "Tirzah technical design",
    )


def test_score_node_match_prefers_intent_section_over_generic_header() -> None:
    generic = {
        "title": "Tirzah Technical Design Document",
        "text_preview": "Header",
        "labels": ["source_root"],
    }
    concept = {
        "title": "1. System Name and Concept",
        "text_preview": "Tirzah is a locally operated memory layer.",
        "labels": ["source_section"],
    }

    assert score_node_match(concept, "Tirzah technical design system") > score_node_match(
        generic,
        "Tirzah technical design system",
    )


def test_score_node_match_keeps_named_project_above_broad_corpus_hits() -> None:
    ams_system = {
        "title": "2. It strengthens whole-system reading",
        "text_preview": "Coherence helps explain system-level dependence.",
        "labels": ["source_section", "ams_domain"],
        "provenance": {"source_path": "/home/cello/domains/AMS/coherence.md"},
    }
    tirzah_concept = {
        "title": "1. System Name and Concept",
        "text_preview": "Tirzah is a locally operated memory layer.",
        "labels": ["source_section"],
        "provenance": {"source_path": "Tirzah_Technical_Design_v0.1.md"},
    }

    assert score_node_match(
        tirzah_concept,
        "What does the Tirzah technical design say the system is for?",
    ) > score_node_match(
        ams_system,
        "What does the Tirzah technical design say the system is for?",
    )


def test_score_node_match_allows_partial_multi_anchor_matches() -> None:
    partial = {
        "title": "Tirzah Technical Design Document",
        "text_preview": "Architecture notes.",
        "labels": ["source_section"],
        "provenance": {"source_path": "Tirzah_Technical_Design_v0.1.md"},
    }
    weak = {
        "title": "Technical notes",
        "text_preview": "Generic notes.",
        "labels": ["source_section"],
        "provenance": {"source_path": "notes.md"},
    }

    assert score_node_match(
        partial,
        "Compare AMS Tirzah Technical design notes",
    ) > score_node_match(
        weak,
        "Compare AMS Tirzah Technical design notes",
    )


def test_score_node_match_ignores_sentence_initial_stopword_anchor() -> None:
    row = {
        "title": "Tirzah Configuration",
        "text_preview": "Configuration notes.",
        "labels": ["source_section"],
    }

    assert score_node_match(row, "This Tirzah configuration") > 0


def test_score_node_match_ignores_command_verb_anchor() -> None:
    row = {
        "title": "Tirzah Configuration",
        "text_preview": "Configuration notes.",
        "labels": ["source_section"],
    }

    assert score_node_match(row, "Find Tirzah configuration") > 0


def test_score_node_match_demotes_document_root() -> None:
    root = {
        "title": "Tirzah Technical Design Document",
        "text_preview": "System Name and Concept",
        "labels": ["source_root"],
    }
    section = {
        "title": "1. System Name and Concept",
        "text_preview": "Tirzah is a memory layer.",
        "labels": ["source_section"],
    }

    assert score_node_match(section, "Tirzah technical design system") > score_node_match(
        root,
        "Tirzah technical design system",
    )


def test_score_node_match_demotes_document_metadata_section_below_concept() -> None:
    metadata_section = {
        "title": "Tirzah Technical Design Document",
        "text_preview": "**Version:** 0.1\n**Status:** For Review\n**Date:** May 2026",
        "labels": ["source_section"],
        "provenance": {"source_path": "Tirzah_Technical_Design_v0.1.md"},
    }
    concept = {
        "title": "1. System Name and Concept",
        "text_preview": "Tirzah is a locally operated memory layer.",
        "labels": ["source_section"],
        "provenance": {"source_path": "Tirzah_Technical_Design_v0.1.md"},
    }

    assert score_node_match(
        concept,
        "What does the Tirzah technical design say the system is for?",
    ) > score_node_match(
        metadata_section,
        "What does the Tirzah technical design say the system is for?",
    )


def test_score_node_match_demotes_separator_only_chunks() -> None:
    separator = {
        "title": "Tirzah Technical Design Document / paragraph 2",
        "text_preview": "---",
        "labels": ["source_chunk"],
        "provenance": {"source_path": "Tirzah_Technical_Design_v0.1.md"},
    }
    concept = {
        "title": "1. System Name and Concept",
        "text_preview": "Tirzah is a locally operated memory layer.",
        "labels": ["source_section"],
        "provenance": {"source_path": "Tirzah_Technical_Design_v0.1.md"},
    }

    assert score_node_match(
        concept,
        "What does the Tirzah technical design say the system is for?",
    ) > score_node_match(
        separator,
        "What does the Tirzah technical design say the system is for?",
    )


def test_combined_query_text_deduplicates_planner_and_original_terms() -> None:
    assert combined_query_text("memory design", "What memory design does") == (
        "memory design What does"
    )


def test_build_query_assembly_extracts_terms_phrases_and_anchors() -> None:
    assembly = build_query_assembly(
        "technical design",
        "What does the Tirzah technical design say the system is for?",
    )

    assert assembly["ranking_query"] == (
        "technical design What does the Tirzah say system is for"
    )
    assert assembly["lexical_terms"] == ["technical", "design", "Tirzah", "system"]
    assert assembly["exact_phrases"][:3] == [
        "technical design",
        "Tirzah technical",
        "design system",
    ]
    assert assembly["anchor_terms"] == ["Tirzah"]
    assert assembly["near_match_terms"] == []


def test_build_query_assembly_can_include_near_match_terms() -> None:
    assembly = build_query_assembly(
        "tecnical desgin",
        vocabulary=["technical", "design", "memory"],
    )

    assert assembly["near_match_terms"] == [
        {
            "source_term": "tecnical",
            "candidate_term": "technical",
            "score": 0.94,
            "reason": "near_token_match",
        },
        {
            "source_term": "desgin",
            "candidate_term": "design",
            "score": 0.83,
            "reason": "near_token_match",
        },
    ]


def test_near_match_terms_skips_exact_matches_and_low_scores() -> None:
    assert near_match_terms(["technical", "zzzzzz"], ["technical", "design"]) == []


def test_vocabulary_terms_extracts_bounded_content_terms() -> None:
    assert vocabulary_terms(["Tirzah Technical Design", "/tmp/source.md"], limit=3) == [
        "Tirzah",
        "Technical",
        "Design",
    ]


def test_near_match_vocabulary_includes_label_definitions_and_documents() -> None:
    assert near_match_vocabulary(FakeVocabularyDb()) == [
        "memory_reference",
        "memory",
        "reference",
        "material",
        "taj_mahal",
        "mahal",
        "Technical",
        "Design",
    ]


def test_build_query_assembly_keeps_underscore_terms_for_near_matches() -> None:
    assembly = build_query_assembly(
        "memry_reference",
        vocabulary=["memory_reference"],
    )

    assert assembly["lexical_terms"] == ["memry_reference"]
    assert assembly["near_match_terms"] == [
        {
            "source_term": "memry_reference",
            "candidate_term": "memory_reference",
            "score": 0.97,
            "reason": "near_token_match",
        }
    ]


def test_fallback_queries_prefers_phrases_before_single_terms() -> None:
    assert fallback_queries("Tirzah technical design system")[:3] == [
        "Tirzah technical",
        "technical design",
        "design system",
    ]


def test_fallback_queries_tries_near_matches_before_original_single_terms() -> None:
    assembly = build_query_assembly("tecnical desgin", vocabulary=["technical", "design"])

    assert fallback_queries(assembly) == [
        "tecnical desgin",
        "technical",
        "design",
        "tecnical",
        "desgin",
    ]


def test_prepare_tool_results_for_answer_keeps_top_two_search_contexts() -> None:
    prepared = prepare_tool_results_for_answer(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "matches": [{"node_id": "top"}, {"node_id": "other"}],
                    "compiled_contexts": [
                        {
                            "focus_node_id": "top",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Top",
                                    "node_id": "top",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Top text.",
                                }
                            ],
                        },
                        {
                            "focus_node_id": "other",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Other",
                                    "node_id": "other",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Other text.",
                                }
                            ],
                        },
                    ],
                },
            }
        ]
    )

    assert prepared[0]["output"]["top_match"] == {"node_id": "top"}
    assert [context["focus_node_id"] for context in prepared[0]["output"]["top_contexts"]] == [
        "top",
        "other",
    ]
    assert prepared[0]["output"]["match_count"] == 2
    assert "matches" not in prepared[0]["output"]


def test_prepare_tool_results_for_answer_keeps_proximity_contexts() -> None:
    prepared = prepare_tool_results_for_answer(
        [
            {
                "tool": "expand_proximity",
                "ok": True,
                "output": {
                    "matches": [
                        {"node_id": "near1", "title": "Nearby", "proximity_score": 0.85},
                        {"node_id": "near2", "title": "Related", "proximity_score": 0.7},
                    ],
                    "compiled_contexts": [
                        {
                            "focus_node_id": "near1",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Nearby",
                                    "node_id": "near1",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Nearby node text.",
                                }
                            ],
                        }
                    ],
                },
            }
        ]
    )

    assert prepared[0]["output"]["top_match"] == {
        "node_id": "near1",
        "title": "Nearby",
        "proximity_score": 0.85,
    }
    assert prepared[0]["output"]["top_contexts"][0]["focus_node_id"] == "near1"
    assert prepared[0]["output"]["match_count"] == 2
    assert "matches" not in prepared[0]["output"]


def test_prepare_tool_results_for_answer_deduplicates_context_records() -> None:
    prepared = prepare_tool_results_for_answer(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "matches": [{"node_id": "top"}, {"node_id": "other"}],
                    "compiled_contexts": [
                        {
                            "focus_node_id": "top",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Top",
                                    "node_id": "shared",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Shared text.",
                                }
                            ],
                        },
                        {
                            "focus_node_id": "other",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Shared",
                                    "node_id": "shared",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Shared text.",
                                },
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Other",
                                    "node_id": "other",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Other text.",
                                },
                            ],
                        },
                    ],
                },
            }
        ]
    )

    contexts = prepared[0]["output"]["top_contexts"]
    assert [record["node_id"] for context in contexts for record in context["records"]] == [
        "shared",
        "other",
    ]


def test_prepare_tool_results_for_answer_skips_contexts_over_budget() -> None:
    prepared = prepare_tool_results_for_answer(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "matches": [{"node_id": "top"}],
                    "compiled_contexts": [
                        {
                            "focus_node_id": "top",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Oversized",
                                    "node_id": "large",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "x" * 5000,
                                }
                            ],
                        }
                    ],
                },
            }
        ]
    )

    assert prepared[0]["output"]["top_contexts"] == []


def test_render_tool_results_renders_context_as_markdown() -> None:
    rendered = render_tool_results(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "arguments": {"query": "system"},
                "details": {
                    "query_assembly": {
                        "lexical_terms": ["system", "Tirzah"],
                        "exact_phrases": ["Tirzah system"],
                        "anchor_terms": ["Tirzah"],
                    },
                    "fallback_queries": [
                        {"query": "Tirzah system", "result_count": 2},
                        {"query": "system", "result_count": 5},
                    ],
                },
                "output": {
                    "match_count": 1,
                    "top_match": {"node_id": "node1", "title": "System Name"},
                    "top_contexts": [
                        {
                            "document": {"title": "Doc", "document_id": "doc1"},
                            "focus_node_id": "node1",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "System Name",
                                    "node_id": "node1",
                                    "labels": ["source_section"],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Tirzah is a memory layer.",
                                }
                            ],
                        },
                        {
                            "document": {"title": "Doc", "document_id": "doc1"},
                            "focus_node_id": "node2",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Purpose",
                                    "node_id": "node2",
                                    "labels": ["source_section"],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "It assembles context.",
                                }
                            ],
                        },
                    ],
                },
            }
        ]
    )

    assert "### search_nodes" in rendered
    assert "# Tirzah Context" in rendered
    assert "Tirzah is a memory layer." in rendered
    assert "It assembles context." in rendered
    assert "Compiled context 1:" in rendered
    assert "Compiled context 2:" in rendered
    assert "- Lexical terms: system, Tirzah" in rendered
    assert "- Exact phrases: Tirzah system" in rendered
    assert "- Named anchors: Tirzah" in rendered
    assert "- Near-match terms: none" in rendered
    assert "- Fallback searches: Tirzah system (2), system (5)" in rendered
    assert "#### Tirzah Context" in rendered
    assert "\n# Tirzah Context" not in rendered
    assert "#### Compiled context" not in rendered
    assert '"top_contexts"' not in rendered


def test_render_tool_results_handles_empty_top_contexts() -> None:
    rendered = render_tool_results(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "arguments": {"query": "missing"},
                "output": {
                    "match_count": 0,
                    "top_match": None,
                    "top_contexts": [],
                },
            }
        ]
    )

    assert "### search_nodes" in rendered
    assert "Compiled context" not in rendered


def test_render_tool_results_renders_proximity_context_as_markdown() -> None:
    rendered = render_tool_results(
        [
            {
                "tool": "expand_proximity",
                "ok": True,
                "arguments": {"node_id": "node1", "direction": "outgoing"},
                "output": {
                    "match_count": 1,
                    "top_match": {
                        "node_id": "near1",
                        "title": "Nearby Node",
                        "proximity_score": 0.85,
                    },
                    "top_contexts": [
                        {
                            "document": {"title": "Doc", "document_id": "doc1"},
                            "focus_node_id": "near1",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Nearby Node",
                                    "node_id": "near1",
                                    "labels": ["source_section"],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Nearby context text.",
                                }
                            ],
                        }
                    ],
                },
            }
        ]
    )

    assert "### expand_proximity" in rendered
    assert "- Source node ID: node1" in rendered
    assert "- Direction: outgoing" in rendered
    assert "- Relation type: <any>" in rendered
    assert "- Proximity score: 0.85" in rendered
    assert "Nearby context text." in rendered
    assert '"top_contexts"' not in rendered


def test_render_tool_results_renders_graph_path_context_as_markdown() -> None:
    rendered = render_tool_results(
        [
            {
                "tool": "expand_graph_paths",
                "ok": True,
                "arguments": {"node_id": "node1", "direction": "outgoing", "max_depth": 2},
                "output": {
                    "match_count": 1,
                    "top_match": {
                        "node_id": "near2",
                        "title": "Two Hop Node",
                        "path_score": 0.5184,
                        "path_depth": 2,
                        "path_edges": [
                            {
                                "relation_type": "supports",
                                "candidate_source": "embedding_similarity",
                                "embedding_similarity": 0.91,
                                "embedding_model": "mock",
                                "embedding_dimensions": 16,
                                "selection_min_similarity": 0.8,
                            }
                        ],
                    },
                    "top_contexts": [
                        {
                            "document": {"title": "Doc", "document_id": "doc1"},
                            "focus_node_id": "near2",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Two Hop Node",
                                    "node_id": "near2",
                                    "labels": ["source_section"],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Two-hop context text.",
                                }
                            ],
                        }
                    ],
                },
            }
        ]
    )

    assert "### expand_graph_paths" in rendered
    assert "- Source node ID: node1" in rendered
    assert "- Max depth: 2" in rendered
    assert "- Path score: 0.5184" in rendered
    assert "- Path depth: 2" in rendered
    assert "- Path edge 1 evidence: embedding_similarity, similarity 0.91, threshold 0.8, mock 16 dims." in rendered
    assert "Two-hop context text." in rendered


def test_render_tool_results_renders_semantic_candidate_context_as_markdown() -> None:
    rendered = render_tool_results(
        [
            {
                "tool": "semantic_candidates",
                "ok": True,
                "arguments": {"node_id": "node1", "include_same_document": True},
                "output": {
                    "match_count": 1,
                    "top_match": {
                        "node_id": "near3",
                        "title": "Label Match",
                        "shared_labels": ["topic", "project"],
                        "shared_label_count": 2,
                    },
                    "top_contexts": [
                        {
                            "document": {"title": "Doc", "document_id": "doc1"},
                            "focus_node_id": "near3",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Label Match",
                                    "node_id": "near3",
                                    "labels": ["source_section", "topic"],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Semantic candidate context text.",
                                }
                            ],
                        }
                    ],
                },
            }
        ]
    )

    assert "### semantic_candidates" in rendered
    assert "- Source node ID: node1" in rendered
    assert "- Include same document: True" in rendered
    assert "- Shared label count: 2" in rendered
    assert "- Shared labels: topic, project" in rendered
    assert "Semantic candidate context text." in rendered


def test_agentic_answer_envelope_includes_structured_context_document() -> None:
    envelope = build_agentic_answer_envelope(
        query="What is Tirzah?",
        tool_results=[
            {
                "index": 0,
                "tool": "search_nodes",
                "arguments": {"query": "Tirzah"},
                "ok": True,
                "details": {
                    "normalized_query": "Tirzah",
                    "ranking_query": "Tirzah",
                    "query_assembly": {"lexical_terms": ["Tirzah"]},
                    "fallback_queries": [{"query": "Tirzah", "result_count": 1}],
                },
                "output": {
                    "matches": [{"node_id": "node1", "title": "System Name"}],
                    "compiled_contexts": [
                        {
                            "document": {"title": "Doc", "document_id": "doc1"},
                            "focus_node_id": "node1",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "System Name",
                                    "node_id": "node1",
                                    "labels": ["source_section"],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {"source_path": "design.md"},
                                    "text": "Tirzah is a memory layer.",
                                }
                            ],
                        }
                    ],
                },
            },
            {
                "index": 1,
                "tool": "get_document_tree",
                "arguments": {"document_id": "doc1"},
                "ok": True,
                "output": [{"node_id": f"node{index}"} for index in range(25)],
            },
            {
                "index": 2,
                "tool": "expand_proximity",
                "arguments": {"node_id": "node1"},
                "ok": True,
                "output": [{"node_id": f"near{index}"} for index in range(25)],
            },
        ],
        token_budget=2000,
        reserved_response_tokens=500,
    )

    context_document = envelope["context_metadata"]["context_document"]
    assert context_document["schema_version"] == 1
    assert context_document["kind"] == "agentic_answer_context"
    assert context_document["query"] == "What is Tirzah?"
    assert context_document["controller_decision"]["mode"] == "agentic"
    assert context_document["controller_decision"]["current_owner"] == "memory_agent_controller"
    assert envelope["context_metadata"]["controller_decision"] == context_document["controller_decision"]
    assert envelope["context_metadata"]["evidence_summary"] == context_document["evidence_summary"]
    assert context_document["evidence_summary"] == {
        "tool_result_count": 3,
        "successful_tool_count": 3,
        "failed_tool_count": 0,
        "match_count": 26,
        "context_count": 1,
        "record_count": 1,
        "included_node_count": 1,
        "included_node_ids": ["node1"],
        "source_documents": [{"document_id": "doc1", "title": "Doc"}],
    }
    search_output = context_document["tool_results"][0]["output"]
    assert search_output["top_match"] == {"node_id": "node1", "title": "System Name"}
    assert search_output["top_contexts"][0]["records"][0] == {
        "node_id": "node1",
        "role": "focus",
        "distance": 0,
        "title": "System Name",
        "labels": ["source_section"],
        "endorsement_label": "unreviewed",
        "provenance": {"source_path": "design.md"},
        "text": "Tirzah is a memory layer.",
            "chars": 25,
        }
    tree_output = context_document["tool_results"][1]["output"]
    assert tree_output["node_count"] == 25
    assert len(tree_output["nodes"]) == 20
    proximity_output = context_document["tool_results"][2]["output"]
    assert proximity_output["top_match"] == {"node_id": "near0"}
    assert proximity_output["match_count"] == 25
    assert proximity_output["top_contexts"] == []
    assert "context_document" not in envelope["prompt_text"]
    assert "## Controller Decision" in envelope["prompt_text"]
    assert "- Action: use_repository_context" in envelope["prompt_text"]


def test_agentic_answer_envelope_records_and_applies_context_proposal() -> None:
    envelope = build_agentic_answer_envelope(
        query="Which node matters?",
        tool_results=[
            {
                "index": 0,
                "tool": "search_nodes",
                "arguments": {"query": "node"},
                "ok": True,
                "output": {
                    "matches": [{"node_id": "node1"}, {"node_id": "node2"}],
                    "compiled_contexts": [
                        {
                            "document": {"title": "Doc 1", "document_id": "doc1"},
                            "focus_node_id": "node1",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Node 1",
                                    "node_id": "node1",
                                    "text": "Less relevant.",
                                }
                            ],
                        },
                        {
                            "document": {"title": "Doc 2", "document_id": "doc2"},
                            "focus_node_id": "node2",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Node 2",
                                    "node_id": "node2",
                                    "text": "Selected by memory agent.",
                                }
                            ],
                        },
                    ],
                },
            }
        ],
        token_budget=2000,
        reserved_response_tokens=500,
        proposed_controller_decision={
            "action": "use_repository_context",
            "reason": "Node 2 should drive the answer.",
            "confidence": 0.75,
        },
        context_proposal={
            "selected_node_ids": ["node2"],
            "rationale": "Node 2 is the better evidence.",
            "organization": ["selected evidence first"],
        },
    )

    context_document = envelope["context_metadata"]["context_document"]
    assert context_document["context_proposal"]["selected_node_ids"] == ["node2"]
    assert context_document["controller_decision"]["proposed_by"] == "memory_agent"
    assert context_document["controller_decision"]["reason"] == "Node 2 should drive the answer."
    assert context_document["controller_decision"]["confidence"] == 0.75
    assert "## Controller Decision" in envelope["prompt_text"]
    assert "- Reason: Node 2 should drive the answer." in envelope["prompt_text"]
    assert "- Confidence: 0.75" in envelope["prompt_text"]
    assert envelope["context_metadata"]["context_proposal"]["rationale"] == (
        "Node 2 is the better evidence."
    )
    search_output = context_document["tool_results"][0]["output"]
    assert search_output["top_contexts"][0]["focus_node_id"] == "node2"
    assert "Node 2" in envelope["context_text"].split("Compiled context 1:", 1)[1]


def test_agentic_controller_decision_corrects_impossible_context_action() -> None:
    envelope = build_agentic_answer_envelope(
        query="Can you answer?",
        tool_results=[
            {
                "index": 0,
                "tool": "search_nodes",
                "arguments": {"query": "missing"},
                "ok": True,
                "output": {"matches": [], "compiled_contexts": []},
            }
        ],
        token_budget=2000,
        reserved_response_tokens=500,
        proposed_controller_decision={
            "action": "use_repository_context",
            "reason": "I think context is available.",
            "confidence": 0.9,
        },
    )

    decision = envelope["context_metadata"]["controller_decision"]
    assert decision["action"] == "answer_after_memory_tools_without_context"
    assert decision["correction"]["original_action"] == "use_repository_context"
    assert decision["validation_issues"] == []
    assert "- Correction: use_repository_context -> answer_after_memory_tools_without_context" in (
        envelope["prompt_text"]
    )


def test_included_nodes_from_tool_results_collects_multiple_contexts() -> None:
    included = included_nodes_from_tool_results(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "top_match": {"node_id": "match-only"},
                    "top_contexts": [
                        {
                            "focus_node_id": "node1",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Node 1",
                                    "node_id": "node1",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "First rendered record.",
                                }
                            ],
                        },
                        {
                            "focus_node_id": "node2",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Node 2",
                                    "node_id": "node2",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Second rendered record.",
                                }
                            ],
                        },
                    ]
                },
            }
        ]
    )

    assert {row["node_id"] for row in included} == {"node1", "node2"}
    assert all(row["chars"] < 500 for row in included)


def test_included_nodes_from_tool_results_collects_proximity_contexts() -> None:
    included = included_nodes_from_tool_results(
        [
            {
                "tool": "expand_proximity",
                "ok": True,
                "output": {
                    "top_match": {"node_id": "match-only", "title": "Nearby"},
                    "top_contexts": [
                        {
                            "focus_node_id": "near1",
                            "records": [
                                {
                                    "role": "focus",
                                    "distance": 0,
                                    "title": "Nearby",
                                    "node_id": "near1",
                                    "labels": [],
                                    "endorsement_label": "unreviewed",
                                    "provenance": {},
                                    "text": "Nearby rendered record.",
                                }
                            ],
                        }
                    ],
                },
            }
        ]
    )

    assert included[0]["node_id"] == "near1"
    assert included[0]["role"] == "focus"


def test_included_nodes_from_tool_results_falls_back_to_top_match_without_contexts() -> None:
    included = included_nodes_from_tool_results(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "top_match": {"node_id": "match-only", "title": "Visible Search Hit"},
                    "top_contexts": [],
                },
            }
        ]
    )

    assert included == [
        {
            "node_id": "match-only",
            "role": "search_match",
            "distance": 0,
            "chars": len("- Top match: Visible Search Hit\n- Top node ID: match-only"),
        }
    ]


def test_included_nodes_from_tool_results_does_not_fall_back_when_context_records_duplicate() -> None:
    record = {
        "role": "focus",
        "distance": 0,
        "title": "Shared Node",
        "node_id": "shared-node",
        "labels": [],
        "endorsement_label": "unreviewed",
        "provenance": {},
        "text": "Shared rendered record.",
    }

    included = included_nodes_from_tool_results(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "top_match": {"node_id": "shared-node", "title": "Shared Node"},
                    "top_contexts": [{"records": [record]}],
                },
            },
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "top_match": {"node_id": "second-match", "title": "Second Match"},
                    "top_contexts": [{"records": [record]}],
                },
            },
        ]
    )

    assert {row["node_id"] for row in included} == {"shared-node"}
    assert included[0]["role"] == "focus"


def test_included_nodes_from_tool_results_preserves_non_search_tool_nodes() -> None:
    included = included_nodes_from_tool_results(
        [
            {
                "tool": "compile_context",
                "ok": True,
                "output": {
                    "focus_node_id": "focus-node",
                    "records": [
                        {
                            "node_id": "record-node",
                            "role": "focus",
                            "distance": 0,
                            "text": "Compiled context text.",
                        }
                    ],
                },
            }
        ]
    )

    assert {row["node_id"] for row in included} == {"focus-node", "record-node"}


def test_included_nodes_from_tool_results_combines_search_and_non_search_nodes() -> None:
    included = included_nodes_from_tool_results(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "top_match": {"node_id": "match-node", "title": "Search Match"},
                    "top_contexts": [],
                },
            },
            {
                "tool": "compile_context",
                "ok": True,
                "output": {
                    "focus_node_id": "focus-node",
                    "records": [{"node_id": "record-node", "role": "focus"}],
                },
            },
        ]
    )

    assert {row["node_id"] for row in included} == {
        "match-node",
        "focus-node",
        "record-node",
    }


def test_prepare_tool_results_for_answer_applies_aggregate_context_budget() -> None:
    record_text = "x" * 2600
    contexts = []
    for index in range(3):
        contexts.append(
            {
                "document": {"title": f"Doc {index}", "document_id": f"doc{index}"},
                "focus_node_id": f"node{index}",
                "records": [
                    {
                        "role": "focus",
                        "distance": 0,
                        "title": f"Node {index}",
                        "node_id": f"node{index}",
                        "labels": [],
                        "endorsement_label": "unreviewed",
                        "provenance": {},
                        "text": record_text,
                    }
                ],
            }
        )

    prepared = prepare_tool_results_for_answer(
        [
            {
                "tool": "search_nodes",
                "ok": True,
                "output": {
                    "matches": [{"node_id": "node0"}, {"node_id": "node1"}, {"node_id": "node2"}],
                    "compiled_contexts": contexts,
                },
            }
        ]
    )

    assembled = prepared[0]["output"]["top_contexts"]
    rendered = render_tool_results(prepared)

    assert len(assembled) == 1
    assert assembled[0]["focus_node_id"] == "node0"
    assert len(rendered) < 5000


def test_build_query_embedding_gating(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction
    from tirzah.config import RuntimeConfig

    class FakeEmbedder:
        def embed(self, text):
            return {"model": "real", "dimensions": 2, "vector": [1.0, 0.0]}

    enabled_real = RuntimeConfig(hybrid_search_enabled=True, embedding_adapter="local_command")

    # disabled / mock adapter / empty text → no query embedding (lexical-only).
    assert (
        interaction.build_query_embedding(
            RuntimeConfig(hybrid_search_enabled=False, embedding_adapter="local_command"), "q"
        )
        is None
    )
    assert (
        interaction.build_query_embedding(
            RuntimeConfig(hybrid_search_enabled=True, embedding_adapter="mock"), "q"
        )
        is None
    )
    assert interaction.build_query_embedding(enabled_real, "") is None

    # enabled + real adapter → payload from the embedder.
    monkeypatch.setattr(interaction, "embedding_adapter", lambda _cfg: FakeEmbedder())
    assert interaction.build_query_embedding(enabled_real, "q") == {
        "model": "real",
        "dimensions": 2,
        "vector": [1.0, 0.0],
    }

    # embedder failure degrades gracefully to lexical-only.
    class BoomEmbedder:
        def embed(self, text):
            raise RuntimeError("boom")

    monkeypatch.setattr(interaction, "embedding_adapter", lambda _cfg: BoomEmbedder())
    assert interaction.build_query_embedding(enabled_real, "q") is None


def test_ranked_focus_matches_forwards_query_embedding(monkeypatch) -> None:
    import tirzah.sessions.interaction as interaction

    captured = []

    def fake_search_nodes(_db, query=None, label=None, document_id=None, limit=5, **kwargs):
        captured.append(kwargs.get("query_embedding"))
        return []

    monkeypatch.setattr(interaction, "search_nodes", fake_search_nodes)
    monkeypatch.setattr(interaction, "weak_match_fallback_needed", lambda *a, **k: False)
    embedding = {"vector": [1.0, 0.0], "dimensions": 2, "model": "m"}

    # Provided → forwarded to every search_nodes call (hybrid in direct mode).
    interaction.ranked_focus_matches(None, "memory", label=None, limit=3, query_embedding=embedding)
    assert captured and all(c == embedding for c in captured)

    # Omitted → not forwarded (lexical path unchanged; kwarg absent).
    captured.clear()
    interaction.ranked_focus_matches(None, "memory", label=None, limit=3)
    assert captured and all(c is None for c in captured)


def test_answer_query_deep_mode_dispatch(monkeypatch) -> None:
    """Deep mode dispatches through the phased flow: answer_query →
    answer_phases._retrieve_deep (deep retrieval collects chunks) →
    _synthesize_deep_and_persist (synthesize_answer produces the text)."""
    import tirzah.retrieval.deep as deep
    import tirzah.sessions.interaction as interaction

    monkeypatch.setattr(interaction, "first_active_agent_identity", lambda db: None)
    # The 'fake' adapter never reaches the factory: the phases resolve it via
    # the interaction namespace, which is the monkeypatch seam.
    monkeypatch.setattr(interaction, "answer_adapter", lambda runtime: object())
    monkeypatch.setattr(interaction, "build_query_embedding", lambda runtime, text: None)
    monkeypatch.setattr(interaction, "render_session_history_block", lambda *a, **k: "")
    monkeypatch.setattr(
        deep,
        "run_deep_retrieval",
        lambda db, query, **k: {
            "useful_chunks": [{"node_id": "n1"}, {"node_id": "n2"}],
            "rounds": [],
            "trace": [{"step": "stop", "reason": "planner_stop"}],
        },
    )
    monkeypatch.setattr(deep, "synthesize_answer", lambda *a, **k: "deep answer")
    monkeypatch.setattr(interaction, "save_exchange", lambda *a, **k: "exch-deep")
    config = AppConfig(runtime=RuntimeConfig(retrieval_mode="deep", answer_adapter="fake"))

    result = answer_query(FakeDb(), config, "find memory")

    assert result["ok"] is True
    assert result["answer"] == "deep answer"
    assert result["used_node_ids"] == ["n1", "n2"]
    assert result["exchange_id"] == "exch-deep"
    assert result["retrieval_status"] == "deep_context"
