from tirzah.retrieval.definitions import definition_pairs, extract_definitions, split_sentences

TERMS = ["T1", "T2", "current", "field", "darkness", "vorton", "magnetism", "T0"]


def one(text, term=None):
    found = [d for d in extract_definitions(text, TERMS) if term is None or d["term"] == term]
    assert len(found) == 1, found
    return found[0]


def test_asserted_definitions_including_equals_and_markup() -> None:
    d = one("- T1 names the primary material baseline.", "T1")
    assert (d["polarity"], d["verb"], d["body"]) == ("asserts", "names", "the primary material baseline")
    d = one("> Current = integrated vorton-slip activity crossing a section of the circuit per second.", "current")
    assert d["verb"] == "=" and d["body"].startswith("integrated vorton-slip activity")
    d = one("Darkness is **matter whose AMS configuration is not expressing radiative torsion**.", "darkness")
    assert d["polarity"] == "asserts" and d["body"].startswith("matter whose AMS configuration")
    assert one("Vortons are substrate-level topological identities (T0-bound).", "vorton")["term"] == "vorton"


def test_inverted_definitions() -> None:
    d = one(
        "A vorton doesn't carry energy. It gets repositioned as the AMS around it is cyclically deformed by a "
        "battery's tension gradient, and its slow net movement is what we call electrical current.",
        "current",
    )
    assert (d["polarity"], d["body"]) == ("asserts", "its slow net movement")


def test_pronoun_continuation_defines_the_same_term() -> None:
    found = extract_definitions(
        "`T1` names stable identity-bearing structures. It no longer names the substrate baseline itself. "
        "It names persistent identities. It is raining.", ["T1"])
    assert [(d["polarity"], d["body"]) for d in found] == [
        ("asserts", "stable identity-bearing structures"),
        ("denies", "the substrate baseline itself"),
        ("asserts", "persistent identities"),
        ("asserts", "raining"),  # chained continuations are kept; the ranker weighs them
    ]
    assert found[1]["sentence"] == "T1 names stable identity-bearing structures. It no longer names the substrate baseline itself."
    # Context is the neighbouring sentences, not the start of the node.
    assert found[0]["context"].startswith("T1 names stable") and "It names persistent identities." in found[0]["context"]
    # No continuation without a definition of the term in the sentence before.
    assert extract_definitions("T1 appears here. It names something else entirely.", ["T1"]) == []


def test_not_a_but_b_denies_a_and_asserts_b() -> None:
    found = extract_definitions("Current is not particle flow but the rate of AMS reconfiguration.", ["current"])
    assert [(d["polarity"], d["body"]) for d in found] == [
        ("denies", "particle flow"), ("asserts", "the rate of AMS reconfiguration")]
    found = extract_definitions(
        "Fields, in AMS, are not fundamental things. They are ways of summarising the state of the substrate.", ["field"])
    assert [(d["polarity"], d["body"]) for d in found] == [
        ("denies", "fundamental things"), ("asserts", "ways of summarising the state of the substrate")]


def test_contrast_is_not_revision_and_shared_rejections_agree() -> None:
    assert one("Darkness is a condition within creation rather than non-being.", "darkness")["revision"] is False
    rows = _defs("energy", "Energy is not a substance.", "Energy is a state of substrate tension, not a stored substance.")
    assert definition_pairs(rows) == []
    rows = _defs("energy", "Energy is not a state of tension.", "Energy is a state of substrate tension, not a stored substance.")
    assert [p["reasons"][0] for p in definition_pairs(rows)] == ["asserted_and_denied"]


def test_reframed_as_is_a_definition() -> None:
    d = one("Fields were reframed as stable torsional configurations in the substrate.", "field")
    assert (d["verb"], d["polarity"]) == ("were reframed as", "asserts")
    assert d["revision"] is True


def test_denials_and_qualifications() -> None:
    d = one("The point is not that current is vorton motion.", "current")
    assert (d["polarity"], d["body"]) == ("denies", "vorton motion")
    assert one("Fields, in AMS, are not fundamental things.", "field")["polarity"] == "denies"
    assert one("Magnetism is not merely decorative accompaniment to electrical behaviour.", "magnetism")["polarity"] == "qualifies"
    d, _continuation = extract_definitions(
        "`T1` names stable identity-bearing structures. It no longer names the substrate baseline itself.", ["T1"])
    assert d["polarity"] == "asserts" and d["revision"] is True  # marker in the following sentence


def test_framed_definitions_are_marked() -> None:
    assert one("AMS is not:\n- a claim that T0 is an energy store", "T0")["framed"] is True
    assert one("Those statements are not coherent if darkness is merely unlit matter.", "darkness")["framed"] is True
    assert one("Darkness is the ordered, non-expressive ground state of the substrate.", "darkness")["framed"] is False
    [d] = extract_definitions("Readers assume time is a neutral container.", ["time"])
    assert d["framed"] is True
    [d] = extract_definitions("One of the quiet assumptions in modern physics is that energy is a thing we store.",
                              ["energy"])
    assert d["framed"] is True
    for sentence, term in (
        ("In conventional physics, fields are often treated as real substances filling space.", "field"),
        ("In common usage, darkness is treated as absence of light.", "darkness"),
        ("Fields are spoken of as though they were substances.", "field"),
    ):
        found = extract_definitions(sentence, [term])
        assert found and all(d["framed"] for d in found), sentence
    # The author's own "X is treated as Y" is still a definition.
    [d] = extract_definitions("Electricity is treated as ordered transfer through pathways.", ["electricity"])
    assert d["framed"] is False


def test_lead_in_and_heading_frames() -> None:
    d = one("The rebuilt book should no longer speak as though:\n- T1 is simply the deeper baseline itself", "T1")
    assert d["framed"] is True
    [d] = extract_definitions("Light is mysterious and unexplained.", ["light"],
                              title="Comparator 2: Romantic ether-poetry model / paragraph 1")
    assert d["framed"] is True
    [d] = extract_definitions("Gravity is a force between masses.", ["gravity"], title="Newton (as usually presented)")
    assert d["framed"] is True
    for heading in ("Replaced Assumption / paragraph 1", "Guardrail / paragraph 2", "Reader starting point"):
        [d] = extract_definitions("Fields are the most basic physical entities.", ["field"], title=heading)
        assert d["framed"] is True, heading
    [d] = extract_definitions("Fields are stable geometric states of the substrate.", ["field"],
                              title="Replace with / paragraph 3")
    assert d["framed"] is False
    [d] = extract_definitions("Light is a transient torsional propagation.", ["light"], title="Chapter 25 / paragraph 2")
    assert d["framed"] is False
    # A lead-in without a framing word leaves its bullets alone.
    assert one("Core terms:\n- T1 names stable identity-bearing structures.", "T1")["framed"] is False


def test_configured_frame_titles_and_run_on_sentences() -> None:
    import re

    from tirzah.retrieval.contradictions import claim_tokens
    from tirzah.retrieval.definitions import definition_strength

    [d] = extract_definitions("Gravity is not a force.", ["gravity"], title="Einstein (General Relativity)")
    assert d["framed"] is False
    [d] = extract_definitions("Gravity is not a force.", ["gravity"], title="Einstein (General Relativity)",
                              frame_titles=[re.compile("general relativity", re.I)])
    assert d["framed"] is True
    [short] = extract_definitions("Current is the rate of reconfiguration.", ["current"])
    run_on = {**short, "sentence": short["sentence"] + " and so" * 40}
    tokens = claim_tokens({"text": short["body"]})
    assert definition_strength(run_on, tokens) == definition_strength(short, tokens) - 1.5


def test_non_definitions_are_ignored() -> None:
    assert extract_definitions("The current manuscript says this for too long.", TERMS) == []
    assert split_sentences("- a\n- short line here. Another sentence follows.") == ["short line here.", "Another sentence follows."]


def _defs(term, *sentences, document_ids=None):
    rows = []
    for index, sentence in enumerate(sentences):
        for d in extract_definitions(sentence, [term]):
            rows.append({**d, "node_id": f"n{index}", "document_id": (document_ids or {}).get(index, f"doc{index}")})
    return rows


def test_pairs_rank_asserted_and_denied_first_and_skip_agreement() -> None:
    rows = _defs(
        "current",
        "Current = integrated vorton-slip activity crossing a section of the circuit per second.",
        "The point is not that current is vorton motion.",
        "Current names the rate of reconfiguration along an available path.",
        "Current names the rate of reconfiguration along an available path.",  # a copy
    )
    pairs = definition_pairs(rows)
    top = pairs[0]
    assert "asserted_and_denied" in top["reasons"]
    assert {top["a"]["body"], top["b"]["body"]} == {
        "integrated vorton-slip activity crossing a section of the circuit per second", "vorton motion"}
    # The duplicate sentence collapses into one definition with two copies.
    rates = [p for p in pairs if "rate of reconfiguration" in p["a"]["body"] + p["b"]["body"]]
    assert all(p["a"]["body"] != p["b"]["body"] for p in rates)


def test_pairs_find_changed_definitions_and_leave_out_framed_and_qualified() -> None:
    rows = _defs(
        "T1",
        "T1 names the primary material baseline.",
        "`T1` names stable identity-bearing structures. It no longer names the substrate baseline itself.",
        "This is not a claim that T1 is a separate world.",
        "T1 is not merely a label.",
    )
    pairs = definition_pairs(rows)
    assert len(pairs) == 2
    # The "It no longer names ..." continuation denies the old baseline reading.
    top = pairs[0]
    assert {top["a"]["polarity"], top["b"]["polarity"]} == {"asserts", "denies"}
    assert "primary material baseline" in top["a"]["sentence"] + top["b"]["sentence"]
    assert set(pairs[1]["reasons"]) >= {"revision_marker", "different_definitions"}


def test_ranking_prefers_real_definitions_over_predications() -> None:
    rows = _defs(
        "energy",
        "Energy names a state of substrate tension.",
        "Energy is the capacity of a configuration to change.",
        "Energy is not stored in a component.",
        "Energy is stored in capacitors.",
    )
    pairs = definition_pairs(rows)
    assert pairs, "the two genuine definitions should pair"
    assert {pairs[0]["a"]["body"], pairs[0]["b"]["body"]} == {
        "a state of substrate tension", "the capacity of a configuration to change"}
    assert "strong_definitions" not in pairs[0]["reasons"]  # only one side uses a strong verb
    # "is not stored" vs "is stored" are passing predications: they may fill a
    # leftover slot, but always below a pair of real definitions.
    weak = [p for p in pairs if "capacitors" in p["a"]["sentence"] + p["b"]["sentence"]]
    assert all(p["score"] < pairs[0]["score"] for p in weak)


def test_definition_cap_keeps_the_most_definitional_sentences() -> None:
    rows = _defs(
        "field",
        "Field is ordered somewhere.",  # weak: passing predication
        "Field is loud.",  # weak: too short
        "Field names a stable torsional configuration.",
        "Field names a summary of substrate state.",
    )
    [pair] = definition_pairs(rows, max_definitions_per_term=2)
    assert {pair["a"]["body"], pair["b"]["body"]} == {"a stable torsional configuration", "a summary of substrate state"}


def test_pairs_group_restatements_of_one_change() -> None:
    rows = _defs(
        "T2",
        "T2 names manifest lived embodiment and conscious interface.",
        "T2 names the secondary ordering conditions governing propagation.",
        "T2 names manifest lived embodiment, perception and conscious interface.",
        "T2 names the secondary ordering conditions governing coupling.",
    )
    [pair] = definition_pairs(rows)  # one change, however many passages state it
    assert pair["group_size"] == 4
    assert len(pair["supporting"]) == 3
    assert all(s["a_node_id"] and s["b_node_id"] for s in pair["supporting"])
    assert len(definition_pairs(rows, group_changes=False)) == 4  # diagnostics rank every pair
    # Genuinely different changes keep their own rows.
    others = _defs(
        "charge",
        "Charge is the directional bias of vorton slip.",
        "Charge is an effective description of stable asymmetry.",
        "Charge is a conserved scalar quantity of particles.",
    )
    assert len(definition_pairs(others)) == 3


def test_no_definition_fills_a_terms_quota() -> None:
    bodies = ["stable torsional configurations", "a potential gradient in space", "an abstraction over forces",
              "the ordered ground of the substrate", "measurable tension around charges", "a ledger of momentum flow"]
    rows = _defs("field", *[f"Field names {body}." for body in bodies])
    pairs = definition_pairs(rows, pairs_per_term=10, max_pairs_per_definition=2)
    usage: dict[str, int] = {}
    for pair in pairs:
        for side in ("a", "b"):
            usage[pair[side]["sentence"]] = usage.get(pair[side]["sentence"], 0) + 1
    assert pairs and max(usage.values()) <= 2
    assert len(pairs) <= 6


def test_router_parses_model_answers(monkeypatch) -> None:
    import tirzah.adapters.answer as answer
    from tirzah.config import RuntimeConfig
    from tirzah.retrieval.definitions import make_definition_router

    replies = iter(['YES\n"Current names the rate of reconfiguration along a path."', "NO - it only mentions it", "maybe"])
    monkeypatch.setattr(answer, "generate_text", lambda *_a, **_k: {"answer": next(replies)})
    route = make_definition_router(RuntimeConfig())
    node = {"title": "Current", "text": "Intro. Current names the rate of reconfiguration along a path. More text."}

    first = route("current", node)
    assert first["status"] == "defines"
    assert first["sentence"] == "Current names the rate of reconfiguration along a path."
    assert route("current", node)["status"] == "not_definition"
    assert route("current", node)["status"] == "unparsed"
    assert make_definition_router(RuntimeConfig(definition_second_pass_enabled=False)) is None


def test_definition_confirmer_asks_a_term_anchored_question(monkeypatch) -> None:
    import tirzah.adapters.answer as answer
    from tirzah.config import RuntimeConfig
    from tirzah.retrieval.definitions import _passage, make_definition_confirmer

    prompts = []
    replies = iter(["CONFLICT\nThey name different things.", "SAME - a restatement", "no idea"])

    def fake(_runtime, prompt, **_kwargs):
        prompts.append(prompt)
        return {"answer": next(replies)}

    monkeypatch.setattr(answer, "generate_text", fake)
    confirm = make_definition_confirmer(RuntimeConfig())
    a = _passage({"term": "T2", "sentence": "T2 names secondary ordering conditions.", "title": "Old",
                  "context": "Chapter 3 context."})
    b = _passage({"term": "T2", "sentence": "T2 names transient field and process structures.", "title": "New",
                  "context": "Chapter 4 context."})
    assert confirm(a, b)["status"] == "confirmed"
    assert 'what "T2" is or names' in prompts[0] and "secondary ordering conditions" in prompts[0]
    assert "Chapter 3 context." in prompts[0] and "Chapter 4 context." in prompts[0]
    assert confirm(a, b)["status"] == "rejected"
    assert confirm(a, b)["status"] == "unparsed"
    assert make_definition_confirmer(RuntimeConfig(contradiction_confirmation_enabled=False)) is None
    assert make_definition_confirmer(None) is None


def test_cli_definition_drift_wires_terms_and_defaults(monkeypatch, capsys) -> None:
    import json
    import sys
    from types import SimpleNamespace

    from tirzah.cli import main

    monkeypatch.setattr(sys, "argv", ["tirzah", "definition-drift", "--term", "T1", "--term", "current", "--pairs-per-term", "5"])
    monkeypatch.setattr("tirzah.cli.load_config", lambda _path: SimpleNamespace(mongo=SimpleNamespace()))
    monkeypatch.setattr("tirzah.cli.get_database", lambda _config: "db")
    monkeypatch.setattr("tirzah.cli.ensure_indexes", lambda _db: None)
    monkeypatch.setattr("tirzah.db.repositories.enqueue_definition_drift_candidates",
                        lambda _db, terms, **kwargs: {"ok": True, "terms": terms, **kwargs})

    main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "ok": True, "terms": ["T1", "current"], "pairs_per_term": 5, "include_same_document": True,
        "second_pass_limit": 20, "router": None, "confirmer": None, "created_by": "user", "dry_run": True,
        "frame_titles": [], "exclude_titles": [],
    }


def test_same_document_pairs_are_optional_and_older_comes_first() -> None:
    rows = _defs("darkness", "Darkness is matter whose configuration is not expressing radiative torsion.",
                 "Darkness is the ordered, non-expressive ground state of the substrate.", document_ids={0: "d", 1: "d"})
    rows[0]["origin_date"], rows[1]["origin_date"] = "2026-03-01", "2025-12-01"
    assert definition_pairs(rows, include_same_document=False) == []
    [pair] = definition_pairs(rows)
    assert pair["a"]["origin_date"] == "2025-12-01"
