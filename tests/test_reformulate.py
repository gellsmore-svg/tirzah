from tirzah.retrieval.reformulate import (
    edit_distance,
    enrich_query_assembly,
    fallback_queries,
    light_stems,
    near_match_terms,
    reformulated_query,
    synonym_hints,
)


def test_light_stems_strip_common_suffixes() -> None:
    assert "vorton" in light_stems("vortons")
    assert "memory" in light_stems("memories")


def test_synonym_hints_are_vocabulary_gated() -> None:
    assert synonym_hints("memory", ["recall", "source"]) == ["recall"]
    assert synonym_hints("memory", ["source"]) == []


def test_edit_distance_detects_single_typo() -> None:
    assert edit_distance("tecnical", "technical") == 1
    assert edit_distance("desgin", "design") == 2


def test_near_match_terms_include_stem_and_keep_best_fuzzy() -> None:
    matches = near_match_terms(["vortons", "tecnical"], ["vorton", "technical", "design"])
    by_source = {item["source_term"]: item for item in matches}
    assert by_source["vortons"]["candidate_term"] == "vorton"
    assert by_source["vortons"]["reason"] == "light_stem"
    assert by_source["tecnical"]["candidate_term"] == "technical"
    assert by_source["tecnical"]["reason"] == "near_token_match"
    assert by_source["tecnical"]["score"] == 0.94


def test_enrich_query_assembly_records_reformulated_query() -> None:
    assembly = enrich_query_assembly(
        {
            "original_query": "tecnical desgin",
            "ranking_query": "tecnical desgin",
            "lexical_terms": ["tecnical", "desgin"],
            "exact_phrases": ["tecnical desgin"],
        },
        ["technical", "design"],
    )
    assert assembly["reformulated_query"] == "technical desgin"
    assert fallback_queries(assembly)[0] == "technical desgin"


def test_reformulated_query_keeps_original_when_no_strong_match() -> None:
    assert (
        reformulated_query(
            "zzzzzz",
            ["zzzzzz"],
            [{"source_term": "zzzzzz", "candidate_term": "design", "score": 0.5, "reason": "near_token_match"}],
        )
        == "zzzzzz"
    )
