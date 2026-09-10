from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

DEFAULT_NEAR_MATCH_MIN_SCORE = 0.78
DEFAULT_NEAR_MATCH_PER_TERM = 3
DEFAULT_NEAR_MATCH_MAX_CANDIDATES = 8

QUERY_SYNONYMS = {
    "memory": ["recall", "remembrance"],
    "document": ["source", "file"],
    "search": ["find", "retrieve"],
    "chunk": ["passage", "section"],
    "graph": ["network"],
    "summary": ["abstract", "overview"],
}


def light_stems(term: str) -> list[str]:
    key = term.lower()
    stems: list[str] = []
    if key.endswith("ies") and len(key) > 5:
        stems.append(key[:-3] + "y")
    if key.endswith("ing") and len(key) > 6:
        stems.append(key[:-3])
        if len(key) > 7:
            stems.append(key[:-3] + "e")
    if key.endswith("ed") and len(key) > 5:
        stems.append(key[:-2])
        stems.append(key[:-1])
    if key.endswith("es") and len(key) > 4 and not key.endswith("sses"):
        stems.append(key[:-2])
    if key.endswith("s") and not key.endswith("ss") and len(key) > 4:
        stems.append(key[:-1])
    return dedupe_preserve_order(stems)


def synonym_hints(term: str, vocabulary: list[str] | None = None) -> list[str]:
    key = term.lower()
    vocab = {item.lower() for item in vocabulary or []}
    hints = []
    for synonym in QUERY_SYNONYMS.get(key, []):
        if not vocab or synonym.lower() in vocab:
            hints.append(synonym)
    for source, targets in QUERY_SYNONYMS.items():
        if key in {item.lower() for item in targets} and (not vocab or source in vocab):
            hints.append(source)
    return dedupe_preserve_order(hints)


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 2:
        return 99
    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        current = [i]
        for j, right_ch in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_ch != right_ch)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def near_match_terms(
    source_terms: list[str],
    vocabulary: list[str],
    min_score: float = DEFAULT_NEAR_MATCH_MIN_SCORE,
    limit: int = DEFAULT_NEAR_MATCH_MAX_CANDIDATES,
    per_term: int = DEFAULT_NEAR_MATCH_PER_TERM,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    vocabulary_by_key = {
        term.lower(): term
        for term in vocabulary
        if is_content_term(term)
    }
    for source_term in source_terms:
        source_key = source_term.lower()
        if source_key in vocabulary_by_key:
            continue
        ranked: list[dict[str, Any]] = []
        for stem in light_stems(source_term):
            candidate = vocabulary_by_key.get(stem)
            if candidate:
                ranked.append(
                    {
                        "source_term": source_term,
                        "candidate_term": candidate,
                        "score": 1.0,
                        "reason": "light_stem",
                    }
                )
        for synonym in synonym_hints(source_term, list(vocabulary_by_key)):
            ranked.append(
                {
                    "source_term": source_term,
                    "candidate_term": vocabulary_by_key.get(synonym, synonym),
                    "score": 0.88,
                    "reason": "synonym",
                }
            )
        for candidate_key, candidate_term in vocabulary_by_key.items():
            if abs(len(source_key) - len(candidate_key)) > 2:
                continue
            if len(source_key) >= 5 and edit_distance(source_key, candidate_key) == 1:
                ranked.append(
                    {
                        "source_term": source_term,
                        "candidate_term": candidate_term,
                        "score": round(1 - (1 / max(len(source_key), len(candidate_key))), 2),
                        "reason": "typo_edit",
                    }
                )
            score = SequenceMatcher(None, source_key, candidate_key).ratio()
            if score >= min_score:
                ranked.append(
                    {
                        "source_term": source_term,
                        "candidate_term": candidate_term,
                        "score": round(score, 2),
                        "reason": "near_token_match",
                    }
                )
        ranked.sort(key=lambda item: (-float(item["score"]), item["candidate_term"].lower()))
        for item in ranked:
            key = (source_key, item["candidate_term"].lower())
            if key in seen:
                continue
            seen.add(key)
            matches.append(item)
            if sum(1 for row in matches if row["source_term"].lower() == source_key) >= per_term:
                break
        if len(matches) >= limit:
            break
    return matches[:limit]


def enrich_query_assembly(
    assembly: dict[str, Any],
    vocabulary: list[str] | None = None,
    *,
    min_score: float = DEFAULT_NEAR_MATCH_MIN_SCORE,
    per_term: int = DEFAULT_NEAR_MATCH_PER_TERM,
    limit: int = DEFAULT_NEAR_MATCH_MAX_CANDIDATES,
) -> dict[str, Any]:
    lexical_terms = list(assembly.get("lexical_terms") or [])
    vocab = vocabulary or []
    near_matches = near_match_terms(
        lexical_terms,
        vocab,
        min_score=min_score,
        limit=limit,
        per_term=per_term,
    )
    stems = []
    for term in lexical_terms:
        for stem in light_stems(term):
            stems.append({"source_term": term, "stem": stem})
    synonyms = []
    for term in lexical_terms:
        for synonym in synonym_hints(term, vocab):
            synonyms.append(
                {
                    "source_term": term,
                    "candidate_term": synonym,
                    "reason": "synonym",
                }
            )
    assembly["near_match_terms"] = near_matches
    assembly["stemmed_terms"] = stems
    assembly["synonym_terms"] = synonyms
    assembly["reformulated_query"] = reformulated_query(
        assembly.get("ranking_query"),
        lexical_terms,
        near_matches,
    )
    assembly["original_query"] = assembly.get("original_query") or assembly.get("ranking_query")
    return assembly


def reformulated_query(
    ranking_query: str | None,
    lexical_terms: list[str],
    near_matches: list[dict[str, Any]],
) -> str | None:
    if not ranking_query:
        return None
    replacements: dict[str, str] = {}
    for item in near_matches:
        source = str(item.get("source_term") or "").lower()
        candidate = str(item.get("candidate_term") or "")
        score = float(item.get("score") or 0)
        reason = item.get("reason")
        if not source or not candidate or source in replacements:
            continue
        if reason in {"light_stem", "typo_edit"} or score >= 0.9:
            replacements[source] = candidate
    if not replacements:
        return ranking_query
    tokens = re.findall(r"[A-Za-z0-9_]+|\s+|[^\sA-Za-z0-9_]+", ranking_query)
    rewritten = []
    for token in tokens:
        replacement = replacements.get(token.lower())
        rewritten.append(replacement if replacement else token)
    text = "".join(rewritten).strip()
    return text or ranking_query


def fallback_queries(assembly: dict[str, Any], *, limit: int = 8) -> list[str]:
    candidates: list[str] = []
    reformulated = assembly.get("reformulated_query")
    ranking = assembly.get("ranking_query")
    if reformulated and reformulated != ranking:
        candidates.append(str(reformulated))
    candidates.extend(assembly.get("exact_phrases") or [])
    candidates.extend(
        item["candidate_term"]
        for item in assembly.get("near_match_terms") or []
        if isinstance(item, dict) and item.get("candidate_term")
    )
    candidates.extend(
        item["candidate_term"]
        for item in assembly.get("synonym_terms") or []
        if isinstance(item, dict) and item.get("candidate_term")
    )
    candidates.extend(sorted(assembly.get("lexical_terms") or [], key=len, reverse=True))
    return dedupe_preserve_order(candidates)[:limit]


def is_content_term(term: str) -> bool:
    return len(term) >= 4


def dedupe_preserve_order(values) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        key = str(value).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(str(value))
    return deduped
