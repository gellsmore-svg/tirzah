"""Term-anchored definition drift: find sentences that define a term, then pair
conflicting definitions of the same term for contradiction review.

In a long-lived notes corpus, real contradictions are mostly definitions that
changed over time ("T1 names the primary material baseline" vs "T1 names
stable identity-bearing structures"). They rarely sit near each other in
embedding space and rarely use a one-sided negation, so neighbour scanning plus
the lexical rule misses them. On mnemosyne_dev (2026-09-11) that pipeline found
0 of 11 hand-labelled contradictions. Comparing what each passage says a term
*is* finds them directly.

The pattern router here is the primary, deterministic stage; a model second
pass (see ``make_definition_router``) only catches phrasings it misses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from tirzah.retrieval.contradictions import claim_tokens

# Longest first so "is better understood as" wins over "is".
DEFINITION_VERBS = (
    "should be understood as", "is better understood as", "is best understood as",
    "is better read as", "is best read as", "is better approached as",
    "was reframed as", "were reframed as", "is reframed as", "are reframed as",
    "is defined as", "are defined as", "should be treated as", "is treated as", "are treated as",
    "refers to", "denotes", "names", "means", "equals", "is", "are", "was", "were", "=",
)
# "X is not merely Y" qualifies X rather than denying Y.
QUALIFIERS = ("merely", "simply", "just", "only", "primarily", "mainly", "necessarily", "fundamentally")
# Text before the term that puts the definition in a hypothetical or reported
# frame: "a claim that X is", "if X is", "Readers assume X is", "one of the
# assumptions ... is that X is", "In conventional physics, X is".
FRAME_PATTERN = re.compile(
    r"\b(?:a claim that|claims? that|claiming that|if|whether|as though|as if|suppose|imagine|unless|"
    r"assumes?|assumed|assuming|assumptions?|believes?|believed|thinks? that|thought that|"
    r"in (?:conventional|common|standard|mainstream|classical|popular) \w+)\b", re.I
)
# A body that reports how X is usually talked about rather than saying what it
# is: "X is often treated as ...", "X is spoken of as though ...".
REPORTED_BODY = re.compile(
    r"^(?:(?:often|commonly|usually|typically|traditionally|conventionally|popularly|widely)\s+"
    r"(?:treated|thought|described|seen|regarded|taken|assumed|said|spoken|understood|pictured|imagined)|"
    r"(?:spoken of|thought of|talked about|treated|described)\s+as\s+(?:though|if))\b",
    re.I,
)
# Change over time. Contrast ("A rather than B") is not revision.
REVISION_PATTERN = re.compile(
    r"\b(?:no longer|reframed|redefined|re-defined|now names|now treated|previously|formerly|"
    r"used to (?:be|mean|name)|was wrong|were wrong|corrected|replaces?)\b",
    re.I,
)
DENIAL_LEADS = r"(?:the point is not that|it is not that|this does not mean that|not that)"
# A bullet under a lead-in line like these reports a view rather than stating one.
LEAD_IN_FRAME_PATTERN = re.compile(
    r"\b(?:not|no longer|never|rejects?|rejected|as though|as if|avoids?|instead of|rather than|should not|"
    r"must not|misreadings?|mistakes?|errors?|objections?|misconceptions?|myths?|wrong|incorrect|"
    r"old(?:er)? views?|claims? that|comparators?)\b",
    re.I,
)
# Sections whose heading marks them as reported, rejected or hypothetical views.
TITLE_FRAME_PATTERN = re.compile(
    r"\b(?:comparators?|objections?|misreadings?|misconceptions?|myths?|failure modes?|anti-?patterns?|"
    r"rejected (?:views?|models?)|what [\w\s]{0,40} is not|errors? to avoid|as usually presented|mainstream|"
    r"conventional|standard (?:views?|models?|accounts?)|textbook (?:views?|accounts?)|"
    r"replaced (?:assumptions?|views?|models?|claims?)|(?:old|prior|common|mainstream) assumptions?|"
    r"guardrails?|readers?'? starting points?)\b",
    re.I,
)
# Verbs that introduce a definition rather than a passing predication.
STRONG_VERBS = frozenset(verb for verb in DEFINITION_VERBS if verb not in {"is", "are", "was", "were"})
DETERMINERS = frozenset({"the", "a", "an"})
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# Sentences kept either side of a definition as its context for the confirmer.
CONTEXT_SENTENCES = 2
CONTEXT_CHARS = 800
# Definitions rarely run this long; run-on sentences are mostly transcribed speech.
RUN_ON_WORDS = 60

_VERB_ALT = "|".join(
    re.escape(verb) if verb == "=" else rf"{re.escape(verb)}\b"
    for verb in sorted(DEFINITION_VERBS, key=len, reverse=True)
)
_QUAL_ALT = "|".join(QUALIFIERS)
_PREDICATE = (
    rf"(?P<pre>no longer\s+|not\s+|never\s+)?(?P<verb>{_VERB_ALT})\s+"
    rf"(?P<post>(?:not|no longer|never)\s+(?:(?P<qual>{_QUAL_ALT})\s+)?)?(?P<body>[^.;\n]{{6,220}})"
)
# "X names A. It no longer names B." -- the second sentence still defines X.
_CONTINUATION = re.compile(rf"^(?:it|they|this term|the term)\s+{_PREDICATE}", re.I)
# "X is not A but B": the denial covers A, and B is asserted.
_DENIAL_TAIL = re.compile(r",?\s+but\s+(?:rather\s+)?|\s+[—–]\s+(?:it|they)\s+(?:is|are)\s+", re.I)
# "X is A, not B" / "A rather than B": B is the reading X is contrasted with.
_CONTRAST = re.compile(r"[,;:]\s+(?:and\s+)?not\s+|\s+rather than\s+|\s+instead of\s+", re.I)


def term_key(term: str) -> str:
    return term.strip().lower()


def term_regex(term: str) -> str:
    return rf"(?<![\w-]){re.escape(term.strip())}(?:e?s)?(?![\w-])"


def clean_sentence(text: str) -> str:
    text = re.sub(r"[*_`]{1,3}", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)
    return " ".join(text.split()).lstrip("-*>• ").strip()


def _sentences_with_frames(text: str) -> list[tuple[str, bool]]:
    """Sentences, each flagged when it is a bullet under a lead-in line (one
    ending in ':') that frames the list as reported, rejected or hypothetical,
    e.g. "The book should no longer speak as though:"."""
    sentences: list[tuple[str, bool]] = []
    lead_framed = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_bullet = bool(_BULLET.match(stripped))
        if not is_bullet:
            lead_framed = stripped.endswith(":") and bool(LEAD_IN_FRAME_PATTERN.search(stripped))
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z`*\"'(>])", stripped):
            cleaned = clean_sentence(part)
            if len(cleaned) >= 8:
                sentences.append((cleaned, is_bullet and lead_framed))
    return sentences


def split_sentences(text: str) -> list[str]:
    return [sentence for sentence, _framed in _sentences_with_frames(text)]


def extract_definitions(
    text: str, terms: Iterable[str], *, title: str | None = None, frame_titles: Iterable[re.Pattern[str]] = ()
) -> list[dict[str, Any]]:
    """Sentences in ``text`` that define one of ``terms``.

    Each definition carries ``polarity`` (``asserts``, ``denies`` or
    ``qualifies``), the defining ``verb`` and ``body``, ``framed`` (a reported,
    rejected or hypothetical view: "a claim that X is ...", a bullet under a
    framing lead-in, or a section ``title`` such as "Comparator ..." or one
    matching ``frame_titles``), and
    ``revision`` (a revision marker such as "no longer" in the sentence or the
    one after it).
    """
    sentences = _sentences_with_frames(text)
    title_framed = bool(title and (TITLE_FRAME_PATTERN.search(title) or any(p.search(title) for p in frame_titles)))
    found: list[dict[str, Any]] = []
    for index, (sentence, lead_framed) in enumerate(sentences):
        outer_frame = lead_framed or title_framed
        following = sentences[index + 1][0] if index + 1 < len(sentences) else ""
        revision = bool(REVISION_PATTERN.search(sentence) or REVISION_PATTERN.search(following))
        for term in terms:
            rx = term_regex(term)
            denied_spans = []
            for m in re.finditer(rf"\b{DENIAL_LEADS}\s+(?P<term>{rx})\s+(?:is|are)\s+(?P<body>[^.;\n]{{3,220}})", sentence, re.I):
                denied_spans.append(m.span("term"))
                found.extend(_split_denial(
                    _definition(term, sentence, "denies", "is", m.group("body"), outer_frame, revision, index)))
            # Inverted definitions: "... is what we call X", "... is called X", "... known as X".
            for m in re.finditer(
                rf"(?P<body>[^.;\n]{{6,220}}?)\s+(?:is|are)\s+(?:what (?:we|you|people) (?:call|mean by)|called|known as)"
                rf"\s+(?:\w+\s+)?(?P<term>{rx})",
                sentence,
                re.I,
            ):
                body = m.group("body").split(",")[-1].strip()
                body = re.sub(r"^(?:and|but|so)\s+", "", body, flags=re.I)
                framed = bool(FRAME_PATTERN.search(sentence[: m.start("body")]))
                found.append(_definition(term, sentence, "asserts", "is what we call", body, framed or outer_frame, revision, index))
            pattern = rf"(?P<term>{rx})\s*(?:,\s*(?:in|within)\s+[\w\s-]{{1,30}},\s*)?{_PREDICATE}"
            before = len(found)
            for m in re.finditer(pattern, sentence, re.I):
                if any(start <= m.start("term") < end for start, end in denied_spans):
                    continue
                framed = bool(FRAME_PATTERN.search(sentence[: m.start("term")]) or REPORTED_BODY.match(m.group("body")))
                found.extend(_split_denial(
                    _definition(term, sentence, _polarity(m), m.group("verb").lower(), m.group("body"),
                                framed or outer_frame, revision, index)
                ))
            if len(found) > before or denied_spans:
                continue
            m = _CONTINUATION.match(sentence)
            prior = next((d for d in reversed(found) if d["term"] == term and d["sentence_index"] == index - 1), None)
            if m and prior:
                # Keep the defining sentence with it so the pair reads on its own.
                sentence_pair = f"{sentences[index - 1][0]} {sentence}"
                found.extend(_split_denial(
                    _definition(term, sentence_pair, _polarity(m), m.group("verb").lower(), m.group("body"),
                                prior["framed"] or outer_frame, revision, index)
                ))
    for definition in found:
        definition["context"] = _window([s for s, _framed in sentences], definition["sentence_index"])
    return found


def _window(sentences: list[str], index: int | None) -> str:
    if index is None:
        return " ".join(sentences)[:CONTEXT_CHARS]
    start = max(0, index - CONTEXT_SENTENCES)
    return " ".join(sentences[start : index + CONTEXT_SENTENCES + 1])[:CONTEXT_CHARS]


def _split_denial(definition: dict[str, Any]) -> list[dict[str, Any]]:
    """"X is not A but B" denies A and asserts B."""
    if definition["polarity"] != "denies":
        return [definition]
    parts = _DENIAL_TAIL.split(definition["body"], maxsplit=1)
    if len(parts) < 2 or len(parts[1].strip()) < 6:
        return [definition]
    return [{**definition, "body": parts[0].strip(" ,:")},
            {**definition, "polarity": "asserts", "body": parts[1].strip(" ,:")}]


def _polarity(m: re.Match[str]) -> str:
    if m.group("qual"):
        return "qualifies"
    if m.group("pre") or m.group("post"):
        return "denies"
    return "asserts"


def _definition(term, sentence, polarity, verb, body, framed, revision, index) -> dict[str, Any]:
    return {
        "term": term,
        "term_key": term_key(term),
        "sentence": sentence,
        "polarity": polarity,
        "verb": verb,
        "body": body.strip(" ,:"),
        "framed": framed,
        "revision": revision,
        "sentence_index": index,
        "source": "pattern",
    }


def body_overlap(left: str, right: str) -> float:
    return _token_overlap(claim_tokens({"text": left}), claim_tokens({"text": right}))


def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def definition_strength(definition: dict[str, Any], tokens: set[str]) -> float:
    """How much a sentence reads as a definition rather than a passing
    predication: strong defining verbs and "X is the/a ..." count for it;
    "X is stored/ordered ...", very short asserted bodies and run-on
    sentences (transcribed speech) count against. Short denials are normal
    ("X is not Y") and are not penalised."""
    words = definition["body"].strip().lower().split()
    first = words[0] if words else ""
    strength = 1.0 if definition["verb"] in STRONG_VERBS else 0.0
    if first in DETERMINERS:
        strength += 0.5
    if definition["verb"] in {"is", "are", "was", "were"} and first.endswith(("ed", "ing")):
        strength -= 1.0
    if definition["polarity"] == "asserts" and len(tokens) < 3:
        strength -= 1.0
    if len(definition["sentence"].split()) > RUN_ON_WORDS:
        strength -= 1.5
    return strength


def _normalized(sentence: str) -> str:
    return re.sub(r"[^\w\s]", "", sentence.lower()).strip()


def definition_pairs(
    definitions: Iterable[dict[str, Any]],
    *,
    pairs_per_term: int = 10,
    include_same_document: bool = True,
    min_score: float = 1.0,
    max_definitions_per_term: int = 400,
    max_pairs_per_definition: int = 2,
) -> list[dict[str, Any]]:
    """Rank pairs of conflicting definitions of the same term.

    Framed definitions and qualifications are left out, and copies of the same
    sentence are collapsed. Agreeing definitions (both assert, heavy overlap)
    never pair. Scoring favours an asserted definition that another passage
    denies, then revision markers, then genuinely different definitions, each
    weighted by ``definition_strength``. No definition appears in more than
    ``max_pairs_per_definition`` selected pairs, so one sentence cannot fill a
    term's quota. Each definition needs ``node_id`` and ``document_id``;
    ``origin_date`` orders the pair older-first when both are known.
    """
    by_term: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    for definition in definitions:
        if definition.get("framed") or definition.get("polarity") == "qualifies":
            continue
        key = (_normalized(definition["sentence"]), definition["polarity"], _normalized(definition["body"]))
        bucket = by_term[definition["term_key"]]
        if key in bucket:
            bucket[key]["copies"] = bucket[key].get("copies", 1) + 1
        else:
            bucket[key] = {**definition, "copies": 1}
    ranked: list[dict[str, Any]] = []
    for term, bucket in by_term.items():
        # Compare what each sentence says X is; keep what it contrasts X with apart.
        scored = []
        for item in bucket.values():
            parts = _CONTRAST.split(item["body"], maxsplit=1)
            head = claim_tokens({"text": parts[0]})
            scored.append((item, head, claim_tokens({"text": parts[1]}) if len(parts) > 1 else set(),
                           definition_strength(item, head)))
        # Past max_definitions_per_term, keep revisions, then the most
        # definitional sentences, then denials, then the most repeated.
        scored.sort(key=lambda s: (not s[0]["revision"], -s[3], s[0]["polarity"] != "denies", -s[0].get("copies", 1)))
        scored = scored[:max_definitions_per_term]
        items = [s[0] for s in scored]
        tokens, contrast, strength = [s[1] for s in scored], [s[2] for s in scored], [s[3] for s in scored]
        pairs = []
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                b = items[j]
                if a.get("node_id") == b.get("node_id"):
                    continue
                same_document = a.get("document_id") == b.get("document_id")
                if same_document and not include_same_document:
                    continue
                overlap = _token_overlap(tokens[i], tokens[j])
                if a["polarity"] == b["polarity"] == "asserts" and overlap >= 0.8:
                    continue  # the same definition, restated
                if a["polarity"] == b["polarity"] == "denies" and not (a["revision"] or b["revision"]):
                    continue  # two denials usually agree
                if a["polarity"] != b["polarity"]:
                    denied, asserted = (i, j) if a["polarity"] == "denies" else (j, i)
                    if _token_overlap(tokens[denied], contrast[asserted]) >= 0.5:
                        continue  # both reject the same reading: "X is not B" / "X is A, not B"
                score, reasons = 0.0, []
                if a["polarity"] != b["polarity"]:
                    if overlap >= 0.4:
                        score += 3.0
                        reasons.append("asserted_and_denied")
                    else:
                        score += 1.0
                        reasons.append("opposite_polarity")
                if a["revision"] or b["revision"]:
                    score += 2.0
                    reasons.append("revision_marker")
                if a["polarity"] == b["polarity"] == "asserts":
                    score += 1.5 * (1.0 - overlap)
                    if overlap < 0.3:
                        reasons.append("different_definitions")
                score += strength[i] + strength[j]
                if strength[i] >= 1.0 and strength[j] >= 1.0:
                    reasons.append("strong_definitions")
                if not same_document:
                    score += 0.25
                if score < min_score:
                    continue
                first, second = (a, b)
                if a.get("origin_date") and b.get("origin_date") and b["origin_date"] < a["origin_date"]:
                    first, second = b, a
                pairs.append({"term": term, "score": round(score, 3), "reasons": reasons,
                              "body_overlap": round(overlap, 3), "a": first, "b": second})
        # Ties go to explicit denials; then spread picks across definitions.
        pairs.sort(key=lambda pair: (-pair["score"], "asserted_and_denied" not in pair["reasons"]))
        used: dict[str, int] = defaultdict(int)
        selected = 0
        for pair in pairs:
            if selected >= pairs_per_term:
                break
            keys = (_normalized(pair["a"]["sentence"]), _normalized(pair["b"]["sentence"]))
            if any(used[key] >= max_pairs_per_definition for key in keys):
                continue
            for key in keys:
                used[key] += 1
            selected += 1
            ranked.append(pair)
    return ranked


# --------------------------------------------------------------------------- #
# Corpus index, model second pass, report
# --------------------------------------------------------------------------- #

DEFINITION_DRIFT_SOURCE = "definition_drift"
DEFINITION_ROUTER_STEP_NAME = "definition_router"
ROUTER_PROMPT = """You are indexing a research-notes corpus by the terms it defines.

Term: {term}

Passage:
{p}

Does this passage state, revise, or reject what "{term}" means or what it is? Passages that deny a meaning ("{term} is not ...", "the point is not that {term} is ...") count.
Answer YES or NO on the first line. If YES, copy the one sentence that does it on the second line."""
INDEXED_LABELS = ("source_chunk", "source_section")


def _node_fields(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": str(node.get("_id")),
        "document_id": str(node.get("document_id")),
        "title": node.get("title") or "",
        "origin_date": node.get("origin_date"),
    }


def definition_index(
    db: Any, terms: list[str], *, second_pass_limit: int = 20, frame_titles: Iterable[str] = ()
) -> dict[str, Any]:
    """Definitions of each term across active chunks and sections.

    Chunks are read before sections, so when a sentence appears in both, the
    duplicate collapse in ``definition_pairs`` keeps the chunk. Chunks whose
    title mentions the term but that yielded no usable definition are returned
    (up to ``second_pass_limit`` per term) for the model second pass.
    """
    from tirzah.models.ingestion import INACTIVE_RETRIEVAL_STATUSES

    # Extra section titles (regexes) whose definitions report someone else's view.
    patterns = [re.compile(pattern, re.I) for pattern in frame_titles]
    definitions: list[dict[str, Any]] = []
    second_pass: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, dict[str, int]] = {}
    for term in terms:
        key = term_key(term)
        title_rx = re.compile(term_regex(term), re.I)
        nodes = list(
            db.nodes.find(
                {
                    "labels": {"$in": list(INDEXED_LABELS)},
                    "status": {"$nin": list(INACTIVE_RETRIEVAL_STATUSES)},
                    "text": {"$regex": rf"\b{re.escape(term.strip())}", "$options": "i"},
                },
                {"text": 1, "title": 1, "document_id": 1, "origin_date": 1, "labels": 1},
            )
        )
        nodes.sort(key=lambda node: "source_section" in (node.get("labels") or []))
        term_stats = {"nodes": len(nodes), "definitions": 0, "framed": 0, "qualified": 0}
        unrouted: list[dict[str, Any]] = []
        for node in nodes:
            found = [d for d in extract_definitions(node.get("text") or "", [term], title=node.get("title"),
                                                    frame_titles=patterns)
                     if d["term_key"] == key]
            usable = [d for d in found if not d["framed"]]
            term_stats["framed"] += len(found) - len(usable)
            term_stats["qualified"] += sum(d["polarity"] == "qualifies" for d in usable)
            term_stats["definitions"] += len(usable)
            definitions.extend({**d, **_node_fields(node)} for d in usable)
            if not usable and "source_chunk" in (node.get("labels") or []) and title_rx.search(node.get("title") or ""):
                unrouted.append(node)
        second_pass[key] = unrouted[:second_pass_limit]
        stats[key] = {**term_stats, "second_pass_candidates": len(unrouted)}
    return {"definitions": definitions, "second_pass_candidates": second_pass, "stats": stats}


def _sentence_in_text(quoted: str, text: str) -> str | None:
    if len(quoted) < 12:
        return None
    if _normalized(quoted) in _normalized(clean_sentence(text)):
        return quoted
    for sentence in split_sentences(text):
        if body_overlap(quoted, sentence) >= 0.7:
            return sentence
    return None


def _first_sentence_with(term: str, text: str) -> str | None:
    rx = re.compile(term_regex(term), re.I)
    return next((s for s in split_sentences(text) if rx.search(s)), None)


def make_definition_router(runtime_config: Any, *, db: Any = None, session_id: str = "consolidation"):
    """Model second pass for passages the patterns did not route.

    Returns a callable ``(term, node) -> verdict`` whose ``status`` is
    ``defines`` (with the quoted ``sentence``), ``not_definition``,
    ``unparsed`` or ``unavailable``; None when disabled. The question is closed
    over one term, so the model never has to name the term itself.
    """
    if runtime_config is None or not getattr(runtime_config, "definition_second_pass_enabled", True):
        return None
    from tirzah.retrieval.contradictions import bounded_model_caller

    call = bounded_model_caller(runtime_config, step_name=DEFINITION_ROUTER_STEP_NAME, db=db, session_id=session_id)

    def route(term: str, node: dict[str, Any]) -> dict[str, Any]:
        text = str(node.get("text") or "")
        answer, info = call(ROUTER_PROMPT.format(term=term, p=f"{node.get('title') or ''}\n{text[:1000]}".strip()))
        if answer is None:
            return info
        lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
        first = lines[0].upper().lstrip("*#> ") if lines else ""
        if first.startswith("NO"):
            return {**info, "status": "not_definition"}
        if not first.startswith("YES"):
            return {**info, "status": "unparsed", "answer": answer[:200]}
        quoted = clean_sentence(lines[1].strip("\"“”'")) if len(lines) > 1 else ""
        return {**info, "status": "defines", "sentence": _sentence_in_text(quoted, text) or _first_sentence_with(term, text)}

    route.trace_id = call.trace_id  # type: ignore[attr-defined]
    return route


DEFINITION_CONFIRMATION_STEP_NAME = "definition_confirmation"
# The general contradiction prompt counts "different aspects" as compatible,
# which is exactly how a redefinition reads, so definition pairs get their
# own closed question about the term.
DEFINITION_CONFIRMATION_PROMPT = """You are checking how a research-notes corpus defines the term "{term}".

Definition A (from "{title_a}"):
{sentence_a}

Definition B (from "{title_b}"):
{sentence_b}

Context around A:
{context_a}

Context around B:
{context_b}

Do A and B give incompatible accounts of what "{term}" is or names, so that if one is the current definition the other must be revised or retired?
Different words for the same idea, or one adding detail to the other, are NOT incompatible.
One denying what the other asserts about "{term}", or the two naming different things, ARE incompatible, even when both are phrased positively.
Answer on the first line with exactly one word:
CONFLICT - incompatible definitions of "{term}"
SAME - the same definition, restated or elaborated
UNRELATED - one of them does not actually define "{term}" (it describes something else, an analogy, or someone else's view)
Then give a one-sentence reason."""
DEFINITION_LABEL_PATTERN = re.compile(r"\b(CONFLICT|SAME|UNRELATED)\b")
DEFINITION_CONTEXT_CHARS = 400


def definition_confirmation_prompt(source: dict[str, Any], target: dict[str, Any]) -> str:
    return DEFINITION_CONFIRMATION_PROMPT.format(
        term=source.get("term") or target.get("term") or "",
        title_a=source.get("title") or "", sentence_a=source.get("sentence") or source.get("text") or "",
        title_b=target.get("title") or "", sentence_b=target.get("sentence") or target.get("text") or "",
        context_a=(source.get("context") or "")[:DEFINITION_CONTEXT_CHARS],
        context_b=(target.get("context") or "")[:DEFINITION_CONTEXT_CHARS],
    )


def make_definition_confirmer(runtime_config: Any, *, db: Any = None, session_id: str = "consolidation"):
    """The model confirmation stage for definition pairs, or None when
    ``contradiction_confirmation_enabled`` is off.

    Takes two ``_passage`` dicts and returns a verdict whose ``status`` is
    ``confirmed`` (CONFLICT), ``rejected`` (SAME or UNRELATED), ``unparsed`` or
    ``unavailable``, like ``make_contradiction_confirmer``, and shares its
    adapter, model, circuit breaker and ``llm_calls`` recording.
    """
    if runtime_config is None or not getattr(runtime_config, "contradiction_confirmation_enabled", True):
        return None
    from tirzah.retrieval.contradictions import bounded_model_caller

    call = bounded_model_caller(runtime_config, step_name=DEFINITION_CONFIRMATION_STEP_NAME, db=db,
                                session_id=session_id)

    def confirm(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        answer, info = call(definition_confirmation_prompt(source, target))
        if answer is None:
            return info
        match = DEFINITION_LABEL_PATTERN.search(answer.upper())
        label = match.group(1) if match else "UNPARSED"
        return {
            **info,
            "status": {"CONFLICT": "confirmed", "UNPARSED": "unparsed"}.get(label, "rejected"),
            "label": label,
            "reason": answer.strip()[:300],
        }

    confirm.trace_id = call.trace_id  # type: ignore[attr-defined]
    return confirm


def model_definition(term: str, node: dict[str, Any], sentence: str) -> dict[str, Any]:
    """A definition the model second pass found, shaped like a pattern one."""
    parsed = [d for d in extract_definitions(sentence, [term]) if not d["framed"]]
    if parsed:
        base = parsed[0]
    else:
        polarity = "denies" if re.search(r"\b(?:not|no longer|never)\b", sentence, re.I) else "asserts"
        base = _definition(term, clean_sentence(sentence), polarity, "model", clean_sentence(sentence), False,
                           bool(REVISION_PATTERN.search(sentence)), 0)
    sentences = split_sentences(node.get("text") or "")
    index = next((i for i, s in enumerate(sentences) if s == clean_sentence(sentence)), None)
    return {**base, "source": "model", **_node_fields(node), "context": _window(sentences, index)}


def _passage(definition: dict[str, Any]) -> dict[str, Any]:
    # Lead with the defining sentence so a general confirmer always sees it.
    context = definition.get("context") or ""
    return {"title": definition.get("title"), "term": definition.get("term"), "sentence": definition["sentence"],
            "context": context, "text": f"{definition['sentence']}\n\n{context}"}


def _slim(definition: dict[str, Any]) -> dict[str, Any]:
    return {key: definition.get(key) for key in
            ("node_id", "document_id", "title", "sentence", "polarity", "verb", "body", "source", "origin_date",
             "copies")}


def definition_drift_report(
    db: Any,
    terms: Iterable[str],
    *,
    pairs_per_term: int = 10,
    include_same_document: bool = True,
    router: Any = None,
    second_pass_limit: int = 20,
    confirmer: Any = None,
    frame_titles: Iterable[str] = (),
) -> dict[str, Any]:
    """Candidate contradictions from definitions of ``terms`` that disagree.

    The pattern router indexes definitions; ``router`` (see
    ``make_definition_router``) classifies up to ``second_pass_limit``
    heading-matched passages per term that the patterns missed; pairs are
    ranked by ``definition_pairs``; ``confirmer`` (see
    ``make_contradiction_confirmer``) then judges each pair.
    """
    terms = [term for term in dict.fromkeys(t.strip() for t in terms) if term]
    index = definition_index(db, terms, second_pass_limit=second_pass_limit if router is not None else 0,
                             frame_titles=frame_titles)
    definitions = list(index["definitions"])
    second_pass = {"enabled": router is not None, "considered": 0, "defines": 0, "not_definition": 0,
                   "unparsed": 0, "unavailable": 0}
    by_key = {term_key(term): term for term in terms}
    for key, nodes in index["second_pass_candidates"].items():
        for node in nodes:
            verdict = router(by_key[key], node)
            status = verdict.get("status") if verdict.get("status") in second_pass else "unparsed"
            second_pass["considered"] += 1
            second_pass[status] += 1
            if status == "defines" and verdict.get("sentence"):
                definitions.append(model_definition(by_key[key], node, verdict["sentence"]))
    candidates = []
    for pair in definition_pairs(definitions, pairs_per_term=pairs_per_term, include_same_document=include_same_document):
        candidate = {key: pair[key] for key in ("term", "score", "reasons", "body_overlap")}
        candidate["a"], candidate["b"] = _slim(pair["a"]), _slim(pair["b"])
        if confirmer is not None:
            candidate["confirmation"] = confirmer(_passage(pair["a"]), _passage(pair["b"]))
        candidates.append(candidate)
    return {
        "ok": True,
        "terms": terms,
        "stats": index["stats"],
        "second_pass": second_pass,
        "confirmation": "llm" if confirmer is not None else "off",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def render_definition_drift_text(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"Definition drift: {result.get('reason')}. {result.get('hint') or ''}".strip()
    queued = result.get("enqueued_count", 0) if not result.get("dry_run", True) else result.get("would_enqueue_count", 0)
    lines = [
        f"Definition drift over {len(result.get('terms') or [])} term(s): {result.get('candidate_count', 0)} candidate "
        f"pair(s); {queued} {'queued' if not result.get('dry_run', True) else 'would be queued'} "
        f"(confirmation: {result.get('confirmation')})",
    ]
    second_pass = result.get("second_pass") or {}
    if second_pass.get("enabled"):
        lines.append(
            f"model second pass: {second_pass.get('considered', 0)} passage(s) checked, "
            f"{second_pass.get('defines', 0)} definition(s) found"
        )
    for term, stats in (result.get("stats") or {}).items():
        lines.append(
            f"  {term}: {stats.get('definitions', 0)} definition(s) in {stats.get('nodes', 0)} passage(s), "
            f"{stats.get('framed', 0)} framed set aside"
        )
    for number, candidate in enumerate(result.get("candidates") or [], start=1):
        verdict = (candidate.get("confirmation") or {}).get("status", "not_run")
        lines += [
            "",
            f"{number}. {candidate['term']} | score {candidate['score']} | {', '.join(candidate['reasons'])} | model: {verdict}",
            f"   A [{(candidate['a'].get('title') or '')[:60]}]: {(candidate['a'].get('sentence') or '')[:200]}",
            f"   B [{(candidate['b'].get('title') or '')[:60]}]: {(candidate['b'].get('sentence') or '')[:200]}",
        ]
    return "\n".join(lines)
