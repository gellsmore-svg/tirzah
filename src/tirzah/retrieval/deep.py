"""Deep retrieval mode — the fixed query-primitive menu (ADR-020).

This is the skeleton the deep retrieval agent sits on: a small, fixed set of
**validated** query primitives that the LLM selects among (it does not compose
raw Mongo queries). Each primitive has a JSON-style argument schema and a handler
that delegates to the existing, read-only `retrieval/queries.py` functions.

Every primitive call is validated as hostile input before dispatch
(`validate_primitive_call`), then run (`run_primitive`). The orchestrator loop
(plan -> validate -> execute -> gate/shortlist -> triage -> synthesise) is built
on top of this menu and is added separately.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tirzah.adapters.answer import answer_adapter
from tirzah.adapters.embedding import embedding_adapter
from tirzah.retrieval.budget import request_budget_plan
from tirzah.retrieval.queries import (
    expand_graph_paths,
    node_context,
    node_identity,
    query_embedding_candidate_nodes,
    search_nodes,
)

# Bound for any integer argument coming from the model (limits, depth).
MAX_PRIMITIVE_INT = 50

# Default deterministic stop threshold: stop a round when the share of newly-seen
# candidates falls below this (low novelty).
DEFAULT_MIN_NOVELTY = 0.1


class PrimitiveError(ValueError):
    """A primitive call was malformed (unknown name, missing/invalid argument)."""


# --------------------------------------------------------------------------- #
# Handlers — each delegates to an existing read-only query function.
# Signature: handler(db, args, *, query_embedding, identity) -> result
# --------------------------------------------------------------------------- #


def _keyword_search(db: Any, args: dict[str, Any], *, query_embedding=None, identity=None, embedder=None):
    return search_nodes(
        db, query=args["query"], label=args.get("label"), limit=args["limit"], identity=identity
    )


def _hybrid_search(db: Any, args: dict[str, Any], *, query_embedding=None, identity=None, embedder=None):
    return search_nodes(
        db,
        query=args["query"],
        label=args.get("label"),
        limit=args["limit"],
        identity=identity,
        query_embedding=query_embedding,
    )


def _semantic_search(db: Any, args: dict[str, Any], *, query_embedding=None, identity=None, embedder=None):
    # Embed this call's focused phrase if an embedder is available; otherwise fall
    # back to the session query embedding. Returns [] when no embedding is usable
    # (hybrid off / mock adapter / embed failure) so the planner degrades to lexical.
    call_embedding = None
    text = args.get("query")
    if text and embedder is not None:
        call_embedding = embedder(text)
    if call_embedding is None:
        call_embedding = query_embedding
    if call_embedding is None:
        return []
    return query_embedding_candidate_nodes(
        db, call_embedding, limit=args["limit"], label=args.get("label")
    )


def _adjacent_context(db: Any, args: dict[str, Any], *, query_embedding=None, identity=None, embedder=None):
    return node_context(db, args["node_id"], child_limit=args["child_limit"])


def _graph_traverse(db: Any, args: dict[str, Any], *, query_embedding=None, identity=None, embedder=None):
    return expand_graph_paths(db, args["node_id"], max_depth=args["depth"], limit=args["limit"])


# --------------------------------------------------------------------------- #
# The fixed menu. `args`: name -> {type, required, [default]}.
# --------------------------------------------------------------------------- #

PRIMITIVES: dict[str, dict[str, Any]] = {
    "keyword_search": {
        "description": "Lexical search over node title/text/labels.",
        "args": {
            "query": {"type": "str", "required": True},
            "label": {"type": "str", "required": False},
            "limit": {"type": "int", "required": False, "default": 10},
        },
        "handler": _keyword_search,
    },
    "hybrid_search": {
        "description": "Lexical + vector blended search (uses the query embedding when available).",
        "args": {
            "query": {"type": "str", "required": True},
            "label": {"type": "str", "required": False},
            "limit": {"type": "int", "required": False, "default": 10},
        },
        "handler": _hybrid_search,
    },
    "semantic_search": {
        "description": (
            "Pure meaning-based (vector) search — finds nodes whose meaning matches "
            "the `query`, even when they share NO keywords with it. Use for "
            "conceptual/paraphrased questions where the wording may differ from the source."
        ),
        "args": {
            "query": {"type": "str", "required": True},
            "label": {"type": "str", "required": False},
            "limit": {"type": "int", "required": False, "default": 10},
        },
        "handler": _semantic_search,
    },
    "adjacent_context": {
        "description": "The node plus its document, parent, and children.",
        "args": {
            "node_id": {"type": "str", "required": True},
            "child_limit": {"type": "int", "required": False, "default": 20},
        },
        "handler": _adjacent_context,
    },
    "graph_traverse": {
        "description": "Follow graph relationships from a node, up to a bounded depth.",
        "args": {
            "node_id": {"type": "str", "required": True},
            "depth": {"type": "int", "required": False, "default": 2},
            "limit": {"type": "int", "required": False, "default": 10},
        },
        "handler": _graph_traverse,
    },
}

_COERCE = {"str": str, "int": int}


def allowed_primitive_specs() -> list[dict[str, Any]]:
    """The menu as plain data (no handlers), for the agent prompt / introspection."""
    return [
        {
            "name": name,
            "description": prim["description"],
            "args": {arg: dict(spec) for arg, spec in prim["args"].items()},
        }
        for name, prim in PRIMITIVES.items()
    ]


def validate_primitive_call(name: str, raw_args: dict[str, Any] | None) -> dict[str, Any]:
    """Validate + normalise a primitive call. Returns the cleaned args, or raises
    PrimitiveError. Unknown args are ignored; required args must be present;
    integers are coerced and bounded."""
    if name not in PRIMITIVES:
        raise PrimitiveError(
            f"unknown primitive '{name}'. Allowed: {', '.join(sorted(PRIMITIVES))}."
        )
    raw_args = raw_args or {}
    if not isinstance(raw_args, dict):
        raise PrimitiveError(f"{name}: arguments must be an object.")
    schema = PRIMITIVES[name]["args"]
    args: dict[str, Any] = {}
    for pname, pspec in schema.items():
        if pname in raw_args and raw_args[pname] is not None:
            try:
                value = _COERCE[pspec["type"]](raw_args[pname])
            except (TypeError, ValueError):
                raise PrimitiveError(f"{name}: argument '{pname}' must be {pspec['type']}.")
            if pspec["type"] == "int":
                value = max(1, min(value, MAX_PRIMITIVE_INT))
            args[pname] = value
        elif pspec["required"]:
            raise PrimitiveError(f"{name}: missing required argument '{pname}'.")
        elif "default" in pspec:
            args[pname] = pspec["default"]
    return args


def run_primitive(
    db: Any,
    name: str,
    raw_args: dict[str, Any] | None,
    *,
    query_embedding: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
    embedder=None,
) -> Any:
    """Validate then dispatch a primitive call to its read-only handler."""
    args = validate_primitive_call(name, raw_args)
    return PRIMITIVES[name]["handler"](
        db, args, query_embedding=query_embedding, identity=identity, embedder=embedder
    )


# --------------------------------------------------------------------------- #
# Deep retrieval session — the Python-authoritative state and deterministic stop
# logic the orchestrator loop is built on (ADR-020). The LLM never owns this.
# --------------------------------------------------------------------------- #


class DeepRetrievalSession:
    """Holds the authoritative state for one deep retrieval session:
    session-scoped exclusion of already-returned chunks, the accumulating
    "useful chunks" bucket, and the deterministic stop signals (novelty,
    no-new-candidates, diminishing returns, and a hard iteration cap).
    """

    def __init__(self, *, max_iterations: int, min_novelty: float = DEFAULT_MIN_NOVELTY) -> None:
        self.max_iterations = max(1, int(max_iterations))
        self.min_novelty = min_novelty
        self.exclusion_ids: set[str] = set()
        self.useful_chunks: list[dict[str, Any]] = []
        self._useful_ids: set[str] = set()
        self.iterations = 0
        self.round_log: list[dict[str, Any]] = []

    def filter_new(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop candidates already returned this session and register the rest as
        seen. Returns only the new (session-deduped) candidates."""
        new: list[dict[str, Any]] = []
        for candidate in candidates:
            nid = node_identity(candidate)
            if not nid or nid in self.exclusion_ids:
                continue
            self.exclusion_ids.add(nid)
            new.append(candidate)
        return new

    def add_useful(self, items: list[dict[str, Any]]) -> None:
        """Add kept chunks to the useful-chunks bucket (deduped by node identity)."""
        for item in items:
            nid = node_identity(item)
            if nid and nid not in self._useful_ids:
                self._useful_ids.add(nid)
                self.useful_chunks.append(item)

    def record_round(self, *, new_count: int, total_count: int, best_score: float = 0.0) -> None:
        """Record one round's outcome and advance the iteration counter."""
        self.iterations += 1
        novelty = (new_count / total_count) if total_count else 0.0
        self.round_log.append(
            {
                "new": new_count,
                "total": total_count,
                "novelty": round(novelty, 4),
                "best_score": float(best_score),
            }
        )

    def should_stop(self) -> tuple[bool, str | None]:
        """Deterministic stop decision. Order: hard cap, then no-new, low-novelty,
        diminishing returns. Returns (stop, reason)."""
        if not self.round_log:
            return (False, None)
        if self.iterations >= self.max_iterations:
            return (True, "max_iterations")
        last = self.round_log[-1]
        if last["new"] == 0:
            return (True, "no_new_candidates")
        if last["novelty"] < self.min_novelty:
            return (True, "low_novelty")
        if self._diminishing_returns():
            return (True, "diminishing_returns")
        return (False, None)

    def _diminishing_returns(self, window: int = 2, min_gain: float = 1e-6) -> bool:
        if len(self.round_log) <= window:
            return False
        recent = self.round_log[-(window + 1):]
        if all(r["best_score"] <= 0 for r in recent):
            return False  # no score signal to judge improvement by
        gains = [recent[i + 1]["best_score"] - recent[i]["best_score"] for i in range(len(recent) - 1)]
        return all(g <= min_gain for g in gains)


# --------------------------------------------------------------------------- #
# The orchestrator loop (ADR-020 §4). Python-authoritative; the two LLM seams
# (planner, triager) are injected so the orchestration is testable without a
# model. The real prompt/parse planner + triager, synthesis, and the
# `retrieval_mode == "deep"` wiring are built on top of this.
# --------------------------------------------------------------------------- #


def _pages(items: list[Any], size: int):
    for i in range(0, len(items), max(1, size)):
        yield items[i : i + size]


def _as_candidates(results: Any) -> list[dict[str, Any]]:
    """Normalise a primitive's result into a flat list of node-like dicts (those
    carrying a node identity), so search results and context results both feed the
    shortlist."""
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict) and node_identity(r)]
    if isinstance(results, dict):
        out: list[dict[str, Any]] = []
        if node_identity(results):
            out.append(results)
        node = results.get("node")
        if isinstance(node, dict) and node_identity(node):
            out.append(node)
        for key in ("children", "nodes", "results"):
            for r in results.get(key) or []:
                if isinstance(r, dict) and node_identity(r):
                    out.append(r)
        return out
    return []


def run_deep_retrieval(
    db: Any,
    query: str,
    *,
    config: Any,
    planner,
    triager,
    scorer=None,
    query_embedding: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
    embedder=None,
) -> dict[str, Any]:
    """Run one deep retrieval session.

    Loop: **plan** (`planner`, the LLM) -> **validate + execute** (the primitive
    menu) -> **dedup + shortlist** (Python, session-scoped) -> **triage**
    (`triager`, the LLM, paged) -> **stop** (`DeepRetrievalSession`, deterministic).

    - `planner(plan_context) -> decision`: returns `{"action": "stop"}` or
      `{"primitive": name, "args": {...}}`.
    - `triager(query, page) -> kept`: returns the subset of the page to keep.

    Returns the gathered `useful_chunks` (for synthesis), the round log, and a
    trace. Guaranteed to terminate (stop signals + invalid-plan cap + hard cap).
    """
    rc = config.retrieval
    session = DeepRetrievalSession(max_iterations=rc.deep_max_iterations)
    trace: list[dict[str, Any]] = []
    invalid_attempts = 0
    hard_cap = 2 * session.max_iterations + 2
    score_history: list[float] = []
    last_sufficiency: dict[str, Any] | None = None
    scoring_on = scorer is not None and getattr(rc, "deep_sufficiency_scoring", False)

    for _ in range(hard_cap):
        plan_context = {
            "query": query,
            "iteration": session.iterations,
            "useful_count": len(session.useful_chunks),
            "seen_count": len(session.exclusion_ids),
            "primitives": allowed_primitive_specs(),
            "rounds": session.round_log,
            "sufficiency": last_sufficiency,  # dynamic planning: target the gaps
        }
        decision = planner(plan_context) or {}
        if decision.get("action") == "stop":
            trace.append({"step": "stop", "reason": "planner_stop"})
            break

        name = decision.get("primitive")
        raw_args = decision.get("args") or {}
        try:
            results = run_primitive(
                db, name, raw_args,
                query_embedding=query_embedding, identity=identity, embedder=embedder,
            )
        except PrimitiveError as exc:
            invalid_attempts += 1
            trace.append({"step": "invalid_plan", "primitive": name, "error": str(exc)})
            if invalid_attempts > session.max_iterations:
                trace.append({"step": "stop", "reason": "repeated_invalid_plans"})
                break
            continue  # bounded repair: let the planner try again

        candidates = _as_candidates(results)
        new = session.filter_new(candidates)
        shortlist = new[: max(1, rc.deep_shortlist_size)]
        kept: list[dict[str, Any]] = []
        for page in _pages(shortlist, rc.deep_page_size):
            kept.extend(triager(query, page) or [])
        session.add_useful(kept)
        session.record_round(new_count=len(new), total_count=len(candidates))
        trace.append({"step": "search", "primitive": name, "new": len(new), "kept": len(kept)})

        # Phase 4: score context sufficiency and stop when confident or plateaued.
        if scoring_on:
            sufficiency = scorer(query, session.useful_chunks, session.round_log) or {}
            value = _clamp_score(sufficiency.get("context_sufficiency_score"))
            sufficiency["context_sufficiency_score"] = value
            score_history.append(value)
            last_sufficiency = sufficiency
            trace.append({
                "step": "sufficiency",
                "recursion": session.iterations,
                "context_sufficiency_score": value,
                "remaining_uncertainty": sufficiency.get("remaining_uncertainty", []),
            })
            if value >= rc.deep_sufficiency_stop:
                trace.append({"step": "stop", "reason": "sufficiency_high"})
                break
            if value >= rc.deep_sufficiency_plateau_floor and detect_plateau(
                score_history, rc.deep_plateau_passes, rc.deep_plateau_epsilon
            ):
                trace.append({"step": "stop", "reason": "sufficiency_plateau"})
                break

        stop, reason = session.should_stop()
        if stop:
            trace.append({"step": "stop", "reason": reason})
            break

    return {
        "useful_chunks": session.useful_chunks,
        "rounds": session.round_log,
        "trace": trace,
        "sufficiency": last_sufficiency,
        "sufficiency_history": score_history,
    }


# --------------------------------------------------------------------------- #
# Real planner + triager — the two LLM seams, built on the answer adapter.
# Prompt-build + parse are separate from the model call so both are testable;
# the model output is treated as hostile input (parse failures degrade safely).
# --------------------------------------------------------------------------- #


def _extract_json(text: str) -> Any:
    """Best-effort: pull the first JSON value out of a model response (tolerating
    code fences and surrounding prose). Returns the parsed value or None."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    decoder = json.JSONDecoder(strict=False)
    for match in re.finditer(r"[{\[]", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start():])
            return parsed
        except json.JSONDecodeError:
            continue
    return None


def build_deep_planner_prompt(query: str, plan_context: dict[str, Any]) -> str:
    return (
        "You are a retrieval planner. Choose ONE primitive to call next, or stop "
        "when further search would add little.\n"
        f"User query: {query}\n"
        f"Round: {plan_context.get('iteration')}; useful chunks kept: "
        f"{plan_context.get('useful_count')}; already-seen nodes: "
        f"{plan_context.get('seen_count')}.\n"
        "Available primitives (pick exactly one):\n"
        + json.dumps(plan_context.get("primitives", []), indent=2)
        + "\n\nGuidance:\n"
        "- For `keyword_search`/`hybrid_search`, use a SHORT, focused keyword or "
        "key term as the `query` (e.g. \"vorton\"), NOT the full question.\n"
        "- For `semantic_search` (meaning-based), prefer a fuller conceptual phrase "
        "or the question itself — it matches meaning, not exact words, so it can "
        "find relevant nodes that use different terminology.\n"
        "- If keyword/hybrid search returned little that was new, try "
        "`semantic_search` (or a different term) rather than repeating the same search.\n"
        "- Stop once you have enough to answer.\n"
        '\nReply with ONLY a JSON object, one of:\n'
        '  {"action": "stop"}\n'
        '  {"primitive": "<name>", "args": { ... }, "rationale": "<why>"}\n'
    )


def parse_deep_decision(text: str) -> dict[str, Any]:
    """Parse a planner response into a decision the orchestrator understands.
    A `stop` is honoured; a usable primitive call is returned as-is; anything
    unparseable becomes an unknown-primitive call so the loop's bounded
    invalid-plan repair handles it (rather than silently stopping)."""
    data = _extract_json(text)
    if isinstance(data, dict) and data.get("action") == "stop":
        return {"action": "stop"}
    if isinstance(data, dict) and isinstance(data.get("primitive"), str) and data["primitive"]:
        args = data.get("args")
        return {"primitive": data["primitive"], "args": args if isinstance(args, dict) else {}}
    return {"primitive": "__unparseable__", "args": {}}


def build_deep_triage_prompt(query: str, page: list[dict[str, Any]]) -> str:
    items = [
        {
            "node_id": node_identity(c),
            "title": c.get("title"),
            "preview": c.get("text_preview") or c.get("text"),
        }
        for c in page
    ]
    return (
        "Keep only the candidate chunks genuinely relevant to the query.\n"
        f"Query: {query}\n"
        "Candidates:\n" + json.dumps(items, indent=2)
        + '\n\nReply with ONLY a JSON array of the node_id strings to KEEP, '
        'e.g. ["id1", "id3"]. Keep none with [].'
    )


def parse_deep_triage(text: str, page: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data = _extract_json(text)
    if not isinstance(data, list):
        return []
    keep = {str(x) for x in data if isinstance(x, (str, int))}
    return [c for c in page if node_identity(c) in keep]


def _adapter_answer_result(adapter: Any, prompt_text: str) -> dict[str, Any]:
    """Call an answer adapter and return the full payload (incl. usage/duration).

    Deep planner/triager/scorer only need the text; synthesis and the
    process-trace path need instrumentation for galeed llm_calls.
    """
    result = adapter.answer(
        {
            "prompt_text": prompt_text,
            "context_text": "",
            "context_metadata": {"included": []},
        }
    )
    if isinstance(result, dict):
        return result
    return {"answer": str(result)}


def _adapter_answer(adapter: Any, prompt_text: str) -> str:
    result = _adapter_answer_result(adapter, prompt_text)
    return str(result.get("answer") or "")


def make_planner(adapter: Any):
    """Wrap an answer adapter into a `planner(plan_context) -> decision`."""

    def planner(plan_context: dict[str, Any]) -> dict[str, Any]:
        prompt = build_deep_planner_prompt(plan_context.get("query", ""), plan_context)
        return parse_deep_decision(_adapter_answer(adapter, prompt))

    return planner


def make_triager(adapter: Any):
    """Wrap an answer adapter into a `triager(query, page) -> kept`."""

    def triager(query: str, page: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not page:
            return []
        prompt = build_deep_triage_prompt(query, page)
        return parse_deep_triage(_adapter_answer(adapter, prompt), page)

    return triager


# --------------------------------------------------------------------------- #
# Phase 4: Context Sufficiency Score — drives recursive stopping + dynamic planning.
# --------------------------------------------------------------------------- #

SUFFICIENCY_COMPONENTS = (
    "relevance",
    "coverage",
    "focus",
    "continuity",
    "process_fit",
    "decision_completeness",
)


def _clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def heuristic_sufficiency(useful_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic fallback score. Grows with the useful-chunks bucket but caps
    below the hard stop (9.0) so it never claims certainty on its own."""
    count = len(useful_chunks)
    score = min(8.5, 3.5 + 1.0 * count)
    return {
        "context_sufficiency_score": round(score, 1),
        "remaining_uncertainty": [] if count else ["no relevant context retrieved yet"],
        "source": "heuristic",
    }


def build_sufficiency_prompt(query: str, useful_chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for chunk in useful_chunks[:12]:
        title = chunk.get("title") or ""
        text = (chunk.get("text") or chunk.get("text_preview") or "")[:300]
        blocks.append(f"[{node_identity(chunk)}] {title}\n{text}".strip())
    context = "\n\n".join(blocks) if blocks else "(no context retrieved yet)"
    return (
        "Rate how SUFFICIENT the retrieved context is to answer the question well, 0-10. "
        'Return ONLY JSON: {"context_sufficiency_score": <0-10>, "relevance": <0-10>, '
        '"coverage": <0-10>, "focus": <0-10>, "continuity": <0-10>, "process_fit": <0-10>, '
        '"decision_completeness": <0-10>, "remaining_uncertainty": ["..."]}. '
        "Score high only if the context actually answers the question; list what is still "
        "missing in remaining_uncertainty.\n\n"
        f"Question: {query}\n\nRetrieved context:\n{context}\n"
    )


def parse_sufficiency(text: str) -> dict[str, Any]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        raise ValueError("sufficiency score is not a JSON object")
    score: dict[str, Any] = {"context_sufficiency_score": _clamp_score(payload.get("context_sufficiency_score"))}
    for component in SUFFICIENCY_COMPONENTS:
        if component in payload:
            score[component] = _clamp_score(payload[component])
    uncertainty = payload.get("remaining_uncertainty")
    score["remaining_uncertainty"] = (
        [str(item) for item in uncertainty][:8] if isinstance(uncertainty, list) else []
    )
    score["source"] = "llm"
    return score


def make_scorer(adapter: Any):
    """Wrap an answer adapter into a scorer(query, useful_chunks, rounds) -> score."""

    def scorer(query: str, useful_chunks: list[dict[str, Any]], rounds: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            return parse_sufficiency(_adapter_answer(adapter, build_sufficiency_prompt(query, useful_chunks)))
        except Exception:
            return heuristic_sufficiency(useful_chunks)

    return scorer


def detect_plateau(history: list[float], passes: int, epsilon: float) -> bool:
    """True when the score has improved by <= epsilon over the last `passes` passes."""
    if passes <= 0 or len(history) <= passes:
        return False
    return (history[-1] - history[-1 - passes]) <= epsilon


# --------------------------------------------------------------------------- #
# Synthesis + the full deep flow (retrieve -> synthesise). Self-contained so
# `deep.py` takes no dependency on the interaction layer (which will call this).
# --------------------------------------------------------------------------- #


def build_synthesis_prompt(
    query: str, useful_chunks: list[dict[str, Any]], history_block: str = ""
) -> str:
    """The exact prompt deep synthesis sends to the model — exposed so the
    answer pipeline can record the full input in the LLM debugging view."""
    prefix = (history_block.rstrip() + "\n\n") if history_block else ""
    if not useful_chunks:
        return (
            prefix
            + "Answer the question. If there is no relevant information available, say so plainly.\n\n"
            f"Question: {query}\n"
        )
    return _context_synthesis_prompt(
        query, [_synthesis_block(chunk) for chunk in useful_chunks], history_block
    )


def _synthesis_block(chunk: dict[str, Any]) -> str:
    text = chunk.get("text") or chunk.get("text_preview") or ""
    title = chunk.get("title") or ""
    return f"[{node_identity(chunk)}] {title}\n{text}".strip()


def _context_synthesis_prompt(query: str, blocks: list[str], history_block: str = "") -> str:
    prefix = (history_block.rstrip() + "\n\n") if history_block else ""
    return (
        prefix
        + "Answer the question using ONLY the context below. Cite the [node_id] sources you "
        "use. If the context is insufficient, say so plainly.\n\n"
        f"Context:\n{chr(10).join(blocks)}\n\nQuestion: {query}\n"
    )


def pack_synthesis_chunks(
    query: str,
    useful_chunks: list[dict[str, Any]],
    plan: Any,
    history_block: str = "",
) -> dict[str, Any]:
    """Bound the synthesis context to the request's budget plan.

    Keeps chunks in rank order while the rendered context fits
    ``plan.context_char_budget`` and the whole prompt fits the prompt token
    budget less the reserved response. The first chunk is truncated rather than
    dropped so evidence never vanishes entirely; every skipped chunk is listed
    so the trace shows what was left out. ``plan=None`` keeps everything.
    """
    chunks = list(useful_chunks)
    if plan is None:
        return {"kept": chunks, "skipped": [], "budget": {}}
    available_tokens = max(0, plan.prompt_token_budget - plan.reserved_response_tokens)
    char_budget = plan.context_char_budget
    overhead_tokens = plan.count(_context_synthesis_prompt(query, [], history_block))
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    truncated = 0
    used_chars = 0
    used_tokens = overhead_tokens
    for chunk in chunks:
        block = _synthesis_block(chunk)
        block_tokens = plan.count(block + "\n")
        fits = used_chars + len(block) <= char_budget and used_tokens + block_tokens <= available_tokens
        if not fits and not kept:
            header_chars = len(_synthesis_block({**chunk, "text": "", "text_preview": ""})) + 1
            token_room_chars = int(max(0, available_tokens - overhead_tokens) * plan.chars_per_token)
            room = min(char_budget, token_room_chars) - header_chars
            if room > 0:
                text = str(chunk.get("text") or chunk.get("text_preview") or "")
                chunk = {**chunk, "text": text[:room], "truncated_from_chars": len(text)}
                block = _synthesis_block(chunk)
                block_tokens = plan.count(block + "\n")
                truncated += 1
                fits = True
        if not fits:
            skipped.append(
                {
                    "node_id": node_identity(chunk),
                    "title": chunk.get("title"),
                    "chars": len(block),
                    "reason": "prompt_budget",
                    "included_as": "omitted",
                }
            )
            continue
        kept.append(chunk)
        used_chars += len(block)
        used_tokens += block_tokens
    prompt_tokens = plan.count(build_synthesis_prompt(query, kept, history_block))
    return {
        "kept": kept,
        "skipped": skipped,
        "budget": {
            "token_budget": plan.prompt_token_budget,
            "reserved_response_tokens": plan.reserved_response_tokens,
            "estimated_overhead_tokens": overhead_tokens,
            "available_context_tokens": max(0, available_tokens - overhead_tokens),
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_total_with_reserved_response_tokens": prompt_tokens
            + plan.reserved_response_tokens,
            "tokenizer": plan.tokenizer,
            "tokenizer_encoding": plan.tokenizer_encoding,
            "chars_per_token": plan.chars_per_token,
            "profile_key": plan.profile_key,
            "model": plan.model,
            "adapter": plan.adapter,
            "context_char_budget": char_budget,
            "used_chars": used_chars,
            "kept_count": len(kept),
            "skipped_count": len(skipped),
            "truncated_count": truncated,
        },
    }


def synthesize_answer(
    query: str, useful_chunks: list[dict[str, Any]], adapter: Any, history_block: str = ""
) -> str:
    """Write the final answer from the kept chunks (the useful-chunks bucket).
    Reads the chunk text, cites node ids, and is told to flag insufficiency.
    ``history_block`` (optional) threads prior conversation turns for continuity."""
    return str(
        synthesize_answer_result(
            query, useful_chunks, adapter, history_block=history_block
        ).get("answer")
        or ""
    )


def synthesize_answer_result(
    query: str, useful_chunks: list[dict[str, Any]], adapter: Any, history_block: str = ""
) -> dict[str, Any]:
    """Like :func:`synthesize_answer` but returns the full adapter payload.

    Carries ``usage`` / ``duration_ms`` when the adapter reports them so the
    answer pipeline can stamp galeed ``llm_calls`` (instrumentation backlog).
    """
    return _adapter_answer_result(
        adapter, build_synthesis_prompt(query, useful_chunks, history_block)
    )


def _build_query_embedding(runtime_config: Any, text: str) -> dict[str, Any] | None:
    """Embed the query for hybrid primitives, or None to stay lexical. None when
    hybrid is off, the adapter is the deterministic mock, or embedding fails.
    (Mirrors interaction.build_query_embedding; kept here to avoid a layer cycle.)"""
    if not text or runtime_config is None:
        return None
    if not getattr(runtime_config, "hybrid_search_enabled", False):
        return None
    if getattr(runtime_config, "embedding_adapter", "mock") == "mock":
        return None
    try:
        return embedding_adapter(runtime_config).embed(text)
    except Exception:
        return None


def run_deep_answer(
    db: Any,
    query: str,
    *,
    config: Any,
    runtime_config: Any,
    identity: dict[str, Any] | None = None,
    adapter: Any = None,
    query_embedding: dict[str, Any] | None = None,
    history_block: str = "",
) -> dict[str, Any]:
    """The full deep retrieval-and-answer flow: build the LLM seams from the
    answer adapter, run the orchestrator loop, then synthesise over the kept
    chunks. `adapter` / `query_embedding` are injectable for testing.
    `history_block` threads prior conversation turns into the synthesis."""
    adapter = adapter or answer_adapter(runtime_config)
    if query_embedding is None:
        query_embedding = _build_query_embedding(runtime_config, query)
    # Per-call embedder so `semantic_search` can embed a focused phrase (not just
    # the original question). Same gate as the session embedding: None when hybrid
    # is off / the adapter is mock / embedding fails.
    embedder = (lambda text: _build_query_embedding(runtime_config, text)) if runtime_config else None
    result = run_deep_retrieval(
        db,
        query,
        config=config,
        planner=make_planner(adapter),
        triager=make_triager(adapter),
        scorer=make_scorer(adapter) if getattr(config.retrieval, "deep_sufficiency_scoring", False) else None,
        query_embedding=query_embedding,
        identity=identity,
        embedder=embedder,
    )
    packed = pack_synthesis_chunks(
        query,
        result["useful_chunks"],
        request_budget_plan(config, runtime=runtime_config),
        history_block,
    )
    synthesis = synthesize_answer_result(
        query, packed["kept"], adapter, history_block=history_block
    )
    return {
        "answer": str(synthesis.get("answer") or ""),
        "useful_chunks": result["useful_chunks"],
        "synthesis_chunks": packed["kept"],
        "budget": packed["budget"],
        "budget_skipped": packed["skipped"],
        "rounds": result["rounds"],
        "trace": result["trace"],
        "sufficiency": result.get("sufficiency"),
        "sufficiency_history": result.get("sufficiency_history", []),
        "usage": synthesis.get("usage"),
        "duration_ms": synthesis.get("duration_ms"),
        "model": synthesis.get("model"),
        "adapter": synthesis.get("adapter"),
    }
