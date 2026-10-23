# Tirzah features in depth

Planning, human-defined processes, interpretive execution, session restart
state, Hoglah routing and embeddings. Moved from the root README (2026-10).

## Recursive Deborah request planning

The browser-facing `/api/ask` path is wrapped by a recursive process planner. It
creates a bounded first-pass Deborah `PLAN`, invokes Tirzah's existing validated
retrieval and answer pipeline, and then revises the same plan from new evidence
or unresolved state. Each revision keeps the plan ID, parent revision, trigger,
stopping conditions, and a complete process backbone.

Plans are operational records in `recursive_plans`, not trusted graph memory, and
they do not grant tools or side-effect authority. Python continues to enforce the
actual tool menu, budgets, writes, and termination. Later information can revise
a stored plan through `POST /api/plans/{plan_id}/revise`. Configure the planner
with `runtime.recursive_planning_*`.

**Deborah plan ownership** (see Deborah `docs/PLAN-OWNERSHIP.md`): Tirzah
authors and revises plans; Deborah owns PLAN conformance and crystallised
*framed* walks. `tirzah.planning.deborah_bridge` seals each plan with
`deborah.validate_plan`, auto-detects substrate/critique graphs, and routes
them through Deborah `run_substrate_slice` when
`runtime.plan_framed_execution_enabled` is true (default). Soft conformance
errors land in `deborah_conformance_errors`; set
`runtime.plan_require_deborah_conformance: true` to fail hard.

When the planner needs a coherence pressure-test or counter-framework research,
it can call [Milcah](https://github.com/gellsmore-svg/Milcah) as an optional
specialist:

```bash
pip install "tirzah[milcah]"
```

Set `runtime.milcah_enabled: true` (or `MILCAH_ENABLED=1`). Tirzah delegates to
Milcah's `coherence_check` contract and degrades to a blocked specialist result
if Milcah is not importable or the call fails. Real Milcah specialist execution
uses Hoglah under the hood, so run the Hoglah worker described below when you
want live model-backed results.

## Human-defined Processes

Processes are first-class, selectable templates that ground agentic work with
the right amount of oversight. A **template** is versioned plain-text prose
(its gates and loops stated in English); an **instance** binds a template
version to a task and carries its own state + audit trace. When a conversation
runs under a process, its text becomes the planner's top-level guide: the
planner plans *within* it and emits AWAIT gate steps where it says to pause for
approval. Gates pause the instance (resumable on approval), deviations are
flagged for approval, and an emergency override needs a justification.

Three presets ship seeded: **Governed** (gates before apply/ship), **Fluid**
(log-only oversight), **Emergency** (act-first, mandatory retrospective).
Manage them in Mahlah's process bar, or via `tirzah process …`
(seed-presets/templates/new-template/start/gate/override/complete/
retrospective/metrics/history) and the `/api/process/*` routes. Every instance
is fully audit-queryable (retrospective, usage metrics, and a "how were similar
tasks handled?" history query).

Authoring and selection are assisted. **`tirzah process review`** (or the
process bar's *Review with Tirzah*) checks a draft for missing gates,
ambiguity, and gaps — structural checks always, plus model-generated
clarifying questions and an optional suggested rewrite; **`tirzah process
trial`** dry-runs a draft against a sample task to confirm the planner places
the gates it asks for. **`tirzah process suggest`** (and the process bar on
open) recommends a fitting process from the task's risk/scope/urgency signals —
advisory, and the choice is recorded as `selection_reason` for the audit. And processes are living
artifacts: **`tirzah process evolve <id>`** mines a template's past runs for
recurring approved deviations, gate friction, heavy override, or high
abandonment, and *proposes* a revised body with rationale — applied only on
approval (`--apply`), as a new version with provenance. Active instances keep
their frozen process, so evolution is backward-compatible.

## Interpretive plan execution & debugging

Set `TIRZAH_PLAN_INTERPRETIVE_EXECUTION_ENABLED=true` and planned requests are
*executed* step-by-step by the Deborah interpreter (SPEC §4.6): dependency-gated
steps, tool gating via `allowed_tools`, ITERATE/DECISION/PARALLEL/RETRY/ERROR
constructs, resumable persisted executions, and **mid-step revision** — the plan
adapts after each completed call. The running plan streams live into Mahlah's
process panel, and every model call's full In→Out lands in galeed's `llm_calls`
debugging store (`galeed trace`, or Mizpah's LLM Calls tab).

## Restart state

Resumable session state is tracked in the database, not a file: each exchange
records a `session_continuity` snapshot — the latest query, focus/used nodes,
active documents, controller decision, evidence summary, and an answer preview,
with older iterations superseded but retained.

```bash
tirzah session-continuity --session-id default --limit 5   # latest + recent
tirzah restart-render --session-id default --output .restart.md  # rendered view
```

It is also exposed over HTTP at `GET /api/session-continuity` (with a panel in
the web UI's developer mode).

## Routing model calls through Hoglah (optional)

Route **answers and embeddings** through [Hoglah](https://github.com/gellsmore-svg/hoglah),
a local-first job queue, so every model call is serialized through one durable
queue (Hoglah's own [known limits](https://github.com/gellsmore-svg/hoglah#know-before-you-rely-on-it) apply):

```bash
pip install "tirzah[hoglah]"
```

Set `runtime.answer_adapter` and/or `runtime.embedding_adapter` to `hoglah`, then
run a **separate** worker daemon pointed at the same queue and output folder:

```bash
HOGLAH_OUTPUT_DIR=data/hoglah/outbox \
  hoglah run --real --db data/hoglah/jobs.sqlite3 \
  --ollama-host http://<host>:11434 -c 1
```

Tirzah becomes a pure submitter (no in-process worker): it enqueues each call and
collects the result by polling the output folder (`hoglah_delivery: poll`) or via
an HTTP callback (`hoglah_delivery: callback`, with poll as fallback). Tirzah
itself does only local IPC; the daemon makes the Ollama call.

## Similarity profiles / embeddings (optional)

Text-similarity profiles default to the deterministic `mock` adapter so tests and
first runs are reproducible. For real local embeddings, install the optional
extra and configure a local embedding adapter:

```bash
pip install -e '.[profiles]'
```

`tirzah init --runtime local_command` writes sensible defaults (e.g.
`BAAI/bge-small-en-v1.5`, 384 dims, worker mode). Existing nodes can be profiled
in bounded, resumable batches (`backfill-profiles`, `queue-profile-backfill`,
`process-profile-backfill`). HTTP-backed embedding adapters are blocked by default
for ingestion/retrieval. See [`docs/install.md`](install.md) for details.
