# TIRZAH — Initial SCL/IOMR Baseline Governance (candidate policy)

**Status:** proposed, unapproved. This document specifies the prerequisite lifecycle for accepting an initial TIRZAH IOMR graph. It does not assert that the lifecycle is implemented, that the alpha passes SCL validation, or that a human has approved it.

**Scope:** the first TIRZAH consumer graph in PR #82 and any recovery from an invalid initial baseline. SCL defines the metamodel and language; TIRZAH owns its IOMR data. This policy is a separately reviewable precondition for bootstrapping TIRZAH; routine model changes after activation follow SCL's normal proposal API.

## Governing invariants

1. **Authority is external to the candidate graph.** Neither a proposed IOMR record, a Markdown statement, a Git commit, nor a PR merge can grant itself authoritative status.
2. **Exactly zero or one authoritative baseline.** A repository may contain any number of candidates and retained historical baselines, but the authoritative-baseline pointer is either unset or points to exactly one independently approved, verified, non-retracted baseline.
3. **Content-bound approval.** A human approval is an independently verifiable signature over one canonical candidate digest, the validation result digest, the SCL contract pin and the approved activation intent. Changing any covered material invalidates the attestation.
4. **Validation precedes signing.** The signed approval record is generated only after an official SCL validation has passed for the exact frozen candidate. An approver may refuse approval even after technical validation.
5. **Failures are explicit and fail closed.** Verification, storage, signature, authorization, validation or audit uncertainty must not promote or silently restore a baseline.
6. **History is append-only while retained.** Superseded/retracted baselines and their decisions remain identifiable and auditable; an old approval is not silently erased or reused.
7. **Successor authority is never inherited from the candidate.** Initial activation, replacement, supersession and retraction require an independently established human authority. A candidate graph, including its own Policy records, may describe the rule but cannot grant or widen the authority to approve itself or its successor.
8. **Every governed entry point enforces current authority.** No application adapter, CLI command, background worker or MCP handler may use a cached, missing, superseded or retracted baseline as though it were active. Retrieval is governed as well as mutation. A special administrative recovery mode must be separately authenticated, audited and prevented from invoking ordinary agent-facing retrieval.

## Required records (proposed contracts, not implemented schema)

- **Candidate:** opaque candidate ID, candidate repository and commit, model/manifest paths, SCL metamodel ID and version, pinned SCL validator/build, and canonical digest of the declared complete candidate input set. Digest computation and ordering must be specified, deterministic and verified.
- **Validation receipt:** candidate ID and digest, validator identity/version, executable command or API contract, validation timestamp, results/errors and result digest. No passing receipt means no approval.
- **Approval attestation (separate from the candidate model):** candidate digest, validation receipt digest, approving human identity, authority/key identifier, signature algorithm and signature, approval timestamp, scope/purpose, and explicit intent to activate this baseline. The verifier must use an independently established trust root, not identity asserted only by the candidate. The attestation must not be included recursively inside the content it signs.
- **Lifecycle event:** event ID, baseline ID/digest, event type, actor, verified authority, time, reason, previous state, next state, predecessor/successor references where applicable, and immutable audit reference.
- **Authority pointer:** the sole current active baseline ID/digest, or null, with a monotonically increasing generation/version, stored in the restricted MongoDB `iomr_baseline_authority` singleton outside the consumer IOMR graph. Atomic compare-and-swap must update this pointer and durably record the linked event in the same database transaction, or fail without any authority change.
- **Independent checkpoint:** an operator-controlled, authenticated record of the latest accepted authority generation and event-chain digest, stored **outside Git and outside the TIRZAH MongoDB instance**. On start and before every governed operation, verification detects disagreement or rollback and fails closed. Checkpoint publication must be crash-consistent with authority transitions; unresolved partial commits must fail closed until authorised reconciliation, not guess a winner.

## Out-of-band initial trust root — mandatory precondition

Before **any** TIRZAH bootstrap validator/activation service can grant authority, a designated human operator must establish and document, **outside the Git repository, candidate graph and TIRZAH database**, (a) which human identity or identities may approve, replace and retract a baseline, (b) their independently verified public signing keys and signature policy, (c) the initial trust-root version and key-revocation/rotation procedure, and (d) the independently administered authority-checkpoint location. The private signing keys must remain outside the repository and TIRZAH database. The verifier is provisioned with that trust root by an operator-controlled channel; it must not accept a root advertised by the candidate.

For alpha, the simplest proposed arrangement is an operator-provisioned, access-controlled local configuration outside the checkout and database, identifying a human-controlled signing key and approved roles. Its creation and custody are a **manual prerequisite**, not a deferred acceptance test or a claim that TIRZAH already provisions it. Human identity-to-key binding, safe rotation and recovery of the external checkpoint require documented out-of-band authority before activation. In the absence of a valid root or checkpoint, the baseline remains non-authoritative.

## Authority storage and rollback boundary

**Chosen location:** TIRZAH's existing MongoDB service, with a **restricted governance namespace** separate from the IOMR graph: `iomr_baseline_authority` (one authoritative pointer record) and `iomr_baseline_events` (retained event and attestation references). Only a dedicated governance identity may transition authority; ordinary ingestion, retrieval, worker and agent identities have read-only verification access at most. If the MongoDB deployment lacks transaction-capable configuration, atomic pointer-plus-event updates are unavailable and **activation is blocked until that prerequisite is met**. This is an explicit operational dependency, not an assertion about the current deployment.

**Why not Git:** a branch revert, force push or rewritten history can restore an old pointer without enforcing a state transition. Git stores candidate artefacts and review history, **never the live authority pointer**.

**Why MongoDB alone is insufficient:** database restores or privileged writes may roll a pointer and its event log backwards even when compare-and-swap worked. The external monotonic signed/checkpointed head must reject a generation or event-chain head older than the last independently accepted transition. If MongoDB, audit ledger and the external checkpoint disagree, fail closed and require a separately authorised reconciliation with its own event; never infer authority from Git or choose the newest-looking value. This counter-rollback control is a hard first-activation gate, and must be demonstrated, not merely described.

## Lifecycle and transitions

| From | To | Required precondition and action |
|---|---|---|
| None | Candidate | Record a new alpha/repair candidate, initially non-authoritative |
| Candidate | Validated | Official SCL validation passes against a frozen, digest-addressed model |
| Validated | Approved | Designated human signs an attestation binding the exact validated digest and validation receipt; independently verify signature and authority |
| Approved | Active | Require a fresh, independently verified **external human activation/replacement authorization** that binds the candidate digest, expected previous active digest (or null), expected generation and action. No authorization may originate from or be delegated by the candidate model. In one atomic transaction, verify the expected pointer state, activate, record the transition and coordinate the independently anchored checkpoint; on any uncertainty fail closed |
| Candidate / Validated / Approved | Rejected | Record explicit rejection; never activate |
| Active | Superseded | Only as part of externally authorised activation of a separately validated and approved successor; atomically replace the prior active pointer, write reciprocal predecessor/successor references and lifecycle event, and anchor the new checkpoint |
| Active | Retracted | Require separately verified **external human retraction authority**, expected generation/digest, reason and audit event; atomically set active pointer to null, anchor the transition and never reactivate a predecessor |
| Superseded / Retracted | Active | **Forbidden** by direct pointer flip; restoration requires a newly validated, approved and activated candidate, even when content is byte-identical |

A Git merge may store reviewable specification text before bootstrap approval, but **merge is not activation**. The lifecycle policy must be separately reviewed and accepted before PR #82's initial model is eligible for attestation or activation. The **initial PR #82 candidate itself MUST include**, before official validation and signature, a Requirement and a Policy linked to its bootstrap governance Model that reflect this external authority, the complete lifecycle and the entry-point gate. It is not permissible to add those records after first activation. They document externally enforced authority and do not grant it.

## Recovery from a defective first baseline

If the initial active graph proves invalid after activation, an externally authorised reviewer can retract it without first having a replacement. The authority pointer becomes null. **All ordinary governed operations, including both retrieval and mutation across every adapter, fail closed**, while separately authorised administrative inspection and recovery remain available. The revoked baseline remains in immutable audit history while retained. A repaired candidate starts a fresh validation/attestation/activation cycle. Retraction does not delete the defective baseline or cause the previous one to become active automatically.

If a baseline is superseded, its approval and provenance remain in history, but only the successor is authoritative after the atomic transition. Concurrent activation attempts must produce one winner and an explicit conflict for others. A pending or failed activation must leave the previous authoritative state intact.

## Gate coverage: operations that must verify the current authority

Every TIRZAH application-level path must resolve and verify the same current authority generation, baseline digest and external checkpoint before using IOMR-governed behaviour; a stale session or cached capability manifest does not satisfy this check. The gate covers:

- **Agent retrieval/read:** search, hybrid/semantic/keyword search, `search_memory`, `retrieve`, `read_memory`, `ask`, context assembly, verbatim reads, graph navigation/traversal and privileged historical reads, whether called via API, MCP, internal adapters or CLI.
- **Writes and processing:** source ingestion, re-ingestion, indexing/rebuilds, graph writes, endorsement/authority-affecting changes, model proposals, validation/commit actions, scheduled background jobs and any CLI/API operation that reads or writes governed data.
- **Downstream use:** workflows reusing evidence or context must recheck authority for any new access; an evidence reference does not carry permission or authorize using a retracted baseline.

During first-bootstrap/retraction states, only explicitly allowlisted, authenticated and auditable *administrative* functions may inspect diagnostic state, validate repair candidates, manage external authority, or perform recovery. They must not call normal retrieval adapters or return governed evidence to ordinary agents. Direct database bypass must be prohibited by service permissions and deployment controls; policy cannot claim to gate a principal with unrestricted database administration rights.

This is a **proposed gate** and a release blocker for any enforcement claim. The current TIRZAH code has not been demonstrated to implement it. Until all listed entry points are instrumented and negative-tested, baseline activation and network-facing governed operations remain blocked.

## Acceptance tests before first activation

1. An unsigned, self-approved, unvalidated or failed-validation candidate cannot activate.
2. Signature verification rejects altered model bytes, changed source declarations, changed SCL pin, mismatched validation receipt, unknown key and an approver without independently established authority.
3. Two concurrent attempts cannot leave two active baselines; an expected-version conflict is explicit.
4. Activating a successor atomically supersedes its predecessor and retains both events and provenance.
5. Retraction of the only active baseline yields zero active baselines and blocks **every ordinary governed retrieval and mutation path**, including CLI, MCP, API, ingestion, background processing, graph reads, historical inspection and old-session retries, until a newly attested activation.
6. Retraction, failed activation and supersession are logged with accountable human authority; audit failure blocks the transition.
7. A previously superseded/retracted baseline cannot be silently restored by reverting Git, rewriting history, restoring an old MongoDB snapshot or resetting the authority pointer. A rolled-back or missing independent checkpoint makes verification fail closed.
8. Repeating the same verified operation must be idempotent or return a typed conflict, never create contradictory authority records.
9. **Precondition (not merely an implementation test):** the independently documented out-of-band trust root, human-key/role binding, key revocation procedure and checkpoint custody exist before testing/activation; negative tests reject missing, revoked or substituted roots.
10. A candidate attempting to define its own approver or authorise its own successor/retraction is rejected; an independently signed replacement/retraction binds the expected active digest and generation.
11. A deliberately bypassing CLI, MCP, API or background-worker retrieval path is denied when the pointer is null/retracted or the external checkpoint cannot be verified.
12. A simulated failed Mongo transaction or interrupted checkpoint publication yields no contradictory active baseline; startup verification fails closed pending authorised reconciliation.

## Sequencing for PR #82

1. **First:** review and merge this lifecycle *policy specification* separately, without representing it as implemented controls; separately provision the out-of-band trust root and choose the checkpoint custody model.
2. **Second:** layer on the alpha model the other review refinements: machine-readable current/target capability state, traceable provenance classifications, reasoned Decision→Requirement constraints, and a **mandatory bootstrap-governance Requirement/Policy/Model linkage** through the SCL metamodel **before candidate validation**.
3. **Third:** implement and negative-test the authority verifier, transactional pointer/event writer, external checkpoint and all path gates; then run the actual SCL validator and proposal/bootstrap checks; record a real, reproducible validation receipt. Freeze the exact candidate digest; changing the candidate restarts this step.
4. **Fourth:** request a human approval signature for that exact validated candidate and verify it using independently established authority.
5. **Fifth:** atomically activate the approved baseline with an audit event and authority pointer; only then call the TIRZAH IOMR graph authoritative. Keep PR #82 in draft until its review gates are met; GitHub draft/merge status is not the authority pointer.

**Implementation gap:** no out-of-band trust root provisioning, transaction-capable governance collection, checkpoint service, entry-point gate, transition implementation, verifier, human attestation, approval record, audit writer, authority pointer, or official SCL validation is asserted by this document. These are explicit exit criteria, not completed work.
