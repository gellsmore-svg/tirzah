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

## Required records (proposed contracts, not implemented schema)

- **Candidate:** opaque candidate ID, candidate repository and commit, model/manifest paths, SCL metamodel ID and version, pinned SCL validator/build, and canonical digest of the declared complete candidate input set. Digest computation and ordering must be specified, deterministic and verified.
- **Validation receipt:** candidate ID and digest, validator identity/version, executable command or API contract, validation timestamp, results/errors and result digest. No passing receipt means no approval.
- **Approval attestation (separate from the candidate model):** candidate digest, validation receipt digest, approving human identity, authority/key identifier, signature algorithm and signature, approval timestamp, scope/purpose, and explicit intent to activate this baseline. The verifier must use an independently established trust root, not identity asserted only by the candidate. The attestation must not be included recursively inside the content it signs.
- **Lifecycle event:** event ID, baseline ID/digest, event type, actor, verified authority, time, reason, previous state, next state, predecessor/successor references where applicable, and immutable audit reference.
- **Authority pointer:** the sole current active baseline ID/digest, or null, with a monotonically increasing generation/version. Changes use atomic compare-and-swap and record the event.

Do not treat a digital signature as sufficient without verifying the approver's independently established authority at approval time. Rules for authority changes and key revocation must be defined and tested before activation.

## Lifecycle and transitions

| From | To | Required precondition and action |
|---|---|---|
| None | Candidate | Record a new alpha/repair candidate, initially non-authoritative |
| Candidate | Validated | Official SCL validation passes against a frozen, digest-addressed model |
| Validated | Approved | Designated human signs an attestation binding the exact validated digest and validation receipt; independently verify signature and authority |
| Approved | Active | Atomically activate if there is no active baseline, or via a separately authorised replacement transaction; retain activation event |
| Candidate / Validated / Approved | Rejected | Record explicit rejection; never activate |
| Active | Superseded | Atomically activate a separately approved successor, writing reciprocal successor/predecessor references and lifecycle events |
| Active | Retracted | Independently authorised retraction with reason and audit event; atomically set active pointer to null; do not implicitly reactivate a predecessor |
| Superseded / Retracted | Active | **Forbidden** by direct pointer flip; restoration requires a newly validated, approved and activated candidate, even when content is byte-identical |

A Git merge may store reviewable specification text before bootstrap approval, but **merge is not activation**. The lifecycle policy must be separately reviewed and accepted before PR #82's initial model is eligible for attestation or activation. The graph itself must eventually include a requirement and traceable Policy/Model links reflecting this external bootstrap authority.

## Recovery from a defective first baseline

If the initial active graph proves invalid after activation, an authorised reviewer can retract it without first having a replacement. The authority pointer becomes null, all governed changes that require an authoritative graph fail closed, and the revoked baseline remains in immutable audit history while retained. A repaired candidate starts a fresh validation/attestation/activation cycle. Retraction does not delete the defective baseline or cause the previous one to become active automatically.

If a baseline is superseded, its approval and provenance remain in history, but only the successor is authoritative after the atomic transition. Concurrent activation attempts must produce one winner and an explicit conflict for others. A pending or failed activation must leave the previous authoritative state intact.

## Acceptance tests before first activation

1. An unsigned, self-approved, unvalidated or failed-validation candidate cannot activate.
2. Signature verification rejects altered model bytes, changed source declarations, changed SCL pin, mismatched validation receipt, unknown key and an approver without independently established authority.
3. Two concurrent attempts cannot leave two active baselines; an expected-version conflict is explicit.
4. Activating a successor atomically supersedes its predecessor and retains both events and provenance.
5. Retraction of the only active baseline yields zero active baselines and blocks governed mutations until a newly attested activation.
6. Retraction, failed activation and supersession are logged with accountable human authority; audit failure blocks the transition.
7. A previously superseded/retracted baseline cannot be silently restored by reverting Git or resetting the authority pointer.
8. Repeating the same verified operation must be idempotent or return a typed conflict, never create contradictory authority records.
9. Implementation demonstrates how verifier trust roots and revocation policy are established independently of the first graph.

## Sequencing for PR #82

1. **First:** review and merge this lifecycle *policy specification* separately, without representing it as implemented controls.
2. **Second:** layer on the alpha model the other review refinements: machine-readable current/target capability state, traceable provenance classifications, reasoned Decision→Requirement constraints, and a bootstrap-governance requirement/Policy linked through the SCL metamodel.
3. **Third:** run the actual SCL validator and proposal/bootstrap checks; record a real, reproducible validation receipt. Freeze the exact candidate digest; changing the candidate restarts this step.
4. **Fourth:** request a human approval signature for that exact validated candidate and verify it using independently established authority.
5. **Fifth:** atomically activate the approved baseline with an audit event and authority pointer; only then call the TIRZAH IOMR graph authoritative. Keep PR #82 in draft until its review gates are met; GitHub draft/merge status is not the authority pointer.

**Implementation gap:** no transition implementation, verifier, human attestation, approval record, audit writer, authority pointer, or official SCL validation is asserted by this document. These are explicit exit criteria, not completed work.
