# TIRZAH — SCL/IOMR alpha bootstrap (review only)

**Status:** Initial AI-authored, unapproved consumer-model bootstrap for review by Graham and Eve. **Do not merge as an authoritative approved IOMR graph without an explicit bootstrap governance decision.** This is not an SCL proposal receipt, and records marked `proposed` do not imply that TIRZAH functionality is absent or present.

## Why this exists
TIRZAH already has extensive prose requirements, architecture decisions and code. This is the first **structured candidate** using SCL's `iomr-evolved` 2.0.0 record and relationship types, so future changes can be governed by SCL's semantic proposal API rather than a second informal specification. The baseline models both the existing product and the proposed agent-facing improvements from [issue #81](https://github.com/gellsmore-svg/tirzah/issues/81).

## Files and coverage
- `imk.json`: consumer project declaration and pinned source pointers.
- `meta/iomr-model/metamodel.json`: SCL `iomr-evolved` 2.0.0 identity and relationship rules.
- `meta/iomr-model/nodes/records.json`: project, 3 intents, 9 measurable outcomes, 11 interaction models, 30 acceptance-bearing requirements, actors/systems, two policies and three proposed Decisions.
- `meta/iomr-model/relationships/links.json`: `delivers`, `serves`, `specifies`, `involves`, `constrains` and `belongs_to` links.
- `docs/iomr-alpha/review-baseline.md`: snapshot of the agreed U-01/U-02/U-03 design.

## Trust and provenance
All requirements are **desired behavioural constraints**, not statements that existing code passes. `metadata.basis=derived` expresses interpretation of the linked in-repository evidence. All records remain `status=proposed`; nothing is promoted to implemented based on prose alone. The issue-derived decisions are jointly agreed **in principle**, not approved SCL graph transitions. Some existing implementation choices are intentionally described in requirements but their actual conformance must be audited.

This **first baseline** is a special bootstrap candidate because no valid TIRZAH consumer graph currently exists for the normal `propose_change` lifecycle to operate on. It is **not** a normal model mutation via SCL API and must be separately reviewed as a trust-root initialization; afterwards changes must use the API and trusted reviewer signature. No fabricated proposal/approval receipt. Never hand-edit canonical model data for routine changes.

## How to validate and review
1. Check out this draft branch in a clean workspace and install a pinned version of [SCL](https://github.com/gellsmore-svg/SCL) compatible with `iomr-evolved 2.0.0`.
2. Run the SCL product-model validator (`imk api validate_model --json '{}'` or the corresponding installed `scl` API operation). Review **all** findings, including evidence anchors, source integrity, participant links, model coverage and traces. No claim is made that official validation was run by this author.
3. Walk the Intent → Outcome → Model → Requirement chains and Decision → `constrains` → Requirement links. Confirm that requirements 04–06/11–13/29 are constrained by U-01, 07/09/11/24–26/30 by U-02, and 13/15/18/20–23/29 by U-03.
4. Challenge current/target status, scope and overreach. In particular, resolve the source-association rule, revision offset unit and migration, capability group/individual fingerprint format, deletion retention policies, privacy/error semantics and actual actor trust boundary.
5. Agree how the bootstrap is approved and merged; **then** use `get_context` → `propose_change` → `validate_proposal` → `preview_change` → externally signed `approve_change` → `commit_change` for subsequent changes.

## Review gates
- No network exposure before issue #18 authentication/scope controls.
- PR #79 is not made merge-ready by this specification; revision-pinning and read-path authorization remain delivery gates.
- Keep 400-character search previews as a separate adapter backlog item.
- Do not silently broaden ACL or treat metadata trust scores as eligibility.
- Validate against **real MongoDB** and cross-adapter agent scenarios before announcing conformance.

**Primary review questions for Eve:** Is the Intent/Outcome breakdown truly product-wide? Are the interaction participants meaningful? Do the 30 proposed requirements correctly distinguish existing behaviour, target changes and policy gaps? Are U-01/U-02/U-03 decisions constraining the right requirements without copying Keturah's own contract? What is missing from lifecycle/security/operational acceptance?
