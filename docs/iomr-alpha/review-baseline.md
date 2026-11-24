# Review baseline — TIRZAH IOMR alpha source

> **Not a formally approved specification.** This captures the Sam + Eve design agreements in [TIRZAH #81](https://github.com/gellsmore-svg/tirzah/issues/81#issuecomment-7147242134) for modelling evidence. Treat as a proposal input, not as a substituted human sign-off.

## U-01 — Stable source identity and immutable revisions
A persistent opaque logical source identifier is independent of path, raw-blob checksum and ingestion event. Source association across renames and replacements must be deliberate and attributable. Separately minted immutable revision IDs bind canonical text, transform/parser/segmentation identity, source hash and a stable node-to-span map. Deterministic fingerprints deduplicate equivalent processing; nondeterminism is a defect. Source/revision/offset references must not silently resolve to different text following rebuild; stale or inaccessible requests fail explicitly. Exact offset units are specified.

## U-02 — Keturah capability and contract compatibility
Extend Keturah's existing capability manifest, registry and MCP projection. Group versions describe safe interaction patterns across search, read, graph and history; per-tool contract fingerprints establish exact definitions. Agents negotiate and pin compatible contracts, and evidence references carry minimum compatibility/format requirements. Unknown filters and unsupported tools fail explicitly. Discovery metadata is not authority: runtime TIRZAH checks caller authorization independently. Keturah owns its published contract; TIRZAH references rather than clones it.

## U-03 — Historical authority, audit and retention deletion
Revision content is immutable while retained, but lifecycle and permissions remain separate. Superseded, pending and rejected states have distinct eligibility. Privileged historical inspection requires explicit authority for the material and mandatory attributable audit events; failed mandatory audit persistence blocks disclosure. Deletion and retention expiry require their own separately attributable authority and audit records. Endorsement and trust never widen visibility. Issue #18's unauthenticated operator API must be addressed before untrusted-client exposure.

## Review scope
Initial baseline must describe TIRZAH as a whole (source ingestion, Mongo-backed document/graph memory, retrieval/context, local interfaces, model/runtime adapters, governance), then represent these newer agent-facing decisions as planned constraints. PR #79 remains draft and is not declared version-safe. The 400-character search preview cap is a separate adapter improvement. This alpha model is meant for critique and correction before SCL proposal approval or any product implementation.
