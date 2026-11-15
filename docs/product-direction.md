# Product direction

Last updated: 2026-05-17

Answered decisions stay here. Open questions are only in `meta/open-questions.md`.

## Answered Decisions

- Stage 1 uses real local MongoDB immediately. A fake repository may still be used for narrow unit tests, but the first working path targets MongoDB.
- MongoDB 8.0.23 has been installed on WSL Ubuntu 24.04 and verified with `mongosh --eval 'db.runCommand({ ping: 1 })'`.
- Stage 1 hardware target is the current machine: 8GB RAM plus GTX 3060. Avoid assuming the later 32GB RAM target.
- The first user surface is a CLI/dev command for ingestion and inspection. The FastAPI web UI follows after the ingestion slice is stable.
- Implicit endorsement is represented as a label/provenance field in the correct MongoDB tree/node structure, not as a filename convention.
- Duplicate source ingestion is rejected by content checksum and the requestor is notified.
- Accepted source files are copied into the local archive.
- Label meanings are stored in MongoDB in a lookup collection with descriptions.
- `llama-cpp-python` is only an optional future adapter path for direct local GGUF model execution; it is not a required Stage 1 dependency.

## Product Scope

- OQ-001
- OQ-002
- OQ-003

## Local Runtime

- Stage 1 assumes current 8GB RAM plus GTX 3060.
- Do not force `llama-cpp-python` into Stage 1. Keep a model adapter boundary and use deterministic/mock output for early tests.
- OQ-004

## MongoDB And Vector Search

- MongoDB 8.0.23 is installed locally and `mongod` is active under systemd.
- Use real MongoDB immediately for the working Stage 1 path.
- OQ-005

## Ingestion Semantics

- Implicit endorsement is a MongoDB tree/node label/provenance value, not a filename convention.
- Copy accepted source documents into the local archive.
- OQ-006

## Interface

- Build a CLI/dev command first for faster ingestion testing.
- OQ-007

## Evaluation

- OQ-008
- OQ-009
- OQ-010

## Post-V1 Enhancements

For a curated list of improvements that would strengthen the implemented V1 features, see `docs/improvements-and-enhancements.md`. New direction questions go in `meta/open-questions.md`.
