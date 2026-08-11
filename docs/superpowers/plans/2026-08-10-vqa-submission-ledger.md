# VQA Submission Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable, accessible ledger and the explicitly authorized final `q35_4b_u12` organizer-upload file without committing raw predictions or integrating its one-off ranking policy into reusable baseline code.

**Architecture:** A Markdown index provides the stable ledger interface, while one focused run record owns the complete method, provenance, validation, artifact-hash, unavailable-video, compute, and Evalbase metadata disclosure. The exact final upload JSON is tracked under `submissions/`; raw artifacts remain under ignored `data/`; executed source is preserved by an archival Git ref.

**Tech Stack:** Markdown, Git, SHA-256, pinned NIST VQA validator.

## Global Constraints

- Run tag is exactly `q35_4b_u12`.
- Commit generated answers only inside the exact authorized final organizer
  upload file; do not commit questions, media, credentials, raw predictions,
  or logs.
- Do not merge the submission-specific repair or deterministic fallback policy into reusable baseline modules.
- Record the run as prepared and locally validated until Evalbase confirms upload acceptance.
- Preserve all 2,443 test queries, including the 15 organizer-designated unavailable-video queries.

---

### Task 1: Add and verify the submission ledger

**Files:**
- Modify: `README.md`
- Create: `docs/submissions/README.md`
- Create: `docs/submissions/2026-q35-4b-u12.md`

**Interfaces:**
- Consumes: the private frozen bundle at `data/vqa/submissions/qwen35-safety-20260810-a1/final/`, its validator identity, the official test topics, and executed commit `c3477a7`.
- Produces: one stable ledger index entry and one accessible, copy-ready submission record.

- [x] **Step 1: Add the ledger index and run record**

  Record lifecycle state, method, test-set boundary, broken-video handling,
  artifact hashes, validation, compute, code isolation, and exact Evalbase
  field text.

- [x] **Step 2: Verify private evidence against the authored record**

  Recompute the topics and submission hashes, confirm 2,443 top-10 rows and all
  15 organizer-listed queries, and require the validator report to contain no
  errors or warnings.

- [x] **Step 3: Verify repository privacy and documentation quality**

  Run a targeted sensitive-content scan, Markdown link checks,
  `git diff --check`, and relevant existing tests.

- [x] **Step 4: Commit and publish a draft pull request**

  Stage only the ledger, design, and plan files; push the isolated branch and
  create a draft PR against `main`.

### Task 2: Include the final organizer-upload artifact

**Files:**
- Create: `submissions/trecvid-2026/vqa-ag/q35_4b_u12/submission.json`
- Modify: `README.md`
- Modify: `docs/submissions/README.md`
- Modify: `docs/submissions/2026-q35-4b-u12.md`

**Interfaces:**
- Consumes: frozen `final/submission-top10.json` with SHA-256 `c01873efb9a5f5cf1090368f1bed4fd91b6ecbf51d806f8670ecabc3d82322a7`.
- Produces: the exact byte-identical Evalbase upload file at a stable tracked path.

- [x] **Step 1: Copy and verify the exact finalized upload JSON**

  Require byte identity, 2,443 rows, ten answers per row, and the frozen
  SHA-256 before staging.

- [x] **Step 2: Update privacy and ledger documentation**

  Record that only the final approved upload is tracked; retain raw
  predictions, questions, media, credentials, and logs under the ignored
  boundary.

- [x] **Step 3: Verify and update the draft PR**

  Re-run the pinned organizer validator, repository tests, diff/privacy checks,
  then commit and push the exact artifact and documentation updates.
