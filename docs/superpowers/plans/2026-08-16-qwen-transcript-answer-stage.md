# Qwen Transcript-to-Answer Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid ASR-quality interpretation with a fail-closed regression and execute one 12-row, two-arm transcript-to-answer diagnostic over the already frozen Qwen and Whisper outputs.

**Architecture:** Add one focused module that prepares label-free matched requests, records one local container attempt, freezes all 24 answers, and opens annotations only for post-freeze review. Reuse the existing `vqa_transformers_runner` with empty frame arrays and a transcript, and keep orchestration in a thin CLI. Preserve every Stage 0.5f artifact unchanged.

**Tech Stack:** Python 3.12, pytest, SHA-256/JSON freezes, rootless Podman, the pinned ROCm image, `Qwen/Qwen3.5-4B`, and Quarto.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-16-qwen-transcript-answer-stage-design.md` exactly.
- Preserve the Stage 0.5f private root and historical `0/12` score byte-for-byte.
- Reuse exactly 12 frozen identities and exactly two transcript arms.
- Run exactly 24 transcript-only answer requests in one local attempt at `$0.00` paid cost.
- Do not open annotations or references before the 24-answer prediction freeze.
- Do not rerun Qwen ASR or Whisper.
- Keep row identities, questions, transcripts, answers, references, and judgments under ignored `data/`.
- Never stage the unrelated `.claude/` directory or anything under `data/`.

---

### Task 1: Make the invalid ASR metric fail closed

**Files:**
- Modify: `src/trec_video_2026/vqa_qwen_aligner_canary.py`
- Modify: `scripts/run_qwen_aligner_canary.py`
- Modify: `tests/test_vqa_qwen_aligner_canary.py`
- Modify: `tests/test_run_qwen_aligner_canary.py`

**Interfaces:**
- `score_postfreeze(...)` continues to support the original two-row literal-phrase mechanism diagnostic.
- A protocol whose `selection_kind` is `qwen_aligner_fresh_english_quality_panel` is rejected with `VQA answers are not transcript references` before a new score artifact can be written.

- [x] **Step 1: Write the failing direct-scorer regression**

  Build a ready synthetic fresh-English root and call:

  ```python
  with pytest.raises(ValueError, match="VQA answers are not transcript references"):
      score_postfreeze(
          root,
          references={video_id: "because he received some news" for video_id in ids},
          whisper=whisper,
          source_bindings=bindings,
      )
  assert not (root / "postfreeze-score.json").exists()
  ```

- [x] **Step 2: Run the direct regression and verify RED**

  Run:

  `uv run pytest -q tests/test_vqa_qwen_aligner_canary.py -k rejects_vqa_answers_as_transcript_references`

  Expected: FAIL because the current scorer still computes exact answer containment.

- [x] **Step 3: Add the minimal fail-closed guard**

  Immediately after reading the verified protocol in `score_postfreeze`, add:

  ```python
  if protocol.get("selection_kind") == FRESH_ENGLISH_SELECTION_KIND:
      raise ValueError(
          "VQA answers are not transcript references; "
          "fresh English answer containment cannot score ASR quality"
      )
  ```

  Add the same protocol check in the CLI before `_score_inputs` opens annotation
  JSON so future invocations fail without reading answer text.

- [x] **Step 4: Run focused scorer and CLI tests**

  Run:

  `uv run pytest -q tests/test_vqa_qwen_aligner_canary.py tests/test_run_qwen_aligner_canary.py`

  Expected: PASS, including the unchanged non-fresh literal-phrase test.

---

### Task 2: Build and freeze the matched transcript-only requests

**Files:**
- Create: `src/trec_video_2026/vqa_qwen_transcript_answer.py`
- Create: `tests/test_vqa_qwen_transcript_answer.py`

**Interfaces:**
- `build_answer_requests(*, selection: Mapping[str, Any], manifest: Mapping[str, Any], qwen_results: Sequence[Mapping[str, Any]], whisper_payload: Mapping[str, Any]) -> list[dict[str, Any]]`
- `prepare_answer_test(root: Path, *, prepared_at: str) -> dict[str, Any]`
- `verify_answer_inputs(root: Path) -> dict[str, Any]`

- [x] **Step 1: Write failing pure request-builder tests**

  The happy-path assertion is:

  ```python
  requests = build_answer_requests(
      selection=selection,
      manifest=manifest,
      qwen_results=qwen_results,
      whisper_payload=whisper_payload,
  )
  assert len(requests) == 24
  assert {row["condition_id"] for row in requests} == {
      "qwen_transcript",
      "whisper_transcript",
  }
  assert all(row["frame_paths"] == [] for row in requests)
  assert all(row["frame_timestamps"] == [] for row in requests)
  assert all("answer" not in row and "reference" not in row for row in requests)
  ```

  Also reject a non-12 selection, duplicate/missing IDs, failed or empty
  transcripts, audio-hash drift, question drift, forbidden answer/reference
  keys, and unequal arm coverage.

- [x] **Step 2: Run builder tests and verify RED**

  Run:

  `uv run pytest -q tests/test_vqa_qwen_transcript_answer.py -k build`

  Expected: collection fails because the module does not exist.

- [x] **Step 3: Implement the minimal pure builder**

  Emit each request with these exact live fields before its canonical hash:

  ```python
  {
      "run_id": "qwen-transcript-answer-stage05g-v1",
      "condition_id": condition_id,
      "video_id": video_id,
      "question": question,
      "frame_paths": [],
      "frame_timestamps": [],
      "transcript": transcript,
  }
  ```

  Sort by frozen selection order, with Qwen then Whisper for each identity.

- [x] **Step 4: Write failing preparation/binding tests**

  Synthetic files must prove that preparation exclusively writes
  `protocol.json`, `requests.json`, and `inputs-freeze.json`; hashes every source
  and output; records the runner/model/runtime pins; and refuses an existing
  root or any post-prepare source mutation.

- [x] **Step 5: Run preparation tests and verify RED**

  Run:

  `uv run pytest -q tests/test_vqa_qwen_transcript_answer.py -k 'prepare or verify'`

- [x] **Step 6: Implement preparation and pass Task 2**

  The production source paths are the canonical constants already used by the
  Qwen aligner harness. `prepare_answer_test` must call the verified Stage 0.5f
  status boundary before reading Qwen output bytes and must verify the frozen
  Whisper checksum entry. It must not accept an annotation path.

---

### Task 3: Record one local answer attempt and freeze predictions

**Files:**
- Modify: `src/trec_video_2026/vqa_qwen_transcript_answer.py`
- Modify: `tests/test_vqa_qwen_transcript_answer.py`

**Interfaces:**
- `build_answer_command(root: Path) -> list[str]`
- `record_answer_attempt(root: Path, *, runner: Callable[..., CompletedProcess[str]] = subprocess.run) -> dict[str, Any]`
- `freeze_answer_predictions(root: Path) -> dict[str, Any]`

- [x] **Step 1: Write failing command and lifecycle tests**

  Assert the command uses `podman run --rm -i --network none --read-only`, the
  digest-pinned ROCm image, read-only repository/data/model-cache mounts,
  `vqa_transformers_runner`, pinned Qwen3.5 revision, `--max-new-tokens 32`, and
  `--answer-policy current`. Reject a TTY, network access, mutable model pins,
  a second attempt, source drift, or a partial/mutated result set.

- [x] **Step 2: Run lifecycle tests and verify RED**

  Run:

  `uv run pytest -q tests/test_vqa_qwen_transcript_answer.py -k 'command or attempt or freeze'`

- [x] **Step 3: Implement exclusive command/terminal evidence**

  Write `command.json` before launch. On either a returned failure or launch
  exception, preserve sanitized `stderr.txt`, `terminal.json`, and
  `validated-results.json`. Never expose transcript text through stderr.

- [x] **Step 4: Implement and test the 24-answer freeze**

  Require exactly 24 unique `(condition_id, video_id)` results, the exact model
  and revision, `status=passed`, and one non-empty answer each. Freeze the raw
  result, terminal, validation, command, and stderr hashes in
  `predictions-freeze.json`; otherwise freeze `not_ready` without annotations.

- [x] **Step 5: Run Task 3 verification**

  Run the complete new test file and:

  `uv run ruff check src/trec_video_2026/vqa_qwen_transcript_answer.py tests/test_vqa_qwen_transcript_answer.py`

---

### Task 4: Add post-freeze review, CLI, execute, and report

**Files:**
- Create: `scripts/run_qwen_transcript_answer.py`
- Create: `tests/test_run_qwen_transcript_answer.py`
- Modify: `src/trec_video_2026/vqa_qwen_transcript_answer.py`
- Modify: `docs/experiments/2026-08-16-qwen-aligner-quality-diagnostic.qmd`
- Regenerate: `docs/experiments/2026-08-16-qwen-aligner-quality-diagnostic.md`
- Modify: `docs/experiments/artifacts/qwen-aligner-quality-20260816/README.md`
- Modify: `docs/experiments/artifacts/qwen-aligner-quality-20260816/SHA256SUMS`
- Create: `docs/experiments/artifacts/qwen-aligner-quality-20260816/transcript-answer-summary.json`

**Interfaces:**
- CLI commands: `prepare`, `run`, `freeze`, `bind-annotations`, `review-packet`, `score`, `status`.
- `bind_answer_annotations(root: Path, *, bound_at: str) -> dict[str, Any]`
- `build_blind_review_packet(root: Path) -> dict[str, Any]`
- `score_answer_review(root: Path, judgments: Mapping[str, Any], *, scored_at: str) -> dict[str, Any]`

- [x] **Step 1: Write failing post-freeze/CLI tests**

  Assert annotation binding refuses to open files before a ready prediction
  freeze; hashes exactly the 12 canonical annotation files before parsing; and
  rejects mutation. The review packet must deterministically blind arm names,
  include exactly 24 answers, and contain no transcript. Scoring accepts only
  `correct`, `incorrect`, or `ambiguous`, requires one judgment per blinded
  answer, reports arm totals plus normalized exact match/token F1, and writes
  no tracked artifact itself.

- [x] **Step 2: Run post-freeze/CLI tests and verify RED**

  Run:

  `uv run pytest -q tests/test_vqa_qwen_transcript_answer.py tests/test_run_qwen_transcript_answer.py`

  Expected: fail until the post-freeze transitions and CLI exist.

- [x] **Step 3: Implement the thin CLI and post-freeze transitions**

  CLI stdout is restricted to status, counts, and decision. It exposes no
  model override, arbitrary source path, retry, delete, or credential option.

- [x] **Step 4: Run the sole local attempt**

  Execute in order against the fixed private root:

  ```bash
  uv run python scripts/run_qwen_transcript_answer.py prepare
  uv run python scripts/run_qwen_transcript_answer.py run
  uv run python scripts/run_qwen_transcript_answer.py freeze
  uv run python scripts/run_qwen_transcript_answer.py bind-annotations
  uv run python scripts/run_qwen_transcript_answer.py review-packet
  ```

  Complete the blinded 24-row review once, then run `score`. Preserve failures;
  do not retry inference.

- [x] **Step 5: Record only the sanitized aggregate**

  The tracked JSON contains row count, completion count, per-arm
  correct/incorrect/ambiguous totals, paired totals, exact-match/token-F1 means,
  outcome, private freeze hashes, `$0.00` paid cost, and a privacy declaration.
  It contains no IDs, questions, transcripts, references, answers, or row-level
  judgments.

- [x] **Step 6: Regenerate and verify the report**

  Run focused pytest, Ruff, `git diff --check`, JSON parsing, tracked checksum
  verification, privacy/credential scans, and the repository's QMD render
  command. Confirm the generated Markdown is newer than its QMD source.

- [x] **Step 7: Commit the verified implementation and result**

  Stage only the plan, module, CLI, tests, QMD/generated Markdown, and sanitized
  artifact files. Never stage `.claude/` or `data/`.
