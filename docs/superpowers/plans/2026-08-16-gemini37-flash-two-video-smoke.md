# Gemini 3.7 Flash Two-Video Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute one immutable two-video native-AV Batch comparison between pinned Gemini 3.1 Pro and stable Gemini 3.7 Flash, then freeze and report the compatibility and two-row diagnostic result.

**Architecture:** Add a dedicated deep module for protocol validation, request construction, provider lifecycle, freeze, and post-freeze scoring; keep a thin CLI as the only live-provider entrypoint. Reuse two existing regular-file WebM deliveries without copying or remuxing them, upload each once, and reuse its active URI across both model arms. Preserve the completed 475-row runner and evidence byte-for-byte.

**Tech Stack:** Python 3.12, `google-genai>=2,<3`, pytest, JSON/SHA-256 freezes, Gemini File and Batch APIs, and Quarto.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-16-gemini37-flash-two-video-smoke-design.md` exactly.
- Preserve `data/vqa/benchmarks/direct-gemini-batch-475-v1/` and all completed Gemini/Qwen evidence byte-for-byte.
- Use exactly two rows: the first two frozen rows in the 475-row protocol, selected without labels or prior outputs.
- Reuse the same two delivery bytes and File API URIs for both arms.
- Submit exactly one two-request Batch job per arm; no paid retry.
- Keep labels, references, prior predictions, and scores closed until both arms freeze.
- Reject any deterministic estimated cost above `$0.15`.
- Keep private row identities, questions, media, requests, responses, and scores under ignored `data/`.
- Never print, stage, hash into tracked evidence, or pass `GEMINI_API_KEY` as a CLI argument.
- Never stage the unrelated `.claude/` directory.

---

### Task 1: Freeze the two-row protocol and arm-specific requests

**Files:**
- Create: `src/trec_video_2026/vqa_gemini37_flash_smoke.py`
- Create: `tests/test_vqa_gemini37_flash_smoke.py`
- Modify: `pyproject.toml`
- Regenerate: `uv.lock`

**Interfaces:**
- Produces `prepare_protocol(*, source_root: Path, root: Path, created_at: str) -> dict[str, Any]`.
- Produces `verify_inputs(root: Path) -> dict[str, Any]`.
- Produces `build_inlined_request(*, row: Mapping[str, Any], upload: Mapping[str, Any], arm: str) -> types.InlinedRequest`.
- Exposes arm IDs `control_31pro` and `candidate_37flash` with exact model/config bindings.

- [x] **Step 1: Add the Gemini SDK extra**

  Add:

  ```toml
  [project.optional-dependencies]
  gemini = ["google-genai>=2.0.0,<3"]
  ```

  Run `uv lock` and verify `uv run --extra gemini python -c 'from google import genai; print(genai.__version__)'` reports a 2.x version.

- [x] **Step 2: Write failing protocol tests**

  Create synthetic historical protocol, upload, delivery, and provenance
  fixtures. The tests must prove that preparation:

  - selects only frozen rows 0 and 1 in order;
  - emits exactly two rows and no label/answer/reference/metric/prior-output key;
  - binds source protocol, historical upload, delivery, and provenance hashes;
  - rejects non-475 source protocols, duplicate IDs, duration drift, missing or
    symlinked delivery files, hash mismatch, forbidden keys, and existing roots;
  - records 39.576 total seconds and a deterministic estimate below `$0.15`.

  Name the primary break:

  ```python
  def test_prepare_protocol_binds_two_regular_deliveries_without_private_scores(...):
      protocol = prepare_protocol(...)
      assert len(protocol["rows"]) == 2
      assert protocol["rows"][0]["source_order_index"] == 0
      assert protocol["rows"][1]["source_order_index"] == 1
      assert protocol["estimated_cost_ceiling_usd"] <= 0.15
  ```

- [x] **Step 3: Run protocol tests and confirm RED**

  Run:

  `uv run --extra gemini pytest -q tests/test_vqa_gemini37_flash_smoke.py -k 'prepare or privacy or cost'`

  Expected: collection fails because `vqa_gemini37_flash_smoke` does not yet
  exist.

- [x] **Step 4: Implement minimal exclusive protocol preparation**

  The protocol schema must include:

  ```python
  {
      "schema_version": 1,
      "experiment_id": "gemini37-flash-two-video-smoke-v1",
      "status": "inputs_frozen",
      "source_protocol": {"path": ..., "sha256": ...},
      "arms": {
          "control_31pro": {"model": "gemini-3.1-pro-preview", "temperature": 0.0},
          "candidate_37flash": {"model": "gemini-3.7-flash", "temperature": None},
      },
      "rows": [...],
      "total_duration_seconds": 39.576,
      "estimated_cost_ceiling_usd": ...,
  }
  ```

  Write `protocol.json` and `inputs-freeze.json` with exclusive creation. The
  verifier must re-hash the historical protocol, delivery bytes, historical
  upload record, delivery provenance, and both new JSON artifacts every time.

- [x] **Step 5: Write failing request serialization tests**

  Assert both real SDK objects serialize to native FileData video followed by
  the unchanged direct-answer prompt, `fps=4.0`, high part-media resolution,
  high thinking with thoughts excluded, seed 42, plain text, and 2,048 output
  tokens. Assert only 3.1 contains `temperature=0.0`; 3.7 must serialize no
  `temperature`, `top_p`, `top_k`, or `candidate_count` key.

- [x] **Step 6: Run request tests and confirm RED**

  Run:

  `uv run --extra gemini pytest -q tests/test_vqa_gemini37_flash_smoke.py -k request`

  Expected: `build_inlined_request` is absent or does not yet satisfy the
  arm-specific config.

- [x] **Step 7: Implement the minimal request builder and pass Task 1**

  Use `google.genai.types.InlinedRequest`, `Content`, `Part`, `FileData`,
  `VideoMetadata`, `PartMediaResolution`, `GenerateContentConfig`, and
  `ThinkingConfig`. Do not add a response schema or map condition.

  Run the complete test file plus:

  `uv run ruff check src/trec_video_2026/vqa_gemini37_flash_smoke.py tests/test_vqa_gemini37_flash_smoke.py`

---

### Task 2: Add one-attempt provider lifecycle and immutable freeze

**Files:**
- Modify: `src/trec_video_2026/vqa_gemini37_flash_smoke.py`
- Modify: `tests/test_vqa_gemini37_flash_smoke.py`

**Interfaces:**
- Produces `preflight_models(root: Path, client: Any, checked_at: str) -> dict[str, Any]`.
- Produces `upload_rows(root: Path, client: Any, *, sleep: Callable[[float], None]) -> dict[str, Any]`.
- Produces `submit_arm(root: Path, arm: str, client: Any, *, submitted_at: str) -> dict[str, Any]`.
- Produces `record_status(root: Path, arm: str, client: Any, *, checked_at: str) -> dict[str, Any]`.
- Produces `freeze_predictions(root: Path, *, frozen_at: str) -> dict[str, Any]`.

- [x] **Step 1: Write failing preflight and upload tests**

  With complete SDK-shaped fakes, assert preflight fetches both exact models,
  records model IDs/versions/capabilities without credentials, and refuses
  submission if either model is missing. Assert upload reads the two verified
  deliveries, uploads each exactly once, polls to `ACTIVE`, and writes two
  exclusive records. Reject FAILED/timeout, mutation, duplicate record, wrong
  MIME type, or a URI reused from the expired historical record.

- [x] **Step 2: Run preflight/upload tests and confirm RED**

  Run:

  `uv run --extra gemini pytest -q tests/test_vqa_gemini37_flash_smoke.py -k 'preflight or upload'`

- [x] **Step 3: Implement preflight and one-upload lifecycle**

  Call `client.models.get(model=...)`, `client.files.upload(...)`, and
  `client.files.get(name=...)`. Sanitize model snapshots to documented model
  metadata and never serialize client/config/environment state.

- [x] **Step 4: Write failing submission tests**

  Assert each arm submits exactly one `client.batches.create` call with the
  arm's exact model and two requests; both arm request metadata binds the same
  ordered video IDs, questions, upload URIs, delivery hashes, MIME types, and
  protocol hash. Reject unknown arms, missing preflight/upload evidence,
  duplicate submissions, mutated inputs, wrong request count, or any
  response/job file already present.

- [x] **Step 5: Run submission tests and confirm RED**

  Run:

  `uv run --extra gemini pytest -q tests/test_vqa_gemini37_flash_smoke.py -k submit`

- [x] **Step 6: Implement one paid transition per arm**

  The job display names are
  `gemini37-smoke-control-31pro` and
  `gemini37-smoke-candidate-37flash`. Write `jobs/<arm>.json` before returning.
  Expose no retry or delete command.

- [x] **Step 7: Write failing status/freeze tests**

  Exercise pending, failed, succeeded, response-level error, empty text,
  duplicate ID, missing ID, wrong arm metadata, wrong model version, and
  mutation cases. A successful freeze contains two responses per arm, response
  IDs, finish reasons, model versions, complete usage metadata, request hashes,
  and provider snapshot hashes, but no references or prior outputs.

  Require each arm's aggregate prompt tokens to fall inclusively between
  `34_193` and `61_548`, the design's documented custom-FPS reconciliation
  band. An out-of-band or missing usage value must freeze a terminal
  `custom_fps_not_verified` decision rather than a quality-ready one.

- [x] **Step 8: Run status/freeze tests and confirm RED**

  Run:

  `uv run --extra gemini pytest -q tests/test_vqa_gemini37_flash_smoke.py -k 'status or freeze or fps'`

- [x] **Step 9: Implement status and freeze, then pass Task 2**

  Status snapshots are additive under `status/<arm>/`. Freeze is exclusive and
  terminal. Calculate observed cost from prompt tokens plus
  `totalTokenCount - promptTokenCount`, using frozen Batch rates for each arm.

  Run the complete test file, Ruff, and `git diff --check`.

---

### Task 3: Add the thin CLI and post-freeze scorer

**Files:**
- Create: `scripts/run_gemini37_flash_smoke.py`
- Create: `tests/test_run_gemini37_flash_smoke.py`
- Modify: `src/trec_video_2026/vqa_gemini37_flash_smoke.py`
- Modify: `tests/test_vqa_gemini37_flash_smoke.py`

**Interfaces:**
- CLI commands: `prepare`, `preflight`, `upload`, `submit`, `status`, `freeze`,
  `bind-annotations`, `score`, and `show-status`.
- Produces `freeze_annotation_bindings(root: Path, annotations_dir: Path, *, bound_at: str) -> dict[str, Any]`.
- Produces `score_postfreeze(root: Path, annotations_dir: Path, *, scored_at: str) -> dict[str, Any]`.

- [x] **Step 1: Write failing CLI boundary tests**

  Run `main(argv, client_factory=...)` against fakes. Assert each command calls
  one domain transition, emits only sanitized aggregate JSON, requires the
  canonical source/private root paths, and returns nonzero with a generic
  sanitized error on invalid input. Assert `--help` exposes no retry, delete,
  model override, arbitrary panel, or credential option.

- [x] **Step 2: Run CLI tests and confirm RED**

  Run:

  `uv run --extra gemini pytest -q tests/test_run_gemini37_flash_smoke.py`

  Expected: the CLI file is absent.

- [x] **Step 3: Implement the thin CLI**

  Live commands construct `genai.Client(api_key=os.environ["GEMINI_API_KEY"])`.
  `prepare`, `freeze`, `score`, and `show-status` do not construct a provider
  client. Every printed object is restricted to status, arm, counts, decision,
  and aggregate cost.

- [x] **Step 4: Write failing post-freeze score tests**

  Assert scoring refuses absent/incomplete/mutated prediction freezes before
  reading any annotation file. Add a hash-only `bind-annotations` transition
  that can run only after a ready two-arm freeze, opens exactly two canonical
  annotation files as raw bytes without parsing them, and exclusively freezes
  their exact identities, paths, and SHA-256 values. Assert scoring refuses an
  absent, incomplete, or mutated binding manifest and verifies each raw hash
  before JSON parsing. With a valid binding, report normalized exact match,
  token F1, paired win/tie/loss counts, and one of the three design decisions.
  Keep the question/reference/answers and row metrics only in the private score.

- [x] **Step 5: Run scorer tests and confirm RED**

  Run:

  `uv run --extra gemini pytest -q tests/test_vqa_gemini37_flash_smoke.py -k score`

- [x] **Step 6: Implement the minimal scorer and pass Task 3**

  Use case-folded alphanumeric tokens. Exact match compares normalized strings;
  token F1 uses multiset overlap. Candidate improvement/regression is decided
  from per-row token F1, with exact match as a secondary diagnostic. Do not call
  any evaluator model or external metric endpoint.

  Run both new test files, Ruff, JSON serialization checks, and
  `git diff --check`.

- [x] **Step 7: Commit the provider-free harness**

  Stage only the spec, plan, dependency files, module, CLI, and tests. Commit:

  `git commit -m "feat: add frozen Gemini 3.7 two-video smoke"`

---

### Task 4: Preflight and execute the sole Batch A/B

**Files:**
- Private create: `data/vqa/benchmarks/gemini37-flash-two-video-smoke-v1/`

**Interfaces:**
- Consumes only the frozen source protocol and two preserved delivery bundles before freeze.
- Produces immutable private protocol, model, upload, job, status, prediction, and score artifacts.

- [ ] **Step 1: Run provider-free final preflight**

  Verify:

  ```bash
  test "$(readlink -f data)" = "/home/npatta01/data/competitions/trec_video_2026/data"
  stat -c '%a' /home/npatta01/.config/trec_video_2026/gemini.env
  uv run --extra gemini pytest -q \
    tests/test_vqa_gemini37_flash_smoke.py \
    tests/test_run_gemini37_flash_smoke.py
  uv run ruff check \
    src/trec_video_2026/vqa_gemini37_flash_smoke.py \
    scripts/run_gemini37_flash_smoke.py \
    tests/test_vqa_gemini37_flash_smoke.py \
    tests/test_run_gemini37_flash_smoke.py
  git diff --check
  ```

  Require credential mode `600`, both delivery hashes valid, estimated cost at
  or below `$0.15`, and no private-source or credential pattern in tracked
  changes.

- [ ] **Step 2: Prepare and inspect only aggregate protocol evidence**

  Run `prepare` once. Inspect only status, row count, duration sum, cost ceiling,
  privacy-scan count, and hashes. Never print the protocol or row identities.

- [ ] **Step 3: Run authenticated no-inference model preflight**

  Source the approved private environment without printing it, invoke
  `preflight` once, and require exact availability of both arm model IDs.
  A failure stops before upload/submission.

- [ ] **Step 4: Upload the two preserved deliveries once**

  Invoke `upload` once. Require 2/2 `ACTIVE` records and re-run input/hash
  verification. Reuse those URIs for both arms.

- [ ] **Step 5: Submit exactly one Batch job per arm**

  Invoke `submit --arm control_31pro` once and
  `submit --arm candidate_37flash` once. Preserve any provider rejection as the
  sole terminal attempt; never retry or replace a row.

- [ ] **Step 6: Monitor to terminal and freeze**

  Poll both job records with `status` at outcome-driven intervals, appending
  snapshots. When both are terminal, run `freeze` once. If either is failed or
  incomplete, preserve `not_ready` and stop before score inputs.

- [ ] **Step 7: Bind and score only after a ready freeze**

  Run `bind-annotations` only if `show-status` reports a complete prediction
  freeze with custom FPS verified. Require its two-row hash-only manifest, then
  run `score`. Inspect row-level answers privately, then expose only aggregate
  completion, usage, cost, paired diagnostic counts, and decision.

---

### Task 5: Correct the Qwen interpretation and record Gemini evidence

**Files:**
- Modify: `docs/experiments/2026-08-16-qwen-aligner-quality-diagnostic.qmd`
- Modify: `docs/project-status/research-speech-transcripts.qmd`
- Modify: `docs/project-status/research-visual-evidence.qmd`
- Create: `docs/experiments/artifacts/gemini37-flash-two-video-smoke-20260816/README.md`
- Create: `docs/experiments/artifacts/gemini37-flash-two-video-smoke-20260816/summary.json`
- Create: `docs/experiments/artifacts/qwen-aligner-quality-20260816/aligned-english-metric-audit.json`
- Modify: `docs/experiments/artifacts/qwen-aligner-quality-20260816/README.md`
- Modify: `docs/experiments/artifacts/qwen-aligner-quality-20260816/SHA256SUMS`
- Regenerate: `docs/experiments/2026-08-16-qwen-aligner-quality-diagnostic.md`
- Regenerate: `docs/project-status/research-speech-transcripts.md`
- Regenerate: `docs/project-status/research-visual-evidence.md`
- Regenerate: `reports/project-status/research-speech-transcripts.html`
- Regenerate: `reports/project-status/research-visual-evidence.html`

**Interfaces:**
- Preserves original Qwen score while correcting its evidence-validity interpretation.
- Publishes only sanitized Gemini/Qwen aggregates and immutable hashes.

- [ ] **Step 1: Freeze the sanitized Qwen metric audit**

  Record only these aggregate facts from the private read-only audit:

  - all source/hash bindings and stored booleans reproduced;
  - 12/12 Qwen outputs were non-empty;
  - 8/12 Qwen and 10/12 Whisper quality statuses were accepted;
  - only one row directly requested a recoverable literal phrase/name, one was
    semantic conversation evidence, and ten did not support exact transcript
    containment;
  - raw Whisper had one exact hit that the frozen timestamp gate excluded;
  - mean unordered official-answer token recall was `0.035` Qwen and `0.212`
    Whisper, explicitly post-hoc and non-promotional.

  Exclude identities, questions, answers, transcripts, and row-level scores.

- [ ] **Step 2: Correct the owning Qwen claim without rewriting history**

  Keep the original frozen `0/12` versus `0/12` score. Change the quality
  evidence state from valid-negative/non-directional to
  `construct-invalid for ASR quality`; retain the valid 12/12 operational
  completion observation. State that the selector's keyword heuristic admitted
  ten non-literal tasks and that scaling the same metric is prohibited.

- [ ] **Step 3: Create the sanitized Gemini aggregate**

  Record two-arm completion, returned model provenance, custom-FPS decision,
  aggregate prompt/output tokens, observed cost estimate, paired exact/token-F1
  counts, decision, protocol/prediction/score hashes, and limitations. Exclude
  every private row field and provider URI/name.

- [ ] **Step 4: Update the primary theme and regenerate outputs**

  Follow `docs/project-status/adding-experiments.qmd`: one canonical Gemini
  claim under Visual and Temporal Evidence, a cross-link from Speech and
  Transcript Evidence, and no duplicated full result. Render Markdown/HTML only
  through Quarto.

- [ ] **Step 5: Verify, independently review, and commit**

  Run new and affected tests, Ruff, JSON parsing, tracked checksums, private-ID
  and credential scans, relative-link checks from every generated output,
  QMD-to-MD/HTML byte comparisons, `git diff --check`, and an independent
  read-only review. Commit only intended files with:

  `git commit -m "docs: record Gemini 3.7 smoke and Qwen metric audit"`.
