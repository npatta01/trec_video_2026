# Full-panel VQA Silver-Oracle and Election Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and score a full-475 silver-oracle evidence factorial and three
reference-blind candidate-election tests without manual full-panel grading.

**Architecture:** A pure contract module validates reference-assisted numeric
selector outputs, builds label-free G/H/L × N/F/S answer requests, and assembles
reference-blind election packets. A pinned Qwen3-Omni runtime performs selector
and judge roles in separate jobs; the existing Qwen3.5-4B runner performs all
nine matched answer cells. Private checksum-bound bundles cross the label
boundary only after answer and ranking freezes.

**Tech Stack:** Python 3.12, pytest, PyAV/FFmpeg, Qwen3-Omni-30B-A3B-Thinking,
Qwen3.5-4B, CUDA dstack, Hugging Face Buckets, the existing METEOR/BERTScore/STS
scorer, and Quarto QMD reporting.

## Global Constraints

- Implement the exact design in
  `docs/superpowers/specs/2026-08-09-vqa-silver-oracle-election-design.md`.
- Use all 475 annotated rows with local media and keep the 1,883 test rows
  untouched.
- Never stage or commit raw media, questions, references, transcripts, frames,
  predictions, rankings, raw logs, or private paths.
- Freeze protocol, prompts, model revisions, runtime images, seeds, panel order,
  metrics, gates, and budgets before selector inference.
- Selector-derived arms are diagnostic silver-oracle arms, never deployable
  baselines or human-oracle claims.
- Answer and judge requests reject reference text and arbitrary selector text
  recursively.
- Run a ten-row operational canary without treatment scoring before full-panel
  GPU work.

---

### Task 1: Freeze pure protocol and selector contracts

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/silver_oracle.py`
- Create: `tests/test_silver_oracle.py`
- Create private: `data/vqa/benchmarks/vqa-silver-oracle-election-20260809/protocol.json`

**Interfaces:**
- Produces: `validate_selector_result(payload, row) -> dict[str, object]`,
  `parse_timestamped_whisper(text, duration) -> list[WhisperSegment]`,
  `build_frame_plans(row, selector, inventory) -> dict[str, list[dict]]`, and
  `protocol_sha256(protocol) -> str`.

- [ ] Write failing tests for exact selector keys, status values, two-interval
  and three-segment limits, interval duration budget, range/order/overlap,
  non-localizable emptiness, invalid transcript indices, forbidden generated
  text, and canonical hashes.
- [ ] Run `uv run pytest -q tests/test_silver_oracle.py` and require failures
  caused by the missing module.
- [ ] Implement immutable dataclasses and validators. Reuse
  `relevant_scene_candidate_plans()` from `temporal_bottleneck.py` for H/L
  allocation; add only a thin adapter for the selector's chosen intervals.
- [ ] Generate the private protocol from the sorted 475-row label-free manifest,
  six explicit missing-media annotations, model/runtime pins, prompts, seeds,
  contrasts, gates, and caps. Assert the frozen panel digest.
- [ ] Run focused tests, Ruff, `git diff --check`, placeholder scan, and a
  recursive credential/private-data scan.
- [ ] Commit the contract, tests, design, plan, panel review, and only the safe
  protocol SHA sidecar as `experiment: freeze VQA silver-oracle protocol`.

### Task 2: Build the private selector bundle and Qwen3-Omni runtime

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/qwen3_omni_structured.py`
- Create: `tests/test_qwen3_omni_structured.py`
- Create: `scripts/build_vqa_silver_oracle_bundle.py`
- Create: `scripts/run-vqa-silver-oracle-dstack.sh`
- Create: `dstack/vqa-silver-oracle-selector-canary.dstack.yml`
- Create: `dstack/vqa-silver-oracle-selector-full-{00,01,02,03}.dstack.yml`

**Interfaces:**
- Consumes: selector rows with `video_path`, `duration_seconds`, `question`,
  `reference`, and indexed `timestamped_whisper`.
- Produces: append-only `selector-results.jsonl`, `runtime.json`,
  `completion.json`, `SHA256SUMS`, and a terminal status for each chunk.

- [ ] Write failing tests for safe relative media paths, exact panel/chunk
  partition, reference-required selector rows, reference-forbidden judge rows,
  terminal JSON extraction, no persisted reasoning, one retry only, seeds
  42/43, append-only resume, model/runtime identity, and failure finalization.
- [ ] Run the two focused test files and confirm the new runtime tests fail.
- [ ] Implement the Qwen3-Omni loader by reusing the exact model, revision,
  runtime, BF16/CUDA, EOS, and first-forward attestation seams from
  `qwen3_omni_dstack.py`. For selector mode, pass one native video with
  `use_audio_in_video=True`; emit only the validated terminal object and raw
  output SHA.
- [ ] Build one private tar bundle containing the 475 exact source videos,
  selector JSONL, timestamped Whisper bytes, protocol, source manifest, and
  checksums. Refuse symlinks, labels outside the selector file, missing files,
  duplicate IDs, and any test-set path.
- [ ] Render four disjoint 119/119/119/118 full chunks plus the fixed ten-row
  canary and one label-free resource sentinel. Each job uses an H100 with at
  least 80GB VRAM, at least 192GB host RAM, `max_price: 3.35`; full shards use
  `max_duration: 200m` (3h20m), and every run uses a unique immutable Bucket prefix.
- [ ] Run tests, shell syntax checks, dstack dry-run previews, source-bundle
  hash verification, and a secret scan before any launch.
- [ ] Commit as `feat: add structured Qwen3-Omni selector runtime`.

### Task 3: Execute and verify the selector canary and full panel

**Files:**
- Create private: `data/vqa/benchmarks/vqa-silver-oracle-election-20260809/selector/`

**Interfaces:**
- Produces: one merged, sorted, checksum-bound 475-row selector freeze consumed
  by Task 4.

- [ ] Upload the selector bundle to its private immutable Bucket prefix and
  verify the remote manifest before provisioning.
- [ ] Launch the B4 canary detached. Require 11/11 terminal success (the frozen
  ten strata plus one label-free maximum-duration resource sentinel), at most
  two repairs among the original ten, in-range intervals/segment IDs,
  native-video/audio and EOS attestation, the frozen H100 memory/runtime/cost
  gates, and no answerer-visible leakage. Do not score answers.
- [ ] If the canary passes, launch all four full chunks in parallel without
  changing code, prompt, seeds, or protocol. If it fails, stop and consume a
  new protocol revision rather than patching outputs.
- [ ] Retrieve each chunk, verify every payload checksum, merge only disjoint
  exact IDs, and require 475 terminal records. Stop if malformed-after-retry
  exceeds 5%; record non-localizable rows without manual repair.
- [ ] Freeze the merged selector file and its digest before building any answer
  request.

### Task 4: Construct and audit the nine evidence cells

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/silver_oracle.py`
- Modify: `tests/test_silver_oracle.py`
- Create: `scripts/build_vqa_silver_evidence.py`
- Create private: `data/vqa/benchmarks/vqa-silver-oracle-election-20260809/evidence/`

**Interfaces:**
- Consumes: merged selector freeze, verified global evidence, source videos,
  timestamped Whisper, and protocol.
- Produces: nine 475-row label-free request files, frame provenance, contact
  sheets, leakage audit, and checksums.

- [ ] Add failing tests for G exact reuse, H six/six and L twelve/zero slot
  counts, actual-PTS uniqueness/order, collision fallback, selected transcript
  verbatim identity, N/F/S semantics, exact nine-cell coverage, recursive
  reference rejection, and cross-cell held-fixed identity.
- [ ] Run focused tests and verify the new cases fail.
- [ ] Implement evidence extraction with source-media hashes, requested and
  actual timestamps, frame indices, time bases, image hashes, command/runtime
  identity, transcript source hash, selected line indices, and request hashes.
- [ ] Build all nine requests. The builder accepts only the validated selector
  object; it cannot open annotation files. Generate one contact sheet per
  row/view, not per transcript cell.
- [ ] Run the leakage auditor over every nested request and compare G/F's full
  model-input digest with the verified baseline. Freeze all evidence checksums.
- [ ] Commit implementation/tests only as `feat: build VQA silver-oracle evidence`.

### Task 5: Run nine matched Qwen3.5-4B answer arms

**Files:**
- Create: `dstack/vqa-silver-answer-{gn,gf,gs,hn,hf,hs,ln,lf,ls}.dstack.yml`
- Create: `scripts/run-vqa-silver-answer-dstack.sh`
- Create private: `data/vqa/benchmarks/vqa-silver-oracle-election-20260809/answers/`

**Interfaces:**
- Produces: nine complete 475-row answer freezes with exact evidence/runtime
  bindings.

- [ ] Write config/runner tests for the exact condition set, model revision,
  current prompt, greedy decoding, pixel bounds, evidence URI, unique output
  prefixes, 475 count, no-label mount, telemetry, checksums, and non-overwrite.
- [ ] Reuse `vqa_transformers_runner.py`; do not add a second answer generator.
  The shell wrapper verifies one selected request file and runs exactly one
  condition per job.
- [ ] Run focused tests, Bash syntax checks, and all nine dry-run previews.
- [ ] Launch the nine jobs in parallel. Retrieve only after terminal success;
  verify all checksums and exact request/prediction identity.
- [ ] Require 475 nonempty answers in every cell and freeze predictions before
  loading references or computing metrics.
- [ ] Commit runner/config/tests as `feat: run VQA silver-oracle answer matrix`.

### Task 6: Build and run three reference-blind elections

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/silver_oracle.py`
- Modify: `src/trec_video_2026/vqa_experiments/qwen3_omni_structured.py`
- Modify: `tests/test_silver_oracle.py`
- Modify: `tests/test_qwen3_omni_structured.py`
- Create: `scripts/build_vqa_election_packets.py`
- Create: `dstack/vqa-election-{a,b,c}-{00,01,02,03}.dstack.yml`

**Interfaces:**
- Produces: complete ranked candidate IDs for Pools A, B, and C; candidate
  strings remain private and are joined only by opaque IDs.

- [ ] Add failing tests for Pool A/B/C membership, normalized exact
  deduplication, deterministic per-row randomization, hidden provenance,
  reference/metric rejection, evidence union, complete permutation output,
  no candidate rewriting, and retry/failure boundaries.
- [ ] Implement packet construction after answer freeze. Pool A uses the four
  factorial answers and common G/F evidence; B uses nine new answers and union
  G/H/L evidence; C uses their deduplicated union. Store the secret candidate
  map separately from judge packets.
- [ ] Extend Qwen3-Omni structured mode to rank opaque candidates from actual
  frame/transcript evidence with native audio disabled. The validated response
  is `{"ranking":["opaque-id", ...]}` containing every unique ID exactly once.
- [ ] Run a ten-row schema-only judge canary, then the three pools over four
  chunks each. Require complete rankings for all 475 rows after at most one
  schema retry.
- [ ] Retrieve, checksum, merge, freeze, and one-time join rankings to candidate
  maps. Preserve randomized-order hashes and ranking hashes.
- [ ] Commit implementation/tests/configs as `feat: add reference-blind VQA election`.

### Task 7: Score metrics, uncertainty, ceilings, and NDCG

**Files:**
- Create: `scripts/score_vqa_silver_oracle.py`
- Create: `tests/test_score_vqa_silver_oracle.py`
- Create: `docs/experiments/artifacts/vqa-silver-oracle-election-475-20260809.json`
- Create private: `data/vqa/benchmarks/vqa-silver-oracle-election-20260809/sts-cache.json`

**Interfaces:**
- Produces: one safe aggregate artifact containing nine arm scores, declared
  contrasts, three election results, reference-aware ceilings, paired
  intervals, NDCG, gate decisions, runtime, and provenance.

- [ ] Write failing tests for exact ID-set joins, prediction freeze precedence,
  metric arithmetic, composite C, 10,000-resample paired bootstrap, evidence
  and election gates, per-row reference-aware maxima, capture fractions,
  NDCG@K, candidate counts, completion denominators, and safe-artifact fields.
- [ ] Reuse the pinned METEOR/BERTScore/STS implementations and private cache
  contract from `score_vqa_factorial.py`; do not change model, endpoint, text
  normalization, retry count, or component weights.
- [ ] Score all candidate answers once, derive arm/election/ceiling aggregates,
  apply gates, and write the safe artifact only after all provenance checks.
- [ ] Independently recompute means, deltas, confidence intervals, factorial
  contrasts, ceiling selections, NDCG, and gate booleans from the private row
  table. Run all scorer/evaluation tests and Ruff.
- [ ] Commit as `results: record full VQA silver-oracle and election scores`.

### Task 8: Record the decision and prepare the competition branch

**Files:**
- Modify: `docs/project-status/research-routing.qmd`
- Modify: `docs/project-status/research-visual-evidence.qmd`
- Modify: `docs/project-status/research-speech-transcripts.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd` only if the active branch changes
- Modify: `docs/superpowers/plans/2026-08-09-vqa-silver-oracle-election.md`

**Interfaces:**
- Produces: canonical, condition-bound claims and one selected next competition
  architecture.

- [ ] Record observed results separately from interpretation. Visual Evidence
  owns visual-timing claims, Speech owns transcript-timing claims, and Routing
  owns election/fusion and the selected architecture; cross-link rather than
  duplicate.
- [ ] Apply the predeclared four-way decision. If election passes, immediately
  plan eight deterministic evidence passes plus ranked-10 output. If evidence
  also passes, include a reference-blind locator. If both fail, switch to
  fold-based fine-tuning/stronger native-audio answerer.
- [ ] Regenerate QMD-derived Markdown and HTML, verify links/fragments/privacy,
  inspect phone and desktop rendering, and run `git diff --check`.
- [ ] Run a Sol code/result review, address consequential findings, verify the
  final branch against `origin/main`, and report exact merge/rebase needs without
  disturbing unrelated dirty files.
