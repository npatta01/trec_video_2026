# VQA Factorial Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and run the semantic-scoring gate plus a bounded
Qwen3.5-4B/Qwen3-VL-8B model-by-prompt VQA factorial on the complete 475-row
annotated development panel.

**Architecture:** Add a reproducible semantic scorer around the existing saved
prediction contract; expose one deterministic question-aware answer policy in
the shared model runner; and drive model/prompt cells through one profile-based
dstack runner that produces checksummed Hugging Face Bucket artifacts. Start
with a three-row integrity/throughput canary, then expand only passing cells to
the full 475-row panel.

**Tech Stack:** Python 3.12, pytest, NLTK METEOR, bert-score 0.3.12,
FacebookAI/roberta-large, UMBC STS HTTP endpoint, Transformers at pinned source
revision, CUDA BF16, dstack 0.20.29, Hugging Face Buckets.

## Global Constraints

- Quality panels contain only VQA question/answer annotations: 475 usable 2025
  training rows for development and 1,883 usable 2025 testing rows reserved for
  one-shot confirmation. VTT and unlabeled 2026 rows are excluded from quality
  scoring.
- Remote answer inference must be label-free. References join locally only
  after prediction files and checksums freeze.
- Models are exactly
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and
  `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- The Transformers source revision remains
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`; runtime image remains
  `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
- Base evidence remains the same uniform-12 frame policy, 65,536–262,144
  pixels per frame, accepted Whisper-large-v3 transcript policy, deterministic
  decoding, and 64 output tokens.
- Metrics are METEOR, BERTScore F1, STS, and ranked NDCG. Exact match/token F1
  are diagnostics. The analysis composite is
  `C = mean(METEOR, BERTScore F1, STS)` and is not called official.
- Single-response NDCG is recorded as the NIST-degenerate value 1.0 and clearly
  marked non-informative.
- Every remote bundle records selected frames and timestamps, transcript
  provenance, request policy, runtime identity, timing/VRAM, predictions, and a
  verified `SHA256SUMS`.
- Do not overwrite an existing local or remote artifact prefix.
- Use `apply_patch` for tracked edits. Preserve all pre-existing dirty worktree
  changes and never stage dataset contents.

---

### Task 1: Reproducible semantic scorer and frozen baseline artifact

**Files:**
- Modify: `src/trec_video_2026/vqa_evaluation.py`
- Modify: `tests/test_vqa_evaluation.py`
- Create: `scripts/score_vqa_semantic_panel.py`
- Create: `tests/test_score_vqa_semantic_panel.py`
- Create (generated): `docs/experiments/artifacts/qwen35-4b-12f-whisper-semantic-20260809-d1.json`

**Interfaces:**
- Produces: `semantic_composite(metrics: Mapping[str, float]) -> float`
- Produces: `ndcg_from_gains(gains: Sequence[float], k: int | None = None) -> float`
- Produces: `ranked_cutoff_metrics(rows: Sequence[Mapping[str, float]], cutoffs: Sequence[int] = (1, 5, 10)) -> dict[str, Any]`
- CLI consumes frozen prediction JSON files, annotation root, BERT model path,
  and output path; it emits an atomic JSON evidence artifact.

- [ ] **Step 1: Write failing pure-metric tests**

Add literal tests that catch a wrong composite field, reversed ranking, and the
single-response NDCG edge:

```python
def test_semantic_composite_averages_three_declared_metrics() -> None:
    assert semantic_composite(
        {"meteor": 0.2, "bertscore_f1": 0.8, "sts": 0.5}
    ) == pytest.approx(0.5)


def test_ndcg_rewards_better_gain_order() -> None:
    assert ndcg_from_gains([1.0, 0.0]) > ndcg_from_gains([0.0, 1.0])


def test_single_response_ndcg_is_declared_degenerate() -> None:
    assert ndcg_from_gains([0.4]) == 1.0
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run --group dev pytest tests/test_vqa_evaluation.py -q
```

Expected: the three new tests fail because the named functions do not exist.

- [ ] **Step 3: Implement the minimal metric functions**

Use the NIST 2025 gain semantics. Validate finite values and cutoff bounds.
`semantic_composite` must require all three semantic fields rather than silently
dropping one.

- [ ] **Step 4: Write failing scorer-boundary tests**

Create two tiny annotations and two frozen conditions. Inject literal metric
functions into the script-level scorer so tests do not contact UMBC or load
RoBERTa. Assert:

```python
assert artifact["panel"]["completed_per_condition"] == {
    "frames": 2,
    "whisper": 2,
}
assert artifact["conditions"]["frames"]["semantic_composite"] == pytest.approx(0.5)
assert artifact["integrity"]["labels_joined_after_prediction_freeze"] is True
```

Also assert duplicate IDs, missing references, non-finite STS, and a pre-existing
output path fail closed.

- [ ] **Step 5: Run the scorer tests and verify RED**

Run:

```bash
uv run --group dev pytest tests/test_score_vqa_semantic_panel.py -q
```

Expected: failure because the script/module API is absent.

- [ ] **Step 6: Implement the scorer CLI**

The CLI must normalize text exactly as NIST 2025, use URL-encoded STS requests
with timeout and bounded retry, calculate BERTScore in one batched call, retain
per-row metrics, calculate paired bootstrap intervals with seed `20260809`,
hash all inputs, and write atomically only after 475/475 completion for each
declared condition.

- [ ] **Step 7: Verify GREEN and generate the baseline artifact**

Run:

```bash
uv run --group dev pytest tests/test_vqa_evaluation.py tests/test_score_vqa_semantic_panel.py -q
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add=keep-groups \
  --security-opt=label=disable -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/npatta01/.codex/worktrees/5f31/trec_video_2026:/repo \
  -v /home/npatta01/data/competitions/trec_video_2026/data:/data:ro \
  -v /home/npatta01/.cache/huggingface:/cache:ro \
  localhost/trec-vqa-gate-rocm:2026-07-20 \
  python /repo/scripts/score_vqa_semantic_panel.py \
  --annotations /data/vqa/2025/training/annotations \
  --condition frames=/data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/predictions.frames-only.json \
  --condition whisper=/data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/predictions.frames-whisper.json \
  --bert-model /cache/hub/models--FacebookAI--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59 \
  --panel-profile vqa-2025-training-475 \
  --expected-rows 475 \
  --output /repo/docs/experiments/artifacts/qwen35-4b-12f-whisper-semantic-20260809-d1.json
```

Expected: tests pass; artifact reports 475 rows/condition and aggregates within
`1e-6` of the Phase 0 values in the design spec.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/trec_video_2026/vqa_evaluation.py tests/test_vqa_evaluation.py \
  scripts/score_vqa_semantic_panel.py tests/test_score_vqa_semantic_panel.py \
  docs/experiments/artifacts/qwen35-4b-12f-whisper-semantic-20260809-d1.json
git commit -m "feat: add official-style VQA semantic scoring"
```

### Task 2: Question-aware answer policy and auditable runner controls

**Files:**
- Modify: `src/trec_video_2026/vqa_models.py`
- Modify: `src/trec_video_2026/vqa_transformers_runner.py`
- Modify: `tests/test_vqa_models.py`
- Modify: `tests/test_vqa_transformers_runner.py`

**Interfaces:**
- Produces: `classify_question_type(question: str) -> str`
- Extends: `build_controlled_messages(..., answer_policy: str = "current")`
- Extends CLI with `--answer-policy {current,question-aware}`,
  `--min-pixels-per-frame`, and `--max-pixels-per-frame`.
- Every result records `answer_policy`, `question_type`, and actual pixel bounds.

- [ ] **Step 1: Write failing prompt classification and serialization tests**

Use literal cases:

```python
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many cups are on the table?", "count"),
        ("What words are written on the sign?", "visible-text"),
        ("What does she say after opening the door?", "speech"),
        ("What happens before the dog runs away?", "temporal"),
        ("Why does the pan catch fire?", "causal-procedural"),
        ("What color is the bicycle?", "general"),
    ],
)
def test_question_type_is_deterministic(question: str, expected: str) -> None:
    assert classify_question_type(question) == expected
```

Assert the current policy remains byte-for-byte compatible with the existing
instruction and the question-aware policy contains the type-specific contract
but no reference/answer field.

- [ ] **Step 2: Run prompt tests and verify RED**

Run:

```bash
uv run --group dev pytest tests/test_vqa_models.py -q
```

Expected: import/signature failures for the new API.

- [ ] **Step 3: Implement the minimal policy**

Use ordered, anchored keyword/phrase rules with the precedence
`visible-text → speech → count → temporal → causal-procedural → general`.
Keep current policy unchanged. Question-aware output must still request only the
final answer and must never ask the model to expose reasoning.

- [ ] **Step 4: Write failing runner control/result tests**

Assert `generate_answer` forwards the selected policy, `parse_args` rejects
inverted/non-positive pixel bounds, `load_processor` receives the requested
bounds, and `_result` records policy/type/bounds. A realistic mutation that
hardcodes 512² must fail at least one test.

- [ ] **Step 5: Run runner tests and verify RED**

Run:

```bash
uv run --group dev pytest tests/test_vqa_transformers_runner.py -q
```

Expected: failures for absent CLI/result fields.

- [ ] **Step 6: Implement runner controls and verify GREEN**

Keep deterministic `do_sample=False`, thinking disabled, and 64 tokens. Validate
pixel bounds before model loading. Record the policy and classifier output per
request without changing the canonical label-free request schema.

Run:

```bash
uv run --group dev pytest tests/test_vqa_models.py tests/test_vqa_transformers_runner.py -q
uv run --group dev ruff check src/trec_video_2026/vqa_models.py \
  src/trec_video_2026/vqa_transformers_runner.py tests/test_vqa_models.py \
  tests/test_vqa_transformers_runner.py
```

Expected: all pass with no lint output.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/trec_video_2026/vqa_models.py \
  src/trec_video_2026/vqa_transformers_runner.py \
  tests/test_vqa_models.py tests/test_vqa_transformers_runner.py
git commit -m "feat: add question-aware VQA answer policy"
```

### Task 3: Profile-driven factorial dstack runner

**Files:**
- Create: `scripts/build_vqa_factorial_manifest.py`
- Create: `tests/test_build_vqa_factorial_manifest.py`
- Create: `scripts/run-vqa-factorial-dstack.sh`
- Create (generated/private; do not stage):
  `configs/vqa/vqa_factorial_canary_3.json`
- Create: `configs/vqa/vqa_factorial_canary_profile.json`
- Create: `configs/vqa/vqa_factorial_full_evidence_profile.json`
- Create: `configs/vqa/vqa_factorial_full_condition_profiles.json`
- Create: `dstack/vqa-factorial-canary.dstack.yml`
- Create: `dstack/vqa-factorial-full-evidence.dstack.yml`
- Create: `dstack/vqa-factorial-full-4b-current.dstack.yml`
- Create: `dstack/vqa-factorial-full-4b-question-aware.dstack.yml`
- Create: `dstack/vqa-factorial-full-8b-current.dstack.yml`
- Create: `dstack/vqa-factorial-full-8b-question-aware.dstack.yml`

**Interfaces:**
- Manifest builder emits rows containing exactly `video_id` and `question`.
- Canary profile contains exact fields `run_name`, `input_uri`, `remote_uri`,
  `manifest`, `expected_count`, `frame_count`, `conditions`, and
  `repeat_condition`.
- Each condition contains exact fields `condition_id`, `model_id`, `revision`,
  `answer_policy`, `min_pixels`, `max_pixels`, and `request_source`.
- The full topology is one immutable evidence/Whisper profile followed by four
  independent condition profiles. One dstack task is one node; the
  implementation must not claim that a single YAML parallelizes four cells.

- [ ] **Step 1: Write failing manifest/profile tests**

Assert the canary deterministically contains one temporal, one speech, and one
OCR/detail question; manifests contain no answer-like fields; unknown profile
fields, duplicate conditions, non-pinned model revisions, and output-prefix
reuse fail closed.

- [ ] **Step 2: Run and verify RED**

```bash
uv run --group dev pytest tests/test_build_vqa_factorial_manifest.py -q
```

Expected: failure because the builder does not exist.

- [ ] **Step 3: Implement the builder and frozen profiles**

Use the existing 475-row label-free manifest as the source of identity and
questions. Freeze the canary IDs to temporal `CtinYSH3nVw`, speech
`LKiaCf1QZFw`, and OCR/detail `IVqpg_1aWXU`; selection remains question-only
and deterministic. The canary declares all four base cells, one 8B
question-aware 768px OCR-only condition, and one repeated 4B-current base
condition. The full profiles declare one evidence-preparation job and exactly
four one-condition jobs over the same canonical 475 IDs.

- [ ] **Step 4: Write a shell contract test before the runner**

Use a temporary fake `hf`/Python environment to exercise profile validation and
remote-prefix refusal without GPU/network. Assert the runner exits non-zero
before bootstrap when a profile contains a label, revision mismatch, duplicate
condition, stale remote prefix, or an unsafe path.

- [ ] **Step 5: Run shell contract test and verify RED**

Run the new test target documented in
`tests/test_build_vqa_factorial_manifest.py`; expected failure is missing runner.

- [ ] **Step 6: Implement the dstack runner and YAML**

Reuse the existing input-bundle verification, frame/audio preparation, Whisper
quality gate, isolated condition execution, prediction validation, status
summary, checksum creation, and bucket sync. Drive model/policy/pixel settings
only from the validated profile. Run each canary condition in a separate child
process so model memory is released on exit. Stage selected frames, timestamps,
audio provenance, transcripts, requests, predictions, telemetry, and status
inside the durable output bundle before writing and verifying `SHA256SUMS`.
Emit utilization samples from `nvidia-smi` every five seconds while inference
runs. The full evidence job prepares frames/audio/Whisper exactly once; the four
condition jobs consume that immutable checksummed bundle and may run in
parallel. Never print the HF token.

- [ ] **Step 7: Verify generated configs and local tests**

```bash
uv run --group dev pytest tests/test_build_vqa_factorial_manifest.py -q
bash -n scripts/run-vqa-factorial-dstack.sh
uv run --group dev python scripts/build_vqa_factorial_manifest.py \
  --source configs/vqa/qwen35_vqa2025_475_label_free.json \
  --canary-output configs/vqa/vqa_factorial_canary_3.json \
  --expected-source-count 475
```

Expected: tests pass, shell parses, canary has three label-free rows, and the
generated full topology contains one evidence job plus four distinct condition
jobs rather than a fictitious multi-cell single task.

- [ ] **Step 8: Commit Task 3**

```bash
git add scripts/build_vqa_factorial_manifest.py \
  tests/test_build_vqa_factorial_manifest.py scripts/run-vqa-factorial-dstack.sh \
  configs/vqa/vqa_factorial_canary_profile.json \
  configs/vqa/vqa_factorial_full_evidence_profile.json \
  configs/vqa/vqa_factorial_full_condition_profiles.json \
  dstack/vqa-factorial-canary.dstack.yml \
  dstack/vqa-factorial-full-evidence.dstack.yml \
  dstack/vqa-factorial-full-4b-current.dstack.yml \
  dstack/vqa-factorial-full-4b-question-aware.dstack.yml \
  dstack/vqa-factorial-full-8b-current.dstack.yml \
  dstack/vqa-factorial-full-8b-question-aware.dstack.yml
git commit -m "feat: add parallel VQA factorial runner"
```

### Task 4: Canary preview, launch, and integrity decision

**Files:**
- Create (generated/private): `data/vqa/benchmarks/vqa-factorial-canary-20260809-e1-failed/`
- Create (generated/private): `data/vqa/benchmarks/vqa-factorial-canary-20260809-e2/`
- Modify: `docs/superpowers/plans/2026-08-09-vqa-factorial-pilot.md`

**Interfaces:**
- Consumes Task 3 canary dstack config.
- Produces a pass/fail gate and measured per-cell runtime/VRAM projection.

- [x] **Step 1: Run infrastructure preflight and offer check**

```bash
/home/npatta01/.agents/skills/running-dstack-hf-experiments/scripts/preflight.sh
dstack offer --gpu L40S:1 --max-offers 20 --json
```

Expected: direct HF mode and at least two available offers.

- [x] **Step 2: Preview without provisioning**

```bash
echo n | dstack apply -f dstack/vqa-factorial-canary.dstack.yml
```

Expected: resolved single-GPU offer, pinned files/env/resources, no run created.

- [x] **Step 3: Submit exactly once per immutable remote prefix**

```bash
dstack apply -f dstack/vqa-factorial-canary.dstack.yml -y -d
```

Record the returned run name. Do not resubmit the same remote prefix.

- [x] **Step 4: Monitor and download durable artifacts**

Use `dstack ps`, `dstack logs`, and HF Bucket sync. Download to a new private
benchmark directory, verify `SHA256SUMS`, and compare remote/local file hashes.

- [x] **Step 5: Apply the canary gate**

Pass only if every request completes, the repeated deterministic answer is
identical, evidence hashes match, no reference field enters requests, no
OOM/NaN occurs, and the observed 8B throughput supports a full-panel run.
Otherwise stop before Task 5 and record the exact failure.

**Observed 2026-08-09:**

- `e1` was preserved as a checksum-valid failed artifact after the canary caught
  an over-strict buffer-dtype validator. Its terminal-bound payload checksum is
  `4597136e079e96f03aee7e58075938c4f8c9dd91b1e40f4d60eae80eb7540307`.
  The artifact proves all 723 parameters were CUDA BF16 and all three Qwen
  buffers were CUDA float32. No answer-quality conclusion was drawn from `e1`.
- Corrected run `vqa-factorial-canary-20260809-e2` completed with dstack run ID
  `8a280e45-6d6b-4d84-96fe-43c4332b6024`, exit 0, and cost `$0.0496` on one
  VastAI L40S. The terminal-bound payload checksum is
  `cc60b3f3fce9302f7828e28355971a3c19cacec6514602cd55078c4a59b11b3f`;
  all 89 payload files passed `SHA256SUMS` verification.
- All 16 expected predictions completed. `4b-current` and its repeated arm had
  byte-identical answer lists. Requested/applied OCR pixel bounds were
  `589824/589824`; base cells were `262144/262144`. No labels entered remote
  requests, and no OOM, NaN, empty answer, or runtime-attestation failure
  occurred.
- Cached-condition wall time was about 11 seconds for three rows. Peak PyTorch
  allocation was 10.1 GB for 4B, 18.6 GB for base 8B, and 19.9 GB for the
  high-resolution OCR row. Per-condition telemetry reached 100% utilization;
  the live dstack sample reached 94%.
- Speech and OCR answers matched the references in every base cell; high
  resolution did not change the already-correct OCR answer. All temporal cells
  missed the single reference wording. Inspection showed the relevant selfie,
  approach, confrontation, and timestamped offensive speech were present in
  the supplied evidence, while several generated confrontation answers remain
  literally before the car entry. Treat this row as both an answerer failure
  and an annotation/temporal-specificity diagnostic, not proof of a locator
  failure.
- The exact speech canary was initially classified as `general`; commit
  `735df43` fixes noun-phrase speech questions before the full run. Operational
  canary gate: **pass**. Proceed to the complete 475-row factorial.

- [ ] **Step 6: Commit only the updated canonical plan record**

Do not commit private benchmark data. Add the run identity, checksums, measured
timing, and gate result to this plan, then commit that plan update.

### Task 5: Full 475-row factorial, semantic scoring, and decision

**Files:**
- Create (generated/private): `data/vqa/benchmarks/vqa-factorial-475-20260809-f1/`
- Create: `scripts/score_vqa_factorial.py`
- Create: `tests/test_score_vqa_factorial.py`
- Create: `docs/experiments/artifacts/vqa-factorial-475-20260809-f1.json`
- Modify: `docs/project-status/research-routing.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify: `docs/superpowers/plans/2026-08-09-vqa-factorial-pilot.md`

**Interfaces:**
- Consumes passing canary, Task 1 scorer, and Task 3 full profile.
- Produces the Phase 1 winner/stop decision and throughput projection for the
  oracle/multi-pass implementation plan.

- [ ] **Step 1: Preview full topology and cost**

Resolve the evidence config and each of four condition configs without
provisioning. Require capacity for four simultaneous condition GPUs and record
estimated evidence time, per-condition time, wall time, total GPU-hours, and
maximum price. If the measured canary contradicts the projection, revise the
profiles before submission and record why.

- [ ] **Step 2: Submit once and monitor to terminal state**

Submit the evidence job once and freeze its remote checksum marker. Only then
submit the four condition jobs detached, each exactly once. Verify every
condition uses the same evidence marker and complete canonical 475-ID set, and
that output prefixes are distinct.

- [ ] **Step 3: Download, checksum, and merge**

Refuse missing/duplicate row IDs, mismatched condition identities, or evidence
marker drift. Preserve each condition bundle and produce one deterministic
factorial index plus checksum file; do not merge different conditions into one
prediction list.

- [ ] **Step 4: Score all four cells**

Run one explicit four-cell scorer in the pinned Podman environment while
preserving Task 1's two-condition scorer unchanged. Validate and align all four
canonical 475-ID prediction sets before joining labels. Reuse the frozen
`4b-current` numeric rows only after proving its artifact/model/panel/scorer
binding; score unique new prediction/reference pairs once. Emit per-row and
aggregate METEOR/BERTScore/STS/composite, paired bootstrap effects versus
`4b-current`, row-paired model and prompt main effects, their interaction,
answer length, strata, completion, wall time, GPU-hours, utilization, and VRAM.
NDCG is marked degenerate because these are one-answer cells. Keep raw text and
metric caches private; the tracked artifact contains IDs, strata, hashes, and
numeric metrics only.

- [ ] **Step 5: Apply the predeclared decision rule**

Advance a cell only at `ΔC >= 0.015`, positive bootstrap lower bound, positive
METEOR/BERTScore deltas, STS no worse than `-0.005`, and 475/475 completion.
Report prompt/model main effects and interaction; call a mechanism dominant
only under the design-spec threshold.

- [ ] **Step 6: Update the primary QMD record and render**

Record reason, condition/panel, observed result, validity, limitations, gate,
decision, evidence link, and next experiment under Routing. Link from the hub
without duplicating the claim. Render from QMD; do not edit generated Markdown
or HTML directly.

- [ ] **Step 7: Verify report and full test suite**

```bash
uv run --group dev pytest -q
uv run --group dev ruff check .
quarto render docs/project-status
```

Verify links, privacy boundary, and that no raw labels/media/logs enter tracked
reports.

- [ ] **Step 8: Commit Task 5**

Stage only source, tests, sanitized artifacts, QMD, and generated report files
that belong to this result. Never stage `data/`.

## Follow-on boundary

After Task 5 freezes the winning answerer, write a separate implementation plan
for Phases 2–4 of the design spec: the 120-row `2 × 3` silver-oracle evidence
factorial, routed spatial-detail test, and ten deterministic evidence passes
with reference-blind election. This separation prevents implementing an oracle
and multi-pass pipeline against an answerer that the full factorial may reject.
