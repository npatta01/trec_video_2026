# Evidence-Routing Benchmark Implementation Plan

> **Historical-status note (2026-07-25):** This plan was restored because
> closed issue #4 links to it. Frozen-30, automatic-transcript, model
> compatibility, semantic correction, Qwen3-Omni, and temporal diagnostics
> ran. The tested directed-12 ten-row and dense-burst-12 two-row diagnostics
> were not promoted as defaults; other selectors remain open. The strict
> human-oracle tier remains deferred in open issue #19. Current decisions live
> in [Project Status](../../project-status/index.md). The checklist below is
> preserved as historical record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine which TRECVID VQA errors come from missing evidence, poor evidence extraction, or failure by the VLM to use available evidence, then benchmark the smallest useful set of alternative omni models.

**Architecture:** Freeze small, stratified diagnostic panels and compare matched conditions that differ in one evidence path at a time. Keep code and configuration in the repository, but write videos, transcripts, model outputs, and scored artifacts only beneath the ignored canonical `data/vqa/benchmarks/` tree.

**Tech Stack:** Python 3.12, pytest, PyTorch/Transformers, ROCm rootless Podman, dstack for large NVIDIA inference, the existing frozen TRECVID scorer, JSON/JSONL artifacts.

## Global Constraints

- The shared `data` path must resolve to `/home/npatta01/data/competitions/trec_video_2026/data`.
- Never copy, move, delete, stage, or commit dataset contents.
- Freeze questions, media hashes, frames, prompts, and labels before comparing conditions.
- Keep reference answers unavailable to inference and transcript creation.
- Use identical frames and decoding settings for the three oracle-transcript conditions.
- Treat the original panel as ten difficult rows plus two regression controls, not twelve universal failures.
- A three-video compatibility run must pass before any 10- or 30-video model run.
- Verify GPU access, precision, model revision, runtime image digest, and peak memory for every model run.
- Do not expose an inference service or publish artifacts.

---

### Task 1: Freeze a speech-dependent oracle panel

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/oracle_panel.py`
- Create: `src/trec_video_2026/vqa_experiments/__init__.py`
- Create: `tests/test_oracle_panel.py`
- Produce, ignored: `data/vqa/benchmarks/oracle-speech-10/manifest.json`

**Interfaces:**
- Consumes: annotation JSON files and local video paths.
- Produces: `build_oracle_panel(annotation_dir: Path, video_dir: Path, selected_ids: Sequence[str]) -> dict`.

- [ ] **Step 1: Write tests for panel isolation and integrity**

```python
def test_oracle_panel_contains_no_reference_answers(tmp_path):
    panel = build_oracle_panel(annotation_dir, video_dir, selected_ids)
    encoded = json.dumps(panel).lower()
    assert "correct_answer" not in encoded
    assert "incorrect_answers" not in encoded
    assert len(panel["examples"]) == 10


def test_oracle_panel_binds_media_hashes():
    panel = build_oracle_panel(annotation_dir, video_dir, selected_ids)
    assert all(len(row["media_sha256"]) == 64 for row in panel["examples"])
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_oracle_panel.py -v`
Expected: FAIL because `oracle_panel` does not exist.

- [ ] **Step 3: Implement the manifest builder**

The builder must include `video_id`, `question`, `media_path`, `media_sha256`,
`language_guess`, and the operational tag `speech_content`. It must reject
missing media, duplicate IDs, and annotations lacking questions. Select ten
questions whose answer depends on lexical speech, including `NN7G28sBOlY` and
`Gp4LVTn3YbY`; do not include prosody, music, or sound-effect questions.

- [ ] **Step 4: Generate and validate the ignored manifest**

Run:

```bash
uv run python -m trec_video_2026.vqa_experiments.oracle_panel \
  --annotations data/vqa/2025/training/annotations \
  --videos data/vqa/2025/training/videos_original \
  --output data/vqa/benchmarks/oracle-speech-10/manifest.json
```

Expected: ten unique examples, no answers, and ten valid SHA-256 hashes.

- [ ] **Step 5: Run focused and repository tests**

Run: `uv run pytest tests/test_oracle_panel.py -v && uv run pytest -q`
Expected: all tests pass.

### Task 2: Create label-blind manually verified transcripts

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/transcript_contract.py`
- Create: `tests/test_transcript_contract.py`
- Produce, ignored: `data/vqa/benchmarks/oracle-speech-10/oracle-transcripts/*.json`
- Produce, ignored: `data/vqa/benchmarks/oracle-speech-10/transcript-review.json`

**Interfaces:**
- Consumes: the frozen oracle manifest and reviewer-authored transcript JSON.
- Produces: `validate_oracle_transcript(record: dict, media_sha256: str) -> None`.

- [ ] **Step 1: Write contract tests**

```python
def test_oracle_transcript_requires_source_and_translation():
    record = valid_record | {"source_text": "", "english_translation": ""}
    with pytest.raises(ValueError, match="transcript text"):
        validate_oracle_transcript(record, MEDIA_SHA)


def test_oracle_transcript_is_bound_to_audio():
    with pytest.raises(ValueError, match="source_audio_sha256"):
        validate_oracle_transcript(valid_record, "0" * 64)
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_transcript_contract.py -v`
Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Implement the transcript contract**

Require `video_id`, `source_language`, `source_text`,
`english_translation`, timestamped segments, `source_audio_sha256`,
`transcriber`, and `reviewer`. Reject answer-only paraphrases, empty
translations, timestamps outside the clip, and mismatched audio hashes.

- [ ] **Step 4: Prepare transcripts without label leakage**

The transcriber and reviewer may receive the audio/video ID but not the
question, correct answer, incorrect answers, or existing model predictions.
Each transcript must preserve uncertainty such as “I don't know” rather than
normalizing it into a guessed factual answer.

- [ ] **Step 5: Validate all ten records**

Run:

```bash
uv run python -m trec_video_2026.vqa_experiments.transcript_contract \
  --manifest data/vqa/benchmarks/oracle-speech-10/manifest.json \
  --transcripts data/vqa/benchmarks/oracle-speech-10/oracle-transcripts \
  --output data/vqa/benchmarks/oracle-speech-10/transcript-review.json
```

Expected: ten accepted, independently reviewed transcript records.

### Task 3: Run the three-condition oracle experiment

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/build_conditions.py`
- Create: `tests/test_build_conditions.py`
- Produce, ignored: `data/vqa/benchmarks/oracle-speech-10/requests.json`
- Produce, ignored: `data/vqa/benchmarks/oracle-speech-10/predictions/*.jsonl`
- Produce, ignored: `data/vqa/benchmarks/oracle-speech-10/evaluation.json`

**Interfaces:**
- Consumes: panel manifest, fixed frame evidence, automatic transcripts, and oracle transcripts.
- Produces: `build_matched_conditions(manifest: dict, frames: dict, automatic: dict, oracle: dict) -> list[dict]`.

- [ ] **Step 1: Write matching tests**

```python
def test_conditions_differ_only_by_transcript():
    requests = build_matched_conditions(manifest, frames, automatic, oracle)
    grouped = group_by_video(requests)
    for rows in grouped.values():
        assert {row["condition"] for row in rows} == {
            "frames_only",
            "frames_automatic_transcript",
            "frames_oracle_transcript",
        }
        assert len({tuple(row["frame_sha256s"]) for row in rows}) == 1
        assert len({row["prompt_sha256"] for row in rows}) == 1
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest tests/test_build_conditions.py -v`
Expected: FAIL because the condition builder does not exist.

- [ ] **Step 3: Implement matched requests**

Use the same Qwen3.5-4B VLM, 12 frames, image resolution, question, prompt,
seed, and decoding parameters in all three conditions. Ask for structured
diagnostic output containing `answer` and `evidence_quote`; score only
`answer`. Use Whisper large-v3 for the automatic condition first.

- [ ] **Step 4: Run inference and freeze predictions before loading labels**

Expected output: 30 successful rows, one for each of three conditions across
ten videos, with no duplicated `(video_id, condition)` pairs.

- [ ] **Step 5: Score and adjudicate**

Use the frozen scorer plus blinded correct/partial/wrong adjudication.
Report paired transitions for each video rather than treating ten rows as an
estimate of leaderboard performance.

- [ ] **Step 6: Apply the ASR decision gate**

- If oracle transcripts do not improve any automatic-transcript failures,
  stop ASR model comparisons.
- If oracle transcripts fix at least one failure, compare Whisper large-v3
  with Qwen3-ASR only on the oracle-positive rows.
- If oracle transcripts contain the answer but the VLM still fails, improve
  grounding/prompting or the VLM; do not attribute the error to ASR.

### Task 4: Run non-Qwen omni compatibility probes

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/omni_contract.py`
- Create: `tests/test_omni_contract.py`
- Produce, ignored: `data/vqa/benchmarks/omni-compatibility-3/`

**Interfaces:**
- Consumes: one lexical-speech, one prosody, and one non-speech-sound example.
- Produces: normalized prediction records with model/runtime attestations.

- [ ] **Step 1: Write modality and attestation tests**

Require every request to bind both frame hashes and an audio hash. Require
every result to record model ID/revision, runtime image digest, precision,
device, elapsed time, and peak memory.

- [ ] **Step 2: Test Phi-4-multimodal-instruct locally**

Supply the same sampled frames and native audio in one request. Treat it as a
joint image/audio model, not as proof of native temporal video processing.
Attempt ROCm BF16 in rootless Podman and document any CUDA-only blocker.

- [ ] **Step 3: Test Gemma 3n E4B-it locally**

Verify that the actual runtime accepts combined visual and audio evidence in
the same inference request. Do not count separate successful image-only and
audio-only calls as compatibility.

- [ ] **Step 4: Apply the promotion gate**

Promote a model to the 30-row stratified panel only if all three rows execute,
both modalities are demonstrably consumed, output parsing succeeds, and no
control row regresses due to an input-contract error.

### Task 5: Probe Qwen3-Omni on dstack

**Files:**
- Create: `dstack/qwen3-omni-compatibility.dstack.yml`
- Create: `tests/test_qwen3_omni_dstack_config.py`
- Produce, ignored: `data/vqa/benchmarks/qwen3-omni-compatibility-3/`

**Interfaces:**
- Consumes: the same three frozen compatibility requests as Task 4.
- Produces: normalized predictions compatible with the omni evaluation contract.

- [ ] **Step 1: Test the dstack configuration statically**

Assert that the model revision and container image use immutable identifiers,
the job does not expose ports, inputs are read-only, and outputs are copied
only to the benchmark artifact directory.

- [ ] **Step 2: Launch on a GPU with adequate headroom**

Use Qwen3-Omni-30B-A3B-Thinking on a GPU with more than 80 GB memory for
30-second clips, or clip the compatibility inputs to a documented shorter
duration. Record exact dstack offer, GPU type, and cost.

- [ ] **Step 3: Validate and score the three rows**

Require three completed predictions and full runtime attestations. Compare
against MiniCPM-o 4.5 audio-on and the best non-Qwen compatibility result.

- [ ] **Step 4: Apply the promotion gate**

Run 30 examples only if Qwen3-Omni materially improves at least one
speech/prosody/sound row without a visual-control regression.

### Task 6: Test question-directed temporal sampling

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/temporal_sampling.py`
- Create: `tests/test_temporal_sampling.py`
- Produce, ignored: `data/vqa/benchmarks/temporal-diagnostic-10/`

**Interfaces:**
- Consumes: temporal question, clip duration, and optional anchor timestamp.
- Produces: `select_temporal_frames(question: str, duration: float, anchor: float | None, count: int) -> list[float]`.

- [ ] **Step 1: Write boundary and direction tests**

```python
def test_after_question_samples_after_anchor():
    times = select_temporal_frames("What happens after the shove?", 30, 18, 12)
    assert sum(t > 18 for t in times) >= 8


def test_end_question_densifies_last_quarter():
    times = select_temporal_frames("What happens at the end?", 40, None, 12)
    assert sum(t >= 30 for t in times) >= 8
```

- [ ] **Step 2: Implement deterministic temporal sampling**

Support `before`, `after`, `beginning`, and `end` profiles. Preserve two to
four global context frames and allocate the remainder to the requested
temporal neighborhood. Clamp timestamps to the clip duration and deduplicate
near-identical frames.

- [ ] **Step 3: Compare matched visual conditions**

On ten temporal questions, compare 12 uniform frames, 24 uniform frames, and
12 question-directed frames using the same VLM and decoding parameters.

- [ ] **Step 4: Apply the promotion gate**

Adopt question-directed sampling only if it beats 12 uniform frames and is
competitive with 24 frames while using fewer visual tokens.

### Task 7: Defer narrow specialists until prevalence is measured

**Files:**
- Modify: `reports/trecvid_vqa_2026/taxonomy_prior_art_and_open_omni_models_2026-07-24.md`
- Produce, ignored: `data/vqa/benchmarks/routing-prevalence.json`

**Interfaces:**
- Consumes: all 481 training questions plus manual review of uncertain cases.
- Produces: estimated prevalence for exact-music, small-spatial-detail, and other routing tags.

- [ ] **Step 1: Estimate specialist prevalence**

Manually validate lexical candidate sets rather than treating keyword counts
as ground truth.

- [ ] **Step 2: Decide whether to build specialist branches**

Add audio fingerprinting only if exact music-identification questions are
frequent enough to affect aggregate score. Add crop/tiling only after the
small-detail subset demonstrates a repeatable gain. Keep these branches out of
the main baseline until then.

- [ ] **Step 3: Publish a private local decision report**

Summarize paired scores, correct/partial/wrong adjudication, runtime, peak
memory, and cost. Derive sanitized HTML under
`/home/npatta01/codex-rendered/plans/` only if requested, and verify the
existing Tailscale Serve mapping remains private.

## Execution order

1. Tasks 1–3: oracle transcript diagnostic.
2. Task 4: Phi-4 and Gemma 3n three-video compatibility.
3. Task 5: Qwen3-Omni three-video dstack compatibility.
4. Task 6: temporal sampling, which can run independently of ASR outcomes.
5. Task 7: narrow specialists only after prevalence and expected value are known.

## Self-review

- Spec coverage: oracle isolation, native omni probes, temporal routing,
  sound/music deferral, spatial-detail deferral, controls, and provenance are
  covered.
- Placeholder scan: no implementation placeholders remain.
- Interface consistency: all runtime artifacts stay under ignored `data/`;
  tracked modules expose explicit builders or validators used by their tests.
