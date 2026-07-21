# Audio and Spatial Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether full Whisper large-v3 translation or question-guided high-resolution crops recover the two failed evidence types in the frozen three-video gate.

**Architecture:** Preserve the original gate and create a separate append-only follow-up run. Add optional per-image descriptions to the existing runner protocol so crops are identified explicitly, then execute one pinned ASR condition and two targeted Qwen 3.5 4B requests using the existing ROCm container boundary.

**Tech Stack:** Python 3.12, pytest, Hugging Face Transformers, Whisper, FFmpeg, rootless Podman, PyTorch ROCm, existing VQA runner/evaluator.

## Global Constraints

- Do not start or construct a 100-video run.
- Reuse the three frozen examples, cached audio, and 12 bin-midpoint frames.
- Never serialize a reference answer or label-derived color word into a runner request, crop description, filename, or prompt.
- Store generated media, transcripts, requests, attempts, and predictions only beneath ignored `data/vqa/benchmarks/` paths.
- Pin model revision `06f233fe06e710322aca913c1bc4249a0d71fce1` for `openai/whisper-large-v3` and retain the existing digest-pinned ROCm runtime.
- Treat manual question-guided crops as diagnostics, not a benchmark-ready condition.

---

### Task 1: Add explicit detail-frame descriptions to the runner protocol

**Files:**
- Modify: `src/trec_video_2026/vqa_models.py`
- Modify: `src/trec_video_2026/vqa_gate.py`
- Modify: `src/trec_video_2026/vqa_transformers_runner.py`
- Test: `tests/test_vqa_models.py`
- Test: `tests/test_vqa_gate.py`
- Test: `tests/test_vqa_transformers_runner.py`

**Interfaces:**
- Consumes: existing ordered `frame_paths` and `frame_timestamps`.
- Produces: optional `frame_descriptions: tuple[str, ...] | None` on `RunnerRequest` and `build_controlled_messages`; when supplied, each non-empty description replaces the default `Frame at ... seconds:` label for the corresponding image.

- [x] **Step 1: Write failing message and serialization tests**

```python
def test_controlled_messages_label_detail_crop(tmp_path: Path) -> None:
    frame = tmp_path / "crop.png"
    frame.write_bytes(b"png")
    messages = build_controlled_messages(
        question="What color is the image?",
        frame_paths=(frame,),
        frame_timestamps=(0.61,),
        frame_descriptions=("Detail crop from frame at 0.610 seconds:",),
        transcript=None,
    )
    assert messages[0]["content"][0]["text"] == (
        "Detail crop from frame at 0.610 seconds:"
    )


def test_runner_request_serializes_frame_descriptions(tmp_path: Path) -> None:
    payload = dataclasses.replace(
        request(tmp_path), frame_descriptions=("Full frame:",)
    ).to_dict()
    assert payload["frame_descriptions"] == ["Full frame:"]
    assert "reference" not in json.dumps(payload).lower()
```

- [x] **Step 2: Run the tests and observe the intended failures**

Run:

```bash
uv run pytest -q \
  tests/test_vqa_models.py::test_controlled_messages_label_detail_crop \
  tests/test_vqa_gate.py::test_runner_request_serializes_frame_descriptions
```

Expected: failures because neither interface accepts `frame_descriptions`.

- [x] **Step 3: Implement the minimal optional field**

Add `frame_descriptions` to `RunnerRequest`, validate that its length equals the
frame count, serialize it, thread it through `generate_answer`, and sort
`(timestamp, path, description)` together. Preserve the existing default text
when the field is `None`.

```python
descriptions = frame_descriptions or tuple(
    f"Frame at {timestamp:.3f} seconds:" for timestamp in frame_timestamps
)
if len(descriptions) != len(frame_paths):
    raise ValueError("Frame descriptions and paths must have equal length")
ordered = sorted(zip(frame_timestamps, frame_paths, descriptions, strict=True))
```

- [x] **Step 4: Verify targeted and full tests**

Run:

```bash
uv run pytest -q tests/test_vqa_models.py tests/test_vqa_gate.py \
  tests/test_vqa_transformers_runner.py
uv run pytest -q
```

Expected: all tests pass.

- [x] **Step 5: Commit the protocol change**

```bash
git add src/trec_video_2026/vqa_models.py \
  src/trec_video_2026/vqa_gate.py \
  src/trec_video_2026/vqa_transformers_runner.py \
  tests/test_vqa_models.py tests/test_vqa_gate.py \
  tests/test_vqa_transformers_runner.py
git commit -m "feat: label diagnostic detail frames"
```

### Task 2: Generate and audit full Whisper large-v3 translations

**Files:**
- Create ignored: `data/vqa/benchmarks/followups/audio-spatial-2026-07-21/transcripts/*.json`
- Create ignored: `data/vqa/benchmarks/followups/audio-spatial-2026-07-21/attempts.jsonl`
- Modify: `docs/superpowers/plans/2026-07-21-audio-spatial-followup.md`

**Interfaces:**
- Consumes: three existing evidence-cache `audio.wav` files and `vqa_whisper` JSON protocol.
- Produces: one immutable transcript result per video with source-audio SHA-256, pinned model revision, QC status, ROCm device, latency, and peak allocation.

- [x] **Step 1: Verify resources before the model download**

Run:

```bash
readlink -f data
podman image exists \
  localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5
rocminfo | rg -m1 'Name:.*gfx|Marketing Name'
df -h /home/npatta01/.cache/huggingface
```

Expected: canonical dataset link, image exit 0, Radeon agent present, and more
than 4 GiB free for the 3.09 GB FP16 checkpoint plus metadata.

- [x] **Step 2: Download only the pinned Transformers files**

```bash
hf download openai/whisper-large-v3 model.safetensors added_tokens.json \
  config.json generation_config.json merges.txt normalizer.json \
  preprocessor_config.json special_tokens_map.json tokenizer.json \
  tokenizer_config.json vocab.json \
  --revision 06f233fe06e710322aca913c1bc4249a0d71fce1
```

Expected: the pinned snapshot completes without downloading FP32 shards.

- [x] **Step 3: Run the three audio requests through the pinned ROCm container**

For each of `BDKkVYmWqdM`, `O60MtMuSpf8`, and `Fk_j6re_Uiw`, invoke
`trec_video_2026.vqa_whisper` with:

```text
--model-id openai/whisper-large-v3
--revision 06f233fe06e710322aca913c1bc4249a0d71fce1
```

Use `task="translate"` through the existing runner, pass the frozen evidence
`audio.wav`, and persist stdout only after it parses as one JSON object whose
`video_id` and `source_audio_sha256` match the request and file.

- [x] **Step 4: Audit transcript quality and record the decision**

Check the mechanical QC result and manually compare audio/text meaning without
consulting the official answer during acceptance. Append every pass or failure;
do not overwrite a prior attempt. Mark the plan with the exact accepted/rejected
statuses and hashes.

Observed: all three passed mechanical QC on ROCm. The bike transcript is coherent
English, the color clip is described as music/outro rather than speech, and the
speech clip is coherent English about Pakistan and money. The speech translation
is accepted as usable evidence but does not itself mention a bag. Source-audio
hashes match the frozen evidence cache.

- [ ] **Step 5: Commit only plan status or source changes**

```bash
git add docs/superpowers/plans/2026-07-21-audio-spatial-followup.md
git commit -m "docs: record full Whisper diagnostic"
```

Do not stage the ignored transcript or attempt artifacts.

### Task 3: Execute the two targeted Qwen diagnostics

**Files:**
- Create ignored: `data/vqa/benchmarks/followups/audio-spatial-2026-07-21/crops/*.png`
- Create ignored: `data/vqa/benchmarks/followups/audio-spatial-2026-07-21/crops/manifest.json`
- Modify ignored: `data/vqa/benchmarks/followups/audio-spatial-2026-07-21/attempts.jsonl`

**Interfaces:**
- Consumes: accepted full-Whisper transcript, original 12-frame evidence, optional `frame_descriptions`, existing Qwen 3.5 4B pinned condition.
- Produces: terminal results for `qwen35-4b-12f-whisper-large-v3` on the speech clip and `qwen35-4b-12f-question-guided-crops` on the color clip.

- [ ] **Step 1: Create and inspect the question-guided crops**

Use FFmpeg's `crop` filter on each color-clip sample visibly containing the
herpes image. Choose bounds from pixels and the question only. Record source
frame path, source timestamp, crop rectangle, source SHA-256, and crop SHA-256
in `manifest.json`. Render a contact sheet and visually verify that each crop
contains the requested image and no annotations derived from the reference.

- [ ] **Step 2: Construct label-free requests and audit before inference**

The speech request contains the same 12 frame paths/timestamps plus the accepted
full-Whisper transcript. The color request contains the same 12 originals plus
the crops, with descriptions such as:

```text
Detail crop from frame at 0.610 seconds:
```

Run a JSON-key/content audit that rejects `reference`, `label`, `pale orange`,
or any stored official answer in either request.

- [ ] **Step 3: Run both requests in one Qwen batch**

Use the pinned Qwen condition:

```text
model: Qwen/Qwen3.5-4B
revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
runtime: localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5
```

Keep the model loaded across the ordered batch. Validate each runner result,
join request identity, and append both attempts even if one fails.

- [ ] **Step 4: Verify terminal records**

Confirm two terminal identity pairs, no request/reference leakage, ROCm device,
pinned model/runtime revisions, matching evidence hashes, and non-overwritten
history.

### Task 4: Score, report, and stop

**Files:**
- Create: `reports/trecvid_vqa_2026/audio_spatial_followup_results_2026-07-21.md`
- Modify: `docs/superpowers/plans/2026-07-21-audio-spatial-followup.md`

**Interfaces:**
- Consumes: frozen baseline results and the two new terminal attempts.
- Produces: a concise causal comparison with predictions, normalized EM, token F1, NIST-local METEOR/BERTScore where available, latency, memory, transcript text/status, and explicit diagnostic caveats.

- [ ] **Step 1: Join labels only during scoring**

Use the existing local evaluator after inference. Show the frozen comparator and
new result side by side for the speech and color examples. Do not run external
STS or interpret single-answer NDCG.

- [ ] **Step 2: State the decision without overgeneralizing**

Report whether each hypothesis was supported, distinguish mechanical English
QC from semantic translation quality, label crops as manually question-guided,
and explicitly state that the 100-video run was not started.

- [ ] **Step 3: Run fresh verification**

```bash
uv run pytest -q
git diff --check
uv run vqa-gate preflight \
  --config configs/vqa/three_video_gate.json --dataset-root data
git status --short
```

Expected: full test suite passes; diff check and preflight exit 0; no dataset,
transcript, crop, request, prediction, raw log, or unrelated untracked file is
staged.

- [ ] **Step 4: Commit the report and completed plan**

```bash
git add reports/trecvid_vqa_2026/audio_spatial_followup_results_2026-07-21.md \
  docs/superpowers/plans/2026-07-21-audio-spatial-followup.md
git commit -m "docs: report audio and spatial diagnostics"
```

- [ ] **Step 5: Handoff and stop**

Give the user the two outcomes, verified evidence, report link, and remaining
unknowns. Do not create or execute any 100-video manifest.
