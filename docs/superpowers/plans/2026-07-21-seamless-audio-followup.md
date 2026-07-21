# SeamlessM4T Audio Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether SeamlessM4T-v2 produces more faithful English speech evidence and improves the frozen Qwen 3.5 4B, 12-frame answer.

**Architecture:** Add one isolated JSON-in/JSON-out Seamless speech-to-text runner, execute it on the three frozen WAV files in the existing ROCm container, then perform one label-free downstream Qwen comparison on the speech clip. Keep all generated evidence append-only and ignored; commit only source, tests, plan status, and the final report.

**Tech Stack:** Python 3.12, pytest, Hugging Face Transformers, SeamlessM4T-v2, SoundFile, PyTorch ROCm, rootless Podman, existing VQA runner/evaluator.

## Global Constraints

- Do not create, construct, or execute a 100-video run.
- Use model `facebook/seamless-m4t-v2-large` at revision `5f8cc790b19fc3f67a61c105133b20b34e3dcb76`.
- Use dedicated class `SeamlessM4Tv2ForSpeechToText`, target language `eng`, FP16, and `cuda:0`; do not silently fall back to CPU, another precision, model, or revision.
- Use runtime image `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5` with networking disabled during inference.
- Treat the CC-BY-NC-4.0 checkpoint as a local research diagnostic, not a production dependency or default baseline.
- Reuse only the three frozen 16 kHz mono WAV files and the existing 12 frozen Qwen frames.
- Never serialize an official reference, `label`, `pale orange`, `expected contents of the bag`, or any answer-derived phrase into an inference request.
- Store requests, transcripts, attempts, stdout, stderr, and predictions only beneath ignored `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/`.
- Append terminal attempts; never overwrite or delete a prior attempt.
- Join references only after inference during scoring.

---

### Task 1: Add the dedicated Seamless speech-to-text runner

**Files:**
- Create: `src/trec_video_2026/vqa_seamless.py`
- Create: `tests/test_vqa_seamless.py`

**Interfaces:**
- Consumes: one JSON request on stdin with `run_id`, `condition_id`, `video_id`, and `audio_path`.
- Produces: one JSON result with the same three identity fields, source audio SHA-256, translated `text`, mechanical `quality`, model/runtime/device/precision provenance, elapsed seconds, peak bytes, and structured error fields.
- Exposes: `prepare_audio_inputs(processor, audio, *, device, dtype)`, `decode_translation(processor, generated_tokens)`, and `translate_audio(model, processor, audio, *, device, dtype, inference_mode)` for focused unit testing.

- [ ] **Step 1: Write failing runner tests**

Create tests with small fakes that require:

```python
def test_prepare_audio_inputs_uses_frozen_sample_rate_and_device() -> None:
    processor = FakeProcessor()
    prepared = prepare_audio_inputs(
        processor, [0.1, -0.1], device="cuda:0", dtype="float16"
    )
    assert processor.kwargs == {
        "audio": [0.1, -0.1],
        "sampling_rate": 16_000,
        "return_tensors": "pt",
    }
    assert prepared.to_kwargs == {"device": "cuda:0", "dtype": "float16"}


def test_decode_translation_decodes_first_sequence() -> None:
    processor = FakeProcessor()
    assert decode_translation(processor, [[11, 12, 3]]) == "Where is the bag?"
    assert processor.decode_args == ([11, 12, 3],)
    assert processor.decode_kwargs == {"skip_special_tokens": True}


def test_translate_audio_targets_english_without_sampling() -> None:
    result = translate_audio(
        FakeModel(), FakeProcessor(), [0.1], device="cuda:0",
        dtype="float16", inference_mode=FakeInferenceMode,
    )
    assert result == "Where is the bag?"
    assert FakeModel.generate_kwargs == {"tgt_lang": "eng"}
```

Also test that an empty decoded string raises `ValueError` and that
`request_identity` copies only `run_id`, `condition_id`, and `video_id`.

- [ ] **Step 2: Verify the tests fail for the missing module**

Run:

```bash
uv run pytest -q tests/test_vqa_seamless.py
```

Expected: collection fails because `trec_video_2026.vqa_seamless` does not exist.

- [ ] **Step 3: Implement the minimal runner**

Implement the helpers exactly around the dedicated model API:

```python
def prepare_audio_inputs(processor, audio, *, device, dtype):
    inputs = processor(audio=audio, sampling_rate=16_000, return_tensors="pt")
    return inputs.to(device=device, dtype=dtype)


def decode_translation(processor, generated_tokens):
    text = processor.decode(generated_tokens[0], skip_special_tokens=True).strip()
    if not text:
        raise ValueError("SeamlessM4T returned an empty translation")
    return text


def translate_audio(model, processor, audio, *, device, dtype, inference_mode):
    inputs = prepare_audio_inputs(
        processor, audio, device=device, dtype=dtype
    )
    with inference_mode():
        generated = model.generate(**inputs, tgt_lang="eng")
    return decode_translation(processor, generated)
```

The CLI must use `vqa_whisper.load_wav_audio` and
`vqa_whisper.transcript_quality`, fingerprint before model loading, reset and
record ROCm peak allocation, pin both model and processor revision, embed the
request identity on success and failure, cap error text at 4,000 characters,
and print exactly one JSON object to stdout.

- [ ] **Step 4: Verify focused and full tests**

Run:

```bash
uv run pytest -q tests/test_vqa_seamless.py tests/test_vqa_whisper.py
uv run pytest -q
```

Expected: all focused tests and the full suite pass.

- [ ] **Step 5: Commit the runner**

```bash
git add src/trec_video_2026/vqa_seamless.py tests/test_vqa_seamless.py
git commit -m "feat: add Seamless speech translation runner"
```

### Task 2: Execute and audit the three Seamless translations

**Files:**
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/requests/asr-*.json`
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/raw-results/asr-*.json`
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/logs/asr-*.stderr.log`
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/attempts.jsonl`
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/execution.json`
- Modify: `docs/superpowers/plans/2026-07-21-seamless-audio-followup.md`

**Interfaces:**
- Consumes: the three audio paths already indexed by `audio-spatial-2026-07-21` and the Task 1 runner.
- Produces: three exact-capture terminal result records and semantic review statuses.

- [ ] **Step 1: Verify local resources and runner compatibility**

Run:

```bash
readlink -f data
podman image exists localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5
df -h /home/npatta01/.cache/huggingface
podman run --rm --network=none \
  localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5 \
  python -c 'from transformers import SeamlessM4Tv2ForSpeechToText; print("ok")'
```

Expected: `data` resolves to the canonical shared directory, the image exists,
more than 12 GB remains free, and the import prints `ok`.

- [ ] **Step 2: Download only pinned Transformers artifacts**

Run:

```bash
hf download facebook/seamless-m4t-v2-large \
  config.json generation_config.json preprocessor_config.json \
  tokenizer_config.json special_tokens_map.json added_tokens.json \
  sentencepiece.bpe.model tokenizer.model model.safetensors.index.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  --revision 5f8cc790b19fc3f67a61c105133b20b34e3dcb76
```

Expected: the pinned snapshot completes without `.pt` or duplicate framework
weights. Record exact file count, byte total, revision, and hashes in
`execution.json`.

- [ ] **Step 3: Create and leakage-audit the three requests**

Use condition `seamless-m4t-v2-large-s2tt-eng`, one request per frozen video,
and these exact video/audio pairs:

| Video ID | Absolute audio path |
|---|---|
| `BDKkVYmWqdM` | `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/evidence/cf3ecd7cdead7540183faa532807c03114db2353f11b305c5f7087e01502f036/audio.wav` |
| `O60MtMuSpf8` | `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/evidence/981707cffe59f296aa71743f271ec73d93863e2ed6ee1e4caa2ebc780e717c49/audio.wav` |
| `Fk_j6re_Uiw` | `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/evidence/1092a0dba1faebacc7fe02bab5778771813b866ed0a8bd6edffca7337c7d7f4e/audio.wav` |

Every request contains exactly the four keys `run_id`, `condition_id`,
`video_id`, and `audio_path`.

Before inference, reject any request containing the forbidden strings from the
Global Constraints and record its SHA-256.

- [ ] **Step 4: Run each request with byte-exact capture**

For each request, use the digest-pinned container with `/dev/kfd`, `/dev/dri`,
`--group-add=keep-groups`, `--network=none`, offline Hugging Face variables,
the repo mounted read-only, and shared data mounted at the same absolute path.
Invoke:

```bash
python -m trec_video_2026.vqa_seamless \
  --model-id facebook/seamless-m4t-v2-large \
  --revision 5f8cc790b19fc3f67a61c105133b20b34e3dcb76
```

Redirect stdout and stderr directly to the per-video result/log files. Validate
that stdout is one JSON object, identity equals the request, audio hash matches,
and model revision/device/precision/runtime image are exact. Append a terminal
attempt with request, stdout, and stderr hashes even if inference fails.

- [ ] **Step 5: Review transcript meaning and record plan status**

Apply the mechanical quality gate, then compare meaning with audible/native
transcript evidence without opening the official reference. Record one of
`accepted-semantically-consistent`, `unverified`, or
`rejected-semantic-conflict` separately from runner status. Add observed text,
latency, peak allocation, and hashes to this plan.

- [ ] **Step 6: Commit only the completed plan status**

```bash
git add docs/superpowers/plans/2026-07-21-seamless-audio-followup.md
git commit -m "docs: record Seamless translation diagnostic"
```

### Task 3: Run downstream Qwen, score, report, and stop

**Files:**
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/requests/qwen-speech.json`
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/raw-results/qwen-speech.json`
- Create ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/logs/qwen-speech.stderr.log`
- Modify ignored: `data/vqa/benchmarks/followups/seamless-audio-2026-07-21/attempts.jsonl`
- Create: `reports/trecvid_vqa_2026/seamless_audio_followup_results_2026-07-21.md`
- Modify: `docs/superpowers/plans/2026-07-21-seamless-audio-followup.md`

**Interfaces:**
- Consumes: the Seamless English text for `Fk_j6re_Uiw`, the frozen 12-frame request, the pinned Qwen runner, and references only at scoring time.
- Produces: one exact-capture Qwen terminal result plus a controlled three-condition metric comparison and recommendation.

- [ ] **Step 1: Construct and audit the label-free Qwen request**

Copy the frozen speech request fields and replace only the condition ID and
transcript:

```text
condition_id = qwen35-4b-12f-seamless-m4t-v2-large
transcript = exact Seamless English text
```

Retain the same 12 frame paths, timestamps, prompt, model ID, and revision.
Reject the request if it contains any forbidden string or a reference/label key.

- [ ] **Step 2: Run the pinned Qwen request with exact capture**

Invoke the existing `trec_video_2026.vqa_transformers_runner` with:

```text
model: Qwen/Qwen3.5-4B
revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
runtime: localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5
max new tokens: 32
```

Redirect stdout/stderr directly, validate embedded identity and provenance, and
append a terminal attempt with request/result/log hashes.

- [ ] **Step 3: Join the reference and score the controlled comparison**

Using the existing local evaluator, compare the speech answer under:

```text
12 frames only: End of the trend
12 frames + Whisper large-v3: Where the money is
12 frames + SeamlessM4T-v2: exact new Qwen answer
reference: joined only now from the frozen labeled set
```

Compute normalized exact match, token F1, and METEOR. State BERTScore as
unavailable if its pinned encoder is still absent; do not download it. Do not
call STS or interpret single-answer NDCG.

- [ ] **Step 4: Write the evidence-backed report**

Report the three Seamless transcripts and semantic statuses, the downstream
prediction and metrics, ASR/Qwen latency and memory, pinned provenance, the
CC-BY-NC-4.0 limitation, exact-capture status, and whether Seamless advances.
Explicitly state that this is a one-example causal diagnostic and the 100-video
run was not started.

- [ ] **Step 5: Run fresh verification**

Run:

```bash
uv run pytest -q
git diff --check
uv run vqa-gate preflight --config configs/vqa/three_video_gate.json --dataset-root data
git status --short
```

Also recompute every indexed request/result/log hash, verify append-only attempt
identities, and rerun the inference-request leakage scan.

- [ ] **Step 6: Commit report and completed plan**

```bash
git add reports/trecvid_vqa_2026/seamless_audio_followup_results_2026-07-21.md \
  docs/superpowers/plans/2026-07-21-seamless-audio-followup.md
git commit -m "docs: report Seamless audio diagnostic"
```

- [ ] **Step 7: Handoff and stop**

Give the user the transcript, downstream answer, metric delta, model/runtime
cost, review verdict, report link, and next decision. Do not create or execute a
100-video manifest.
