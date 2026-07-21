# SeamlessM4T Audio Follow-up Design

Date: 2026-07-21

## Goal

Determine whether a non-Whisper speech-translation family produces more faithful
English evidence for the frozen speech-heavy video and improves the existing
Qwen 3.5 4B, 12-frame answer.

## Selected approach

Use `facebook/seamless-m4t-v2-large` through Transformers' dedicated
`SeamlessM4Tv2ForSpeechToText` interface with target language `eng`. This is the
cleanest different-family comparison because it translates speech directly to
English, supports Hindi and Urdu speech input, and does not introduce a second
text-translation model.

The alternatives are intentionally deferred:

1. An IndicConformer plus IndicTrans2 cascade is smaller and language-specific,
   but adds source-language selection and two separately fallible stages.
2. A native-audio VLM tests a different question—joint audio/video reasoning—
   and would not isolate transcript fidelity.

The selected 2.3B-parameter checkpoint is pinned to revision
`5f8cc790b19fc3f67a61c105133b20b34e3dcb76`. Its roughly 9.2 GB FP32
safetensors and CC-BY-NC-4.0 license make this a local research diagnostic, not
a production dependency or default baseline.

## Architecture

Add a focused JSON-in/JSON-out runner, `trec_video_2026.vqa_seamless`, rather
than adding Seamless-specific branches to the Whisper runner. The new runner:

- reads one request containing `run_id`, `condition_id`, `video_id`, and an
  absolute 16 kHz mono `audio_path`;
- fingerprints the source audio before inference;
- loads `AutoProcessor` and `SeamlessM4Tv2ForSpeechToText` at the pinned
  revision in FP16 on `cuda:0` through the existing digest-pinned ROCm image;
- preprocesses audio without supplying a source language and generates text
  deterministically with `tgt_lang="eng"`;
- decodes the generated token IDs and applies the existing mechanical transcript
  quality checks;
- emits structured success or failure JSON containing request identity, model
  and runtime provenance, elapsed time, peak allocation, source hash, text, and
  quality status.

Unit tests cover audio preprocessing, device/dtype transfer, token unwrapping,
decoding, and request-identity preservation before model execution is attempted.

## Diagnostic flow

Run the new translator on the same three cached WAV files used by the previous
Whisper comparison. Retain each request before execution and redirect stdout and
stderr directly to immutable ignored artifacts. Append every terminal attempt
to a new run index beneath:

`data/vqa/benchmarks/followups/seamless-audio-2026-07-21/`

Mechanically accepted English is not automatically considered faithful. Review
the outputs against audible/native-transcript evidence without using the official
answer. Record semantic status separately from the automatic quality gate.

Construct one downstream Qwen request for `Fk_j6re_Uiw` using the frozen 12
frames and the Seamless English text. Run the same pinned Qwen 3.5 4B condition
even if semantic review is disputed, but label it exploratory in that case. Do
not place a reference answer, label, or answer-derived phrase in the request.

## Scoring and decision

After inference, join the official reference only in the local evaluator and
compare:

- frozen 12-frame Qwen prediction;
- frozen 12-frame plus Whisper large-v3 prediction;
- frozen 12-frame plus SeamlessM4T-v2 prediction.

Report normalized exact match, token F1, and locally available METEOR, together
with transcript text/status, runtime, memory, and provenance limitations. Do not
download an encoder solely for BERTScore, call external STS, or interpret
single-answer NDCG.

Advance SeamlessM4T only if the speech translation is semantically credible and
the downstream answer improves. Stop after this diagnostic; do not create or
execute a 100-video run.

## Failure handling

- A model download or ROCm incompatibility is a recorded terminal result, not a
  reason to silently switch device, precision, revision, or model family.
- A failed one-video request does not erase successful attempts.
- An English-looking but meaning-conflicted translation remains available for
  audit and is marked semantically rejected or disputed.
- Dataset media and generated artifacts remain ignored and are never staged.
