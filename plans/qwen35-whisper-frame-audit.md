# Matched Qwen3.5-4B Whisper and frame-evidence audit

## Objective

Run the same 475 locally available VQA 2025 training rows through Qwen3.5-4B
with 12 selected visual frames in two label-free conditions: frames-only and
the same frames plus accepted Whisper large-v3 transcript evidence. Preserve
the selected frame timestamps, per-frame hashes, and contact sheets as a
private artifact. After predictions freeze, score locally and inspect every
wrong row using the captured visual evidence and source audio/transcript.

## Scope and integrity boundary

- Panel: 475 rows with both a VQA 2025 training annotation and local video.
- Excluded: six annotation records without local video, VQA testing, VQA 2026,
  and the separate VTT caption/search collection.
- Remote input: the existing private, hash-verified 475-video bundle.
- Remote inference inputs: video IDs, questions, selected frames/timestamps,
  and label-free Whisper output only. No answers, annotations, or references.
- Conditions: `qwen35-4b-12f-frames-only` and
  `qwen35-4b-12f-whisper-large-v3`.
- Whisper source: `openai/whisper-large-v3` at revision
  `06f233fe06e710322aca913c1bc4249a0d71fce1`; accepted transcript text is
  timestamped translated text under the existing deterministic quality policy.

## Artifact contract

- `frame-evidence.json`: one record per row with media/audio hashes, duration,
  requested and selected frame timestamps, individual frame hashes, and the
  contact-sheet hash.
- `frame-contact-sheets/*.jpg`: twelve selected frames per row with timestamp
  labels; no answers or references.
- `predictions.frames-only.json` and `predictions.frames-whisper.json`.
- `whisper-transcripts.json`: label-free ASR output, timestamps, quality status,
  and provenance.
- Private `wrong-row-audit.json`/`.html`: all label-bearing wrong-row joins
  with frame/audio evidence links; private `manual-review.json`: the 48-row
  visual/transcript review subset and its findings.
- `requests.*.json`, summaries, runtime attestations, and `SHA256SUMS`.

## Decision and audit

Score locally only after both prediction files are complete and hash-verified.
For each wrong prediction, classify whether the question-answer evidence was
present in the selected frames, present in the audio/transcript, present in
both, or absent/uncertain. Do not call a row a model error when the required
event was not supplied by the selected evidence.

## Status

- [x] Implement and validate the matched evidence/Whisper runner.
- [x] Complete a CUDA Whisper/Qwen canary.
- [x] Run and verify the 475-row matched conditions.
- [x] Score locally and build the wrong-row frame/audio audit.
- [x] Record the result and next decision.

## Completed run

- Run: `q35-12f-wa-20260803-c2`; dstack exited `0` and the remote checksum
  manifest verified all 500 output files locally.
- Both conditions completed 475/475 rows. Frames-only wall time was 709.7 s;
  frames plus Whisper was 751.8 s. Whisper accepted 424/475 transcripts in a
  single CUDA translation pass; 51 were rejected by the deterministic quality
  gate and remained null evidence.
- Paired deltas on the full panel: normalized exact match `+0.0484` (27 wins,
  4 losses), token F1 `+0.06095`, and lexical METEOR `+0.06228`. Transitions
  were 40 both-correct, 404 both-wrong, 27 frames-only-only-wrong, and 4
  Whisper-only-wrong.
- Among the 424 quality-accepted transcripts, 307 answer strings changed;
  token F1 increased on 122 rows and decreased on 58. This is why the next
  test must measure transcript/question relevance rather than use ASR quality
  acceptance as the route gate.
- The private audit indexes all 435 rows with an exact-match error and stores
  the question, reference, both answers, selected PTS, frame hashes/contact
  sheet, source-audio hash/duration, and timestamped Whisper evidence. A
  stratified manual review covers all 31 transition rows plus 17 both-wrong
  rows: 41 had answer-relevant evidence present and 7 were uncertain from the
  uniform 12-frame sample. Raw audio playback was unavailable, so the audio
  review is transcript/timestamp based.

## Decision

Whisper is useful additional evidence at this checkpoint, but ASR quality
acceptance is not question relevance. The next experiment is a matched
semantic-relevance route with a shuffled-transcript negative control, reusing
the frozen 475-row evidence rather than moving immediately to a larger model
or broader frame budget. The route must fall back to frames-only when the
question/transcript relevance gate fails and must reject the clear unrelated-
audio failure `ITL429RMscI`.

Predeclared promotion criteria:

- all 475 rows receive a deterministic route or frames-only fallback;
- paired F1 and lexical METEOR bootstrap lower bounds exceed zero versus
  frames-only, with no exact-match decrease;
- retain at least 80% of this run's global gain: F1 delta at least `+0.0488`
  and METEOR delta at least `+0.0498`;
- reduce accepted-transcript F1-decrease rows from the current `58` to at
  most `29`;
- aligned transcripts beat the deterministic shuffled-transcript negative
  control by at least `+0.03` F1 and `+0.03` METEOR, with paired lower bounds
  above zero.
