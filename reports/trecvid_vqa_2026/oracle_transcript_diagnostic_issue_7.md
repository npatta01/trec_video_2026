# Oracle-transcript diagnostic (issue #7)

## Outcome

**Status: blocked before inference.** The repository contained no transcript
records with both a label-blind human transcriber and a distinct human reviewer.
Existing automatic ASR output was not promoted to “oracle” status. Consequently,
no matched requests were submitted, no predictions were produced, and no answer
labels were loaded.

The ASR decision gate is therefore **not evaluated**. The safe recommendation is
to stop until genuine manual verification is available; there is no evidence yet
to promote either an ASR comparison or a VLM-grounding investigation.

## Prepared, label-isolated work

- A ten-row lexical-speech panel is frozen in the ignored canonical data tree.
  It includes the horse-age example and excludes the known prosody and temporal-
  action rows.
- The panel builder emits questions and media hashes but excludes reference
  answers and distractors.
- The transcript contract requires full-clip source text, English translation,
  timestamped segments, an input hash, a label-access attestation, and accepted
  review by a person distinct from the transcriber.
- The matched-condition builder fixes Qwen3.5-4B, 12 frame hashes, image limits,
  prompt, seed, and decoding settings across frames-only, automatic-transcript,
  and oracle-transcript conditions.
- The prediction-freeze contract requires one successful prediction for every
  `(video_id, condition)` pair and rejects reference-label fields. Scoring is a
  later phase.

The implementation resolves one source-plan inconsistency in favor of issue #7:
the plan named a temporal-action video while also requiring lexical speech and
excluding non-lexical rows. The temporal row and a prosody row were excluded;
the required horse-age row remains.

## Ignored artifacts

All artifacts remain beneath:

`data/vqa/benchmarks/oracle-speech-10/`

| Artifact | SHA-256 | State |
| --- | --- | --- |
| `manifest.json` | `c13b2d7743449bd2f86f48b6a826c64a73a36df2c12b49e459cc82de798fc3b9` | 10 unique label-free rows |
| `transcript-review.json` | `f962a7de43aeea3558c0b7dec35b87231ea88c0848618f737d966eea620e0889` | blocked, 0/10 accepted |
| `evaluation.json` | `1307fb90d0ff47d15358497593f0cdfc28d79a14911fbe8f3228b5569d962618` | blocked before inference |

No transcript, frame, request, prediction, score, or raw-log artifact is included
in Git.

## Resume protocol

1. Give the transcriber and reviewer only the clip/audio and opaque video ID.
   Do not expose the question, correct answer, distractors, prior predictions,
   or this panel manifest.
2. Collect full-clip records from a human transcriber and a distinct human
   reviewer. Preserve uncertainty literally; do not normalize it into a guessed
   factual answer.
3. Validate all ten records with `transcript_contract`; do not continue unless
   the review result is 10/10 accepted.
4. Freeze one 12-frame set and Whisper large-v3 output per row, build the 30
   matched requests, run inference, and freeze all 30 predictions.
5. Only after the prediction freeze, load labels for blinded adjudication and
   apply the issue #7 ASR decision gate.
