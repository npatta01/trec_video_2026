# Qwen3.5-4B 475-row dstack run

## Objective

Measure the current Qwen3.5-4B 12-frame visual baseline on every VQA 2025
training row that has both an annotation and local video, using a pinned NVIDIA
CUDA dstack runtime. Keep evaluation answers out of the remote inference
inputs; score only after predictions are frozen and retrieved.

## Scope

- Quality panel: 475 local VQA 2025 training videos/questions.
- Excluded from this run: six annotation records without local media; VQA test
  and VQA 2026 rows; VTT caption records.
- Treatment: Qwen3.5-4B, pinned model revision, 12 uniform temporal frames,
  current controlled answer prompt, `max_new_tokens=64`.
- Control for this first substantive run: the current visual-only baseline.
- Not included: answers, incorrect answers, annotation JSON, VTT captions, or
  Whisper transcripts.

## Operational evidence

- Local source media: 475 files, 1.61 GiB.
- dstack canary: three label-free rows completed on NVIDIA L40S; all three
  passed, mean generation time 2.56 seconds/row, one-time model load 38.9
  seconds, remote task cost about $0.05.
- Full run `qwen35-4b-12f-full-20260803-c1`: 475/475 rows passed on an NVIDIA
  L40S; frame preparation took 1,932.15 seconds, workload wall time took
  531.50 seconds, and model load took 23.21 seconds. The dstack task cost
  approximately $0.72 and completed in about 44 minutes end to end.
- Remote input/output artifacts remain private in the existing HF bucket.

## Decision rule

Use the complete baseline as the reference prediction set. Inspect exact match,
token F1, answer-type breakdowns, and failure clusters locally. The frozen
baseline scored 0.090526 normalized exact match, 0.240696 token F1, and 0.156063
lexical METEOR on the 475-row panel. The ledger currently supports
quality-gated/selective Whisper for lexical-speech work, not a broad transcript
default, so this result does not by itself justify attaching Whisper to every
question.

## Status

- [x] Read the experiment-recording guide and current ledger.
- [x] Verify 481 annotation files and 475 matching local videos.
- [x] Complete a label-free dstack CUDA canary.
- [x] Build the 475-row label-free manifest and private input bundle.
- [x] Run and verify the full baseline.
- [x] Score locally and update the owning research record.

## Next action

Use the frozen baseline as the control for the next matched arm. The highest-
value follow-up is a label-blind Whisper/grounding comparison; decide whether
to run it on the declared lexical-speech route or as a route-wide diagnostic.
Two independent dstack GPUs can run separate treatment arms in parallel. If
the frame evidence must be regenerated, first consider a one-pass frame
decoder or cached frame bundle because preparation was about four times longer
than Qwen inference in this run.
