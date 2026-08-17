# Qwen Transcript-to-Answer Diagnostic Design

Date: 2026-08-16

## Objective

Test the user's concrete hypothesis that the Stage 0.5f `0/12` result was
caused by comparing raw transcripts with differently worded VQA answers, and
that a normal answer-generation step can turn the frozen transcript evidence
into a useful short answer.

This is a transcript-to-answer diagnostic. It is not a transcription-fidelity
benchmark and cannot rank Qwen ASR against Whisper by WER or CER.

## Approved test

Reuse the exact 12 Stage 0.5f rows and their already frozen Qwen and Whisper
transcripts. Build two matched, label-free arms:

1. original question plus the frozen Qwen transcript;
2. the same question plus the frozen Whisper transcript.

Give all 24 requests to the same pinned `Qwen/Qwen3.5-4B` answer model with no
frames. Hold the prompt, model revision, runtime image, deterministic decoding,
32-token cap, and request order fixed. Freeze all answers before opening the
official annotations.

No ASR inference is rerun. Local execution uses the existing cached model and
ROCm image and has `$0.00` paid cost.

## Source and model bindings

- Stage 0.5f root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-quality-v1/`
- Canonical 475-row question manifest and frozen Whisper payload are the exact
  paths already enforced by `vqa_qwen_aligner_canary.py`.
- Qwen answer model: `Qwen/Qwen3.5-4B`.
- Answer-model revision:
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime image:
  `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`.
- Answer policy: repository `current` policy, greedy decoding, 32 new tokens.

Preparation must verify the Stage 0.5f prediction freeze, the 12 selected
identities and audio hashes, the Qwen raw-result bytes, the canonical manifest
bytes, the frozen Whisper bytes/checksum binding, and the runner source hash.
Each arm must have exactly one non-empty transcript for every selected row.

## Label isolation

Preparation and inference may read questions, identities, transcript text,
source hashes, and operational metadata. They must not open annotation files,
correct or incorrect answers, prior answer predictions, scores, or review
labels.

Requests contain no reference or answer field. The official annotations are
opened only after a 24/24 non-empty prediction freeze verifies.

## Scoring and interpretation

Post-freeze scoring reports normalized exact match and token F1, plus a private
semantic inspection packet containing the question, reference, Qwen-conditioned
answer, and Whisper-conditioned answer. Because the diagnostic has only 12
rows and the original problem is paraphrase, manual semantic correctness is
the primary interpretation; lexical metrics are secondary diagnostics.

The private review packet is arm-blinded and deterministically ordered before
adjudication. Each answer is labeled `correct`, `incorrect`, or `ambiguous`
based on whether it answers the question with meaning compatible with the
official reference. A tracked report may contain only aggregate counts and
hashes, never row identities, questions, transcripts, references, or answers.

The outcome is:

- `formulation_bridge_observed` if at least one answer previously scored as an
  exact-containment miss is semantically correct after the answer stage;
- `no_formulation_bridge_observed` if all 24 generated answers are incorrect or
  ambiguous;
- `not_ready` if any request, result, source binding, or freeze is incomplete.

The paired Qwen-versus-Whisper correctness counts are descriptive. They do not
become an ASR-quality ranking because no verbatim transcript references exist
for these 12 TRECVID rows.

## Regression boundary

The Stage 0.5f historical `0/12` artifact remains immutable but must be labeled
construct-invalid for ASR quality. New code must fail closed if a VQA
`correct_answer` is supplied as a transcript-fidelity reference. A synthetic
regression test must demonstrate that a faithful transcript can omit the exact
short-answer wording while still supporting a correct generated answer.

## Private and tracked artifacts

Private root:

`data/vqa/benchmarks/model-verifier-canaries-20260814/stage05g-qwen-transcript-answer-v1/`

If the run reaches a terminal freeze, add only a sanitized aggregate beneath:

`docs/experiments/artifacts/qwen-aligner-quality-20260816/`

and update the existing Qwen experiment QMD as the single owning result.
