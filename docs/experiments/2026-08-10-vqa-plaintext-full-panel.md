# Full-panel AV timeline and scene-selection diagnostic


Date: 2026-08-10

Evidence validity: **Verified for operational completion; diagnostic for
answer quality and promotion**

Decision disposition: **Not promoted under the tested condition**

## Completed evidence record

- **Primary theme:** visual and temporal evidence.
- **Related themes:** speech/transcripts, models and adaptation, and
  routing.
- **Question:** can a full-video audio-video timeline followed by
  question-conditioned scene selection provide a stronger answer path
  than the current uniform-frame baseline?
- **Hypothesis:** a timeline pass should expose the relevant scene,
  while a second pass should answer from a narrow, question-relevant
  window.
- **Falsifier:** no improvement over the existing matched baseline,
  material completion failures, or evidence/provenance failures.
- **Canonical artifact directory:**
  `docs/experiments/artifacts/vqa-plaintext-full-20260810/`.

The full local inspection bundle is retained under
`selected-evidence/frames/`, `selected-evidence/audio/`, and
`selected-evidence/contact-sheets/`. It is intentionally not committed
because it is private derived media and is approximately 782 MB. The
tracked manifests retain its hashes and timestamps.

## Tested condition and panel

The run used `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` at
revision `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`, served by the
digest-pinned vLLM runtime
`sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`.
The map pass received normalized full video plus native audio. The
selector received only the resulting plain-text timeline and the
question. Both passes used greedy decoding, no thinking, no guided JSON
decoding, and no retries.

The panel was the complete local-media subset of the VQA-2025 training
annotations: 475 videos from 481 annotation files, with six annotation
rows excluded because local media was unavailable. Expected answers were
loaded only after predictions were frozen for scoring.

## Completion and quality result

| Measure                      |             Observed |
|------------------------------|---------------------:|
| Timeline maps                |            474 / 475 |
| Selector answers             |            462 / 475 |
| Selector completion rate     |               97.26% |
| Selector-parse failures      |                   12 |
| Map-parse failures           |                    1 |
| Selected frames              | 5,700 (12 per video) |
| Extracted audio              |            475 / 475 |
| Accepted Whisper transcripts |            471 / 475 |

Post-freeze semantic scoring was computed on the 462 completed selector
rows. The second line treats the 13 incomplete rows as zero quality
credit; it is the more honest full-panel operational view.

| Panel view | METEOR | BERTScore-F1 | STS | Semantic composite |
|----|---:|---:|---:|---:|
| Selector-completed rows (462) | 0.2186 | 0.8963 | 0.3129 | 0.4759 |
| Full panel, failures at zero (475) | 0.2126 | 0.8717 | 0.3044 | 0.4629 |

NDCG is recorded as `1.0` with `ndcg_degenerate=true`: there is only one
free-form answer per video, so no answer ranking was evaluated. It must
not be read as a quality win.

For context, the existing same-panel Qwen3.5-4B run scored semantic
composite `0.4711` with frames only and `0.4962` with the matched
Whisper arm. That is not a perfectly controlled model comparison because
this experiment uses Nemotron and a different evidence protocol, but it
is sufficient to reject promotion of this event-map answer path as the
current default.

## What the failure inspection supports

All 13 failed rows have complete local visual/audio evidence artifacts.
The 12 selector failures use a deterministic first-event fallback for
inspection; the one map failure uses a uniform fallback. Contact sheets
were inspected for all 13 failures. The frames contain visible actions,
text, or people in every case; several also have answer-bearing audio or
on-screen text. Therefore the 13 failures are not evidence of 13 missing
videos.

The failure mechanism is mixed:

- timeline output sometimes concatenated multiple event records into one
  malformed description;
- the selector then failed its plain-text response contract;
- accepted Whisper is not equivalent to answer-relevant speech, with
  some short boilerplate or poor translations;
- one selected three-second window is often too narrow for counting,
  temporal, or causal questions that require context across scenes.

The full-panel stratum result reinforces this diagnosis: counting had
the lowest completed-row semantic composite (`0.3866`) and STS
(`0.0773`), while audio/text questions were strongest (`0.5510`) but
only had 14 rows. This is a diagnostic pattern, not a claim about the
hidden competition test set.

## What this rules out now

- Do not promote this one-selector Nemotron pipeline over the existing
  Qwen baseline.
- Do not interpret the technical canary or valid timestamps as
  answer-quality evidence.
- Do not treat NDCG `1.0` as a meaningful ranking result.
- Do not blame all failures on missing frames: the failed-row inspection
  shows relevant visual/audio material was usually available.

## What it does not rule out

This result does not rule out a multistage system. It leaves open a
stronger design with top-k candidate scenes, adjacent temporal context,
question-type specific answering, and a final answer verifier. It also
does not establish that Whisper is globally helpful: the existing
matched Qwen run is the evidence for that separate frames-only versus
Whisper comparison.

## Next decision test

The next high-value full-panel test should use the existing Qwen3.5-4B
answerer to hold model quality fixed:

1.  retain uniform-12 as the control;
2.  use the full-video map to produce the top three candidate windows;
3.  answer from those windows plus the relevant Whisper span, when
    accepted;
4.  use question-type-specific answer instructions for counting,
    temporal, causal, and speech questions;
5.  use a verifier/election pass to reject answers unsupported by the
    evidence.

Promote only if the 475-row run completes, improves semantic composite
over the existing Whisper baseline, and does not regress the counting or
temporal strata under the same scorer. This is a stronger bet than
another one-shot selector pass while reusing the already-saved evidence
and avoiding new dstack provisioning until the implementation is ready.

## Artifact index and reproducibility

- [Artifact retention
  map](artifacts/vqa-plaintext-full-20260810/README.md)
- [Full timeline/selector
  results](artifacts/vqa-plaintext-full-20260810/results.jsonl)
- [Full timeline maps](artifacts/vqa-plaintext-full-20260810/maps.jsonl)
- [Semantic
  score](artifacts/vqa-plaintext-full-20260810/selector-quality.json)
- [Selected-evidence
  manifest](artifacts/vqa-plaintext-full-20260810/selected-evidence/evidence.json)
- [Frames-only
  requests](artifacts/vqa-plaintext-full-20260810/selected-evidence/requests.frames.json)
- [Whisper
  requests](artifacts/vqa-plaintext-full-20260810/selected-evidence/requests.whisper.json)

The final compact output hashes are recorded in `summary.json`:

- `maps.jsonl`:
  `26760a015b88075aa1e8a8028c4c5ec02689262488a416d311d3bc0e90ff3d04`
- `results.jsonl`:
  `1010b48418f2d5053971ba2699be320c1f08f20e615c94102bf02a2df3c51bbb`

No dstack reprovisioning is required to inspect or score the saved
artifacts.
