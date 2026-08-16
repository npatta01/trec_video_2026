# Gemini 3.7 Flash Two-Video Native-AV Smoke Design

Date: 2026-08-16

## Objective

Determine whether stable `gemini-3.7-flash` can execute the repository's
native-audio/video direct-answer condition on two preserved TRECVID development
videos under a controlled comparison with the pinned
`gemini-3.1-pro-preview` condition.

This is a compatibility and directional diagnostic, not a model promotion
benchmark. It cannot replace the completed 475-row Gemini 3.1 Pro result or
support a whole-panel ranking.

## Approved approach and alternatives

The approved approach is a fresh two-arm Batch A/B:

1. upload each of two preserved delivery files once;
2. reuse the same active File API URI in both arms;
3. submit one two-request Batch job to Gemini 3.1 Pro and one to Gemini 3.7
   Flash;
4. freeze both arms before opening references, prior predictions, or scores.

Reusing only the historical Gemini 3.1 outputs would cost less but would add a
time/provider-runtime confound. Running Gemini 3.7 alone would establish basic
availability but no matched directional evidence. The fresh A/B is preferred
because its expected maximum token charge is below `$0.15` and it preserves the
strongest comparison boundary.

## Question, hypothesis, and falsifiers

Question: can Gemini 3.7 Flash accept the frozen native-AV request surface and
produce two non-empty direct answers under the same media, prompt, sampling,
resolution, thinking, seed, and output-cap contract as the fresh Gemini 3.1
control?

- **Compatibility hypothesis:** both Gemini 3.7 requests complete successfully,
  return non-empty text, identify a 3.7 model version, and report usage
  consistent with native media processing.
- **Compatibility falsifier:** model preflight fails, submission is rejected,
  either response is missing/empty/errored, returned model provenance is wrong,
  or the custom-FPS evidence cannot be reconciled with the frozen request and
  usage metadata.
- **Directional quality hypothesis:** after prediction freeze, the two 3.7
  answers are at least as useful as the matched 3.1 answers on the frozen
  references and qualitative evidence review.
- **Quality falsifier:** either 3.7 answer is materially worse than its matched
  control, or the two rows disagree. A two-row outcome never authorizes
  promotion; at most it authorizes a separately designed larger canary.

## Panel and selection

The source-of-truth panel is the immutable private protocol at:

`data/vqa/benchmarks/direct-gemini-batch-475-v1/protocol.json`

Select exactly the first two rows in its frozen order using only protocol
metadata. The selected durations are `18.308` and `21.268` seconds. Do not read
the completed 475-row predictions, scores, annotations, answers, failure audit,
or row grades during preparation.

For each selected row, reuse the already preserved WebM delivery under the
completed private root. Preparation must verify that the delivery is a regular
file rather than a symlink and bind its path, SHA-256, byte size, MIME type,
source SHA-256, duration, question, and prior delivery-provenance SHA-256. Do
not copy or remux the delivery.

The two-row selection is a compatibility sample in historical panel order. It
is not category-balanced and cannot estimate prevalence or average quality.

## Arms and held-fixed request surface

| Field | Control | Candidate |
|---|---|---|
| Model | `gemini-3.1-pro-preview` | `gemini-3.7-flash` |
| Condition | direct answer | direct answer |
| Media | same File API URI | same File API URI |
| Prompt | same frozen prompt | same frozen prompt |
| Video metadata | `fps=4.0` | `fps=4.0` |
| Media resolution | `HIGH` | `HIGH` |
| Thinking | `HIGH`, thoughts excluded | `HIGH`, thoughts excluded |
| Seed | `42` | `42` |
| Output | plain text | plain text |
| Maximum output | 2,048 tokens | 2,048 tokens |
| Sampling override | `temperature=0.0` | omitted |

The temperature difference is intentional and must remain visible. The pinned
3.1 condition used `temperature=0.0`; Google deprecates `temperature`, `top_p`,
and `top_k` for Gemini 3.6 and later. Supplying a deprecated sampling override
to 3.7 would test a known-incompatible request rather than the model.

No candidate count, tools, grounding, response schema, system turn, prior
answer, or map is supplied. The video part precedes the prompt part.

## Label isolation and privacy

Before both prediction arms freeze, provider inputs and private preparation
artifacts may contain only experiment/arm identity, model identity, selected
video identity, question, duration, delivery path/hash/MIME data, File API
records, request configuration, and provider job/status metadata.

They must not contain annotations, correct or incorrect answers, labels,
references, grades, metrics, prior Gemini outputs, Qwen outputs, Whisper text,
or qualitative verdicts. Credentials are sourced only from the approved mode
`0600` private environment file and are never printed, copied, hashed into a
tracked artifact, or passed as CLI arguments.

All row identities, questions, provider records, responses, references, and
row-level comparisons remain private under `data/`. Tracked reporting contains
only counts, decisions, aggregate usage/cost, and immutable hashes.

## Provider lifecycle and one-attempt rule

The additive runner exposes these transitions:

1. `prepare`: create an exclusive protocol and input freeze after validating
   the historical protocol and both delivery bundles;
2. `preflight`: authenticate and fetch both exact model records without an
   inference call, then freeze sanitized model metadata;
3. `upload`: upload each delivery once and wait for `ACTIVE`, recording one
   exclusive upload record per row;
4. `submit`: create exactly one inline Batch job for the named arm, containing
   exactly two requests;
5. `status`: append provider snapshots without modifying earlier evidence;
6. `freeze`: require both terminal jobs to succeed with exactly one non-empty
   response per frozen row and no response-level error, then create one
   exclusive prediction freeze;
7. `score`: only after freeze, load the two canonical annotations and produce
   a private matched diagnostic.

There is no automatic or manual paid retry in this experiment. A provider or
row error is terminal, preserved, and yields `not_ready`. Duplicate submission,
incomplete output, duplicate response, mutation, or overwrite attempts fail
closed.

## Custom-FPS and provenance evidence

Request serialization must prove `fps=4.0` and high media resolution in all
four requests. The freeze records returned `model_version`, response ID,
finish reason, and complete usage metadata.

Google documents approximately 280 tokens per high-resolution video frame and
32 audio tokens per second. At four frames per second, the selected 39.576
seconds imply approximately 45,591 media tokens per arm before prompt/metadata
overhead. Compare each arm's summed returned prompt-token usage with this
frozen estimate. The accepted reconciliation band is inclusive `0.75x` through
`1.35x` (`34,193` through `61,548` prompt tokens per arm); it is deliberately
wide for metadata and rounding but excludes the documented one-FPS scale. A
value outside that band does not get silently explained away; it yields
`custom_fps_not_verified` and blocks a capability or quality claim.

## Cost boundary

The two videos total 39.576 seconds. Using the documented high-resolution
four-FPS approximation and the two 2,048-token output caps:

- Gemini 3.1 Pro Batch is bounded at roughly `$0.071`;
- Gemini 3.7 Flash Batch is bounded at roughly `$0.025` through 2026-12-31;
- the frozen experiment cap is `$0.15` total.

The runner rejects a protocol whose deterministic estimate exceeds `$0.15`.
The observed usage-based estimate is recorded after freeze. This allocation is
within the existing `$1.00` Gemini verifier stream cap and the `$10.00` program
cap.

## Post-freeze diagnostic

The prediction freeze is followed by a separate hash-only annotation-binding
transition. That transition first re-verifies the complete `quality_ready`
prediction freeze, locates exactly the two canonical ID-derived annotation
files, hashes their raw bytes without parsing JSON or extracting a reference,
and writes one exclusive private manifest. It is the first permitted access to
annotation bytes and cannot run before both model arms freeze.

The score step fails closed unless that exact two-row manifest exists and
validates. It hashes each annotation's raw bytes and compares the result with
the frozen manifest before parsing JSON or reading a reference. It then records
normalized exact match and token-F1 for each arm, plus a private side-by-side
inspection packet containing the question, reference, and both answers. These
two dependency-free metrics are diagnostics, not a substitute for the
established full-panel semantic composite.

The aggregate decision is one of:

- `not_ready`: either arm is incomplete or provenance/FPS validation fails;
- `compatible_no_directional_claim`: 3.7 completes but the two-row quality
  evidence is tied, mixed, or worse;
- `ready_for_larger_canary`: 3.7 completes, has no matched regression, and has
  at least one matched diagnostic improvement.

No outcome changes the production baseline or triggers a 475-row rerun.

### Implementation erratum: explicit annotation binding

The executable CLI therefore includes `bind-annotations` between `freeze` and
`score`. This is an additive integrity correction to the original command list,
not an inference arm or quality result. The binding manifest remains private,
contains the two identities, canonical paths, and SHA-256 values, and is itself
hash-bound into the private score and score freeze. No tracked artifact may
contain those row-level fields.

## Artifacts

Private root:

`data/vqa/benchmarks/gemini37-flash-two-video-smoke-v1/`

Tracked sanitized evidence, if the run reaches a terminal freeze:

`docs/experiments/artifacts/gemini37-flash-two-video-smoke-20260816/`

The owning result is the Visual and Temporal Evidence theme. The speech/Qwen
report may link to it only as a separate model test; it must not copy the full
claim.

## Verification before paid execution

Provider-free tests must prove arm-specific sampling configuration, identical
media/question controls, exact two-row/two-request boundaries, privacy-key
rejection, immutable source and delivery binding, cost rejection, exclusive
transitions, one submission per arm, no retry path, and complete/duplicate/error
freeze rejection. A synthetic SDK fake must exercise provider serialization,
job collection, and model-version retention.

Before submission, also verify canonical data resolution, credential-file mode,
both authenticated model preflights, delivery hashes, privacy scan, focused
tests, Ruff, and `git diff --check`. Any failure stops before a paid call.
