# TimeLens2 Selector Quality Gate Design

## Objective

Decide whether pinned TimeLens2-2B improves relevant-frame selection and
downstream VQA answers at the same 12-frame budget as uniform sampling. This
is a focused development-panel gate, not evidence that the selector
generalizes to the separate frozen-30 benchmark.

## Frozen inputs and revisions

- Selector: `MCG-NJU/TimeLens2-2B` at revision
  `90dd30b3eeff3520a30c830eff2209f4ad788ca9`.
- Answer model: `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime image:
  `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
- Evaluation source: only
  `data/vqa/benchmarks/temporal-relevant-scene-oracle-10/annotations/adjudicated.json`
  and its accepted freeze/checksum sidecars. Its accepted SHA-256 is
  `774326c801b85ac3e542c29be44040630b6ba1afadbfa94971366b425c08de98`.
- The temporal-required-moment oracle and the similarly named `-r2` benchmark
  are out of scope.

The source must contain exactly ten distinct accepted video identities with
valid media hashes and durations. A mismatch stops the experiment before
inference.

## Sampling contract

The matched conditions contain exactly 12 distinct chronological frames per
row:

1. **uniform-12:** twelve equal-bin midpoint timestamps over the full clip.
2. **hybrid-12:** six equal-bin midpoint timestamps over the full clip plus
   six local timestamps inside normalized TimeLens intervals.

TimeLens raw intervals remain unchanged and are validated before
normalization. Non-finite, reversed, or out-of-duration intervals are
rejected. Sorting and overlap merging occur only in the linked normalized
representation.

No time padding is added outside predicted intervals. Local seats are
allocated in proportion to normalized interval duration using deterministic
Hamilton allocation, then placed at equal-bin midpoints within each interval.
The six global frames provide context and fallback coverage. If decoded frame
rounding produces duplicate images, the row fails rather than silently
changing the budget.

## Label-blind data flow

The workload has two physically separated phases:

1. **Inference phase:** prepare a label-free ten-row request manifest; run
   TimeLens; validate and freeze raw and normalized selector results; build
   and checksum both frame recipes; extract and checksum the exact frames;
   run the pinned answer model for both matched conditions; freeze all twenty
   answers and the completion denominator.
2. **Evaluation phase:** only after the inference freeze verifies, load the
   accepted adjudication file; score required-component coverage for
   hybrid-12 and compare matched answer grades against uniform-12.

Inference code must have no access to the adjudication path. Evaluation must
refuse incomplete, mutable, hash-mismatched, or wrong-revision inputs.

## Runtime and artifact contract

- CUDA/dstack only; no ROCm and no local model download.
- Use one 48 GB-class GPU from the already compatible A40-or-better set, with
  at least 24 GB VRAM requested, on-demand capacity, detached submission, and
  foreground workload execution.
- Preview with `echo "n" | dstack apply -f ...` and retain the complete
  declined preview before submission.
- The exact-image canary and three-video model compatibility gate are already
  valid and must not be repeated for quality reasons.
- Use a new immutable run name and an empty durable prefix below
  `hf://buckets/Npatta01/trec_video_2026/experiments/<run-name>/`.
- Keep `HF_TOKEN` only in dstack secrets. Never print it or use secret-bearing
  diagnose output.
- Set hard `max_price` and `max_duration`; total TimeLens-track spend must stay
  within the original $4 allocation. The recorded spend before this gate is
  $0.6580, leaving $3.3420.
- Persist phase state, per-row attempts, failures, latency, peak memory,
  device, precision, model/runtime revisions, request hashes, frame hashes,
  and completion denominator incrementally and again in a finalizer.
- Upload only the exact private panel inputs and durable outputs. List,
  download to a fresh temporary directory, and checksum-verify every durable
  artifact after completion.
- Raw competition rows, media, labels, outputs, logs, and private manifests
  remain ignored and out of Git. Git records only sanitized aggregate
  evidence.

## Failure behavior

Setup, model-load, OOM, decoding, parsing, frame-extraction, answer, freeze,
or evaluation failures retain partial observations and the denominator. A
row is accepted only when its selector output, both 12-frame recipes, and both
answers complete and verify. Any incomplete accepted row makes the quality
gate invalid rather than counting as a regression.

The first infrastructure/runtime failure permits at most one narrowly
corrective immutable attempt. A second failure of the same gate is persistent
and stops the track. A technically valid quality result is never repeated
because its quality is poor.

## Evaluation and decision

After the answer freeze, evaluate:

- whether hybrid-12 covers every adjudicated required component per row;
- the number of rows with complete required-component coverage;
- matched answer transitions from uniform-12 to hybrid-12; and
- aggregate correct/partial/wrong counts for both conditions.

Promote TimeLens2 to a frozen-30 confirmation only if:

- all ten accepted rows complete;
- every required component is covered in at least eight rows;
- at least two rows improve over uniform-12; and
- at most one row regresses.

A complete result that misses any threshold is a valid negative result and
ends the TimeLens2 selector track. A pass authorizes designing the separate
30-video confirmation; it does not by itself change the uniform-12 default.

## Tests and verification

Tests are written and observed failing before implementation. They cover the
ten-row source binding, label-key rejection, interval validation,
no-padding/inside-interval sampling, deterministic Hamilton allocation,
distinct chronological frames, freeze-before-label enforcement, paired
completion, partial failure persistence, promotion thresholds, dstack pins,
immutable artifact URI, price/duration bounds, and private-path exclusions.

Before committing implementation, run model-specific tests, relevant VQA
runner tests, configuration tests, a private-content scan, `git diff --check`,
and the declined dstack preview. Final evidence records the exact commands and
results.
