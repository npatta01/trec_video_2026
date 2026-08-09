# VQA Factorial, Oracle-Evidence, and Multi-Pass Design

**Date:** 2026-08-09  
**Status:** Approved direction; revised after Sol/Claude/Gemini panel review  
**Primary theme:** Routing, fusion, and answer selection  
**Related themes:** Models and adaptation; visual and temporal evidence;
speech and transcript evidence; experiment integrity and evaluation

## Objective

Determine whether the current VQA quality gap is driven mainly by answer
elicitation, model capacity, spatial detail, evidence localization, or answer
election, then retain the smallest competition-credible stack that improves
the official semantic metrics under a one-shot confirmation.

The design intentionally makes a large bet—full-panel model-by-prompt testing,
an automated oracle-evidence factorial, and multiple evidence passes—without
collapsing all mechanisms into one uninterpretable treatment.

## Evaluation populations

Only VQA question/answer annotations count as quality labels.

- **Development:** all 475 VQA 2025 training rows with local media. The six
  annotated rows without media are reported as unavailable, not failed.
- **Confirmation:** all 1,883 VQA 2025 testing rows with public ground truth and
  local media. The 117 rows without media are reported separately. This panel
  is scored once after the architecture is frozen.
- **Submission/operations:** VQA 2026 testing is unlabeled and is not used for
  promotion decisions. VTT captions/search collections are out of scope.

## Metric contract

Score every completed quality arm with the NIST 2025 implementation semantics:

- METEOR;
- BERTScore F1 using
  `FacebookAI/roberta-large@722cf37b1afa9454edce342e7895e588b6ff1d59`,
  `bert-score==0.3.12`, layer 17, no IDF or rescaling;
- UMBC STS through URL-encoded requests with explicit timeout/retry and a
  completion denominator; and
- NDCG over ranked answer lists.

Normalized exact match and token F1 remain diagnostics. For a one-answer run,
NIST's NDCG implementation returns 1.0 mechanically; record it as degenerate,
not as answer quality.

For analysis only, define each row's semantic composite as:

`C = mean(METEOR, BERTScore F1, STS)`

Do not call `C` an official metric. For ranked lists, report rank-1, best@5,
best@10, mean@1/5/10, and NDCG@1/5/10 under NIST 2025 semantics, labeled
provisional for 2026 until NIST publishes the updated scorer.

## Phase 0: Frozen baseline semantic recovery

Do not rerun inference. Score the existing 475-row frames-only and
frames-plus-accepted-Whisper predictions. The scorer must:

- prove parity on a hand-checked fixture;
- score 475/475 rows per condition;
- preserve prediction, reference, dependency, and model hashes; and
- emit per-row metrics, aggregates, paired deltas, bootstrap intervals, answer
  lengths, and question strata.

Observed preliminary gate result, to be regenerated as a durable artifact:

| Condition | METEOR | BERTScore F1 | STS |
|---|---:|---:|---:|
| Frames only | 0.186590 | 0.902943 | 0.320919 |
| Frames + accepted Whisper | 0.231053 | 0.910964 | 0.343400 |

The Whisper deltas are positive on all three metrics. This means the baseline
is genuinely weak semantically, not merely under exact match, while accepted
ASR is useful additional evidence.

## Phase 1: Model-by-prompt factorial

Hold fixed the same uniform-12 frame timestamps, base pixel budget, accepted
Whisper transcript policy, deterministic decoding, and 64-token output budget.
Run a complete `2 × 2` factorial on all 475 development rows:

| ID | Model | Answer policy |
|---|---|---|
| `4b-current` | pinned Qwen3.5-4B | current short contract |
| `4b-question-aware` | pinned Qwen3.5-4B | question-aware contract |
| `8b-current` | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | current short contract |
| `8b-question-aware` | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | question-aware contract |

The frozen `4b-current` predictions may be reused only after a three-row
repeat proves the new runner reproduces them. If it does not, regenerate all
four cells under the same runtime.

The question-aware policy is label-blind and deterministic:

- count questions request only the number and counted entity;
- visible-text and quoted-speech questions request exact wording;
- temporal questions request a complete answer preserving order and timing;
- causal/procedural questions request a complete sentence containing the
  cause or key steps; and
- other questions request a concise but complete answer covering every part.

The model may reason internally, but output contains only the final answer.
The policy, classification, and exact instruction are recorded in each result.

### Phase 1 promotion rule

Against the matched `4b-current` cell, a cell advances when:

- `ΔC >= 0.015`;
- the paired 95% bootstrap lower bound for `ΔC` is above zero;
- mean METEOR and mean BERTScore changes are positive;
- mean STS changes by at least `-0.005`; and
- completion is 475/475 with no integrity failure.

Call one mechanism *dominant* only when its paired gain is at least 0.020 and
its interval separates from the competing effect. Otherwise report a mixed
bottleneck.

## Phase 2: Routed spatial-detail test

Do not judge resolution from a global average dominated by non-detail rows.
Before looking at new resolution predictions, freeze a question-only route:

- all identified OCR/visible-text rows;
- up to 45 additional small-object/spatial-detail rows selected from question
  text only; and
- 30 seeded non-detail controls.

Under the Phase 1 winning prompt, compare base frames with a single declared
higher-resolution/tiling policy under both models. Promote only as a route when
the detail stratum has `ΔC >= 0.030` with a positive lower bound and controls
regress by no more than 0.005.

## Phase 3: Reference-conditioned silver oracle

Freeze 120 unique baseline semantic failures before inspecting Phase 1
predictions:

- 30 temporal;
- 30 causal/procedural;
- 30 speech-dependent;
- all 15 identified OCR/text rows; and
- 15 seeded general-visual rows.

Use disjoint precedence `OCR → speech → temporal → causal → general`. Fill a
short stratum from the seeded general evidence-sensitive failure pool and
record every substitution.

Under the winning answerer, run a `2 × 3` evidence factorial:

- frames: uniform-12 versus reference-conditioned selected-12;
- transcript: none, full accepted Whisper, or reference-conditioned bounded
  Whisper spans.

Also run the combined selected-frame/selected-span condition once through the
original 4B/current-prompt answerer. This tests whether better evidence remains
unusable by the original answerer.

The selector may see the question and reference. Its output schema contains
only source frame IDs/timestamps and source transcript segment IDs/time ranges.
It cannot emit rationale, captions, answer text, crops, or free text. A separate
builder reconstructs raw answerer inputs from those IDs and proves that the
reference and selector output text do not enter the request. The answerer sees
only the question, raw selected frames and timestamps, and raw source transcript
spans. All selection manifests and requests are saved with checksums.

Evidence localization is material when the combined silver-oracle condition
improves `C` by at least 0.050 with a positive lower bound. It is unlikely to be
the leading bottleneck only when the upper bound is below 0.020. Intermediate
results imply a hybrid bottleneck. Because this is a failure-enriched panel,
its effect is diagnostic and must not be presented as a population estimate.

## Phase 4: Ten deterministic evidence passes and election

Use the winning answerer, fixed answer policy, deterministic decoding, and ten
auditable evidence presentations:

1. midpoint uniform grid;
2. start-aligned uniform grid;
3. end-aligned uniform grid;
4. quarter-phase grid;
5. three-quarter-phase grid;
6. scene-boundary coverage;
7. question-conditioned temporal peaks;
8. coarse-to-local frames around the strongest peak;
9. speech-centered intervals; and
10. routed OCR/detail crops, with a frozen temporal fallback when inapplicable.

The ten passes generate candidates; they do not vote blindly. A reference-blind
ranker scores candidate support against the exact evidence that generated it,
clusters normalized duplicates, and uses deterministic tie-breaking. If fewer
than ten unique normalized answers remain, a frozen reference-blind alternate-
generation policy backfills candidates, and every backfill is labeled.

Promote multi-pass election when elected rank-1 improves `C` by at least 0.010,
its paired lower bound is positive, and no semantic metric regresses by more
than 0.005. A `best@10 - elected@1` gap of at least 0.020 establishes useful
remaining election headroom. Best@k uses references only after prediction
freeze and is never a deployable ranker.

## Runtime and dstack contract

- Run a three-row canary first: temporal, speech-dependent, and OCR/detail.
- Canary cells: both models × both answer policies at base resolution, one
  high-resolution OCR request, and one repeated base request for determinism.
- Stop after 30 minutes or on the first integrity failure.
- Pass only with complete outputs, identical repeated deterministic answers,
  exact evidence/checksum identity, no label leakage, no OOM/NaN, recorded peak
  VRAM, and measured throughput.
- Use independent one-GPU L40S row shards, pinned image/model/runtime revisions,
  disjoint stable manifests, per-shard outputs, and checksum-based merge. Do not
  use tensor parallelism.
- Report both wall time and total GPU-hours. Linear sharding changes wall time,
  not compute.
- A 2,443-row single-pass 2026 arm targets at most 90 minutes wall time on the
  declared shard count. Do not claim ten passes meet that target without an
  actual throughput projection; reduce passes or increase shards only after the
  canary measurement.

Before experimental promotion is known, preserve a validator-passing fallback
path based on the current 4B + accepted-Whisper stack so the deadline cannot be
missed by research failure.

## Artifacts and integrity

Every run produces a private, checksummed bundle containing:

- frozen label-free manifest and code/config revisions;
- exact model, processor, runtime image, and dependency revisions;
- question-type and answer policy per request;
- source media identity;
- selected frame images, requested and actual timestamps, per-frame hashes,
  contact sheets, and selection policy;
- audio identity, Whisper model/revision, timestamped transcript segments,
  accepted/rejected status, and selected span IDs/times;
- raw predictions for every condition and pass;
- candidate normalization, clustering, backfill, ranking, and validation data;
- wall time, GPU-hours, utilization samples, and peak VRAM;
- scorer versions, per-row metrics, aggregates, bootstrap intervals, and
  completion denominators; and
- a final `SHA256SUMS` verified after download.

Labels remain local until remote predictions freeze. The silver-oracle selector
is the sole exception and runs in an explicitly label-bearing boundary; its
schema-only outputs cross into a separately verified reference-blind answerer
boundary.

## Panel review disposition

All three lanes supported the staged architecture and rejected the original
confounded combined arm. Sol recommended the complete model-by-prompt factorial,
separate routed resolution test, schema-only `2 × 3` oracle-evidence factorial,
and ten deterministic evidence views. Claude favored a larger oracle panel and
fewer mixed deterministic/stochastic passes; Gemini favored five diverse passes.
This design chooses 120 oracle rows for confidence and ten deterministic passes
to honor the user's multi-pass objective and preserve reproducibility, while
using the canary throughput gate to prevent an unaffordable full run.

The panel packet is
`plans/advisor-packets/2026-08-09-vqa-multipass-full-panel.md`.
