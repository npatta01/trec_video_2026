# Frozen-30 video QA benchmark: protocol-v2 results

Date: 2026-07-24 UTC

Status: **protocol-v2 inference and scoring complete**. The clean run contains
120/120 valid VQA predictions and 30/30 valid Whisper executions. Final
independent review is complete.

## Technical summary

There is no single winner across all objectives:

- **Best normalized exact match and token F1:** Qwen3.5-4B with 12 uniformly
  sampled frames plus a quality-gated Whisper large-v3 transcript, at
  `0.366667` EM (11/30) and `0.506795` token F1.
- **Best METEOR and speed:** MiniCPM-o 4.5 with native audio disabled, at
  `0.374089` METEOR and 4.50 minutes end-to-end outer wall time. It was about
  7.9× faster than the transcript condition's downstream Qwen pass, based on
  exact outer totals of 2,140.85 versus 269.86 seconds. This comparison
  excludes the transcript arm's separate 526.90-second ASR phase.
- **Twenty-four frames are not the default:** versus matched 12-frame Qwen,
  24 frames added no exact answers and produced only `+0.038083` F1 and
  `+0.019465` METEOR. Both paired 95% intervals include zero. Median short-row
  latency rose from 9.67 to 22.12 seconds and peak allocation rose from 10.56
  to 15.30 GiB.
- **The transcript result is exploratory, not a universal promotion:** it
  improved dialogue strongly, but reduced temporal/action F1 by `0.055159`
  and visual-control METEOR by `0.103233`. A router or stricter audio gate is
  required before using transcripts by default.

These are descriptive model-selection results on a balanced 30-example panel
from labeled 2025 training data. They do not estimate performance on the
unlabeled 2026 evaluation set.

## Scope and metric definitions

The frozen panel contains 30 videos, with six examples in each of:
dialogue/speech, non-speech sound/music, temporal/action,
spatial/OCR/color, and visual controls. Every condition used the same ordered
video IDs and questions.

The four clean protocol-v2 conditions were:

1. Qwen3.5-4B with 12 uniformly sampled frames;
2. Qwen3.5-4B with 24 uniformly sampled frames;
3. MiniCPM-o 4.5 native-video input with audio disabled;
4. Qwen3.5-4B with the exact 12-frame evidence plus a quality-accepted Whisper
   large-v3 transcript, or null when transcript quality failed.

Primary metrics are:

- **Normalized exact match (EM):** one only when normalized prediction and
  reference strings are identical.
- **Token F1:** harmonic mean of normalized prediction/reference token
  precision and recall.
- **METEOR:** NLTK 3.9.2 METEOR with frozen punkt-tab and WordNet resources.

Paired intervals below are exploratory 95% bootstrap intervals using 2,000
resamples and seed `20260724`; this 30-example panel was not sized for a
confirmatory superiority claim.

## Accuracy and measured local cost

All four conditions completed 30/30 rows. Row timing is
`minimum / median / maximum`. Outer wall time includes container, model-load,
inference, and orchestration time across the scheduled resumable chunks.

| Condition | EM | Token F1 | METEOR | Row inference seconds | Outer wall | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 12f + Whisper large-v3 | **0.366667** | **0.506795** | 0.338955 | 9.197 / 10.233 / 338.291 | 35.68 min downstream + 8.78 min ASR | 10.885 downstream / 4.479 ASR |
| MiniCPM-o 4.5, audio off | 0.266667 | 0.482014 | **0.374089** | 5.623 / 7.022 / 14.679 | **4.50 min** | 19.067 |
| Qwen3.5-4B, 24f | 0.300000 | 0.468808 | 0.305530 | 20.280 / 22.124 / 423.005 | 36.64 min | 15.302 |
| Qwen3.5-4B, 12f | 0.300000 | 0.430725 | 0.286064 | 9.031 / 9.671 / 337.811 | 35.48 min | **10.559** |

The Qwen totals are dominated by a small number of content-dependent long
generations; ordinary 12-frame rows took about ten seconds. MiniCPM did not
show that long-tail behavior on this panel. Exact-answer coverage was 11/30
for Qwen+Whisper, 9/30 for each visual Qwen condition, and 8/30 for MiniCPM.
The transcript condition had two exact wins over matched visual-only Qwen and
no exact losses; MiniCPM contributed one exact answer missed by all Qwen
conditions.

Each VQA condition executed in four append-only checkpoint chunks ending at
3, 10, 20, and 30 valid rows. The workflow resumed from every checkpoint
without replacing accepted rows. All 120 VQA rows and all 30 Whisper rows
passed on ROCm/BF16; the clean run had **zero ROCm, CPU-fallback, OOM, timeout,
or runner failures**.

## Results differ by question type

Each cell is `EM / token F1 / METEOR` over six examples.

| Condition | Dialogue/speech | Sound/music | Temporal/action | Spatial/OCR/color | Visual controls |
|---|---|---|---|---|---|
| Qwen 12f + Whisper | `0.666667 / 0.700000 / 0.552793` | `0 / 0.166667 / 0.135859` | `0 / 0.222863 / 0.208448` | `1 / 1 / 0.627523` | `0.166667 / 0.444444 / 0.170155` |
| MiniCPM audio-off | `0.333333 / 0.604286 / 0.368408` | `0 / 0.138889 / 0.165679` | `0 / 0.228005 / 0.214642` | `0.833333 / 0.900000 / 0.708664` | `0.166667 / 0.538889 / 0.413052` |
| Qwen 24f | `0.333333 / 0.472222 / 0.286648` | `0 / 0.083333 / 0.053996` | `0 / 0.260706 / 0.199140` | `1 / 1 / 0.627523` | `0.166667 / 0.527778 / 0.360341` |
| Qwen 12f | `0.333333 / 0.431159 / 0.267789` | `0 / 0 / 0.026693` | `0 / 0.278022 / 0.234929` | `1 / 1 / 0.627523` | `0.166667 / 0.444444 / 0.273388` |

The transcript arm's aggregate improvement comes primarily from speech.
None of the four conditions produced an exact match in the non-speech
sound/music stratum, so Whisper text is not a substitute for native
non-speech audio understanding. The spatial stratum was already saturated by
both visual Qwen conditions, so this run provides no evidence that global
tiling is worth its cost.

## Matched ablations

### Twenty-four frames versus 12 frames

Treatment minus baseline over the same 30 examples:

| Metric | Paired delta | Exploratory 95% interval |
|---|---:|---:|
| Normalized exact match | `+0.000000` | `[0.000000, 0.000000]` |
| Token F1 | `+0.038083` | `[-0.006003, +0.095388]` |
| METEOR | `+0.019465` | `[-0.022226, +0.076680]` |

Exact-match wins/losses were `0 / 0`. More frames slightly improved dialogue,
sound/music, and visual-control overlap, but temporal/action F1 changed by
`-0.017316` and METEOR by `-0.035789`. The matched result does not support
doubling frames globally; it supports a future targeted temporal-routing
experiment.

### Whisper transcript versus visual-only Qwen

Treatment minus baseline over the same 30 examples:

| Metric | Paired delta | Exploratory 95% interval |
|---|---:|---:|
| Normalized exact match | `+0.066667` | `[0.000000, +0.166667]` |
| Token F1 | `+0.076070` | `[+0.003251, +0.167495]` |
| METEOR | `+0.052891` | `[-0.024898, +0.139180]` |

Exact-match wins/losses were `2 / 0`. The five stratum deltas
(`EM / F1 / METEOR`) were:

| Stratum | Transcript minus visual-only |
|---|---|
| Dialogue/speech | `+0.333333 / +0.268841 / +0.285004` |
| Non-speech sound/music | `0 / +0.166667 / +0.109165` |
| Temporal/action | `0 / -0.055159 / -0.026482` |
| Spatial/OCR/color | `0 / 0 / 0` |
| Visual controls | `0 / 0 / -0.103233` |

Because the original four-audio promotion gate had no visual-control examples,
protocol v2 predeclared this condition as `explore`, not `promote`. The clean
rerun reproduced the material temporal and visual-control regressions that
make universal transcript attachment unsafe.

## Whisper quality outcome

Whisper large-v3 completed 30/30 ASR requests on ROCm/BF16. The label-blind
quality policy accepted 29 transcripts and rejected one repetition
hallucination. The rejected `EMI_anpHII4` output contained 284 words
(`9.462941` words/second), above the frozen five-words/second threshold; its
downstream Qwen request used a null transcript. The policy also rejects text
with fewer than 80% Latin alphabetic characters. No reference answer was used
to accept, reject, clean, or select a transcript.

Whisper row inference was 9.068 / 12.412 / 37.713 seconds
(minimum / median / maximum), outer wall time was 8.78 minutes, and peak
allocation was 4.479 GiB.

## Leakage-resistant protocol and reproducibility

Protocol v2 used a strict two-phase scoring boundary:

1. A network-disabled, read-only **freeze container** could see only the staged
   runtime source, label-free configs, exact media/frame inputs, and the run
   directory. It did not mount the reference manifest or annotation files.
   It selected and froze all 120 prediction rows.
2. A separate **evaluation container** then mounted the exact reference
   manifest and 30 annotation files read-only. It treated the phase-one
   selected prediction and freeze bytes as immutable inputs and failed if
   either changed.

`evaluation.json` records
`references_loaded_only_after_prediction_freeze=true`. Repeating evaluation
reproduced the scored-prediction and evaluation files byte-for-byte.

Every inference row is bound to the exact request, current evidence manifest,
pre-launch decision, staged source tree, immutable model revision, and
digest-pinned image. Qwen live attestation observed 723/723 parameters on
`cuda:0` in BF16, three buffers on `cuda:0` in FP32, and HIP
`7.2.53211`. MiniCPM and Whisper independently recorded
`cuda:0 (ROCm)`/`bfloat16`. Competing GPU-process audits and the global serial
lock ran at launch/checkpoints.

### Model and image identities

| Role | Model and immutable revision | Digest-pinned image |
|---|---|---|
| Qwen 12f, 24f, and transcript downstream | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5` |
| Whisper ASR | `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1` | same ROCm image above |
| MiniCPM visual-only | `openbmb/MiniCPM-o-4_5@1f761131fa83f5ed3cd6f2f22b225c4501d154fa` | `localhost/trec-vqa-gate-minicpm@sha256:9fd8ffb14c39440b6a036781315558ae5ac6e93893582fff8d9eceaf8ec7a12a` |
| Scorer | NLTK `3.9.2` | `localhost/trec-vqa-gate-qwen-omni@sha256:effff2801d7677aa4894a641adc5793d6e5072a8cec71817bf925574862c88de` |

The scorer verified punkt-tab English tree SHA-256
`6073f4561d752e90f3b02bbb80a955100eb90b7f62c5fcfe17c86e8030909522`
and WordNet ZIP SHA-256
`6c4365b1018d44f39b284f9a5e44766548b18db9c436c9cfc1b7116230e35711`.

### Exact artifact hashes

| Artifact or identity | SHA-256 |
|---|---|
| Gate selection | `099dedfa4d74aa7ed1c135a013c8b342a62e422a6692934507c94839a2803f69` |
| Label-free inference config | `12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499` |
| Protocol-v2 decision config | `a279ceedb5b3e56597e16e9945585014c3db89747f075079917b79fb9bbb4249` |
| Bound source tree | `c281fbe28c883a3d3dbc9c761426e44603ffe40d4116a94a37cb99c2401b354d` |
| Staged runtime-source manifest | `c7fd797537654a694e9afc1412390d2c2e13944f655113114be3c29a02b63dd5` |
| `preflight.json` | `310e04c18358fbd32d1c4213c188d7b1eaf07a4068a80918eb0da193f3c992d2` |
| `promotion-decisions.jsonl` | `42c78918676e478679dad0198d4669abb45f91758adec694d57574880709a37b` |
| VQA attempts / selected predictions | `8aa8c76e809402a97e32277e0846b1cdae033c9727535717724b1be0b0235879` |
| `whisper-attempts.jsonl` | `f893f3f9a15bc0b113cb1b1ee4db0845df13b798c6a99c18caf0bdd2a0cfbefa` |
| `transcript-sources.json` | `4cf2c01d8d04e9898a6350b52f9abf61a3a5c208c3864bcc6df59415a40190cd` |
| Scored predictions | `c95669135b29551ef407e2889ac5e203053e944b00a0968cbe20da1fdfd0be7a` |
| Final `evaluation.json` | `f610dbd3d468f3b72074c7ace38dec95176c7f47fa213b5ce374d0a9429ad9c0` |

## Why protocol v1 is superseded

The earlier v1 numbers were reproducible but not protocol-valid: its inference
containers mounted a dataset root that contained labels, preparation opened
annotation content, and Qwen's saved ROCm/BF16 strings were configured rather
than live-attested. That run could not prove filesystem/process label
isolation or actual Qwen execution identity. Protocol v2 reused no v1
prediction: it reran all four conditions and Whisper under minimal mounts,
staged-source binding, live device/dtype attestation, outer wall timing, and
the two-container scoring boundary above. V1 remains immutable historical
audit evidence only; all **frozen-30 comparative conclusions** in this report
use v2. The explicitly labeled pre-v2 diagnostics below still inform which
future conditions are worth retesting.

### Earlier visual/native-AV screening is diagnostic only

The following conditions were screened or stopped before the clean frozen-30
rerun. They are not mixed into the four-condition protocol-v2 ranking.
Completed gate metrics are exact pre-v2 diagnostic values; `—` means no
quality run was launched.

| Pre-v2 condition | Evidence | EM | Token F1 | METEOR | Mean s / peak GiB | Stop decision |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B, 12f | 10/10 gate | 0.300 | 0.493 | 0.333 | 48.18 / 19.68 | Redundant with cheaper 4B/24f; no exclusive exact win |
| Gemma4-12B, 12f | 10/10 gate | 0.200 | 0.405 | 0.372 | 16.09 / 25.66 | Below leaders; no complement |
| InternVL3.5-4B | 10/10 gate | 0.200 | 0.315 | 0.241 | 14.53 / 22.44 | Below leaders; no complement |
| Molmo2-4B | 10/10 gate | 0.100 | 0.149 | 0.184 | 0.87 / 9.88 | Below leaders; no complement |
| MiniCPM-o 4.5, audio on | 10/10 gate | 0.300 | 0.471 | 0.283 | 10.35 / 19.82 | Versus audio-off: EM `0`, F1 `-0.115`, METEOR `-0.131` |
| Qwen2.5-Omni-3B, audio off | 10/10 gate | 0.300 | 0.467 | 0.287 | 398.73 / 75.68 | Exceeded cost and memory bounds |
| Qwen2.5-Omni-3B, audio on | 10/10 gate | 0.300 | 0.488 | 0.273 | 415.18 / 80.35 | Versus off: EM `0`, F1 `+0.021`, METEOR `-0.015`; exceeded bounds |
| Qwen2.5-Omni-7B, off/on | One off + one on probe (2/2 total) | 0 / 0 | 0.154 / 0.154 | 0.132 / 0.045 | 74.33 / 20.57; 29.07 / 21.10 | Both answers wrong; no probe quality signal |
| Baichuan-Omni-1d5 | Stopped before acquisition/probe | — | — | — | — | Additional unpublished community-license text made local clearance ambiguous |
| Qwen3-Omni-30B-A3B-Thinking | Stopped before acquisition/probe | — | — | — | 63.44 GB official shards | Larger-machine only: local runtime lacked the official FlashAttention-2 memory path, so safe headroom was not established |

The Baichuan and Qwen3-Omni rows are pre-launch safety decisions, not negative
quality findings. In particular, no local answer-quality claim is made for
either model.

Two other transcript models were stopped on the four-audio pre-v2 diagnostic:
Whisper large-v3-turbo reached downstream F1 `0.321` (paired gain `+0.071`),
and SeamlessM4T-v2-large reached `0.333` (paired gain `+0.083`). Both were
inferior to the large-v3 transcript gain used for the exploratory v2 arm.

### Earlier ASR screening is diagnostic only

Source-model completion and transcript acceptance are separate from downstream
Qwen completion: rejected or unavailable transcripts became null, while the
matched visual request still ran. METEOR was not computed for this diagnostic.

| Pre-v2 path | Source generation | Quality accepted | Downstream Qwen | EM / F1 / METEOR | Mean inference / peak |
|---|---|---:|---:|---|---:|
| Voxtral transcription + Qwen | 3 passed; 1 stopped before launch because source language was not validated | 3/4 | 4/4 | `0.250 / 0.250 / unavailable` | 92.050 s downstream / 10.68 GiB |
| Voxtral translation + Qwen | 4/4 generated | 2/4 | 4/4 | `0.250 / 0.250 / unavailable` | 9.999 s downstream / 10.66 GiB |
| Qwen3-ASR + Qwen | 3 passed; 1 failed with empty output | 3/4 | 4/4 | `0.250 / 0.250 / unavailable` | 9.978 s downstream / 10.67 GiB |
| Voxtral direct audio QA | 4/4 diagnostic outputs | not applicable | not applicable | `0.000 / 0.050 / unavailable` | 4.880 s direct / 9.65 GiB |

None improved the matched downstream answers. Any renewed comparison must be
rerun under protocol v2 with a complete matched control and pinned scoring;
the unavailable METEOR values must not be inferred from EM/F1. Voxtral
translation's two rejected rows were a 128-token repetition hallucination and
non-transcript clarification boilerplate. Qwen3-ASR's three accepted rows
included one mechanically accepted but semantically questionable song
transcript; its accepted French row alone used a separately frozen
SeamlessM4T-v2 text-translation step. The fourth Qwen3-ASR row failed empty and
therefore supplied null downstream text.

## Recommended stack and next experiments

For a first larger benchmark:

1. **Fast screening baseline:** MiniCPM-o 4.5 with audio off. It is the best
   METEOR result and the only condition with consistently short local latency.
2. **Accuracy baseline:** Qwen3.5-4B with 12 frames. Add Whisper large-v3 only
   as a separately reported exploratory arm until a label-blind
   speech/question router passes temporal and visual-control guards.
3. **Frame policy:** keep 12 frames globally. Test 24 frames only behind a
   predeclared temporal-question route; the global 24-frame treatment did not
   improve exact coverage.
4. **Spatial policy:** test selective high-resolution crops/2×2 tiles only on
   genuinely small OCR/object cases. This panel's spatial questions were
   already saturated, so global tiling has no measured justification.

Prioritized experiments after the baseline:

1. **True native-audio-on matched condition:** run audio-off versus audio-on
   with identical visual evidence on a modern native-AV model and retain
   separate speech and non-speech strata. This priority is partly informed by
   the explicitly pre-v2 ten-row MiniCPM audio-on diagnostic, which was
   negative; the new test must therefore be controlled, not assumed to help.
2. **Transcript routing:** attach Whisper only when a label-blind detector
   finds speech or the question asks about dialogue; predeclare temporal and
   visual-control non-regression thresholds.
3. **Non-speech audio model:** test an audio encoder/classifier for sound and
   music, where every current condition had zero exact matches.
4. **Different model families:** rerun current non-Qwen families such as
   MiniCPM, InternVL, Molmo, and Gemma under the same protocol-v2 gate when
   newer checkpoints or materially different adapters are available. Their
   lower rankings above are pre-v2 screening evidence, not frozen-30 v2
   comparisons.
5. **Larger models on a larger host:** test a larger native-AV model such as
   Qwen3-Omni-30B-A3B and a larger non-Qwen VLM as diversity checks. These are
   future conditions; this recommendation incorporates the pre-v2 local
   headroom stop decision, not model-quality evidence.

## Honest 100-video local ETA

Linear extrapolation of measured protocol-v2 **outer wall time** on this host:

| Recipe | 30-video measured wall | 100-video estimate |
|---|---:|---:|
| MiniCPM audio-off | 4.50 min | **15.0 min** |
| Qwen 12f | 35.48 min | **118.3 min** |
| Qwen 24f | 36.64 min | **122.1 min** |
| Whisper ASR only | 8.78 min | **29.3 min** |
| Qwen 12f + Whisper, sequential | 44.46 min | **148.2 min (2 h 28 min)** |

These estimates include observed container/model/orchestration overhead but
assume the 100 videos have a similar mix of Qwen long-generation cases. A
practical budget is about 15–20 minutes for MiniCPM and 2.5–3 hours for the
Whisper+Qwen arm. Persistent batching could reduce overhead; a different
content mix could increase or decrease the Qwen tail.

## Reproducible evidence

Canonical ignored evidence is under
`data/vqa/benchmarks/runs/native-audio-benchmark-v2/`. The key files are
`preflight.json`, `promotion-decisions.jsonl`, `attempts.jsonl`,
`whisper-attempts.jsonl`, `transcript-sources.json`,
`selected-predictions.jsonl`, `selected-predictions-freeze.json`,
`scored-predictions.jsonl`, and `evaluation.json`.
