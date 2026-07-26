# Visual and Temporal Evidence


<span id="back-to-research-themes" class="legacy-fragment-alias"
aria-hidden="true"></span>
<nav class="status-breadcrumb" aria-label="Breadcrumb">

<a href="index.md">Start Here</a> <span aria-hidden="true">/</span>
<a href="experiment-ledger.md">Research Themes</a>
<span aria-hidden="true">/</span> <span aria-current="page">Visual and
Temporal Evidence</span>
</nav>

<nav class="theme-local-nav" aria-label="On this theme page">

<a href="#current-working-answer">Working answer</a>
<a href="#what-has-been-tried">Tried</a>
<a href="#known-failures-and-cases">Failures</a>
<a href="#cheapest-valid-next-experiments">Next tests</a>
</nav>

<span id="why-this-theme-matters" class="legacy-fragment-alias"
aria-hidden="true"></span>

## New to this theme?

VQA fails either because needed evidence was not selected (retrieval) or
because selected evidence was not used well (answering). More frames, a
local burst, or crops can cost compute and still miss the required
moment, so the fixed evidence budget and selection mechanism must be
visible.

For example, **uniform-12** spends a 12-frame budget across the whole
clip, while a directed selector spends some of the same budget near a
question-chosen time. If neither selects the brief decisive action, that
is a retrieval problem; if both select it and the answer is still wrong,
the likely bottleneck is downstream.

<span id="technical-summary" class="legacy-fragment-alias"
aria-hidden="true"></span>

## Current working answer

Use uniform-12 as the current default: it takes midpoint frames from 12
equal time bins. Uniform-24 applies the same whole-video coverage with
twice the frame proxy. Do not make the tested directed-12 or
dense-burst-12 diagnostic a default; each is **Not promoted under the
tested condition**.

<div class="theme-decision-strip"
aria-label="Visual-evidence decisions">

- **Use now:** Qwen3.5-4B with 12 uniformly sampled frames.
- **Do not default:** global uniform-24 or the tested directed/burst
  heuristics.
- **Still possible:** learned, hybrid, shot/event, OCR, or
  coarse-to-fine retrieval.
- **Next test:** required-moment labels, selector recall, and
  oracle-frame answering at one fixed budget.

</div>

## Evidence at a glance

- **Strongest evidence:** matched uniform-12 versus uniform-24 on
  frozen-30 v2.
- **Diagnostic evidence:** ten selected temporal rows for directed-12
  and two rows for dense-burst-12.
- **Audit gaps:** no tracked general-workflow caller for directed-12;
  the exact dense-burst implementation is absent.
- **Condition boundary:** none of these results rejects smarter visual
  sampling or retrieval as a research branch.

## What has been tried

<a name="frozen-30-protocol-v2"></a>

### Frozen-30 protocol v2: global frame budget

- **Reason for testing:** learn whether twice the uniform evidence
  budget improved the global visual baseline.
- **Tested condition and panel:** matched Qwen3.5-4B uniform-12 versus
  uniform-24 across 30 labeled 2025 examples under protocol v2.
- **Observed result:** uniform-24 added no exact answers over uniform-12
  while increasing the visual evidence proxy.
- **Current condition-bound status:** uniform-12 remains the **Current
  default**; global uniform-24 is **Not promoted under the tested
  condition**.
- **What this rules out now:** making uniform-24 the global default for
  this model and protocol-v2 panel.
- **What it does not rule out:** a declared temporal-only route or
  another fixed-budget evidence design.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—30
  labeled 2025 rows do not establish a universal frame-budget effect.
- **Reconsider when:** a matched routed panel beats uniform-12 under
  predeclared controls and non-regression rules.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

Both arms use whole-video bin-midpoint coverage, not query-driven
retrieval. The comparison is the strongest global visual evidence here
but remains a 30-row 2025 training-data result, not an estimate of 2026
accuracy. See the [protocol-v2 report · Evidence of record · Evidence
report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md).
</details>

### Directed temporal sampling

Directed-12 retains broad context and reallocates the rest of its fixed
12-frame budget toward a keyword/anchor-selected region. It differs from
uniform-12, which uses midpoint frames from 12 equal time bins, and from
uniform-24, which doubles that same whole-video coverage.

- **Reason for testing:** test whether a question-directed allocation
  could beat uniform coverage without increasing the 12-frame budget.
- **Tested condition and panel:** ten temporal rows with identical model
  and pixel bounds; directed-12 versus uniform-12 and uniform-24.
- **Observed result:** directed-12 tied uniform-12 at `2C/5P/3W`;
  uniform-24 was `4C/3P/3W` in the same comparison.
- **Current condition-bound status:** **Not promoted under the tested
  condition**; it rejects this heuristic as the current default, not all
  question-aware sampling.
- **What this rules out now:** using this keyword/anchor directed-12
  heuristic as the default temporal selector.
- **What it does not rule out:** learned, query-only, hybrid, shot,
  event, or coarse-to-fine selection under a new controlled design.
- **Possible defect, audit gap, or uncertainty:** **Audit gap**—source
  and unit tests exist, but no tracked general-workflow caller proves
  the attachment path.
- **Reconsider when:** required-moment recall and oracle-frame answer
  quality show a predeclared improvement with matched controls.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The selector source and tests establish deterministic timestamps for a
supplied anchor, not required-evidence recall or answer relevance. The
tracked repository has no general-workflow caller for the selector: that
is an **Audit gap**, not a confirmed execution bug. The [temporal
experiment record · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/docs/experiments/temporal-sampling-issue-8.md),
[selector · Current code · Source code ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_experiments/temporal_sampling.py),
and [tests · Current code · Source code ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/tests/test_temporal_sampling.py)
document the narrow contract.
</details>

### Dense-burst sampling

Dense-burst-12 is only recoverable as a two-row aggregate: its exact
tracked implementation and schedule are absent. It is therefore a weakly
auditable diagnostic, not a general selector result.

- **Reason for testing:** ask whether a dense local burst could beat
  broad coverage on rows requiring short-lived evidence.
- **Tested condition and panel:** a two-row diagnostic comparing
  uniform-24 with dense-burst-12.
- **Observed result:** uniform-24 was `2C/0P/0W` versus burst
  `0C/1P/1W`, with one C→P and one C→W regression.
- **Current condition-bound status:** **Not promoted under the tested
  condition**; reject the tested diagnostic as a default.
- **What this rules out now:** making this tested dense-burst-12
  diagnostic the default local selector.
- **What it does not rule out:** another local schedule or selector
  evaluated with recoverable source, provenance, and a larger declared
  panel.
- **Possible defect, audit gap, or uncertainty:** **Audit gap**—the
  exact dense-burst implementation and schedule are absent from tracked
  source.
- **Reconsider when:** a fully tracked implementation passes a
  fixed-budget, label-isolated comparison with non-regression controls.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The two-row boundary means the evidence cannot estimate prevalence,
effect size, or performance of all local selectors. It also cannot
recover the exact burst schedule or launch path, an **Audit gap** rather
than proof of an execution bug. The aggregate’s recorded source digest
is `d4825b4530b181f1b4414552a80136e2f0e5a7ee64def23131d8c163acc458f1`;
the [closed issue \#16 · Issue/PR history ·
GitHub](https://github.com/npatta01/trec_video_2026/issues/16) records
the decision without exposing rows or raw artifacts.
</details>

### Spatial-crop diagnostic

- **Reason for testing:** inspect whether question-guided spatial detail
  could recover failures attributed to small visual evidence.
- **Tested condition and panel:** manual question-guided crop on a
  targeted spatial failure, alongside audio and translation diagnostics.
- **Observed result:** the crop added pixels but did not make the
  downstream answer correct.
- **Current condition-bound status:** **Diagnostic signal**; global
  tiling or crops are not promoted.
- **What this rules out now:** promoting this manual crop or global
  tiling from the targeted failure diagnostic.
- **What it does not rule out:** a category-routed spatial-detail or OCR
  policy with a declared crop budget.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—one
  manual, targeted crop cannot separate retrieval from fusion failure.
- **Reconsider when:** a matched target stratum and visual controls show
  a predeclared route benefit without regressions.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The crop was manually selected and not scale-ready. One targeted failure
cannot show that spatial detail is generally useless, nor distinguish a
selector failure from a fusion/reasoning failure. The original mixed
follow-up comprised exactly two audio/spatial failures, then one speech
failure; the speech and Seamless record now lives under [Speech and
Transcript
Evidence](research-speech-transcripts.md#seamless-diagnostic). See the
[audio and spatial follow-up · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/audio_spatial_followup_results_2026-07-21.md)
and the frozen-30 spatial policy discussion in the [protocol-v2 report ·
Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md).
</details>

## Known failures and cases

[CASE-004](failure-library.md#case-004) is a useful cross-theme warning:
a brief end-of-clip event can involve temporal retrieval, native sound,
and answer specificity at once. One selected case cannot identify which
bottleneck dominates without matched evidence.

### What the evidence supports

- Uniform-12 is the present simple visual default under the tested
  global frozen-30 conditions.
- The tested directed-12 heuristic did not earn default status on its
  ten-row temporal diagnostic.
- The tested dense-burst-12 diagnostic regressed on its two rows and
  should not be expanded as a default without a new controlled design.
- A manual crop can add spatial pixels without repairing an answer; that
  is a reason to route and diagnose detail needs rather than tile
  globally.

### What it does not rule out

- Global uniform-24’s result does not rule out a temporal-only route or
  a different evidence budget.
- Directed-12 does not rule out learned, query-only,
  half-uniform/half-query, shot-based, event-based, or coarse-to-fine
  selectors.
- Dense-burst-12 does not estimate prevalence, effect size, or all local
  selection schedules.
- The crop diagnostic does not rule out category-routed spatial detail
  or OCR.

<span id="possible-bugs-confounds-and-audit-gaps"
class="legacy-fragment-alias" aria-hidden="true"></span>

### Confirmed defects, audit gaps, and uncertainties

- **Audit gap:** no tracked general-workflow caller establishes how the
  directed selector was attached to a general workflow. This is not a
  confirmed execution bug.
- **Audit gap:** dense-burst-12’s exact implementation and schedule are
  absent from tracked source; only its two-row aggregate is recoverable.
- **Uncertainty:** directed-12’s ten selected temporal rows have no
  required-moment labels, selector-recall measure, or oracle-frame
  answer quality measure.
- **Uncertainty:** manual spatial crops are targeted and non-scalable,
  so their negative result cannot rank global tiling or OCR routes.

## Open hypotheses and ideas

- Some rows may fail because the required moment was never selected;
  others may have the moment but fail during fusion or answer selection.
- A fixed budget that combines broad context and query-local evidence
  may improve recall without the narrowness of the tested directed
  heuristic.
- Spatial detail may be worthwhile only for routed OCR or small-object
  cases.

## Cheapest valid next experiments

### Required moments, selector recall, and oracle frames

- **Hypothesis:** missing required moments, rather than answer reasoning
  alone, explain a measurable share of temporal failures.
- **Smallest valid panel or work item:** label required moment/span on a
  small frozen temporal panel, then score each selector’s recall and run
  an oracle-frame answer-quality condition.
- **Matched controls and integrity safeguards:** freeze labels before
  selector scoring; hold model, prompt, decoding, frame count, and pixel
  budget fixed; keep required-moment labels out of ordinary inference.
- **Decision or promotion rule:** advance a selector only if it improves
  recall and downstream answer quality without a predeclared control
  regression.
- **Boundary on what a positive result would establish:** it would
  separate retrieval from answering on that panel, not prove general
  temporal accuracy.

### Fixed-budget temporal selector comparison

- **Hypothesis:** a controlled query-aware policy can outperform uniform
  coverage at the same evidence budget.
- **Smallest valid panel or work item:** a frozen temporal panel
  comparing uniform, query-only, half-uniform/half-query, and
  coarse-to-fine selectors.
- **Matched controls and integrity safeguards:** identical model,
  prompt, decoding, frame/pixel budget, source version, and
  label-after-freeze scoring; record selector inputs and timestamps.
- **Decision or promotion rule:** promote only a condition that meets a
  predeclared paired improvement and non-regression threshold against
  uniform.
- **Boundary on what a positive result would establish:** it would
  support that one controlled selector on the declared panel, not every
  question-aware sampler.

### Category-routed spatial-detail/OCR diagnostic

- **Hypothesis:** spatial detail helps routed small-text or small-object
  cases more than it helps a global visual stream.
- **Smallest valid panel or work item:** a label-blind, declared
  OCR/spatial stratum plus matched visual-control cases.
- **Matched controls and integrity safeguards:** identical base
  frames/model/ prompt/decoding; one declared crop or detail budget;
  fixed routing rule; and frozen predictions before scoring.
- **Decision or promotion rule:** keep a route only if it improves its
  target stratum without exceeding a predeclared visual-control
  regression limit.
- **Boundary on what a positive result would establish:** it would
  support a category route, not global tiling or universal OCR benefit.

## Method, result, validity risks, and evidence

<details>

<summary>

How to audit this theme’s records
</summary>

The visible record summaries preserve the exact panel, budget, result,
limits, and reconsideration condition. Open each record’s native details
block for source links, selector provenance, and validity risks. Compare
selectors only at a fixed frame/pixel budget and with labels hidden from
selection.
</details>

## Related themes

- [Experiment Integrity and Evaluation](research-integrity.md) defines
  the freeze and audit boundaries for these comparisons.
- [Speech and Transcript Evidence](research-speech-transcripts.md)
  covers time-aligned language evidence that may need its own route.
- The [Code Map](code-map.md#remaining-gaps) records the
  general-workflow and routing gaps.
