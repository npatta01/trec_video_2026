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
tested condition**. A clean r4 Stage 1 audit found the frozen preferred
evidence represented in oracle-hybrid-12 for `10/10` selected rows. The
completed matched Stage 2 test did not turn that stronger representation
into better answers: treatment scored `1C/5P/4W` versus control at
`2C/4P/4W`, with one improvement, two regressions, seven ties, and
ordinal gain `-1`. The answer gate failed, so uniform-12 remains the
operational default. A post-result frozen-timestamp audit then found
that uniform-12 already covered every required interval in `6/10` rows
and at least one interval in `9/10`. The matched test therefore had
limited selection headroom and does not by itself isolate downstream
evidence use as the failure mechanism.

<div class="theme-decision-strip"
aria-label="Visual-evidence decisions">

- **Use now:** Qwen3.5-4B with 12 uniformly sampled frames.
- **Do not default:** global uniform-24, the tested directed/burst
  heuristics, or oracle-hybrid-12 under the matched r4 condition.
- **Still possible:** learned, hybrid, shot/event, OCR, or
  coarse-to-fine retrieval.
- **Next test:** prospectively freeze a selection-sensitive panel where
  uniform-12 demonstrably misses required evidence before testing
  another selector or answering prompt.

</div>

## Evidence at a glance

- **Strongest evidence:** matched uniform-12 versus uniform-24 on
  frozen-30 v2.
- **Diagnostic evidence:** ten selected temporal rows for directed-12
  and two rows for dense-burst-12; the historical primary
  annotation-agreement stop remains separate from the verified r4
  ten-row representation audit.
- **Matched answer evidence:** a completed 10-pair, 20-prediction
  comparison found oracle-hybrid-12 slightly worse than uniform-12 under
  the frozen gate.
- **Selection-headroom audit:** uniform-12 covered all required
  intervals in `6/10` rows, some but not all in `3/10`, and none in
  `1/10`; it covered `12/17` required intervals overall.
- **Audit gaps:** no tracked general-workflow caller for directed-12;
  the exact dense-burst implementation is absent.
- **Condition boundary:** none of these results rejects smarter visual
  sampling or retrieval as a research branch.

## What has been tried

The [2026-08-10 full-panel AV timeline and scene-selection
diagnostic](../experiments/2026-08-10-vqa-plaintext-full-panel.qmd) is
the canonical dated record for the complete 475-video event-map pass. It
completed 474/475 maps and 462/475 selector answers, but was **Not
promoted under the tested condition**: the failed rows had
answer-relevant evidence, which shifts the next test toward multi-scene
context and downstream answer use rather than another one-shot selector.

### Gemini native-AV event-map canary

- **Reason for testing:** determine whether one frontier
  native-video/audio pass can understand the question, map key events
  and state transitions, localize supporting spans, and produce one
  grounded top-1 answer.
- **Tested condition and panel:** two frozen VQA-2025 training examples,
  one sound and one temporal, with one Gemini content inference per
  video and no Whisper, independent audio check, or candidate-diversity
  pass.
- **Observed result:** both paid response bodies contained
  strict-schema-valid maps after deterministic exact-fence parsing:
  seven events and 15 selected timestamps for \$0.063730. The temporal
  map localized the core sequence but the answer added one detail
  unsupported by the inspected frames. The sound answer was too generic
  relative to the post-freeze reference. Two-row mean METEOR was 0.2532,
  BERTScore-F1 0.8949, STS 0.4392, and semantic composite 0.5291;
  single-answer NDCG was explicitly degenerate.
- **Current condition-bound status:** **Diagnostic signal; Not promoted
  under the tested condition.** Gemini native-AV mapping is
  operationally viable, but the current answer-to-evidence contract is
  not ready for the frozen 35-row panel.
- **What this rules out now:** interpreting the earlier MKV/schema
  failures as proof that Gemini or event maps do not work, and scaling
  the unchanged contract merely because calls are inexpensive.
- **What it does not rule out:** a one-pass contract with clause-level
  event/span support, audio-specific attributes, later multi-stage
  evidence acquisition, or a matched direct-answer-only control.
- **Possible defect, audit gap, or uncertainty:** **Confirmed
  defect**—the preflight freeze omitted raw-file hashes for
  failed-but-billable responses; this is now fixed. **Uncertainty**—two
  selected rows, provider audio use is not independently attested, and
  the private audit packet is valid but not reviewer-frozen.
- **Reconsider when:** a tightened contract on unused videos across the
  remaining categories produces schema-valid maps with supported answer
  claims and preserves label-critical specificity.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

Original MKV streams were losslessly copied to WebM and verified through
codec, packet-hash, duration, file-hash, and uploaded-byte bindings.
Native provider schema translation was unreliable, so the working route
uses prompt-JSON, strict local validation, and no semantic repair. Sol,
Fable, and Gemini Pro independently reviewed the same sanitized packet
and agreed that the primary failure is answer–evidence decoupling rather
than failed map localization. Fable noted that a merely shorter answer
would not solve both observed errors: one answer embellished while the
other lost specificity. See the [dated evidence
record](../experiments/2026-08-11-frontier-video-map-diagnostic.qmd).
</details>

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

### r4 required-moment Stage 1 representation audit

- **Reason for testing:** determine whether fixed-budget oracle frame
  candidates could represent the frozen preferred evidence before any
  model answer comparison.
- **Tested condition and panel:** a frame-first then full-video
  independent audit of all ten selected temporal development rows.
  Oracle-hybrid-12 uses six global and six local frames; oracle-local-12
  uses 12 local frames.
- **Observed result:** r4 had `10/10` labelable and
  still-frame-representable rows, zero incompatible/uncertain rows, and
  zero repair replacements. Hybrid was visible, relationship-preserving,
  answer-sufficient, and a final pass for `10/10`; local-only was `9/10`
  for each measure. The independently reviewed sanitized extraction
  aggregate (`r4-stage1-extraction-summary-sanitized.json`, SHA-256
  `a5d069160df74a315a290767932c68c13af72d01c80d7592157f24dce3d91bfe`)
  reports 360 real frames and 120 unique PNGs per condition, with all 60
  hybrid local frames and all 120 local-only frames inside frozen target
  intervals.
- **Current condition-bound status:** **Verified** Stage 1
  representation evidence. The hybrid-first rule selected
  `oracle-hybrid-12` for the matched Stage 2 test recorded below;
  uniform-12 remains the **Current default**.
- **What this rules out now:** advancing oracle-local-12 as the selected
  candidate for this exact Stage 2 comparison.
- **What it does not rule out:** whether another model or prompt uses
  the represented evidence well enough to improve answers, whether the
  oracle condition is deployable, or whether retrieval is the dominant
  end-to-end bottleneck.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—this
  is a selected ten-row representation audit, not a model-answer or
  generalization result.
- **Reconsider when:** a new matched model, prompt, or deployable
  selector passes a predeclared paired improvement and non-regression
  gate.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The clean r4 evidence bundle is the sole evidence-of-record. Independent
raw auditor agreement was `10/10` representability, `33/34` interval
visibility, `40/42` relationship preservation, and `19/20` answer
sufficiency. The candidate table is intentionally the primary visual:
with two candidates and four exact audit measures, a chart would add no
useful relationship. r1 (incomplete full-audit packet contract), r2
(illegal reviewer order), and r3 (unbound Python bytecode before the
correct freeze) are invalid operational attempts, not scientific rows or
evidence for this claim. No GPU/model inference, predictions,
references, grades, or transcript work had run when this Stage 1 bundle
froze; the later matched Stage 2 result is recorded below. Before report
edits, full r4 verification passed at the frozen original worktree path
with 1,162 sidecars validated and forbidden artifacts absent. It now
correctly refuses this dirty reporting worktree; an alternate checkout
cannot reproduce the path-bound protocol because it binds absolute
dataset, prior-bundle, and script paths. This is a verifier portability
limitation, not r4 artifact drift. See the [dated evidence
record](../experiments/2026-07-30-temporal-required-moment-oracle.md).
</details>

### Matched r4 oracle-hybrid-12 inference comparison

- **Reason for testing:** determine whether the r4-selected
  relevant-scene representation improves model answers at the same
  12-frame budget.
- **Tested condition and panel:** matched Qwen3.5-4B uniform-12 control
  versus oracle-hybrid-12 treatment on the frozen ten-row r4 cohort:
  `10/10` pairs and `20/20` predictions completed. The model, prompt,
  decoding, and visual budget were fixed; no transcript was created or
  loaded.
- **Observed result:** control was `2C/4P/4W` with ordinal score `8`;
  treatment was `1C/5P/4W` with ordinal score `7`. Transitions were one
  `C→C`, one `C→P`, three `P→P`, one `P→W`, one `W→P`, and three `W→W`:
  one improvement, two regressions, and seven ties. Treatment ordinal
  gain was `-1`. A later frozen-timestamp coverage audit found that
  uniform-12 already hit all required intervals in `6/10` rows, hit some
  but not all in `3/10`, and hit none in `1/10`. It covered `12/17`
  intervals with 15 of its 120 frames.
- **Current condition-bound status:** **Verified; Not promoted under the
  tested condition.** The predeclared answer gate required at least two
  improvements, zero regressions, and ordinal gain of at least two; all
  three requirements failed.
- **What this rules out now:** promoting oracle-hybrid-12 as the default
  visual condition for this model, prompt, selected panel, and fixed
  budget.
- **What it does not rule out:** a deployable learned selector, a
  different model or answering prompt, another compact evidence
  representation, or an effect on a broader panel—especially one
  selected prospectively for genuine uniform-12 evidence misses.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—this
  is a selected ten-row 2025 training panel and the oracle condition is
  not deployable. Because uniform-12 already covered all required
  intervals in a majority of rows, the comparison mostly tested evidence
  concentration rather than relevant versus missing evidence. Interval
  contact is also a timestamp proxy, not a fresh human judgment of
  uniform-frame answer sufficiency.
- **Reconsider when:** a new matched condition passes the same
  improvement, non-regression, and ordinal-gain gate on frozen evidence.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The sealed Task 2 run remained unchanged. Its controller
(`e432f4b8c642321f46a92481e79a3e415185a4a5d357d833d0a92e7e22cb22fb`) and
complete prediction freeze
(`111e1215da648bd7fb9e02cb412ff45ce174dc7d0a2dc5019cf8ca11c301f393`)
were reverified through a reference-denied, identity-preserving
namespace. All new grading work stayed in a private external workspace.
Two independent condition-blind graders agreed on `19/20` candidate
labels; a fresh third reviewer adjudicated the one disagreement before
the one-time unblind. The unblind-complete record is
`cac66433bb9db9bb653795c878656075aa7c03e8f390c943c68f497624bbe17b`. The
grading packet omitted condition and row identities, paths, provenance,
and the secret map. No transcript was mounted, opened, or used. An
independent final review passed 217 checks across sidecars, lineage, the
packet/secret bijection, grade coverage, adjudication, unblinding, and
arithmetic. Its deliberately restricted scope excluded independently
rehashing two upstream source files; it validated every in-scope sidecar
and cross-phase link, while the earlier isolation review supplied those
pinned hashes. These controls support the narrow answer-quality
comparison; they do not make the oracle condition deployable or the
ten-row result general.

The read-only selection-headroom audit bound the frozen r4 frame plans
(`ee21e7eb9e53026b958c9a32c164a406db0d5b0fa7d1d146afadbdafccd18d2e`),
their extraction manifest
(`212a41bfff3f35af3c7fc571c9b7e269b84da11f20821c8ad8ae7234aa7558a5`),
and the exact Stage 2 uniform manifest
(`e8132e81d261729be2df08de410cdc8e189d4d841c2ac5ffb9296c1123c9c559`).
Its sanitized aggregate record is
`933a01cf5227c089a6459c3ac22ed67f134fd3bb522fa6cd0cbf4e2ec370755f`. A
separate jq calculation reproduced every aggregate. The audit opened no
predictions, references, grades, model material, or transcript and
launched no model run.
</details>

### Historical primary annotation-agreement stop

- **Reason for testing:** separate missing-moment retrieval from
  downstream answering by labeling the minimal visual evidence needed
  for each selected temporal row.
- **Tested condition and panel:** three blind, de novo full-video
  annotation passes on ten selected temporal development rows from
  labeled 2025 training data; the planned uniform-12/oracle-hybrid-12
  comparison was conditional on primary reviewer agreement.
- **Observed result:** the two primary reviewers matched eligibility on
  `9/10` rows and structure on `8/10`, exactly meeting the structural
  threshold, but median tIoU was `0.4791666666666665` against the
  required `0.50`. Six rows were flagged. The agreement gate failed and
  all later stages were not run.
- **Current condition-bound status:** **Diagnostic signal**; evidence is
  **Incomplete** for the intended oracle comparison, and no selector was
  promoted. Uniform-12 remains the **Current default**.
- **What this rules out now:** using these annotations for
  boundary-sensitive required-moment recall or oracle-frame answer
  claims.
- **What it does not rule out:** whether oracle or deployable selectors
  can improve retrieval, whether retrieval is the dominant bottleneck,
  or whether transcript work should take priority.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the
  selected ten-row panel and narrowly failed agreement gate diagnose
  this rubric and process, not every required-moment annotation design.
- **Reconsider when:** a calibrated rubric and three fresh blind
  annotations pass the same predeclared structural and median-tIoU gate.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

All three reviewers annotated `10/10` rows independently. The A/B files
froze before their comparison. Reviewer C remained blind while A/B were
compared, then froze its de novo file before learning the result; it was
supplemental and was not used to rescue the failed gate. Reviewer A
marked `9/10` rows eligible; reviewers B and C marked `10/10`. A/B
eligibility matched on `9/10`; eight eligible rows structurally matched.
The tIoU interquartile range was `0.49476190476190574`. Per the
preregistered stop rule, adjudication, actual-PTS inventory and frame
extraction, selector recall, local ROCm inference, references,
predictions, grading, and transcript work were not run.

See the [dated evidence
record](../experiments/2026-07-30-temporal-required-moment-oracle.md).

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
- The r4 audit shows that oracle-hybrid-12 can represent the frozen
  preferred evidence for all ten selected rows, whereas oracle-local-12
  did so for nine.
- The matched r4 Stage 2 comparison shows that the represented relevant
  scenes did not improve answers under this model and prompt; treatment
  was slightly worse and failed the predeclared gate.
- The post-result overlap audit shows why that negative should not be
  read as a clean evidence-use diagnosis: uniform-12 already covered
  every required interval in six rows and at least one in nine.
- The historical required-moment diagnostic shows that structural
  agreement alone was not enough for its original boundary-sensitive
  selector-recall branch; that agreement stop remains historical and
  separate from r4.

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
- The failed r4 answer comparison does not show that all relevant-scene
  selectors are ineffective or that this model cannot use relevant
  scenes. Limited selection headroom means it does not isolate retrieval
  from evidence use without a panel where uniform-12 actually misses
  required evidence.

<span id="possible-bugs-confounds-and-audit-gaps"
class="legacy-fragment-alias" aria-hidden="true"></span>

### Confirmed defects, audit gaps, and uncertainties

- **Audit gap:** no tracked general-workflow caller establishes how the
  directed selector was attached to a general workflow. This is not a
  confirmed execution bug.
- **Audit gap:** dense-burst-12’s exact implementation and schedule are
  absent from tracked source; only its two-row aggregate is recoverable.
- **Uncertainty:** the matched r4 answer result covers one oracle
  condition, model, prompt, and selected ten-row panel. It does not
  estimate a deployable selector’s recall or broader oracle-frame answer
  quality.
- **Uncertainty:** interval overlap shows timestamp coverage, not
  whether the exact uniform frames preserve every visual relationship or
  are independently answer-sufficient.
- **Uncertainty:** manual spatial crops are targeted and non-scalable,
  so their negative result cannot rank global tiling or OCR routes.

## Open hypotheses and ideas

- The current ten-row panel mixes four rows with a measurable uniform-12
  interval-coverage gap and six rows without one, limiting causal
  diagnosis.
- Rows with a verified uniform miss may fail at retrieval; rows with
  full interval coverage may instead fail at visual representation,
  fusion, or answer selection.
- A fixed budget that combines broad context and query-local evidence
  may improve recall without the narrowness of the tested directed
  heuristic.
- Spatial detail may be worthwhile only for routed OCR or small-object
  cases.

## Cheapest valid next experiments

### Selection-sensitive relevant-scene panel

- **Hypothesis:** relevant-scene selection helps when uniform-12
  demonstrably misses at least one required interval, rather than when
  both arms already contact the needed moments.
- **Smallest valid panel or work item:** prospectively freeze a new
  ten-row temporal panel whose required evidence is
  still-frame-representable and for which uniform-12 misses at least one
  interval. Treat the four such rows in the current panel as post-hoc
  diagnostic seeds, not a promotion panel.
- **Matched controls and integrity safeguards:** select rows from frozen
  interval/frame evidence before any new prediction; keep model, prompt,
  decoding, seed, frame/pixel budget, and blind post-freeze grading
  fixed.
- **Decision or promotion rule:** retain a selector only if it produces
  at least two improvements, zero regressions, and ordinal gain of at
  least two.
- **Boundary on what a positive result would establish:** it would
  support the tested selector on uniform-miss temporal rows, not a
  global default or a deployable learned selector.

### Conditional fixed-evidence answering-use diagnostic

- **Hypothesis:** the failure is partly in evidence use or answer
  extraction, so explicitly eliciting the decisive visual observation
  before the final answer can improve oracle-hybrid-12 answers without
  changing its frames.
- **Smallest valid panel or work item:** run only after a
  selection-sensitive panel still shows failures despite verified
  treatment coverage; compare the current prompt with an
  observation-then-answer prompt on those frozen frames.
- **Matched controls and integrity safeguards:** keep model revision,
  decoding, seed, frames, pixel budget, and blind post-freeze grading
  fixed; do not expose references or prior grades to the model.
- **Decision or promotion rule:** retain the new prompt only if it
  produces at least two improvements, zero regressions, and ordinal gain
  of at least two.
- **Boundary on what a positive result would establish:** it would
  implicate evidence use or answer elicitation for this panel, not prove
  a deployable selector or general temporal accuracy.

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
