# Temporal required-moment oracle: historical agreement stop and r4 Stage 1 representation result

Date: 2026-07-30

Primary theme: **Visual and Temporal Evidence**

Related theme: **Routing, Fusion, and Answer Selection**

Decision: **The clean r4 Stage 1 audit selected `oracle-hybrid-12` for the
predeclared matched Stage 2 inference test after it fully represented the
selected cohort. This is not a model-answer result: Stage 2 has not started,
and uniform-12 remains the operational visual default.**

## Technical summary

The r4 evidence bundle establishes a narrow Stage 1 extraction and
representation result on the selected ten-row temporal development panel.
`oracle-hybrid-12` passed visibility, relationship preservation,
answer-sufficiency, and final representation for all `10/10` rows.
`oracle-local-12` passed each measure for `9/10`. The predeclared all-cohort
rule therefore produced `select_oracle_hybrid_12` for a future matched Stage 2
inference test. The sanitized derived r4 extraction aggregate
`r4-stage1-extraction-summary-sanitized.json` (SHA-256
`a5d069160df74a315a290767932c68c13af72d01c80d7592157f24dce3d91bfe`),
independently reviewed against its immutable sources, reports 360 real frames
and 120 unique PNGs per candidate condition; all 60 hybrid local frames and
all 120 local-only frames fell within their frozen target intervals, with zero
outside.

This is **Verified** evidence for the r4 Stage 1 representation claim and a
selection decision, not evidence that a model answers better. No GPU or other
model inference, predictions, reference loading, grading, or transcript work
ran. Uniform-12 remains the operational default until a matched answer-quality
comparison is completed and passes its declared rule.

## Question, scope, and candidate definitions

- **Question:** on the selected ten-row temporal development panel, can two
  fixed 12-frame oracle candidates represent the adjudicated visual evidence
  well enough to select one for a future matched inference comparison?
- **Panel and data boundary:** all ten selected temporal development rows from
  labeled 2025 training data. This is a fixed, small, non-random diagnostic
  cohort; it does not estimate prevalence, general temporal accuracy, or 2026
  evaluation performance.
- **Control and candidates:** uniform-12 remains the operational baseline.
  `oracle-hybrid-12` combines six global frames with six local frames;
  `oracle-local-12` uses 12 local frames. The two oracle candidates are audit
  conditions, not a deployed default or an answer-quality comparison.
- **Measures and denominator:** each candidate is measured on all ten
  still-frame-representable rows for interval visibility, relationship
  preservation, answer sufficiency, and final pass. A final pass requires the
  preceding representation checks; the comparison never substitutes a
  favorable candidate-specific subset.

## r4 Stage 1 result: hybrid fully represented the selected cohort

The clean r4 scoreboard has `10/10` all rows, labelable rows, and
still-frame-representable rows; it has zero still-incompatible and zero
uncertain rows. The sanitized derived r4 extraction aggregate reports zero
repair replacements; the sanitized scoreboard independently exposes the
remaining exact aggregate values below. A table is more useful than a chart
because these are two conditions and four discrete audit lookup measures, not
a trend or a continuous relationship.

| Candidate | Visibility | Relations preserved | Answer-sufficient | Final pass | Repair replacements |
|---|---:|---:|---:|---:|---:|
| oracle-hybrid-12 | `10/10` | `10/10` | `10/10` | `10/10` | `0` |
| oracle-local-12 | `9/10` | `9/10` | `9/10` | `9/10` | `0` |

Independent-auditor raw agreement was `10/10` for representability, `33/34`
for interval visibility, `40/42` for relationship preservation, and `19/20`
for answer sufficiency. The extraction result and the representation result are
separate: the independently reviewed sanitized extraction aggregate reports
the material frames, while the sanitized-scoreboard table establishes candidate
representation on the ten-row denominator.

## Audit method and decision boundary

The audit used a frame-first review followed by a full-video relation review,
with independent auditor inputs frozen before the final adjudication. The
final audit checked whether sampled still frames visibly covered the frozen
preferred evidence, preserved the required relationship, and were sufficient
for answering. The predeclared hybrid-first rule selected
`oracle-hybrid-12` because it had a final pass for every eligible cohort row;
the local-only candidate did not. The completion denominator is `10/10` rows
for each candidate.

This result selects a candidate for a matched Stage 2 test; it does not
establish selector recall, end-to-end answer quality, causal benefit over
uniform-12, or a production routing rule. Stage 2/model answering is not yet
tested at this checkpoint.

## Evidence validity, limitations, and operational history

- **Canonical r4 claim:** under this frozen Stage 1 audit and selected panel,
  `oracle-hybrid-12` represented all ten rows and was selected for the next
  matched inference test.
- **Evidence validity:** **Verified** for the r4 extraction/representation and
  selection result. It is not an answer-quality result.
- **Decision disposition:** `select_oracle_hybrid_12` for the predeclared
  Stage 2 test; uniform-12 remains the **Current default** for operations.
- **Limitations and uncertainty:** the cohort is selected and small; the
  oracle candidates use adjudicated preferred intervals and are not deployable
  selectors. A static representation audit cannot prove model use of evidence.
- **Invalid operational attempts, not scientific evidence:** r1 used an
  incomplete full-audit packet contract; r2 froze full-audit work in an
  illegal reviewer order; and r3 acquired an unbound Python bytecode file
  during inspection before the correct full audit could freeze. They are
  retained as integrity lessons only. r4 is the sole evidence-of-record for
  this Stage 1 result.
- **Verifier portability limitation:** before these tracked report edits, the
  full r4 verifier passed at the frozen original worktree path with 1,162
  sidecars validated and forbidden artifacts absent. The verifier deliberately
  rejects this reporting worktree while it is dirty; a detached checkout at a
  different path instead reports protocol-core drift because its frozen
  protocol binds absolute dataset, prior-bundle, and orchestration-script
  paths. This is a verifier portability limitation, not evidence drift; frozen
  r4 artifacts, protocol, and script were not modified.

## Immutable r4 evidence

The private evidence bundle is identified by hashes rather than linked or
copied here:

- public/sanitized scoreboard file SHA-256:
  `355b65aab8f5b316b04a623dbce3baabdd300700414986cdf5e643251f656524`
- semantic scoreboard SHA-256:
  `5276e294e760d6d1da5a1fc0e5255dcb1c60d8e640d67b7c5a13ac64f72b6ff6`
- protocol SHA-256:
  `514c17d4cd56f05f07d6e0b80219d6edd5f7e53675ce3dfe4838618cf892758c`
- frozen orchestration script SHA-256:
  `a3efc6dac3f83d2db3e93c15e387f62019f6f6fe90a9877085db3187da48e310`
- frozen tracked revision:
  `d294b7c6e83523a885f84ffa2d682ae9524e39bd`
- sanitized derived extraction aggregate SHA-256:
  `a5d069160df74a315a290767932c68c13af72d01c80d7592157f24dce3d91bfe`

No private questions, row identifiers, labels, intervals, paths, frames,
answers, predictions, references, grades, transcripts, or logs are included
in this report. The sanitized scoreboard is the public representation source;
the separate sanitized extraction aggregate is independently reviewed and
bound by the hashes above.

## Historical primary annotation-agreement stop

The current annotation rubric and labeling process are not reliable enough to
support boundary-sensitive required-moment recall on this selected panel. Two
blind primary reviewers reached the preregistered structural threshold exactly
(`8/10` rows), but their median temporal intersection-over-union (tIoU) was
`0.4791666666666665`, below the required `0.50`. The protocol therefore stopped
before adjudication, decoded-frame planning, selector recall, or model
inference.

This is **Incomplete** evidence for the intended oracle comparison and a valid
**Diagnostic signal** about the labeling process. It does not show whether
oracle sampling would succeed, whether retrieval is the dominant bottleneck,
or whether transcript work should take priority.

### Original question, hypothesis, and falsifier

- **Question:** on a frozen temporal development panel, does a fixed-budget
  oracle-hybrid frame plan retrieve independently labeled required visual
  moments more reliably than fresh uniform-12, and does that evidence improve
  the pinned visual model's answers?
- **Hypothesis:** missing required visual moments explain a measurable share
  of temporal failures, so the oracle-hybrid plan would improve required-moment
  recall and downstream answer quality without regressions.
- **Falsifier and prerequisite:** the branch would not advance if primary
  annotation agreement failed, the CPU recall gate failed, or the paired
  answer comparison missed its predeclared improvement and non-regression
  rule. The first prerequisite failed, so later falsifiers were not evaluated.

### Original panel and intended comparison

The frozen panel contains ten selected temporal development rows drawn from
labeled 2025 training data. It is a deliberately small diagnostic panel, not a
random or representative sample. The result cannot estimate prevalence,
general temporal accuracy, or 2026 evaluation performance.

The planned control was fresh uniform-12. The planned treatment was
oracle-hybrid-12: six global and six label-derived local frames. Had the CPU
gate passed, both conditions would have held the model revision, prompt,
decoding, seed, 12-frame budget, and maximum-pixel proxy fixed. Required-moment
metadata would have controlled offline frame construction only and would not
have entered the model prompt. Because the agreement prerequisite failed, no
treatment or control requests were constructed or run.

### Annotation agreement failed narrowly

Three reviewers independently annotated all ten rows de novo from the full
videos while blind to references, predictions, grades, selector timestamps,
condition outcomes, and one another's annotations. The two primary reviews
were frozen before comparison. The third reviewer remained blind while A/B
agreement was computed, then froze the de novo review before learning that
result. It is supplemental only and was not used to rescue the failed primary
gate.

| Measure | Observed | Predeclared requirement | Outcome |
|---|---:|---:|---|
| Primary reviewer A eligible | `9/10` | descriptive | — |
| Primary reviewer B eligible | `10/10` | descriptive | — |
| Supplemental reviewer C eligible | `10/10` | descriptive | — |
| A/B eligibility match | `9/10` | part of structural rule | — |
| A/B structural match | `8/10` | at least `8/10` | pass at threshold |
| Eligible structural matches | `8` | descriptive | — |
| Median tIoU | `0.4791666666666665` | at least `0.50` | fail |
| tIoU interquartile range | `0.49476190476190574` | descriptive | — |
| Flagged rows | `6` | descriptive | — |
| Combined agreement gate | `false` | all requirements pass | stop |

tIoU measures temporal overlap between the primary reviewers' preferred
sufficient evidence sets after the required structural match. The high spread
and six flagged rows reinforce that the aggregate miss is not suitable for
boundary-sensitive recall scoring, even though the median threshold was missed
by only about `0.021`.

### The preregistered stop prevented the original downstream measurements

| Planned stage | Completion | Result |
|---|---:|---|
| Blind full-video annotation | three reviewers, `10/10` rows each | completed and frozen |
| Primary A/B agreement | `10/10` rows compared | failed median-tIoU gate |
| Third-reviewer adjudication | not run | prohibited after failed primary gate |
| Actual-PTS inventory and frame extraction | not run | prerequisite failed |
| Uniform-12 and oracle-hybrid-12 selector recall | not run | no admissible adjudicated labels |
| CPU recall gate | not run | stopped at annotation agreement |
| Local ROCm inference | not run | CPU prerequisites did not pass |
| Reference loading, prediction grading, and paired answer decision | not run | no predictions were generated |
| Transcript work | not run | outside this visual checkpoint |

Accordingly, there are no selector-recall values, missed-row counts, answer
scores, C/P/W transitions, or inference runtime to report. “Not run” must not
be interpreted as zero performance.

### Historical validity, decision, and limits

- **Canonical claim:** the current required-moment rubric and blind annotation
  process did not meet its own agreement requirement on this selected panel.
- **Evidence validity:** **Incomplete** for the intended oracle comparison; the
  completed agreement stage is a valid **Diagnostic** observation.
- **Decision disposition:** **Diagnostic signal**. No oracle condition or
  retrieval branch is promoted; uniform-12 remains the **Current default**.
- **Decision rule:** predeclared. Agreement required both structural match on
  at least `8/10` rows and median tIoU of at least `0.50`; the latter failed.
- **Gate outcome:** `stop_before_inference` at the primary agreement gate.
- **Completion denominator:** all three blind annotations cover `10/10` rows;
  downstream stages were not run because the stop rule fired, so they produced
  no admissible measurement.
- **Supersedes / superseded by:** neither. This result does not replace earlier
  fixed-budget selector findings.
- **Uncertainty:** the panel is selected and small; the result diagnoses this
  rubric/process/panel combination, not all annotation protocols or temporal
  questions.

This result rules out using the current frozen annotations to make a
boundary-sensitive required-moment recall claim. It does not rule out oracle,
hybrid, learned, shot/event, or coarse-to-fine sampling; it does not establish
whether retrieval, fusion, or answer selection dominates; and it says nothing
about the relative value of a future transcript experiment.

### Historical immutable evidence and tracked implementation

The private evidence bundle is identified by hashes rather than linked or
copied into this report:

- protocol SHA-256:
  `0b58f26faa8ce2eb5da0545254956712ff97ef4536311202fa90e39bb709b2d6`
- annotation packet file SHA-256:
  `f7e57def9f296f5c1d4804a45b802ed4abee30bf1d7c1690bc9dc5e39861a122`
- annotation packet canonical SHA-256:
  `10e63a7b438b44d246352591e68332f4bc31ed434d5165e296b2ba5740b93e21`
- frozen orchestration script SHA-256:
  `7dd414fc8de454e3abad7a5493a5079c2262b822c48560705d367436818832b1`
- frozen temporal-contract source SHA-256:
  `f3876e50bdcedae4dd43c14fe1732806eb131c5a44e54ccd1f9c9038414312b4`
- frozen Git revision:
  `ac5ce849c1ba635cf39220af94d7451fccba8256`
- primary agreement record SHA-256:
  `d4d94bb2f149ed3314130d7bd6f6bbab60583ab9e43a10c294227a998d87a1ea`
- agreement stop-decision SHA-256:
  `0ab6bdaa76a8b433d5d240983e7f97ff80636885488477db82421b8d5e4450cf`
- freeze record SHA-256:
  `83e78f15afa38cf84cba0ccb11cb5363424ebc4b5dc88b353d6e8b09d881f416`
- binding SHA-256:
  `9bf33413622cbf299110acb581a18ecec8a018272de755f1c509936254cb7e77`

The tracked validation and scoring contract lives in
[`temporal_bottleneck.py`](../../src/trec_video_2026/vqa_experiments/temporal_bottleneck.py),
with contract tests in
[`test_temporal_bottleneck.py`](../../tests/test_temporal_bottleneck.py). The
preregistered execution plan is
[`2026-07-30-temporal-required-moment-oracle.md`](../superpowers/plans/2026-07-30-temporal-required-moment-oracle.md).

No private questions, row identifiers, labels, intervals, paths, frames,
answers, predictions, or logs are included here.

## Next decision

The next matched inference test must compare the selected `oracle-hybrid-12`
condition with uniform-12 under frozen model, prompt, decoding, frame/pixel
budget, and label-after-freeze scoring controls. It must report model-answer
results separately from the r4 extraction/representation evidence and apply
its predeclared paired improvement and non-regression rule. No Stage 2 work
has started at this checkpoint. The historical agreement-stop result remains
separate: it is neither rescued nor superseded by this exploratory r4 audit.
