# Temporal required-moment oracle diagnostic

Date: 2026-07-30

Primary theme: **Visual and Temporal Evidence**

Related theme: **Routing, Fusion, and Answer Selection**

Decision: **Diagnostic signal—the primary annotation-agreement gate failed,
so the planned selector-recall and oracle-frame comparison did not run.
Uniform-12 remains the visual baseline.**

## Technical summary

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

## Question, hypothesis, and falsifier

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

## Panel and intended comparison

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

## Annotation agreement failed narrowly

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

## The preregistered stop prevented downstream measurements

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

## Validity, decision, and limits

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

## Immutable evidence and tracked implementation

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

Revise and calibrate the interval rubric before repeating the comparison. The
next run should use a small calibration exercise that tests consistent
eligibility, preferred-set topology, anchor/event coverage, and boundary
placement, then freeze three fresh de novo annotations and reapply the same
primary agreement gate. Only a passing fresh agreement result should resume
adjudication and the CPU recall gate.
