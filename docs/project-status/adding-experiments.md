# Adding Experiments and Results


<nav class="status-breadcrumb" aria-label="Breadcrumb">

<a href="index.md">Start Here</a> <span aria-hidden="true">/</span>
<a href="experiment-ledger.md">Research Themes</a>
<span aria-hidden="true">/</span> <span aria-current="page">Adding
Experiments and Results</span>
</nav>

<nav class="theme-local-nav" aria-label="On this guide page">

<a href="#the-rule-in-one-minute">Rule</a>
<a href="#where-does-this-update-belong">Routing</a>
<a href="#before-the-run">Before</a> <a href="#after-the-run">After</a>
<a href="#update-the-living-report">Report</a>
<a href="#render-and-verify">Render</a>
<a href="#agent-handoff">Handoff</a>
</nav>

<div class="lead">

This guide helps contributors turn an idea, a run, or a finding into
durable research memory without forcing every scientific question into
the same experiment design. It keeps the evidence and the current
project decision readable, auditable, and safely separated from private
run artifacts.

</div>

## The rule in one minute

Static audits, compatibility probes, matched ablations, qualitative
diagnostics, and benchmarks can use different scientific designs. Each
still needs a visible question; treatment and control; panel boundary;
integrity safeguards; observed result; interpretation; evidence status;
limitations; and next decision.

> [!NOTE]
>
> ### The reporting contract
>
> **Standardize the handoff, not the hypothesis.**

The question decides the scientific shape. The reporting envelope makes
clear what was tested, what was observed, and what the result can
support now.

## Where does this update belong?

The theme that owns the research question is the **primary theme**. A
result with cross-theme relevance is linked from related themes, not
copied into a second current record. Dated evidence preserves
provenance; the primary theme states the readable current conclusion.

| Update | Canonical destination |
|----|----|
| Early idea | The primary theme’s [Open hypotheses and ideas](experiment-ledger.md#six-theme-research-map). |
| Scoped next experiment | The primary theme’s [Cheapest valid next experiments](experiment-ledger.md#six-theme-research-map). |
| Completed result | One primary theme’s [What has been tried](experiment-ledger.md#six-theme-research-map), plus sanitized dated evidence as located in the [repository map](code-map.md#repository-map). Extend an existing evidence-of-record report when it already owns the result; do not force a new file for every diagnostic. |
| Cross-theme relevance | A link from the related [Research Themes](experiment-ledger.md#six-theme-research-map), not a copied record. |
| Durable selected failure | The [Failure Library](failure-library.md#how-to-read-the-records), only when its evidence and disclosure boundary fit. |
| Theme status or queue change | The [theme matrix and branch-complete queue](experiment-ledger.md#branch-complete-next-test-queue). |
| Supersession | The primary theme and the [supersession table](experiment-ledger.md#what-superseded-what); retain the historical evidence. |
| Baseline or router change | [Start Here](index.md#current-recommendation), the primary theme, and the [hub status](experiment-ledger.md). |
| Reusable-code boundary | The [Code Map](code-map.md#code-layers), and affected code documentation when needed. |
| Correction without new evidence | Correct the owning [Research Themes record](experiment-ledger.md#six-theme-research-map), mark the correction, and preserve its provenance. |

## Before the run

Write enough for a reader to understand the planned claim before
outcomes can influence it. A **falsifier** is a result that would show
the proposed change did not help under the declared condition. A
**freeze** saves predictions before expected answers or grades can
influence them.

- **Why and question:** why the decision matters and the question being
  tested.
- **Hypothesis and falsifier:** what is expected, and the result that
  would not advance it.
- **Treatment, control, and held-fixed variables:** what changes, the
  matched comparison, and what intentionally remains the same.
- **Panel and data boundary:** composition, size, selection, and data
  split.
- **Material implementation details:** model and revision, prompt,
  configuration, seed, code revision, and environment when they affect
  interpretation.
- **Integrity and scoring:** label isolation, freeze, provenance, and
  scoring plan.
- **Decision rule:** the predeclared promotion, rejection, or next-step
  rule.
- **Operational boundary:** privacy, dataset and media rights, cost, and
  run authorization.

A field can be **not applicable** with a reason, but it may not
disappear when it affects interpretation. These are reporting questions,
not a required panel size, metric, model family, or experiment topology.

Keep private competition data, evaluation rows, credentials, raw logs,
and unapproved media out of report sources and generated outputs.
Include source-video links or limited snapshots only when the
repository’s recorded rights and disclosure policy authorizes them.

## After the run

1.  Preserve the smallest immutable evidence needed to audit the claim.
2.  Record the **observed result** apart from the project’s
    interpretation.
3.  Assign an existing [evidence
    status](experiment-ledger.md#how-to-interpret-results); do not
    invent a parallel status system.
4.  State the panel, selection, data-split, and generalization
    boundaries.
5.  Classify a possible **confirmed defect**, **audit gap**, or
    **uncertainty**.
6.  Say what the result rules out now and what it does not rule out.
7.  Make the resulting decision and name the condition that would
    trigger reconsideration.
8.  Preserve corrections and superseded claims instead of silently
    erasing them.

If labels, expected answers, or evaluation data may have influenced
inference, stop before making a quality claim and classify the validity
problem. If immutable evidence is missing, record an **audit gap**
rather than upgrading a result from memory. If a run is incomplete,
preserve the partial observation but do not score it as a completed
promotion comparison.

The [Failure Library](failure-library.md#how-to-read-the-records) is for
selected, fully disclosed development cases. It is not a place to turn
every incomplete run into a prevalence claim.

## Update the living report

For a completed result, add or revise one primary-theme **What has been
tried** record with these visible fields:

- Reason for testing
- Tested condition and panel
- Observed result
- Current condition-bound status
- What this rules out now
- What it does not rule out
- Possible defect, audit gap, or uncertainty
- Reconsider when
- Collapsed method, result, validity risks, and evidence

Keep the method, result, validity risks, and evidence in that collapsed
disclosure so the current conclusion remains scannable while audit
detail stays available.

For an upcoming test, use the established **Cheapest valid next
experiments** fields:

- Hypothesis
- Smallest valid panel or work item
- Matched controls and integrity safeguards
- Decision or promotion rule
- Boundary on what a positive result would establish

Also update [Start Here](index.md) when the operational baseline or
router changes; the [theme matrix or
queue](experiment-ledger.md#branch-complete-next-test-queue) when status
or next work changes; the [supersession
table](experiment-ledger.md#what-superseded-what) when a conclusion is
replaced; the [Failure Library](failure-library.md) when a durable
selected failure meets its boundary; and the [Code Map](code-map.md)
when a reusable-code boundary changes.

## Render and verify

QMD is editable. Its sibling Markdown and HTML report are generated
outputs and must not be hand-edited. From the repository root,
regenerate all report surfaces with:

``` bash
guide_render_cache=$(mktemp -d /tmp/trecvid-guide-render.XXXXXX)
for source in docs/project-status/*.qmd; do
  page=$(basename "$source" .qmd)
  env XDG_CACHE_HOME="$guide_render_cache/cache" quarto render "$source" --to gfm --output-dir docs/project-status --output "$page.md"
done
env XDG_CACHE_HOME="$guide_render_cache/cache" quarto render docs/project-status --to html
```

Use checks proportional to the change: compare a clean isolated
regeneration, check links and authored fragments, retain the three-item
navbar, inspect light and dark desktop and phone layouts, scan outputs
for private data or remote runtime dependencies, and run
`git diff --check`. Do not add scripts or CI for this reporting
workflow.

## Agent handoff

Use this copy-and-adapt packet as guidance for a human-readable handoff.
It is not a machine schema.

``` markdown
## Experiment or result name

- **Primary theme / related themes:**
- **Question / hypothesis / decision rule:**
- **Treatment / control / panel / data boundary:**
- **Revisions and immutable evidence:**
- **Observed result / evidence status / limits:**
- **Decision / reconsideration condition / superseded claim:**
- **Report pages and code paths changed:**
- **Verification:**
- **Unresolved defects / audit gaps / unknowns:**
```
