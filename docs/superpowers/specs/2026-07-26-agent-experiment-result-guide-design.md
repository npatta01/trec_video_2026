# Agent Guide for Adding Experiments and Results

**Date:** 2026-07-26

**Audience:** Agents and project contributors who propose, execute, interpret,
or document TRECVID VQA experiments

**Canonical guide source:** `docs/project-status/adding-experiments.qmd`

**Delivery:** GitHub-readable Markdown plus self-contained HTML on the existing
private tailnet-only portal until the repository is intentionally made public

## Objective

Add one concise, discoverable guide that tells an agent how to incorporate a
new experiment or result into the project's living research memory. The guide
must allow experiment design to vary with the research question while making
the evidence, interpretation, documentation, and handoff consistently
auditable.

The guide must answer four practical questions:

1. What must be declared before a run, regardless of experiment shape?
2. What must be preserved and distinguished after a run?
3. Which existing page owns each kind of update?
4. How does an agent regenerate and verify the report without creating drift?

## Approved direction

Create a standalone reader-invoked page named **Adding Experiments and
Results**. Link it from repository agent instructions and the existing report,
but do not add a fourth top-level navigation item.

Use a **flexible experiment core inside a standard reporting envelope**:

- do not prescribe one experiment design, workflow, panel size, metric, or
  model family;
- require every experiment to declare its question, falsifier, controls,
  evidence boundary, and decision rule;
- require every completed result to separate observation, interpretation,
  evidence status, limitations, and reconsideration conditions;
- keep exactly one primary research-theme record as the current readable
  conclusion;
- preserve dated evidence as provenance rather than turning it into a second
  current-state ledger;
- keep QMD as the sole editable source for the status report;
- do not add a schema, generator, CI workflow, custom validator, or application
  test suite.

## Alternatives considered

### 1. Standalone reader-invoked guide — selected

Add `docs/project-status/adding-experiments.qmd`, generate its Markdown and
HTML outputs, and link it from `AGENTS.md`, README, Start Here, Research Themes,
and Code Map. This is the most discoverable option for agents while preserving
the existing three-item navbar and theme-first research structure.

### 2. Section inside Code Map

Add the workflow directly to `code-map.qmd`. This produces a smaller diff, but
it overloads a page about implementation boundaries and makes research-memory
maintenance harder to find.

### 3. `docs/experiments/README.md`

Place the workflow next to dated evidence records. This is convenient during
Git work, but it risks implying that chronological evidence files, rather than
the six research themes, own the current conclusion.

## Information architecture

The new page is available through contextual links, not the navbar:

- `AGENTS.md` points agents to the canonical QMD before they add or revise an
  experiment or result;
- README's Start Here section points contributors to the GitHub-readable guide;
- Start Here adds a short contributor path near the existing reading paths;
- Research Themes links the guide near the branch-complete next-test queue;
- Code Map replaces the current documentation seed for an update procedure
  with a link to the completed guide.

The page uses six compact sections:

1. **The rule in one minute** — bespoke scientific design, standard evidence
   and reporting contract.
2. **Where does this update belong?** — a routing table for proposals, results,
   failures, changed decisions, implementation changes, and corrections.
3. **Before the run** — the minimum declaration needed for interpretable work.
4. **After the run** — preservation, interpretation, and evidence-status steps.
5. **Update, render, and verify the living report** — source/output ownership
   and proportional checks.
6. **Agent handoff packet** — a concise copy-and-adapt checklist, not a machine
   schema.

## Standard envelope

### Before the run

The guide requires an agent to record:

- research question and why it matters;
- hypothesis and a result that would falsify or fail to advance it;
- treatment, matched control, and intentionally held-fixed variables;
- panel composition, size, selection method, and data boundary;
- model, revision, prompt, configuration, seed, code revision, and environment
  when they can affect interpretation;
- label-isolation, prediction-freeze, provenance, and scoring plan;
- predeclared decision or promotion rule;
- privacy, dataset, media-rights, cost, and run-authorization boundaries.

The guide explicitly says these are reporting questions, not a mandatory
experimental topology. A static audit, compatibility probe, matched ablation,
qualitative diagnostic, or larger benchmark may answer them differently.

### After the run

The guide requires an agent to:

- freeze and preserve the smallest immutable evidence needed to audit the
  claim;
- record the observed result separately from the project's interpretation;
- assign the existing evidence-status vocabulary rather than inventing a new
  status system;
- state panel size, selection effects, data split, and generalization limits;
- distinguish a confirmed defect from an audit gap and from ordinary
  uncertainty;
- say what the result rules out now and what remains possible;
- name the resulting decision and the condition that would trigger
  reconsideration;
- preserve corrections and superseded conclusions instead of silently erasing
  them.

## Update routing

The guide routes changes as follows:

| Update | Canonical destination |
|---|---|
| Early idea | The relevant theme's **Open hypotheses and ideas** section |
| Scoped next experiment | The relevant theme's **Cheapest valid next experiments** section |
| Completed result | A sanitized dated evidence record plus exactly one primary theme's **What has been tried** section |
| Cross-theme relevance | Links from related themes; do not duplicate the full record |
| Durable selected failure | Failure Library only after its evidence and disclosure boundary is satisfied |
| Theme status or queue change | Research Themes matrix and next-test queue |
| Superseded conclusion | Primary theme plus the hub's supersession table; retain historical evidence |
| Baseline or router decision | Start Here plus the primary theme and hub status |
| Reusable-code boundary | Code Map and, when needed, the affected code documentation |
| Correction without new evidence | Correct the owning record, label the correction, and preserve provenance |

The dated evidence location may extend an existing report when that report is
already the experiment's evidence of record. The guide must not force a new
file for every diagnostic.

## Source and output policy

- `docs/project-status/*.qmd` files are the only editable report sources.
- Sibling Markdown files and `reports/project-status/*.html` are generated
  outputs and must not be hand-edited.
- A completed result has one primary thematic home, even when it informs
  several themes.
- Evidence links should be immutable where practical and visibly labelled as
  current code, evidence of record, or issue/PR history.
- Private competition data, evaluation rows, credentials, raw logs, and
  unapproved media must not enter generated outputs.
- Source-video links and limited snapshots follow the repository's recorded
  rights and disclosure policy.

## Agent workflow and handoff

The guide provides a compact copy-and-adapt packet containing:

- experiment or result name;
- primary theme and related themes;
- question, hypothesis, and decision rule;
- treatment, control, panel, and data boundary;
- revisions and immutable evidence links;
- observed result, evidence status, and limitations;
- current decision, reconsideration condition, and superseded claim if any;
- report pages and code paths changed;
- render, link, privacy, accessibility, and diff checks performed;
- unresolved defects, audit gaps, and unknowns.

This packet is prose guidance. It is not a serialized schema and does not need
custom tooling.

## Guardrails and failure handling

- If no single primary theme is obvious, choose the theme that owns the
  research question and add related-theme links. Do not create a seventh theme
  or duplicate the record.
- If labels, expected answers, or evaluation data may have influenced
  inference, stop and classify the validity problem before making a quality
  claim.
- If immutable evidence is missing, record an audit gap rather than upgrading
  the result from memory.
- If a run is incomplete, preserve the partial observation but do not silently
  score it as a completed promotion comparison.
- If a result conflicts with a current recommendation, update the owning theme
  and supersession trail together; do not change Start Here in isolation.
- If rendering changes an unrelated generated file, diagnose the source or
  environment difference before publication.

## Verification

This documentation-only change uses proportional checks:

1. render every project-status QMD to sibling GitHub-readable Markdown and the
   self-contained HTML site;
2. compare an isolated clean render with committed generated outputs;
3. verify repository-relative links, fragment targets, and the three-item
   navbar;
4. confirm the guide is reachable from every declared entry point;
5. inspect desktop and 390-pixel layouts in both light and dark modes;
6. scan generated outputs for private data, credentials, raw artifacts, remote
   runtime dependencies, and unapproved media;
7. run `git diff --check` and confirm only intended source and derived files
   changed;
8. refresh only the sanitized HTML copy on the authorized tailnet-only portal
   after review, then verify live routes and unchanged Serve/Funnel scope.

Do not add or run the application test suite for this documentation-only
change.

## Success criteria

The change is successful when a new agent can identify the right thematic home,
declare a trustworthy experiment without being forced into a standard design,
record a condition-bound result without overstating it, regenerate the three
report surfaces, and hand off the remaining uncertainty without relying on
institutional memory.
