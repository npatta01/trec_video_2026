# Agent Experiment and Result Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one concise, agent-facing guide for proposing, recording, and publishing nonstandard experiments through the project's standard evidence-and-reporting envelope.

**Architecture:** Create one reader-invoked Quarto page without changing the three-item navbar. Keep each experiment's scientific design flexible, route its current conclusion to exactly one existing research theme, and use dated records only as provenance. Link the guide from repository instructions and the existing report, then regenerate deterministic Markdown and self-contained HTML outputs and refresh only the authorized tailnet-only copy.

**Tech Stack:** Quarto 1.9.38, QMD/GFM, self-contained HTML, existing light/dark SCSS and CSS, headless Chrome, Git.

## Global Constraints

- `docs/project-status/adding-experiments.qmd` is the canonical editable source; its sibling Markdown and `reports/project-status/adding-experiments.html` are generated outputs.
- Do not add a fourth navbar item; preserve Start Here, Research Themes, and Code Map and Gaps.
- Do not prescribe one experiment design, panel size, metric, workflow, or model family.
- A completed result has exactly one primary research-theme record; related themes link to it instead of duplicating it.
- Dated evidence records preserve provenance and do not become a second current-state ledger.
- Do not add a schema, generator, CI workflow, custom validator, or application test suite.
- Do not run the application test suite for this documentation-only change.
- Do not expose private competition data, evaluation rows, credentials, raw logs, private URLs, or unapproved media.
- Keep the existing report accessible in light and dark modes at desktop and 390-pixel phone widths.
- Publish only sanitized derived HTML to the existing tailnet-only Tailscale Serve portal; do not enable Funnel or change sharing scope.

---

### Task 1: Author the canonical guide and its entry points

**Files:**
- Create: `docs/project-status/adding-experiments.qmd`
- Modify: `AGENTS.md:1-17`
- Modify: `README.md:27-39`
- Modify: `docs/project-status/index.qmd:210-233`
- Modify: `docs/project-status/experiment-ledger.qmd:172-192`
- Modify: `docs/project-status/code-map.qmd:140-169`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-26-agent-experiment-result-guide-design.md`, the six existing `research-*.qmd` page contracts, and the existing evidence-status vocabulary at `experiment-ledger.qmd#how-to-interpret-results`.
- Produces: one canonical guide with stable anchors `the-rule-in-one-minute`, `where-does-this-update-belong`, `before-the-run`, `after-the-run`, `update-the-living-report`, `render-and-verify`, and `agent-handoff`; five contextual entry points; no navbar change.

- [x] **Step 1: Confirm the source ownership and clean baseline**

Run:

```bash
git status --short --branch
rg -n 'href: (index|experiment-ledger|code-map)\.qmd|text: (Start Here|Research Themes|Code Map and Gaps)' docs/project-status/_quarto.yml
rg -n '^## (Open hypotheses and ideas|Cheapest valid next experiments|What has been tried)' docs/project-status/research-*.qmd
```

Expected: the branch is clean; `_quarto.yml` has exactly the existing three
navbar destinations; all six research-theme pages expose the established
current-record and next-experiment sections.

- [x] **Step 2: Create the standalone QMD guide**

Create `adding-experiments.qmd` with:

```yaml
---
title: "Adding Experiments and Results"
description: "A flexible scientific workflow inside a consistent evidence, interpretation, and reporting contract."
---
```

Follow the established deep-page pattern: a quiet breadcrumb/local-navigation
row back to Start Here and Research Themes, then a visible lead paragraph. Use
plain language before specialist terms. Define a **falsifier** as a result that
would show the proposed change did not help under the declared condition, and
define **freeze** as saving predictions before expected answers or grades can
influence them.

Use these visible sections and content:

1. `## The rule in one minute {#the-rule-in-one-minute}`
   - state that static audits, compatibility probes, ablations, qualitative
     diagnostics, and benchmarks may use different scientific designs;
   - state that every design must still expose its question, treatment/control,
     panel boundary, integrity safeguards, observed result, interpretation,
     evidence status, limitations, and next decision;
   - include a short callout: “Standardize the handoff, not the hypothesis.”
2. `## Where does this update belong? {#where-does-this-update-belong}`
   - add a compact routing table for early idea, scoped next experiment,
     completed result, cross-theme relevance, durable selected failure, theme
     status/queue change, supersession, baseline/router change, reusable-code
     boundary, and correction without new evidence;
   - link every destination to its actual report page or stable section;
   - explain that the theme owning the research question is primary and a
     cross-theme record is linked, not copied.
3. `## Before the run {#before-the-run}`
   - teach and list: why/question; hypothesis/falsifier; treatment/control and
     held-fixed variables; panel composition/size/selection/data split; model,
     revision, prompt, configuration, seed, code revision, and environment when
     material; label isolation, freeze, provenance, and scoring; predeclared
     decision rule; privacy, rights, cost, and authorization;
   - explicitly say a field may be “not applicable” with a reason, but it may
     not disappear when it affects interpretation.
4. `## After the run {#after-the-run}`
   - order the workflow: preserve immutable evidence; record observation apart
     from interpretation; assign an existing status; state panel and
     generalization boundaries; classify confirmed defect versus audit gap
     versus uncertainty; state what is ruled out and not ruled out; make a
     decision; name a reconsideration condition; preserve corrections and
     superseded claims;
   - link the existing status vocabulary and Failure Library boundary.
5. `## Update the living report {#update-the-living-report}`
   - show the established **What has been tried** record fields: Reason for
     testing; Tested condition and panel; Observed result; Current
     condition-bound status; What this rules out now; What it does not rule
     out; Possible defect/audit gap/uncertainty; Reconsider when; collapsed
     method/evidence;
   - show the established next-experiment fields: Hypothesis; Smallest valid
     panel or work item; Matched controls and integrity safeguards; Decision or
     promotion rule; Boundary on what a positive result would establish;
   - state when Start Here, the theme matrix/queue, supersession table, Failure
     Library, and Code Map must also change.
6. `## Render and verify {#render-and-verify}`
   - state that QMD is editable and Markdown/HTML are generated;
   - include the repository-root commands from Task 2 Step 1;
   - list the proportional checks without adding scripts or CI.
7. `## Agent handoff {#agent-handoff}`
   - provide a copy-and-adapt Markdown packet with the exact fields from the
     approved design: name; primary/related themes; question/hypothesis/
     decision rule; treatment/control/panel/data boundary; revisions/evidence;
     observed result/status/limits; decision/reconsideration/supersession;
     report/code paths; verification; unresolved defects/audit gaps/unknowns;
   - label it guidance rather than a machine schema.

Keep detailed lists compact and use native tables, callouts, and disclosures
already supported by the site. Do not add CSS unless a verified overflow or
contrast defect requires it.

- [x] **Step 3: Add the five contextual entry points**

Make these source edits:

- In `AGENTS.md`, add a short `## Adding or updating research` section before
  Shared local dataset. Link directly to
  `docs/project-status/adding-experiments.qmd`, tell agents to choose one
  primary theme, and remind them never to hand-edit generated Markdown/HTML.
- In README's Start Here list, add **Add an experiment or result** linking to
  `docs/project-status/adding-experiments.md`; keep historical context as the
  final path.
- In Start Here's reading paths, add a short contributor path linking to
  `adding-experiments.{{< meta report-ext >}}` without changing the newcomer
  reading sequence.
- Immediately before Research Themes' branch-complete queue, add one sentence
  directing agents to the guide before turning a branch into a run or result.
- In Code Map, add a short maintenance paragraph near the repository map and
  remove the now-completed documentation seed that asks for the QMD/Markdown/
  HTML update procedure. Leave unrelated seeds unchanged.

- [x] **Step 4: Verify the source-level contract**

Run:

```bash
test "$(rg -c '^      - href:' docs/project-status/_quarto.yml)" -eq 3
test "$(rg -l 'adding-experiments' AGENTS.md README.md docs/project-status/index.qmd docs/project-status/experiment-ledger.qmd docs/project-status/code-map.qmd | wc -l)" -eq 5
test "$(rg -c '^## (The rule in one minute|Where does this update belong\?|Before the run|After the run|Update the living report|Render and verify|Agent handoff)' docs/project-status/adding-experiments.qmd)" -eq 7
rg -n 'primary theme|not.*schema|must not be hand-edited|observed result|audit gap|uncertainty|reconsider' docs/project-status/adding-experiments.qmd
git diff --check
```

Expected: three navbar items, all five entry points, seven guide sections, the
flexible-core/standard-envelope guardrails, and no whitespace errors.

- [x] **Step 5: Review and commit the canonical source change**

Inspect:

```bash
git diff -- AGENTS.md README.md docs/project-status/*.qmd docs/project-status/_quarto.yml
git status --short
```

Confirm `_quarto.yml` is unchanged, no generated output is hand-edited, and no
private identifier or local artifact path entered the guide. Then commit:

```bash
git add AGENTS.md README.md docs/project-status/adding-experiments.qmd docs/project-status/index.qmd docs/project-status/experiment-ledger.qmd docs/project-status/code-map.qmd
git commit -m "docs: add experiment result workflow"
```

### Task 2: Render, audit, and review all report surfaces

**Files:**
- Create: `docs/project-status/adding-experiments.md`
- Create: `reports/project-status/adding-experiments.html`
- Modify: generated Markdown and HTML corresponding to the QMD entry-point edits
- Modify only if a verified defect requires it: `docs/project-status/styles.css`
- Modify: `docs/superpowers/plans/2026-07-26-agent-experiment-result-guide.md`

**Interfaces:**
- Consumes: the committed QMD and entry-point sources from Task 1.
- Produces: exactly 12 QMD, 12 sibling GFM, and 12 self-contained HTML pages; deterministic outputs; source/link/privacy/accessibility evidence; an independent review verdict.

- [x] **Step 1: Render every canonical output**

From the repository root, run:

```bash
guide_render_cache=$(mktemp -d /tmp/trecvid-guide-render.XXXXXX)
for source in docs/project-status/*.qmd; do
  page=$(basename "$source" .qmd)
  env XDG_CACHE_HOME="$guide_render_cache/cache" quarto render "$source" --to gfm --output-dir docs/project-status --output "$page.md"
done
env XDG_CACHE_HOME="$guide_render_cache/cache" quarto render docs/project-status --to html
```

Expected: exactly 12 QMD, 12 sibling Markdown, and 12 HTML files. Remove only
task-generated Quarto transients after resolving their exact paths; do not
delete or overwrite unrelated files.

- [x] **Step 2: Prove deterministic regeneration from an isolated copy**

Create a task-specific temporary directory at the same relative repository
depth, copy only the Quarto sources and required styles into it, render all 12
GFM and HTML pages with a separate cache, and compare all 24 generated files
byte-for-byte with `cmp`.

Expected: `12/12` Markdown and `12/12` HTML matches. Stop and diagnose any
mismatch before review or portal refresh.

- [x] **Step 3: Run focused report checks**

Verify all of the following without adding a repository test or validator:

```text
12 QMD / 12 sibling Markdown / 12 self-contained HTML files.
Exactly three top-level navbar links on every HTML page.
The new guide route and all seven authored fragment targets resolve.
All five declared entry points reach the correct QMD, Markdown, or HTML form.
All repository-relative links and fragments resolve.
No duplicate authored guide IDs; report Quarto-owned resource IDs separately.
No remote runtime src/poster/srcset/import/stylesheet dependency.
No credential, bucket URI, private URL, raw-log fragment, local filesystem URL,
or 2026 evaluation identifier in source or generated output.
No source-video ID embedded as media by this change.
```

Run `git diff --check`. Confirm the generated Markdown presents the full guide
without requiring HTML-only disclosure behavior.

- [x] **Step 4: Inspect representative responsive views**

Use authorized headless Chrome to inspect:

- the new guide and Start Here;
- `1440x1000` and `390x844` viewports;
- normal/light preference and `--force-dark-mode`.

Check navigation collapse, reading width, routing-table overflow, callout and
link contrast, focus visibility, code-block wrapping/scrolling, and zero page
overflow. Reuse the existing styles. If a verified defect requires a CSS fix,
make the smallest scoped change and repeat the affected render and checks.

- [x] **Step 5: Obtain an independent documentation review**

Give a fresh reviewer the design spec, this implementation plan, source diff,
new guide, generated outputs, and verification evidence. Ask the reviewer to
report Critical, Important, and Minor findings for:

- faithfulness to the flexible-core/standard-envelope design;
- clarity to a new agent and a new-to-field reader;
- correct routing into the six current themes;
- absence of a second ledger/schema or prescriptive experiment topology;
- evidence/privacy/accessibility accuracy;
- consistency across QMD, GFM, and HTML.

Resolve all Critical and Important findings, rerender, and repeat affected
checks. Record the verdict and remediation in this plan.

- [x] **Step 6: Commit reviewed generated outputs**

Inspect the exact diff and confirm it contains only intended sources, derived
pages, any justified scoped style correction, and this plan's evidence update.
Then commit:

```bash
git add docs/project-status reports/project-status docs/superpowers/plans/2026-07-26-agent-experiment-result-guide.md
git commit -m "docs: render experiment result guide"
```

**Task 1 and Task 2 completion evidence (2026-07-26):**

- Task 1 is committed at `e2cba4fe0927b016db77ebd71e0e0a9b9c64564b`.
  Its five source steps passed with the three-item navbar unchanged, all five
  contextual entry points present, and the seven guide sections anchored.
- Quarto `1.9.38` rendered exactly 12 QMD sources into 12 sibling GFM files and
  12 self-contained HTML files. After the final review remediation, a
  same-depth isolated copy containing only the Quarto sources and styles used a
  separate cache and reproduced all 24 generated files byte-for-byte (`12/12`
  GFM and `12/12` HTML). Task-created `.quarto`, nested `docs/`, `.gitignore`,
  `search.json`, and `site_libs` transients were removed after their paths were
  confirmed.
- The final resolver checked 51 QMD, 180 GFM, and 351 HTML repository-relative
  links, including 33 QMD, 106 GFM, and 217 HTML fragments, with zero
  unresolved. All seven guide fragments and all 11 source/generated forms of
  the five declared entry points resolve. Every HTML page has exactly the
  original three top-level navbar links, and the generated GFM exposes the
  complete guide without an HTML-only disclosure.
- Authored content and control IDs are unique. Quarto's dual-theme renderer
  separately emits its resource IDs `quarto-bootstrap` and
  `quarto-text-highlighting-styles` three times on each page; these are
  renderer-owned resource IDs, not authored fragment collisions. Generated
  output has zero remote runtime `src`, `poster`, `srcset`, import, or
  stylesheet dependencies.
- Publication-boundary scans found zero credential patterns, bucket URIs,
  private or local filesystem URLs, raw-log fragments, 2026 evaluation
  identifiers, or source-video media in the new guide. The required generic
  `/tmp/trecvid-guide-render.XXXXXX` cache template is documentation, not a
  local filesystem URL. No source-video ID is embedded as media by this change.
- Authorized Chrome `150` inspected Adding Experiments and Results and Start
  Here at `1440x1000` and `390x844` in light and forced-dark modes (`8/8`
  scenarios). The correct color scheme and three-link navigation were present;
  the phone menu collapsed and expanded; main reading widths were 799 pixels
  desktop and about 340 pixels phone; routing tables remained within the page;
  code containers used horizontal scrolling; and document scroll width equaled
  viewport width in every scenario. Guide text, link, and callout samples had a
  minimum measured contrast of `6.288:1` light and `10.000:1` dark.
- Chrome found two presentation defects before review. Bootstrap specificity
  suppressed the mobile navbar toggle's keyboard indicator, and the new routing
  table split words at 390 pixels. The scoped CSS correction restores a bright
  3-pixel link-colored focus ring with a 2-pixel separating outline, keeps the
  routing table at 340 pixels with balanced 170-pixel columns, and preserves
  whole-word wrapping. Final light/dark desktop/phone checks reproduced zero
  page overflow and `overflow-x: auto` on both guide code containers.
- The first fresh, read-only documentation review reported zero Critical, two
  Important, and one Minor finding: missing explicit invalid/incomplete-run
  handling, a missing report-output privacy boundary, and wording that could
  imply one new evidence file per diagnostic. The canonical QMD now says to
  stop and classify possible label/evaluation influence, record missing
  immutable evidence as an audit gap, preserve incomplete observations without
  scoring them as completed comparisons, exclude the specified private
  material from report outputs, follow the source-media rights boundary, and
  extend an existing evidence-of-record report when it already owns the result.
  All 24 outputs were rerendered and reproduced byte-for-byte after remediation.
  The same read-only reviewer then reported zero Critical, Important, or Minor
  findings and approved Task 2.
- Self-review confirmed the intended QMD remediation, four generated GFM
  changes, all 12 generated HTML pages, the two justified scoped CSS fixes, and
  this evidence record. `git diff --check` passes. No application tests,
  repository validator, schema, CI workflow, dataset read, portal action,
  public deployment, push, pull request, or issue mutation occurred.

### Task 3: Refresh the authorized portal and close the implementation record

**Files:**
- Modify outside Git: `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/*.html`
- Modify: `docs/superpowers/plans/2026-07-26-agent-experiment-result-guide.md`

**Interfaces:**
- Consumes: 12 independently reviewed, sanitized repository HTML files from Task 2.
- Produces: a byte-identical 12-page tailnet-only preview, live-route evidence, unchanged Serve/Funnel scope, and a clean committed branch.

- [x] **Step 1: Reconfirm publication scope before copying**

Run read-only checks for `tailscale serve status`, `tailscale funnel status`,
the portal directory contents, and the current portal index link. Record the
existing mapping and confirm it remains tailnet-only.

- [x] **Step 2: Refresh only the report directory**

Copy the 12 sanitized files from `reports/project-status/` into the existing
`/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/` directory.
Remove no unrelated portal artifact. Confirm the portal directory contains
exactly the intended report files and all 12 copies match repository bytes.

- [x] **Step 3: Verify live HTTPS routes and containment**

Verify HTTP 200 and byte identity for the Start Here directory route and all
11 named HTML routes, including `adding-experiments.html`. Confirm the portal
root still links to the report and its description is unchanged. Re-run Serve
and Funnel status and confirm no public listener, Funnel, or mapping change.

- [x] **Step 4: Record final evidence and commit**

Update this plan with:

- final commit IDs and intended file counts;
- render version and `12/12` deterministic comparisons;
- link, fragment, privacy, runtime-resource, and responsive-view results;
- independent review verdict and resolved findings;
- portal byte-match/live-route results;
- before/after tailnet-only Serve/Funnel status;
- statement that no application tests, public deployment, issue, PR, or push
  occurred.

Run:

```bash
git diff --check
git status --short
git diff -- docs/superpowers/plans/2026-07-26-agent-experiment-result-guide.md
```

Commit the evidence record:

```bash
git add docs/superpowers/plans/2026-07-26-agent-experiment-result-guide.md
git commit -m "docs: record experiment guide verification"
```

- [x] **Step 5: Final clean-state check**

Run:

```bash
git status --short --branch
git log -4 --oneline --decorate
git diff HEAD~3..HEAD --check
```

Expected: clean `codex/repository-cleanup-report`, three implementation commits
after the design/plan commits, and no whitespace errors. Do not push, open a
pull request, or create GitHub issues without separate authorization.

**Task 3 completion evidence (2026-07-26):**

- The implementation commits are Task 1
  `e2cba4fe0927b016db77ebd71e0e0a9b9c64564b` and Task 2
  `bf3a460d15410d6e1540d7bfd889345798500eca`; this evidence-record commit
  closes Task 3. Quarto `1.9.38` had already produced the intended 12 QMD,
  12 GFM, and 12 self-contained HTML report surfaces, with isolated,
  post-remediation deterministic comparisons of `12/12` GFM and `12/12` HTML.
- Task 2's focused checks found zero unresolved repository-relative links or
  fragments; all seven guide fragments and 11 declared entry-point forms
  resolve. It found zero remote runtime dependencies, credential patterns,
  bucket URIs, private or local filesystem URLs, raw-log fragments, 2026
  evaluation identifiers, or source-video media added by the guide. Its eight
  light/dark desktop/phone Chrome inspections found no page overflow, contained
  the routing table, preserved the three-link navigation and visible focus, and
  measured at least `6.288:1` light and `10.000:1` dark contrast.
- The independent read-only documentation re-review approved the completed
  sources and outputs with zero Critical, Important, or Minor findings after
  the explicit invalid/incomplete-run, report-output privacy, and existing
  evidence-of-record routing fixes were made.
- Before refresh, `tailscale serve status` and `tailscale funnel status` both
  reported only `https://npatta01-framework.tail481212.ts.net (tailnet only)`
  with `/` mapped to `/home/npatta01/codex-rendered`. The portal root retained
  the `plans/trecvid-vqa-current-state/index.html` link and its unchanged
  TRECVID 2026 VQA current-state description.
- Only the 12 reviewed `reports/project-status/*.html` files were copied to
  `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`; its HTML
  file set now exactly matches the intended 12 files and repository comparisons
  pass `12/12`. The live Start Here directory route plus all 11 named HTML
  routes, including `adding-experiments.html`, returned HTTP 200 and matched
  repository bytes (`12/12`).
- After refresh, both Serve and Funnel status commands reported the same
  tailnet-only root mapping. No public listener, Funnel enablement, mapping or
  portal-index change, application test, public deployment, issue, pull
  request, push, or unrelated portal-artifact removal occurred.

## Post-handoff dark code-surface correction (2026-07-26)

- A user dark-mode screenshot exposed a theme-cascade defect: Quarto's dark
  syntax colors were rendered over the inherited light
  `rgba(233, 236, 239, 0.65)` code surface. The focused ephemeral regression
  check failed on that value before the correction.
- The source-level correction adds `$code-block-bg: #1d2a40` to
  `docs/project-status/theme-dark.scss`. The same regression check now finds
  `#1d2a40` for dark fenced and inline code while the light surface remains
  unchanged.
- All 12 self-contained HTML pages were regenerated. A clean isolated render
  matched the repository outputs `12/12`; `git diff --check` passes, and no
  remote runtime resource was added. Generated-content review confirmed that
  only the embedded dark stylesheet changed.
- Combined before/after inspection plus separate dark desktop, dark 390 px
  phone, and light desktop captures show readable code blocks with overflow
  contained inside the code area. The minimum syntax-token contrast is
  `5.04:1`; base code text is `12.90:1`. Independent review reported zero
  Critical, Important, or Minor findings.
- Only the 12 sanitized HTML copies were refreshed under the existing portal
  report directory. The portal copy and live tailnet-only HTTPS routes match
  repository bytes `12/12`; Serve and Funnel status remain tailnet-only with
  the same root mapping. No application test, public deployment, issue, pull
  request, or push occurred.
