# Research Report Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing theme-first report robust to cross-theme claims,
invalid or incomplete work, theme evolution, and continued experiment growth
without introducing a schema-heavy ledger or another report page.

**Architecture:** Keep QMD as the only editable report source. Add a lightweight
governance model to the contributor guide and hub, use canonical page anchors as
claim identities, keep evidence validity separate from decision disposition,
and provide a collapsed static experiment index. Regenerate sibling Markdown
and the existing 12-page self-contained HTML site after source review.

**Tech Stack:** Quarto 1.9.x, Pandoc Markdown/QMD, Bootstrap/SCSS theme already
in the project, self-contained HTML, headless Chrome for responsive inspection.

## Global Constraints

- The approved design is
  `docs/superpowers/specs/2026-07-26-research-report-governance-design.md`.
- QMD is the sole editable report source; sibling Markdown and
  `reports/project-status/*.html` are generated outputs.
- Keep exactly the existing 12 report pages and the three-item primary navbar.
- Preserve historical paths and fragments, including
  `#six-theme-research-map`; do not break older GitHub or report links.
- One primary theme owns each claim; one evidence bundle may support several
  linked claims; related themes link rather than copy the full record.
- Use canonical theme-page section anchors as stable claim identities; do not
  introduce opaque IDs, a database, JSON/YAML registry, or schema.
- Keep evidence validity and decision disposition as visibly separate axes.
- Keep the static experiment index collapsed by default and grouped by theme.
- No new custom tests, scripts, validators, CI, search runtime, or application
  test-suite run.
- No raw logs, private evaluation rows, credentials, transcripts, predictions,
  or unapproved media in source or generated outputs.
- Preserve light/dark behavior, semantic headings, progressive disclosure,
  keyboard-readable tables/details, and 390 px reflow.
- Do not create GitHub issues, push, open a pull request, or publish publicly.
- Refresh only sanitized HTML through the existing tailnet-only portal after
  independent review.

---

### Task 1: Contributor governance and hub vocabulary

**Files:**
- Modify: `docs/project-status/adding-experiments.qmd`
- Modify: `docs/project-status/index.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd`

**Interfaces:**
- Consumes: the approved design's three-concept model, two status axes, theme
  lifecycle, invalid-result protocol, and Qwen3-Omni cross-theme example.
- Produces: the canonical contributor rules and hub vocabulary that Task 2
  follows when amending existing claim records.

- [ ] **Step 1: Record the pre-change gaps**

Run:

```bash
rg -n "Evidence bundle|Claim record|Decision disposition|Create a theme|Current research-theme map|Experiment index" \
  docs/project-status/adding-experiments.qmd \
  docs/project-status/index.qmd \
  docs/project-status/experiment-ledger.qmd
```

Expected: the complete governance vocabulary is absent; the hub still displays
`Six-theme research map` and mixes evidence and decision labels.

- [ ] **Step 2: Add the three-concept model and lifecycle to the contributor guide**

In `adding-experiments.qmd`:

- add `Lifecycle` and `Themes` to the local navigation;
- add a **Result lifecycle** section containing the ordered text path
  `idea → planned test → evidence bundle → claim record → project decision → reconsider, supersede, or invalidate`;
- define **Evidence bundle**, **Claim record**, and **Project decision** using
  the approved design wording;
- state that canonical page plus authored section anchor is claim identity;
- add a routing-table row for `New, split, or merged theme`;
- add **When should a theme change?** with the four create conditions and the
  split/merge rules from the design;
- list the required standard theme headings and hub/matrix/queue/related-link
  update steps;
- add the Qwen3-Omni shared-evidence/two-claim example under the routing rule.

Use this exact lifecycle concept table:

```markdown
| Concept | What it owns | Stable reference |
|---|---|---|
| **Evidence bundle** | What executed: frozen configuration, completion count, outputs, and immutable evidence. | Evidence-of-record link. |
| **Claim record** | One condition-bound scientific conclusion. | Canonical theme page plus authored section anchor. |
| **Project decision** | What the project does now because of one or more claims. | Start Here, the owning theme's working answer, and the hub when globally relevant. |
```

- [ ] **Step 3: Make invalid and incomplete handling deterministic**

Under **After the run**, add a compact table with these exact states and actions:

```markdown
| Evidence state | Required action |
|---|---|
| **Valid but negative** | Preserve the claim and use **Not promoted under the tested condition**. |
| **Incomplete** | Preserve the evidence bundle and completion denominator; do not score it as a completed promotion comparison. |
| **Invalid** | Preserve the evidence, mark the intended claim unusable, identify affected conclusions, and require a fresh isolated run. |
| **Audit gap** | State what cannot be reconstructed without claiming an unproven defect. |
| **Corrected interpretation** | Amend the owning claim, state what changed, and preserve the earlier wording and provenance. |
| **Superseded valid claim** | Retain both claims, link `supersedes` and `superseded by`, and update current-answer surfaces when the decision changes. |
```

State immediately below that invalid/incomplete attempts become theme claims
only when their failure changes scientific or operational understanding.

- [ ] **Step 4: Extend the visible completed-result contract and agent handoff**

Require these additional fields for new or materially updated durable claims:

```markdown
- Canonical claim page and anchor
- Shared evidence bundle and immutable evidence link
- Evidence validity
- Decision disposition
- Decision rule and whether it was predeclared
- Gate outcome and completion denominator
- Supersedes / superseded by
```

State that a missing historical field must be reported as `not recorded`, not
inferred, and does not force a wholesale historical migration.

- [ ] **Step 5: Separate evidence validity from decision disposition on current-state surfaces**

In `index.qmd`, rename **Evidence strength** to **Evidence validity** and use
the seven validity labels from the design: Verified, Diagnostic,
Compatibility-only, Incomplete, Invalid, Blocked, Unknown. Remove
`Not promoted under the tested condition` from this table and link readers to
the hub's separate decision-disposition table.

In `experiment-ledger.qmd`:

- rename the visible heading to `Current research-theme map` while retaining
  the explicit `{#six-theme-research-map}` fragment;
- add a near-top contributor callout linking to `adding-experiments`;
- rename **How to interpret results** to **How to interpret evidence and decisions**;
- add the seven-label **Evidence validity** table before the existing decision
  labels;
- retitle the existing table **Decision disposition** and remove `Blocked`
  from it because Blocked belongs to the evidence/readiness axis;
- keep **Current default**, **Fast alternate**, **Routed candidate**,
  **Diagnostic signal**, **Not promoted under the tested condition**,
  **Superseded**, and **Open hypothesis**.

- [ ] **Step 6: Add direct evidence links and a collapsed experiment index**

Link each theme card's **Strongest evidence** text to these anchors:

```text
Integrity  -> research-integrity#frozen-30-protocol-v2
Visual     -> research-visual-evidence#frozen-30-protocol-v2
Speech     -> research-speech-transcripts#automatic-transcript-ablation
Audio      -> research-native-audio#minicpm-audio-on-off
Models     -> research-models#minicpm-fast-alternate
Routing    -> research-routing#taxonomy-audit
```

Add `## Experiment index` after the theme status matrix. Keep it inside a
closed `<details>` element with summary `Expand named experiments by theme`.
Inside, provide six theme headings and direct links to every `###` record under
each theme's **What has been tried** section. Do not include next-test records,
failure-library cases, or chronology rows.

- [ ] **Step 7: Verify Task 1 source consistency**

Run:

```bash
rg -n "Evidence bundle|Claim record|Project decision|Evidence validity|Decision disposition|When should a theme change|Experiment index" \
  docs/project-status/adding-experiments.qmd \
  docs/project-status/index.qmd \
  docs/project-status/experiment-ledger.qmd
rg -n "Six-theme research map" docs/project-status/experiment-ledger.qmd
git diff --check
```

Expected: all new concepts are present; the old visible heading is absent but
the legacy fragment string remains; no whitespace errors.

- [ ] **Step 8: Commit Task 1**

```bash
git add docs/project-status/adding-experiments.qmd \
  docs/project-status/index.qmd \
  docs/project-status/experiment-ledger.qmd
git commit -m "docs: add report governance model"
```

### Task 2: Complete touched claim records and cross-theme ownership

**Files:**
- Modify: `docs/project-status/research-speech-transcripts.qmd`
- Modify: `docs/project-status/research-models.qmd`
- Modify: `docs/project-status/research-native-audio.qmd`

**Interfaces:**
- Consumes: Task 1's completed-result contract and one-primary-theme-per-claim
  rule.
- Produces: complete high-impact claim records and an explicit shared-evidence
  example without duplicated ownership.

- [ ] **Step 1: Complete Automatic transcript ablation**

After its current condition-bound status, add:

```markdown
- **What this rules out now:** dismissing quality-accepted ASR as having no
  diagnostic value on the tested selected lexical-speech panel.
- **What it does not rule out:** routing prevalence, global transcript value,
  non-lexical vocalization performance, transcript-fidelity failures, or 2026
  accuracy.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the selected
  ten-row panel cannot establish prevalence or a global policy.
- **Reconsider when:** a representative label-blind routing panel measures
  accepted and rejected ASR strata, including regressions, under one frozen
  decision rule.
```

Do not change its observed `4/10` to `7/10` result or evidence links.

- [ ] **Step 2: Complete MiniCPM-o 4.5 fast alternate**

After its current condition-bound status, add:

```markdown
- **What this rules out now:** calling MiniCPM-o the global accuracy winner
  under frozen-30 merely because it was fastest or led METEOR.
- **What it does not rule out:** using it when latency dominates, quality on
  other representative panels, or a future accuracy/latency frontier under a
  different controlled environment.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the outer-wall
  timing is environment-specific and frozen-30 is descriptive 2025 training
  evidence.
- **Reconsider when:** a repeated same-hardware matched panel jointly reports
  completion, latency, exact/F1, and the declared speed-quality decision rule.
```

Do not change its METEOR, wall-time, or frozen-30 claims.

- [ ] **Step 3: Make Qwen3-Omni shared evidence and separate claims explicit**

In `research-models.qmd`, label the claim as the canonical model eligibility
claim, state that its stable identity is `#qwen3-omni-quality`, and link the
audio-specific claim at
`research-native-audio#qwen3-omni-native-audio`.

In `research-native-audio.qmd`, label the claim as the canonical native-audio
effect claim, state that the quality-10 neutral A2 evidence bundle is shared,
and leave strict eligibility details canonical on Models. Keep the audio
result, condition-bound fields, and immutable evidence link; shorten any prose
that merely repeats the Models claim rather than supporting the audio claim.

Use the exact labels `Canonical model-eligibility claim` and
`Canonical native-audio-effect claim` so the cross-theme example is easy to
trace.

- [ ] **Step 4: Verify Task 2 scientific boundaries**

Run:

```bash
rg -n "What this rules out now|What it does not rule out|Possible defect, audit gap, or uncertainty|Reconsider when" \
  docs/project-status/research-speech-transcripts.qmd \
  docs/project-status/research-models.qmd
rg -n "Canonical model-eligibility claim|Canonical native-audio-effect claim|quality-10 neutral A2" \
  docs/project-status/research-models.qmd \
  docs/project-status/research-native-audio.qmd
git diff --check
```

Expected: both previously incomplete records now expose all four fields; the
two Qwen3-Omni claims point to one shared evidence bundle without changing
their recorded results.

- [ ] **Step 5: Commit Task 2**

```bash
git add docs/project-status/research-speech-transcripts.qmd \
  docs/project-status/research-models.qmd \
  docs/project-status/research-native-audio.qmd
git commit -m "docs: clarify cross-theme claim ownership"
```

### Task 3: Generated outputs and report verification

**Files:**
- Modify: `docs/project-status/adding-experiments.md`
- Modify: `docs/project-status/index.md`
- Modify: `docs/project-status/experiment-ledger.md`
- Modify: `docs/project-status/research-speech-transcripts.md`
- Modify: `docs/project-status/research-models.md`
- Modify: `docs/project-status/research-native-audio.md`
- Modify: `reports/project-status/*.html` (the existing 12 files only)
- Modify: `docs/superpowers/plans/2026-07-26-research-report-governance.md`

**Interfaces:**
- Consumes: reviewed QMD sources from Tasks 1 and 2.
- Produces: GitHub-readable Markdown, a byte-reproducible 12-page HTML site,
  responsive/accessibility evidence, and a completed canonical plan record.

- [ ] **Step 1: Render the six changed QMD sources to sibling Markdown**

Run from the repository root:

```bash
governance_render_cache=$(mktemp -d /tmp/trecvid-governance-render.XXXXXX)
for source in \
  docs/project-status/adding-experiments.qmd \
  docs/project-status/index.qmd \
  docs/project-status/experiment-ledger.qmd \
  docs/project-status/research-speech-transcripts.qmd \
  docs/project-status/research-models.qmd \
  docs/project-status/research-native-audio.qmd
do
  page=$(basename "$source" .qmd)
  env XDG_CACHE_HOME="$governance_render_cache/cache" quarto render "$source" \
    --to gfm --output-dir docs/project-status --output "$page.md"
done
```

Expected: six sibling Markdown files update; no root-level Markdown outputs.

- [ ] **Step 2: Render the existing HTML site**

Run:

```bash
env XDG_CACHE_HOME="$governance_render_cache/cache" \
  quarto render docs/project-status --to html
```

Expected: Quarto reports `1/12` through `12/12` and writes
`reports/project-status/index.html`; no thirteenth page appears.

- [ ] **Step 3: Remove only known render transients**

After confirming they were generated by Step 2, remove exactly:

```text
docs/project-status/.quarto/
docs/project-status/.gitignore
reports/project-status/search.json
reports/project-status/site_libs/
```

Do not remove unrelated or pre-existing files.

- [ ] **Step 4: Prove isolated reproducibility**

Create a temporary repository-shaped directory containing all 12 QMD files,
`_quarto.yml`, `styles.css`, `theme.scss`, and `theme-dark.scss`. Render GFM
and HTML from that isolated copy and compare:

```text
Markdown byte matches: 12/12
HTML byte matches: 12/12
```

Even though only six Markdown files are expected to change, compare all 12 to
prove the source/generated ownership contract.

- [ ] **Step 5: Check links, fragments, privacy, and runtime containment**

Verify at minimum:

```text
adding-experiments#result-lifecycle
adding-experiments#when-should-a-theme-change
adding-experiments#after-the-run
experiment-ledger#six-theme-research-map
experiment-ledger#experiment-index
experiment-ledger#how-to-interpret-evidence-and-decisions
research-models#qwen3-omni-quality
research-native-audio#qwen3-omni-native-audio
```

Check every repository-relative link in changed pages and confirm no generated
output contains credentials, private-data patterns, local preview URLs, local
filesystem paths, or remote runtime `<script>`, `<link>`, image, audio, or video
dependencies.

- [ ] **Step 6: Inspect responsive and color-scheme behavior**

Using headless Chrome against a loopback-only preview, capture and inspect:

```text
Start Here: light desktop
Research Themes: light and dark desktop; light and dark 390 px
Adding Experiments: light and dark desktop; light and dark 390 px
Speech and Models amended records: light desktop
```

Confirm the experiment index starts collapsed, expands without overflow,
theme cards and tables reflow, local navigation remains usable, code surfaces
stay dark-mode friendly, and the mobile brand/content do not overlap. This is a
visual and interaction smoke check, not a WCAG conformance claim.

- [ ] **Step 7: Record evidence in this plan**

Append a dated completion section containing exact render counts, isolated
comparison counts, link/fragment results, responsive captures, privacy/runtime
scan results, independent review verdict, and the statement that no custom or
application tests were added or run.

- [ ] **Step 8: Commit Task 3**

```bash
git add docs/project-status/*.md reports/project-status/*.html \
  docs/superpowers/plans/2026-07-26-research-report-governance.md
git commit -m "docs: render report governance update"
```

Stage only the six changed sibling Markdown files, the existing 12 HTML files,
and this plan if unchanged Markdown files remain byte-identical.

### 2026-07-26 — Task 3 completion evidence

- Rendered the six reviewed QMD sources to their sibling Markdown files with
  Quarto 1.9.38, then rendered the site `1/12` through `12/12`; the output
  directory contains exactly 12 existing HTML pages. The changed generated
  files are the six Task 3 Markdown pages and their corresponding six HTML
  pages; no QMD, project configuration, or stylesheet source changed.
- Confirmed the four named render transients were absent before the render and
  created by it, then removed only
  `docs/project-status/.quarto/`, `docs/project-status/.gitignore`,
  `reports/project-status/search.json`, and
  `reports/project-status/site_libs/`. Quarto also created one untracked
  source-relative `docs/project-status/docs/project-status/index.html` during
  the requested GFM invocation; it was confirmed generated in this task and
  removed immediately, leaving no extra page or directory.
- Copied all 12 QMD files plus `_quarto.yml`, `styles.css`, `theme.scss`, and
  `theme-dark.scss` to `/tmp/trecvid-governance-isolated.AR9F0Q`, rendered GFM
  and HTML there, and compared bytes with the repository outputs:
  **Markdown 12/12** and **HTML 12/12** matches.
- Checked all 263 repository-relative links in the six changed rendered HTML
  pages and found zero target or fragment failures. All required anchors were
  present: `adding-experiments#result-lifecycle`,
  `adding-experiments#when-should-a-theme-change`,
  `adding-experiments#after-the-run`,
  `experiment-ledger#six-theme-research-map`,
  `experiment-ledger#experiment-index`,
  `experiment-ledger#how-to-interpret-evidence-and-decisions`,
  `research-models#qwen3-omni-quality`, and
  `research-native-audio#qwen3-omni-native-audio`.
- Targeted generated-output scans found **0** credentials, sensitive local
  paths, or local preview URLs, and inspected **72** runtime resource
  attributes across all HTML pages with **0** remote script, stylesheet,
  image, audio, or video dependencies.
- Used a loopback-only `127.0.0.1:8765` preview and headless Chrome 150 for
  the following inspected captures:
  `/tmp/trecvid-governance-captures/start-here-light-desktop.png`,
  `/tmp/trecvid-governance-captures/research-themes-light-desktop.png`,
  `/tmp/trecvid-governance-captures/research-themes-dark-desktop.png`,
  `/tmp/trecvid-governance-captures/research-themes-light-mobile.png`,
  `/tmp/trecvid-governance-captures/research-themes-dark-mobile.png`,
  `/tmp/trecvid-governance-captures/adding-experiments-light-desktop.png`,
  `/tmp/trecvid-governance-captures/adding-experiments-dark-desktop.png`,
  `/tmp/trecvid-governance-captures/adding-experiments-light-mobile.png`,
  `/tmp/trecvid-governance-captures/adding-experiments-dark-mobile.png`,
  `/tmp/trecvid-governance-captures/speech-amended-light-desktop.png`, and
  `/tmp/trecvid-governance-captures/models-amended-light-desktop.png`.
  The static index starts closed; Chrome interaction opened it without
  overflow (`documentScrollWidth 1425 <= viewportWidth 1440`,
  `detailsScrollWidth 797 = detailsClientWidth 797`). Theme cards and the
  status table reflow, local navigation remains usable, the dark code/table
  surfaces are readable, and no mobile brand/content overlap appeared. This
  is a visual smoke check, not a WCAG conformance claim.
- Task 3 self-review verdict: no Critical or Important finding in the scoped
  generated-output diff. `git diff --check` passed. No custom tests,
  application tests, scripts, validators, or CI were added or run.

## Final review and authorized portal refresh

After all three task reviews pass:

1. Run the subagent-driven whole-branch review over the design/plan
   implementation range and resolve all Critical or Important findings.
2. Re-run `git diff --check`, clean-state, and live-output checks.
3. Confirm `tailscale serve status` and `tailscale funnel status` still report
   only the existing tailnet-only `/home/npatta01/codex-rendered` mapping.
4. Copy only the 12 sanitized repository HTML files into
   `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`.
5. Verify the portal directory matches `12/12`, the directory route and all 11
   named pages return HTTP 200 and match repository bytes, and Serve/Funnel
   status is unchanged.
6. Append the publication evidence to this plan and commit it. Do not push or
   open a pull request without separate authorization.
