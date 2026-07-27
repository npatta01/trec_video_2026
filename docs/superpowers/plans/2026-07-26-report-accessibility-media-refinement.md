# Report Accessibility and Media Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the living Quarto report easier to scan in light or dark mode,
integrate old experiment bookmarks into the six current research themes, and
link each failure case to its original YouTube source without redistributing
video media.

**Architecture:** Keep the existing three-item site and QMD-only authoring
model. The Research Themes hub owns zero-height compatibility aliases and one
collapsed history appendix; canonical experiment prose remains on the six
theme pages. Failure records expose source links and a visible evidence trace,
while Quarto's native dual-theme support and shared semantic CSS tokens provide
accessible light/dark presentation without runtime dependencies.

**Tech Stack:** Quarto 1.9.38, Pandoc GFM, Bootstrap/Cosmo Sass, native
`<details>/<summary>`, self-contained HTML, system fonts, and Google Chrome 150
for documentation-only responsive inspection.

## Global Constraints

- Work only on `codex/repository-cleanup-report`; leave detached worktrees and
  their loose files untouched.
- QMD files remain the sole editable source for report content; sibling
  Markdown and `reports/project-status/*.html` are generated outputs.
- Keep exactly three top-level navigation choices: Start Here, Research Themes,
  and Code Map and Gaps.
- Add no page, redirect JavaScript, dependency, schema, CI workflow, custom test
  suite, or application test.
- Preserve all nineteen legacy fragment IDs, but remove the visible migration
  list and do not create a seventh research theme.
- A cross-theme experiment has one primary theme and explicit related-theme
  links; do not duplicate its experiment record.
- Source-video links may identify the four already-tracked 2025 training
  sources. Do not add copied frames, thumbnails, contact sheets, audio,
  transcript excerpts, expected answers, or any 2026 evaluation artifact.
- Never infer a within-video timestamp from the annotations' ISO ingestion
  timestamp. Add a `Watch relevant moment` link only when tracked evidence
  records the actual offset; otherwise state that the relevant offset is not
  recorded.
- Keep HTML self-contained: no remote font, stylesheet, script, image,
  thumbnail, iframe, video, audio, or poster request.
- Communicate status with visible text and a non-color cue; color is redundant.
- Do not stage or commit `data/`. Before reading source annotations, verify the
  `data` symlink resolves exactly to
  `/home/npatta01/data/competitions/trec_video_2026/data`.
- Publish only sanitized derived HTML through the existing tailnet-only portal.
  Do not change Serve mappings, enable Funnel, or publish through Codex Sites.

---

## File structure

### Modified source files

- `docs/project-status/_quarto.yml` — native light/dark theme selection and
  operating-system preference support.
- `docs/project-status/theme.scss` — shared system-font and light-theme Sass
  defaults.
- `docs/project-status/styles.css` — mode-aware semantic tokens, status chips,
  failure-trace accents, focus, forced-colors, print, and mobile behavior.
- `docs/project-status/experiment-ledger.qmd` — legacy aliases integrated into
  current theme cards and one unlisted History and provenance appendix.
- `docs/project-status/failure-library.qmd` — four source-linked, falsifiable
  evidence traces.
- This plan — task status and final verification evidence.

### New source file

- `docs/project-status/theme-dark.scss` — dark Cosmo Sass defaults, including
  body, link, border, component, and Mermaid colors.

### Generated files

- Regenerate all eleven `docs/project-status/*.md` files.
- Regenerate all eleven `reports/project-status/*.html` files.
- Refresh the eleven sanitized HTML copies under
  `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/` only after
  final review.

---

### Task 1: Native light/dark shell and semantic visual system

**Files:**
- Modify: `docs/project-status/_quarto.yml`
- Modify: `docs/project-status/theme.scss`
- Create: `docs/project-status/theme-dark.scss`
- Modify: `docs/project-status/styles.css`

**Interfaces:**
- Consumes: existing Cosmo layout, system-font override, and CSS component
  classes.
- Produces: `body.quarto-light` and `body.quarto-dark` token sets plus reusable
  `.status-chip` and `.case-field-*` presentation classes for Tasks 2 and 3.

- [x] **Step 1: Configure Quarto's native two-theme output**

Replace the current single theme list with:

```yaml
theme:
  light: [cosmo, theme.scss]
  dark: [cosmo, theme.scss, theme-dark.scss]
respect-user-color-scheme: true
```

Keep `embed-resources: true`. Quarto must provide the standard color-scheme
toggle and remember the reader's explicit selection.

- [x] **Step 2: Add the dark Cosmo Sass defaults**

Create `theme-dark.scss` with `/*-- scss:defaults --*/`, repeat
`$web-font-path: false`, and set these values:

```scss
$body-bg: #0f1722;
$body-color: #edf3fa;
$link-color: #8fd3ff;
$border-color: #4b5c74;
$navbar-bg: #162235;
$navbar-fg: #edf3fa;
$table-bg: #172235;
$table-color: #edf3fa;
$code-color: #f1b8ca;
$mermaid-bg-color: $body-bg;
$mermaid-edge-color: #a9b7c9;
$mermaid-node-bg-color: #1d2a40;
$mermaid-node-fg-color: #edf3fa;
$mermaid-label-bg-color: $body-bg;
$mermaid-label-fg-color: #8fd3ff;
```

Retain the same system-font stack as `theme.scss`; do not introduce a web-font
URL.

- [x] **Step 3: Replace light-only CSS values with mode-aware tokens**

Define the current light palette on `body.quarto-light` and this dark palette
on `body.quarto-dark`:

```css
--status-page: #0f1722;
--status-surface: #172235;
--status-surface-raised: #1d2a40;
--status-ink: #edf3fa;
--status-muted: #b7c2d2;
--status-border: #4b5c74;
--status-link: #8fd3ff;
--status-use: #63d5bd;
--status-route: #8fd3ff;
--status-open: #f2c36b;
--status-not-promoted: #f1a3b5;
--status-unknown: #c5b8ef;
```

Use this exact light token set without changing the existing high-contrast
appearance:

```css
--status-page: #ffffff;
--status-surface: #f5f8fc;
--status-surface-raised: #ffffff;
--status-ink: #172033;
--status-muted: #536176;
--status-border: #c7d2e2;
--status-link: #0d3d66;
--status-use: #0b665c;
--status-route: #15558a;
--status-open: #7a4700;
--status-not-promoted: #7f2f42;
--status-unknown: #55506f;
```

Replace the hard-coded `#fff` failure-card background and direct Mermaid label
overrides with tokens/Quarto Mermaid variables.

- [x] **Step 4: Add redundant status and failure-trace styling**

Add outlined `.status-chip` variants for `use`, `route`, `open`,
`not-promoted`, and `unknown`. Add subtle left-border variants for evidence,
result, diagnosis, limit, and next-test fields. Preserve the visible field
heading in every case; never make color the only cue.

Use a two-layer `:focus-visible` treatment that remains visible on both page
modes. Give `summary` and the Quarto theme toggle a minimum 24-pixel target.
Add `@media (forced-colors: active)` and `@media print` rules, and preserve the
existing reduced-motion and 390-pixel stacking behavior.

- [x] **Step 5: Render an isolated shell check**

Run:

```bash
theme_check_dir=$(mktemp -d /tmp/trecvid-theme-check.XXXXXX)
mkdir -p "$theme_check_dir/docs"
cp -a docs/project-status "$theme_check_dir/docs/project-status"
cd "$theme_check_dir/docs/project-status"
env XDG_CACHE_HOME="$theme_check_dir/cache" quarto render . --to html
```

Expected: eleven HTML pages under `$theme_check_dir/reports/project-status`, a
visible Quarto color-scheme toggle in `index.html`, both light and dark theme
styles embedded, and no external font/resource warning.

- [x] **Step 6: Commit the shell and token system**

```bash
git add docs/project-status/_quarto.yml docs/project-status/theme.scss docs/project-status/theme-dark.scss docs/project-status/styles.css
git commit -m "docs: add accessible light and dark report themes"
```

---

### Task 2: Integrate legacy experiments into the six current themes

**Files:**
- Modify: `docs/project-status/experiment-ledger.qmd`

**Interfaces:**
- Consumes: the six existing `.theme-card` blocks and nineteen legacy IDs.
- Produces: unchanged old bookmark URLs that land on a current theme card, plus
  one compact historical appendix.

- [x] **Step 1: Move generic former anchors to the six-theme map**

Place zero-height aliases for `research-questions` and
`fig-experiment-timeline` immediately before the current six-theme map. Do not
add visible migration prose.

- [x] **Step 2: Group the remaining aliases inside current theme cards**

Use this exact primary mapping:

```text
Experiment integrity:
  trustworthy-experiments, field-guide, data-readiness,
  compatibility-gate, frozen-30-protocol-v2
Visual and temporal evidence:
  global-default, directed-temporal-sampling, dense-burst-sampling
Speech and transcript evidence:
  automatic-transcript-ablation, human-oracle-tier
Native audio and sound:
  audio-spatial-seamless
Models and adaptation:
  model-promotion, failure-panel, phi-gemma-semantic-stop,
  qwen3-omni-quality
Routing, fusion, and answer selection:
  specialist-evidence, taxonomy-audit
```

Render aliases for both GFM and HTML without creating duplicate HTML IDs. Each
card's visible description remains the meaningful destination.

- [x] **Step 3: Remove the visible compatibility index**

Delete the `Compatibility index for older links` heading and nineteen-item
migration list. Retain no second theme hierarchy.

- [x] **Step 4: Consolidate cross-cutting history**

Create `## History and provenance {.unlisted}` after result interpretation.
Nest the existing issue/PR trail, `What superseded what`, and `Chronology index`
under it. Preserve the existing `what-superseded-what` and `chronology-index`
fragments and keep detailed tables inside native disclosures.

- [x] **Step 5: Verify all old bookmarks in an isolated render**

Render `experiment-ledger.qmd` to GFM and HTML in a temporary copy. Check each
of the nineteen IDs exactly once in both outputs, check that authored and
legacy IDs are unique, report renderer-owned resource IDs separately, and
confirm `Compatibility index for older links` is absent from visible source
and output. Confirm the page contents omits History and provenance.

- [x] **Step 6: Commit the integrated history**

```bash
git add docs/project-status/experiment-ledger.qmd
git commit -m "docs: integrate legacy experiments into research themes"
```

---

### Task 3: Make failure records source-linked evidence traces

**Files:**
- Modify: `docs/project-status/failure-library.qmd`

**Interfaces:**
- Consumes: the four verified 2025 training source IDs and Task 1's trace/status
  classes.
- Produces: four visible case traces with original-video links, explicit media
  boundaries, failure location, bug status, and falsifying next tests.

- [x] **Step 1: Verify the source boundary before authoring**

Run:

```bash
test "$(readlink -f data)" = "/home/npatta01/data/competitions/trec_video_2026/data"
for id in NN7G28sBOlY NW3Mhr76ClM NrB4LPosWBk IWSPAfCKtV8; do
  jq -e '.video_url | startswith("https://www.youtube.com/watch?v=")' "data/vqa/2025/training/annotations/$id.json"
done
```

Expected: all four sources have platform-hosted watch URLs. The annotation
`timestamp` field is an ISO dataset timestamp, not a video offset, so it must
not become a `t=` link.

- [x] **Step 2: Correct the library-wide disclosure wording**

State that source IDs and original-video links are disclosed, while exact
questions, expected answers, transcripts, copied media, and within-video
offsets remain withheld or unrecorded under the recorded policy. Replace the
false claim that all source identifiers are withheld. Keep the selected-case,
non-prevalence warning above the cases.

- [x] **Step 3: Add one source-media band to every case**

Use these exact canonical links:

```text
CASE-001: https://www.youtube.com/watch?v=NN7G28sBOlY
CASE-002: https://www.youtube.com/watch?v=NW3Mhr76ClM
CASE-003: https://www.youtube.com/watch?v=NrB4LPosWBk
CASE-004: https://www.youtube.com/watch?v=IWSPAfCKtV8
```

Each band shows `Watch original video · YouTube`, the source ID, `2025 training
source`, `external source; media not redistributed`, and `Relevant offset: not
recorded in tracked evidence`. Do not embed a thumbnail or player.

- [x] **Step 4: Reorder each visible card as an evidence trace**

Keep the prediction, grade, competing explanation, limit, and next test already
present. Add visible `Evidence received`, `Possible failure location`, and `Bug
or audit status` fields:

```text
CASE-001 — answer selection/output; post-hoc diagnosis, no confirmed defect.
CASE-002 — evidence extraction plus gate/routing; confirmed defect in the tested
           generic gate, with repair verification and a dedicated non-lexical
           check or route still open.
CASE-003 — fusion/evidence use or task instruction; post-hoc diagnosis,
           no confirmed defect.
CASE-004 — answer specificity/evidence use; a separate container mount-path
           defect was fixed before the recorded graded result.
```

Use `case-field-evidence`, `case-field-result`, `case-field-diagnosis`,
`case-field-limit`, and `case-field-next` classes from Task 1.

- [x] **Step 5: Reconcile each expert disclosure**

Add Source URL, source availability, relevant-offset status, and link-only media
status to every details table. Preserve model revisions, frame/audio/transcript
conditions, frozen predictions, grades, rater/blinding gaps, immutable report
links, and supersession statements. Keep `Model observation: Not recorded`; do
not invent hidden reasoning.

- [x] **Step 6: Render and inspect the library alone**

Render `failure-library.qmd` to GFM and HTML in a temporary copy. Confirm all
four source links appear, no source ID appears in an `img`, `iframe`, `video`,
`audio`, `source`, or `poster` attribute, and all visible failure claims retain
their condition and `n=1` boundary.

- [x] **Step 7: Commit the linked failure traces**

```bash
git add docs/project-status/failure-library.qmd
git commit -m "docs: link failure cases to source videos"
```

---

### Task 4: Generate, review, and refresh the private report

**Files:**
- Generate: all eleven `docs/project-status/*.md`
- Generate: all eleven `reports/project-status/*.html`
- Modify: this plan with completed checkboxes and verification evidence
- Refresh: eleven sanitized HTML files in the existing private portal directory

**Interfaces:**
- Consumes: Tasks 1–3 source changes.
- Produces: deterministic GitHub Markdown, self-contained light/dark HTML, and a
  verified tailnet-only preview for user review.

- [x] **Step 1: Render all canonical outputs**

From the repository root, run:

```bash
render_cache=$(mktemp -d /tmp/trecvid-report-render.XXXXXX)
for source in docs/project-status/*.qmd; do
  page=$(basename "$source" .qmd)
  env XDG_CACHE_HOME="$render_cache/cache" quarto render "$source" --to gfm --output-dir docs/project-status --output "$page.md"
done
env XDG_CACHE_HOME="$render_cache/cache" quarto render docs/project-status --to html
```

Expected: exactly eleven QMD, eleven sibling Markdown, and eleven HTML files.
Do not hand-edit generated output.

- [x] **Step 2: Prove deterministic regeneration**

Create a fresh temporary directory with `docs/project-status/` at the same
relative depth as the repository, render all eleven GFM and HTML outputs there
with a separate task-specific cache, and compare all twenty-two generated files
byte-for-byte using `cmp`. Stop and investigate any mismatch before portal
refresh.

- [x] **Step 3: Run focused source and output checks**

Verify:

```text
19/19 legacy IDs resolve exactly once in experiment-ledger HTML.
4/4 source-video links exist in QMD, Markdown, and HTML.
0 copied/embedded media elements reference any of the four source IDs.
0 remote runtime src/poster/import URLs exist in generated HTML/CSS.
0 duplicate authored/content/interactive/case/legacy ID groups exist; report
renderer-owned framework resource-ID counts separately.
0 unresolved repository-relative links or fragments exist.
```

Run `git diff --check` and scan QMD, Markdown, and HTML for credential patterns,
bucket URIs, local filesystem paths/URLs, raw-log fragments, and 2026 evaluation
identifiers. External `href` links to GitHub, Quarto, the Copyright Office, and
the four YouTube watch pages are intentional.

- [x] **Step 4: Inspect representative light/dark desktop and phone views**

Use authorized headless Chrome because the in-app browser has no automation
surface. Capture Start Here, Research Themes, Failure Library, one theme page,
and Code Map at `1440×1000` and `390×844`, once with normal preference and once
with `--force-dark-mode`. Inspect navigation, toggle, contrast, focus,
disclosures, card stacking, overflow, history placement, and the failure trace.
Use the user's current screenshots as the visual baseline; do not redesign
unrelated layout.

- [x] **Step 5: Obtain an independent review before publishing**

Give a fresh reviewer the approved spec, source diff, rendered pages, and
verification evidence. Require findings to distinguish critical accessibility
or evidence errors, important usability issues, and optional polish. Resolve
all critical/important findings and rerun the affected render/check before the
portal is refreshed.

- [x] **Step 6: Refresh only the authorized portal copy**

Copy the eleven sanitized HTML files to
`/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`. Replace the
current portal-index description with:

```text
Start-here guidance, a synthetic worked example, six research-theme records,
source-linked public-safe failure traces, accessible light/dark presentation,
and a code map with remaining research and engineering gaps.
```

Do not remove any uncertain or user-owned artifact.

Verify all eleven live HTTPS routes return `200` and match the repository HTML
bytes. Confirm `tailscale serve status` and `tailscale funnel status` still say
tailnet-only and that no Serve mapping changed.

- [x] **Step 7: Record evidence and commit the completed refinement**

Update this plan with render counts, deterministic comparison, fragment/link
results, media/privacy scan, responsive light/dark scenarios, independent
review outcome, portal byte matches, and tailnet-only status.

Stage only the plan, intended source files, eleven Markdown files, and eleven
HTML files. Confirm no `data/` path is staged, then commit:

```bash
git commit -m "docs: publish accessible source-linked status report"
```

**Earlier Task 4 status:** The then-scoped fresh re-review approved private
publication with zero Critical and zero Important findings; the authorized
portal refresh, live-byte checks, and final commit then completed that scope. A
later holistic review found the evidence contradiction recorded below, so that
earlier verdict is not the final review state for the current sources.

**Pre-publication verification evidence:**

- Quarto `1.9.38` rendered exactly 11 QMD sources to 11 sibling GFM files and
  11 self-contained HTML files with task-specific caches. Both the original
  pass and the review-fix pass were reproduced from physically separate copies;
  every pass matched all 22 generated files byte-for-byte (`11/11` GFM and
  `11/11` HTML). No generated output was hand-edited, and Quarto's `.quarto`,
  nested `docs/`, `.gitignore`, `search.json`, and `site_libs` transients were
  removed from the repository worktree.
- The focused audit resolved all 19 legacy experiment-ledger IDs exactly once.
  All four approved YouTube watch URLs occur in QMD, GFM, and HTML; each occurs
  once in the visible source-media band and once in the expert disclosure on
  every surface. No media element references a source ID, and generated HTML
  contains no remote runtime `src`, `poster`, `srcset`, stylesheet link, CSS
  import, or CSS resource URL.
- Authored/content, interactive-control, case, and all 19 legacy IDs are
  unique. Quarto `1.9.38`'s native dual-theme renderer separately emits the
  framework stylesheet resource IDs `quarto-bootstrap` and
  `quarto-text-highlighting-styles` three times on every page. This known
  renderer behavior is not an authored fragment collision; the current result
  must not be summarized as literal zero duplicate IDs.
- Repository-relative resolution checked 149 QMD links/fragments, 149 GFM
  links/fragments, and 280 rendered-HTML links/fragments with zero unresolved.
  Credential, bucket-URI, local-path/URL, raw-log-fragment, and 2026 evaluation
  identifier scans found zero matches. `git diff --check` passes.
- Authorized Chrome `150` captured Start Here, Research Themes, Failure
  Library, Visual and Temporal Evidence, and Code Map at `1440x1000` and
  `390x844` in light and dark modes (`20/20` scenarios). All scenarios retained
  the three-item navbar, correct mode, at least 24-pixel theme/disclosure
  controls, one-column phone stacking, and zero page overflow. Keyboard Tab
  focus produced the intended two-layer ring in both modes. The History and
  provenance section is absent from the TOC but remains a valid deep-link
  target below the fixed navbar.
- The first independent review reported zero Critical and two Important
  usability/evidence findings: provenance labels collapsed at phone width, and
  the visible failure traces omitted explicit `What failed` and `Change tried`
  steps. The focused fix preserves a usable first-column width (`133.125px` in
  the `390px` Chrome scenario), normal label wrapping, and zero page overflow;
  all four visible traces now follow the nine-field evidence order while
  preserving source boundaries, conditions, predictions, grades, diagnoses,
  provenance, limits, and next tests. Fresh re-review reported zero Critical
  and zero Important findings and approved the portal refresh.
- Only the 11 sanitized repository HTML files were copied into the existing
  authorized portal directory; its 11 filesystem copies match the repository
  byte-for-byte. The canonical Start Here directory route and ten named HTML
  routes all returned HTTP 200 and matched repository bytes (`11/11`). The
  explicit `index.html` spelling returns HTTP 301 to the canonical directory
  route. The portal root returned 200 and contains the exact approved
  description.
- Before and after the copy, `tailscale serve status` and
  `tailscale funnel status` reported the same single mapping:
  `https://npatta01-framework.tail481212.ts.net (tailnet only)` with `/` served
  from `/home/npatta01/codex-rendered`. No Serve mapping was changed, Funnel
  was not enabled, and no public listener was introduced.

**Later holistic-review correction (2026-07-26):**

- The later review found one Important contradiction in CASE-002: the report
  claimed the tested generic-gate defect was fixed and superseded, while the
  immutable issue #7 evidence only confirms that the tested gate incorrectly
  accepted an unrelated outro phrase and records a dedicated non-lexical check
  as future work. The corrected source records the confirmed tested-gate defect,
  states that no repair or supersession evidence is recorded, and leaves repair
  verification plus a dedicated non-lexical-vocalization check or route open.
- The same review found one Minor stale numeric record-count phrase after the
  case traces had expanded. The hub now uses the durable description `full
  evidence traces and disclosure boundary`.
- Fresh scoped re-review reported zero Critical, zero Important, and zero Minor
  findings and approved the private portal refresh.
- The refresh replaced only `experiment-ledger.html` and
  `failure-library.html`. The authorized directory still contains exactly 11
  HTML files, and all 11 filesystem copies match the repository outputs
  byte-for-byte. The portal index was not modified; its SHA-256 remained
  `1597e111f1a1ea6e67f44d54f4bf33f083ac9efbc8cc32183d6fee9a53f57c8e`,
  and its approved description remained unchanged.
- The canonical Start Here directory route and all ten named HTML routes
  returned HTTP 200 and matched the repository bytes (`11/11`). The portal root
  returned HTTP 200 and contained the unchanged approved description.
- Before and after the refresh, `tailscale serve status` and
  `tailscale funnel status` reported the same single mapping:
  `https://npatta01-framework.tail481212.ts.net (tailnet only)` with `/` served
  from `/home/npatta01/codex-rendered`. No Serve mapping command was run, Funnel
  was not enabled, and no public listener was introduced.
- **Final commit status:** Completed after the post-plan diff check and exact
  seven-path stage audit as the local commit
  `docs: correct transcript gate evidence status`. No push, pull request, or
  issue action was performed.

## Success check

- A newcomer can scan each page in either light or dark mode without color
  being the only status signal.
- An old bookmark lands on the relevant current theme card, not a migration
  list.
- Every failure case exposes the original source link, what evidence the system
  received, what it predicted, where the failure might be, known bug/audit
  status, and the smallest falsifying next test.
- The report redistributes no source-video pixels or audio and embeds no remote
  runtime resource.
- GitHub Markdown, repository HTML, and the private portal are deterministic,
  current, and tailnet-only.
