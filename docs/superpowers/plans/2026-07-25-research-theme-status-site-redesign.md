# Research-Theme Status Site Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the long decision-question Experiment Ledger with a concise Research Themes hub and six focused, evidence-backed pages that connect completed experiments, condition-bound conclusions, validity risks, and next experiments.

**Architecture:** Keep the existing three top-level navigation choices, preserving `experiment-ledger` as the Research Themes hub URL. Add six self-contained QMD theme pages; use native disclosure controls for audit depth, a semantic responsive card grid instead of a linear Mermaid path, and a secondary expandable decision diagram on Start Here. QMD is the sole editable report source; sibling GFM and self-contained HTML are generated outputs.

**Tech Stack:** Quarto 1.9.x, Pandoc GFM, native HTML `<details>/<summary>`, Mermaid, repository CSS, self-contained HTML, headless Chrome for responsive/accessibility verification.

## Global Constraints

- Work only on `codex/repository-cleanup-report`; do not switch or modify detached or issue worktrees.
- Historical reports and plans remain at their current paths.
- The ignored `data` symlink must continue to resolve to `/home/npatta01/data/competitions/trec_video_2026/data`; never read, copy, move, delete, stage, or commit dataset contents.
- QMD files are the sole editable source for report prose. Markdown and HTML are generated outputs.
- Keep three top-level navigation choices: Start Here, Research Themes, and Code Map and Gaps.
- Add no custom JavaScript, structured ledger schema, application test framework, CI, or dependency.
- Do not run the application test suite; this is a documentation-only revision.
- Use native disclosures only when the same QMD remains accessible in HTML and GitHub Markdown. Leave content expanded if the cross-format check fails.
- Keep conclusions, statuses, and next-test summaries visible outside closed disclosures.
- Replace permanent-sounding negative language with condition-bound language. Name the tested panel, implementation, model, prompt, budget, or gate that bounds the conclusion.
- Classify validity concerns as `Confirmed defect`, `Audit gap`, or `Uncertainty`; never present an audit gap as a confirmed bug.
- Preserve all existing aggregates, panel sizes, generalization limits, eligibility qualifications, evidence links, and legacy fragment targets.
- Keep dense-burst evidence aggregate-only; expose no row IDs, answers, predictions, timestamps, paths, or raw artifacts.
- Keep all generated pages sanitized: no private frames, transcripts, predictions, media, datasets, raw logs, local paths, credentials, or bucket URIs.
- Publish only sanitized HTML through the existing tailnet-only portal after final verification. Do not use Funnel, a public listener, Codex Sites, or new sharing permissions.
- GitHub issue creation remains out of scope until the revised report is reviewed.

---

## File structure

### New canonical QMD sources

- `docs/project-status/research-integrity.qmd` — protocol validity, reproducibility, scoring, and panel-design evidence.
- `docs/project-status/research-visual-evidence.qmd` — frame budgets, temporal selection, spatial evidence, and selector auditability.
- `docs/project-status/research-speech-transcripts.qmd` — automatic, reference, multilingual, and human-transcript evidence tiers.
- `docs/project-status/research-native-audio.qmd` — prosody, non-speech sound, music, and matched audio-on/off evidence.
- `docs/project-status/research-models.qmd` — model compatibility, semantic quality, promotion gates, and adaptation ideas.
- `docs/project-status/research-routing.qmd` — taxonomy, routing, retrieval-versus-fusion, grounding, fallback, and answer election.

### Modified canonical sources

- `docs/project-status/experiment-ledger.qmd` — concise Research Themes hub, priority map, compatibility index, supersession, and chronology.
- `docs/project-status/index.qmd` — condition-bound wording and expandable full decision diagram.
- `docs/project-status/code-map.qmd` — replace detached research seeds with links into the six theme pages without duplicating their backlog.
- `docs/project-status/_quarto.yml` — render the six new QMD files and relabel the hub navigation item `Research Themes`.
- `docs/project-status/styles.css` — responsive theme cards, disclosures, status labels, and expanded-diagram scrolling/focus.

### Generated outputs

- Create sibling Markdown for all six new theme sources.
- Regenerate `docs/project-status/index.md`, `experiment-ledger.md`, and `code-map.md`.
- Create six new self-contained HTML pages under `reports/project-status/`.
- Regenerate the three existing HTML pages under `reports/project-status/`.

---

### Task 1: Integrity, visual-evidence, and speech/transcript theme pages

**Files:**
- Create: `docs/project-status/research-integrity.qmd`
- Create: `docs/project-status/research-visual-evidence.qmd`
- Create: `docs/project-status/research-speech-transcripts.qmd`

**Interfaces:**
- Consumes: the approved design at `docs/superpowers/specs/2026-07-25-research-theme-status-site-redesign.md`; existing evidence files listed below; hub path `experiment-ledger.{{< meta report-ext >}}`.
- Produces: three standalone QMD pages with stable experiment anchors, visible condition-bound summaries, native evidence disclosures, and theme-specific next experiments. Task 3 links to these exact filenames and anchors.

- [ ] **Step 1: Verify scope and evidence sources**

Run:

```bash
git status --short --branch --untracked-files=all
readlink -f data
```

Expected: clean `codex/repository-cleanup-report`; `data` resolves to the canonical shared path. Do not inspect dataset contents.

Read these tracked sources before drafting claims:

```text
reports/trecvid_vqa_2026/three_video_gate_results_2026-07-20.md
reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md
reports/trecvid_vqa_2026/nist_scorer_reuse_audit_2026-07-20.md
docs/experiments/temporal-sampling-issue-8.md
src/trec_video_2026/vqa_benchmark.py
src/trec_video_2026/vqa_experiments/temporal_sampling.py
tests/test_temporal_sampling.py
reports/trecvid_vqa_2026/audio_spatial_followup_results_2026-07-21.md
reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md
reports/trecvid_vqa_2026/external_gold_transcript_benchmark_research_2026-07-24.md
reports/trecvid_vqa_2026/asr_model_followup_options_2026-07-20.md
reports/trecvid_vqa_2026/multilingual_qwen36_followup_2026-07-24.md
reports/trecvid_vqa_2026/seamless_audio_followup_results_2026-07-21.md
plans/issue-7-automatic-transcript-ablation.md
```

- [ ] **Step 2: Create the integrity and evaluation page**

Use this YAML header and page spine:

```yaml
---
title: "Experiment Integrity and Evaluation"
description: "What makes a TRECVID VQA experiment interpretable, which protocol defects were confirmed, and what integrity work remains."
---
```

Visible sections, in order:

```text
Back to Research Themes
Technical summary
Why this theme matters
Current working answer
What has been tried
What the evidence supports
What it does not rule out
Possible bugs, confounds, and audit gaps
Open hypotheses and ideas
Cheapest valid next experiments
Related themes
```

Include the July 12 field guide (`field-guide`), data readiness
(`data-readiness`), three-video compatibility gate (`compatibility-gate`),
frozen-30 v1/v2 protocol transition, and scorer-reuse audit. Keep these exact
distinctions visible:

- frozen-30 v1 label isolation is a **Confirmed defect**;
- protocol v2 replaces v1 for comparative conclusions;
- reproducible execution and compatibility do not establish answer quality;
- a reusable integrity gate is **Open work**, not an existing supported CLI.

Define two next experiments/work items using the full five-field contract:

1. reusable `prepare → run → freeze → score → report` integrity gate with
   deliberately invalid fixtures;
2. representative-panel and scorer-validation gate before expensive runs.

- [ ] **Step 3: Create the visual and temporal evidence page**

Use title `Visual and Temporal Evidence` and description `What frame budgets,
temporal selectors, spatial crops, and retrieval diagnostics have—and have
not—established.`

Include `frozen-30-protocol-v2`, `directed-temporal-sampling`,
`dense-burst-sampling`, and the spatial-crop diagnostic. Explain before using
the names:

- uniform-12 takes midpoint frames from 12 equal time bins;
- uniform-24 applies the same whole-video coverage with twice the frame proxy;
- directed-12 retains broad context and reallocates the rest of the 12-frame
  budget toward a keyword/anchor-selected region;
- dense-burst-12 is only recoverable as a two-row aggregate; its exact tracked
  implementation and schedule are absent.

For directed-12, retain the ten-row `2C/5P/3W` tie with uniform-12 and the
uniform-24 `4C/3P/3W` comparison. Label the missing tracked general-workflow
caller as an **Audit gap**, not a confirmed execution bug. Explain that the
selector's tests establish deterministic timestamps, not required-evidence
recall or answer relevance.

For dense-burst-12, retain uniform-24 `2C/0P/0W` versus burst `0C/1P/1W` and
the two-row boundary. State that the result rejects that tested diagnostic as a
default but cannot estimate prevalence, effect size, or all local selectors.

Define three next experiments:

1. required-moment labels plus selector recall and oracle-frame answer quality;
2. fixed-budget uniform, query-only, half-uniform/half-query, and coarse-to-fine
   comparison with identical model/prompt/decoding controls;
3. category-routed spatial-detail/OCR diagnostic rather than global tiling.

- [ ] **Step 4: Create the speech and transcript evidence page**

Use title `Speech and Transcript Evidence` and description `What automatic,
reference, multilingual, and human transcripts can contribute without
confusing their provenance.`

Include the automatic transcript ablation (`automatic-transcript-ablation`),
multilingual/translation follow-ups, Seamless diagnostic, external
dataset-reference research, and blocked human tier (`human-oracle-tier`).
Retain the ten-row automatic-transcript change from `4/10` to `7/10`, three
rescues, and no observed degradation on that selected lexical-speech panel.

Define the four provenance tiers before comparing them:

```text
automatic ASR output
uploader caption
dataset recording-prompt reference
independently produced human transcript
```

Do not call the external dataset-reference protocol a TRECVID human oracle or
an effectiveness result. Keep the strict human tier blocked until independent
transcriber/reviewer records exist.

Define three next experiments:

1. accepted Whisper with current prompt versus accepted Whisper with a required
   timestamp/evidence-span grounding prompt;
2. label-blind human-transcript upper bound with independent provenance;
3. alternative/multilingual ASR only on a matched frozen panel with the same
   downstream frames/model/prompt and explicit null/rejection controls.

- [ ] **Step 5: Verify the three source pages without creating repository outputs**

Copy the project-status source directory to a task-specific temporary directory
so Quarto project resolution cannot write an output into the repository:

```bash
theme_task1_dir=$(mktemp -d /tmp/trecvid-theme-task1.XXXXXX)
cp -a docs/project-status "$theme_task1_dir/source"
cd "$theme_task1_dir/source"
```

In the same shell session, render each source with these exact relative paths.
Expected: all three commands exit 0 and write only under the temporary
directory.

```bash
env XDG_CACHE_HOME="$theme_task1_dir/cache" quarto render research-integrity.qmd --to gfm --output-dir ../gfm --output research-integrity.md
env XDG_CACHE_HOME="$theme_task1_dir/cache" quarto render research-visual-evidence.qmd --to gfm --output-dir ../gfm --output research-visual-evidence.md
env XDG_CACHE_HOME="$theme_task1_dir/cache" quarto render research-speech-transcripts.qmd --to gfm --output-dir ../gfm --output research-speech-transcripts.md
```

Inspect all three temporary files. Confirm every page has one H1-equivalent
title, all nine page-contract sections, visible status/boundary text, disclosure
summaries, and repository-relative evidence links. Run `git diff --check`.

- [ ] **Step 6: Commit Task 1**

```bash
git add docs/project-status/research-integrity.qmd docs/project-status/research-visual-evidence.qmd docs/project-status/research-speech-transcripts.qmd
git commit -m "docs: add integrity visual and speech research themes"
```

---

### Task 2: Native-audio, model, and routing theme pages

**Files:**
- Create: `docs/project-status/research-native-audio.qmd`
- Create: `docs/project-status/research-models.qmd`
- Create: `docs/project-status/research-routing.qmd`

**Interfaces:**
- Consumes: the approved design; existing evidence files listed below; hub path `experiment-ledger.{{< meta report-ext >}}`; cross-theme filenames from Task 1.
- Produces: three standalone QMD pages with exact theme-page contracts and links that Task 3 exposes from the hub.

- [ ] **Step 1: Read the source evidence before drafting**

```text
reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md
reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md
reports/trecvid_vqa_2026/audio_spatial_followup_results_2026-07-21.md
reports/trecvid_vqa_2026/seamless_audio_followup_results_2026-07-21.md
reports/trecvid_vqa_2026/taxonomy_prior_art_and_open_omni_models_2026-07-24.md
reports/trecvid_vqa_2026/three_video_gate_results_2026-07-20.md
docs/experiments/2026-07-24-nonqwen-omni-compatibility.md
docs/experiments/2026-07-24-omni-semantic-quality.md
docs/experiments/2026-07-25-qwen3-omni-quality-10.md
docs/experiments/2026-07-25-qwen3-omni-quality-10-neutral-a2.md
plans/issue-17-qwen3-omni-quality-10.md
reports/trecvid_vqa_2026/queries/literature_review.sql
docs/superpowers/plans/2026-07-24-evidence-routing-benchmark.md
```

- [ ] **Step 2: Create the native audio and sound page**

Use title `Native Audio and Sound` and description `What matched audio-on/off
diagnostics say about speech, prosody, sound events, music, and specialist
routing.`

Include the matched MiniCPM audio-on/off evidence, the failure-selected panel's
prosody/non-speech transitions, sound/music failures, and relevant Qwen3-Omni
audio evidence. Cross-link model-specific promotion details to
`research-models.{{< meta report-ext >}}` instead of duplicating their full
entry.

State visibly that existing evidence rejects a global audio-on switch under the
tested panels, not category-routed native audio. Separate lexical speech,
prosody/emotion, non-speech events, and exact music-title retrieval.

Define three next experiments:

1. matched audio-off/audio-on specialist panel with speech, prosody,
   non-speech, temporal, and visual-control strata;
2. non-speech audio classifier with abstention and an audio-off/native-VLM
   control;
3. exact-music fingerprint/retrieval route that abstains rather than asking a
   general VLM to guess.

- [ ] **Step 3: Create the models and adaptation page**

Use title `Models and Adaptation` and description `Which model candidates ran,
which quality gates they met, and what a valid reconsideration would require.`

Include the compatibility gate, failure-selected Qwen3.6 panel
(`failure-panel`), Phi/Gemma semantic screen (`phi-gemma-semantic-stop`),
MiniCPM fast-alternate role, and Qwen3-Omni compatibility/quality evidence
(`qwen3-omni-quality`).

Retain exact small-panel boundaries. Rephrase Qwen3-Omni as: the unchanged
strict run completed 19/20 outputs and was therefore ineligible under that
predeclared 20/20 promotion gate; the model remains a candidate for a repaired,
complete, blinded matched run. Rephrase Qwen3.6, Phi, and Gemma as not promoted
under their tested versions/panels, not permanently disqualified model families.

Define three next experiments:

1. repaired, complete, neutral Qwen3-Omni run with the strict contract;
2. representative matched specialist-model panel after compatibility;
3. adaptation/fine-tuning only after a measured retrieval, fusion, or answer-
   selection bottleneck justifies it.

- [ ] **Step 4: Create the routing, fusion, and answer-selection page**

Use title `Routing, Fusion, and Answer Selection` and description `How to decide
which evidence path to invoke and distinguish missing evidence from failure to
use evidence correctly.`

Include the taxonomy audit (`taxonomy-audit`), current router concept, lower-
bound lexical-tag limitation, transcript residuals where evidence was present
but the VLM selected the wrong answer, and the retrieval-versus-fusion
distinction from the tracked literature synthesis.

Define retrieval failure and fusion/reasoning failure in plain language. State
that the repository has route seeds, not a versioned evaluated router. Require
abstention to fall back to the baseline and route-level non-regression before a
specialist becomes a default policy.

Define three next experiments:

1. versioned label-blind router with multi-label tags, held-out validation,
   abstention, and baseline fallback;
2. required-evidence labels plus selector recall, oracle-evidence answer quality,
   and end-to-end answer quality;
3. answer grounding/election or a targeted ensemble only after matched evidence
   paths are frozen.

- [ ] **Step 5: Verify the three source pages in temporary GFM**

```bash
theme_task2_dir=$(mktemp -d /tmp/trecvid-theme-task2.XXXXXX)
cp -a docs/project-status "$theme_task2_dir/source"
cd "$theme_task2_dir/source"
```

In the same shell session, run:

```bash
env XDG_CACHE_HOME="$theme_task2_dir/cache" quarto render research-native-audio.qmd --to gfm --output-dir ../gfm --output research-native-audio.md
env XDG_CACHE_HOME="$theme_task2_dir/cache" quarto render research-models.qmd --to gfm --output-dir ../gfm --output research-models.md
env XDG_CACHE_HOME="$theme_task2_dir/cache" quarto render research-routing.qmd --to gfm --output-dir ../gfm --output research-routing.md
```

Expected: all commands exit 0; disclosures preserve their summaries and inner
Markdown; no raw/private artifact reference appears; repository worktree shows
only the three intended QMD files. Run `git diff --check`.

- [ ] **Step 6: Commit Task 2**

```bash
git add docs/project-status/research-native-audio.qmd docs/project-status/research-models.qmd docs/project-status/research-routing.qmd
git commit -m "docs: add audio model and routing research themes"
```

---

### Task 3: Research hub, navigation, disclosures, and expandable diagram

**Files:**
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify: `docs/project-status/index.qmd`
- Modify: `docs/project-status/code-map.qmd`
- Modify: `docs/project-status/_quarto.yml`
- Modify: `docs/project-status/styles.css`

**Interfaces:**
- Consumes: the six exact theme filenames from Tasks 1–2.
- Produces: three stable top-level navigation destinations, a non-linear hub, all legacy fragment targets, cross-theme links, native disclosures, and reusable CSS classes consumed by generated HTML.

- [ ] **Step 1: Replace the linear ledger with the Research Themes hub**

Keep the source path `experiment-ledger.qmd`; change the visible title to
`Research Themes and Experiment Ledger` and description to an answer-first
summary of tried evidence, uncertainty, and next experiments.

Remove the linear Mermaid research path. Add a `.theme-grid` containing six
`.theme-card` fenced divs. Each card must include:

```text
theme name and real link
current working answer
strongest evidence level
most important open question
```

Use links to these exact targets:

```text
research-integrity.{{< meta report-ext >}}
research-visual-evidence.{{< meta report-ext >}}
research-speech-transcripts.{{< meta report-ext >}}
research-native-audio.{{< meta report-ext >}}
research-models.{{< meta report-ext >}}
research-routing.{{< meta report-ext >}}
```

Follow the grid with six concise priorities, one per theme. Do not repeat the
full backlog from theme pages.

- [ ] **Step 2: Add condition-bound status and validity guidance**

Add an always-visible section defining:

```text
Current default
Fast alternate
Routed candidate
Diagnostic signal
Not promoted under the tested condition
Blocked
Superseded
Open hypothesis
```

Add a three-item validity classification:

```text
Confirmed defect — demonstrated fault
Audit gap — tracked evidence cannot prove the exact path
Uncertainty — small/selected panel or generalization boundary without bug evidence
```

Use frozen-30 v1, directed integration, and small diagnostic panels as the
respective examples.

- [ ] **Step 3: Preserve legacy anchors in a compatibility index**

Every legacy fragment must exist exactly once on the new hub and visibly link
to its theme-page replacement:

```text
trustworthy-experiments
global-default
specialist-evidence
model-promotion
research-questions
fig-experiment-timeline
field-guide
data-readiness
compatibility-gate
frozen-30-protocol-v2
directed-temporal-sampling
dense-burst-sampling
audio-spatial-seamless
automatic-transcript-ablation
taxonomy-audit
human-oracle-tier
failure-panel
phi-gemma-semantic-stop
qwen3-omni-quality
what-superseded-what
chronology-index
```

Use stable HTML/GFM-compatible anchor elements or the existing format-specific
fallback pattern. The compatibility index text must say where the record moved;
do not add JavaScript redirects.

Keep the six-row supersession information and five-row July 12/20/21/24/25
chronology. Put each inside a native disclosure while keeping its section
heading/anchor and one-sentence purpose visible.

- [ ] **Step 4: Make the Start Here decision diagram secondary and expandable**

Keep the readable status table as the primary decision interface. Wrap the
dense Mermaid current-decision diagram in a native disclosure with summary
`Expand the full decision diagram`. Place the Mermaid in a
`.diagram-scroll` fenced div with `tabindex="0"` and an accessible label. Keep
caption, `fig-alt`, Mermaid `accTitle`/`accDescr`, explanatory paragraph, and
the visible table equivalent.

Replace standalone `Stop as current defaults` wording with `Not promoted as
current defaults under the tested conditions`. Describe the Qwen3-Omni strict
run as gate-specific ineligibility, not permanent model ineligibility.

- [ ] **Step 5: Update site navigation and Code Map links**

In `_quarto.yml`, retain the three navbar items but change the second label to
`Research Themes`. Keep `render: ["*.qmd"]`, which includes the new pages.

In `code-map.qmd`, keep engineering/documentation gaps. Replace the detached
research-seed bullets with a short link list to the six theme pages and state
that exact hypotheses, controls, and decision rules live there. Do not duplicate
the research backlog.

- [ ] **Step 6: Add responsive, accessible CSS**

Add styles with these responsibilities:

```css
.theme-grid         /* two equal columns, gap, normal document width */
.theme-card         /* bordered panel, no fixed height, clear heading */
.theme-card:focus-within
.theme-status       /* compact plain-language status */
details             /* border, spacing, readable open state */
summary             /* pointer cursor, font weight, focus-visible outline */
.diagram-scroll     /* overflow-x:auto, focusable outline */
.diagram-details .mermaid-js svg /* legible minimum width while expanded */
```

At `max-width: 767px`, set `.theme-grid` to one column. Do not apply a fixed
height to cards. Preserve the existing reduced-motion rule and existing table
wrapping behavior.

- [ ] **Step 7: Render a temporary HTML site and inspect interactions**

Copy `docs/project-status/` into a temporary directory, render there with a
task-specific `XDG_CACHE_HOME`, and inspect all nine pages without modifying
tracked generated outputs.

At 1440 and 390 pixels verify:

- hub grid is two columns then one;
- all disclosures are keyboard operable and their summaries describe content;
- headings and conclusions remain visible when disclosures are closed;
- the expanded decision diagram is legible and horizontally scrollable without
  causing page-level overflow;
- old hub fragments exist exactly once and point to the correct new pages;
- every page has exactly one H1 and no forced new window.

Run `git diff --check`.

- [ ] **Step 8: Commit Task 3**

```bash
git add docs/project-status/experiment-ledger.qmd docs/project-status/index.qmd docs/project-status/code-map.qmd docs/project-status/_quarto.yml docs/project-status/styles.css
git commit -m "docs: center status site on research themes"
```

---

### Task 4: Generate and verify Markdown and self-contained HTML

**Files:**
- Create: `docs/project-status/research-integrity.md`
- Create: `docs/project-status/research-visual-evidence.md`
- Create: `docs/project-status/research-speech-transcripts.md`
- Create: `docs/project-status/research-native-audio.md`
- Create: `docs/project-status/research-models.md`
- Create: `docs/project-status/research-routing.md`
- Modify: `docs/project-status/index.md`
- Modify: `docs/project-status/experiment-ledger.md`
- Modify: `docs/project-status/code-map.md`
- Create: six matching `reports/project-status/research-*.html` files
- Modify: `reports/project-status/index.html`
- Modify: `reports/project-status/experiment-ledger.html`
- Modify: `reports/project-status/code-map.html`

**Interfaces:**
- Consumes: all committed QMD/CSS/config sources from Tasks 1–3.
- Produces: the complete GitHub-readable and browser-readable site consumed by Task 5.

- [ ] **Step 1: Render every QMD to sibling GFM**

Use one task-specific Quarto cache. Render these nine sources with output names
matching their basenames:

```text
index
experiment-ledger
code-map
research-integrity
research-visual-evidence
research-speech-transcripts
research-native-audio
research-models
research-routing
```

Render the exact nine-name list with this loop:

```bash
for theme_page in index experiment-ledger code-map research-integrity research-visual-evidence research-speech-transcripts research-native-audio research-models research-routing; do
  env XDG_CACHE_HOME=/tmp/trecvid-theme-quarto-cache quarto render "docs/project-status/${theme_page}.qmd" --to gfm --output-dir docs/project-status --output "${theme_page}.md"
done
```

Remove only Quarto transients created by the GFM render, never canonical source
or generated deliverables.

- [ ] **Step 2: Render the complete self-contained HTML site**

```bash
env XDG_CACHE_HOME=/tmp/trecvid-theme-quarto-cache quarto render docs/project-status --to html
```

An optional Google Fonts fetch warning is acceptable in the network-restricted
environment only if generated HTML contains no Google Fonts URL and the browser
records no external resources. Remove `search.json`, `site_libs`, `.quarto`, or
other transient outputs not part of the declared self-contained site.

- [ ] **Step 3: Run source/generated consistency and content-preservation checks**

Compare the pre-redesign ledger at commit `7a837db` with the nine final
QMD/GFM/HTML pages. Confirm all thirteen historical entries, exact panels,
aggregates, decisions, boundaries, and evidence links remain represented.
Allow the former mixed audio/spatial/Seamless entry to be split across themes,
but do not duplicate its quantitative claims.

Confirm these sensitive facts remain exact:

```text
frozen-30: 30 rows, four arms, 120 predictions, 30 Whisper executions
automatic transcript: 4/10 to 7/10, three rescues, no observed degradation
directed temporal: ten rows; 2C/5P/3W tie with uniform-12; uniform-24 4C/3P/3W
dense burst: two rows; uniform-24 2C/0P/0W; burst-12 0C/1P/1W
Phi/Gemma: three-row screen and exact C/P/W boundaries
Qwen3-Omni: strict 19/20 completion plus diagnostic 3C/3P/4W to 5C/2P/3W
human transcript: blocked pending independent provenance
2026 answer quality: unknown
```

- [ ] **Step 4: Check links, fragments, disclosures, visuals, and privacy**

Run a repository-relative link check across changed QMD/Markdown files. Check
all legacy fragments in GFM and HTML and reject missing or duplicate targets.

At desktop and 390 pixels, inspect all nine pages in headless Chrome. Require:

- one H1 per page;
- no document-level horizontal overflow;
- no unreadably compressed diagram labels;
- zero external runtime resources;
- keyboard-operable disclosure summaries;
- hub grid 2→1 columns;
- expanded diagram scroll contained inside its region;
- the four retained Mermaid visuals preserve caption, alternative text,
  accessible SVG title/description, explanatory prose, and text equivalent;
- the semantic theme-card grid remains complete and operable without CSS.

Scan the nine QMD, nine Markdown, and nine HTML files for local paths, dataset
links, bucket URIs, credentials, tokens, raw media/log links, structured private
records, YouTube watch URLs, and private artifact names. Expected: zero
findings.

- [ ] **Step 5: Isolated deterministic render check**

Copy only the canonical project-status sources into a fresh temporary directory,
render all nine GFM and HTML pages there, and compare each result byte-for-byte
with the committed working outputs. Expected: eighteen matching file pairs.

Run:

```bash
git diff --check
git status --short --branch --untracked-files=all
git diff --name-only 814e554..HEAD -- data/
```

Expected: no whitespace errors, only intended report outputs before commit, and
no dataset path changes.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  docs/project-status/index.md \
  docs/project-status/experiment-ledger.md \
  docs/project-status/code-map.md \
  docs/project-status/research-integrity.md \
  docs/project-status/research-visual-evidence.md \
  docs/project-status/research-speech-transcripts.md \
  docs/project-status/research-native-audio.md \
  docs/project-status/research-models.md \
  docs/project-status/research-routing.md \
  reports/project-status/index.html \
  reports/project-status/experiment-ledger.html \
  reports/project-status/code-map.html \
  reports/project-status/research-integrity.html \
  reports/project-status/research-visual-evidence.html \
  reports/project-status/research-speech-transcripts.html \
  reports/project-status/research-native-audio.html \
  reports/project-status/research-models.html \
  reports/project-status/research-routing.html
git commit -m "docs: render research-theme status site"
```

---

### Task 5: Independent evidence audit, private portal refresh, and final gate

**Files:**
- No intended repository edits.
- Refresh only: `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/*.html`

**Interfaces:**
- Consumes: the committed nine-page site from Task 4.
- Produces: independently approved content and a byte-identical tailnet-only private site.

- [ ] **Step 1: Run an independent content and framing review**

Ask a fresh reviewer to compare the final site against:

```text
docs/superpowers/specs/2026-07-25-research-theme-status-site-redesign.md
docs/superpowers/plans/2026-07-25-research-theme-status-site-redesign.md
the pre-redesign source at 7a837db
the tracked reports cited by each theme
```

The reviewer must verify:

- all historical evidence remains, without unsupported new experimental claims;
- negative results are condition-bound and every one includes what it does not
  rule out plus a reconsideration trigger;
- confirmed defects, audit gaps, and uncertainty are correctly classified;
- future ideas are clearly untested and include valid controls/decision rules;
- the new structure is substantially less overwhelming at first view;
- no private artifact, application test, dataset change, or unrelated scope
  entered the branch.

Address only actionable findings, re-render affected outputs, and repeat the
scoped verification before proceeding.

- [ ] **Step 2: Refresh the authorized private portal**

Copy exactly these nine verified HTML pages from `reports/project-status/`:

```text
index.html
experiment-ledger.html
code-map.html
research-integrity.html
research-visual-evidence.html
research-speech-transcripts.html
research-native-audio.html
research-models.html
research-routing.html
```

Use this destination directory:

```text
/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/
```

Do not change the portal index outside this report directory or any sharing
configuration. Verify source and portal-copy SHA-256 for every page. Fetch each
live URL and require HTTPS 200 plus the same hash. Run `tailscale funnel status`
and require `(tailnet only)` with `/` mapped to `/home/npatta01/codex-rendered`.

- [ ] **Step 3: Run final clean-branch checks**

```bash
git diff --check 814e554..HEAD
git status --short --branch --untracked-files=all
git diff --name-only 814e554..HEAD -- data/
readlink -f data
```

Expected: clean `codex/repository-cleanup-report`, no dataset changes, and the
canonical shared-data resolution. Record that the application test suite was
intentionally not run because the user explicitly scoped this to report-only
work.

- [ ] **Step 4: Hand off for browser review**

Provide:

```text
https://npatta01-framework.tail481212.ts.net/plans/trecvid-vqa-current-state/experiment-ledger.html
```

Also link the canonical hub QMD, six theme QMD files, design specification, and
this implementation plan. Do not create GitHub issues until the user reviews
the revised site.
