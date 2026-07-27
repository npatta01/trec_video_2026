# Experiment Ledger Thematic-Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the Experiment Ledger around four motivating research questions while retaining a compact five-row chronological audit path.

**Architecture:** Keep `experiment-ledger.qmd` as the sole editable report source. Replace the date-first timeline and 13-entry chronicle with an answer-first research-question map, four thematic sections, compact evidence entries, the existing supersession table, and a final chronology index; then regenerate the sibling GFM and self-contained HTML outputs.

**Tech Stack:** Quarto 1.9.38, QMD/GFM, Mermaid, self-contained HTML, headless Chrome, Git.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-25-experiment-ledger-thematic-hybrid-design.md` exactly.
- Preserve all 13 experiments, their panel sizes, aggregate results, decisions, limitations, eligibility status, and evidence links.
- Keep frozen-30 as the strongest matched global comparison and keep 2026 answer quality unknown.
- Do not expose private frames, transcripts, predictions, datasets, logs, row identifiers, answers, or filesystem paths.
- Keep the QMD as the only editable ledger source; Markdown and HTML are generated outputs.
- Do not add a ledger schema, custom test suite, CI, application tests, or new framework.
- Do not modify or stage `data/`; its symlink must continue to resolve to `/home/npatta01/data/competitions/trec_video_2026/data`.
- Refresh only the sanitized Experiment Ledger HTML page in the existing tailnet-only portal.

---

### Task 1: Rewrite the source as a thematic decision trail

**Files:**
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify only if structurally stale: `docs/superpowers/plans/2026-07-25-repository-cleanup-current-state-report.md`

**Interfaces:**
- Consumes: the approved design spec and the 13 existing ledger entries in `experiment-ledger.qmd`.
- Produces: stable theme/entry anchors and one QMD source ready for both GFM and HTML rendering.

- [ ] **Step 1: Replace the page framing and date-first visual**

Change the YAML description to say the ledger is organized by research question with a compact chronological index.

Replace `## Timeline` with this visible structure:

````markdown
## Research questions

Read this page by research question to understand why the project ran each
experiment and what decision followed. Use the chronology index at the end
only when you need the date-ordered audit trail.

```{mermaid}
%%| label: fig-experiment-timeline
%%| fig-cap: "Research-question map showing how foundation, baseline, specialist-evidence, and model-promotion experiments led to the current system."
%%| fig-alt: "Flowchart with four research questions. Trustworthy-experiment work established reproducible execution but not quality. Baseline work selected Qwen3.5-4B with 12 uniform frames and MiniCPM-o audio-off as a fast alternate. Specialist-evidence work retained routed Whisper and kept native audio experimental. Model-promotion work stopped Qwen3.6, Phi, and Gemma while leaving Qwen3-Omni promising but ineligible."
flowchart TD
  accTitle: Research-question map
  accDescr: Four research questions led to the current baseline, specialist routes, stopped alternatives, and open candidates.
  P[Project decisions] --> T{Can experiments be trusted?}
  T --> TR[Reproducible execution<br/>quality still separate]
  P --> B{What should every question start with?}
  B --> BR[Qwen3.5-4B + 12 frames<br/>MiniCPM audio-off for speed]
  P --> S{When is specialist evidence useful?}
  S --> SR[Routed Whisper<br/>native audio remains experimental]
  P --> M{Should another model replace the baseline?}
  M --> MR[No promotion yet<br/>Qwen3-Omni remains ineligible]
```

**Figure caption:** Research-question map showing how foundation, baseline,
specialist-evidence, and model-promotion experiments led to the current system.

The map is a decision index, not an evidence ranking. The table below gives the
same four paths in text and states the strongest evidence each path can support.

| Research question | Why it was tested | Current answer | Evidence strength |
|---|---|---|---|
| Can we run trustworthy experiments? | Model comparisons require private, reproducible, label-isolated execution. | The repository can run controlled work; readiness and compatibility do not establish answer quality. | Operational checks and a three-row compatibility panel. |
| What should every question start with? | The project needed a simple default before adding specialists. | Qwen3.5-4B with 12 frames is the accuracy baseline; MiniCPM-o audio-off is the fast alternate. | Matched frozen-30 plus narrower temporal diagnostics. |
| When is specialist evidence worth adding? | Speech and sound may help selected questions but can distract globally. | Route quality-gated Whisper to lexical speech; keep native audio experimental and the human-oracle tier blocked. | Selected diagnostics, including a ten-row transcript ablation. |
| Should another multimodal model replace the baseline? | Larger and native-audio models might recover baseline failures. | None is promoted; Qwen3-Omni is promising but formally ineligible. | Failure-selected and three-/ten-row quality screens. |
````

The Mermaid block must retain the label `fig-experiment-timeline` so cross-format figure identity remains stable, but rename its accessible/caption text to “Research-question map.” Add `fig-alt`, `fig-cap`, `accTitle`, and `accDescr` that describe all four branches.

- [ ] **Step 2: Add the four thematic sections and map each entry once**

Use these exact theme anchors and primary mappings:

```text
#trustworthy-experiments
  field-guide, data-readiness, compatibility-gate
#global-default
  frozen-30-protocol-v2, directed-temporal-sampling, dense-burst-sampling
#specialist-evidence
  audio-spatial-seamless, automatic-transcript-ablation, taxonomy-audit,
  human-oracle-tier
#model-promotion
  failure-panel, phi-gemma-semantic-stop, qwen3-omni-quality
```

Start every theme with its exact heading, motivating uncertainty, and current
answer from this table:

| Heading | Why this mattered | Current answer |
|---|---|---|
| `## Can we run trustworthy experiments? {#trustworthy-experiments}` | Before comparing models, the project needed private, reproducible acquisition, label isolation, runnable adapters, and a clear line between compatibility and quality. | The repository can prepare and execute controlled local work, but operational readiness and compatibility do not establish answer quality. |
| `## What should every question start with? {#global-default}` | The project needed a simple global baseline before adding routing, more frames, or specialist models. | Qwen3.5-4B with 12 uniform frames remains the global accuracy baseline; MiniCPM-o 4.5 audio-off is the fast alternate. Global 24 frames and the tested temporal heuristics did not earn default status. |
| `## When is specialist evidence worth adding? {#specialist-evidence}` | Some questions depend on speech, sound, spatial detail, or evidence provenance, but adding every modality globally can distract the model or waste compute. | Retain quality-gated Whisper for lexical speech; study native audio only as a category-routed specialist; keep the independent human-oracle tier blocked until its provenance records exist. |
| `## Should another multimodal model replace the baseline? {#model-promotion}` | Larger or native-audio-capable models could potentially recover baseline failures, but runtime success is not evidence of semantic quality. | None is promoted. Qwen3.6, Phi, and Gemma failed their relevant quality gates; Qwen3-Omni remains promising but formally ineligible after only 19 of 20 strict outputs completed. |

Render each table row as two short paragraphs labeled `**Why this mattered.**`
and `**Current answer.**` directly after the corresponding heading.

Keep the failure-panel entry only under `#model-promotion`; add one short cross-reference from the specialist-evidence introduction to its native-audio result instead of duplicating the entry.

- [ ] **Step 3: Rewrite every experiment using the compact motivation-first pattern**

Use explicit stable entry anchors and exactly this field order. Frozen-30 should
take this concrete form, preserving the existing evidence links:

```markdown
### Frozen-30 protocol v2 {#frozen-30-protocol-v2}

- **Why tested:** The three-video gate established runnable adapters but was too
  small to choose a default; the project needed a matched comparison of model,
  frame-budget, and transcript conditions.
- **Test and panel:** Four matched VQA arms across 30 labeled 2025 training
  examples balanced across five declared strata; 120 valid predictions and 30
  valid Whisper executions.
- **Result → decision:** Qwen12+Whisper led exact match at 11/30 and token F1 at
  0.506795; MiniCPM audio-off led METEOR at 0.374089 and outer-wall speed;
  Qwen24 added no exact answers over Qwen12. Keep Qwen12 as the accuracy
  baseline, MiniCPM audio-off as the fast alternate, and Whisper as a separately
  reported route.
- **Boundary:** This is the strongest verified repository comparison, but it is
  descriptive model selection on 2025 training data—not a 2026 estimate.
- **Evidence:** [protocol-v2 report][frozen30] and [recovered benchmark work in
  PR #21][pr21].
```

Use these evidence-backed motivations for the remaining entries; preserve their
existing test/panel, result/decision, boundary, and evidence content:

| Entry anchor | Why tested |
|---|---|
| `field-guide` | The repository began with candidate architectures and prior work but lacked local executed evidence to choose among them. |
| `data-readiness` | Controlled inference required resumable acquisition, source validation, shared-data linking, and label isolation without committing private artifacts. |
| `compatibility-gate` | Before paying for larger panels, the project needed to learn which model-family adapters could execute reproducibly on the local AMD/ROCm host. |
| `directed-temporal-sampling` | Global frame increases were costly and inconclusive, so the project tested whether reallocating a fixed 12-frame budget toward question-relevant time spans could help. |
| `dense-burst-sampling` | A narrower temporal hypothesis asked whether a dense local burst could beat a broader uniform-24 view on rows requiring short-lived evidence. |
| `audio-spatial-seamless` | Early baseline failures suggested missing speech or spatial detail, so the project tested the simplest transcript, crop, and translation escalations before designing a router. |
| `automatic-transcript-ablation` | A transcript had changed an answer in an early diagnostic, but the project needed a matched, label-blind test to learn whether accepted ASR reliably helps lexical-speech rows. |
| `taxonomy-audit` | Specialist routing needed evidence about which question types exist before the project could justify route categories or stratified follow-ups. |
| `human-oracle-tier` | Automatic ASR gains do not establish the value of an independently produced transcript, so the project specified a provenance-clean upper-bound tier. |
| `failure-panel` | Known baseline failures created a diagnostic opportunity to test whether a larger Qwen model or native audio could recover cases that the global baseline missed. |
| `phi-gemma-semantic-stop` | Joint audio/vision runtime compatibility left open the separate question of whether Phi and Gemma produced semantically useful answers. |
| `qwen3-omni-quality` | Qwen3-Omni's repaired joint runtime made it a candidate, but promotion required a neutral matched quality panel and a complete strict-output contract. |

Do not use collapsed `<details>` blocks: the Start Here page deep-links to the frozen-30 and automatic-transcript anchors, and hiding those targets would weaken navigation and accessibility. Compact wording plus thematic summaries is the verified fallback allowed by the design.

Preserve these existing externally referenced anchors exactly:

```text
#frozen-30-protocol-v2
#automatic-transcript-ablation
#what-superseded-what
```

- [ ] **Step 4: Retain supersession and add the chronology index**

Keep `## What superseded what` and all six rows, changing wording only if the new thematic structure makes a narrow reference stale.

Append:

```markdown
## Chronology index

| Date | Milestone | Research questions |
|---|---|---|
| July 12 | Field guide | Trustworthy experiments |
| July 20 | Data readiness and compatibility gate | Trustworthy experiments |
| July 21 | Audio/spatial diagnostics and frozen-30 | Global default; specialist evidence |
| July 24 | Routing, temporal, and model-promotion diagnostics | Global default; specialist evidence; model promotion |
| July 25 | Strict Qwen3-Omni quality result | Model promotion |
```

Link each research-question cell to the relevant theme anchor. The table is a lookup index, not a second detailed ledger.

- [ ] **Step 5: Verify source coverage and commit**

Run:

```bash
rg -n '^## |^### |\*\*Why tested:\*\*|\*\*Test and panel:\*\*|\*\*Result → decision:\*\*|\*\*Boundary:\*\*|\*\*Evidence:\*\*' docs/project-status/experiment-ledger.qmd
rg -n '30 labeled|120 valid|4/10 to 7/10|12 deliberately|481 training|ten temporal|two rows|1C/1P/1W|19/20|none—the strict tier did not run' docs/project-status/experiment-ledger.qmd
git diff --check
```

Expected: 4 theme headings, 13 experiment headings, 13 instances of every compact field, all named evidence boundaries present, and no whitespace errors.

Review the source diff against the design, then commit only the source and any directly necessary canonical-plan correction:

```bash
git add docs/project-status/experiment-ledger.qmd docs/superpowers/plans/2026-07-25-repository-cleanup-current-state-report.md
git commit -m "docs: organize experiment ledger by research question"
```

If the canonical plan did not change, omit it from `git add`.

---

### Task 2: Render and verify both delivery formats

**Files:**
- Modify generated: `docs/project-status/experiment-ledger.md`
- Modify generated: `reports/project-status/experiment-ledger.html`

**Interfaces:**
- Consumes: the thematic QMD and stable anchors from Task 1.
- Produces: GitHub-readable Markdown and self-contained HTML with equivalent content/navigation.

- [ ] **Step 1: Render GFM beside the QMD**

Run from the repository root:

```bash
env XDG_CACHE_HOME=/tmp/trecvid-ledger-quarto-cache quarto render docs/project-status/experiment-ledger.qmd --to gfm --output-dir docs/project-status --output experiment-ledger.md
```

Remove only Quarto transients created by this command:

```bash
rm -f docs/project-status/.gitignore docs/project-status/index.html
rm -rf docs/project-status/.quarto docs/project-status/docs
```

- [ ] **Step 2: Render the self-contained HTML site**

Run:

```bash
env XDG_CACHE_HOME=/tmp/trecvid-ledger-quarto-cache quarto render docs/project-status --to html
```

The optional Google-font fetch may warn in the network-restricted sandbox; the render succeeds only if generated HTML contains no Google-font URL and the browser loads zero external resources.

Remove only Quarto transients:

```bash
rm -f docs/project-status/.gitignore reports/project-status/search.json
rm -rf docs/project-status/.quarto reports/project-status/site_libs
```

- [ ] **Step 3: Check generated scope, links, privacy, and accessibility**

Run:

```bash
git status --short --untracked-files=all
python3 /tmp/trecvid-project-status-links.py
python3 /tmp/trecvid-project-status-private-scan.py
node /tmp/trecvid-project-status-probe.mjs
git diff --check
```

Expected:

- only the ledger Markdown/HTML outputs differ from Task 1's clean commit;
- 90 or more repository-relative links have zero failures;
- the nine report files have zero private-data findings;
- all three pages load at 1440px and 390px with one H1, no page overflow, no external runtime resources, and no forced new windows;
- the five mini-site figures still expose captions, SVG titles, and SVG descriptions;
- the ledger figure is a research-question map with an adjacent explanatory paragraph and four-row table equivalent.

Open the ledger at desktop and 390px and inspect the theme-at-a-glance table, section hierarchy, chronology index, and anchor navigation. Do not create a new committed browser-test suite.

- [ ] **Step 4: Commit generated outputs**

Review the generated diff, verify no `data/` path is staged, then run:

```bash
git add docs/project-status/experiment-ledger.md reports/project-status/experiment-ledger.html
git diff --cached --check
git commit -m "docs: render thematic experiment ledger"
```

---

### Task 3: Independent review, private refresh, and final branch gate

**Files:**
- No intended repository edits.
- Refresh external sanitized copy: `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/experiment-ledger.html`

**Interfaces:**
- Consumes: committed QMD, GFM, and HTML from Tasks 1–2.
- Produces: reviewer-approved branch and byte-identical tailnet-only Experiment Ledger page.

- [ ] **Step 1: Run an independent content review**

Ask a fresh reviewer to compare the final branch against the design and verify:

- all 13 entries have one thematic home and an evidence-backed motivation;
- no aggregate, panel size, limitation, decision, evidence link, or eligibility status was lost;
- the thematic summaries do not overstate diagnostic panels;
- chronology and supersession remain auditable;
- no unrelated files, private artifacts, or application tests entered scope.

Address only actionable findings and repeat the narrow render/verification step if the QMD changes.

- [ ] **Step 2: Refresh the authorized private portal page**

Copy only the verified ledger HTML:

```bash
cp reports/project-status/experiment-ledger.html /home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/experiment-ledger.html
```

Verify the local and portal-copy SHA-256 hashes match, the live URL returns HTTP 200, and `tailscale funnel status` still reports `(tailnet only)`.

- [ ] **Step 3: Run the final clean-branch checks**

Run:

```bash
git diff --check 814e554..HEAD
git status --short --branch --untracked-files=all
git diff --name-only 814e554..HEAD -- data/
```

Expected: no whitespace errors, a clean `codex/repository-cleanup-report` worktree, and no dataset changes. Record that the application test suite was intentionally not run because this is a report-only revision.
