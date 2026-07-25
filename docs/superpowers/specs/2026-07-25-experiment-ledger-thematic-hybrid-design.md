# Experiment Ledger Thematic-Hybrid Redesign

**Date:** 2026-07-25
**Audience:** Technical newcomers to the TRECVID 2026 VQA repository
**Primary source:** `docs/project-status/experiment-ledger.qmd`

## Objective

Make the Experiment Ledger explain **why each experiment was run** without
forcing a newcomer to reconstruct the research logic from a long date-ordered
chronicle. Preserve dates, panels, limitations, decisions, supersession, and
evidence links so the page remains auditable.

The redesigned page should answer two questions at different depths:

1. **Newcomer path:** What research questions did the project investigate, and
   what is the current answer to each one?
2. **Audit path:** When did each experiment happen, what panel did it use, and
   where is its evidence?

## Approved organization

Use a **thematic main body with a compact chronology at the end**. Chronology
remains available for lookup, but it no longer determines the reading order.

The main page is organized around four research questions:

### 1. Can we run trustworthy experiments?

**Why this mattered:** Before comparing models, the project needed private,
reproducible acquisition, label isolation, runnable adapters, and a clear line
between compatibility and quality.

Primary entries:

- July 12 field guide
- Data readiness
- Three-video compatibility gate

Current answer: the repository can prepare and execute controlled local work,
but operational readiness and compatibility do not establish answer quality.

### 2. What should every question start with?

**Why this mattered:** The project needed a simple global baseline before
adding routing, more frames, or specialist models.

Primary entries:

- Frozen-30 protocol v2
- Directed temporal sampling
- Dense-burst sampling

Current answer: Qwen3.5-4B with 12 uniform frames remains the global accuracy
baseline; MiniCPM-o 4.5 audio-off is the fast alternate. Global 24 frames and
the tested temporal heuristics did not earn default status.

### 3. When is specialist evidence worth adding?

**Why this mattered:** Some questions depend on speech, sound, spatial detail,
or evidence provenance, but adding every modality globally can distract the
model or waste compute.

Primary entries:

- Audio, spatial, and Seamless diagnostics
- Automatic-transcript ablation
- Taxonomy audit
- Human-oracle tier

Cross-cutting evidence: the failure-selected Qwen3.6/audio panel supplies the
strongest current caution against a global native-audio switch and links to its
full entry under model promotion.

Current answer: retain quality-gated Whisper for lexical speech; study native
audio only as a category-routed specialist; keep the independent human-oracle
tier blocked until its provenance records exist.

### 4. Should another multimodal model replace the baseline?

**Why this mattered:** Larger or native-audio-capable models could potentially
recover baseline failures, but runtime success is not evidence of semantic
quality.

Primary entries:

- Failure-selected Qwen3.6 and audio panel
- Phi and Gemma compatibility, then semantic stop
- Qwen3-Omni compatibility and quality-10

Current answer: none is promoted. Qwen3.6, Phi, and Gemma failed their relevant
quality gates. Qwen3-Omni remains promising but formally ineligible after only
19 of 20 strict outputs completed.

Every existing ledger entry has exactly one primary thematic home. A short
cross-reference may expose a cross-cutting result, but the full entry is not
duplicated.

## Page structure

Use this reading order:

1. **Technical summary:** the four current answers and the evidence boundary.
2. **Research-question map:** replace the date-first timeline visual with a
   four-theme map leading to the current baseline and specialist routes.
3. **Theme-at-a-glance table:** theme, motivating uncertainty, current answer,
   and strongest evidence level.
4. **Four thematic sections:** each starts with the motivating question and
   current answer, followed by its experiment summaries.
5. **What superseded what:** retain the existing compact correction table.
6. **Chronology index:** five milestone rows—July 12, July 20, July 21,
   July 24, and July 25—with links back to thematic entries.

The research-question map remains one of the mini-site's five accessible
visuals. It receives a meaningful caption, alt text, Mermaid accessible title
and description, an explanatory paragraph, and the theme-at-a-glance table as
its text equivalent.

## Experiment summary pattern

Replace the repeated `Tried / Panel / Learned / Decision / Limitation / Evidence`
shape with a more direct sequence:

- **Why tested:** the uncertainty or preceding result that motivated the run.
- **Test and panel:** what changed, comparison basis, and exact panel size.
- **Result → decision:** the observed result and what it changed.
- **Boundary:** what the panel cannot establish.
- **Evidence:** existing report, issue, PR, or sanitized digest links.

Keep each field to one compact paragraph or bullet. Preserve every currently
reported denominator, exact aggregate, generalization limit, and eligibility
qualification. Do not add new experimental claims.

To reduce initial page length, put each theme's detailed experiment summaries
inside one native, clearly labeled disclosure block after its always-visible
answer and summary table. Use that structure only if one QMD construct renders
as an accessible disclosure in both HTML and GitHub-readable Markdown. If the
render check fails, keep the entries expanded rather than maintaining two
different editorial versions.

## Evidence and claim boundaries

- Frozen-30 remains the strongest matched global comparison.
- Selected diagnostics remain explicitly non-representative.
- Compatibility, diagnostic quality, formal eligibility, and promotion remain
  separate statuses.
- The dense-burst result remains aggregate-only and contains no row identifiers,
  answers, predictions, paths, or raw artifacts.
- No private frames, transcripts, predictions, datasets, or logs enter any
  output.
- The quality of the unlabeled 2026 evaluation set remains unknown.

## Files and generated outputs

Edit only the QMD source and directly dependent report metadata or styling:

- `docs/project-status/experiment-ledger.qmd`
- `docs/project-status/experiment-ledger.md` (generated)
- `reports/project-status/experiment-ledger.html` (generated)
- the canonical cleanup plan, only if its description of the ledger structure
  becomes stale

Do not reorganize historical evidence files, add a structured ledger schema,
create a new test framework, or change the other report pages except for a
directly dependent navigation or description correction.

Refresh only the sanitized HTML page in the existing tailnet-only portal after
verification.

## Verification

- Render the Experiment Ledger to GitHub-readable Markdown and self-contained
  HTML from the QMD source.
- Confirm every original experiment, panel size, decision, limitation, and
  evidence link remains represented once.
- Confirm every entry begins with an evidence-backed reason for testing.
- Check the thematic map's alt text, caption, accessible SVG title/description,
  explanatory paragraph, and table equivalent.
- Check navigation, disclosure controls if used, readability, and overflow at
  desktop and 390px phone widths.
- Re-run repository-relative link, generated-output privacy, and
  `git diff --check` checks.
- Do not run or add application tests for this report-only revision.

## Success criteria

A first-time reader can answer these without scanning thirteen chronological
entries:

1. What question did a group of experiments investigate?
2. Why was that question worth testing at that point?
3. What is the current answer and confidence boundary?
4. Which experiment and source support that answer?
5. Where does the work appear in time if an audit requires chronology?
