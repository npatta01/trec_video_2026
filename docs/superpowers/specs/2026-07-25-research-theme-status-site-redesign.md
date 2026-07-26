# Research-Theme Status Site Redesign

**Date:** 2026-07-25
**Audience:** Technical newcomers and project researchers
**Canonical sources:** `docs/project-status/*.qmd`

## Objective

Recenter the current-state report around the project's broader research themes,
not around a chronology or a short sequence of past decision questions. A new
reader should be able to see, in one place, what was tried, what the evidence
actually supports, what may have gone wrong, what remains possible, and what a
valid next experiment would look like.

The redesign must correct two misleading impressions in the current report:

1. A condition that failed a small promotion gate is not permanently ruled out
   under every model, prompt, panel, budget, or implementation.
2. A recorded result is not automatically trustworthy merely because a run
   completed; confirmed defects, audit gaps, small-panel uncertainty, and
   untested hypotheses must remain distinct.

## Reporting job

- **Primary question:** What research directions has the project explored, and
  what should be tested next?
- **Decision supported:** Choose the next small, controlled experiments without
  repeating superseded work or overgeneralizing narrow diagnostics.
- **Audience:** Technical readers who did not participate in the prior runs.
- **Evidence window:** Repository evidence through 2026-07-25.
- **Strongest comparison:** Frozen-30 protocol v2; smaller selected panels remain
  diagnostic.
- **Unknown:** Quality on the unlabeled 2026 evaluation set.
- **Delivery:** QMD source, GitHub-readable Markdown, and self-contained HTML on
  the existing private tailnet-only portal.

## Information architecture

Keep three top-level navigation destinations so the site still feels small:

1. **Start Here** — current operational recommendation.
2. **Research Themes** — the existing `experiment-ledger` URL becomes the hub.
3. **Code Map and Gaps** — implementation ownership and engineering gaps.

The Research Themes hub links to six focused pages:

1. **Experiment integrity and evaluation**
2. **Visual and temporal evidence**
3. **Speech and transcript evidence**
4. **Native audio and sound**
5. **Models and adaptation**
6. **Routing, fusion, and answer selection**

This deliberately supersedes the earlier three-rendered-page constraint. The
site retains three top-level navigation choices but adds six reader-invoked
detail pages. The existing historical evidence files stay at their current
paths.

## Research Themes hub

`docs/project-status/experiment-ledger.qmd` remains the canonical hub and keeps
its existing URL. Its visible title changes from a date-like ledger emphasis to
**Research Themes and Experiment Ledger**.

The hub uses this reading order:

1. **Technical summary** — current baseline, strongest evidence, and the rule
   that negative results are condition-bound.
2. **Six-theme research map** — independent linked cards, not arrows or a
   linear path.
3. **What to investigate next** — one concise priority from each theme.
4. **How to interpret negative results** — the status vocabulary and the
   difference between a confirmed defect, an audit gap, and uncertainty.
5. **Compatibility index** — all historical entry anchors point to their new
   thematic homes.
6. **What superseded what** — compact and collapsed by default.
7. **Chronology index** — the existing five milestones, collapsed by default.

The theme map is a semantic two-column card grid on wide screens and a
single-column list below 768 pixels. Each card is a real keyboard-focusable
link and contains the theme, current working answer, evidence strength, and
most important open question. GitHub Markdown receives the same information in
normal document flow. There are no arrows, so the map does not imply that the
themes are sequential dependencies.

## Theme-page contract

Every theme page follows the same visible structure:

1. **Why this theme matters**
2. **Current working answer**
3. **What has been tried**
4. **What the evidence supports**
5. **What it does not rule out**
6. **Possible bugs, confounds, and audit gaps**
7. **Open hypotheses and ideas**
8. **Cheapest valid next experiments**
9. **Related themes**

Each completed experiment receives a short, always-visible orientation:

- hypothesis or reason for testing;
- tested condition and panel size;
- observed result;
- current condition-bound status.

Extended method, exact results, limitations, provenance, and evidence links go
inside a native disclosure labeled **Method, result, validity risks, and
evidence**. Keep the experiment heading and stable anchor outside the closed
disclosure so deep links land on visible content. If native disclosure does not
render accessibly in both Quarto HTML and GitHub Markdown, leave the details
expanded rather than maintaining separate editorial versions.

Every negative or incomplete result must include:

- **What this rules out now**
- **What it does not rule out**
- **Reconsider when**

Every proposed next experiment must include:

- hypothesis;
- smallest valid panel or work item;
- matched controls and integrity safeguards;
- decision or promotion rule;
- boundary on what a positive result would establish.

## Theme scope

### 1. Experiment integrity and evaluation

Primary evidence includes the July 12 field guide, data readiness, the
three-video compatibility gate, frozen-30 v1's verified label-isolation defect,
protocol v2's repaired two-phase boundary, and the scorer-reuse audit.

Open work includes a reusable `prepare → run → freeze → score → report`
integrity gate, deliberately invalid fixtures, source/model/runtime attestation,
and representative-panel design. This page must say that integrity proves a run
is interpretable, not that its answers are good.

### 2. Visual and temporal evidence

Primary evidence includes the 12-versus-24-frame frozen-30 comparison,
directed-12, dense-burst-12, and spatial-crop diagnostics.

Directed-12 is defined in plain language: retain broad context and spend the
remaining fixed 12-frame budget near a keyword/anchor-derived region. Its
ten-row result rejects that tested heuristic as the current default, not all
question-aware sampling. The page records the tracked audit gap that the
selector has source/tests but no tracked general-workflow caller; this is not a
confirmed execution bug.

Dense-burst-12 remains a two-row aggregate. The tracked repository lacks its
exact implementation and schedule, so the page labels it a weakly auditable
diagnostic rather than a general result.

Open work includes required-moment labeling, selector recall, oracle-frame
answer quality, fixed-budget uniform/query/hybrid selectors, shot or event
boundaries, coarse-to-fine retrieval, spatial detail, and OCR routing.

### 3. Speech and transcript evidence

Primary evidence includes frozen-30 Whisper results, the ten-row automatic
transcript ablation, multilingual/translation follow-ups, Seamless diagnostics,
the blocked strict human-transcript tier, and external dataset-reference
transcript research.

The page distinguishes:

- automatic ASR output;
- uploader captions;
- dataset recording-prompt references;
- independently produced human transcripts.

None may be relabeled as another provenance tier. A dataset-reference control
can validate plumbing but cannot establish a TRECVID human-oracle upper bound.

Open work includes transcript-grounded answers, alternative or multilingual
ASR, timestamps and evidence spans, and a true label-blind human-transcript
upper bound with independent transcriber/reviewer provenance.

### 4. Native audio and sound

Primary evidence includes matched MiniCPM audio-off/audio-on diagnostics,
failure-selected prosody and non-speech findings, sound/music failures, and
native-audio evidence from Omni candidates.

The current evidence rejects a global native-audio switch under the tested
conditions. It does not reject category-routed native audio. Open work includes
matched speech/prosody/non-speech/control strata, audio event classifiers with
abstention, music fingerprinting or retrieval, and transcript-plus-native-audio
combinations.

### 5. Models and adaptation

Primary evidence includes the compatibility gate, Qwen3.6 failure panel,
Phi/Gemma semantic screen, MiniCPM's speed role, and Qwen3-Omni compatibility
and quality runs.

The Qwen3-Omni strict run was ineligible under its predeclared 20/20 completion
gate because only 19 strict outputs passed. That makes the run ineligible for
that promotion decision; it does not make the model permanently ineligible for
a repaired, complete, blinded comparison.

Open work includes repaired complete candidates, matched representative panels,
specialist models, model revisions, and adaptation or fine-tuning only after a
specific evidence bottleneck is established.

### 6. Routing, fusion, and answer selection

Primary evidence includes the taxonomy audit, specialist-routing proposal,
transcript residual-error attribution, and failure panels where the needed
evidence may have been present but the model chose the wrong answer.

The page distinguishes retrieval failure (the needed evidence was not selected)
from fusion/reasoning failure (the evidence was present but not used correctly).
Open work includes a versioned label-blind router, abstention and baseline
fallback, route-level non-regression, transcript/evidence grounding, answer
election, and targeted ensembles.

## Status and uncertainty vocabulary

Use these statuses consistently:

- **Current default** — strongest available evidence for present use.
- **Fast alternate** — an explicit cost/latency trade-off.
- **Routed candidate** — considered only for a declared question type.
- **Diagnostic signal** — useful for choosing the next test, not for prevalence
  or general quality claims.
- **Not promoted under the tested condition** — failed this implementation,
  panel, budget, or gate; not a permanent verdict.
- **Blocked** — cannot run or interpret the experiment until a named prerequisite
  exists.
- **Superseded** — a later valid result replaces the earlier conclusion.
- **Open hypothesis** — proposed but not yet tested.

Reserve **formally ineligible** for a specific predeclared gate and name that
gate in the same sentence. Avoid the standalone words `stop`, `failed model`,
or `ineligible model` when the evidence only supports a narrower decision.

Classify validity concerns as:

1. **Confirmed defect** — demonstrated protocol or implementation fault, such
   as frozen-30 v1 label isolation.
2. **Audit gap** — tracked evidence is insufficient to confirm the launch path
   or exact implementation, such as directed-run attachment or dense-burst
   provenance.
3. **Uncertainty** — small/selected panel, model revision, prompt, annotation,
   or generalization limit without evidence of a bug.

Do not imply that an audit gap is a confirmed bug.

## Interaction design

### Disclosures

Use native `<details>/<summary>` controls with visible focus styling and no
custom JavaScript. Summaries must state what opens; do not use vague labels such
as `More`. Default closed state is allowed only for evidence already summarized
outside the control.

### Expandable decision diagram

On Start Here, keep the readable decision table/cards as the primary interface.
Move the dense Mermaid decision diagram into a native disclosure labeled
**Expand the full decision diagram**. When opened, render it at a legible
minimum width inside a keyboard-focusable horizontal-scroll region. The text
equivalent remains outside the disclosure, so the diagram is never the sole
source of a decision.

### Deep-link compatibility

The hub retains every existing ledger fragment target in its compatibility
index, including `directed-temporal-sampling`, `dense-burst-sampling`,
`automatic-transcript-ablation`, `failure-panel`, and the theme-level anchors.
Each target visibly links to the experiment's new page and anchor. No hash-based
JavaScript redirect is added.

The chronology and supersession anchors remain stable on the hub even when
their contents are collapsed.

## Source and generated files

Canonical editable sources:

- `docs/project-status/index.qmd`
- `docs/project-status/experiment-ledger.qmd`
- `docs/project-status/research-integrity.qmd`
- `docs/project-status/research-visual-evidence.qmd`
- `docs/project-status/research-speech-transcripts.qmd`
- `docs/project-status/research-native-audio.qmd`
- `docs/project-status/research-models.qmd`
- `docs/project-status/research-routing.qmd`
- `docs/project-status/code-map.qmd` only for directly dependent navigation or
  gap-link changes
- `docs/project-status/_quarto.yml`
- `docs/project-status/styles.css`

Generate GitHub-readable Markdown beside every QMD source. Generate
self-contained HTML under `reports/project-status/`. QMD remains the sole
editable source for report prose.

Do not move historical reports/plans, add a structured ledger schema, create a
new application test framework, expose private artifacts, or create GitHub
issues in this revision.

## Verification

- Render every QMD page to GFM and the full site to self-contained HTML.
- Confirm the six theme pages and three top-level pages have working navigation
  and exactly one H1.
- Verify the theme grid is two columns on desktop and one column at 390 pixels.
- Verify disclosure controls are keyboard operable, accurately labeled, and
  contain no unique conclusion hidden from the closed state.
- Verify the expanded decision diagram is legible and scrollable without page
  overflow at desktop and 390 pixels.
- Confirm all original ledger entries, aggregates, limitations, and evidence
  links remain represented, allowing the former mixed diagnostic entry to be
  split across themes without duplicating quantitative claims.
- Confirm every negative result has `What this does not rule out` and
  `Reconsider when` content.
- Confirm validity risks are labeled as confirmed defect, audit gap, or
  uncertainty.
- Confirm all legacy fragments resolve on the hub and link to the correct new
  page anchor.
- Run repository-relative link, generated-output privacy, external-resource,
  phone/desktop overflow, and `git diff --check` checks.
- Re-render in an isolated temporary directory and confirm committed generated
  outputs are byte-identical.
- Refresh only the sanitized project-status HTML pages through the existing
  tailnet-only portal; verify hashes, HTTPS 200, and tailnet-only status.
- Do not run or add the application test suite for this documentation-only
  revision.

## Success criteria

A first-time reader can answer these without reconstructing the chronology:

1. What broader research problem does this experiment belong to?
2. What exactly was tested, on how many examples, and under which conditions?
3. What does the result support, and what does it not rule out?
4. Is a concern a confirmed defect, an audit gap, or ordinary uncertainty?
5. What is the cheapest valid next test, with which controls and decision rule?
6. Where can the reader expand the exact method and evidence?

The rendered site should feel like a small research program map with optional
audit depth, not a long chronological notebook or a permanent ranking of ideas.
