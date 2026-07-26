# Living Research Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current-state Quarto site understandable from first
principles while adding a public-ready worked example, living failure library,
clear experiment rationale, and visible next research directions.

**Architecture:** Preserve the three-item top navigation and make Research
Themes the non-linear research-memory hub. Add `worked-example.qmd` and
`failure-library.qmd`, layer existing theme pages from newcomer explanation to
collapsible expert evidence, and keep QMD as the sole editable report source.
Generated Markdown and self-contained HTML remain derived outputs.

**Tech Stack:** Quarto 1.9.x, Pandoc GFM, native HTML
`<details>/<summary>`, Mermaid, existing repository CSS, optimized cleared
image assets when authorized, headless Chrome for responsive/accessibility
inspection, and the existing tailnet-only Tailscale Serve portal.

## Global Constraints

- Work only on `codex/repository-cleanup-report`; leave detached worktrees and
  their loose files untouched.
- QMD files are the sole editable source for report prose. Markdown and HTML
  are generated outputs.
- Keep exactly three top-level navigation choices: Start Here, Research Themes,
  and Code Map and Gaps.
- Preserve historical reports, plans, issue/PR links, and stable legacy
  fragments at their existing paths.
- Add no structured ledger service, case schema framework, custom JavaScript,
  application test framework, CI, dependency, or schema validator.
- Do not run the application test suite; this is a documentation-only revision.
- Keep important conclusions, statuses, definitions, and next tests visible
  outside closed disclosures.
- Treat negative evidence as condition-bound. Separate Confirmed defect, Audit
  gap, and Uncertainty.
- Never stage or commit `data/` content. Before reading a local case artifact,
  confirm `data` resolves to
  `/home/npatta01/data/competitions/trec_video_2026/data`.
- No 2026 evaluation row, reference answer, frame, audio, transcript,
  prediction, or diagnosis may enter source or generated output.
- Pixel, audio, waveform, spectrogram, and transcript assets fail closed unless
  their case contains a public-safe rights attestation with asset ID, rights
  class, source/license, allowed use, verifier, date, and attribution.
- A 2025 label-informed diagnosis is explicitly post-hoc and is never presented
  as pre-freeze model or router evidence.
- Until explicit public release, publish only sanitized derived HTML and
  required cleared assets through the existing tailnet-only portal. Do not use
  Funnel, a public listener, Codex Sites, or new sharing permissions.
- GitHub issue creation remains deferred until the revised report is reviewed.

---

## File structure

### New canonical QMD sources

- `docs/project-status/worked-example.qmd` — one traced question from evidence
  selection through frozen prediction, grading, diagnosis, and next test.
- `docs/project-status/failure-library.qmd` — policy plus three to five
  approachable, auditable case records.

### Modified canonical QMD sources

- `docs/project-status/index.qmd` — first-principles opening, lifecycle banner,
  compact policy, worked-example link, and newcomer definitions.
- `docs/project-status/experiment-ledger.qmd` — Research Themes hub, compact
  tried/learned/open/next matrix, failure preview, and visible branch-complete
  next-test queue.
- `docs/project-status/research-integrity.qmd` — newcomer primer, decision strip,
  evidence-at-a-glance, cases, and expert-depth ordering.
- `docs/project-status/research-visual-evidence.qmd` — frame-budget and temporal
  sampling primer with concrete uniform/directed/burst explanations.
- `docs/project-status/research-speech-transcripts.qmd` — ASR/transcript primer,
  provenance examples, and failure-library links.
- `docs/project-status/research-native-audio.qmd` — speech/prosody/non-speech
  primer, routing examples, and failure-library links.
- `docs/project-status/research-models.qmd` — compatibility-versus-quality
  primer, condition-bound model conclusions, and failure-library links.
- `docs/project-status/research-routing.qmd` — retrieval-versus-fusion primer,
  concrete diagnostic flow, and failure-library links.
- `docs/project-status/code-map.qmd` — traced flow-to-code crosswalk, link
  classes, contribution path, and remaining documentation/engineering gaps.

### Modified site shell

- `docs/project-status/_quarto.yml` — render the two new pages, refresh lifecycle
  copy, and preserve the three-item navbar.
- `docs/project-status/styles.css` — lifecycle banner, decision strip, glossary,
  breadcrumb/local navigation, failure cards, contact-sheet/timeline layout,
  link-kind labels, and responsive disclosure behavior.

### Optional cleared assets

- `docs/project-status/assets/cases/` — only optimized stills or contact sheets
  whose public-safe rights attestations appear in the corresponding QMD case.
  Leave this directory absent if no candidate clears the rights gate.

### Generated outputs

- Create `docs/project-status/worked-example.md` and
  `docs/project-status/failure-library.md`.
- Regenerate all sibling Markdown under `docs/project-status/`.
- Create two new HTML pages and regenerate the existing nine under
  `reports/project-status/`.
- Refresh the sanitized derived copy under
  `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/` and its portal
  index only after final verification.

---

### Task 1: Case, rights, and disclosure preflight

**Files:**
- Read: `reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md`
- Read: `reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md`
- Read: `docs/experiments/2026-07-24-omni-semantic-quality.md`
- Read: ignored 2025-training run artifacts only when needed to recover exact
  recorded outputs; never edit, stage, or copy those artifacts
- Modify: `docs/superpowers/plans/2026-07-26-living-research-memory.md`
  checkboxes only after the selection is verified

**Interfaces:**
- Consumes: design policy in
  `docs/superpowers/specs/2026-07-26-living-research-memory-design.md`.
- Produces: a verified list of three to five admissible case IDs, exact tracked
  observation/prediction/grade fields, and a media decision for Tasks 2 and 3.

- [x] **Step 1: Verify checkout and shared-data boundary**

Run:

```bash
git status --short --branch --untracked-files=all
readlink -f data
git diff --check
```

Expected: branch `codex/repository-cleanup-report`; `data` resolves exactly to
the canonical shared dataset path; only this design/plan work is uncommitted.

- [x] **Step 2: Select candidate cases from tracked evidence**

Start with these documented mechanisms, not with visually interesting cherry
picks:

```text
NN7G28sBOlY — accepted transcript contained the needed answer; wrong span chosen
NW3Mhr76ClM — non-lexical chuckle missed; generic ASR gate accepted wrong text
NrB4LPosWBk — automatic and uploader text contained the reference word; answer unchanged and wrong
EMI_anpHII4 — music/no-speech Whisper repetition or timestamp-overrun failure
IWSPAfCKtV8 — native-audio distraction: church bell answered as wedding-march music
```

Choose three to five only when the tracked report and frozen artifact agree on
question, condition, prediction, grade, and post-hoc diagnosis. Record no hidden
reasoning; `model observation` is present only if the run explicitly emitted an
observation field.

- [x] **Step 3: Verify split and label boundary**

For every selected case, confirm it is a 2025 training row or an external,
self-recorded, or synthetic example. Reject any row whose split cannot be
proven. Mark every answer-informed diagnosis `Post-hoc diagnosis`.

- [x] **Step 4: Decide media separately from textual admissibility**

For each candidate asset, locate an authoritative license or project rights
record. Classify it as one of:

```text
cleared for public report
text-only because rights are unclear
inadmissible
```

Do not infer redistribution rights merely from local dataset access or a public
video URL. For an external CC-BY/self-recorded/synthetic worked example, retain
the exact source/license and required attribution. If no TRECVID pixel clears
the gate, use timestamps and plain-language evidence descriptions for TRECVID
cases and reserve imagery for the cleared worked example.

- [x] **Step 5: Check extracted text before authoring**

Search the selected fields for local paths, bucket URIs, tokens, credentials,
raw log fragments, and 2026 identifiers. Expected: none survive into the
authoring notes. Keep raw evidence in place; do not create a scratch export in
the repository.

---

### Task 2: Start Here and the worked current-approach example

**Files:**
- Modify: `docs/project-status/index.qmd`
- Create: `docs/project-status/worked-example.qmd`
- Modify: `docs/project-status/styles.css`

**Interfaces:**
- Consumes: Task 1's verified example and media decision; the existing router
  anchor `fig-recommended-router`.
- Produces: the newcomer path and stable `worked-example` route linked by Tasks
  3–5.

- [ ] **Step 1: Rewrite the Start Here opening**

Put these visible sections before model/metric detail:

```text
Lifecycle banner
Project in one minute
What the system sees
Current recommendation
See one answer from start to finish
```

Define VQA, frame budget, uniform sampling, ASR, lexical speech, frozen
prediction, and label-blind at first use. Preserve the evidence boundary that
2026 evaluation quality remains unknown.

- [ ] **Step 2: Convert the current decisions to a compact decision strip**

Use text labels with short plain-language descriptions:

```text
Use now — Qwen3.5-4B, 12 uniform frames
Use for speed — MiniCPM-o 4.5, audio off
Route selectively — accepted Whisper for spoken-word questions
Still testing — category-routed native audio and repaired Qwen3-Omni
Do not default — global 24 frames and tested directed/burst heuristics
```

Keep the full router diagram in a native disclosure titled `Expand the full
evidence router`; retain its alt text, caption, and plain-text equivalent.

- [ ] **Step 3: Author `worked-example.qmd`**

Use this exact section order:

```text
Breadcrumb: Start Here / How One Answer Is Made
Why this example
1. Question and difficulty
2. What the baseline receives
3. Route decision
4. Evidence added
5. Recorded model observation
6. Frozen prediction
7. Expected-answer disclosure and grade
8. Post-hoc diagnosis and next test
Evidence received, run details, grading, and provenance (details)
What this example cannot establish
```

Label the page `one traced example, not the policy`. If Task 1 clears a
publishable asset, show an optimized contact sheet/timeline with timestamps;
otherwise show a timestamp timeline and text equivalent without pixel media.
Never invent intermediate reasoning.

- [ ] **Step 4: Add a compact newcomer glossary**

Define `C/P/W`, exact match, token F1, METEOR, panel, stratum, promotion gate,
and protocol version. Place each definition near first use and keep a lookup
table at the end of Start Here.

- [ ] **Step 5: Render the two sources in an isolated temporary copy**

Run:

```bash
status_task2_dir=$(mktemp -d /tmp/trecvid-status-task2.XXXXXX)
cp -a docs/project-status "$status_task2_dir/source"
cd "$status_task2_dir/source"
env XDG_CACHE_HOME="$status_task2_dir/cache" quarto render index.qmd --to gfm --output-dir ../gfm --output index.md
env XDG_CACHE_HOME="$status_task2_dir/cache" quarto render worked-example.qmd --to html --output-dir ../html --output worked-example.html
```

Expected: both render 0; no repository output changes yet. Inspect that the
worked example remains understandable with all disclosures closed.

---

### Task 3: Living failure library

**Files:**
- Create: `docs/project-status/failure-library.qmd`
- Modify: `docs/project-status/styles.css`
- Optional create: cleared optimized assets under
  `docs/project-status/assets/cases/`

**Interfaces:**
- Consumes: Task 1's verified cases and Task 2's glossary terms.
- Produces: stable case anchors used by the hub and theme pages.

- [ ] **Step 1: Author the library boundary**

State above the first case:

```text
This is a selected development-case library, not a prevalence estimate or leaderboard.
Repeatedly inspected cases are development evidence, never held-out validation.
No 2026 evaluation row is admissible.
Expected answers and media appear only under the recorded disclosure and rights policy.
```

Add status and diagnosis keys in plain language.

- [ ] **Step 2: Author three to five six-field case cards**

For each stable `case-*` anchor, keep this content visible:

```text
What failed
Change tried
Result
Post-hoc diagnosis plus competing explanation
Limit
Next falsifying test
```

Show a compact metadata line with case ID, lifecycle, owner, last verified,
source/split role, and linked theme.

- [ ] **Step 3: Add expert disclosures**

Each disclosure is titled `Evidence received, run details, grading, and
provenance` and contains only verified fields: evidence configuration, selected
timestamps/spans/events, recorded model observation if available, frozen
prediction, expected-answer disclosure, metric or C/P/W grade, rater/blinding/
adjudication provenance, uncertainty, next comparison, immutable evidence
links, and supersession.

- [ ] **Step 4: Add media only when the case attestation is complete**

Place the public-safe attestation next to the asset in the same QMD disclosure:

```text
Asset ID
Rights class
Source/license identifier
Allowed report use
Verifier
Verification date
Required attribution
```

Every image receives alt text, a caption, and a nearby textual evidence
equivalent. Audio or derivative media receives a transcript/event description.

- [ ] **Step 5: Render and inspect the failure page in isolation**

Render both GFM and HTML from a temporary source copy. Confirm cases are
readable with disclosures closed, stable anchors exist, no row is from 2026,
and every media-bearing case has the complete rights attestation.

---

### Task 4: Layer the six research themes and hub

**Files:**
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify: all six `docs/project-status/research-*.qmd` files
- Modify: `docs/project-status/styles.css`

**Interfaces:**
- Consumes: Task 3 case anchors and Task 2 definitions.
- Produces: a non-linear hub plus six consistent newcomer-to-expert theme pages.

- [ ] **Step 1: Update the hub without creating another ledger**

Keep the six independent theme cards. Add a compact matrix with columns:

```text
Theme | Why it matters | Tried | Learned now | Still open | Next falsifying test
```

Use short status tokens plus theme-record anchors; do not duplicate detailed
results. Add a three-case preview linked to the Failure Library.

- [ ] **Step 2: Make every major research branch visible in the next-test queue**

Include one concise, linked item for smarter visual sampling/retrieval,
alternative ASR, human/reference transcript controls, routed native
audio/sound, repaired/specialist models or bottleneck-first adaptation, and
routing/fusion/answer election.

- [ ] **Step 3: Apply the layered theme contract**

On each of the six theme pages, use this order:

```text
Breadcrumb and local navigation
New to this theme?
Current working answer: use now / do not default / still possible / next test
Evidence at a glance
What has been tried
Known failures and cases
Open hypotheses and ideas
Cheapest valid next experiments
Method, result, validity risks, and evidence (details)
Related themes
```

Explain each theme's basic concept with one concrete example before metrics.
Keep exact experiment headings and legacy anchors visible outside disclosures.

- [ ] **Step 4: Preserve condition and audit boundaries**

For every negative/incomplete record, confirm the visible summary includes:

```text
What this rules out now
What it does not rule out
Possible confirmed defect, audit gap, or uncertainty
Reconsider when
```

Do not turn directed sampling, native audio, alternate models, or transcripts
into permanent rejections.

- [ ] **Step 5: Render all seven sources in a temporary copy**

Render the hub and six themes to GFM and HTML. Confirm each theme is
understandable before opening methodology details, all legacy fragments land on
visible content, and no theme implies a linear research sequence.

---

### Task 5: Code flow, link semantics, lifecycle, and site shell

**Files:**
- Modify: `docs/project-status/code-map.qmd`
- Modify: `docs/project-status/_quarto.yml`
- Modify: `docs/project-status/styles.css`
- Modify: `README.md` only if its report links or description become stale

**Interfaces:**
- Consumes: stable routes and anchors from Tasks 2–4.
- Produces: final site navigation, lifecycle metadata, link legend, and
  implementation crosswalk.

- [ ] **Step 1: Add one traced flow-to-code example**

Map the worked example through acquisition → evidence preparation → route →
inference → freeze → score → report. For each stage name the reusable CLI or
module and mark bespoke scripts explicitly. Keep operational warnings before
any expensive command.

- [ ] **Step 2: Label link kinds and access state**

Use visible labels for Report page, Evidence report · GitHub, Issue/PR history
· GitHub, Source code · GitHub, Evidence of record, and Current code. Mark
GitHub evidence as requiring repository access until public release. Pin
historical evidence-of-record links to a commit SHA or release tag; reserve
`main` for evolving code links.

- [ ] **Step 3: Update lifecycle and navigation**

Keep navbar text exactly `Start Here`, `Research Themes`, and `Code Map and
Gaps`. Add the two new pages to Quarto rendering without flattening them into
the navbar. Replace the stale fixed footer date with the reviewed evidence date
and phase. Keep search off for the initial three-to-five-case library.

- [ ] **Step 4: Finish responsive and focus styles**

Reuse the current Cosmo theme and existing spacing/color language. Verify
visible keyboard focus, text-plus-color statuses, disclosure summaries,
scrollable large figures, one-column mobile cards, and no clipped code/URLs.
Do not add custom JavaScript.

---

### Task 6: Generate, verify, and refresh the private portal

**Files:**
- Generate: all sibling `docs/project-status/*.md`
- Generate: all `reports/project-status/*.html`
- Refresh: sanitized derived HTML under the authorized rendered-artifact portal
- Modify: this plan's checkboxes/status with final verification evidence

**Interfaces:**
- Consumes: all canonical QMD and styles from Tasks 2–5.
- Produces: reviewable GitHub Markdown, self-contained HTML, and a verified
  private live preview.

- [ ] **Step 1: Render GFM and HTML from QMD**

Run the repository's established Quarto render commands for every QMD. Expect
eleven QMD, eleven sibling Markdown files, and eleven self-contained HTML pages.
Do not hand-edit generated files.

- [ ] **Step 2: Prove deterministic generated outputs**

Render an isolated copy with a task-specific `XDG_CACHE_HOME` and compare all
22 generated files byte-for-byte with repository outputs. Investigate any
unexpected nondeterminism before publication.

- [ ] **Step 3: Inspect desktop and phone flows**

Using the user's existing browser flow or authorized headless Chrome, inspect
Start Here, worked example, Research Themes hub, Failure Library, one
representative theme, and Code Map at 1440 pixels and 390 pixels. Check
navigation, focus, disclosures, reading order, image scaling, table/diagram
overflow, and whether newcomer content appears before expert metrics.

- [ ] **Step 4: Run accessibility, link, and disclosure checks**

Confirm every visual has meaningful alt text, caption, and nearby text/table
equivalent. Check repository-relative paths, stable fragments, and external
links; classify private-GitHub authentication separately from broken 404s.

- [ ] **Step 5: Run publication-boundary scans**

Scan QMD, Markdown, HTML, and any assets for secrets, credentials, bucket URIs,
local paths, raw logs, 2026 identifiers, unapproved frames/audio/transcripts,
and private licensing records. Confirm all included media have their public-safe
attestation and required attribution. Run `git diff --check`.

- [ ] **Step 6: Refresh only the authorized private portal copy**

Copy the eleven sanitized HTML files and cleared required assets to
`/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`, refresh
`/home/npatta01/codex-rendered/index.html`, and remove no uncertain or
user-owned artifact. Verify the live HTTPS routes return the intended bytes and
that `tailscale funnel status` remains tailnet-only.

- [ ] **Step 7: Record final evidence and commit the reviewed revision**

Update this plan with exact render counts, responsive checks, link results,
privacy/media results, live byte comparisons, and portal status. Commit only
the intended source, generated, design, plan, and cleared asset files; confirm
no `data/` path is staged.

## Success check

Before handoff, a newcomer must be able to explain the baseline, transcript
route, freeze boundary, and condition-bound meaning of a negative result after
reading Start Here plus one example. An expert must be able to trace a case to
its exact condition, panel, grader provenance, validity boundary, next
falsifying test, immutable evidence, and implementation without relying on an
unlabeled GitHub login-wall link.
