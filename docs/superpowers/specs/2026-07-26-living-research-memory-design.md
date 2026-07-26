# Living Research Memory and Failure Library

**Date:** 2026-07-26

**Audience:** Newcomers to video question answering, project contributors, and
expert reviewers

**Canonical sources:** `docs/project-status/*.qmd`

**Delivery:** GitHub-readable Markdown plus self-contained HTML on the existing
private tailnet-only portal until the repository is intentionally made public

**Approved refinement:** Integrate legacy experiment bookmarks into the six
current research themes, add source-video and relevant-moment links without
redistributing media, and provide a native light/dark presentation with
text-labelled evidence states.

## Objective

Turn the current status site into an approachable living research memory. A
reader who is new to the field should understand the problem, the current
system, and one concrete answer trace before encountering dense experiment
records. An expert should still be able to inspect exact conditions, panels,
validity limits, decision rules, and immutable evidence.

The revision must answer five questions without requiring institutional memory:

1. What does the project currently do, in plain language?
2. Why did the project test each broad research direction?
3. What was tried, what was learned, and what remains possible?
4. Where and how does the current approach fail on concrete cases?
5. What is the smallest trustworthy experiment that could change the current
   answer?

## Approved direction

Use a **focused hybrid** rather than either a single long report or a large
documentation portal:

- retain three top-level navigation choices;
- make the existing Research Themes page the research-memory hub;
- layer each theme page from newcomer explanation to expert audit trail;
- add one worked end-to-end example and one living failure library;
- keep chronology, supersession, and exact methodology available but secondary;
- preserve QMD as the only editable report source;
- do not add a structured ledger service, custom schema framework, CI suite, or
  new application tests.

Review advice is a design input; repository evidence remains authoritative for
experimental claims.

## Alternatives considered

### 1. Minimal retrofit

Add definitions, examples, and failure records directly to the existing pages.
This is the smallest diff, but it would make already long theme pages more
overwhelming and give the living case library no stable home.

### 2. Focused hybrid — selected

Keep the small top-level site, add two focused reader-invoked pages, and use a
consistent layered contract on every theme page. This provides room for growth
without turning the report into a documentation maze.

### 3. Full research portal

Create separate primer, glossary, evidence matrix, provenance, worked-example,
case-library, and change-log destinations. This scales furthest but creates too
many pages and too much maintenance for the current evidence base.

## Information architecture

Keep the top navigation compact:

1. **Start Here** — project in one minute, current recommendation, one traced
   example, evidence limits, and reading paths.
2. **Research Themes** — the project's living research memory: broad themes,
   current state matrix, failure library, open ideas, supersession, and
   chronology. Preserve the familiar label and existing `experiment-ledger`
   URL.
3. **Code Map and Gaps** — acquisition through reporting, reproducible commands,
   bespoke seams, engineering gaps, and contribution path.

Reader-invoked pages linked from Start Here and Research Themes:

- **How One Answer Is Made** — a concrete, fully traced current-approach example
  at `worked-example.qmd`, rendered beside it as `worked-example.md` and at the
  stable HTML route `worked-example.html`.
- **Failure Library** — living case records with evidence, prediction,
  diagnosis, uncertainty, and next test.
- the existing six broad research-theme pages.

Do not put every page in a flat navbar. Use breadcrumbs and a short local
navigation row on deep pages. Replace the large `Back to Research Themes`
heading with a quiet breadcrumb. The worked example's primary breadcrumb parent
is Start Here; the Failure Library and six theme pages belong to Research
Themes.

### Legacy experiment routing

Earlier experiments are not a seventh research area. Group each one into its
current thematic home:

| Current theme | Earlier records grouped there |
|---|---|
| Experiment integrity and evaluation | Field guide, data readiness, compatibility gate, and Frozen-30 protocol integrity |
| Visual and temporal evidence | Frame budgets, directed sampling, dense bursts, and spatial-crop diagnostics |
| Speech and transcript evidence | Automatic-transcript ablation, Seamless, and human/reference transcript controls |
| Native audio and sound | Audio-on/off, prosody, non-speech, and sound-event diagnostics |
| Models and adaptation | Qwen3.6, Phi/Gemma, and Qwen3-Omni compatibility and quality screens |
| Routing, fusion, and answer selection | Specialist routing, taxonomy, retrieval, evidence use, and answer selection |

Remove the visible nineteen-row compatibility index. Preserve its existing
fragment IDs as zero-height aliases inside the corresponding six theme cards on
the hub, so an old `experiment-ledger.html#...` bookmark lands on a meaningful
current destination. Generic former anchors such as the research-question map
belong at the six-theme map itself. A record spanning themes receives one
primary home and explicit related-theme links; it is not duplicated.

Keep only cross-cutting supersession, chronology, and issue/PR history in a
bottom-of-page **History and provenance** section. Make that section unlisted
from the page contents and put its detailed tables in native disclosures. Do
not create a separate legacy page or add redirect JavaScript.

## Progressive disclosure contract

Every major page should follow the same depth order:

1. **Plain-language answer** — what this means and why it matters.
2. **Concrete example** — what a question, evidence choice, or failure looks
   like.
3. **Evidence at a glance** — current status, strongest evidence, boundary, and
   cheapest valid next test.
4. **Detailed records** — exact condition, panel, result, decision, and
   reconsideration rule.
5. **Expert audit trail** — methods, metrics, validity risks, and provenance in
   clearly named native disclosure controls.

No claim may exist only inside a closed disclosure. The visible summary must be
enough to decide whether opening the detail is worthwhile.

## Start Here design

The first screen should contain:

- a lifecycle banner: `Active pre-competition research`, evidence-through date,
  and the explicit unknown about the 2026 evaluation set;
- **Project in one minute:** a first-principles explanation of video question
  answering and the fixed evidence budget;
- the current default in ordinary language before model names: twelve evenly
  spaced still frames go to Qwen3.5-4B; a quality-gated automatic transcript is
  added only for questions that ask about spoken words;
- a compact decision strip with text labels: **Use now**, **Use for speed**,
  **Route selectively**, **Still testing**, and **Do not default**;
- one link to the worked example and one link to the failure library.

Define jargon at first use. A compact glossary later on the page remains useful
for lookup, but it must not be the first place a reader can learn terms such as
VQA, uniform sampling, frame budget, lexical speech, ASR, panel, frozen,
label-blind, exact match, token F1, METEOR, and C/P/W.

The existing recommended-router figure remains the canonical policy diagram.
Its large rendering should be expandable, with the short textual route visible
by default. A worked example instantiates this policy; it must not introduce a
second competing router.

## Worked-example design

Create one page titled **How One Answer Is Made**. Select a verified 2025
training example whose disclosure and media rights are documented. Prefer a
lexical-speech case because it demonstrates both the frame-only baseline and
the routed transcript specialist. If no suitable real case is publishable, use
a clearly labeled synthetic example and keep the same system trace.

The storyboard is:

1. **Question and difficulty** — what is being asked and why visual frames alone
   may be insufficient.
2. **What the baseline receives** — a 12-position timeline or authorized contact
   sheet with timestamps and a plain-language explanation of uniform sampling.
3. **Route decision** — why the question is or is not eligible for transcript
   evidence, including the quality gate and frame-only matched control.
4. **Selected evidence** — accepted frame timestamps, transcript span, or audio
   event; distinguish automatic ASR from human text.
5. **Model observation** — the model's intermediate description if the recorded
   run produced one; do not invent hidden reasoning.
6. **Frozen prediction** — the exact or sanitized output and the label boundary.
7. **Expected answer and grade** — only when disclosure is allowed; otherwise
   show the allowed answer class or state that it is withheld.
8. **Diagnosis and next test** — retrieval failure, evidence-use/fusion failure,
   output-contract failure, grading artifact, or unknown; include what this one
   case cannot establish.

Expert-only run metadata belongs in a native disclosure below the storyboard:
model revision, prompt identifier, decoding settings, frame/pixel budget, ASR
model and gate, run digest, grader provenance, and evidence link.

## Research Themes hub

The hub remains thematic and non-linear. It contains:

1. a concise explanation of why broad research themes are the organizing unit;
2. a six-theme card map without arrows or implied sequence;
3. a compact **tried / learned / open / next falsifying test** matrix;
4. a preview of recent or representative failure records;
5. a living next-test queue grouped by research theme, with one always-visible
   item for each major branch: smarter visual sampling/retrieval, alternative
   ASR, human/reference transcript controls, routed native audio/sound, model
   repair/adaptation, and routing/fusion;
6. status-vocabulary and result-interpretation guidance;
7. a collapsed, unlisted **History and provenance** appendix containing
   supersession, chronology, and the issue/PR trail.

The matrix is an index, not another prose ledger. Each cell contains a short
status token and a link to the canonical theme-page record. Detailed findings
remain owned by the theme pages. Shared one-line status content may live in a
QMD include so Start Here and the hub cannot drift, but no separate structured
data system is introduced.

Legacy experiment aliases live on the six existing theme cards rather than in
a second visible hierarchy. The theme page's **What has been tried** section is
the canonical readable home for both current and earlier experiment records.

## Theme-page contract

Keep the current rigorous experiment records, but change their reading order:

1. breadcrumb and local navigation;
2. **New to this theme?** — first-principles background, one concrete example,
   and definitions;
3. **Current working answer** — a four-part decision strip: use now, do not
   default, still possible, and next smallest test;
4. **Evidence at a glance** — condition, panel size, result, strength, and
   generalization boundary;
5. **What has been tried** — compact experiment summaries;
6. **Known failures and cases** — links into the Failure Library;
7. **Open hypotheses and ideas** — including smarter sampling, alternative ASR,
   human/reference transcript controls, routed native audio, retrieval,
   grounding, ensembles, and adaptation where relevant;
8. **Cheapest valid next experiments**;
9. collapsible methodology, validity risks, and evidence;
10. related themes.

Every negative result continues to state:

- what it rules out under the tested condition;
- what it does not rule out;
- what could have gone wrong or remains unaudited;
- the condition under which it should be reconsidered.

This prevents `not promoted` from being misread as `can never work` and keeps
possible bugs, audit gaps, small-panel uncertainty, and confirmed defects
separate.

## Living failure library

Create `docs/project-status/failure-library.qmd` as the initial canonical
source. Keep the first release to three to five carefully selected cases. Do
not create per-case files or a schema framework until page size or contribution
volume demonstrates the need.

Each case uses a consistent QMD section template. Its visible content reads as
one evidence trace:

`source and question → evidence received → frozen prediction → grade → possible
failure location → bug status → next falsifying test`

The card stays compact:

- **What failed** — the question or failure in plain language;
- **Source media** — source ID, original-video link, relevant-moment link when a
  trustworthy timestamp exists, and `external source; media not redistributed`;
- **Evidence received** — frame policy/timestamps and transcript or native-audio
  treatment, without reconstructing hidden reasoning;
- **Change tried** — the intervention and matched control;
- **Result** — prediction versus allowed expected-answer disclosure and grade;
- **Possible failure location** — acquisition/runtime, evidence
  extraction/retrieval, gate/routing, fusion/evidence use, answer
  selection/output contract, scoring/adjudication, or unknown;
- **Diagnosis and bug status** — leading explanation plus competing explanation,
  separately labelled as confirmed defect, audit gap, post-hoc diagnosis, open
  hypothesis, or unknown;
- **Limit** — what this selected case cannot establish;
- **Next test** — the smallest falsifying comparison.

The following identity fields remain visible in a compact metadata line: stable
case ID, lifecycle, owner, last verified date, source/split role, and linked
theme. Put the remaining reproducibility and provenance fields inside a native
disclosure named **Evidence received, run details, grading, and provenance**:

- question and why it is hard;
- source URL, source availability, relevant interval, and media-rights status;
- evidence configuration: model/revision, frame policy/count, audio/transcript
  treatment, prompt ID, and decoding settings where known;
- what the model received: publishable contact sheet or timestamps, accepted
  transcript span, and/or sound evidence;
- recorded model observation, if available;
- frozen prediction;
- expected-answer disclosure: full, answer-class only, or withheld;
- grade and provenance: metric or C/P/W label, rubric, rater count, blinding,
  and adjudication status;
- diagnosis and competing explanation;
- uncertainty and the limit of an `n=1` case;
- next falsifying test and held-constant comparison;
- evidence-of-record and supersession links;
- media-rights record when any pixel, audio asset or derivative, transcript
  excerpt, or other licensed material is included.

Repeatedly inspected cases are development evidence, never held-out validation.
The page must say this near the top. Case counts describe a selected failure
library, not prevalence or a leaderboard.

## Media and answer-disclosure policy

The user intends the canonical Git report to become public after the
competition, so the design supports real examples rather than a permanently
private gallery. Publication intent does not itself grant redistribution
rights, and private/internal access does not automatically change that rights
boundary. The report therefore uses a link-only media policy unless a later
rights record clears redistribution:

- no 2026 evaluation row, reference answer, frame, transcript, or prediction is
  admissible;
- a verified 2025 training case may expose its already-tracked source ID, a
  direct platform-hosted original-video URL, and a timestamped relevant-moment
  URL; those links do not copy the video into the repository or report;
- the visible question remains a verified paraphrase unless its annotation text
  has a separate disclosure basis; expected answers and transcript excerpts
  remain subject to their existing disclosure gates;
- pixel media requires a recorded rights class, source/license, verifier, and
  verification date;
- audio clips, generated audio previews, waveforms, and spectrogram derivatives
  require the same documented rights, provenance, verifier, and verification
  date; when unclear, use text-only timing or sound-event descriptions;
- expected answers use `full`, `answer-class only`, or `withheld` disclosure;
- ambiguous rights use an outbound source link plus a text-only
  timestamp/timeline, or a self-recorded, synthetic, public-domain, or otherwise
  cleared stand-in;
- raw videos and large contact sheets do not enter Git history; use optimized
  stills only when they materially aid understanding;
- do not embed a YouTube player, remote thumbnail, or other runtime third-party
  media request in the self-contained report;
- self-contained HTML embeds included media, so the same rights gate applies to
  both source assets and rendered output.

This is a conservative publication policy, not a legal conclusion that a small
research contact sheet could never qualify as fair use. Fair use is
fact-specific, and a private or noncommercial purpose is only one factor. Until
the project records a redistribution basis, link to the source rather than copy
its pixels. See the [U.S. Copyright Office fair-use
overview](https://www.copyright.gov/fair-use/more-info.html) and [YouTube Terms
of Service](https://uk.youtube.com/t/terms).

The canonical public-safe rights attestation lives inside the media-bearing
case's QMD section, under its provenance disclosure. It records: asset ID,
rights class, source or license identifier, allowed report use, verifier handle,
verification date, and any required attribution. Private contracts or source
licensing files are not copied into the report or portal.

Case selection, narrative drafting, and displayed intermediate observations may
not use any 2026 evaluation row or reference answer. For a disclosed 2025
training case, any diagnosis made after comparing with the expected answer must
be labeled **post-hoc diagnosis** and may never be presented as evidence the
router or model had before prediction freeze.

Until explicit public release, rendered HTML remains on the existing
tailnet-only portal. Copy only the sanitized derived HTML and required cleared
assets under `/home/npatta01/codex-rendered/plans/`, and refresh
`/home/npatta01/codex-rendered/index.html`. Do not expose repository trees, raw
media, datasets, logs, private licensing records, or unrelated files. No public
deployment is part of this revision.

## Evidence-link strategy

The current absolute GitHub links were a conscious way to keep the portal
small, but they are currently a login wall because the repository is private.
The revision makes link type and access explicit:

- **Report page** — local, always available in the rendered mini-site;
- **Evidence report · GitHub** — repository evidence; marked `requires repository
  access until public release` while private;
- **Issue/PR history · GitHub** — discussion and decision trail;
- **Source code · GitHub** — implementation, not quality evidence;
- **Evidence of record** — commit-SHA or release-tag permalink for historical
  claims;
- **Current code** — `main` link only when following the evolving branch is the
  intended behavior.

Every page must summarize the evidence needed for its visible conclusion; a
reader should not need GitHub access merely to understand the claim. Link
checks must distinguish an authentication-required private URL from a genuinely
broken path.

## Lifecycle and maintenance

Use a small shared lifecycle banner rather than many hand-maintained metadata
fields:

- document phase: pre-competition research, frozen for submission, or
  post-competition retrospective;
- evidence through date;
- owner.

Case records additionally carry their own last-verified date and lifecycle.
The post-competition transition must replace present-tense unknowns with a
retrospective outcome while preserving the state of knowledge at submission.

The report should eventually expose search when the repository becomes public
or the case library grows enough to need it. Search is not required for the
first three-to-five-case revision.

## Visual system

### Light, dark, and status presentation

Use Quarto's native dual-theme support with the same offline Cosmo foundation
in both modes. Show the built-in light/dark selector, respect the reader's
system preference, remember the explicit selection, and keep all required CSS
and toggle assets inside the self-contained HTML. Use mode-specific page,
surface, text, muted-text, border, link, focus, and Mermaid tokens; remove
hard-coded white surfaces and dark-only diagram text.

Use a restrained semantic status vocabulary, always with visible text and a
non-color cue such as an outlined chip or left border:

| Meaning | Visual family |
|---|---|
| **Use now / verified** | teal |
| **Routed / informational** | blue |
| **Diagnostic / experimental / open** | amber |
| **Not promoted under the tested condition** | muted rose |
| **Blocked / unknown / historical** | slate or violet |

Color never distinguishes the six research themes themselves; theme identity
comes from headings and links. In failure traces, use subtle field accents for
evidence, observed result, diagnosis/uncertainty, limit/provenance, and next
test without tinting entire cards. Provide a two-color keyboard focus ring,
forced-colors support, minimum target sizing for disclosures and the theme
toggle, and a light, ink-efficient print treatment.

Use approximately five information-bearing visuals across the site, reusing
existing figures where they remain canonical:

1. **Current evidence router** — compact by default, expandable full diagram,
   with visible text equivalent.
2. **One-answer storyboard** — question through freeze and grade.
3. **What twelve frames means** — a schematic timestamp timeline by default,
   with an authorized contact sheet only after rights clearance.
4. **Anatomy of a failure** — retrieval versus evidence-use versus output or
   grading failure.
5. **Research-theme state map** — non-linear theme cards or matrix, with no
   false chronology.

New visual assets may be generated when a real asset is unavailable, but they
must be explicitly labeled as conceptual or synthetic and may not imply that a
generated scene is a real experiment frame. Every visual requires meaningful
alt text, a caption, and a nearby prose or table equivalent. Status is always
communicated in text, never by color alone.

## Source and output policy

- QMD remains the sole editable report source.
- Markdown and HTML remain generated outputs.
- Existing historical evidence stays at its current path.
- No raw logs, unrestricted datasets, secrets, or 2026 evaluation artifacts
  enter the report.
- No application code, application tests, CI, schema validators, or test
  framework changes are in scope.
- The root README continues to link to Start Here, operational data
  instructions, the field-guide background, and report indexes.

## Verification

Use lightweight documentation verification only:

1. render every QMD to GitHub-readable Markdown and self-contained HTML;
2. verify desktop and 390-pixel phone layouts in light and dark modes,
   navigation, the theme selector, disclosure controls, focus, readability, and
   horizontal overflow;
3. verify every visual has alt text, caption, and nearby text/table equivalent;
4. verify every failure record contains the agreed visible fields and that no
   2026 evaluation row appears;
5. verify every selected failure case has a working original-video link,
   relevant-moment link when supported, and an explicit link-only media status;
6. inspect any committed media rights records and generated HTML embedding;
7. check repository-relative links and classify GitHub links as accessible,
   authentication-required, or broken;
8. run `git diff --check` and a brief secret/private-data scan over generated
   outputs;
9. verify all legacy fragments resolve to their intended current theme card;
10. compare source QMD, generated Markdown, local HTML, and live portal copies;
11. confirm the Tailscale mapping remains tailnet-only.

Do not run the application test suite and do not add a custom report test suite.

## Success criteria

A newcomer can answer, after following Start Here and one example:

- what evidence the baseline receives;
- why twelve uniform frames are the default;
- when a transcript may be added;
- what a frozen prediction means;
- why one failed experiment does not reject an entire research direction.

An expert can answer:

- which exact condition and panel support a decision;
- whether a record is representative, failure-selected, or a tiny diagnostic;
- whether a suspected problem is a confirmed defect, audit gap, or uncertainty;
- what evidence the model actually received and produced on a failure case;
- what matched, label-blind experiment could falsify the current answer;
- where the immutable evidence, decision history, and implementation live.

A returning reader with an old deep link lands on the appropriate current
research theme without reading a separate migration index. A failure-library
reader can reach the original source and relevant moment without the repository
redistributing video pixels.

The report succeeds when both readers can reach those answers without scanning
a chronology, following an unexplained login-wall link, or mistaking a narrow
negative result for a universal conclusion.
