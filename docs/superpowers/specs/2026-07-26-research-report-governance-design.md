# Research Report Governance Design

**Status:** Approved on 2026-07-26 through the user's instruction to proceed
after reviewing the two-advisor synthesis.

## Problem

The theme-first report is readable and scientifically cautious, but four
governance questions are not yet explicit enough for sustained iteration:

1. what owns a run, a scientific claim, and a current project decision;
2. how evidence validity differs from the decision the project makes;
3. where invalid, incomplete, corrected, and superseded work belongs; and
4. when an idea should remain inside a theme versus create, split, or merge a
   theme.

Named experiments are also becoming harder to find because search is disabled.
The solution must preserve the current approachable theme pages rather than
introduce a database, schema-heavy ledger, or large dashboard.

## Goals

- Keep the existing theme-centered information architecture.
- Define one primary theme per scientific claim, while allowing one evidence
  bundle to support several linked claims.
- Distinguish evidence validity from project decision disposition.
- Give agents deterministic paths for cross-theme work, new themes, invalid or
  incomplete runs, corrections, and supersession.
- Make named experiments directly discoverable through a compact static index.
- Complete two high-impact older records that omit fields now required by the
  contributor guide.
- Preserve historical links, progressive disclosure, phone readability, dark
  mode, and the existing disclosure boundary.

## Non-goals

- No database, JSON/YAML registry, generated schema, custom validator, CI job,
  or new application test suite.
- No new report page, navigation category, leaderboard, Gantt chart, long
  chronology graphic, or raw evidence dump.
- No opaque identifier migration across every historical record. The canonical
  page plus authored section anchor is the stable claim identity.
- No broad rewrite of historical experiments whose visible records already
  satisfy the reporting contract.
- No issue creation, push, pull request, or public deployment.

## Core model

The report will define three linked concepts:

- **Evidence bundle:** the immutable record of what executed, including the
  frozen configuration, completion count, outputs, and evidence-of-record link.
- **Claim record:** one condition-bound scientific conclusion. Its stable
  identity is the canonical theme page plus authored section anchor.
- **Project decision:** what the project does now because of one or more claims.

One primary theme owns each claim. A single run may support several claims when
the claims have independently testable conclusions and decision rules. Each
claim links to the shared evidence bundle and to related claims; the full
record is not copied between themes.

## Two independent status axes

The report will name two axes rather than treating every label as one status.

### Evidence validity

- **Verified:** the declared protocol completed and supports the stated
  condition-bound result.
- **Diagnostic:** deliberately limited evidence that selects a next test but
  does not estimate prevalence or general quality.
- **Compatibility-only:** execution or interface success without quality
  evidence.
- **Incomplete:** the declared evidence bundle did not complete; partial
  observations cannot satisfy the planned comparison.
- **Invalid:** a demonstrated integrity failure makes the affected comparison
  unusable for its intended claim.
- **Blocked:** a named prerequisite is absent, so the planned evidence cannot
  yet be produced or interpreted.
- **Unknown:** no admissible evidence establishes the claim.

### Decision disposition

- **Current default**
- **Fast alternate**
- **Routed candidate**
- **Diagnostic signal**
- **Not promoted under the tested condition**
- **Superseded**
- **Open hypothesis**

Evidence validity answers whether and how the evidence can be used. Decision
disposition answers what the project does now. A diagnostic bundle can justify
a diagnostic signal, while an incomplete or invalid bundle cannot satisfy a
promotion rule.

## Theme lifecycle

An idea stays under the nearest theme's **Open hypotheses and ideas** until a
new theme passes all four conditions:

1. its working answer cannot be expressed as a refinement of an existing
   theme;
2. it has an independent decision or promotion rule;
3. it needs its own evidence boundary and next-test queue; and
4. it is expected to produce at least two durable claims, or it is one
   consequential multi-iteration research program.

Split a theme only when two subareas can change their working answers
independently and each has its own decision rule, evidence program, and queue.
Merge themes when those elements no longer vary independently and most new
claims require the same evidence and controls. Page length alone never creates
or splits a theme.

A new theme uses the established page headings, adds a hub card and matrix row,
links its next work from the branch-complete queue, adds reciprocal related-theme
links, and is included in normal render, link, privacy, accessibility, and
responsive checks. The visible hub heading becomes **Current research-theme
map** while the legacy `#six-theme-research-map` fragment remains valid.

## Invalid, incomplete, corrected, and superseded work

- **Valid but negative:** retain the claim and use the condition-bound
  disposition **Not promoted under the tested condition**.
- **Incomplete:** preserve the evidence bundle and completion denominator; do
  not score it as a completed promotion comparison.
- **Invalid:** preserve the evidence bundle, mark the intended claim unusable,
  identify affected downstream conclusions, and require a fresh isolated run.
- **Audit gap:** record what cannot be reconstructed without claiming a defect
  that has not been demonstrated.
- **Corrected interpretation:** amend the owning claim, state what changed, and
  preserve the earlier wording and provenance.
- **Superseded valid claim:** retain both claims and link `supersedes` and
  `superseded by`; update current-answer surfaces only when the decision changes.

Invalid or incomplete attempts receive a theme claim only when the failure
changes scientific or operational understanding. Otherwise they remain in the
evidence-of-record trail without becoming a current conclusion card.

## Cross-theme example

The Qwen3-Omni neutral A2 work demonstrates the intended model:

- one immutable quality-10 evidence bundle;
- a Models claim about `19/20` strict completion and promotion ineligibility;
- a Native Audio claim about the selected `3C/3P/4W` to `5C/2P/3W` semantic
  signal; and
- direct reciprocal links instead of duplicated full experiment records.

The theme whose decision changes owns each claim. The modalities or models that
happened to execute do not determine ownership by themselves.

## Information architecture changes

### Adding Experiments and Results

- Add the three-concept model and a short result lifecycle.
- Add the theme creation, split, and merge decision rule.
- Add an invalid/incomplete/corrected/superseded routing table.
- Add the Qwen3-Omni cross-theme example.
- Extend the completed-result and agent-handoff fields with canonical claim
  anchor, shared evidence bundle, evidence validity, decision disposition,
  decision rule and whether it was predeclared, gate outcome, completion
  denominator, and supersession links.

### Research Themes hub

- Rename the visible map while preserving its legacy fragment.
- Add a prominent contributor-guide entry near the introduction.
- Link each card's strongest evidence directly to the relevant claim anchor.
- Add a collapsed static experiment index grouped by theme.
- Present evidence validity and decision disposition as separate tables.

### Start Here

- Keep the newcomer explanation compact.
- Make the evidence-validity vocabulary consistent with the hub and point to
  the separate decision-disposition vocabulary.

### Existing claim records

- Complete **Automatic transcript ablation** and **MiniCPM-o 4.5 fast
  alternate** with the four missing condition-bound fields: what the result
  rules out, what it does not rule out, defect/gap/uncertainty, and reconsider
  condition.
- Do not force a wholesale migration of already complete historical records.

## Accessibility and display

- Reuse existing headings, callouts, tables, links, and collapsed disclosures.
- Keep the experiment index collapsed by default so the hub remains scannable.
- Provide all lifecycle and status information in text and tables; no decorative
  diagram or image is required.
- Preserve semantic breadcrumbs, authored fragments, visible link purpose,
  contained table overflow, light/dark modes, and 390 px reflow.

## Verification

- Render the five changed QMD files to sibling GitHub-readable Markdown.
- Render all 12 existing QMD pages to self-contained HTML; do not add a page.
- Compare a clean isolated regeneration with repository outputs.
- Check repository-relative links and authored fragments, including the legacy
  theme-map fragment.
- Inspect Start Here, Research Themes, Adding Experiments, and the two amended
  theme pages in light/dark desktop and 390 px layouts.
- Confirm the static index is keyboard-readable and collapsed by default.
- Run `git diff --check` and a focused private-data/remote-runtime scan.
- Do not run the application test suite or add custom tests, schemas, scripts,
  validators, or CI.
- Refresh only the sanitized HTML files through the existing tailnet-only
  portal after independent review.
