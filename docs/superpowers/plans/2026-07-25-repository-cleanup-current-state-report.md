# Repository Cleanup and Current-State Report

**Objective:** Make the repository understandable to a newcomer without
reorganizing or rerunning the experiment system.

**Status:** Complete on `codex/repository-cleanup-report`, based on
`origin/main@814e554`. The report has been reviewed in the private portal;
GitHub issue creation remains intentionally deferred until the user chooses
how to integrate this branch.

## Supersession note

This plan originally specified a three-page mini-site. That version was built
and reviewed, but the long Experiment Ledger was still overwhelming and made
the reasons for experiments hard to follow. The subsequently approved
[research-theme redesign](2026-07-25-research-theme-status-site-redesign.md)
superseded only the report structure: the site still has the original three
top-level navigation choices, while Research Themes now links to six focused
detail pages. The final deliverable is therefore nine pages, not three.

## Scope and decisions

- Preserve all pre-cleanup loose files in the named Git stash
  `pre-cleanup backup 2026-07-25` before changing the checkout.
- Restore and reconcile the evidence-routing plan linked by closed issue #4.
- Keep historical files at their current paths; add indexes and narrow status
  banners instead of moving them.
- Original delivery target: create a three-page Quarto mini-site under
  `docs/project-status/`. The later approved redesign expanded this to nine
  pages while preserving **Start Here**, **Research Themes**, and **Code Map
  and Gaps** as the three top-level choices.
- Treat QMD as the editable report source. Generate GitHub-readable Markdown
  beside each QMD and self-contained HTML under `reports/project-status/`.
- Use sanitized diagrams and aggregate findings only. Do not include media,
  frames, transcripts, predictions, labels, raw logs, credentials, or private
  artifact URLs.
- Defer GitHub issue creation until this report has been reviewed.
- Do not add a test framework, CI, a schema validator, or application tests for
  this documentation-only change.

## Task 1: Cleanup and historical-document clarification

- Shorten the root README into a landing page.
- Add small indexes for implementation plans and historical reports.
- Reconcile the recovered evidence-routing plan.
- Add status or supersession banners only where current wording is misleading.
- Preserve the other three loose files in the recovery stash only.

## Task 2: Original three-page Quarto report, later expanded

- Create **Start Here**, **Experiment Ledger**, and **Code Map and Remaining
  Gaps** pages. This original milestone was completed; the later redesign
  retained these paths and converted Experiment Ledger into the Research
  Themes hub with six linked theme pages.
- Include five accessible visuals: system pipeline, research-question map,
  decision map, recommended router, and repository map.
- Give every visual meaningful alternative text, a caption, and a nearby
  list/table equivalent.
- Link claims to tracked reports, issues, PRs, and sanitized tracked artifacts.
- Record the two-row dense-burst aggregate without exposing row identifiers,
  answers, or raw artifacts.

## Task 3: Minimal verification and private delivery

- Render the final nine QMD sources to nine GitHub-readable Markdown files and
  nine self-contained HTML pages with Quarto.
- Check navigation, heading order, readability, and page overflow at desktop
  and 390-pixel widths.
- Check local links, figure accessibility metadata, `git diff --check`, and
  generated-output privacy.
- Publish only the sanitized derived HTML through the existing tailnet-only
  portal; do not use Funnel or a public listener.

## Verification evidence

- Before the fast-forward, all four loose files were preserved in named stash
  `pre-cleanup backup 2026-07-25` (`399280bfdd4f521bc4d220276f3d6cc42f701de7`).
  The checkout then fast-forwarded to `origin/main@814e554`; only the
  evidence-routing plan was restored. Detached worktrees were not changed.
- Quarto 1.9.38 rendered nine canonical QMD sources to nine sibling
  GitHub-Flavored Markdown files and nine self-contained HTML pages under
  `reports/project-status/`. An isolated rerender matched all 18 generated
  files byte-for-byte (`18/18`).
- Headless Chrome checked all nine HTML pages at 1440 and 390 CSS pixels. The
  final site passed the heading, navigation, keyboard-disclosure, responsive
  layout, diagram containment, accessibility, and page-overflow gates.
- Repository and isolated-portal link checks passed, including legacy ledger
  fragments and the canonical GitHub evidence links used by the private copy.
  Visuals retain meaningful alternative text, captions, accessible SVG
  titles/descriptions, and nearby prose or table equivalents.
- Privacy and external-resource scans across the nine QMD, nine Markdown, and
  nine HTML files found no private data, local paths, credentials, bucket URIs,
  media/log/data links, or unintended runtime resources.
- `git diff --check` passes. A narrow `.gitattributes` rule marks only the
  self-contained HTML outputs as generated and excludes Quarto's embedded
  vendor-code whitespace from that diagnostic. No `data/` path is staged.
- The published directory contains exactly the nine sanitized HTML files.
  All nine live pages returned HTTPS 200 and matched their repository outputs
  byte-for-byte (`9/9`). `tailscale funnel status` reported the portal as
  `tailnet only`; no Funnel/public listener is enabled.
- No application tests, custom test suite, CI, schema validator, dataset read,
  or GitHub issue mutation was performed for this documentation-only change.

## Next action

The report review is complete. The user now chooses whether to merge the
branch, open a pull request, or keep it as-is. GitHub issue creation remains
deferred; after integration is chosen, the documented engineering and research
gaps can be converted into scoped issues without treating condition-bound
negative results as permanent disqualifications.
