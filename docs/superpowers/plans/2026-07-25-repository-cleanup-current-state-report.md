# Repository Cleanup and Current-State Report

**Objective:** Make the repository understandable to a newcomer without
reorganizing or rerunning the experiment system.

**Status:** In progress on `codex/repository-cleanup-report`, based on
`origin/main@814e554`.

## Scope and decisions

- Preserve all pre-cleanup loose files in the named Git stash
  `pre-cleanup backup 2026-07-25` before changing the checkout.
- Restore and reconcile the evidence-routing plan linked by closed issue #4.
- Keep historical files at their current paths; add indexes and narrow status
  banners instead of moving them.
- Create a three-page Quarto mini-site under `docs/project-status/`.
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

## Task 2: Three-page Quarto report

- Create **Start Here**, **Experiment Ledger**, and **Code Map and Remaining
  Gaps** pages.
- Include five accessible visuals: system pipeline, experiment timeline,
  decision map, recommended router, and repository map.
- Give every visual meaningful alternative text, a caption, and a nearby
  list/table equivalent.
- Link claims to tracked reports, issues, PRs, and sanitized tracked artifacts.
- Record the two-row dense-burst aggregate without exposing row identifiers,
  answers, or raw artifacts.

## Task 3: Minimal verification and private delivery

- Render Markdown and HTML with Quarto.
- Check navigation, heading order, readability, and page overflow at desktop
  and 390-pixel widths.
- Check local links, figure accessibility metadata, `git diff --check`, and
  generated-output privacy.
- Publish only the sanitized derived HTML through the existing tailnet-only
  portal; do not use Funnel or a public listener.

## Verification evidence

- Pending.

## Next action

Complete Task 1, then render the report before doing any portal copy.
