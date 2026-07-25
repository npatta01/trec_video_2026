# Repository Cleanup and Current-State Report

**Objective:** Make the repository understandable to a newcomer without
reorganizing or rerunning the experiment system.

**Status:** Complete on `codex/repository-cleanup-report`, based on
`origin/main@814e554`. GitHub issue creation remains intentionally deferred
until the report is reviewed.

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
- Include five accessible visuals: system pipeline, research-question map,
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

- Before the fast-forward, all four loose files were preserved in named stash
  `pre-cleanup backup 2026-07-25` (`399280bfdd4f521bc4d220276f3d6cc42f701de7`).
  The checkout then fast-forwarded to `origin/main@814e554`; only the
  evidence-routing plan was restored. Detached worktrees were not changed.
- Quarto 1.9.38 rendered all three QMD sources to sibling GitHub-Flavored
  Markdown and to three self-contained HTML pages under
  `reports/project-status/`. The final HTML render processed exactly three QMD
  files.
- Headless Chrome loaded all three final pages at 1440 and 390 CSS pixels. Each
  page had one H1, working local navigation and landmarks, no page-wide
  horizontal overflow, no forced-new-window links, and no external runtime
  resources. The five Mermaid SVGs each had a visible caption plus a native
  accessible title and description; each also has an adjacent list or table
  equivalent in the QMD/HTML and a visible caption in generated Markdown.
- Six representative desktop/phone screenshots were inspected for readable
  text, responsive navigation, tables, captions, and diagrams. Long inline
  code wraps without creating page-wide overflow; tables retain native table
  semantics, wrap long cell content, and fit within the phone viewport.
- A bounded link check covered 21 changed Markdown/QMD documents and 90
  repository-relative links, with zero missing files or anchors.
- A bounded privacy scan covered the three QMD sources, three generated
  Markdown files, and three HTML pages, with zero findings for local home
  paths, private bucket URIs, credentials, YouTube watch links, structured raw
  records, or media/log/data links.
- `git diff --check` passes. A narrow `.gitattributes` rule marks only the
  self-contained HTML outputs as generated and excludes Quarto's embedded
  vendor-code whitespace from that diagnostic. No `data/` path is staged.
- The published directory contains exactly the three sanitized HTML files.
  Their hashes match the repository outputs. The live start page returned HTTP
  200 and matched the repository SHA-256 exactly. `tailscale funnel status`
  reported the portal as `tailnet only`; no Funnel/public listener is enabled.
- No application tests, custom test suite, CI, schema validator, dataset read,
  or GitHub issue mutation was performed for this documentation-only change.

## Next action

Review the current-state report. After that review, convert the documentation,
engineering, and research gaps into scoped GitHub issues without reopening
superseded experiment directions.
