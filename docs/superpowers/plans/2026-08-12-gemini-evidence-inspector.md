# Gemini Evidence Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one private static page that makes the seven Gemini evidence cases easy to compare and dissect.

**Architecture:** A Python generator validates and projects the frozen private artifacts, then renders one dependency-free HTML page. The page uses relative URLs into a tailnet-served symlink to the existing private bundle, so no media is copied or committed.

**Tech Stack:** Python standard library, HTML/CSS/vanilla JavaScript, pytest, Tailscale Serve.

## Global Constraints

- One read-only static page; no backend, framework, persistence, or model calls.
- Never stage, commit, embed, or publish private video, frames, questions, references, or provider responses.
- Keep canonical evidence under the existing shared dataset path.
- Render only under `$HOME/codex-rendered/plans/` and expose only through the existing tailnet-only portal.

---

### Task 1: Artifact projection and validation

**Files:**
- Create: `scripts/build_gemini_evidence_inspector.py`
- Create: `tests/test_build_gemini_evidence_inspector.py`

**Interfaces:**
- Consumes: private run root containing `postfreeze-metrics.json` and seven `rows/*` bundles.
- Produces: `load_cases(run_root: Path, media_prefix: str) -> list[dict]` with video, question, reference, assessment, answers, timeline, selected spans, and frame URLs.

- [x] Write failing tests requiring exactly seven rows, all verdicts, safe relative media URLs, and rejection of missing artifacts.
- [x] Run `uv run pytest tests/test_build_gemini_evidence_inspector.py -q` and verify failure because the generator does not exist.
- [x] Implement the minimal validated projection, reusing the frozen manual assessments from `score_direct_gemini_evidence_canary.py`.
- [x] Run the focused tests and Ruff; require clean output.
- [x] Commit the validated projection.

### Task 2: Static inspector and private delivery

**Files:**
- Modify: `scripts/build_gemini_evidence_inspector.py`
- Modify: `tests/test_build_gemini_evidence_inspector.py`
- Generate privately: `/home/npatta01/codex-rendered/plans/trec-video-gemini-evidence-inspector.html`
- Modify privately: `/home/npatta01/codex-rendered/index.html`
- Create privately: `/home/npatta01/codex-rendered/private/trec-video-gemini-evidence-canary` symlink

**Interfaces:**
- Consumes: `load_cases(...)` output.
- Produces: `render_html(cases: list[dict]) -> str` and one executable CLI accepting `--run-root`, `--media-prefix`, and `--output`.

- [x] Add failing tests for seven video controls, answer/verdict cards, video seeking hooks, timeline buttons, frame buttons, responsive CSS, and escaped artifact text.
- [x] Run the focused test and verify expected failures.
- [x] Implement the minimal single-page HTML/CSS/JavaScript renderer and CLI.
- [x] Render the private page and create only the narrow symlink to the existing seven-row bundle.
- [x] Add one portal-index link without altering unrelated entries.
- [x] Verify focused tests, Ruff, `git diff --check`, seven case IDs in generated HTML, HTTP 200 for page/video/frame URLs, and `tailscale serve status` showing the existing tailnet HTTPS mapping.
- [ ] Commit and push only tracked generator/test/plan changes; keep generated page, portal index, symlink, and private media untracked.
