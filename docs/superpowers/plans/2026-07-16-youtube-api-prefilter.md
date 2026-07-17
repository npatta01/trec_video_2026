# YouTube API Prefilter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict yt-dlp to API-confirmed available, unarchived IDs and cool down immediately on YouTube bot checks.

**Architecture:** Add secure API-key loading and a batch API prefilter to the existing downloader module. Detached starts refresh atomic ignored manifests, while the worker consumes only those manifests and preserves all existing archive/proxy behavior.

**Tech Stack:** Python 3.12 standard library, pytest, YouTube Data API v3, existing yt-dlp integration.

## Global Constraints

- Never expose the API key or proxy credentials in arguments, logs, exceptions, or tracked files.
- Query at most 50 IDs per API request.
- Abort safely on API failure; never fall back to the unfiltered download manifests.
- Run only one yt-dlp process at a time.

---

### Task 1: Bot-check rate-limit classification

**Files:**
- Modify: `tests/test_vqa_download.py`
- Modify: `src/trec_video_2026/vqa_download.py`

**Interfaces:**
- Consumes: `run_url(command: list[str], popen=...) -> AttemptResult`
- Produces: bot-check output classified as `AttemptResult(False, True)`

- [ ] Write parameterized failing tests for straight and typographic apostrophes and retain negative tests for private/unavailable output.
- [ ] Run the focused tests and confirm failure because bot checks are currently ordinary errors.
- [ ] Add normalized bot-check text to the rate-limit marker predicate.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Secure YouTube API prefilter

**Files:**
- Modify: `tests/test_vqa_download.py`
- Modify: `src/trec_video_2026/vqa_download.py`

**Interfaces:**
- Produces: `load_youtube_api_key(path: Path) -> str`
- Produces: `fetch_available_video_ids(ids: list[str], api_key: str, opener=...) -> set[str]`
- Produces: `prepare_available_manifests(repo_root: Path, opener=...) -> dict[str, dict[str, int]]`

- [ ] Write failing tests for file ownership/mode, ambient-variable scrubbing, 50-ID batching, archive exclusion, atomic available/omitted inventories, and API errors.
- [ ] Run focused tests and confirm failures are caused by missing interfaces.
- [ ] Implement minimal secure loading, standard-library HTTPS requests, response validation, and atomic runtime outputs.
- [ ] Run focused tests and confirm they pass.

### Task 3: Start and worker integration

**Files:**
- Modify: `tests/test_vqa_download.py`
- Modify: `src/trec_video_2026/vqa_download.py`
- Modify: `README.md`

**Interfaces:**
- `start` refreshes API availability before detached spawn.
- `run_downloads` consumes `training.pending-available.urls.txt` and `testing.pending-available.urls.txt` only.

- [ ] Write failing integration tests proving start aborts on prefilter failure and workers never consume unfiltered manifests.
- [ ] Run focused tests and confirm the existing start/worker paths violate the new behavior.
- [ ] Integrate the prefilter, add status inventory counts, and document the private key file plus fail-closed behavior.
- [ ] Run focused tests and confirm they pass.

### Task 4: Verification and restart

**Files:**
- Runtime only: `data/vqa/download_runtime/`

- [ ] Run `uv run pytest` and require all tests to pass.
- [ ] Run `git diff --check` and inspect the tracked diff for secrets.
- [ ] Run the real API precheck and verify the pending/omitted partition against unarchived inputs.
- [ ] Commit the implementation and documentation intentionally, excluding unrelated `dstack/` and `notes.txt`.
- [ ] Start the detached downloader and verify one child whose ID exists in the pending-available manifest.

