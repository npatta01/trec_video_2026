# Generic YouTube Dataset Engine and Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one generic YouTube dataset download engine used by both a generic CLI and the existing TRECVID wrapper, plus a repository-local skill that operates the generic CLI.

**Architecture:** Move all reusable API, proxy, yt-dlp, archive, resume, pacing, cooldown, detached-worker, and status behavior into `youtube_dataset.py`. Keep only TRECVID source parsing, default paths, and adapter CLI behavior in `vqa_download.py`; expose the generic engine through `youtube-dataset-download` and teach the repository-local skill to use that command.

**Tech Stack:** Python 3.12 standard library, yt-dlp, pytest, Codex repository skills.

## Global Constraints

- There is exactly one implementation of download behavior.
- Existing VQA data, archives, partial files, runtime paths, proxy/API files, command names, and resume semantics remain compatible.
- Credentials never appear in process arguments, logs, exceptions, tracked files, or skill examples.
- API checks fail closed, yt-dlp runs one video at a time, and repeated proxy transport failures stop and report promptly.
- Preserve unrelated `dstack/` and `notes.txt` files.

---

### Task 1: Characterization and baseline skill evaluation

**Files:**
- Modify: `tests/test_vqa_download.py`
- Create: `tests/test_youtube_dataset.py`
- Runtime only: `.superpowers/sdd/youtube-skill-baseline.md`

**Interfaces:**
- Captures the current public behavior before moving code.
- Records baseline-agent omissions for the skill test-first cycle.

- [ ] Run a fresh-agent baseline scenario without the skill and record whether it checks API availability, keeps secrets out of argv/logs, verifies proxy health, limits concurrency, detects 429/bot checks, stops on repeated 502s, and reports terminal failures.
- [ ] Add failing ownership tests asserting generic functions come from `trec_video_2026.youtube_dataset`, while TRECVID parsing/default construction remains in `vqa_download`.
- [ ] Run focused tests and confirm failure because the generic module does not exist.

### Task 2: Extract the single generic engine

**Files:**
- Create: `src/trec_video_2026/youtube_dataset.py`
- Modify: `src/trec_video_2026/vqa_download.py`
- Modify: `tests/test_youtube_dataset.py`
- Modify: `tests/test_vqa_download.py`

**Interfaces:**
- Produces: `DatasetSpec`, `DownloadConfig`, `AttemptResult` dataclasses.
- Produces generic credential/API/proxy helpers, `build_ytdlp_command`, `run_url`, `run_dataset`, detached-worker helpers, and status helpers.
- `vqa_download` imports and delegates to these interfaces while retaining `read_video_ids`, `read_unavailable_ids`, `write_manifest`, `prepare_manifests`, and TRECVID defaults.

- [ ] Move each generic test to `tests/test_youtube_dataset.py` before moving its implementation; run it to confirm the missing generic import fails.
- [ ] Move the minimum corresponding implementation into `youtube_dataset.py`, re-export compatibility names from `vqa_download`, and run the focused tests green after each group: API/cache, proxy, yt-dlp/archive, cooldown/failure classification, detached execution/status.
- [ ] Add a failing test for repeated proxy 502 classification and prompt stop; implement a bounded consecutive-proxy-failure threshold without treating ordinary video errors as proxy failures.
- [ ] Run both test modules and confirm all existing VQA behavior remains green.

### Task 3: Add the generic CLI and thin VQA adapter

**Files:**
- Modify: `src/trec_video_2026/youtube_dataset.py`
- Modify: `src/trec_video_2026/vqa_download.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_youtube_dataset.py`
- Modify: `tests/test_vqa_download.py`

**Interfaces:**
- Produces entry point: `youtube-dataset-download = trec_video_2026.youtube_dataset:main`.
- Generic commands: `prepare`, `run`, `start`, `status` with explicit `--source`, `--output-dir`, `--state-dir`, `--api-env`, and `--proxy-env` configuration.
- `vqa-download` constructs training/testing `DatasetSpec` values and invokes the same engine.

- [ ] Write failing CLI tests for ID/URL normalization, required explicit paths, prepare snapshots, detached start/PID state, status, Git-tracked-path refusal, and fail-closed API/proxy preflight.
- [ ] Implement the generic parser/config adapter without duplicating engine logic.
- [ ] Write and pass equivalence tests proving both CLIs call the same engine interfaces and the existing VQA runtime paths remain unchanged.
- [ ] Run `uv run youtube-dataset-download --help`, `uv run vqa-download status`, and both test modules.

### Task 4: Create and forward-test the repository skill

**Files:**
- Create: `.agents/skills/downloading-youtube-datasets/SKILL.md`
- Create: `.agents/skills/downloading-youtube-datasets/agents/openai.yaml`

**Interfaces:**
- The skill always invokes `youtube-dataset-download`; it contains no downloader implementation.

- [ ] Initialize the skill with `skill-creator/scripts/init_skill.py` under `.agents/skills` and generate interface metadata.
- [ ] Write a concise `SKILL.md` addressing baseline omissions: private credentials, Git-ignore checks, API-first preparation, proxy preflight, one-at-a-time execution, archive/partial resume, 429 cooldown, repeated-502 stop/notify, status monitoring, and categorized final reporting.
- [ ] Run `quick_validate.py` and verify `agents/openai.yaml` matches the skill.
- [ ] Run a fresh-agent forward scenario with only the skill and record whether it uses the generic CLI and obeys every safety gate; revise and rerun if it finds a loophole.

### Task 5: Capture final dataset outcome and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-17-generic-youtube-dataset-skill.md`

- [ ] Record the final 1,883 testing-media result and the three API-visible inaccessible IDs with their exact categories.
- [ ] Run `uv run pytest`, `git diff --check`, both CLI help/status smoke tests, skill validation, and a tracked-diff secret scan.
- [ ] Confirm existing ignored media/archive state is unchanged and no downloader starts during verification.
- [ ] Commit only intended tracked files, leaving `dstack/` and `notes.txt` untouched.
