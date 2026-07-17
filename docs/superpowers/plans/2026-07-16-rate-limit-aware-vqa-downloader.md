# Rate-Limit-Aware VQA Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume original-quality VQA downloads with per-URL pacing and automatic one-/two-hour recovery from YouTube guest-session rate limits.

**Architecture:** Keep detached orchestration in `vqa_download.py`, but replace whole-manifest yt-dlp calls with sequential single-URL calls. Stream each child process's combined output into the existing worker log, classify rate-limit output, and let the Python worker own ordinary pacing and cooldown state.

**Tech Stack:** Python 3.12, yt-dlp 2026.7.4 with EJS/Node, pytest, uv, FFmpeg.

## Global Constraints

- Preserve original `bestvideo*+bestaudio/best` selection and MKV stream-copy merging; never transcode or downscale.
- Preserve manifests, yt-dlp archives, partial-file continuation, metadata, detached execution, and training-before-testing ordering.
- Process one URL at a time with 1.5 seconds between extraction requests and 10 seconds after each ordinary URL attempt.
- On the documented rate-limit message, pause one hour; use two hours after consecutive rate limits without an archived success.
- Do not use cookies, accounts, proxies, parallel jobs, or publish any data.
- Leave the pre-existing untracked `notes.txt` untouched.

---

### Task 1: Single-URL command and result classification

**Files:**
- Modify: `src/trec_video_2026/vqa_download.py`
- Test: `tests/test_vqa_download.py`

**Interfaces:**
- Produces: `AttemptResult(completed: bool, rate_limited: bool)`.
- Produces: `read_archive_ids(archive: Path) -> set[str]`.
- Produces: `run_url(command: list[str], popen: Callable[..., object] = subprocess.Popen) -> AttemptResult`.
- Changes: `build_ytdlp_command(url: str, output_dir: Path, archive: Path) -> list[str]` accepts one URL, uses `--sleep-requests 1.5`, and contains no media sleep options.

- [ ] **Step 1: Write failing tests for archive parsing and command pacing**

```python
def test_read_archive_ids_accepts_ytdlp_archive_rows(tmp_path: Path):
    archive = tmp_path / "archive.txt"
    archive.write_text("youtube abcDEF123_0\nyoutube xyzXYZ987-1\n", encoding="utf-8")
    assert read_archive_ids(archive) == {"abcDEF123_0", "xyzXYZ987-1"}


def test_ytdlp_command_paces_requests_without_per_stream_sleep(tmp_path, monkeypatch):
    monkeypatch.setenv("VQA_NODE_PATH", "/opt/node/bin/node")
    command = build_ytdlp_command(
        "https://www.youtube.com/watch?v=abcDEF123_0",
        tmp_path / "videos",
        tmp_path / "archive.txt",
    )
    assert command[command.index("--sleep-requests") + 1] == "1.5"
    assert "--sleep-interval" not in command
    assert "--max-sleep-interval" not in command
    assert "--batch-file" not in command
    assert command[-1] == "https://www.youtube.com/watch?v=abcDEF123_0"
```

- [ ] **Step 2: Run the focused tests and verify they fail for missing archive parsing and the old batch/sleep command**

Run: `uv run pytest tests/test_vqa_download.py -k 'archive_ids or paces_requests' -v`

Expected: FAIL because `read_archive_ids` does not exist and the command still takes a manifest with 20–40 second sleeps.

- [ ] **Step 3: Implement archive parsing and the single-URL command**

```python
def read_archive_ids(archive: Path) -> set[str]:
    if not archive.exists():
        return set()
    return {
        fields[-1]
        for line in archive.read_text(encoding="utf-8").splitlines()
        if (fields := line.split()) and VIDEO_ID.fullmatch(fields[-1])
    }
```

Update `build_ytdlp_command` to remove `--batch-file`, `--sleep-interval`, and `--max-sleep-interval`; set `--sleep-requests` to `1.5`; append the URL as the last argument. Retain the existing format, archive, merge, metadata, continuation, retry, JavaScript runtime, and error options.

- [ ] **Step 4: Write and verify a failing output-classification test**

```python
def test_run_url_streams_output_and_detects_rate_limit(capsys):
    class Child:
        stdout = iter(["progress\n", f"ERROR: {RATE_LIMIT_MARKER}\n"])
        def wait(self):
            return 1

    result = run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())
    assert result == AttemptResult(completed=False, rate_limited=True)
    assert "progress" in capsys.readouterr().out
```

Run: `uv run pytest tests/test_vqa_download.py::test_run_url_streams_output_and_detects_rate_limit -v`

Expected: FAIL because `run_url`, `AttemptResult`, and `RATE_LIMIT_MARKER` do not exist.

- [ ] **Step 5: Implement result classification and run focused tests**

Implement a frozen `AttemptResult` dataclass. `run_url` must call `Popen(command, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`, print every output line with flushing, detect the rate-limit marker, wait for the child, and return the classification. Treat return code zero as completed only when no rate-limit marker was seen; archive state remains the final authority in Task 2.

Run: `uv run pytest tests/test_vqa_download.py -k 'archive_ids or paces_requests or run_url' -v`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/trec_video_2026/vqa_download.py tests/test_vqa_download.py
git commit -m "refactor: process VQA videos individually"
```

### Task 2: Per-URL pacing and automatic cooldown

**Files:**
- Modify: `src/trec_video_2026/vqa_download.py`
- Modify: `tests/test_vqa_download.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `read_archive_ids`, `run_url`, `AttemptResult`, and single-URL `build_ytdlp_command` from Task 1.
- Produces: `run_dataset(manifest: Path, output_dir: Path, archive: Path, *, url_runner: Callable[[list[str]], AttemptResult] = run_url, sleeper: Callable[[float], None] = time.sleep, ordinary_delay: float = 10.0, initial_cooldown: float = 3600.0, maximum_cooldown: float = 7200.0) -> None`.
- Changes: `run_downloads` invokes `run_dataset` for training and testing in order.

- [ ] **Step 1: Write failing tests for ordinary pacing and cooldown escalation/reset**

```python
def test_run_dataset_paces_every_ordinary_url(tmp_path: Path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "https://www.youtube.com/watch?v=abcDEF123_0\n"
        "https://www.youtube.com/watch?v=xyzXYZ987-1\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive.txt"
    calls, sleeps = [], []
    run_dataset(
        manifest, tmp_path / "videos", archive,
        url_runner=lambda command: calls.append(command) or AttemptResult(False, False),
        sleeper=sleeps.append,
    )
    assert len(calls) == 2
    assert sleeps == [10.0, 10.0]


def test_run_dataset_cools_down_and_resumes_from_archive(tmp_path: Path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text("https://www.youtube.com/watch?v=abcDEF123_0\n", encoding="utf-8")
    archive = tmp_path / "archive.txt"
    outcomes = iter([
        AttemptResult(False, True),
        AttemptResult(False, True),
        AttemptResult(True, False),
    ])
    sleeps = []
    run_dataset(
        manifest, tmp_path / "videos", archive,
        url_runner=lambda command: next(outcomes),
        sleeper=sleeps.append,
    )
    assert sleeps[:2] == [3600.0, 7200.0]
```

Use the fake runner to append `youtube abcDEF123_0` to the archive before returning the final successful result, proving the next pass observes archive state and terminates.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `uv run pytest tests/test_vqa_download.py -k 'run_dataset' -v`

Expected: FAIL because `run_dataset` does not exist.

- [ ] **Step 3: Implement the dataset loop**

Read URLs from the manifest, skip IDs already returned by `read_archive_ids`, and call `run_url` sequentially. After ordinary results, sleep 10 seconds. On a rate-limit result, log the cooldown and UTC/local retry timestamp, sleep for the current cooldown, double it up to 7200 seconds, then restart the manifest pass. If the attempted ID appears in the archive after an ordinary result, reset the next cooldown to 3600 seconds. Terminate after a full pass without rate limiting.

- [ ] **Step 4: Update orchestration and README**

Change `run_downloads` to call `run_dataset` for training and testing in order. Update the README downloader section to say that every attempted URL is paced by 10 seconds, extraction requests by 1.5 seconds, separate streams no longer receive duplicate long sleeps, and rate limits trigger automatic one-/two-hour resumable cooldowns.

- [ ] **Step 5: Run all tests and static checks**

Run: `uv run pytest -v`

Expected: all tests PASS.

Run: `git diff --check && uv run vqa-download status`

Expected: no whitespace errors; status remains stopped until the verified implementation is committed and explicitly restarted.

- [ ] **Step 6: Commit Task 2**

```bash
git add README.md src/trec_video_2026/vqa_download.py tests/test_vqa_download.py
git commit -m "feat: recover VQA downloads from rate limits"
```

### Task 3: Live resume verification

**Files:**
- Runtime only: `data/vqa/download_runtime/download.log`
- Runtime only: `data/vqa/download_runtime/download.pid`

**Interfaces:**
- Consumes: `uv run vqa-download start` and `uv run vqa-download status`.
- Produces: a detached worker using ignored runtime state; no tracked source changes.

- [ ] **Step 1: Perform cheap preflight checks**

Run: `git status --short && git check-ignore -v data/vqa && command -v node && command -v ffmpeg && df -h .`

Expected: only pre-existing `notes.txt` is untracked; VQA data is ignored; Node and FFmpeg exist; sufficient disk remains.

- [ ] **Step 2: Start the resumable worker**

Run: `uv run vqa-download start && uv run vqa-download status`

Expected: a new detached PID is running with unchanged archive/media counts at launch.

- [ ] **Step 3: Verify paced live behavior**

Inspect `tail -n 80 data/vqa/download_runtime/download.log` and status after at least one attempt.

Expected: the worker processes one URL, the command uses Node challenge solving and original video+audio selection when downloadable, attempts do not race through the manifest, and rate-limit detection logs a one-hour cooldown instead of continuing.

- [ ] **Step 4: Install and verify the completion watcher**

Start a detached, process-owned watcher that waits for the new downloader PID and sends the existing local desktop completion notification with final archive counts. Record its PID under ignored runtime state and verify its parent is PID 1.

- [ ] **Step 5: Final verification**

Run: `uv run pytest -v && git diff --check && git status --short && uv run vqa-download status && ps -o pid,ppid,etime,stat,cmd -p <downloader-pid>,<watcher-pid>`

Expected: all tests pass; no tracked changes remain; only `notes.txt` is untracked; downloader and watcher are detached and running.
