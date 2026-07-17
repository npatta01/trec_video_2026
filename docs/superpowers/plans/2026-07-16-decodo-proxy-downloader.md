# Decodo Proxy Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Decodo can download a public YouTube video and one previously bot-blocked VQA item, then integrate a credential-safe proxy config into the resumable downloader without starting the full batch.

**Architecture:** A one-time gate creates an ignored mode-`0600` yt-dlp config from the protected Decodo environment file and uses it for two live proofs. Only after both proofs pass, focused Python helpers reproduce the same secure config generation for detached production runs, while yt-dlp command lines contain only the config path and never proxy credentials.

**Tech Stack:** Python 3.12, Bash, yt-dlp 2026.7.4 with EJS/Node, FFmpeg/ffprobe, pytest, uv.

## Global Constraints

- Read credentials only from `/home/npatta01/.config/trec-video/decodo.env`, owned by the current user with mode `0600`.
- Require `DECODO_ENDPOINT`, `DECODO_USER`, and `DECODO_PASSWORD`; never print, log, commit, or place their values in process arguments.
- URL-encode username and password and store `--proxy` only in `data/vqa/download_runtime/yt-dlp-proxy.conf`, atomically written with mode `0600`.
- Public proof `BaW_jenozKc` must pass before attempting VQA proof `emArU432qfE`; both proofs must pass before modifying production downloader source.
- Preserve `bestvideo*+bestaudio/best`, MKV stream-copy, Node/EJS, archives, partials, sequential processing, pacing, retries, and cooldown behavior.
- Do not start the full VQA batch in this plan.
- Do not commit credentials, private config, runtime state, logs, media, archives, `notes.txt`, or the unrelated `dstack/` working directory.

---

### Task 1: Secure public-video proxy proof

**Files:**
- Read: `/home/npatta01/.config/trec-video/decodo.env`
- Runtime create: `data/vqa/download_runtime/yt-dlp-proxy.conf`
- Runtime create/remove: `data/vqa/proxy_smoke/`
- Source changes: none

**Interfaces:**
- Produces: a mode-`0600` yt-dlp config whose contents are never displayed.
- Produces: verified public-video evidence or a sanitized blocking diagnostic.

- [ ] **Step 1: Validate credential metadata without reading values to output**

Run:

```bash
stat -c 'mode=%a owner_uid=%u' /home/npatta01/.config/trec-video/decodo.env
bash -c 'set -a; source "$1"; for n in DECODO_ENDPOINT DECODO_USER DECODO_PASSWORD; do test -n "${!n:-}" || exit 1; done' _ /home/npatta01/.config/trec-video/decodo.env
```

Expected: mode `600`, owner UID equal to `id -u`, and exit 0 with no credential output.

- [ ] **Step 2: Generate the private yt-dlp config atomically**

Run a non-logging shell that sources the environment and passes values only through inherited environment variables to this Python body:

```python
import os
from pathlib import Path
from urllib.parse import quote

target = Path("data/vqa/download_runtime/yt-dlp-proxy.conf")
temporary = target.with_suffix(".conf.tmp")
endpoint = os.environ["DECODO_ENDPOINT"]
username = quote(os.environ["DECODO_USER"], safe="")
password = quote(os.environ["DECODO_PASSWORD"], safe="")
proxy_url = f"http://{username}:{password}@{endpoint}"
target.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write(f"--proxy\n{proxy_url}\n")
os.chmod(temporary, 0o600)
os.replace(temporary, target)
os.chmod(target, 0o600)
```

After the process exits, unset the three `DECODO_*` variables. Verify only metadata:

```bash
stat -c 'mode=%a owner_uid=%u size=%s' data/vqa/download_runtime/yt-dlp-proxy.conf
```

Expected: mode `600`, current user ownership, nonzero size; never print the file.

- [ ] **Step 3: Download the public smoke video through the config**

Run:

```bash
rm -rf data/vqa/proxy_smoke
mkdir -p data/vqa/proxy_smoke/media
uv run yt-dlp \
  --config-locations data/vqa/download_runtime/yt-dlp-proxy.conf \
  --format 'bestvideo*+bestaudio/best' \
  --js-runtimes "node:$(command -v node)" \
  --merge-output-format mkv \
  --output 'data/vqa/proxy_smoke/media/%(id)s.%(ext)s' \
  --write-info-json --continue --no-overwrites \
  --download-archive data/vqa/proxy_smoke/archive.txt \
  --no-playlist --newline \
  'https://www.youtube.com/watch?v=BaW_jenozKc'
```

Expected: exit 0, archive row `youtube BaW_jenozKc`, and one finalized ID-named media file. Command/log output must not contain the endpoint, username, password, or a URL containing `@`.

- [ ] **Step 4: Verify streams and clean public smoke media**

Run `ffprobe -v error -show_entries stream=codec_type -of csv=p=0 <media>` and require exact `video` and `audio` output rows. Record only ID, container, size, codec types, and pass/fail. Remove `data/vqa/proxy_smoke/` after verification, retaining the private shared config for Task 2.

- [ ] **Step 5: Gate decision**

If any Task 1 step fails, stop the plan, preserve sanitized diagnostics, and do not execute Tasks 2 or 3. If all pass, record the proof in the progress ledger and continue.

### Task 2: Canonical VQA proxy proof

**Files:**
- Runtime modify: `data/vqa/2025/testing/videos_original/`
- Runtime modify: `data/vqa/download_runtime/testing.archive.txt`
- Read: `data/vqa/download_runtime/yt-dlp-proxy.conf`
- Source changes: none

**Interfaces:**
- Consumes: secure proxy config and public proof from Task 1.
- Produces: finalized `emArU432qfE` media/metadata and an authoritative testing archive row.

- [ ] **Step 1: Prove the target is currently unarchived**

Run:

```bash
test -z "$(rg '^youtube emArU432qfE$' data/vqa/download_runtime/testing.archive.txt || true)"
```

Expected: exit 0 and no output.

- [ ] **Step 2: Attempt the VQA video through the private config**

Run:

```bash
uv run yt-dlp \
  --config-locations data/vqa/download_runtime/yt-dlp-proxy.conf \
  --format 'bestvideo*+bestaudio/best' \
  --js-runtimes "node:$(command -v node)" \
  --sleep-requests 1.5 \
  --merge-output-format mkv \
  --output 'data/vqa/2025/testing/videos_original/%(id)s.%(ext)s' \
  --write-info-json --continue --no-overwrites \
  --download-archive data/vqa/download_runtime/testing.archive.txt \
  --retries 10 --fragment-retries 10 --extractor-retries 3 \
  --no-playlist --newline \
  'https://www.youtube.com/watch?v=emArU432qfE'
```

Expected: exit 0 without bot-check output, finalized media and metadata, and archive row `youtube emArU432qfE`.

- [ ] **Step 3: Verify canonical result**

Require exactly one finalized `emArU432qfE` media file, successful `ffprobe`, at least one video stream, at least one audio stream, and the exact archive row. Run `uv run vqa-download status` and confirm testing archive/media counts both increase by one.

- [ ] **Step 4: Gate decision**

If Task 2 fails, stop with sanitized evidence and do not modify production source. If it passes, record proof evidence and continue.

### Task 3: Production proxy integration

**Files:**
- Modify: `src/trec_video_2026/vqa_download.py`
- Modify: `tests/test_vqa_download.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `load_proxy_credentials(env_path: Path, runner: Callable[..., object] = subprocess.run) -> tuple[str, str, str]`.
- Produces: `write_proxy_config(env_path: Path, config_path: Path, runner: Callable[..., object] = subprocess.run) -> Path`.
- Changes: `build_ytdlp_command(url: str, output_dir: Path, archive: Path, proxy_config: Path | None = None) -> list[str]`.
- Changes: `start_detached(..., proxy_env_path: Path | None = None) -> int` prepares the secure config and passes only `VQA_PROXY_CONFIG_PATH` to the child.

- [ ] **Step 1: Write failing credential validation and encoding tests**

Add tests using a fake mode-`0600` environment file and injected runner. Cover URL encoding of reserved username/password characters, exact config contents with placeholder credentials, output mode `0600`, rejection of wrong permissions, rejection of symlinks, missing variables, and sanitized exceptions that contain no placeholder secret.

- [ ] **Step 2: Run focused tests to verify RED**

Run: `uv run pytest tests/test_vqa_download.py -k 'proxy_credentials or proxy_config' -v`

Expected: collection/import failure because the new helpers do not exist, or assertion failures for missing behavior.

- [ ] **Step 3: Implement secure credential loading and atomic config creation**

Use `lstat()` to reject symlinks, require `stat.S_IMODE(st_mode) == 0o600` and `st_uid == os.getuid()`, source with `bash -c` using the env path as positional data, capture three NUL-delimited values without printing them, validate a host/port endpoint without scheme/credentials/whitespace, URL-encode user/password with `quote(..., safe="")`, and atomically replace a mode-`0600` temporary file. Raise sanitized `RuntimeError` messages only.

- [ ] **Step 4: Write failing command and detached-start tests**

Assert `build_ytdlp_command(..., proxy_config=Path("/private/proxy.conf"))` contains `--config-locations /private/proxy.conf`, contains no `--proxy`, and contains no credential-shaped `@`. Assert `start_detached` prepares `data/vqa/download_runtime/yt-dlp-proxy.conf`, sets only its path in `VQA_PROXY_CONFIG_PATH`, and never places secrets in the child command or captured log output.

- [ ] **Step 5: Run focused tests to verify RED**

Run: `uv run pytest tests/test_vqa_download.py -k 'ytdlp_command or start_detached' -v`

Expected: FAIL because the command and detached interfaces do not yet accept proxy configuration.

- [ ] **Step 6: Implement command and detached integration**

`start_detached` calls `write_proxy_config` before opening the worker log, defaults the env path to `/home/npatta01/.config/trec-video/decodo.env`, and passes the generated config path through `VQA_PROXY_CONFIG_PATH`. `build_ytdlp_command` uses its explicit `proxy_config` argument or the environment path and inserts `--config-locations <path>` before the URL. No secret values cross either interface.

- [ ] **Step 7: Update README and run complete verification**

Document the protected environment location and variable names, private generated config, smoke/VQA proof results, sticky-session rule, and the fact that the full batch remains stopped. Run:

```bash
uv run pytest -v
uvx ruff check src/trec_video_2026/vqa_download.py tests/test_vqa_download.py
git diff --check
git status --short
uv run vqa-download status
```

Expected: all tests and checks pass; downloader remains stopped; only intended source/docs plus pre-existing `notes.txt` and `dstack/` appear.

- [ ] **Step 8: Commit production integration**

```bash
git add README.md src/trec_video_2026/vqa_download.py tests/test_vqa_download.py
git commit -m "feat: route VQA downloads through private proxy config"
```

### Task 4: Final review and secret audit

**Files:**
- Review only: all Task 3 tracked changes and runtime metadata (never config contents)

**Interfaces:**
- Consumes: Tasks 1–3 evidence and commits.
- Produces: reviewed, stopped production state ready for a separately authorized batch start.

- [ ] **Step 1: Generate a whole-feature review package and request independent code review**

Review against the design with emphasis on credential leakage, file permissions, command arguments, error sanitization, archive integrity, and preservation of downloader behavior. Fix all Critical/Important findings and re-review.

- [ ] **Step 2: Run a secret audit without printing matches**

Search tracked files and commit diff for the endpoint, username, password, proxy URL authority syntax, and `DECODO_PASSWORD` values using a checker that returns only pass/fail and filenames for variable-name documentation. Never print secret match text.

- [ ] **Step 3: Final verification**

Freshly run the full test suite, Ruff, diff check, Git status, config metadata check, archive/media/count verification, and stopped-process check. Confirm the proxy config is Git-ignored and mode `0600`.
