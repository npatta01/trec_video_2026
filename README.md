# TRECVID 2026 Video Question Answering

Local tools and private, Git-ignored datasets for the TRECVID 2026 Video Question Answering track.

## Data inventory

Everything under `data/` is local-only. The large media and runtime files are intentionally excluded from Git by `.gitignore`.

### Shared data across Git worktrees

The canonical checkout owns the only physical dataset copy:
`/home/npatta01/data/competitions/trec_video_2026/data`. Linked Git worktrees
use an ignored, absolute `data` symlink to that directory, so repository tools
can use the same relative paths without duplicating approximately 25 GiB.

Configure every currently registered worktree from any checkout:

```bash
scripts/link-shared-data.sh --all
```

After creating one future worktree, run `scripts/link-shared-data.sh` inside it.
The helper is safe to repeat and refuses to replace any conflicting file,
directory, or incorrect symlink. It also adds `/data` to Git's shared local
exclude file, so links stay ignored even in older detached worktrees that do
not yet contain the tracked `.gitignore` rule.

| Path | Contents | Expected usable items |
| --- | --- | ---: |
| `data/vtt/` | NIST TRECVID Video-to-Text data from 2016–2024: videos, captions, URL metadata, and checksums | 10,152 local WebM videos; captions cover 16,627 IDs |
| `data/vqa/2025/training/` | Original NIST training ZIP/README, extracted annotations, and downloaded YouTube media | 481 usable videos/annotations from 500 rows after 19 NIST exclusions |
| `data/vqa/2025/testing/` | NIST 2025 test CSV/README and downloaded YouTube media | 1,982 usable videos from 2,000 rows after 18 NIST exclusions |
| `data/vqa/2026/testing/` | NIST 2026 AG README, topics JSON, retained/extracted video ZIP, and downloaded YouTube media | 1,946/1,956 unique test videos local; 10 YouTube IDs API-omitted |
| `data/vqa/download_runtime/` | 2025 generated manifests, archives, logs, and completion state | Runtime state; safe to retain for resume |
| `data/vqa/2026/download_runtime/` | 2026 generated manifest, API snapshot, archive, log, and completion state | Cleanly stopped; safe to retain for rechecking omitted IDs |

Important files:

- `data/vtt/vtt_ground_truth.txt`: VTT captions.
- `data/vtt/Video_Files/`: VTT WebM media.
- `data/vtt/checklist.chk`: checksums for all 10,152 VTT media files.
- `data/vqa/2025/training/annotations/train.2025.csv`: 500 original training rows.
- `data/vqa/2025/training/annotations/*.json`: extracted question/answer annotations; the working copy excludes 19 unavailable videos.
- `data/vqa/2025/testing/testing.dataset.vqa.2025.csv`: 2,000 prior test queries.
- `data/vqa/2025/*/videos_original/`: highest-quality YouTube video+audio streams, merged without transcoding.
- `data/vqa/2026/testing/vqa.2026.testingDataset.Task1.json`: 2,443 current AG queries covering 1,956 unique videos.
- `data/vqa/2026/testing/videos/`: all 1,088 NIST-supplied WebM clips extracted from the retained ZIP.
- `data/vqa/2026/testing/videos_original/`: 858 downloaded YouTube videos plus matching metadata.
- `data/vqa/2026/testing/DATASET_STATUS.md`: current 2026 coverage, evidence, and recheck instructions.
- `data/vqa/2026/testing/youtube_not_downloaded.csv`: the 10 API-omitted YouTube IDs and affected-query counts.

The original NIST ZIPs and metadata are retained even when extracted working copies exclude unavailable videos.

## Competition-data completeness

As of 2026-07-20:

- VTT: 10,152/10,152 media files, 3.2 GiB; the full NIST checksum list passed.
- VQA 2025 training package: the original 500-item NIST ZIP passes its archive
  test. The extracted working set contains 481 annotations after the 19
  exclusions named by NIST; 475/481 corresponding YouTube videos are local and
  6 are API-omitted. This is optional 2026 development data.
- VQA 2025 testing list: the original 2,000-query NIST CSV is present. After
  the 18 exclusions named by NIST, 1,883/1,982 corresponding YouTube videos are
  local. The remaining 99 comprise 96 API omissions and three downloader-level
  terminal cases documented below. This is optional 2026 development data.
- The YouTube Data API omitted 96 unarchived testing IDs. Three API-visible IDs
  also remained inaccessible to an unauthenticated downloader:
  - `y2Ouj_3UnII`: members-only.
  - `pau76KtNeeg`: unavailable when yt-dlp attempted it.
  - `mOP7QZfMEjE`: age-restricted and requires authentication.
- VQA 2026 AG test package: README, 2,443-query JSON, and the 1,088-video NIST
  ZIP are present; both ZIP integrity and extracted filename coverage pass.
  Of 868 referenced YouTube IDs, 858 have finalized media and metadata and 10
  were omitted by the final API check. Overall local test-video coverage is
  1,946/1,956 (99.49%); the 10 missing IDs affect 10 queries.
- VQA 2026 MC topics/options are not yet expected. NIST schedules their release
  for 2026-08-21 and states that AG and MC share the same testing videos.
- All data uses approximately 25 GiB at this snapshot.
- The first VQA media file was verified with `ffprobe` as 1080×1920 VP9 video plus Opus audio.

The VQA counts change while a download is running. Use the status command for live values.

## Downloader

Dependencies are managed by `uv`; no system Python package installation is required. FFmpeg/ffprobe and Node.js must already be available on `PATH`.

```bash
uv sync
uv run vqa-download prepare  # regenerate filtered URL manifests
uv run vqa-download status   # show PID, archive counts, and media counts
uv run vqa-download start    # detached, resumable training-then-testing run
```

`vqa-download` is a TRECVID path/configuration adapter over the same shared
engine exposed by `youtube-dataset-download`. For generic newline-delimited
YouTube ID/URL datasets, use the repository skill at
`.agents/skills/downloading-youtube-datasets/` and the generic CLI; there is no
second downloader implementation.

Downloader behavior:

- Selects the best available video stream plus best available audio, falling back to the best combined stream.
- Merges streams into MKV with stream copy; it does not re-encode or downscale.
- Saves `.info.json` metadata next to each media file.
- Uses separate training/testing download archives, partial-file continuation, retries, and per-video error skipping.
- Paces every attempted URL by 10 seconds and extraction requests by 1.5 seconds; separate video/audio streams do not receive duplicate long sleeps.
- On a YouTube rate limit, automatically pauses for one hour, then up to two hours on repeated limits, and resumes safely from the download archive.
- Treats YouTube's `Sign in to confirm you're not a bot` response as a rate
  limit and enters the same cooldown immediately.
- Uses the locked `yt-dlp-ejs` component and an absolute Node.js path for complete format extraction in detached sessions.
- Processes training before testing.
- Excludes the unavailable IDs listed in the NIST README files.
- Before every detached start, checks only unarchived IDs through the YouTube
  Data API in batches of 50 and passes only returned IDs to yt-dlp. API failures
  abort startup without falling back to the unfiltered list.
- Resuming is safe: archived successes are skipped and partial files continue.

Operational files:

- Progress/errors: `data/vqa/download_runtime/download.log`
- Downloader PID: `data/vqa/download_runtime/download.pid`
- Completed IDs: `training.archive.txt` and `testing.archive.txt`
- Filtered inputs: `training.urls.txt` and `testing.urls.txt`
- API-confirmed pending inputs: `training.pending-available.urls.txt` and
  `testing.pending-available.urls.txt`
- API omissions: `training.unavailable-or-private.txt` and
  `testing.unavailable-or-private.txt`

### Private YouTube API configuration

Detached starts require `/home/npatta01/.config/trec-video/youtube.env`, owned
by the current user with mode `0600`, containing a nonempty `YOUTUBE_API_KEY`.
The downloader binds the validated file through a private file descriptor and
scrubs any ambient value before sourcing it. The key is not placed in process
arguments, logs, generated inventories, or Git-tracked files.

The API may omit private, deleted, invalid, or otherwise inaccessible IDs
without distinguishing the reason. Runtime inventories therefore label all
omitted IDs `unavailable-or-private`.

### Private proxy configuration

Detached starts require `/home/npatta01/.config/trec-video/decodo.env`, owned
by the current user with mode `0600`. Use either `PROXY_URL` alone or all four
structured variables: `PROXY_SCHEME`, `PROXY_ENDPOINT`, `PROXY_USER`, and
`PROXY_PASSWORD`. Legacy `DECODO_*` variables remain supported only for VQA
compatibility. The downloader sources
that file in a subprocess, URL-encodes the user and password, and atomically
writes `data/vqa/download_runtime/yt-dlp-proxy.conf` with mode `0600`. Child
commands receive only that config path; credentials and the proxy URL are not
placed on the command line or in Git-tracked files.

`PROXY_ENDPOINT` accepts a DNS hostname or IPv4 literal followed by a numeric
port (for example, `proxy.example:7000`). Schemes, embedded credentials,
whitespace, and bracketed IPv6 endpoints are intentionally rejected.

The proxy uses the existing United States sticky session for every request
needed to complete one video. Do not rotate the session between metadata,
video, and audio requests. This integration does not alter the provider's
session-name syntax.

Proxy gates passed on 2026-07-16:

- The planned public control `BaW_jenozKc` was unavailable. Its authorized
  replacement, `jNQXAC9IVRw`, downloaded through the proxy with exit `0`, an
  exact archive row, one MKV, and verified video and audio streams. The isolated
  smoke media was removed afterward.
- Previously blocked VQA item `emArU432qfE` downloaded into the canonical test
  collection with exit `0`, one media file, one metadata file, an exact archive
  row, and verified video and audio streams. Testing archive and media counts
  both increased from 1,052 to 1,053.
- Protected-output scans found no raw or encoded credential or proxy-authority
  leakage. The full batch remained stopped; these proofs did not start it.

## Data provenance and privacy

- VTT data came from the authenticated NIST TRECVID VTT distribution.
- VQA annotations and query lists came from the authenticated NIST active-participant distribution.
- VQA media is fetched from the YouTube IDs supplied by NIST and may become unavailable, private, geo-blocked, or rate-limited.
- Authentication credentials are not stored in this repository or in the download manifests.
- Do not publish or change sharing permissions for these datasets without confirming the applicable NIST agreements and source-video terms.
## Research report

Research and submission-planning materials for the TRECVID 2026 Video Question Answering (VQA) task. The report is written for readers who are new to video QA and uses progressive disclosure: a short executive overview first, followed by task rules, 2025 results, team architectures, fine-tuning guidance, public models, literature, and an implementation plan.

### Report

- [Open the self-contained interactive report](reports/trecvid_vqa_2026/report.html)
- [Canonical report artifact](reports/trecvid_vqa_2026/artifact.json)
- [Sources, reconciliation decisions, and QA notes](reports/trecvid_vqa_2026/source_notes.md)

The HTML report is self-contained and can be opened directly in a modern browser. It includes a contents card, nine numbered chapters, accessible architecture diagrams, sortable evidence tables, and source details.

### What the report covers

- The 2026 Answer Generation and Multiple Choice tasks, output contracts, scoring, data, and tentative schedule.
- The complete published 2025 submission inventory and the strongest current appendix results.
- Plain-language explanations of video-language model components and four team architecture patterns.
- Fine-tuning choices: frozen baselines, ranker-only training, LoRA, QLoRA, and full-model tuning.
- Training-data design, leakage controls, GPU planning bands, and a staged experiment plan.
- Official public-weight video models and originator-released task fine-tunes, including access and license caveats.
- A 25-work competition-adjacent literature review tied to concrete TRECVID experiments.
- A recommended shared evidence engine with separate AG and MC output heads.

### Repository layout

```text
reports/
└── trecvid_vqa_2026/
    ├── report.html          # portable interactive report
    ├── artifact.json        # canonical manifest and bounded data snapshot
    ├── source_notes.md      # evidence ledger, calculations, and QA contract
    ├── assets/              # accessible SVG architecture explainers
    ├── queries/             # reproducible SQL-shaped evidence mirrors
    └── *.mjs                # deterministic report update/delivery scripts
```

### Rebuild the report

Run the update scripts in this order so content expansions occur before the chapter labels are applied:

```bash
cd reports/trecvid_vqa_2026
node expand_literature.mjs
node embed_diagrams.mjs
node expand_finetuning.mjs
node add_navigation.mjs
node deliver_report.mjs
```

`deliver_report.mjs` uses the Data Analytics report runtime installed in the local Codex environment and an installed Chrome/Chromium browser for desktop and phone-width verification. The committed `report.html` does not require that runtime to be read.

### Evidence boundary

Competition facts were checked against the official NIST pages and TREC 2025 proceedings through July 11, 2026. Model cards, fine-tuning documentation, and the literature supplement were checked July 12, 2026. See the report's source controls and `source_notes.md` for exact links, denominator reconciliation, limitations, and reproducibility notes.

Primary task page: [NIST TRECVID 2026 VQA](https://www-nlpir.nist.gov/projects/tv2026/vqa.html).
