# TRECVID 2026 VQA

Local tools and private, Git-ignored datasets for the TRECVID 2026 Video Question Answering track.

## Data inventory

Everything under `data/` is local-only. The large media and runtime files are intentionally excluded from Git by `.gitignore`.

| Path | Contents | Expected usable items |
| --- | --- | ---: |
| `data/vtt/` | NIST TRECVID Video-to-Text data from 2016–2024: videos, captions, URL metadata, and checksums | 10,152 local WebM videos; captions cover 16,627 IDs |
| `data/vqa/2025/training/` | Original NIST training ZIP/README, extracted annotations, and downloaded YouTube media | 481 usable videos/annotations from 500 rows after 19 NIST exclusions |
| `data/vqa/2025/testing/` | NIST 2025 test CSV/README and downloaded YouTube media | 1,982 usable videos from 2,000 rows after 18 NIST exclusions |
| `data/vqa/download_runtime/` | Generated URL manifests, yt-dlp archives, PID files, progress logs, and completion-monitor state | Runtime state; safe to retain for resume |

Important files:

- `data/vtt/vtt_ground_truth.txt`: VTT captions.
- `data/vtt/Video_Files/`: VTT WebM media.
- `data/vtt/checklist.chk`: checksums for all 10,152 VTT media files.
- `data/vqa/2025/training/annotations/train.2025.csv`: 500 original training rows.
- `data/vqa/2025/training/annotations/*.json`: extracted question/answer annotations; the working copy excludes 19 unavailable videos.
- `data/vqa/2025/testing/testing.dataset.vqa.2025.csv`: 2,000 prior test queries.
- `data/vqa/2025/*/videos_original/`: highest-quality YouTube video+audio streams, merged without transcoding.

The original NIST ZIPs and metadata are retained even when extracted working copies exclude unavailable videos.

## Verified snapshot

As of 2026-07-15:

- VTT: 10,152/10,152 media files, 3.2 GiB; the full NIST checksum list passed.
- VQA training originals: 464/481 archived as complete.
- VQA testing originals: 482/1,982 archived as complete.
- VQA data used approximately 3.8 GiB at this snapshot.
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

Downloader behavior:

- Selects the best available video stream plus best available audio, falling back to the best combined stream.
- Merges streams into MKV with stream copy; it does not re-encode or downscale.
- Saves `.info.json` metadata next to each media file.
- Uses separate training/testing download archives, partial-file continuation, retries, and per-video error skipping.
- Paces every attempted URL by 10 seconds and extraction requests by 1.5 seconds; separate video/audio streams do not receive duplicate long sleeps.
- On a YouTube rate limit, automatically pauses for one hour, then up to two hours on repeated limits, and resumes safely from the download archive.
- Uses the locked `yt-dlp-ejs` component and an absolute Node.js path for complete format extraction in detached sessions.
- Processes training before testing.
- Excludes the unavailable IDs listed in the NIST README files.
- Resuming is safe: archived successes are skipped and partial files continue.

Operational files:

- Progress/errors: `data/vqa/download_runtime/download.log`
- Downloader PID: `data/vqa/download_runtime/download.pid`
- Completed IDs: `training.archive.txt` and `testing.archive.txt`
- Filtered inputs: `training.urls.txt` and `testing.urls.txt`

## Data provenance and privacy

- VTT data came from the authenticated NIST TRECVID VTT distribution.
- VQA annotations and query lists came from the authenticated NIST active-participant distribution.
- VQA media is fetched from the YouTube IDs supplied by NIST and may become unavailable, private, geo-blocked, or rate-limited.
- Authentication credentials are not stored in this repository or in the download manifests.
- Do not publish or change sharing permissions for these datasets without confirming the applicable NIST agreements and source-video terms.
