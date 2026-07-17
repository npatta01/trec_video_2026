---
name: downloading-youtube-datasets
description: Use when preparing, downloading, resuming, or monitoring a dataset of YouTube video IDs or watch URLs, especially for unattended, proxied, original-quality, or failure-categorized collection.
---

# Download YouTube Datasets

Always use this repository's `youtube-dataset-download`; never implement a downloader or invoke `yt-dlp` or `vqa-download` directly. Run `uv run youtube-dataset-download --help` if the interface may have changed.

## Required inputs and safety gates

Use explicit source, output, state, API credential, and proxy credential paths:

```bash
uv run youtube-dataset-download COMMAND \
  --source SOURCE --output-dir OUTPUT --state-dir STATE \
  --api-env API_ENV --proxy-env PROXY_ENV
```

- Keep `API_ENV`, `PROXY_ENV`, output, and state private and Git-ignored. Confirm with `git check-ignore` and `git ls-files`; do not edit `.gitignore` implicitly or expose secrets in arguments, logs, reports, or generated files.
- Require current-user-owned, mode-`0600` credential files. Use separate files containing the documented API and proxy variables.
- In `PROXY_ENV`, use either `PROXY_URL` alone or all four structured variables: `PROXY_SCHEME`, `PROXY_ENDPOINT`, `PROXY_USER`, and `PROXY_PASSWORD`. Legacy `DECODO_*` variables exist only for VQA compatibility.
- Do not use `--allow-git-tracked` unless the user explicitly authorizes tracked runtime paths.
- Before a substantial run, verify available space and required local dependencies without downloading media.

## Workflow

1. Run `prepare`. It normalizes and deduplicates IDs, excludes archived successes, queries the YouTube Data API first in batches of at most 50, and atomically caches visible and omitted IDs. Treat any API or snapshot failure as blocking; never replace it with metadata probes.
2. Run `run` for an attended job or `start` for one detached worker. These commands repeat preparation and independently preflight the proxy against neutral HTTPS and YouTube before downloading. If either preflight fails, stop and notify the user.
3. Let the repository command own original-quality selection, one-video-at-a-time execution, ten-second pacing, archive deduplication, partial-file resume, and no-overwrite behavior. Never start concurrent workers for the same state.
4. Monitor with `status` and the reported log. Track worker/PID state, counts, last progress time, and any cooldown deadline. HTTP 429 or bot-check output must stop the pass and trigger the built-in one-hour, then at-most-two-hour cooldown. Repeated proxy HTTP 502 transport failures must stop promptly; preserve state and notify the user instead of restarting blindly.
5. Finish by reporting downloaded/archive and media counts, API-omitted IDs, terminally inaccessible categories (private, unavailable, members-only, authentication-required), retryable failures, worker state, and the exact resumable next command. Never claim API omissions identify a more specific cause than the stored evidence supports.
