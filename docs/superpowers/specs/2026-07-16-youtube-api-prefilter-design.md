# YouTube API Prefilter and Bot-Check Cooldown Design

## Goal

Before invoking yt-dlp, classify every unarchived TRECVID 2025 VQA YouTube ID with the YouTube Data API. Download only IDs returned by the API, and treat YouTube's bot-check response as a rate limit that triggers the existing cooldown.

## Inputs and private configuration

- Source IDs remain the NIST-filtered `training.urls.txt` and `testing.urls.txt` manifests.
- Completed IDs remain authoritative in the yt-dlp archive files.
- The API key is read from `/home/npatta01/.config/trec-video/youtube.env`, which must be a regular file owned by the current user with mode `0600` and contain a nonempty `YOUTUBE_API_KEY`.
- The key must never appear in process arguments, logs, tracked files, exceptions, or generated inventories.

## Availability prefilter

The downloader queries the YouTube Data API `videos.list` endpoint in batches of at most 50 IDs. It checks only IDs absent from the corresponding yt-dlp archive. IDs returned by the API are written as URLs to an ignored `*.pending-available.urls.txt` runtime manifest. IDs omitted by the API are written to an ignored `*.unavailable-or-private.txt` inventory and are not passed to yt-dlp.

Every start refreshes the snapshot before spawning the detached downloader. A failed, malformed, or incomplete API response aborts the start; it must never fall back to probing the unfiltered manifest. Empty pending manifests are valid. Generated files are atomically replaced so a failed refresh cannot leave a partial snapshot.

The API cannot distinguish all reasons an ID is omitted. Reports therefore use the label `unavailable-or-private`, not `deleted` or `private` individually.

## Download and cooldown behavior

The sequential downloader consumes only the pending-available manifests. Existing archive skipping, partial-file continuation, original-quality format selection, proxy handling, one-video-at-a-time execution, 10-second pacing, and one-/two-hour cooldowns remain unchanged.

Both straight and typographic apostrophe variants of `Sign in to confirm you're not a bot` are classified as rate limiting. The current attempt stops, the downloader waits one hour, and a repeated block waits up to two hours. Ordinary private/unavailable messages are not rate limits.

## Verification

- Unit tests cover secure API-key loading, batches of 50, archive exclusion, output partitioning, API failure without fallback, automatic prefilter use, and both bot-check apostrophe variants.
- The complete test suite must pass.
- A real API-only precheck must reproduce the current inventory without running yt-dlp.
- Before restart, verify generated pending IDs and omitted IDs are disjoint and partition every unarchived input ID.
- After restart, verify exactly one yt-dlp child and confirm its ID belongs to the API-available pending manifest.

