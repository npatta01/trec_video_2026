# Rate-Limit-Aware VQA Downloader Design

## Objective

Resume the original-quality TRECVID VQA YouTube download faster than the current configuration while keeping a single guest session below YouTube's practical request limits and recovering automatically when the session is rate-limited.

## Scope

- Preserve the existing NIST manifests, yt-dlp archives, partial downloads, original video/audio format selection, MKV stream-copy merge, metadata files, and detached execution.
- Change pacing and recovery only.
- Do not use account cookies, proxies, parallel download jobs, transcoding, or downscaling.

## Design

The worker processes manifest URLs sequentially rather than giving an entire batch to one opaque yt-dlp process. Archive-completed URLs are removed before work begins. Each remaining URL is passed to yt-dlp individually with the same archive and output configuration.

After every attempted URL, including deleted or private videos, the worker waits 10 seconds before starting the next URL. yt-dlp also waits 1.5 seconds between extraction requests. The old 20–40 second pre-download delay is removed because it can run separately for video and audio streams while failing to pace unavailable URLs at all.

The worker streams yt-dlp output into the existing log and inspects it for the documented YouTube session-rate-limit message. When detected, it stops scheduling URLs, waits one hour, then begins another resumable pass. A second consecutive rate-limit event before a successful URL increases the cooldown to two hours. Any successful archived download resets the cooldown to one hour.

## Data Flow

1. Regenerate the filtered training and testing manifests.
2. Read each dataset's yt-dlp archive and skip completed IDs locally.
3. Run one yt-dlp subprocess for the next unarchived URL.
4. Forward output to the detached job log.
5. Classify the result as completed, unavailable/error, or rate-limited.
6. Pause 10 seconds after ordinary attempts, or apply the cooldown after rate limiting.
7. Repeat passes until a complete pass adds no videos and encounters no rate limit.

## Failure and Resume Behavior

- Download and fragment retries remain enabled within yt-dlp.
- Deleted, private, and other unavailable videos remain errors in the log but do not stop the dataset.
- A stopped host or killed worker can be restarted safely because yt-dlp archives and partial files remain authoritative.
- A rate-limit cooldown must be logged with its duration and next retry time so status is understandable.
- The downloader remains single-process to avoid multiplying requests against the same guest-session quota.

## Verification

- Unit tests prove archive filtering, per-URL invocation, 1.5-second extraction pacing, removal of pre-stream sleeps, ordinary inter-URL pacing, and one-/two-hour cooldown behavior.
- The full existing test suite remains green.
- A controlled fake-run test proves a rate-limit result triggers cooldown and resume without real waiting.
- A live smoke test after the existing cooldown proves the detached worker starts, selects original video plus audio, and either archives a new item or reports a genuine unavailable video without immediately scanning the manifest at the old rate.

## Expected Performance

The guest-session ceiling documented by yt-dlp is approximately 300 videos per hour. This design targets a practical rate below that ceiling while eliminating duplicate 20–40 second sleeps for separate video/audio streams. The estimated remaining runtime is roughly 10–16 hours, plus any one- or two-hour cooldowns required by YouTube.
