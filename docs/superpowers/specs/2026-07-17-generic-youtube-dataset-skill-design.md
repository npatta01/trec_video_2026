# Generic YouTube Dataset Downloader Skill Design

## Goal

Create one generic YouTube dataset download engine in the repository and a repository-local Codex skill that delegates to it. The existing `vqa-download` command becomes a thin TRECVID-specific adapter over the same engine; no download behavior is duplicated.

## Location and contents

The implementation has these boundaries:

- `src/trec_video_2026/youtube_dataset.py`: the only implementation of API validation, caching, proxy handling, yt-dlp execution, archives, resume, cooldowns, detached execution, and status.
- `src/trec_video_2026/vqa_download.py`: TRECVID manifest preparation and path/configuration defaults, delegating all generic behavior to `youtube_dataset`.
- `youtube-dataset-download`: generic CLI entry point accepting explicit source, output, state, API-key file, and proxy file paths.
- `.agents/skills/downloading-youtube-datasets/SKILL.md`: triggering conditions, required workflow, safety gates, and generic CLI guidance.
- `.agents/skills/downloading-youtube-datasets/agents/openai.yaml`: generated UI metadata consistent with `SKILL.md`.

The skill contains no downloader script and never reimplements download behavior. It invokes `youtube-dataset-download` from this repository.

The repository README records the completed TRECVID result: 1,883 downloaded testing videos and the three API-visible IDs that remained inaccessible because they were members-only, unavailable, or age-restricted.

## Inputs and state

The generic CLI accepts a source text file containing one YouTube ID or watch URL per line, an output directory, and a state directory. State includes:

- An atomic YouTube Data API availability snapshot.
- A yt-dlp download archive.
- A terminal-failure inventory for private, unavailable, members-only, and authentication-required videos.
- A resumable log and status summary.

The source, output, and state paths are explicit CLI arguments. The script refuses output or state paths tracked by Git unless the user supplies an explicit override. It never modifies `.gitignore` implicitly.

## Credentials and proxy

The YouTube API key and proxy settings come from separate user-specified environment files. Each file must be a regular, current-user-owned file with mode `0600`, opened with `O_NOFOLLOW`. Ambient credential variables are scrubbed before sourcing. TRECVID defaults remain `/home/npatta01/.config/trec-video/youtube.env` and `/home/npatta01/.config/trec-video/decodo.env`; the generic CLI has no machine-specific defaults.

The API file contains `YOUTUBE_API_KEY`. The proxy file supports either a complete `PROXY_URL` or `PROXY_SCHEME`, `PROXY_ENDPOINT`, `PROXY_USER`, and `PROXY_PASSWORD`. The script URL-encodes proxy authentication and supplies it to yt-dlp through a private mode-`0600` configuration file bound at spawn, never directly in process arguments or logs.

## Workflow

1. Validate input IDs and remove duplicates while preserving order.
2. Read the yt-dlp archive and exclude completed IDs.
3. Query YouTube Data API `videos.list` for remaining IDs in batches of at most 50, using the API key in the `X-goog-api-key` header.
4. Atomically cache returned and omitted IDs. A failed or malformed API response aborts without invoking yt-dlp.
5. Verify the proxy can reach a neutral HTTPS endpoint and YouTube before downloading.
6. Invoke yt-dlp for exactly one API-visible ID at a time, selecting `bestvideo*+bestaudio/best`, merging into MKV without transcoding, writing metadata, resuming partial files, skipping archived successes, and refusing overwrites.
7. Wait ten seconds between ordinary attempts. On HTTP 429 or a bot-check response, stop the current pass and cool down for one hour, then at most two hours on repeated limits.
8. On repeated proxy transport failures such as HTTP 502, stop promptly, preserve state, and report the blocker rather than cycling through the dataset.
9. Finish with counts for downloaded, API-omitted, terminally inaccessible, and retryable failures.

## Commands

The generic CLI exposes `prepare`, `run`, `start`, and `status` commands. `start` launches one detached worker and records its PID. `status` reports archive/media counts, current state, latest progress time, cooldown deadline, and categorized failures without reading or printing secrets. `vqa-download` preserves its existing command surface and runtime compatibility while constructing the equivalent generic configuration.

## Verification

- Unit tests cover ID normalization, archive skipping, API batching and fail-closed behavior, credential-file validation, proxy secret containment, one-at-a-time execution, partial resume flags, bot/429 cooldowns, proxy-502 termination, detached PID behavior, and status output.
- A baseline skill evaluation demonstrates that an agent without the skill omits at least one required safety behavior.
- A forward evaluation demonstrates that an agent using the skill selects `youtube-dataset-download`, performs preflight checks, and stops/notifies on proxy failure.
- Equivalence tests prove the generic CLI and `vqa-download` route through the same library functions and that existing VQA archives, partial files, PID files, and runtime paths remain compatible.
- The skill passes `quick_validate.py`, its help command runs, the complete repository test suite passes, and tracked diffs contain no credentials.
