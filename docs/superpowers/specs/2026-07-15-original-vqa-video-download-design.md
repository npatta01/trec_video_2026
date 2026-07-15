# Original VQA Video Download Design

## Objective

Download the original highest-quality media currently available for every usable YouTube ID in the TRECVID 2025 VQA training and testing datasets. Preserve the source video and audio streams without transcoding. Downscaled or normalized copies are outside this scope.

## Dependency management

Initialize a minimal `uv` project in the repository and add `yt-dlp` as a locked dependency. Do not install a system package. Use `uv run yt-dlp` for every download and validation command.

## Inputs

- Training IDs come from `data/vqa/2025/training/annotations/train.2025.csv`.
- Testing IDs come from `data/vqa/2025/testing/testing.dataset.vqa.2025.csv`.
- Exclude the video IDs that the corresponding NIST README files identify as unavailable.
- Generate deterministic YouTube URL manifests under ignored `data/` paths.

## Download behavior

- Select the best available video stream plus the best available audio stream, falling back to the best combined stream.
- Merge streams into a compatible container when necessary, using stream copy only. Never re-encode video or audio.
- Store original media separately under:
  - `data/vqa/2025/training/videos_original/`
  - `data/vqa/2025/testing/videos_original/`
- Name media by YouTube video ID so annotations join directly to files.
- Preserve useful metadata with an adjacent `.info.json` file.
- Use a download archive, partial-file continuation, retries, and ignore-per-video failures so the collection can resume safely.
- Process training first, followed by testing, while preventing duplicate downloads.

## Nonblocking execution

Provide a terminal-friendly launcher that starts the download in a detached background process and immediately returns control. Keep the PID, download archive, progress log, and error log under ignored `data/vqa/download_runtime/`.

The launcher must refuse to start a duplicate active job. Re-running it after interruption must resume partial files and skip archived successes.

## Failure handling

- Record unavailable, private, geo-blocked, and extractor-failed videos in the log without terminating the batch.
- Retry transient network and fragment errors.
- Do not use browser cookies by default; authentication can be added later only if YouTube requires it.
- Do not delete a valid completed media file during retry or resume operations.

## Verification

Before handing off the background job:

1. Verify the generated training and testing URL counts against the NIST inputs and exclusion lists.
2. Run `yt-dlp` simulation on a small sample.
3. Start the detached job and verify its PID is alive and its log is advancing.
4. Verify the first completed media file with `ffprobe`, confirming both the original video and audio streams are present when the source provides audio.
5. Confirm all downloaded media and runtime files are ignored by Git and that no credentials are persisted.

## Deferred work

Generating 720p/H.264 copies, deleting originals, deduplicating by visual content, and model-specific frame extraction are explicitly deferred.
