# Decodo Proxy Downloader Design

## Objective

Prove that the Decodo sticky residential proxy can download public YouTube media and one previously bot-blocked VQA item, then integrate the proxy into the resumable VQA downloader without exposing credentials.

## Scope and execution gates

Work proceeds in three strictly ordered stages:

1. Download yt-dlp's known-public test video `BaW_jenozKc` into an ignored temporary directory and verify its streams with `ffprobe`.
2. Delete the public smoke media, download unarchived VQA testing video `emArU432qfE` into the canonical testing directory, and confirm its final media and testing archive entry.
3. Only after both proofs pass, add proxy support to the production downloader and its tests. Do not start the full batch as part of this work.

If either proof fails, stop at that gate, retain sanitized diagnostics, and do not modify production downloader behavior.

## Credential handling

- Read credentials only from `/home/npatta01/.config/trec-video/decodo.env`, which must be owned by the current user and have mode `0600`.
- Source the file in a subprocess and require non-empty `DECODO_ENDPOINT`, `DECODO_USER`, and `DECODO_PASSWORD` values.
- URL-encode the username and password before constructing `http://<encoded-user>:<encoded-password>@<endpoint>`.
- Write the resulting `--proxy` option to `data/vqa/download_runtime/yt-dlp-proxy.conf` using an atomic mode-`0600` write.
- Pass only `--config-locations <config-path>` to yt-dlp. Never put the proxy URL in command arguments, logs, Git, test output, or status output.
- The config and all smoke artifacts remain under Git-ignored `data/vqa/` paths.

## Proxy and session behavior

Use the existing United States sticky session for each complete individual video so metadata, video, and audio requests share one exit. Do not rotate during a video. This initial integration does not invent or modify Decodo session-name syntax; rotation is deferred until the provider's supported session controls are explicitly available.

Preserve original `bestvideo*+bestaudio/best` selection, MKV stream-copy merge, Node/EJS extraction, archives, partial downloads, sequential processing, request pacing, ordinary URL delay, retry behavior, and cooldown behavior.

## Smoke tests

The public test uses an isolated temporary output directory and temporary archive under `data/vqa/proxy_smoke/`. Success requires:

- yt-dlp adds `BaW_jenozKc` to the temporary archive;
- one finalized ID-named media file exists;
- `ffprobe` reports at least one video stream and one audio stream;
- the proxy URL and credentials do not appear in captured command output or the process command line.

After verification, remove the public smoke media and metadata while retaining only a sanitized pass/fail record if useful.

The VQA proof uses the canonical testing output and archive so a successful download is immediately resumable production state. Success requires finalized `emArU432qfE` media, metadata, an archive row, and successful `ffprobe` stream inspection.

## Production integration

Add focused helpers to validate/source the credential file and atomically generate the private yt-dlp config. `start_detached` prepares the config before launching the worker. `build_ytdlp_command` accepts a proxy-config path and adds `--config-locations` without containing secret material. Failure to load credentials or write a secure config aborts startup with a sanitized error.

Unit tests use fake credential files and placeholder values. Tests cover URL encoding, `0600` permissions, missing/unsafe credential files, command construction without credential leakage, and detached environment/config propagation. Existing downloader tests must remain green.

## Documentation and completion

Update the README with the credential-file location, required variable names, security guarantees, proxy smoke command/workflow, and explicit statement that the full batch is not automatically started. Commit only source, tests, and documentation; never commit runtime configuration, credentials, logs, media, archives, or the unrelated `notes.txt` and `dstack/` working files.
