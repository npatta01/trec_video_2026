# Task 2 report: shared YouTube engine extraction

## Outcome

Extracted reusable YouTube download behavior into `src/trec_video_2026/youtube_dataset.py` and reduced `vqa_download.py` to a TRECVID adapter. Compatibility exports for reusable functions are direct aliases, preserving object identity.

## Boundaries

- `youtube_dataset.py` owns credential validation, API batching, atomic cache writes, proxy configuration, yt-dlp command construction/execution, archive reading, cooldown handling, proxy-failure stopping, detached worker primitives, PID/status helpers, and the generic dataclasses.
- `vqa_download.py` retains the TRECVID credential defaults, NIST CSV/README parsing, manifest construction, TRECVID path construction, and the VQA command adapter.
- The generic module defines no `DEFAULT_*` credential paths. Generic fallback entry behavior requires explicit credential file environment values.
- Existing compatibility names point to the exact shared function objects.

## Dataclasses

- `DatasetSpec`: explicit source, output, and state paths.
- `DownloadConfig`: explicit API/proxy credential paths and pacing/cooldown values.
- `AttemptResult`: completion, rate-limit, and proxy-transport classifications.

## Proxy failure behavior

`run_url` recognizes a proxy-associated HTTP 502 as `proxy_failure=True`. `run_dataset` stops after two consecutive proxy transport failures, preserving the archive and avoiding a full-dataset failure loop. Any non-proxy result resets the consecutive counter; ordinary private/unavailable/video failures remain ordinary failures and do not trigger the proxy stop.

## TDD evidence

The ownership test initially failed during collection because `trec_video_2026.youtube_dataset` did not exist. New regression tests then covered:

1. HTTP 502 proxy output classification.
2. Prompt stop after two consecutive proxy failures.
3. Processing all URLs when failures are ordinary video failures.

Focused verification: `uv run pytest -q tests/test_youtube_dataset.py tests/test_vqa_download.py` -> 83 passed.

Full verification: `uv run pytest -q` -> 83 passed.

Diff hygiene: `git diff --check` passed. Untracked `dstack/` and `notes.txt` were not touched or staged. No downloader or network operation was started.

## Self-review

- Confirmed TRECVID defaults exist only in the adapter.
- Confirmed ownership/identity assertions pass.
- Confirmed proxy classification requires both a 502 indicator and proxy wording.
- Confirmed bounded stopping is consecutive rather than cumulative.
- Confirmed the existing VQA tests remain green after moving monkeypatch targets for shared implementation internals.

## Review-fix wave

The initial Task 2 review found that the extracted module still contained VQA path, manifest, worker, and status assumptions, and that the adapter had lost two status counters. The correction made the shared interfaces operational rather than nominal:

- `DatasetSpec` now drives arbitrary source, output, and state paths through `prepare_dataset`, `run_spec`, `dataset_status`, and `start_detached`.
- `DownloadConfig` now drives the API-key file, proxy file, pacing, cooldowns, node runtime, and explicit detached worker command.
- Removed generic-module copies of NIST CSV/README parsing, TRECVID manifest construction, training/testing loops, `data/vqa` paths, VQA environment names/messages, and the hard-coded `trec_video_2026.vqa_download` worker spawn.
- The VQA adapter constructs the two TRECVID datasets and explicit detached worker command while retaining compatibility exports as the shared function objects.
- Restored the `API-available pending` and `API unavailable/private` status counters using the shared `count_lines` helper.

### Added regression coverage

- A non-VQA `DatasetSpec` rooted at arbitrary source/output/state directories proves API snapshot preparation, archive/output routing, configuration-driven API key loading, execution, and status counts.
- Detached execution proves the explicit `worker_command`, proxy config, node path, state directory, and PID handling are bound from generic configuration.
- VQA status explicitly verifies both API inventory counters.

### Verification evidence

Commands run without downloader or network activity:

```text
uv run pytest -q tests/test_youtube_dataset.py tests/test_vqa_download.py
86 passed in 0.08s

uv run pytest -q
86 passed in 0.40s

python -m compileall -q src
exit 0

git diff --check
exit 0
```

Boundary audit command:

```text
rg -n 'data/vqa|training|testing|VQA_|vqa_download|NIST|DEFAULT_' src/trec_video_2026/youtube_dataset.py
no matches
```

Untracked `dstack/` and `notes.txt` remained untouched and unstaged.
