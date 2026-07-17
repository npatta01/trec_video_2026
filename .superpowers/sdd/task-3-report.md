# Task 3 report: generic YouTube CLI and VQA adapter

## Outcome

- Added the `youtube-dataset-download` entry point with `prepare`, `run`, `start`, and `status`.
- Required explicit source, output, state, API environment, and proxy environment paths for every generic invocation.
- Routed generic preparation, execution, detached launch, and status through the shared `youtube_dataset` engine.
- Refused Git-tracked output/state paths unless `--allow-git-tracked` is explicit.
- Made direct run/start fail closed on API availability and proxy connectivity preflights before any download.
- Kept `vqa-download` commands, defaults, runtime paths, archive names, partial-file behavior, PID behavior, and compatibility exports unchanged.
- Added equivalence coverage proving the VQA adapter constructs shared `DatasetSpec` objects while retaining legacy paths.

## Verification

- `uv run pytest -q`: 94 passed.
- `uv run youtube-dataset-download --help`: exit 0; all four commands and explicit paths shown.
- Generic `status` smoke test with `/var/tmp/task3-safe/...`: exit 0, stopped/zero state.
- `uv run vqa-download status`: exit 0; existing runtime reported without mutation.
- `git diff --check`: clean.

## Notes

- No credentials or datasets were read by the implementation tests; API/proxy behavior was mocked.
- No network calls or workers were started.
- Existing untracked `dstack/` and `notes.txt` were not touched.
