# Full-panel plaintext AV event-map artifacts

This directory is the compact evidence-of-record bundle for the 2026-08-10
475-video experiment. The canonical interpretation is in
[`../../2026-08-10-vqa-plaintext-full-panel.qmd`](../../2026-08-10-vqa-plaintext-full-panel.qmd).

## Shared checkout after merge

After this branch is merged to `main`, this tracked bundle will be available
from the shared checkout at:

`/home/npatta01/data/competitions/trec_video_2026/docs/experiments/artifacts/vqa-plaintext-full-20260810/`

The report will be at:

`/home/npatta01/data/competitions/trec_video_2026/docs/experiments/2026-08-10-vqa-plaintext-full-panel.qmd`

The repository-relative links in the report and this README are portable
across worktrees. The raw inspection media below is ignored and remains in the
producing worktree; it is not copied into the shared checkout by a merge.

## Tracked compact artifacts

| File | Purpose |
|---|---|
| `summary.json` | Completion counts and SHA-256 hashes for maps and selector results. |
| `correction.json` | Final-run correction and salvage policy. |
| `maps.jsonl` | One label-free full-video timeline result per annotated video. |
| `results.jsonl` | One label-free scene-selection/answer result per annotated video. |
| `selector-quality.json` | Post-freeze METEOR, BERTScore-F1, STS, token diagnostics, and failure-penalized summary. |
| `selected-evidence/evidence.json` | Selected windows, requested timestamps, decoded PTS, frame/audio hashes, and Whisper provenance. |
| `selected-evidence/manifest.json` | 475-row label-free evidence manifest. |
| `selected-evidence/requests.frames.json` | Answer requests without Whisper transcript. |
| `selected-evidence/requests.whisper.json` | Matched answer requests with accepted Whisper transcript where available. |

## Local inspection media

The complete derived inspection media remains next to these files in the local
worktree:

- `selected-evidence/frames/`: 5,700 selected JPEG frames;
- `selected-evidence/audio/`: 475 extracted WAV tracks;
- `selected-evidence/contact-sheets/`: 475 twelve-frame contact sheets.

Those directories are intentionally ignored by Git because they are private
derived media and total roughly 782 MB. The files are retained locally and
their paths, hashes, and decoded timestamps are recorded in `evidence.json`.
