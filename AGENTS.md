# Repository agent instructions

## Adding or updating research

Before proposing, running, interpreting, or recording an experiment, read the
[Adding Experiments and Results guide](docs/project-status/adding-experiments.qmd).
Choose one primary theme for each current result; related themes should link to
that record rather than copy it. QMD is the editable report source: generated
Markdown and HTML must not be hand-edited.

## Shared local dataset

- The only physical dataset copy is
  `/home/npatta01/data/competitions/trec_video_2026/data`.
- Every linked Git worktree must access it through an ignored `data` symlink.
- Before work that needs the dataset, verify that `data` resolves to the
  canonical directory above. If it is missing, run:

  ```bash
  /home/npatta01/data/competitions/trec_video_2026/scripts/link-shared-data.sh --all
  ```

- The helper is idempotent. If it reports a conflicting path, stop and report
  the conflict; never remove or overwrite that path automatically.
- Never copy, move, delete, stage, or commit dataset contents.

## Full-panel VQA artifact reference

The completed 2026-08-10 full-panel AV event-map experiment is documented in
[`docs/experiments/2026-08-10-vqa-plaintext-full-panel.qmd`](docs/experiments/2026-08-10-vqa-plaintext-full-panel.qmd).
Its compact, reviewable evidence-of-record files are under
[`docs/experiments/artifacts/vqa-plaintext-full-20260810/`](docs/experiments/artifacts/vqa-plaintext-full-20260810/):

After this branch is merged to `main`, the same repository-relative links are
available from the shared checkout at
`/home/npatta01/data/competitions/trec_video_2026/`:

- report: `/home/npatta01/data/competitions/trec_video_2026/docs/experiments/2026-08-10-vqa-plaintext-full-panel.qmd`;
- artifact bundle: `/home/npatta01/data/competitions/trec_video_2026/docs/experiments/artifacts/vqa-plaintext-full-20260810/`.

Do not copy the tracked bundle manually between worktrees; merging the branch
to `main` makes it available in the shared checkout. The raw inspection media
is intentionally worktree-local and ignored by Git; use the local bundle in
the producing worktree or regenerate/link it separately if shared raw-media
inspection is required.

- `summary.json`, `correction.json`, `maps.jsonl`, and `results.jsonl` preserve
  the complete 475-row timeline/selector run and hashes.
- `selector-quality.json` contains the post-freeze METEOR, BERTScore-F1, STS,
  and explicitly degenerate NDCG summary.
- `selected-evidence/evidence.json`, `manifest.json`,
  `requests.frames.json`, and `requests.whisper.json` preserve selected
  timestamps, decoded PTS provenance, hashes, and the two answer-input arms.
- The local `selected-evidence/frames/`, `audio/`, and `contact-sheets/`
  directories are the complete private inspection bundle (5,700 JPEG frames,
  475 WAV files, and 475 contact sheets). They are intentionally ignored by
  Git because they are private derived media and total roughly 782 MB; do not
  delete them when inspecting this run. The manifests and hashes remain the
  audit record.

Do not reprovision dstack to recover this run. Inference is complete and the
remote development pod was stopped. If the local inspection bundle is absent,
classify that as an audit gap and use the tracked manifests rather than
silently rerunning or changing the experiment.
