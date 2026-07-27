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
