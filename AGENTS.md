# Repository agent instructions

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
