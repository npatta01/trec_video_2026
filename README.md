# TRECVID 2026 Video Question Answering

Local tooling and evidence records for the TRECVID 2026 Video Question
Answering (VQA) work. This repository contains code and sanitized aggregate
findings; the competition dataset, media, predictions, transcripts, and raw
logs stay local and ignored.

## Current baseline

The current frozen-30 evidence-backed baseline is Qwen3.5-4B with 12 uniformly
sampled frames. Add Whisper large-v3 only through a label-blind quality gate
for speech-dependent questions; it is not a global default. Frozen-30 found no
exact-answer gain from 24 versus 12 frames, so 24 is not the current global
default. The tested directed-12 and dense-burst-12 implementations were not
promoted as defaults on their respective ten- and two-row panels; other
smarter sampling methods remain open. Later three-row semantic screens did
not earn promotion for Gemma or Phi; they did not rule out either model under
other conditions. Qwen3-Omni's neutral A2 run was ineligible for its
predeclared promotion decision because 19/20 outputs passed the contract; a
repaired comparison can reconsider the model.

Read the maintained [project-status pages](docs/project-status/index.md) for
the current decision ledger and remaining gaps. The July 12
[HTML report](reports/trecvid_vqa_2026/report.html) is a background field
guide, not current experiment status.

## Start here

Choose the path that matches your goal:

1. **Run or change repository tooling:** read the CLI table, then the code map
   below. Preserve the shared-data and privacy boundaries before running a
   command.
2. **Understand current experiment decisions:** begin with the
   [project-status pages](docs/project-status/index.md), then the frozen-30
   [benchmark results](reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md).
3. **Add an experiment or result:** follow the
   [experiment and result guide](docs/project-status/adding-experiments.md).
4. **Understand historical context:** use the [report index](reports/trecvid_vqa_2026/README.md)
   and the [execution-plan index](plans/README.md). Historical checklists and
   earlier conclusions are records, not live work queues.

## Supported CLI

Install the locked environment once with `uv sync`, then use `uv run`:

| Command | Purpose | Safe starting command |
| --- | --- | --- |
| `vqa-download` | Prepare, monitor, or resume VQA media acquisition. | `uv run vqa-download status` |
| `vqa-frozen30` | Run the frozen-30 benchmark workflow. | `uv run vqa-frozen30 --help` |
| `vqa-gate` | Preflight the small compatibility/quality gate. | `uv run vqa-gate --help` |
| `youtube-dataset-download` | Run the general YouTube-ID downloader. | `uv run youtube-dataset-download --help` |

The download commands can create or resume ignored local runtime state. Do not
start a large download or benchmark merely to inspect this repository.

## Shared data: safe setup

The only physical dataset copy is
`/home/npatta01/data/competitions/trec_video_2026/data`. Each linked worktree
must use an ignored absolute `data` symlink to that path. Before any task that
needs data, confirm the link resolves there. If it is missing, run:

```bash
scripts/link-shared-data.sh --all
```

The helper is repeatable and refuses to replace a conflicting path. If it
reports a conflict, stop; never delete, copy, move, stage, or commit dataset
contents to repair it.

## Private Gemini environment: safe setup

The tracked `private-env/` directory is a discoverable placeholder; its
contents other than `.gitignore` and `.gitkeep` are ignored. Link the private
Gemini environment into the current worktree with:

```bash
scripts/link-private-env.sh
```

Use `scripts/link-private-env.sh --all` to link every registered worktree.
The helper uses the private source
`/home/npatta01/.config/trec_video_2026/gemini.env` by default (or the same
relative path under `XDG_CONFIG_HOME`), requires a current-user-owned `0600`
regular file, and refuses conflicting paths. It never reads or prints the
credential value. Do not stage or commit `private-env/gemini.env`.

## Privacy boundary

`data/` may contain licensed NIST material, source-video media, annotations,
transcripts, predictions, and runtime artifacts. Keep it local and ignored.
Credentials belong in private `0600` configuration files outside the
repository. Do not publish data, raw artifacts, credentials, or private URLs,
and do not change sharing permissions without confirming the applicable terms.

## Repository map

| Path | What it contains |
| --- | --- |
| `src/trec_video_2026/` | Python CLI implementations and experiment workflows. |
| `scripts/` | Safe operational helpers, including shared-data linking. |
| `tests/` | Automated checks for tracked code and contracts. |
| `private-env/` | Ignored per-worktree links to private credentials. |
| `docs/project-status/` | Current newcomer-oriented status and decision pages. |
| `docs/experiments/` | Sanitized, dated experiment findings. |
| `plans/` | Historical GitHub-issue execution plans and their outcomes. |
| `docs/superpowers/plans/` | Historical implementation plans; unchecked boxes are not status. |
| `reports/trecvid_vqa_2026/` | July 12 field guide plus later Markdown experiment evidence. |
| `data/` | Ignored shared local dataset and private experiment artifacts. |

For historical plan context, see [implementation plans](docs/superpowers/plans/README.md).
