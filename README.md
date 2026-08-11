# TRECVID 2026 Video Question Answering

Local tooling and evidence records for the TRECVID 2026 Video Question
Answering (VQA) work. This repository contains code and sanitized aggregate
findings. The competition dataset, media, transcripts, raw predictions, and
logs stay local and ignored; explicitly approved final organizer-upload files
may be tracked under `submissions/` in this private repository.

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
4. **Review prepared or submitted organizer runs:** use the
   [submission ledger](docs/submissions/README.md). Private run files remain
   local and are identified there by SHA-256.
5. **Understand historical context:** use the [report index](reports/trecvid_vqa_2026/README.md)
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

## Privacy boundary

`data/` may contain licensed NIST material, source-video media, annotations,
transcripts, predictions, and runtime artifacts. Keep it local and ignored.
Credentials belong in private `0600` configuration files outside the
repository. Do not publish data, raw artifacts, credentials, or private URLs,
and do not change sharing permissions without confirming the applicable terms.
An explicitly authorized final submission file belongs under `submissions/`,
never under a reusable source or report directory.

## Repository map

| Path | What it contains |
| --- | --- |
| `src/trec_video_2026/` | Python CLI implementations and experiment workflows. |
| `scripts/` | Safe operational helpers, including shared-data linking. |
| `tests/` | Automated checks for tracked code and contracts. |
| `docs/project-status/` | Current newcomer-oriented status and decision pages. |
| `docs/experiments/` | Sanitized, dated experiment findings. |
| `docs/submissions/` | Sanitized organizer-submission ledger and upload metadata. |
| `submissions/` | Explicitly approved final organizer-upload files in this private repository. |
| `plans/` | Historical GitHub-issue execution plans and their outcomes. |
| `docs/superpowers/plans/` | Historical implementation plans; unchecked boxes are not status. |
| `reports/trecvid_vqa_2026/` | July 12 field guide plus later Markdown experiment evidence. |
| `data/` | Ignored shared local dataset and private experiment artifacts. |

For historical plan context, see [implementation plans](docs/superpowers/plans/README.md).
