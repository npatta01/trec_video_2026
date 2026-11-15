# TRECVID 2026 Video Question Answering

NP Labs’ entry for the NIST TRECVID 2026 Video Question Answering track. Given
a video and a question, we generated ranked answers and ranked the organizer’s
multiple-choice options. We compared Qwen on sampled frames and on video with
and without Whisper transcripts, and Gemini on video with audio.

[Project page](https://npatta01.github.io/trec_video_2026/) ·
[Paper (PDF)](docs/paper/trecvid2026.pdf) ·
[Workshop slides source](docs/workshop/trecvid2026-talk.qmd) ·
[Submission ledger](docs/submissions/README.md)

## Official test results

| Task | Our best result on the metric shown | Leaderboard rank |
| --- | --- | --- |
| Answer generation | **0.318244 first-answer METEOR** with Gemini first and Qwen alternatives | Tied 2nd of 44 runs |
| Multiple choice | **84.514% top-choice accuracy** with Gemini ranking from video, up from 78.748% when matching the saved answer | 2nd of 42 runs |

METEOR compares wording with reference answers. The answer-generation result
uses human and alternative references. The [paper](docs/paper/trecvid2026.pdf)
reports all eight runs, including STS, BERTScore and mean reciprocal rank.

## Repository guide

- [Development decisions](docs/project-status/index.md) and [experiment records](docs/experiments/README.md) document the baselines and follow-up studies.
- [Python implementation](src/trec_video_2026/) and [configuration files](configs/) contain the runners and experiment settings.
- [Workshop directory](docs/workshop/) contains the editable talk and local render instructions.
- [Experiment and result guide](docs/project-status/adding-experiments.md) explains how to record new work. [Reports](reports/trecvid_vqa_2026/README.md) and [plans](plans/README.md) preserve historical context.

## Local setup

Install the locked environment with `uv sync`. Run `uv run vqa-frozen30 --help`
to inspect the development benchmark, `uv run vqa-gate --help` for compatibility
checks, or `uv run vqa-download --help` and
`uv run youtube-dataset-download --help` for media tooling.

The competition dataset and raw experiment outputs remain local. On Framework,
linked worktrees use the ignored `data` symlink to the shared dataset. See the
[repository instructions](AGENTS.md#shared-local-dataset) before working with data.
