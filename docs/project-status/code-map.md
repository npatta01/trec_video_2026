# Code Map and Remaining Gaps


<div class="lead">

The repository has reusable acquisition and benchmark seams, four
supported project commands, isolated model-family adapters, and several
bespoke research workflows. For a newcomer, **`vqa-frozen30` is the most
complete end-to-end workflow**; `vqa-gate` currently exposes preflight
only, and there is no general `vqa-score` command.

</div>

Use [Start Here](index.qmd) to choose a baseline and the [Experiment
Ledger](experiment-ledger.qmd) to understand why it was chosen.

## System pipeline

``` mermaid
flowchart TD
  accTitle: Project system pipeline
  accDescr: Acquisition feeds evidence preparation, inference, scoring after predictions freeze, and sanitized aggregate reports.
  A[Acquisition<br/>resumable local media] --> P[Evidence preparation<br/>frames, manifests, requests]
  P --> I[Inference<br/>isolated family adapters]
  I --> S[Scoring<br/>labels joined after freeze]
  S --> R[Reports<br/>sanitized aggregates]
```

**Figure caption:** The repository pipeline from source-video
acquisition through sanitized reports.

| Stage | Main tracked code | Boundary |
|----|----|----|
| Acquisition | [`youtube_dataset.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/youtube_dataset.py), [`vqa_download.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_download.py) | Media and runtime state stay local and ignored. |
| Evidence preparation | [`vqa_benchmark.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_benchmark.py), [`vqa_gate.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_gate.py), tracked configs | Questions and evidence are bound without reference labels in inference requests. |
| Inference | family runners plus [`vqa_frozen30_workflow.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_frozen30_workflow.py) | Heavy model runtimes are isolated; request/result identity is checked. |
| Scoring | [`vqa_evaluation.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_evaluation.py) and experiment-specific scoring modules | Labels are joined only after predictions are frozen. |
| Reports | `docs/experiments/`, `reports/trecvid_vqa_2026/`, and these status pages | Commit sanitized summaries, never private runtime artifacts. |

The arrow sequence is operational, not a promise of a single universal
command. Only frozen-30 currently packages preparation, guarded
execution, resume/status, and scoring into one supported workflow.

## Supported commands

The four installed project scripts are declared in
[`pyproject.toml`](https://github.com/npatta01/trec_video_2026/blob/main/pyproject.toml).
After `uv sync`, inspect them with `uv run <command> --help`; do not
launch a large download or GPU run merely to explore the code.

| Command | Role | What a newcomer should know |
|----|----|----|
| `youtube-dataset-download` | Generic YouTube-ID/watch-URL acquisition | Reusable engine with `prepare`, `run`, `start`, and read-only `status` commands for a configured YouTube dataset. |
| `vqa-download` | TRECVID-specific acquisition adapter | Supplies repository-specific paths and operational defaults around the generic engine; commands can mutate ignored local runtime state. |
| `vqa-gate` | Small compatibility-gate entry point | Its public CLI exposes **`preflight` only**. The completed family runs used tested Python seams and adapters rather than a general one-command gate. |
| `vqa-frozen30` | Frozen-30 benchmark workflow | The most complete workflow: CPU preparation, read-only status, Whisper generation, transcript request binding, explicit condition authorization/run, resume controls, and scoring. |

There is **no general `vqa-score` CLI**. General scoring primitives live
in
[`vqa_evaluation.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_evaluation.py);
frozen-30 exposes its own `score` action, and later experiments have
purpose-built scoring modules or scripts.

## Code layers

| Layer | Representative paths | Intended reuse |
|----|----|----|
| Supported CLIs | `youtube_dataset.py`, `vqa_download.py`, `vqa_gate.py`, `vqa_frozen30_workflow.py` | Stable user entry points declared by `pyproject.toml`. |
| Reusable modules | [`vqa_benchmark.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_benchmark.py), [`vqa_frozen30.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_frozen30.py), [`vqa_evaluation.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_evaluation.py), [`vqa_models.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_models.py) | Evidence construction, manifests, fail-closed resume controls, evaluation, and shared prompt behavior. |
| Specialist adapters | [`vqa_transformers_runner.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_transformers_runner.py), [`vqa_specialist_runner.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_specialist_runner.py), [`vqa_omni_runner.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_omni_runner.py), [`vqa_whisper.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_whisper.py), [`vqa_audio_runner.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_audio_runner.py), [`vqa_seamless.py`](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_seamless.py) | Isolated `python -m trec_video_2026.<module>` request/result adapters for pinned model families or modalities; they are not separate project scripts. |
| Bespoke experiment workflows | [`vqa_experiments/`](https://github.com/npatta01/trec_video_2026/tree/main/src/trec_video_2026/vqa_experiments) and `scripts/` experiment helpers | Frozen contracts, transport, scoring, and collection for a specific research question; reuse only after reading that experiment’s report and limitations. |

The useful architectural split is between orchestration and adapters:
host-side code freezes requests and validates provenance, while
model-family runners import heavy dependencies only inside their pinned
execution runtime.

## Repository map

``` mermaid
flowchart TD
  accTitle: Repository map
  accDescr: The repository has newcomer entry points, source modules, operations files, tests, sanitized evidence, and an ignored local-data boundary.
  ROOT[Repository root] --> ENTRY[README + pyproject<br/>start and commands]
  ROOT --> SRC[src/trec_video_2026<br/>CLIs, modules, adapters]
  ROOT --> OPS[configs + scripts<br/>frozen inputs and helpers]
  ROOT --> EVID[docs + reports<br/>sanitized evidence]
  ROOT --> TESTS[tests<br/>contracts and tooling]
  ROOT --> LOCAL[data<br/>ignored local boundary]
```

**Figure caption:** Repository map showing the tracked newcomer entry
points and the ignored local-data boundary.

| Path | Read it when… | Caution |
|----|----|----|
| [`README.md`](https://github.com/npatta01/trec_video_2026/blob/main/README.md) and [`pyproject.toml`](https://github.com/npatta01/trec_video_2026/blob/main/pyproject.toml) | You need the operating boundary or supported command names. | A help command is safer than starting work. |
| [`src/trec_video_2026/`](https://github.com/npatta01/trec_video_2026/tree/main/src/trec_video_2026) | You are changing reusable behavior, orchestration, or an adapter. | Identify the layer before widening an experiment-specific seam. |
| [`configs/vqa/`](https://github.com/npatta01/trec_video_2026/tree/main/configs/vqa) and [`scripts/`](https://github.com/npatta01/trec_video_2026/tree/main/scripts) | You need frozen run definitions or operational helpers. | Some helpers launch expensive or stateful work; inspect first. |
| [`docs/experiments/`](https://github.com/npatta01/trec_video_2026/tree/main/docs/experiments) | You need recent sanitized experiment contracts and results. | Compatibility reports may be superseded by quality reports. |
| [`reports/trecvid_vqa_2026/`](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/README.md) | You need the July 12 field guide or later analytical reports. | Historical recommendations are not the current default. |
| `tests/` | You are changing tracked behavior and need its executable contract. | This documentation task did not run or add application tests. |
| `data/` | A specifically authorized run requires the shared local dataset. | Ignored and private: never copy, stage, commit, or expose it. |

## Remaining gaps

These are issue **seeds**, not newly created GitHub issues. Validate and
scope them before implementation.

### Documentation seeds

- Add one safe, copy-paste walkthrough from `uv sync` through a
  read-only preflight/status inspection, without assuming dataset
  access.
- Document which experiment-specific modules are reusable contracts
  versus frozen historical workflows.
- Keep a small claim-to-report-to-code crosswalk as new experiments
  supersede older recommendations.
- Document the separate Quarto source, GitHub-readable Markdown, and
  self-contained HTML update procedure after the first render is
  reviewed.

### Engineering seeds

- Extend `vqa-gate` beyond `preflight` only if one general
  prepare/run/report contract can preserve the existing label and
  provenance boundaries.
- Decide whether a general `vqa-score` CLI is valuable, or whether
  scoring should remain owned by frozen workflows; do not silently
  expose labels to inference-side code.
- Consolidate the common request/result contract across Transformers,
  specialist, audio, and omni adapters without erasing required
  family-specific attestations.
- Turn the recommended question/evidence routing policy into a
  label-blind, versioned configuration with explicit fallback and
  non-regression gates.
- Define a cheap representative end-to-end smoke path before any larger
  local or rented-GPU run.

### Research seeds

- Measure 2026 performance when valid evaluation evidence becomes
  available; it is currently unknown.
- Validate lexical-speech, prosody, non-speech, counting, and
  visible-text routing prevalence on a larger label-blind question
  sample.
- Run a matched native-audio specialist benchmark with separate speech,
  prosody, non-speech, temporal, and visual-control strata.
- Re-evaluate Qwen3-Omni only with a strict output-complete run and a
  larger, predeclared panel; do not promote the 19/20 diagnostic.
- Design a different temporal selector only if it addresses the directed
  and dense-burst failure modes under a fixed evidence budget.
- Keep the independent human-oracle tier blocked until the required
  transcriber/reviewer provenance exists; track it in [open issue
  \#19](https://github.com/npatta01/trec_video_2026/issues/19).

The current safe next action is therefore determined by the question:
use the [Experiment Ledger](experiment-ledger.qmd) for a research
decision, or begin at the supported-command table above for a code
change.
