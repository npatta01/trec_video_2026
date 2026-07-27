# Code Map and Remaining Gaps


<div class="lead">

The repository has reusable acquisition and benchmark seams, four
supported project commands, isolated model-family adapters, and several
bespoke research workflows. For a newcomer, **`vqa-frozen30` is the most
complete end-to-end workflow**; `vqa-gate` currently exposes preflight
only, and there is no general `vqa-score` command.

</div>

Use [Start Here · Report page](index.md) to choose a baseline and
[Research Themes · Report page](experiment-ledger.md) to understand why
it was chosen.

> [!NOTE]
>
> ### How links in this report work
>
> **Evidence of record · Evidence report · GitHub** links use the
> immutable repository snapshot
> `814e55429f6dd1dc8570bc08f1dad849f74e36a1`. **Current code · Source
> code · GitHub** links use `main` because those implementation paths
> can evolve. **Issue/PR history · GitHub** links preserve the project
> history. **Report page** links stay inside this report. GitHub links
> require repository access until public release.

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

| Stage | Main tracked code · Current code · Source code · GitHub | Boundary |
|----|----|----|
| Acquisition | [`youtube_dataset.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/youtube_dataset.py), [`vqa_download.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_download.py) | Media and runtime state stay local and ignored. |
| Evidence preparation | [`vqa_benchmark.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_benchmark.py), [`vqa_gate.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_gate.py), tracked configs | Questions and evidence are bound without reference labels in inference requests. |
| Inference | family runners plus [`vqa_frozen30_workflow.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_frozen30_workflow.py) | Heavy model runtimes are isolated; request/result identity is checked. |
| Scoring | [`vqa_evaluation.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_evaluation.py) and experiment-specific scoring modules | Labels are joined only after predictions are frozen. |
| Reports | `docs/experiments/`, `reports/trecvid_vqa_2026/`, and these status pages | Commit sanitized summaries, never private runtime artifacts. |

The arrow sequence is operational, not a promise of a single universal
command. Only frozen-30 currently packages preparation, guarded
execution, resume/status, and scoring into one supported workflow.

## Trace the synthetic example to code

The [worked example · Report page](worked-example.md) is a deliberately
invented explanation: no video was downloaded, no model ran, and no
benchmark row was written. The crosswalk below asks a narrower question:
**if a real, authorized row followed the same shape, which repository
seam would own each stage?** It does not convert the synthetic answer
into executed evidence.

> [!WARNING]
>
> ### Do not copy the stateful or expensive commands casually
>
> `vqa-download` can download media and mutate ignored local runtime
> state. `vqa-frozen30 run-whisper` and `vqa-frozen30 run CONDITION` can
> launch expensive GPU inference. Inspect `status`/`--help`, confirm
> dataset authorization and the frozen protocol, and obtain any required
> run authorization before using them. The commands below are a
> code-path map, not a walkthrough to execute now.

| Stage in the teaching trace | Exact reusable CLI or module | What really happened in the synthetic page |
|----|----|----|
| **1. Acquisition** | Reusable CLIs: `uv run vqa-download …` (TRECVID adapter) and `uv run youtube-dataset-download …` (generic engine); modules: [`vqa_download.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_download.py) and [`youtube_dataset.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/youtube_dataset.py). | **Bespoke/missing execution:** the 22-second clip and scene notes were invented, so neither acquisition CLI ran and no media exists. |
| **2. Evidence preparation** | Reusable frozen workflow: `uv run vqa-frozen30 prepare`; reusable frame builder: `build_frame_evidence()` in [`vqa_benchmark.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_benchmark.py). Transcript binding, when separately authorized evidence exists, is `uv run vqa-frozen30 build-transcript-requests`. | **Bespoke explanation:** twelve timestamps and one ASR line were authored for teaching; no frames, audio, cache, manifest, or request artifact was produced. |
| **3. Route** | **Missing general seam:** no versioned label-blind question router CLI/module exists. `build_transcript_requests()` in [`vqa_frozen30_workflow.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_frozen30_workflow.py) reuses accepted ASR quality status, but it does not implement the proposed lexical-speech question classifier. | The route string `lexical_speech → request ASR → quality accepted` is an explanatory project-authored decision, not a router output. |
| **4. Inference** | Supported orchestrator: `uv run vqa-frozen30 run CONDITION`; isolated current adapter for the illustrated Qwen family: `python -m trec_video_2026.vqa_transformers_runner`, invoked by the workflow rather than as a separate supported project CLI. | **Bespoke/missing execution:** `green tea` was invented. No model, container, runner request, or result existed. |
| **5. Freeze** | No standalone freeze CLI. The reusable workflow validates and appends attempts through `append_attempt_rows()` and `validate_frozen_attempt()` in [`vqa_frozen30.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_frozen30.py); `freeze_selected_predictions()` in [`vqa_frozen30_workflow.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_frozen30_workflow.py) selects validated saved rows. | “Frozen prediction” describes the narrated label boundary only; there is no append-only attempt row, hash, or run identifier. |
| **6. Score** | Workflow CLI: `uv run vqa-frozen30 score`; reusable join-and-score module: `evaluate_saved_predictions()` in [`vqa_evaluation.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_evaluation.py). There is no general `vqa-score` CLI. | Exact match and token F1 of `1.0` were explanatory arithmetic on invented strings, not benchmark metrics. |
| **7. Report** | **Bespoke/missing general CLI:** `vqa-frozen30 score` writes scored artifacts, but publication remains a manual, sanitized reporting step into tracked evidence pages. No reusable report generator promotes a run automatically. | This local worked-example page reports teaching content only and is not an evidence-of-record benchmark report. |

For an executed benchmark row, the minimum distinguishing evidence would
be a frozen run/condition identity, validated request and attempt
records, artifact hashes, runtime attestations, and post-freeze scoring
provenance. The synthetic page explicitly records that all of those are
absent.

## Supported commands

The four installed project scripts are declared in [`pyproject.toml` ·
Current code · Source code ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/pyproject.toml).
After `uv sync`, inspect them with `uv run <command> --help`; do not
launch a large download or GPU run merely to explore the code.

| Command | Role | What a newcomer should know |
|----|----|----|
| `youtube-dataset-download` | Generic YouTube-ID/watch-URL acquisition | Reusable engine with `prepare`, `run`, `start`, and read-only `status` commands for a configured YouTube dataset. |
| `vqa-download` | TRECVID-specific acquisition adapter | Supplies repository-specific paths and operational defaults around the generic engine; commands can mutate ignored local runtime state. |
| `vqa-gate` | Small compatibility-gate entry point | Its public CLI exposes **`preflight` only**. The completed family runs used tested Python seams and adapters rather than a general one-command gate. |
| `vqa-frozen30` | Frozen-30 benchmark workflow | The most complete workflow: CPU preparation, read-only status, Whisper generation, transcript request binding, explicit condition authorization/run, resume controls, and scoring. |

There is **no general `vqa-score` CLI**. General scoring primitives live
in [`vqa_evaluation.py` · Current code · Source code ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_evaluation.py);
frozen-30 exposes its own `score` action, and later experiments have
purpose-built scoring modules or scripts.

## Code layers

| Layer | Representative paths · Current code · Source code · GitHub | Intended reuse |
|----|----|----|
| Supported CLIs | `youtube_dataset.py`, `vqa_download.py`, `vqa_gate.py`, `vqa_frozen30_workflow.py` | Stable user entry points declared by `pyproject.toml`. |
| Reusable modules | [`vqa_benchmark.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_benchmark.py), [`vqa_frozen30.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_frozen30.py), [`vqa_evaluation.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_evaluation.py), [`vqa_models.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_models.py) | Evidence construction, manifests, fail-closed resume controls, evaluation, and shared prompt behavior. |
| Specialist adapters | [`vqa_transformers_runner.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_transformers_runner.py), [`vqa_specialist_runner.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_specialist_runner.py), [`vqa_omni_runner.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_omni_runner.py), [`vqa_whisper.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_whisper.py), [`vqa_audio_runner.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_audio_runner.py), [`vqa_seamless.py` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_seamless.py) | Isolated `python -m trec_video_2026.<module>` request/result adapters for pinned model families or modalities; they are not separate project scripts. |
| Bespoke experiment workflows | [`vqa_experiments/` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/tree/main/src/trec_video_2026/vqa_experiments) and `scripts/` experiment helpers | Frozen contracts, transport, scoring, and collection for a specific research question; reuse only after reading that experiment’s report and limitations. |

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

| Path · Current code · Source code · GitHub | Read it when… | Caution |
|----|----|----|
| [`README.md` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/README.md) and [`pyproject.toml` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/blob/main/pyproject.toml) | You need the operating boundary or supported command names. | A help command is safer than starting work. |
| [`src/trec_video_2026/` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/tree/main/src/trec_video_2026) | You are changing reusable behavior, orchestration, or an adapter. | Identify the layer before widening an experiment-specific seam. |
| [`configs/vqa/` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/tree/main/configs/vqa) and [`scripts/` · Current code · Source code · GitHub](https://github.com/npatta01/trec_video_2026/tree/main/scripts) | You need frozen run definitions or operational helpers. | Some helpers launch expensive or stateful work; inspect first. |
| [`docs/experiments/` · Evidence of record · Evidence report · GitHub](https://github.com/npatta01/trec_video_2026/tree/814e55429f6dd1dc8570bc08f1dad849f74e36a1/docs/experiments) | You need recent sanitized experiment contracts and results. | Compatibility reports may be superseded by quality reports. |
| [July 12 field guide · Evidence of record · Evidence report · GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/report.html) | You need the background field guide or later analytical reports. | Historical recommendations are not the current default. |
| `tests/` | You are changing tracked behavior and need its executable contract. | This documentation task did not run or add application tests. |
| `data/` | A specifically authorized run requires the shared local dataset. | Ignored and private: never copy, stage, commit, or expose it. |

For research-memory maintenance, use [Adding Experiments and Results ·
Report page](adding-experiments.md). Edit the QMD report sources, then
regenerate their GitHub-readable Markdown and self-contained HTML; never
hand-edit those generated outputs.

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

### Research directions

Exact hypotheses, smallest valid panels, controls, integrity safeguards,
and decision rules live on the six theme pages rather than being
duplicated here:

- [Experiment integrity and evaluation](research-integrity.md)
- [Visual and temporal evidence](research-visual-evidence.md)
- [Speech and transcript evidence](research-speech-transcripts.md)
- [Native audio and sound](research-native-audio.md)
- [Models and adaptation](research-models.md)
- [Routing, fusion, and answer selection](research-routing.md)

The current safe next action is therefore determined by the question:
use [Research Themes · Report page](experiment-ledger.md) for a research
decision, or begin at the supported-command table above for a code
change.
