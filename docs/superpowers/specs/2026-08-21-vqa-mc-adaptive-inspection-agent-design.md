# VQA MC Adaptive Inspection Agent Design

Date: 2026-08-21

## Objective

Test whether a bounded evidence-seeking Gemini agent improves the frozen
20-video TRECVID multiple-choice diagnostic over the existing direct native-AV
control. The prototype is deliberately limited to two model calls per video,
but its inspection request is generic enough to support a more involved future
flow without replacing its contract.

## Scope

This is an experiment-mode prototype, not a submission pipeline. It adds one
new Gemini inference arm, a label-isolated scorer, focused tests, and a durable
result record. It does not alter the frozen direct-Gemini or Qwen artifacts,
implement an unbounded tool loop, add OCR or ASR services, or change final
submission selection.

The shared dataset remains outside Git. Questions, options, labels, media,
provider responses, inspection plans, and row-level predictions remain under
the ignored private `data/` tree. Only sanitized aggregate evidence may enter
tracked reports.

## Scientific design

### Question

Does one model-selected second inspection improve top-1 accuracy over the
agent's own first pass and the frozen 15/20 direct Gemini control without
creating more regressions than repairs?

### Hypothesis

A first pass can identify the evidence it most needs and request a temporally
or spatially better native-AV view. A second pass over that view can resolve
close alternatives more reliably than a single uniform view.

### Falsifier

Do not advance this design if the final stage fails its strict contract, does
not improve over its own first-stage ranking, or creates at least as many
regressions among first-stage-correct rows as it repairs among first-stage
misses. A result on this selected panel never establishes hidden-set accuracy.

### Treatment and control

- Frozen external control: existing Gemini 3.7 Flash native-AV result,
  4 FPS, low media resolution, 15/20 top-1.
- Within-run control: stage-one rankings from the new agent prompt, using the
  same complete video, 4 FPS, low media resolution, model, seed, high thinking,
  option strings, and canonical option order.
- Treatment: one model-selected native-AV inspection followed by a final
  evidence-bearing ranking.

The within-run comparison isolates the second inspection from the changed
planning prompt. The frozen external comparison says whether the complete
agent arm improves on the current operational baseline.

### Panel and integrity boundary

- Use exactly the 20 canonical rows in the frozen 2025 training diagnostic.
- The inference runner must reject label-bearing request bundles.
- Labels remain in the separate frozen label artifact and are loaded only by
  the scorer after both inference-stage predictions are frozen.
- No row is selected or configured using its correctness label.
- Record stage-one and final rankings for all rows, including failures and
  fallbacks.

### Decision rule

- **Promising prototype:** final top-1 exceeds both the new stage-one result
  and 15/20, with repairs strictly greater than regressions and 20/20 valid
  final rankings.
- **Uniform second look is unnecessary:** the planner overwhelmingly requests
  the same settings and a later matched uniform arm performs equivalently.
- **More involved agent justified:** different inspection plans repair
  different mechanisms, but one inspection leaves evidence-localization or
  backtracking failures.
- **Not promoted:** no net improvement, contract failure, or regressions at
  least equal repairs.

## Agent protocol

### Stage 1: observe, rank, and plan

Gemini receives the complete native video and audio at 4 FPS and low media
resolution, followed by the question, all four exact option strings, and the
video duration. It must return strict JSON containing:

- a complete `ranking` permutation of A-D;
- an `inspection` object with:
  - `start_seconds` and `end_seconds` within the video;
  - `fps`, restricted to `1`, `4`, or `8`;
  - `resolution`, restricted to `low` or `high`;
  - `evidence_gap`, one of `text_detail`, `rapid_or_repeated_action`,
    `speech_or_sound`, `global_composition`, `temporal_order`, or `other`;
  - `evidence_question`, a short request describing what the second pass must
    verify.

The planner must always request one inspection. This avoids relying on
uncalibrated self-reported confidence and makes every row an auditable
first-versus-final pair.

### Controller validation and fallback

The controller validates the request without labels. It rejects non-finite or
out-of-duration intervals, intervals with `end <= start`, unsupported FPS or
resolution values, missing fields, and malformed rankings.

If the response exists but only the inspection is invalid, the predetermined
fallback is the complete video at 4 FPS/high resolution with evidence gap
`other`. If the whole response is invalid or the provider call fails, the same
fallback inspection is still run, without a stage-one top-two hint. Every
fallback is explicitly recorded; nothing is silently clamped or repaired.

### Stage 2: inspect and decide

Gemini receives the requested native-AV interval at the requested FPS and
resolution. The prompt includes the original question and four exact options,
the inspection question, and the stage-one top-two set sorted by option ID so
the first-stage preference order is not disclosed. It returns strict JSON
containing:

- a complete final `ranking` permutation of A-D;
- one to three evidence entries, each with a `timestamp_seconds` inside the
  original video and a concise `observation`.

If stage two is invalid, the final ranking falls back to a valid stage-one
ranking and records `final_source = stage1_fallback`. If both are invalid, the
row remains invalid. This distinguishes answer quality from transport or
schema reliability.

## Evidence lifecycle

The runner writes immutable files with exclusive creation semantics:

1. run manifest and cost preflight;
2. stage-one per-response raw files and raw freeze;
3. parsed stage-one plans and predictions freeze;
4. validated inspection-request freeze;
5. stage-two per-response raw files and raw freeze;
6. final predictions freeze and completion/cost summary.

The scorer joins only the frozen prediction artifacts to labels. It reports
stage-one and final top-1/MRR, per-stratum metrics, repairs, regressions,
unchanged hits/misses, final-source counts, fallback counts, and the
distribution of requested FPS, resolution, evidence gaps, and interval
fractions.

## Cost and operational limits

- One upload per video, reused across both calls.
- Exactly 20 stage-one calls and 20 stage-two calls.
- Worst-case stage two assumes every planner requests the complete video at
  8 FPS/high resolution.
- Preflight must reject a worst-case estimate above `$0.75` before credentials
  are loaded or provider inference begins.
- The output directory must not already exist.
- No retry changes semantics. A terminal provider failure is recorded once.

## Implementation boundaries

- `src/trec_video_2026/vqa_experiments/vqa_mc_adaptive_agent.py` owns schemas,
  prompts, parsing, plan validation, deterministic fallback, final selection,
  and aggregate helpers. It contains no provider credentials or labels.
- `scripts/run_vqa_mc_gemini_adaptive_agent_poc20.py` owns preflight, uploads,
  Gemini calls, raw freezes, usage accounting, and the CLI.
- `scripts/score_vqa_mc_adaptive_agent_poc20.py` owns the post-freeze label
  join and comparison metrics.
- Focused tests cover contracts, interval validation, fallback behavior,
  request metadata, cost cap, freeze ordering, label rejection, and scoring.

## Future extension seam

The version-one `inspection` object is intentionally a single validated action.
A later agent may replace it with a bounded list plus actions such as OCR,
transcript search, scene-boundary navigation, or backtracking. That extension
must retain label isolation, per-action evidence freezes, a maximum action
budget, and an explicit stopping rule. Version one does not pre-commit to those
tools; it establishes whether adaptive evidence acquisition has value first.

## Verification

- Focused unit and runner tests pass with `PYTHONPATH=src`.
- Existing MC contract, runner, and scoring tests remain passing.
- `ruff check` passes on changed Python files.
- Preflight verifies the canonical shared dataset link, 20 preserved videos,
  strict request identity, and the `$0.75` cap without loading labels.
- After inference, hashes and completion counts are verified before scoring.
- The result report separates observed results, interpretation, limits, and
  the next decision.
