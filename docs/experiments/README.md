# TRECVID VQA experiment evidence

This directory is the file-level index for sanitized, dated experiment
evidence. Start with [Project Status](../project-status/index.md) for the
current baseline, condition-bound decisions, and next-test queue. This index
routes to evidence records; it does not replace the project-status ledger or
repeat its conclusions.

## How to use this index

- Use the **primary theme** link beside each record to see how the evidence is
  interpreted in the living report.
- Use the dated record for method, panel, provenance, validity, limitations,
  and immutable evidence links.
- The `2026-08-01-nemotron3-omni-frozen30.qmd` file is the editable QMD source
  for its sibling Markdown evidence record; it is listed once below.
- Supporting JSON files are machine-readable score artifacts. The linked human
  record remains the source for interpretation and decision boundaries.
- For the July 12 background field guide and earlier analytical reports, use
  the [report archive](../../reports/trecvid_vqa_2026/README.md). For historical
  execution plans, use the [plan archive](../../plans/README.md).

## Dated evidence records

### Models and adaptation

- **2026-07-24 — [Non-Qwen omni compatibility probe](2026-07-24-nonqwen-omni-compatibility.md)** · Compatibility probe · [Models and adaptation](../project-status/research-models.md#compatibility-gate-models)
- **2026-07-24 — [Omni semantic-quality correction](2026-07-24-omni-semantic-quality.md)** · Semantic-quality correction · [Models and adaptation](../project-status/research-models.md#phi-gemma-semantic-stop)
- **2026-07-24 — [Qwen3-Omni dstack compatibility probe](2026-07-24-qwen3-omni-dstack-compatibility.md)** · Runtime compatibility evidence · [Models and adaptation](../project-status/research-models.md#qwen3-omni-quality)
- **2026-07-25 — [Qwen3-Omni quality-10 neutral A2 result](2026-07-25-qwen3-omni-quality-10-neutral-a2.md)** · Strict quality diagnostic · [Models and adaptation](../project-status/research-models.md#qwen3-omni-quality)
- **2026-07-30 — [Nemotron 3 Nano Omni native audio-video pilot](2026-07-30-nemotron3-omni-pilot.md)** · Native audio-video pilot · [Models and adaptation](../project-status/research-models.md#current-working-answer)
- **2026-07-30 — [Qwen3-Omni frozen-30 infrastructure attempts](2026-07-30-qwen3-omni-frozen30-infrastructure-attempts.md)** · Historical infrastructure-attempt record · [Models and adaptation](../project-status/research-models.md#qwen3-omni-quality)
- **2026-07-31 — [Qwen3-Omni frozen-30 A4 result](2026-07-31-qwen3-omni-frozen30.md)** · Frozen-30 model comparison · [Models and adaptation](../project-status/research-models.md#qwen3-omni-quality)
- **2026-08-01 — [Nemotron 3 Nano Omni frozen-30 baseline replacement](2026-08-01-nemotron3-omni-frozen30.md)** · Frozen-30 model comparison · [Models and adaptation](../project-status/research-models.md#current-working-answer) · [QMD source](2026-08-01-nemotron3-omni-frozen30.qmd)

### Native audio and sound

- **2026-07-25 — [Qwen3-Omni native-audio benchmark](2026-07-25-qwen3-omni-quality-10.md)** · Native-audio benchmark · [Native audio and sound](../project-status/research-native-audio.md#qwen3-omni-native-audio)

### Visual and temporal evidence

- **2026-07-30 — [Temporal required-moment oracle](2026-07-30-temporal-required-moment-oracle.md)** · Temporal representation audit · [Visual and temporal evidence](../project-status/research-visual-evidence.md#r4-required-moment-stage-1-representation-audit)
- **2026-07-30 — [TimeLens2 selector compatibility pilot](2026-07-30-timelens2-selector-pilot.md)** · Selector compatibility pilot · [Visual and temporal evidence](../project-status/research-visual-evidence.md#current-working-answer)
- **2026-08-01 — [TimeLens2 three-condition frozen-30](2026-08-01-timelens2-three-condition-frozen30.md)** · Selector quality-gate record · [Visual and temporal evidence](../project-status/research-visual-evidence.md#current-working-answer)
- **Issue 8 — [Question-directed temporal sampling](temporal-sampling-issue-8.md)** · Temporal sampling diagnostic; date not recorded in the source · [Visual and temporal evidence](../project-status/research-visual-evidence.md#directed-temporal-sampling)

## Supporting score artifacts

These JSON files preserve machine-readable scoring and gate summaries. They are
not independent claims or promotion decisions.

| Artifact | Human-readable record |
| --- | --- |
| [`qwen3-omni-a4-semantic-score.json`](artifacts/qwen3-omni-a4-semantic-score.json) | [Omni semantic-quality correction](2026-07-24-omni-semantic-quality.md) |
| [`qwen3-omni-frozen30-a4-score.json`](artifacts/qwen3-omni-frozen30-a4-score.json) | [Qwen3-Omni frozen-30 A4 result](2026-07-31-qwen3-omni-frozen30.md) |
| [`qwen3-omni-quality-10-neutral-a2-score.json`](artifacts/qwen3-omni-quality-10-neutral-a2-score.json) | [Qwen3-Omni quality-10 neutral A2 result](2026-07-25-qwen3-omni-quality-10-neutral-a2.md) |
| [`qwen3-omni-quality-10-score.json`](artifacts/qwen3-omni-quality-10-score.json) | [Qwen3-Omni native-audio benchmark](2026-07-25-qwen3-omni-quality-10.md) |
